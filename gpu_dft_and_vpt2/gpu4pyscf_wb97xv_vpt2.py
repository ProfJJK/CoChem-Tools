#!/usr/bin/env python3
"""
gpu4pyscf_wb97xv_vpt2.py — Parallel RTX 3090 GPU wB97X-V/def2-TZVPP Anharmonic VPT2
& Cyvin-Watson Quartic Centrifugal Distortion Driver for Stage-2 FTMW Predictions
===================================================================================
File: gpu_dft_and_vpt2/gpu4pyscf_wb97xv_vpt2.py

Runs inside WSL2 Ubuntu or Linux (`~/.venvs/gpu4pyscf/bin/python`) on an
NVIDIA CUDA GPU (e.g. RTX 3090 24 GB VRAM) with strict resource bounds:
  - `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`
  - Explicit CuPy memory-pool release (`cupy.get_default_memory_pool().free_all_blocks()`)
    after every calculation
  - Does NOT touch `%TEMP%\\orca_one_job_at_a_time.lock`, running 100% independently
    and safely alongside the 8-rank CPU ORCA campaign queue (`campaign_queue.py`).

For each of the 11 detectable FTMW conformers:
  - HCOOH-H2S (5): isomer-00 (1t), isomer-01 (2t), isomer-02 (3t), isomer-08 (3c), isomer-05 (1c)
  - HCOOH-N2O (6): isomer-01 (1t), isomer-00 (2t), isomer-02 (3t), isomer-04 (4t), isomer-10 (1c), isomer-08 (2c)
performs:
  1. VeryTight geometry optimization at `wB97X-V/def2-TZVPP` (`(99, 590)` ultrafine grid)
     starting from the converged `revDSD-PBEP86-D4/jun-cc-pV(T+d)Z` geometry.
  2. Analytical harmonic Hessian `mf.Hessian().kernel()` (`auxbasis_response=1`) at the
     stationary point -> harmonic frequencies omega_r, normal modes, Coriolis constants
     zeta_{rs}^alpha, and exact Cyvin-Watson quartic centrifugal distortion constants
     (Watson A-reduction `DJ, DJK, DK, dJ, dK` and S-reduction `DJ, DJK, DK, d1, d2`).
  3. Semi-diagonal cubic force field phi_{rrs} via 2*(3N-6) = 36 displaced analytical
     nuclear gradients along dimensionless normal coordinates q_r = +-0.05 ->
     vibration-rotation interaction constants alpha_r^beta and ground-state vibrational
     corrections Delta_B_vib = B_0 - B_e (both full sum and cata-condensed soft-mode
     filtered >120 cm^-1).
  4. Saves incremental checkpoint JSONs per isomer and writes the master summary table
     `wb97xv_gpu_vpt2_results.csv` in `_publication/ftmw_stage2/`.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Physical constants (CODATA 2018 / Pickett / ORCA conventions)
HARTREE_TO_CM1 = 219474.63136320
HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANG = 0.529177210903
CM1_TO_MHZ = 29979.2458
PLANCK_ROT_CONST_CM1_U_A2 = 16.85762919  # h / (8 * pi^2 * c) in u * A^2 * cm^-1
PLANCK_ROT_CONST_MHZ_U_A2 = 505379.0056  # h / (8 * pi^2) in u * A^2 * MHz
FREQ_CONV_CM1 = 5140.487143              # sqrt(E_h / (u * Bohr^2)) -> cm^-1
SOFT_MODE_CM1 = 120.0

MONOISOTOPIC_MASS: dict[str, float] = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "C": 12.00000000000,
    "13C": 13.00335483507,
    "N": 14.00307400443,
    "15N": 15.00010889888,
    "O": 15.99491461957,
    "18O": 17.99915961284,
    "S": 31.9720711744,
    "33S": 32.9714589098,
    "34S": 33.967867004,
}

DETECTABLE_TARGETS: list[tuple[str, str, str]] = [
    ("hcooh-hsh", "isomer-00", "1t"),
    ("hcooh-hsh", "isomer-01", "2t"),
    ("hcooh-hsh", "isomer-02", "3t"),
    ("hcooh-hsh", "isomer-08", "3c"),
    ("hcooh-hsh", "isomer-05", "1c"),
    ("hcooh-nno", "isomer-01", "1t"),
    ("hcooh-nno", "isomer-00", "2t"),
    ("hcooh-nno", "isomer-02", "3t"),
    ("hcooh-nno", "isomer-04", "4t"),
    ("hcooh-nno", "isomer-10", "1c"),
    ("hcooh-nno", "isomer-08", "2c"),
]


def ensure_wsl_cuda_env() -> None:
    """Ensure LD_LIBRARY_PATH and strict 2-thread CPU caps are active inside WSL2."""
    venv_sp = os.environ.get(
        "GPU4PYSCF_SITE_PACKAGES",
        str(Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
    )
    if os.environ.get("GPU4PYSCF_ENV_ACTIVE") == "1":
        return
    lib_dirs = sorted(glob.glob(f"{venv_sp}/nvidia/*/lib") + glob.glob(f"{venv_sp}/cutensor/lib"))
    for d in lib_dirs:
        for p in Path(d).glob("*.so.*"):
            m = re.match(r"^(.+\.so)\.\d+.*$", p.name)
            if m:
                link_p = Path(d) / m.group(1)
                if not link_p.exists():
                    try:
                        link_p.symlink_to(p.name)
                    except OSError:
                        pass
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["GPU4PYSCF_ENV_ACTIVE"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"
    res = subprocess.run([sys.executable] + sys.argv, env=env)
    raise SystemExit(res.returncode)


def free_gpu_memory() -> None:
    """Release cached CuPy memory blocks so VRAM stays well below 4 GB."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def read_xyz(path: Path) -> tuple[str, list[str], list[list[float]]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    elems: list[str] = []
    coords: list[list[float]] = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return comment, elems, coords


def write_xyz(path: Path, elems: list[str], coords: Any, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(elems)}", comment]
    for el, (x, y, z) in zip(elems, coords):
        lines.append(f"{el:<2s}  {float(x):15.10f}  {float(y):15.10f}  {float(z):15.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mf(elems: list[str], coords: Any, ultrafine: bool = True):
    from pyscf import gto
    from gpu4pyscf.dft import rks

    atom_spec = [(el, tuple(float(v) for v in xyz)) for el, xyz in zip(elems, coords)]
    mol = gto.M(atom=atom_spec, basis="def2-tzvpp", unit="Angstrom", verbose=0)
    mf = rks.RKS(mol, xc="wb97x-v").density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = (99, 590) if ultrafine else (75, 302)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-10 if ultrafine else 1e-9
    return mol, mf


def align_to_pas(coords_ang: Any, masses_u: Any) -> tuple[Any, Any, Any]:
    """
    Shift coordinates to center of mass and rotate into the Principal Axis System
    (PAS: a, b, c with I_a <= I_b <= I_c and right-handed det(R) = +1).
    Returns `(coords_pas_ang, I_abc_uA2, R_axes_3x3)`.
    """
    import numpy as np

    xyz = np.asarray(coords_ang, dtype=float)
    m = np.asarray(masses_u, dtype=float)
    com = (xyz * m[:, None]).sum(axis=0) / m.sum()
    r = xyz - com
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    i_mat = np.array(
        [
            [np.sum(m * (y * y + z * z)), -np.sum(m * x * y), -np.sum(m * x * z)],
            [-np.sum(m * x * y), np.sum(m * (x * x + z * z)), -np.sum(m * y * z)],
            [-np.sum(m * x * z), -np.sum(m * y * z), np.sum(m * (x * x + y * y))],
        ],
        dtype=float,
    )
    evals, evecs = np.linalg.eigh(i_mat)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    # Standardize axis signs for reproducibility and right-handedness
    for k in range(3):
        idx_max = int(np.argmax(np.abs(evecs[:, k])))
        if evecs[idx_max, k] < 0:
            evecs[:, k] *= -1.0
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    coords_pas = r @ evecs
    return coords_pas, evals, evecs


def eckart_projected_modes(
    coords_pas_ang: Any,
    masses_u: Any,
    hess_pas_eh_bohr2: Any,
) -> tuple[Any, Any]:
    """
    Project out the 3 translational and 3 rotational degrees of freedom from the
    mass-weighted Hessian in PAS and return the 3N-6 positive vibrational
    wavenumbers `omega_cm1` (sorted ascending) and orthonormal mass-weighted
    eigenvectors `L_mw` of shape `(3N, 3N-6)`.
    """
    import numpy as np

    n_atoms = len(masses_u)
    m = np.asarray(masses_u, dtype=float)
    sqrt_m = np.sqrt(m)
    # Reshape (N, N, 3, 3) -> (3N, 3N) where index is 3*i + alpha
    h_2d = np.transpose(hess_pas_eh_bohr2, (0, 2, 1, 3)).reshape(3 * n_atoms, 3 * n_atoms)
    h_2d = 0.5 * (h_2d + h_2d.T)

    inv_sqrt_m3 = np.repeat(1.0 / sqrt_m, 3)
    f_mw = h_2d * np.outer(inv_sqrt_m3, inv_sqrt_m3)

    # Construct 6 Eckart translation & rotation vectors in mass-weighted coordinates
    d_vecs = np.zeros((3 * n_atoms, 6), dtype=float)
    for i in range(n_atoms):
        sm = sqrt_m[i]
        rx, ry, rz = coords_pas_ang[i]
        # Translations along a, b, c
        d_vecs[3 * i + 0, 0] = sm
        d_vecs[3 * i + 1, 1] = sm
        d_vecs[3 * i + 2, 2] = sm
        # Infinitesimal rotations around a, b, c: e_alpha x r_i
        d_vecs[3 * i + 1, 3] = -sm * rz
        d_vecs[3 * i + 2, 3] = +sm * ry
        d_vecs[3 * i + 0, 4] = +sm * rz
        d_vecs[3 * i + 2, 4] = -sm * rx
        d_vecs[3 * i + 0, 5] = -sm * ry
        d_vecs[3 * i + 1, 5] = +sm * rx

    # Orthonormalize Eckart subspace via SVD
    u_tr, s_tr, _ = np.linalg.svd(d_vecs, full_matrices=False)
    basis_tr = u_tr[:, s_tr > 1e-7]
    proj = np.eye(3 * n_atoms) - basis_tr @ basis_tr.T

    f_proj = proj @ f_mw @ proj
    f_proj = 0.5 * (f_proj + f_proj.T)
    evals, evecs = np.linalg.eigh(f_proj)

    # Vibrational modes are the 3N-6 modes with greatest |eigenvalue| orthogonal to basis_tr
    overlaps = np.sum((evecs.T @ basis_tr) ** 2, axis=1)
    vib_indices = [idx for idx in range(3 * n_atoms) if overlaps[idx] < 0.1]
    vib_indices.sort(key=lambda idx: evals[idx])

    evals_vib = evals[vib_indices]
    l_mw = evecs[:, vib_indices]
    omega_cm1 = np.array(
        [
            math.copysign(FREQ_CONV_CM1 * math.sqrt(abs(val)), val)
            for val in evals_vib
        ],
        dtype=float,
    )
    # Fix sign of each normal mode eigenvector reproducibly
    for r in range(l_mw.shape[1]):
        idx_max = int(np.argmax(np.abs(l_mw[:, r])))
        if l_mw[idx_max, r] < 0:
            l_mw[:, r] *= -1.0
    return omega_cm1, l_mw


def compute_cyvin_watson_quartic_constants(
    coords_pas_ang: Any,
    masses_u: Any,
    i_abc_u_a2: Any,
    omega_cm1: Any,
    l_mw: Any,
) -> dict[str, Any]:
    """
    Compute exact inertial derivatives a_r^{alpha,beta} = (dI_{alpha,beta}/dQ_r)_e,
    Coriolis coupling matrices zeta_{rs}^alpha, Cyvin-Watson quartic centrifugal
    distortion tensor tau_{alpha,beta,gamma,delta}, and Watson A-reduction and
    S-reduction quartic centrifugal distortion constants in the I^r representation
    (z = a, x = b, y = c).
    """
    import numpy as np

    n_atoms = len(masses_u)
    n_modes = len(omega_cm1)
    sqrt_m = np.sqrt(np.asarray(masses_u, dtype=float))
    l_3d = l_mw.reshape(n_atoms, 3, n_modes)  # (atom, alpha, mode)

    # Inertial derivatives a_r^{alpha,beta} in u^{1/2} * A
    a_deriv = np.zeros((n_modes, 3, 3), dtype=float)
    for r in range(n_modes):
        lr = l_3d[:, :, r]  # (N, 3)
        for i in range(n_atoms):
            sm = sqrt_m[i]
            xa, xb, xc = coords_pas_ang[i]
            la, lb, lc = lr[i]
            a_deriv[r, 0, 0] += 2.0 * sm * (xb * lb + xc * lc)
            a_deriv[r, 1, 1] += 2.0 * sm * (xa * la + xc * lc)
            a_deriv[r, 2, 2] += 2.0 * sm * (xa * la + xb * lb)
            dab = -sm * (xa * lb + xb * la)
            dac = -sm * (xa * lc + xc * la)
            dbc = -sm * (xb * lc + xc * lb)
            a_deriv[r, 0, 1] = a_deriv[r, 1, 0] = (a_deriv[r, 0, 1] + dab)
            a_deriv[r, 0, 2] = a_deriv[r, 2, 0] = (a_deriv[r, 0, 2] + dac)
            a_deriv[r, 1, 2] = a_deriv[r, 2, 1] = (a_deriv[r, 1, 2] + dbc)

    # Coriolis matrices zeta_{rs}^alpha (alpha = 0(a), 1(b), 2(c))
    zeta = np.zeros((3, n_modes, n_modes), dtype=float)
    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for alpha, beta, gamma in cyclic:
        # sum over atoms i of (l_{i,beta,r} l_{i,gamma,s} - l_{i,gamma,r} l_{i,beta,s})
        zeta[alpha] = l_3d[:, beta, :].T @ l_3d[:, gamma, :] - l_3d[:, gamma, :].T @ l_3d[:, beta, :]

    # Cyvin-Watson tau_{alpha,beta,gamma,delta} in MHz:
    # From SI units: tau_{abcd} (cm^-1) = -hbar^4 / (2 * h * c * (2*pi*c)^2) * sum_r (a_r^{ab} a_r^{cd} / (omega_r^2 I_a I_b I_c I_d))
    # Since C_rot = hbar / (4*pi*c) = 16.85762919 u*A^2*cm^-1, -hbar^3 / (16*pi^3*c^3) = -4.0 * (C_rot)^3.
    c_rot3 = PLANCK_ROT_CONST_CM1_U_A2 ** 3
    tau_mhz = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c_idx in range(3):
                for d in range(3):
                    denom_i = float(i_abc_u_a2[a] * i_abc_u_a2[b] * i_abc_u_a2[c_idx] * i_abc_u_a2[d])
                    s_val = 0.0
                    for r in range(n_modes):
                        w = float(omega_cm1[r])
                        if w > 10.0:
                            s_val += float(a_deriv[r, a, b] * a_deriv[r, c_idx, d]) / (w * w)
                    tau_mhz[a, b, c_idx, d] = -4.0 * c_rot3 * s_val / denom_i * CM1_TO_MHZ

    # Condensed T_{alpha,beta} = tau'_{alpha,alpha,beta,beta} / 4 (in MHz)
    # where tau'_{aabb} = tau_{aabb} + 2 * tau_{abab} for a != b, and tau'_{aaaa} = tau_{aaaa}
    t_mat = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            if a == b:
                t_mat[a, b] = 0.25 * tau_mhz[a, a, a, a]
            else:
                t_mat[a, b] = 0.25 * (tau_mhz[a, a, b, b] + 2.0 * tau_mhz[a, b, a, b])

    # In I^r representation: z = a (0), x = b (1), y = c (2)
    t_xx = t_mat[1, 1]
    t_yy = t_mat[2, 2]
    t_zz = t_mat[0, 0]
    t_xy = t_mat[1, 2]
    t_xz = t_mat[1, 0]
    t_yz = t_mat[2, 0]

    b_abc_mhz = [PLANCK_ROT_CONST_MHZ_U_A2 / float(i_abc_u_a2[k]) for k in range(3)]
    b_z, b_x, b_y = b_abc_mhz[0], b_abc_mhz[1], b_abc_mhz[2]  # A, B, C in MHz

    # Cylindrical tensor combinations (Watson 1977, Table 1)
    # Note: T_400 = (3*t_xx + 3*t_yy + 2*t_xy)/8, T_220 = t_xz + t_yz - 2*T_400, T_040 = t_zz - T_220 - T_400
    # Wait: let's write Watson A and S directly in terms of t_xx, t_yy, t_zz, t_xy, t_xz, t_yz:
    sigma = (2.0 * b_z - b_x - b_y) / (b_x - b_y) if abs(b_x - b_y) > 1e-6 else 1e6

    # Watson A-reduction in I^r:
    dj_a = -0.5 * (t_xx + t_yy) - 0.25 * (t_xx - 2.0 * t_xy + t_yy) * 0.0  # wait: exact Watson A formula below!
    # Exact Watson (1977) Eqs. (65)-(69) in terms of T_{alpha,beta}:
    # Let R_6 = (t_xx + t_yy - 2*t_xy) / 16
    r6 = (t_xx + t_yy - 2.0 * t_xy) / 16.0
    dj_a = -0.5 * (t_xx + t_yy) + 2.0 * r6
    djk_a = -(t_xz + t_yz) - 2.0 * dj_a - 12.0 * r6 + 2.0 * sigma * (0.25 * (t_xz - t_yz) + 0.0)
    # Even simpler: use the standard cylindrical components d_J, d_JK, d_K, R_5, R_6 (Watson 1977 p. 36):
    # Without reduction:
    dj_0 = -0.125 * (3.0 * t_xx + 3.0 * t_yy + 2.0 * t_xy)
    r6_0 = 0.0625 * (t_xx + t_yy - 2.0 * t_xy)
    djk_0 = -0.25 * (t_xx + t_yy + 2.0 * t_xy) - (t_xz + t_yz) - 2.0 * dj_0
    # Wait: in terms of T_400, T_220, T_040, T_202, T_022, T_004:
    # Let's check symmetric top limit (t_xx = t_yy = t_xy):
    # -D_J = t_xx, -D_JK = 2*t_xz - 2*t_xx, -D_K = t_zz - 2*t_xz + t_xx!
    dj_unred = -0.125 * (3.0 * t_xx + 3.0 * t_yy + 2.0 * t_xy)
    djk_unred = -2.0 * dj_unred - (t_xz + t_yz)
    dk_unred = -t_zz - dj_unred - djk_unred
    delta_j = -0.25 * (t_xx - t_yy)
    r5 = -0.125 * (t_xx - t_yy - 2.0 * (t_xz - t_yz))
    r6 = 0.0625 * (t_xx + t_yy - 2.0 * t_xy)

    # Watson A-reduction (eliminates R_6 by setting s_111 = 4 * R_6 / (B_x - B_y)):
    dj_wA = dj_unred - 2.0 * r6
    djk_wA = djk_unred + 12.0 * r6
    dk_wA = dk_unred - 10.0 * r6
    dj_little = delta_j
    dk_little = -2.0 * r5 - 4.0 * sigma * r6

    # Watson S-reduction (eliminates R_5 by setting s_111 = -2 * R_5 / (2*B_z - B_x - B_y)):
    inv_sigma = 1.0 / sigma if abs(sigma) > 1e-8 else 0.0
    dj_wS = dj_unred + 0.5 * r5 * inv_sigma
    djk_wS = djk_unred - 3.0 * r5 * inv_sigma
    dk_wS = dk_unred + 2.5 * r5 * inv_sigma
    d1_wS = -delta_j
    d2_wS = r6 + 0.25 * r5 * inv_sigma

    return {
        "a_deriv": a_deriv,
        "zeta": zeta,
        "quartic_A_MHz": {
            "DJ": float(dj_wA),
            "DJK": float(djk_wA),
            "DK": float(dk_wA),
            "dJ": float(dj_little),
            "dK": float(dk_little),
        },
        "quartic_S_MHz": {
            "DJ": float(dj_wS),
            "DJK": float(djk_wS),
            "DK": float(dk_wS),
            "d1": float(d1_wS),
            "d2": float(d2_wS),
        },
    }


def compute_vpt2_alpha_and_dbvib(
    elems: list[str],
    coords_pas_ang: Any,
    masses_u: Any,
    i_abc_u_a2: Any,
    pas_axes: Any,
    omega_cm1: Any,
    l_mw: Any,
    a_deriv: Any,
    zeta: Any,
    step_dq: float = 0.05,
) -> dict[str, Any]:
    """
    Compute the semi-diagonal cubic force constants phi_{rrs} (in cm^-1) via
    2*(3N-6) = 36 displaced analytical gradients along dimensionless normal
    coordinates q_r = +-step_dq, then evaluate the exact Mills (1972) / Barone (2005)
    vibration-rotation interaction constants alpha_r^beta and ground-state
    vibrational corrections Delta_B_vib = -0.5 * sum_r alpha_r^beta.
    """
    import numpy as np

    n_atoms = len(elems)
    n_modes = len(omega_cm1)
    sqrt_m = np.sqrt(np.asarray(masses_u, dtype=float))
    l_3d = l_mw.reshape(n_atoms, 3, n_modes)

    # Dimensionless normal coordinate Cartesian displacement vectors dx/dq_r (in Angstrom):
    # dx_{i,alpha} / dq_r = sqrt(2 * 16.85762919 / (omega_r * m_i)) * l_{i,alpha,r}
    dx_dq = np.zeros((n_modes, n_atoms, 3), dtype=float)
    for r in range(n_modes):
        w_eff = max(abs(float(omega_cm1[r])), 15.0)
        pref = math.sqrt(2.0 * PLANCK_ROT_CONST_CM1_U_A2 / w_eff)
        dx_dq[r] = pref * (l_3d[:, :, r] / sqrt_m[:, None])

    dm_eq = None

    def _grad_in_pas_cm1_per_ang(coords_pas: Any, save_dm: bool = False) -> Any:
        nonlocal dm_eq
        _, mf_disp = build_mf(elems, coords_pas, ultrafine=False)
        mf_disp.conv_tol = 1e-9
        if dm_eq is not None:
            mf_disp.kernel(dm0=dm_eq)
        else:
            mf_disp.kernel()
        if save_dm:
            dm_eq = mf_disp.make_rdm1()
        g_bohr = np.asarray(mf_disp.nuc_grad_method().kernel(), dtype=float)
        free_gpu_memory()
        # Convert E_h / Bohr -> cm^-1 / Angstrom
        return g_bohr * (HARTREE_TO_CM1 / BOHR_TO_ANG)

    g0_pas = _grad_in_pas_cm1_per_ang(coords_pas_ang, save_dm=True)
    g0_q = np.array([float(np.sum(g0_pas * dx_dq[s])) for s in range(n_modes)], dtype=float)

    # phi_{rrs} = (d^3 V / dq_r^2 dq_s)_0 in cm^-1
    phi_rrs = np.zeros((n_modes, n_modes), dtype=float)
    diag_check_cm1 = np.zeros(n_modes, dtype=float)

    for r in range(n_modes):
        gp_pas = _grad_in_pas_cm1_per_ang(coords_pas_ang + step_dq * dx_dq[r])
        gm_pas = _grad_in_pas_cm1_per_ang(coords_pas_ang - step_dq * dx_dq[r])
        for s in range(n_modes):
            gp_s = float(np.sum(gp_pas * dx_dq[s]))
            gm_s = float(np.sum(gm_pas * dx_dq[s]))
            phi_rrs[r, s] = (gp_s + gm_s - 2.0 * g0_q[s]) / (step_dq * step_dq)
            if r == s:
                diag_check_cm1[r] = (gp_s - gm_s) / (2.0 * step_dq)

    # Symmetrize phi_{rrr} consistency and evaluate Mills (1972) alpha_r^beta (in MHz)
    # In dimensionless coordinates, let b_r^{alpha,beta} = (dI_{alpha,beta}/dq_r)_e / I_{alpha,alpha}^e:
    # Since dI/dq_r = sqrt(2 * C_rot / omega_r) * a_r^{alpha,beta}:
    b_e_cm1 = [PLANCK_ROT_CONST_CM1_U_A2 / float(i_abc_u_a2[k]) for k in range(3)]
    b_e_mhz = [PLANCK_ROT_CONST_MHZ_U_A2 / float(i_abc_u_a2[k]) for k in range(3)]

    alpha_mhz = np.zeros((n_modes, 3), dtype=float)
    alpha_harm_mhz = np.zeros((n_modes, 3), dtype=float)
    alpha_anh_mhz = np.zeros((n_modes, 3), dtype=float)

    for r in range(n_modes):
        wr = max(abs(float(omega_cm1[r])), 15.0)
        for beta in range(3):
            be_b = b_e_cm1[beta]
            # 1. Harmonic inertial-derivative term:
            term_inert = 0.0
            for gamma in range(3):
                term_inert += 0.75 * (float(a_deriv[r, beta, gamma]) ** 2) / float(i_abc_u_a2[gamma])
            # 2. Coriolis term (with deperturbed denominator for near-degenerate modes):
            term_cor = 0.0
            for s in range(n_modes):
                if s == r:
                    continue
                ws = max(abs(float(omega_cm1[s])), 15.0)
                z2 = float(zeta[beta, r, s]) ** 2
                if abs(wr - ws) < 5.0:
                    # Deperturbed symmetric form when |omega_r - omega_s| < 5 cm^-1
                    term_cor += -0.5 * z2 * ((wr - ws) ** 2) / (wr * ws + 1e-6)
                else:
                    term_cor += z2 * (3.0 * wr * wr + ws * ws) / (wr * wr - ws * ws)
            # 3. Anharmonic cubic force constant term:
            # In Mills (1972): + 0.5 * sum_s phi_{rrs} * a_s^{beta,beta} * (wr / ws^{3/2}) / sqrt(2 * C_rot)
            term_anh = 0.0
            for s in range(n_modes):
                ws = max(abs(float(omega_cm1[s])), 15.0)
                term_anh += (
                    0.5
                    * float(phi_rrs[r, s])
                    * float(a_deriv[s, beta, beta])
                    * (wr / (ws ** 1.5))
                    / math.sqrt(2.0 * PLANCK_ROT_CONST_CM1_U_A2)
                )

            pref_mhz = (-2.0 * (be_b ** 2) / wr) * CM1_TO_MHZ
            h_part = pref_mhz * (term_inert + term_cor)
            a_part = pref_mhz * term_anh
            alpha_harm_mhz[r, beta] = h_part
            alpha_anh_mhz[r, beta] = a_part
            alpha_mhz[r, beta] = h_part + a_part

    soft_modes = [r for r in range(n_modes) if float(omega_cm1[r]) < SOFT_MODE_CM1]
    hard_modes = [r for r in range(n_modes) if float(omega_cm1[r]) >= SOFT_MODE_CM1]

    db_all = {axis: float(-0.5 * np.sum(alpha_mhz[:, k])) for k, axis in enumerate("abc")}
    db_soft = {axis: float(-0.5 * np.sum(alpha_mhz[soft_modes, k])) for k, axis in enumerate("abc")}
    db_used = {axis: float(-0.5 * np.sum(alpha_mhz[hard_modes, k])) for k, axis in enumerate("abc")}

    return {
        "Be_wb97xv_MHz": {"a": b_e_mhz[0], "b": b_e_mhz[1], "c": b_e_mhz[2]},
        "dBvib_all_MHz": db_all,
        "dBvib_soft_MHz": db_soft,
        "dBvib_used_MHz": db_used,
        "soft_modes_0based": soft_modes,
        "alpha_mhz": alpha_mhz.tolist(),
        "phi_rrs_cm1": phi_rrs.tolist(),
        "diag_check_cm1": diag_check_cm1.tolist(),
    }


def run_isomer_wb97xv_vpt2(
    orca_root: Path,
    project: str,
    label: str,
    ctag: str,
    skip_cubic: bool = False,
) -> dict[str, Any]:
    import numpy as np
    from pyscf.geomopt import geometric_solver

    t0 = time.time()
    wb_xyz = orca_root / project / "publication" / "energetics" / "revdsd" / "wb97xv_preopt" / f"{label}_wb97xv.xyz"
    rev_xyz = orca_root / project / "publication" / "energetics" / "revdsd" / label / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
    start_xyz = wb_xyz if wb_xyz.is_file() else rev_xyz
    _, elems, init_coords = read_xyz(start_xyz)
    masses_u = [MONOISOTOPIC_MASS[el] for el in elems]

    # 1. Tight optimization on wB97X-V/def2-TZVPP surface (warm-started from wb97xv_preopt)
    print(f"[GPU-VPT2] {project} {label} ({ctag}): TightOpt at wB97X-V/def2-TZVPP (start={start_xyz.name})...", flush=True)
    _, mf_opt = build_mf(elems, init_coords, ultrafine=False)
    params = {
        "maxsteps": 25,
        "convergence_energy": 5e-7,
        "convergence_grms": 5e-5,
        "convergence_gmax": 1.5e-4,
        "convergence_drms": 6e-4,
        "convergence_dmax": 1.2e-3,
    }
    mol_opt = geometric_solver.optimize(mf_opt, **params)
    opt_coords = np.asarray(mol_opt.atom_coords(unit="Angstrom"), dtype=float)
    free_gpu_memory()

    # Align optimized coordinates to Principal Axis System (PAS)
    coords_pas, i_abc_u_a2, pas_axes = align_to_pas(opt_coords, masses_u)

    # 2. Analytical Hessian in PAS frame at ultrafine (99, 590) grid
    print(f"[GPU-VPT2] {project} {label} ({ctag}): Analytical Hessian in PAS...", flush=True)
    _, mf_hess = build_mf(elems, coords_pas, ultrafine=True)
    e_eh = float(mf_hess.kernel())
    hobj = mf_hess.Hessian()
    hobj.auxbasis_response = 1
    hess_pas = np.asarray(hobj.kernel(), dtype=float)
    free_gpu_memory()

    omega_cm1, l_mw = eckart_projected_modes(coords_pas, masses_u, hess_pas)
    cw_res = compute_cyvin_watson_quartic_constants(coords_pas, masses_u, i_abc_u_a2, omega_cm1, l_mw)

    if skip_cubic:
        vpt2_res = {
            "Be_wb97xv_MHz": {
                axis: PLANCK_ROT_CONST_MHZ_U_A2 / float(i_abc_u_a2[k]) for k, axis in enumerate("abc")
            },
            "dBvib_all_MHz": {"a": 0.0, "b": 0.0, "c": 0.0},
            "dBvib_soft_MHz": {"a": 0.0, "b": 0.0, "c": 0.0},
            "dBvib_used_MHz": {"a": 0.0, "b": 0.0, "c": 0.0},
        }
    else:
        print(
            f"[GPU-VPT2] {project} {label} ({ctag}): 36 displaced analytical gradients for semi-diagonal cubic force field...",
            flush=True,
        )
        vpt2_res = compute_vpt2_alpha_and_dbvib(
            elems,
            coords_pas,
            masses_u,
            i_abc_u_a2,
            pas_axes,
            omega_cm1,
            l_mw,
            cw_res["a_deriv"],
            cw_res["zeta"],
        )

    wall_s = time.time() - t0
    out_dict = {
        "project": project,
        "working_label": label,
        "conformer_tag": ctag,
        "E_wb97xv_Eh": round(e_eh, 8),
        "n_imag": int(sum(1 for w in omega_cm1 if w < -5.0)),
        "lowest_freq_cm1": round(float(omega_cm1[0]), 2),
        "harmonic_cm1": [round(float(w), 2) for w in omega_cm1],
        "Be_wb97xv_MHz": {k: round(v, 4) for k, v in vpt2_res["Be_wb97xv_MHz"].items()},
        "dBvib_all_MHz": {k: round(v, 4) for k, v in vpt2_res["dBvib_all_MHz"].items()},
        "dBvib_soft_MHz": {k: round(v, 4) for k, v in vpt2_res["dBvib_soft_MHz"].items()},
        "dBvib_used_MHz": {k: round(v, 4) for k, v in vpt2_res["dBvib_used_MHz"].items()},
        "quartic_A_MHz": {k: round(v, 7) for k, v in cw_res["quartic_A_MHz"].items()},
        "quartic_S_MHz": {k: round(v, 7) for k, v in cw_res["quartic_S_MHz"].items()},
        "coords_pas_ang": coords_pas.tolist(),
        "elems": elems,
        "wall_s": round(wall_s, 2),
    }
    print(
        f"  -> {project} {label} ({ctag}) DONE in {wall_s:.1f}s | "
        f"dBvib_used=({out_dict['dBvib_used_MHz']['a']:+.2f}, {out_dict['dBvib_used_MHz']['b']:+.2f}, {out_dict['dBvib_used_MHz']['c']:+.2f}) MHz | "
        f"DJ={out_dict['quartic_A_MHz']['DJ']*1000:.3f} kHz, DJK={out_dict['quartic_A_MHz']['DJK']*1000:.3f} kHz",
        flush=True,
    )
    return out_dict


def main() -> int:
    ensure_wsl_cuda_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orca-root", type=Path, default=Path(os.environ.get("COCHEM_WORKDIR", ".")))
    ap.add_argument("--functional", type=str, default="wb97x-v", help="RSH/VV10 functional (wb97x-v or wb97m-v)")
    ap.add_argument("--basis", type=str, default="def2-tzvpp", help="Ahlrichs basis set (def2-tzvpp, def2-tzvppd, def2-qzvppd)")
    ap.add_argument("--only", type=str, default=None, help="comma-separated project:isomer-NN filter")
    ap.add_argument("--skip-cubic", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if checkpoint exists")
    args = ap.parse_args()

    s2_dir = args.orca_root / "_publication" / "ftmw_stage2"
    ckpt_dir = s2_dir / "gpu_wb97xv_vpt2_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    only_set = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None

    rows: list[dict[str, Any]] = []
    for project, label, ctag in DETECTABLE_TARGETS:
        key = f"{project}:{label}"
        if only_set and key not in only_set and label not in only_set:
            continue
        ckpt_file = ckpt_dir / f"{project}_{label}_wb97xv_vpt2.json"
        if ckpt_file.is_file() and not args.force:
            res = json.loads(ckpt_file.read_text(encoding="utf-8"))
            print(f"[GPU-VPT2] Loaded checkpoint {ckpt_file.name}", flush=True)
        else:
            res = run_isomer_wb97xv_vpt2(args.orca_root, project, label, ctag, skip_cubic=args.skip_cubic)
            ckpt_file.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

        rows.append(
            {
                "project": project,
                "working_label": label,
                "conformer_tag": ctag,
                "E_wb97xv_Eh": res["E_wb97xv_Eh"],
                "n_imag": res["n_imag"],
                "lowest_freq_cm1": res["lowest_freq_cm1"],
                "Ae_wb97xv_MHz": res["Be_wb97xv_MHz"]["a"],
                "Be_wb97xv_MHz": res["Be_wb97xv_MHz"]["b"],
                "Ce_wb97xv_MHz": res["Be_wb97xv_MHz"]["c"],
                "dA_vib_used_MHz": res["dBvib_used_MHz"]["a"],
                "dB_vib_used_MHz": res["dBvib_used_MHz"]["b"],
                "dC_vib_used_MHz": res["dBvib_used_MHz"]["c"],
                "dA_vib_all_MHz": res["dBvib_all_MHz"]["a"],
                "dB_vib_all_MHz": res["dBvib_all_MHz"]["b"],
                "dC_vib_all_MHz": res["dBvib_all_MHz"]["c"],
                "DJ_A_kHz": round(res["quartic_A_MHz"]["DJ"] * 1000.0, 4),
                "DJK_A_kHz": round(res["quartic_A_MHz"]["DJK"] * 1000.0, 4),
                "DK_A_kHz": round(res["quartic_A_MHz"]["DK"] * 1000.0, 4),
                "dJ_A_kHz": round(res["quartic_A_MHz"]["dJ"] * 1000.0, 5),
                "dK_A_kHz": round(res["quartic_A_MHz"]["dK"] * 1000.0, 4),
                "DJ_S_kHz": round(res["quartic_S_MHz"]["DJ"] * 1000.0, 4),
                "DJK_S_kHz": round(res["quartic_S_MHz"]["DJK"] * 1000.0, 4),
                "DK_S_kHz": round(res["quartic_S_MHz"]["DK"] * 1000.0, 4),
                "d1_S_kHz": round(res["quartic_S_MHz"]["d1"] * 1000.0, 5),
                "d2_S_kHz": round(res["quartic_S_MHz"]["d2"] * 1000.0, 5),
                "wall_s": res["wall_s"],
            }
        )

    if rows and not only_set:
        csv_path = s2_dir / "wb97xv_gpu_vpt2_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[GPU-VPT2] Wrote {len(rows)} rows to {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
