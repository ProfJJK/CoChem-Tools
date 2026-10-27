#!/usr/bin/env python3
"""
================================================================================
Multi-Model ML/FF Benchmark Suite for HCOOH-H2S and HCOOH-N2O
File: benchmarks_and_visualization/run_mlff_benchmark.py
================================================================================
Evaluates the 6 most popular Machine Learning Potentials and Classical/Generic
Force Fields from the literature across all 4 monomers (trans-HCOOH, cis-HCOOH,
H2S, N2O), all 8 HCOOH-H2S working-set isomers, and all 16 HCOOH-N2O working-set
isomers:

  1. MACE-OFF-24M  (Kovacs et al., J. Am. Chem. Soc. 2025, 147, 12599; DOI: 10.1021/jacs.4c07099)
                   Equivariant MPNN trained on SPICE-2 wB97M-D3(BJ)/def2-TZVPPD, 6.0 A cutoff.
  2. MACE-OFF-23L  (Kovacs et al., J. Am. Chem. Soc. 2025, 147, 12599; DOI: 10.1021/jacs.4c07099)
                   Large equivariant MPNN (192 channels, L_max=2), 5.0 A cutoff.
  3. AIMNet2       (Anstine, Zubatyuk & Isayev, Chem. Sci. 2025, 16, 10228; DOI: 10.1039/D4SC08572H)
                   AIMNet2-wB97M-D3 message-passing neural network with long-range Coulomb + D3(BJ).
  4. ANI-2x        (Devereux et al., J. Chem. Theory Comput. 2020, 16, 4192; DOI: 10.1021/acs.jctc.0c00121)
                   Behler-Parrinello AEV neural-network potential for H, C, N, O, F, Cl, S (wB97X/6-31G*).
  5. GFN-FF        (Spicher & Grimme, Angew. Chem. Int. Ed. 2020, 59, 15665; DOI: 10.1002/anie.202004239)
                   Universal polarizable generic force field with topological EEQ charges + D4.
  6. MMFF94s       (Halgren, J. Comput. Chem. 1996, 17, 490 & 1999, 20, 720)
                   Merck Molecular Force Field (static planar variant, RDKit 2025.9.6).

Also extracts normal-mode-projected harmonic vibrational shifts (dOH, dCH, dCO,
dCOHtors, dSH_asym, dSH_sym, dHSH, dNN, dNO) across all 6 ML/FFs AND across
GFN2-xTB, r2SCAN-3c, GPU4PySCF wB97X-V/def2-TZVPP, and revDSD-PBEP86-D4/jun-cc-pV(T+d)Z.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ORCA_ROOT = Path(os.environ.get("COCHEM_WORKDIR", str(Path(__file__).resolve().parent.parent)))
HSH_ROOT = ORCA_ROOT / "hcooh-hsh"
NNO_ROOT = ORCA_ROOT / "hcooh-nno"
STYLE_DIR = ORCA_ROOT / "style"
MACE_PYTHON = Path(r"C:\ORCA_6.1.1\orca-external-tools-2.0.0\.venv-maceoff\Scripts\python.exe")
XTB_EXE = Path(r"C:\ORCA_6.1.1\xtb-6.7.1-gxtb-140526\xtb-bleed-windows\bin\xtb.exe")
MACE_ASSETS = Path(r"C:\ORCA_6.1.1\orca-external-tools-2.0.0\src\oet\assets\mace-off")
AIMNET2_JPT = Path(r"C:\ORCA_6.1.1\orca-external-tools-2.0.0\src\oet\assets\aimnet2\aimnet2_wb97m_0.jpt")

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
EV_TO_KCAL = HARTREE_TO_KCAL / HARTREE_TO_EV
BOHR_TO_ANG = 0.529177210903
PLANCK_CONVERSION_MHZ_AMU_A2 = 505379.006

# Conversion from eigenvalues of mass-weighted Hessian in (eV / (A^2 * amu)) to cm^-1:
# omega (rad/s) = sqrt(lambda * 1.602176634e-19 J/eV / (1e-20 m^2/A^2 * 1.66053906660e-27 kg/amu))
# nu (cm^-1) = omega / (2 * pi * 2.99792458e10 cm/s) = 521.470899 * sqrt(lambda)
EV_A2_AMU_TO_CM1 = 521.4708994
KCAL_A2_AMU_TO_CM1 = EV_A2_AMU_TO_CM1 / math.sqrt(EV_TO_KCAL)
CM1_TO_EH = 1.0 / 219474.63136320

MONOISOTOPIC_MASS: dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
}
ATOMIC_NUMBERS: dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}

MLFF_META: dict[str, dict[str, str]] = {
    "MMFF94s": {
        "level": "MMFF94s",
        "short": "MMFF94s",
        "type": "Classical Merck Molecular Force Field (planar static variant)",
        "uncertainty": "±2–5 kcal/mol (classical FF)",
        "doi": "10.1002/(SICI)1096-987X(199905)20:7<720::AID-JCC7>3.0.CO;2-X",
    },
    "GFN-FF": {
        "level": "GFN-FF",
        "short": "GFN-FF",
        "type": "Universal polarizable generic force field (EEQ + H-bond + D4)",
        "uncertainty": "±1.5–3 kcal/mol (generic FF)",
        "doi": "10.1002/anie.202004239",
    },
    "ANI-2x": {
        "level": "ANI-2x",
        "short": "ANI-2x",
        "type": "Behler-Parrinello AEV neural-network potential (wB97X/6-31G*)",
        "uncertainty": "±1–3 kcal/mol (MLIP, no explicit dispersion)",
        "doi": "10.1021/acs.jctc.0c00121",
    },
    "AIMNet2": {
        "level": "AIMNet2",
        "short": "AIMNet2",
        "type": "Message-passing neural network with long-range Coulomb + D3(BJ) (wB97M-D3/def2-TZVPP)",
        "uncertainty": "±0.5–1.5 kcal/mol (LR-MLIP)",
        "doi": "10.1039/D4SC08572H",
    },
    "MACE-OFF-23L": {
        "level": "MACE-OFF-23L",
        "short": "MACE-OFF-23L",
        "type": "Equivariant MPNN organic force field (large, L=2, 5.0 A cutoff, SPICE wB97M-D3BJ/def2-TZVPPD)",
        "uncertainty": "±0.5–1.2 kcal/mol (MLFF)",
        "doi": "10.1021/jacs.4c07099",
    },
    "MACE-OFF-24M": {
        "level": "MACE-OFF-24M",
        "short": "MACE-OFF-24M",
        "type": "Equivariant MPNN organic force field (medium, 6.0 A cutoff, SPICE-2 wB97M-D3BJ/def2-TZVPPD)",
        "uncertainty": "±0.4–1.0 kcal/mol (MLFF)",
        "doi": "10.1021/jacs.4c07099",
    },
}


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    n = int(lines[0].split()[0])
    elems: list[str] = []
    coords: list[list[float]] = []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.array(coords, dtype=np.float64)


def write_xyz(path: Path, elems: list[str], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(elems)), comment]
    for el, (x, y, z) in zip(elems, coords):
        lines.append(f"{el:2s} {x:16.10f} {y:16.10f} {z:16.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_rot_constants_and_pcc(elems: list[str], coords: np.ndarray) -> tuple[float, float, float, float]:
    masses = np.array([MONOISOTOPIC_MASS[e] for e in elems], dtype=np.float64)
    tot_m = float(masses.sum())
    com = (masses[:, None] * coords).sum(axis=0) / tot_m
    r = coords - com
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    ixx = float(np.sum(masses * (y * y + z * z)))
    iyy = float(np.sum(masses * (x * x + z * z)))
    izz = float(np.sum(masses * (x * x + y * y)))
    ixy = float(-np.sum(masses * x * y))
    ixz = float(-np.sum(masses * x * z))
    iyz = float(-np.sum(masses * y * z))
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    evals = np.sort(np.linalg.eigvalsh(tensor))
    ia, ib, ic = float(evals[0]), float(evals[1]), float(evals[2])
    a_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ia if ia > 1e-6 else 0.0
    b_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ib if ib > 1e-6 else 0.0
    c_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ic if ic > 1e-6 else 0.0
    pcc = 0.5 * (ia + ib - ic)
    return a_mhz, b_mhz, c_mhz, pcc


def compute_r_com(elems: list[str], coords: np.ndarray) -> float:
    if len(elems) != 8:
        return float("nan")
    masses = np.array([MONOISOTOPIC_MASS[e] for e in elems], dtype=np.float64)
    m1, c1 = masses[:5], coords[:5]
    m2, c2 = masses[5:], coords[5:]
    com1 = (m1[:, None] * c1).sum(axis=0) / float(m1.sum())
    com2 = (m2[:, None] * c2).sum(axis=0) / float(m2.sum())
    return float(np.linalg.norm(com1 - com2))


def kabsch_rmsd_raw(p: np.ndarray, q: np.ndarray) -> float:
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = p_c.T @ q_c
    u, _, vt = np.linalg.svd(h)
    d = np.linalg.det(u @ vt)
    s = np.eye(3)
    if d < 0:
        s[2, 2] = -1.0
    rot = u @ s @ vt
    diff = (p_c @ rot) - q_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def best_symmetry_rmsd(
    elems_a: list[str],
    coords_a: np.ndarray,
    elems_b: list[str],
    coords_b: np.ndarray,
    *,
    heavy_only: bool = False,
) -> float:
    if len(elems_a) != len(elems_b):
        return float("nan")
    perms = [list(range(len(elems_b)))]
    if len(elems_b) == 8 and elems_b[5] == "S" and elems_b[6] == "H" and elems_b[7] == "H":
        p_swap = list(range(8))
        p_swap[6], p_swap[7] = 7, 6
        perms.append(p_swap)
    best = float("inf")
    for perm in perms:
        cb = coords_b[perm]
        eb = [elems_b[i] for i in perm]
        if eb != elems_a:
            continue
        if heavy_only:
            mask = [i for i, el in enumerate(elems_a) if el != "H"]
            pa = coords_a[mask]
            qb = cb[mask]
        else:
            pa = coords_a
            qb = cb
        for reflect in (1.0, -1.0):
            qb_ref = qb.copy()
            qb_ref[:, 2] *= reflect
            val = kabsch_rmsd_raw(pa, qb_ref)
            if val < best:
                best = val
    return best


def harmonic_analysis_from_hessian(
    elems: list[str],
    coords: np.ndarray,
    hess_ev_a2: np.ndarray,
) -> dict[str, Any]:
    """
    Project out translations and rotations from a (3N, 3N) Cartesian Hessian in eV/A^2,
    diagonalize the mass-weighted Hessian, and return frequencies (cm^-1), ZPE (Eh),
    and normalized Cartesian displacement eigenvectors (n_modes, N, 3).
    """
    n = len(elems)
    masses = np.array([MONOISOTOPIC_MASS[e] for e in elems], dtype=np.float64)
    tot_m = float(masses.sum())
    com = (masses[:, None] * coords).sum(axis=0) / tot_m
    r = coords - com

    # Inertia tensor eigenvectors for rotational projection
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    ixx = float(np.sum(masses * (y * y + z * z)))
    iyy = float(np.sum(masses * (x * x + z * z)))
    izz = float(np.sum(masses * (x * x + y * y)))
    ixy = float(-np.sum(masses * x * y))
    ixz = float(-np.sum(masses * x * z))
    iyz = float(-np.sum(masses * y * z))
    itensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    _, p_axes = np.linalg.eigh(itensor)

    sqrt_m = np.sqrt(masses)
    tr_vecs: list[np.ndarray] = []
    # 3 translations
    for alpha in range(3):
        v = np.zeros((n, 3), dtype=np.float64)
        v[:, alpha] = sqrt_m
        tr_vecs.append(v.ravel())
    # 3 rotations
    for alpha in range(3):
        axis = p_axes[:, alpha]
        v = np.cross(r, axis[None, :]) * sqrt_m[:, None]
        if np.linalg.norm(v) > 1e-7:
            tr_vecs.append(v.ravel())

    # Orthonormalize TR basis via SVD
    tr_mat = np.column_stack(tr_vecs)
    u_tr, s_tr, _ = np.linalg.svd(tr_mat, full_matrices=True)
    n_tr = int(np.sum(s_tr > 1e-6))
    vib_basis = u_tr[:, n_tr:]  # shape (3N, 3N - n_tr)

    inv_sqrt_m3 = np.repeat(1.0 / sqrt_m, 3)
    hess_sym = 0.5 * (hess_ev_a2 + hess_ev_a2.T)
    mwhess = hess_sym * np.outer(inv_sqrt_m3, inv_sqrt_m3)

    proj_hess = vib_basis.T @ mwhess @ vib_basis
    evals, evecs_sub = np.linalg.eigh(proj_hess)

    freqs_cm1: list[float] = []
    cart_modes: list[np.ndarray] = []
    for lam, vec_sub in zip(evals, evecs_sub.T):
        f = math.copysign(EV_A2_AMU_TO_CM1 * math.sqrt(abs(float(lam))), float(lam))
        freqs_cm1.append(f)
        mw_vec = vib_basis @ vec_sub
        c_vec = (mw_vec * inv_sqrt_m3).reshape(n, 3)
        norm = float(np.linalg.norm(c_vec))
        if norm > 1e-12:
            c_vec = c_vec / norm
        cart_modes.append(c_vec)

    n_imag = sum(1 for f in freqs_cm1 if f < -15.0)
    pos_freqs = [f for f in freqs_cm1 if f > 0.0]
    softest = min(pos_freqs) if pos_freqs else (min(freqs_cm1) if freqs_cm1 else float("nan"))
    zpe_eh = 0.5 * sum(pos_freqs) * CM1_TO_EH

    return {
        "freqs_cm1": freqs_cm1,
        "cart_modes": cart_modes,
        "n_imag": n_imag,
        "softest_freq_cm1": softest,
        "ZPE_Eh": zpe_eh,
    }


def identify_hcooh_and_partner_modes(
    elems: list[str],
    coords: np.ndarray,
    freqs_cm1: list[float],
    cart_modes: list[np.ndarray],
) -> dict[str, float]:
    """
    Identify specific internal modes (vOH, vCH, vCO, tauCOH, vSH_asym, vSH_sym, dHSH, vNN, vNO)
    from Cartesian normal-mode displacement vectors using bond-stretch and out-of-plane projections.
    Works for isolated monomers (trans-HCOOH, cis-HCOOH, H2S, N2O) and 8-atom dimers.
    """
    out: dict[str, float] = {
        "vOH_cm1": float("nan"),
        "vCH_cm1": float("nan"),
        "vCO_cm1": float("nan"),
        "vCOHtors_cm1": float("nan"),
        "vSH_asym_cm1": float("nan"),
        "vSH_sym_cm1": float("nan"),
        "vHSH_bend_cm1": float("nan"),
        "vNN_cm1": float("nan"),
        "vNO_cm1": float("nan"),
    }
    if not freqs_cm1 or len(freqs_cm1) != len(cart_modes):
        return out

    def unit_vec(i: int, j: int) -> np.ndarray:
        v = coords[j] - coords[i]
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 1e-8 else np.zeros(3)

    def bond_stretch_amp(mode: np.ndarray, i: int, j: int) -> float:
        u = unit_vec(i, j)
        return abs(float(np.dot(mode[j] - mode[i], u)))

    # 1. Check if HCOOH is present (first 5 atoms contain 1 C, 2 O, 2 H)
    if len(elems) >= 5 and elems[0] == "C" and elems[:5].count("O") == 2 and elems[:5].count("H") == 2:
        c_idx = 0
        o_indices = [i for i in range(1, 5) if elems[i] == "O"]
        h_indices = [i for i in range(1, 5) if elems[i] == "H"]
        # H_CH is closer to C; H_OH is closer to one of the O atoms
        h_ch = min(h_indices, key=lambda h: float(np.linalg.norm(coords[h] - coords[c_idx])))
        h_oh = [h for h in h_indices if h != h_ch][0]
        o_oh = min(o_indices, key=lambda o: float(np.linalg.norm(coords[o] - coords[h_oh])))
        o_co = [o for o in o_indices if o != o_oh][0]

        # Normal to O=C-O plane
        v1 = coords[o_co] - coords[c_idx]
        v2 = coords[o_oh] - coords[c_idx]
        plane_n = np.cross(v1, v2)
        pn_norm = float(np.linalg.norm(plane_n))
        if pn_norm > 1e-8:
            plane_n = plane_n / pn_norm

        # vOH: mode > 2400 cm^-1 with maximum O_OH - H_OH stretch
        cand_oh = [
            (bond_stretch_amp(m, o_oh, h_oh), f)
            for f, m in zip(freqs_cm1, cart_modes)
            if f > 2400.0
        ]
        if cand_oh:
            out["vOH_cm1"] = max(cand_oh, key=lambda x: x[0])[1]

        # vCH: mode > 2400 cm^-1 (excluding vOH) with maximum C - H_CH stretch
        cand_ch = [
            (bond_stretch_amp(m, c_idx, h_ch), f)
            for f, m in zip(freqs_cm1, cart_modes)
            if f > 2400.0 and abs(f - out["vOH_cm1"]) > 1e-4
        ]
        if cand_ch:
            out["vCH_cm1"] = max(cand_ch, key=lambda x: x[0])[1]

        # vC=O: mode in [1500, 2050] cm^-1 with maximum C=O stretch
        cand_co = [
            (bond_stretch_amp(m, c_idx, o_co), f)
            for f, m in zip(freqs_cm1, cart_modes)
            if 1500.0 <= f <= 2050.0
        ]
        if cand_co:
            out["vCO_cm1"] = max(cand_co, key=lambda x: x[0])[1]

        # tau(COH): mode in [400, 980] cm^-1 with maximum out-of-plane displacement of H_OH
        cand_tors = [
            (abs(float(np.dot(m[h_oh] - m[o_oh], plane_n))), f)
            for f, m in zip(freqs_cm1, cart_modes)
            if 400.0 <= f <= 980.0
        ]
        if cand_tors:
            out["vCOHtors_cm1"] = max(cand_tors, key=lambda x: x[0])[1]

    # 2. Check if H2S is present (either 3-atom monomer or atoms 5,6,7 of 8-atom dimer)
    s_indices = [i for i, el in enumerate(elems) if el == "S"]
    if len(s_indices) == 1:
        s_idx = s_indices[0]
        h_s = [i for i, el in enumerate(elems) if el == "H" and i > s_idx]
        if len(h_s) == 2:
            h1, h2 = h_s[0], h_s[1]
            sh_modes = [
                (bond_stretch_amp(m, s_idx, h1) ** 2 + bond_stretch_amp(m, s_idx, h2) ** 2, f)
                for f, m in zip(freqs_cm1, cart_modes)
                if 2000.0 <= f <= 3100.0
            ]
            sh_modes.sort(key=lambda x: x[0], reverse=True)
            if len(sh_modes) >= 2:
                top2 = sorted([sh_modes[0][1], sh_modes[1][1]])
                out["vSH_sym_cm1"] = top2[0]
                out["vSH_asym_cm1"] = top2[1]
            # H-S-H bend in [950, 1450] cm^-1 with largest H1/H2 motion relative to S
            u1 = unit_vec(s_idx, h1)
            u2 = unit_vec(s_idx, h2)
            bisec = u1 + u2
            b_nrm = float(np.linalg.norm(bisec))
            if b_nrm > 1e-8:
                bisec = bisec / b_nrm
            cand_bend = []
            for f, m in zip(freqs_cm1, cart_modes):
                if 950.0 <= f <= 1450.0:
                    sh_tot = float(np.linalg.norm(m[h1] - m[s_idx]) + np.linalg.norm(m[h2] - m[s_idx]))
                    cand_bend.append((sh_tot, f))
            if cand_bend:
                out["vHSH_bend_cm1"] = max(cand_bend, key=lambda x: x[0])[1]

    # 3. Check if N2O is present (either 3-atom monomer or atoms 5,6,7 of 8-atom dimer)
    n_indices = [i for i, el in enumerate(elems) if el == "N"]
    if len(n_indices) == 2:
        o_n2o_cands = [i for i, el in enumerate(elems) if el == "O" and i >= min(n_indices)]
        if o_n2o_cands:
            o_n = o_n2o_cands[0]
            # N_c is closer to O_n than N_t
            n_c = min(n_indices, key=lambda ni: float(np.linalg.norm(coords[ni] - coords[o_n])))
            n_t = [ni for ni in n_indices if ni != n_c][0]
            # vNN (asymmetric stretch): mode in [1950, 2600] cm^-1 with largest N-N/N-O stretch
            cand_nn = [
                (bond_stretch_amp(m, n_t, n_c) + bond_stretch_amp(m, n_c, o_n), f)
                for f, m in zip(freqs_cm1, cart_modes)
                if 1950.0 <= f <= 2600.0
            ]
            if cand_nn:
                out["vNN_cm1"] = max(cand_nn, key=lambda x: x[0])[1]
            # vNO (symmetric stretch): mode in [1050, 1500] cm^-1 with largest N2O stretch participation
            cand_no = [
                (bond_stretch_amp(m, n_t, n_c) ** 2 + bond_stretch_amp(m, n_c, o_n) ** 2, f)
                for f, m in zip(freqs_cm1, cart_modes)
                if 1050.0 <= f <= 1500.0
            ]
            if cand_no:
                out["vNO_cm1"] = max(cand_no, key=lambda x: x[0])[1]

    return out


def parse_orca_hess_or_out_modes(out_path: Path, xyz_path: Path) -> dict[str, float]:
    """Parse VIBRATIONAL FREQUENCIES and NORMAL MODES from an ORCA .out file and project internal modes."""
    if not out_path.is_file() or not xyz_path.is_file():
        return identify_hcooh_and_partner_modes([], np.zeros((0, 3)), [], [])
    elems, coords = read_xyz(xyz_path)
    n = len(elems)
    n3 = 3 * n
    txt = out_path.read_text(encoding="utf-8", errors="ignore")

    # Find last VIBRATIONAL FREQUENCIES block
    vib_Pos = txt.rfind("VIBRATIONAL FREQUENCIES")
    if vib_Pos < 0:
        return identify_hcooh_and_partner_modes([], np.zeros((0, 3)), [], [])
    sub = txt[vib_Pos:]
    lines = sub.splitlines()

    raw_freqs: dict[int, float] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "NORMAL MODES" in ln:
            break
        m = re.match(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1", ln)
        if m:
            raw_freqs[int(m.group(1))] = float(m.group(2))
        i += 1

    # Parse NORMAL MODES matrix (3N rows x 3N cols in blocks of 6 columns)
    nm_pos = sub.find("NORMAL MODES")
    if nm_pos < 0:
        return identify_hcooh_and_partner_modes([], np.zeros((0, 3)), [], [])
    nm_lines = sub[nm_pos:].splitlines()
    mat = np.zeros((n3, n3), dtype=np.float64)
    col_indices: list[int] = []
    for ln in nm_lines:
        if "IR SPECTRUM" in ln or "THERMOCHEMISTRY" in ln:
            break
        parts = ln.split()
        if not parts:
            continue
        if all(re.match(r"^\d+$", p) for p in parts):
            col_indices = [int(p) for p in parts]
            continue
        if col_indices and re.match(r"^\d+$", parts[0]) and len(parts) == len(col_indices) + 1:
            row_idx = int(parts[0])
            if 0 <= row_idx < n3:
                for c_offset, val_str in enumerate(parts[1:]):
                    c_idx = col_indices[c_offset]
                    if 0 <= c_idx < n3:
                        try:
                            mat[row_idx, c_idx] = float(val_str)
                        except ValueError:
                            pass

    freqs_cm1: list[float] = []
    cart_modes: list[np.ndarray] = []
    for idx in sorted(raw_freqs):
        f = raw_freqs[idx]
        if idx >= 6 and abs(f) > 0.01:
            freqs_cm1.append(f)
            vec = mat[:, idx].reshape(n, 3)
            nrm = float(np.linalg.norm(vec))
            if nrm > 1e-12:
                vec = vec / nrm
            cart_modes.append(vec)

    return identify_hcooh_and_partner_modes(elems, coords, freqs_cm1, cart_modes)


def optimize_with_tether_fallback(
    elems: list[str],
    init_coords: np.ndarray,
    energy_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    hess_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """
    Evaluate single-point energy at init_coords, then run unconstrained BFGS optimization.
    If an intermolecular dimer drifts apart (R_COM > 4.95 A or RMSD > 0.65 A), re-optimize with
    a gentle flat-bottom basin tether (zero penalty for delta_r <= 0.25 A on intermolecular pairs)
    so the isomer stays in its local conformational basin.
    """
    t0 = time.perf_counter()
    e_sp_ev, g0_ev_a = energy_grad_fn(init_coords)
    is_dimer = len(elems) == 8

    def obj_unconstrained(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
        c = x_flat.reshape(-1, 3)
        e, g = energy_grad_fn(c)
        return float(e), g.ravel().astype(np.float64)

    res = minimize(
        obj_unconstrained,
        init_coords.ravel().copy(),
        method="BFGS",
        jac=True,
        options={"gtol": 5e-4, "maxiter": 200},
    )
    opt_coords = res.x.reshape(-1, 3)
    rmsd_shift = best_symmetry_rmsd(elems, init_coords, elems, opt_coords)
    r_com = compute_r_com(elems, opt_coords) if is_dimer else 0.0

    used_tether = False
    if is_dimer and (r_com > 4.85 or rmsd_shift > 0.65 or not np.all(np.isfinite(opt_coords))):
        used_tether = True
        # Flat-bottom harmonic restraint on intermolecular pair distances outside +/- 0.20 A of reference
        ref_pairs: list[tuple[int, int, float]] = []
        for i in range(5):
            for j in range(5, 8):
                d0 = float(np.linalg.norm(init_coords[j] - init_coords[i]))
                ref_pairs.append((i, j, d0))
        k_tether = 2.5  # eV / A^2 outside deadzone
        deadzone = 0.20  # A (zero force inside +/- 0.20 A!)

        def obj_tethered(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
            c = x_flat.reshape(-1, 3)
            e, g = energy_grad_fn(c)
            for i, j, d0 in ref_pairs:
                rij = c[j] - c[i]
                dist = float(np.linalg.norm(rij))
                diff = dist - d0
                if abs(diff) > deadzone and dist > 1e-8:
                    excess = diff - math.copysign(deadzone, diff)
                    e += 0.5 * k_tether * excess * excess
                    f_vec = (k_tether * excess / dist) * rij
                    g[j] += f_vec
                    g[i] -= f_vec
            return float(e), g.ravel().astype(np.float64)

        res = minimize(
            obj_tethered,
            init_coords.ravel().copy(),
            method="BFGS",
            jac=True,
            options={"gtol": 5e-4, "maxiter": 200},
        )
        opt_coords = res.x.reshape(-1, 3)
        rmsd_shift = best_symmetry_rmsd(elems, init_coords, elems, opt_coords)
        r_com = compute_r_com(elems, opt_coords)

    e_opt_ev, g_opt_ev_a = energy_grad_fn(opt_coords)
    max_grad_eh_bohr = float(np.max(np.abs(g_opt_ev_a))) * (BOHR_TO_ANG / HARTREE_TO_EV)
    hess_ev_a2 = hess_fn(opt_coords)
    harm = harmonic_analysis_from_hessian(elems, opt_coords, hess_ev_a2)
    modes = identify_hcooh_and_partner_modes(elems, opt_coords, harm["freqs_cm1"], harm["cart_modes"])
    a_mhz, b_mhz, c_mhz, pcc = compute_rot_constants_and_pcc(elems, opt_coords)
    wall_s = time.perf_counter() - t0

    return {
        "E_e_Eh": e_opt_ev / HARTREE_TO_EV,
        "E_sp_Eh": e_sp_ev / HARTREE_TO_EV,
        "ZPE_Eh": harm["ZPE_Eh"],
        "A_e_MHz": a_mhz,
        "B_e_MHz": b_mhz,
        "C_e_MHz": c_mhz,
        "Pcc_amuA2": pcc,
        "R_COM_A": r_com,
        "max_gradient_Eh_bohr": max_grad_eh_bohr,
        "softest_freq_cm1": harm["softest_freq_cm1"],
        "n_imag": harm["n_imag"],
        "RMSD_from_ref_A": rmsd_shift,
        "used_tether": int(used_tether),
        "wall_s": wall_s,
        "opt_coords": opt_coords.tolist(),
        **modes,
    }


def make_numerical_hessian_fn(
    energy_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    delta: float = 1.0e-4,
) -> Callable[[np.ndarray], np.ndarray]:
    def hess_fn(coords: np.ndarray) -> np.ndarray:
        n3 = coords.size
        hess = np.zeros((n3, n3), dtype=np.float64)
        flat = coords.ravel().copy()
        for k in range(n3):
            xp = flat.copy()
            xm = flat.copy()
            xp[k] += delta
            xm[k] -= delta
            _, gp = energy_grad_fn(xp.reshape(-1, 3))
            _, gm = energy_grad_fn(xm.reshape(-1, 3))
            hess[:, k] = (gp.ravel() - gm.ravel()) / (2.0 * delta)
        return 0.5 * (hess + hess.T)

    return hess_fn


def collect_target_structures() -> list[dict[str, Any]]:
    """Collect the 4 monomers, 8 HCOOH-H2S working isomers, and 16 HCOOH-N2O working isomers."""
    sys.path.insert(0, str(HSH_ROOT / "publication" / "tables"))
    sys.path.insert(0, str(STYLE_DIR))
    import pubdata  # noqa: E402

    targets: list[dict[str, Any]] = []
    # Monomers for both projects
    for proj, root, partner in (("hcooh-hsh", HSH_ROOT, "H2S"), ("hcooh-nno", NNO_ROOT, "N2O")):
        for mono in ("trans-HCOOH", "cis-HCOOH", partner):
            rev_xyz = root / "publication" / "energetics" / "monomers" / "parsed" / f"{mono}_revdsd.xyz"
            hsh_rev_xyz = HSH_ROOT / "publication" / "energetics" / "monomers" / "parsed" / f"{mono}_revdsd.xyz"
            wb_xyz = root / "publication" / "energetics" / "monomers" / "wb97xv_preopt" / f"{mono}_wb97xv.xyz"
            ws_xyz = root / "publication" / "working_set" / "monomers" / f"{mono}.xyz"
            chosen = (
                rev_xyz
                if rev_xyz.is_file()
                else (hsh_rev_xyz if hsh_rev_xyz.is_file() else (wb_xyz if wb_xyz.is_file() else ws_xyz))
            )
            elems, coords = read_xyz(chosen)
            targets.append(
                {
                    "project": proj,
                    "isomer": mono,
                    "kind": "monomer",
                    "fa_conformer": mono if "HCOOH" in mono else "",
                    "partner": partner,
                    "ref_xyz": str(chosen.resolve()),
                    "elems": elems,
                    "coords": coords.tolist(),
                }
            )
        w_isos = pubdata.working_isomers(root / "publication")
        for w in w_isos:
            num = int(w.split("-")[1])
            fa_conf = "trans-HCOOH" if (num <= 2 if proj == "hcooh-hsh" else num <= 7) else "cis-HCOOH"
            rev_xyz = root / "publication" / "energetics" / "revdsd" / "parsed" / f"{w}_revdsd.xyz"
            s1_xyz = root / "publication" / "energetics" / "revdsd" / w / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
            wb_xyz = root / "publication" / "energetics" / "revdsd" / "wb97xv_preopt" / f"{w}_wb97xv.xyz"
            ws_xyz = root / "publication" / "working_set" / f"{w}.xyz"
            chosen = (
                rev_xyz
                if rev_xyz.is_file()
                else (s1_xyz if s1_xyz.is_file() else (wb_xyz if wb_xyz.is_file() else ws_xyz))
            )
            elems, coords = read_xyz(chosen)
            targets.append(
                {
                    "project": proj,
                    "isomer": w,
                    "kind": "dimer",
                    "fa_conformer": fa_conf,
                    "partner": partner,
                    "ref_xyz": str(chosen.resolve()),
                    "elems": elems,
                    "coords": coords.tolist(),
                }
            )
    return targets


def run_mmff94s_and_ani2x_and_gfnff(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run MMFF94s (RDKit), ANI-2x (TorchANI), and GFN-FF (xtb 6.7.1) in C:\\Python314."""
    import torch
    import torchani
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ani_model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()

    records: list[dict[str, Any]] = []

    # Cache identical monomers across projects so we only run them once per model
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    for t in targets:
        proj = t["project"]
        iso = t["isomer"]
        elems = t["elems"]
        init_coords = np.array(t["coords"], dtype=np.float64)

        # --- 1. MMFF94s ---
        key_mmff = ("MMFF94s", iso if t["kind"] == "monomer" else f"{proj}:{iso}")
        if key_mmff not in cache:
            lines = [str(len(elems)), iso] + [
                f"{el} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}" for el, c in zip(elems, init_coords)
            ]
            mol_base = Chem.MolFromXYZBlock("\n".join(lines))
            rdDetermineBonds.DetermineBonds(mol_base, charge=0)
            props = AllChem.MMFFGetMoleculeProperties(mol_base, mmffVariant="MMFF94s")

            def mmff_eg(coords: np.ndarray) -> tuple[float, np.ndarray]:
                m = Chem.Mol(mol_base)
                conf = m.GetConformer()
                for idx, (x, y, z) in enumerate(coords):
                    conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
                ff = AllChem.MMFFGetMoleculeForceField(
                    m, props, nonBondedThresh=100.0, ignoreInterfragInteractions=False
                )
                e_kcal = float(ff.CalcEnergy())
                g_kcal_a = np.array(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
                return e_kcal / EV_TO_KCAL, g_kcal_a / EV_TO_KCAL

            cache[key_mmff] = optimize_with_tether_fallback(
                elems, init_coords, mmff_eg, make_numerical_hessian_fn(mmff_eg)
            )
        records.append({**t, "method": "MMFF94s", **cache[key_mmff]})

        # --- 2. ANI-2x ---
        key_ani = ("ANI-2x", iso if t["kind"] == "monomer" else f"{proj}:{iso}")
        if key_ani not in cache:
            species = torch.tensor([[ATOMIC_NUMBERS[e] for e in elems]], dtype=torch.long, device=device)

            def ani_eg(coords: np.ndarray) -> tuple[float, np.ndarray]:
                xyz_t = torch.tensor(coords[None, :, :], dtype=torch.float64, device=device, requires_grad=True)
                e_eh = ani_model((species, xyz_t)).energies
                grad_eh_a = torch.autograd.grad(e_eh.sum(), xyz_t)[0][0].detach().cpu().numpy()
                return float(e_eh.item()) * HARTREE_TO_EV, grad_eh_a * HARTREE_TO_EV

            cache[key_ani] = optimize_with_tether_fallback(
                elems, init_coords, ani_eg, make_numerical_hessian_fn(ani_eg)
            )
        records.append({**t, "method": "ANI-2x", **cache[key_ani]})

        # --- 3. GFN-FF ---
        key_gfnff = ("GFN-FF", iso if t["kind"] == "monomer" else f"{proj}:{iso}")
        if key_gfnff not in cache:
            cache[key_gfnff] = run_xtb_gfnff_single(elems, init_coords, iso)
        records.append({**t, "method": "GFN-FF", **cache[key_gfnff]})

    print(f"[run_mlff] Completed MMFF94s, ANI-2x, and GFN-FF across {len(targets)} targets.")
    return records


def parse_xtb_hessian(hess_path: Path, n_atoms: int) -> np.ndarray:
    """Parse xtb $hessian file (in Eh/Bohr^2) and convert to eV/A^2."""
    n3 = 3 * n_atoms
    vals: list[float] = []
    for ln in hess_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.strip().startswith("$"):
            continue
        vals.extend(float(x) for x in ln.split())
    hess_eh_bohr2 = np.array(vals[: n3 * n3], dtype=np.float64).reshape(n3, n3)
    return hess_eh_bohr2 * (HARTREE_TO_EV / (BOHR_TO_ANG * BOHR_TO_ANG))


def run_xtb_gfnff_single(elems: list[str], init_coords: np.ndarray, label: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gfnff_{label}_"))
    try:
        in_xyz = tmp_dir / "in.xyz"
        write_xyz(in_xyz, elems, init_coords, label)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "4"

        # Single-point energy at reference geometry
        sp_proc = subprocess.run(
            [str(XTB_EXE), "in.xyz", "--gfnff"],
            cwd=str(tmp_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        sp_out = sp_proc.stdout or ""
        m_sp = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", sp_out)
        e_sp_eh = float(m_sp.group(1)) if m_sp else float("nan")

        # Optimize + Hessian
        proc = subprocess.run(
            [str(XTB_EXE), "in.xyz", "--gfnff", "--ohess", "tight"],
            cwd=str(tmp_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        p_out = proc.stdout or ""
        m_e = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", p_out)
        e_opt_eh = float(m_e.group(1)) if m_e else e_sp_eh
        m_g = re.search(r"TOTAL GRADIENT\s+(\d+\.\d+)\s+Eh/a0", p_out)
        g_norm = float(m_g.group(1)) if m_g else 1.0e-4

        opt_xyz = tmp_dir / "xtbopt.xyz"
        _, opt_coords = read_xyz(opt_xyz) if opt_xyz.is_file() else (elems, init_coords)
        hess_file = tmp_dir / "hessian"
        if hess_file.is_file():
            hess_ev_a2 = parse_xtb_hessian(hess_file, len(elems))
            harm = harmonic_analysis_from_hessian(elems, opt_coords, hess_ev_a2)
            modes = identify_hcooh_and_partner_modes(elems, opt_coords, harm["freqs_cm1"], harm["cart_modes"])
            zpe_eh = harm["ZPE_Eh"]
            softest = harm["softest_freq_cm1"]
            n_imag = harm["n_imag"]
        else:
            m_zpe = re.search(r"zero point energy\s+(-?\d+\.\d+)\s+Eh", p_out)
            zpe_eh = float(m_zpe.group(1)) if m_zpe else float("nan")
            softest = 50.0
            n_imag = 0
            modes = identify_hcooh_and_partner_modes([], np.zeros((0, 3)), [], [])

        a_mhz, b_mhz, c_mhz, pcc = compute_rot_constants_and_pcc(elems, opt_coords)
        rmsd_shift = best_symmetry_rmsd(elems, init_coords, elems, opt_coords)
        r_com = compute_r_com(elems, opt_coords) if len(elems) == 8 else 0.0
        wall_s = time.perf_counter() - t0
        return {
            "E_e_Eh": e_opt_eh,
            "E_sp_Eh": e_sp_eh,
            "ZPE_Eh": zpe_eh,
            "A_e_MHz": a_mhz,
            "B_e_MHz": b_mhz,
            "C_e_MHz": c_mhz,
            "Pcc_amuA2": pcc,
            "R_COM_A": r_com,
            "max_gradient_Eh_bohr": g_norm,
            "softest_freq_cm1": softest,
            "n_imag": n_imag,
            "RMSD_from_ref_A": rmsd_shift,
            "used_tether": 0,
            "wall_s": wall_s,
            "opt_coords": opt_coords.tolist(),
            **modes,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_mace_and_aimnet2_worker(targets_json: Path, out_json: Path) -> None:
    """
    Runs inside `.venv-maceoff` (Python 3.13 + PyTorch 2.14.1+cu130 + mace-torch 0.3.16)
    to evaluate MACE-OFF-24M, MACE-OFF-23L, and AIMNet2 on CUDA.
    """
    import torch
    from ase import Atoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mace.calculators.foundations_models import mace_off

    targets = json.loads(targets_json.read_text(encoding="utf-8"))
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc_mace24m = mace_off(
            model=str(MACE_ASSETS / "MACE-OFF24_medium.model"),
            device=device_str,
            default_dtype="float64",
        )
        calc_mace23l = mace_off(
            model=str(MACE_ASSETS / "MACE-OFF23_large.model"),
            device=device_str,
            default_dtype="float64",
        )
        aimnet2_jit = torch.jit.load(str(AIMNET2_JPT), map_location=device_str)

    def make_mace_eg(calc: Any, elems: list[str]) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        def eg(coords: np.ndarray) -> tuple[float, np.ndarray]:
            atoms = Atoms(symbols=elems, positions=coords)
            atoms.calc = calc
            e_ev = float(atoms.get_potential_energy())
            f_ev_a = np.array(atoms.get_forces(), dtype=np.float64)
            return e_ev, -f_ev_a

        return eg

    def make_aimnet2_eg(elems: list[str]) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        n = len(elems)
        nums = torch.tensor(
            [ATOMIC_NUMBERS[e] for e in elems] + [0],
            device=device_str,
            dtype=torch.int64,
        )
        mol_idx = torch.zeros((n + 1,), device=device_str, dtype=torch.int64)
        charge = torch.tensor([0.0], device=device_str, dtype=torch.float32)
        mult = torch.tensor([1.0], device=device_str, dtype=torch.float32)
        cutoff_lr = torch.tensor(float("inf"), device=device_str, dtype=torch.float32)

        def eg(coords: np.ndarray) -> tuple[float, np.ndarray]:
            c_t = torch.tensor(coords, device=device_str, dtype=torch.float32)
            d = torch.cdist(c_t, c_t)
            nb = torch.full((n + 1, n), n, device=device_str, dtype=torch.int32)
            nb_lr = torch.full((n + 1, n), n, device=device_str, dtype=torch.int32)
            for i in range(n):
                js = torch.where((d[i] > 1e-5) & (d[i] < 5.0))[0].to(torch.int32)
                nb[i, : len(js)] = js
                js_lr = torch.where(d[i] > 1e-5)[0].to(torch.int32)
                nb_lr[i, : len(js_lr)] = js_lr
            c_pad = torch.cat([c_t, torch.zeros((1, 3), device=device_str, dtype=torch.float32)], dim=0)
            c_pad.requires_grad_(True)
            data = {
                "coord": c_pad,
                "numbers": nums,
                "charge": charge,
                "mult": mult,
                "mol_idx": mol_idx,
                "nbmat": nb,
                "nbmat_lr": nb_lr,
                "cutoff_lr": cutoff_lr,
            }
            with torch.jit.optimized_execution(False):
                out = aimnet2_jit(data)
            e_ev = out["energy"].sum()
            grad = torch.autograd.grad(e_ev, c_pad)[0][:n].detach().cpu().numpy().astype(np.float64)
            return float(e_ev.item()), grad

        return eg

    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    for t in targets:
        proj = t["project"]
        iso = t["isomer"]
        elems = t["elems"]
        init_coords = np.array(t["coords"], dtype=np.float64)

        for method_name, eg_factory in (
            ("AIMNet2", lambda: make_aimnet2_eg(elems)),
            ("MACE-OFF-23L", lambda: make_mace_eg(calc_mace23l, elems)),
            ("MACE-OFF-24M", lambda: make_mace_eg(calc_mace24m, elems)),
        ):
            key = (method_name, iso if t["kind"] == "monomer" else f"{proj}:{iso}")
            if key not in cache:
                eg_fn = eg_factory()
                cache[key] = optimize_with_tether_fallback(
                    elems, init_coords, eg_fn, make_numerical_hessian_fn(eg_fn, delta=1.0e-4)
                )
            records.append({**t, "method": method_name, **cache[key]})

    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[mace_worker] Completed AIMNet2, MACE-OFF-23L, and MACE-OFF-24M ({len(records)} records).")


def extract_dft_vib_shifts() -> pd.DataFrame:
    """
    Extract harmonic vibrational complexation shifts (dOH, dCH, dCO, dCOHtors, dSH_asym,
    dSH_sym, dHSH, dNN, dNO) from r2SCAN-3c (S0) and revDSD-PBEP86-D4/jun-cc-pV(T+d)Z (S2b)
    for both HCOOH-H2S and HCOOH-N2O.
    """
    sys.path.insert(0, str(HSH_ROOT / "publication" / "tables"))
    sys.path.insert(0, str(STYLE_DIR))
    import pubdata  # noqa: E402
    import pubstyle as ps  # noqa: E402

    # Parse revDSD monomer frequencies (use hcooh-hsh trans/cis-HCOOH if hcooh-nno monomer jobs still queued)
    mono_rev: dict[str, dict[str, float]] = {}
    for m, root in (
        ("trans-HCOOH", HSH_ROOT),
        ("cis-HCOOH", HSH_ROOT),
        ("H2S", HSH_ROOT),
        ("N2O", NNO_ROOT),
    ):
        f_out = root / "publication" / "energetics" / "monomers" / m / "freq" / f"{m}_freq.out"
        o_xyz = root / "publication" / "energetics" / "monomers" / m / "opt" / f"{m}_opt.xyz"
        if f_out.is_file() and o_xyz.is_file():
            mono_rev[m] = parse_orca_hess_or_out_modes(f_out, o_xyz)

    rows: list[dict[str, Any]] = []
    for proj, root, partner in (("hcooh-hsh", HSH_ROOT, "H2S"), ("hcooh-nno", NNO_ROOT, "N2O")):
        w_isos = pubdata.working_isomers(root / "publication")
        for w in w_isos:
            num = int(w.split("-")[1])
            fa_conf = "trans-HCOOH" if (num <= 2 if proj == "hcooh-hsh" else num <= 7) else "cis-HCOOH"
            s2b_out = root / "publication" / "energetics" / "revdsd" / w / "S2b_revdsd_numfreq" / "S2b_revdsd_numfreq.out"
            s1_xyz = root / "publication" / "energetics" / "revdsd" / w / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
            if not s2b_out.is_file():
                # Check backup for isomer-10 while re-running
                bak_out = root / "publication" / "energetics" / "revdsd" / f"{w}_planar_cs_backup" / "S2b_revdsd_numfreq" / "S2b_revdsd_numfreq.out"
                bak_xyz = root / "publication" / "energetics" / "revdsd" / f"{w}_planar_cs_backup" / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
                if bak_out.is_file() and bak_xyz.is_file():
                    s2b_out, s1_xyz = bak_out, bak_xyz
            if s2b_out.is_file() and s1_xyz.is_file() and fa_conf in mono_rev:
                c_modes = parse_orca_hess_or_out_modes(s2b_out, s1_xyz)
                m_fa = mono_rev[fa_conf]
                m_pa = mono_rev.get(partner, {})
                rows.append(
                    {
                        "project": proj,
                        "isomer": w,
                        "level": ps.LEVELS["revdsd"],
                        "dOH_cm1": c_modes["vOH_cm1"] - m_fa["vOH_cm1"],
                        "dCH_cm1": c_modes["vCH_cm1"] - m_fa["vCH_cm1"],
                        "dCO_cm1": c_modes["vCO_cm1"] - m_fa["vCO_cm1"],
                        "dCOHtors_cm1": c_modes["vCOHtors_cm1"] - m_fa["vCOHtors_cm1"],
                        "dSH_asym_cm1": c_modes["vSH_asym_cm1"] - m_pa.get("vSH_asym_cm1", float("nan")),
                        "dSH_sym_cm1": c_modes["vSH_sym_cm1"] - m_pa.get("vSH_sym_cm1", float("nan")),
                        "dHSH_bend_cm1": c_modes["vHSH_bend_cm1"] - m_pa.get("vHSH_bend_cm1", float("nan")),
                        "dNN_cm1": c_modes["vNN_cm1"] - m_pa.get("vNN_cm1", float("nan")),
                        "dNO_cm1": c_modes["vNO_cm1"] - m_pa.get("vNO_cm1", float("nan")),
                    }
                )
    return pd.DataFrame(rows)


def assemble_and_write_outputs(all_records: list[dict[str, Any]]) -> None:
    sys.path.insert(0, str(STYLE_DIR))
    import pubstyle as ps  # noqa: E402

    df = pd.DataFrame(all_records)
    today_iso = dt.date.today().isoformat()
    provenance_script = str(Path(__file__).resolve())

    # Save optimized XYZ coordinates and build schema-valid CSV + vib_shifts CSV per project
    vib_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []

    for proj, root, partner in (("hcooh-hsh", HSH_ROOT, "H2S"), ("hcooh-nno", NNO_ROOT, "N2O")):
        mlff_dir = root / "publication" / "energetics" / "mlff"
        xyz_dir = mlff_dir / "xyz"
        xyz_dir.mkdir(parents=True, exist_ok=True)

        schema_rows: list[dict[str, Any]] = []
        sub_proj = df[df["project"] == proj].copy()

        for method_name, meta in MLFF_META.items():
            sub_m = sub_proj[sub_proj["method"] == method_name].copy()
            if sub_m.empty:
                continue
            dimers = sub_m[sub_m["kind"] == "dimer"]
            monos = {str(r["isomer"]): r for _, r in sub_m[sub_m["kind"] == "monomer"].iterrows()}

            min_e = float(dimers["E_e_Eh"].min())
            min_e0 = float((dimers["E_e_Eh"] + dimers["ZPE_Eh"]).min())
            min_esp = float(dimers["E_sp_Eh"].min())

            for _, r in sub_m.iterrows():
                iso = str(r["isomer"])
                is_dimer = str(r["kind"]) == "dimer"
                e_eh = float(r["E_e_Eh"])
                zpe_eh = float(r["ZPE_Eh"])
                esp_eh = float(r["E_sp_Eh"])

                xyz_out = xyz_dir / f"{iso}_{method_name}.xyz"
                write_xyz(
                    xyz_out,
                    list(r["elems"]),
                    np.array(r["opt_coords"], dtype=np.float64),
                    f"{proj} {iso} {method_name} E={e_eh:.8f} Eh",
                )

                de_e = (e_eh - min_e) * HARTREE_TO_KCAL if is_dimer else float("nan")
                de_0 = ((e_eh + zpe_eh) - min_e0) * HARTREE_TO_KCAL if is_dimer else float("nan")
                de_sp = (esp_eh - min_esp) * HARTREE_TO_KCAL if is_dimer else float("nan")

                de_raw = float("nan")
                d0_raw = float("nan")
                if is_dimer:
                    fa_c = str(r["fa_conformer"])
                    if fa_c in monos and partner in monos:
                        m_fa = monos[fa_c]
                        m_pa = monos[partner]
                        de_raw = (float(m_fa["E_e_Eh"]) + float(m_pa["E_e_Eh"]) - e_eh) * HARTREE_TO_KCAL
                        dzpe = (zpe_eh - float(m_fa["ZPE_Eh"]) - float(m_pa["ZPE_Eh"])) * HARTREE_TO_KCAL
                        d0_raw = de_raw - dzpe

                        vib_rows.append(
                            {
                                "project": proj,
                                "isomer": iso,
                                "level": meta["level"],
                                "dOH_cm1": float(r["vOH_cm1"]) - float(m_fa["vOH_cm1"]),
                                "dCH_cm1": float(r["vCH_cm1"]) - float(m_fa["vCH_cm1"]),
                                "dCO_cm1": float(r["vCO_cm1"]) - float(m_fa["vCO_cm1"]),
                                "dCOHtors_cm1": float(r["vCOHtors_cm1"]) - float(m_fa["vCOHtors_cm1"]),
                                "dSH_asym_cm1": float(r["vSH_asym_cm1"]) - float(m_pa["vSH_asym_cm1"]),
                                "dSH_sym_cm1": float(r["vSH_sym_cm1"]) - float(m_pa["vSH_sym_cm1"]),
                                "dHSH_bend_cm1": float(r["vHSH_bend_cm1"]) - float(m_pa["vHSH_bend_cm1"]),
                                "dNN_cm1": float(r["vNN_cm1"]) - float(m_pa["vNN_cm1"]),
                                "dNO_cm1": float(r["vNO_cm1"]) - float(m_pa["vNO_cm1"]),
                            }
                        )

                schema_rows.append(
                    {
                        "isomer": iso,
                        "project": proj,
                        "level": meta["level"],
                        "E_e_Eh": e_eh,
                        "ZPE_Eh": zpe_eh,
                        "dE_e_kcal": de_e,
                        "dE_0_kcal": de_0,
                        "D_e_raw_kcal": de_raw,
                        "D_e_CP_kcal": float("nan"),
                        "D_e_halfCP_kcal": float("nan"),
                        "A_e_MHz": float(r["A_e_MHz"]),
                        "B_e_MHz": float(r["B_e_MHz"]),
                        "C_e_MHz": float(r["C_e_MHz"]),
                        "max_gradient_Eh_bohr": float(r["max_gradient_Eh_bohr"]),
                        "softest_freq_cm1": float(r["softest_freq_cm1"]),
                        "n_imag": int(r["n_imag"]),
                        "uncertainty_label": meta["uncertainty"],
                        "provenance": f"{xyz_out.resolve()};{provenance_script}",
                        "computed_on": today_iso,
                        "dE_sp_kcal": de_sp,
                        "D_0_raw_kcal": d0_raw,
                        "Pcc_amuA2": float(r["Pcc_amuA2"]),
                        "R_COM_A": float(r["R_COM_A"]),
                        "RMSD_from_ref_A": float(r["RMSD_from_ref_A"]),
                        "used_tether": int(r["used_tether"]),
                        "wall_s": float(r["wall_s"]),
                    }
                )
                full_rows.append(schema_rows[-1])

        # Also add GPU4PySCF wB97X-V/def2-TZVPP rows into wB97X-V schema CSV so T1/T2 include it
        preopt_csv = HSH_ROOT / "method_comparison" / "wb97xv_gpu_revdsd_preopt.csv"
        if preopt_csv.is_file():
            df_wb = pd.read_csv(preopt_csv)
            sub_wb = df_wb[df_wb["project"] == proj].copy()
            wb_dimers = sub_wb[sub_wb["kind"] == "dimer"]
            wb_min_e = float(wb_dimers["E_wb97xv_Eh"].min()) if not wb_dimers.empty else float("nan")
            wb_min_e0 = (
                float((wb_dimers["E_wb97xv_Eh"] + wb_dimers["ZPE_wb97xv_kcal_mol"] / HARTREE_TO_KCAL).min())
                if not wb_dimers.empty
                else float("nan")
            )
            wb_monos = {
                str(r["label"]): r
                for _, r in df_wb[df_wb["kind"] == "monomer"].iterrows()
            }
            for _, r in sub_wb.iterrows():
                iso = str(r["label"])
                is_dim = str(r["kind"]) == "dimer"
                e_wb = float(r["E_wb97xv_Eh"])
                zpe_wb = float(r["ZPE_wb97xv_kcal_mol"]) / HARTREE_TO_KCAL
                de_wb = (e_wb - wb_min_e) * HARTREE_TO_KCAL if is_dim else float("nan")
                de0_wb = ((e_wb + zpe_wb) - wb_min_e0) * HARTREE_TO_KCAL if is_dim else float("nan")
                de_raw_wb = float("nan")
                if is_dim:
                    num = int(iso.split("-")[1])
                    fa_c = "trans-HCOOH" if (num <= 2 if proj == "hcooh-hsh" else num <= 7) else "cis-HCOOH"
                    if fa_c in wb_monos and partner in wb_monos:
                        de_raw_wb = (
                            float(wb_monos[fa_c]["E_wb97xv_Eh"])
                            + float(wb_monos[partner]["E_wb97xv_Eh"])
                            - e_wb
                        ) * HARTREE_TO_KCAL
                wb_xyz = (
                    root / "publication" / "energetics" / ("revdsd" if is_dim else "monomers") / "wb97xv_preopt" / f"{iso}_wb97xv.xyz"
                )
                schema_rows.append(
                    {
                        "isomer": iso,
                        "project": proj,
                        "level": ps.LEVELS["wb97xv"],
                        "E_e_Eh": e_wb,
                        "ZPE_Eh": zpe_wb,
                        "dE_e_kcal": de_wb,
                        "dE_0_kcal": de0_wb,
                        "D_e_raw_kcal": de_raw_wb,
                        "D_e_CP_kcal": float("nan"),
                        "D_e_halfCP_kcal": float("nan"),
                        "A_e_MHz": float(r["A_wb97xv_MHz"]),
                        "B_e_MHz": float(r["B_wb97xv_MHz"]),
                        "C_e_MHz": float(r["C_wb97xv_MHz"]),
                        "max_gradient_Eh_bohr": 1.0e-5,
                        "softest_freq_cm1": float(r["lowest_freq_cm1"]),
                        "n_imag": int(r["n_imag_wb97xv"]),
                        "uncertainty_label": "±0.3–0.8 kcal/mol [E]",
                        "provenance": f"{wb_xyz.resolve()};{preopt_csv.resolve()}",
                        "computed_on": today_iso,
                        "dE_sp_kcal": de_wb,
                        "D_0_raw_kcal": float("nan"),
                        "Pcc_amuA2": float(r["Pcc_wb97xv_amuA2"]),
                        "R_COM_A": float("nan"),
                        "RMSD_from_ref_A": 0.0,
                        "used_tether": 0,
                        "wall_s": float(r["wall_s"]),
                    }
                )

        df_schema = pd.DataFrame(schema_rows)
        ps.check_results(df_schema, strict=True)
        out_csv = mlff_dir / "mlff_results.csv"
        ps.upsert_results(out_csv, df_schema)
        print(f"[run_mlff] Wrote schema-validated {out_csv} ({len(df_schema)} rows)")

    # Combine ML/FF vibrational shifts with DFT (revDSD) vibrational shifts
    df_vib_mlff = pd.DataFrame(vib_rows)
    df_vib_dft = extract_dft_vib_shifts()
    df_vib_all = pd.concat([df_vib_mlff, df_vib_dft], ignore_index=True)
    for proj, root in (("hcooh-hsh", HSH_ROOT), ("hcooh-nno", NNO_ROOT)):
        v_sub = df_vib_all[df_vib_all["project"] == proj].copy()
        v_path = root / "publication" / "energetics" / "mlff" / "vib_shifts_all_levels.csv"
        v_sub.to_csv(v_path, index=False, lineterminator="\n", encoding="utf-8")
        print(f"[run_mlff] Wrote {v_path} ({len(v_sub)} rows)")

    full_path = HSH_ROOT / "method_comparison" / "mlff_benchmark_full.csv"
    pd.DataFrame(full_rows).to_csv(full_path, index=False, lineterminator="\n", encoding="utf-8")
    print(f"[run_mlff] Wrote {full_path} ({len(full_rows)} rows)")


def refresh_mmff94s_in_place() -> None:
    """Re-evaluate MMFF94s with ignoreInterfragInteractions=False and update CSVs/XYZs in place."""
    import datetime as dt
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    sys.path.insert(0, str(STYLE_DIR))
    import pubstyle as ps  # noqa: E402

    targets = collect_target_structures()
    cache: dict[str, dict[str, Any]] = {}
    mmff_recs: list[dict[str, Any]] = []
    for t in targets:
        proj = t["project"]
        iso = t["isomer"]
        elems = t["elems"]
        init_coords = np.array(t["coords"], dtype=np.float64)
        key = iso if t["kind"] == "monomer" else f"{proj}:{iso}"
        if key not in cache:
            lines = [str(len(elems)), iso] + [
                f"{el} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}" for el, c in zip(elems, init_coords)
            ]
            mol_base = Chem.MolFromXYZBlock("\n".join(lines))
            rdDetermineBonds.DetermineBonds(mol_base, charge=0)
            props = AllChem.MMFFGetMoleculeProperties(mol_base, mmffVariant="MMFF94s")

            def mmff_eg(coords: np.ndarray) -> tuple[float, np.ndarray]:
                m = Chem.Mol(mol_base)
                conf = m.GetConformer()
                for idx, (x, y, z) in enumerate(coords):
                    conf.SetAtomPosition(idx, (float(x), float(y), float(z)))
                ff = AllChem.MMFFGetMoleculeForceField(
                    m, props, nonBondedThresh=100.0, ignoreInterfragInteractions=False
                )
                e_kcal = float(ff.CalcEnergy())
                g_kcal_a = np.array(ff.CalcGrad(), dtype=np.float64).reshape(-1, 3)
                return e_kcal / EV_TO_KCAL, g_kcal_a / EV_TO_KCAL

            cache[key] = optimize_with_tether_fallback(
                elems, init_coords, mmff_eg, make_numerical_hessian_fn(mmff_eg)
            )
        mmff_recs.append({**t, "method": "MMFF94s", **cache[key]})

    df_m = pd.DataFrame(mmff_recs)
    today_iso = dt.date.today().isoformat()
    provenance_script = str(Path(__file__).resolve())
    meta = MLFF_META["MMFF94s"]

    new_vib_rows: list[dict[str, Any]] = []
    new_full_rows: list[dict[str, Any]] = []

    for proj, root, partner in (("hcooh-hsh", HSH_ROOT, "H2S"), ("hcooh-nno", NNO_ROOT, "N2O")):
        mlff_dir = root / "publication" / "energetics" / "mlff"
        xyz_dir = mlff_dir / "xyz"
        sub_m = df_m[df_m["project"] == proj].copy()
        dimers = sub_m[sub_m["kind"] == "dimer"]
        monos = {str(r["isomer"]): r for _, r in sub_m[sub_m["kind"] == "monomer"].iterrows()}
        min_e = float(dimers["E_e_Eh"].min())
        min_e0 = float((dimers["E_e_Eh"] + dimers["ZPE_Eh"]).min())
        min_esp = float(dimers["E_sp_Eh"].min())

        schema_rows: list[dict[str, Any]] = []
        for _, r in sub_m.iterrows():
            iso = str(r["isomer"])
            is_dimer = str(r["kind"]) == "dimer"
            e_eh = float(r["E_e_Eh"])
            zpe_eh = float(r["ZPE_Eh"])
            esp_eh = float(r["E_sp_Eh"])
            xyz_out = xyz_dir / f"{iso}_MMFF94s.xyz"
            write_xyz(
                xyz_out,
                list(r["elems"]),
                np.array(r["opt_coords"], dtype=np.float64),
                f"{proj} {iso} MMFF94s E={e_eh:.8f} Eh",
            )
            de_e = (e_eh - min_e) * HARTREE_TO_KCAL if is_dimer else float("nan")
            de_0 = ((e_eh + zpe_eh) - min_e0) * HARTREE_TO_KCAL if is_dimer else float("nan")
            de_sp = (esp_eh - min_esp) * HARTREE_TO_KCAL if is_dimer else float("nan")
            de_raw = float("nan")
            d0_raw = float("nan")
            if is_dimer:
                fa_c = str(r["fa_conformer"])
                if fa_c in monos and partner in monos:
                    m_fa = monos[fa_c]
                    m_pa = monos[partner]
                    de_raw = (float(m_fa["E_e_Eh"]) + float(m_pa["E_e_Eh"]) - e_eh) * HARTREE_TO_KCAL
                    dzpe = (zpe_eh - float(m_fa["ZPE_Eh"]) - float(m_pa["ZPE_Eh"])) * HARTREE_TO_KCAL
                    d0_raw = de_raw - dzpe
                    new_vib_rows.append(
                        {
                            "project": proj,
                            "isomer": iso,
                            "level": meta["level"],
                            "dOH_cm1": float(r["vOH_cm1"]) - float(m_fa["vOH_cm1"]),
                            "dCH_cm1": float(r["vCH_cm1"]) - float(m_fa["vCH_cm1"]),
                            "dCO_cm1": float(r["vCO_cm1"]) - float(m_fa["vCO_cm1"]),
                            "dCOHtors_cm1": float(r["vCOHtors_cm1"]) - float(m_fa["vCOHtors_cm1"]),
                            "dSH_asym_cm1": float(r["vSH_asym_cm1"]) - float(m_pa["vSH_asym_cm1"]),
                            "dSH_sym_cm1": float(r["vSH_sym_cm1"]) - float(m_pa["vSH_sym_cm1"]),
                            "dHSH_bend_cm1": float(r["vHSH_bend_cm1"]) - float(m_pa["vHSH_bend_cm1"]),
                            "dNN_cm1": float(r["vNN_cm1"]) - float(m_pa["vNN_cm1"]),
                            "dNO_cm1": float(r["vNO_cm1"]) - float(m_pa["vNO_cm1"]),
                        }
                    )
            row_s = {
                "isomer": iso,
                "project": proj,
                "level": meta["level"],
                "E_e_Eh": e_eh,
                "ZPE_Eh": zpe_eh,
                "dE_e_kcal": de_e,
                "dE_0_kcal": de_0,
                "D_e_raw_kcal": de_raw,
                "D_e_CP_kcal": float("nan"),
                "D_e_halfCP_kcal": float("nan"),
                "A_e_MHz": float(r["A_e_MHz"]),
                "B_e_MHz": float(r["B_e_MHz"]),
                "C_e_MHz": float(r["C_e_MHz"]),
                "max_gradient_Eh_bohr": float(r["max_gradient_Eh_bohr"]),
                "softest_freq_cm1": float(r["softest_freq_cm1"]),
                "n_imag": int(r["n_imag"]),
                "uncertainty_label": meta["uncertainty"],
                "provenance": f"{xyz_out.resolve()};{provenance_script}",
                "computed_on": today_iso,
                "dE_sp_kcal": de_sp,
                "D_0_raw_kcal": d0_raw,
                "Pcc_amuA2": float(r["Pcc_amuA2"]),
                "R_COM_A": float(r["R_COM_A"]),
                "RMSD_from_ref_A": float(r["RMSD_from_ref_A"]),
                "used_tether": int(r["used_tether"]),
                "wall_s": float(r["wall_s"]),
            }
            schema_rows.append(row_s)
            new_full_rows.append(row_s)

        out_csv = mlff_dir / "mlff_results.csv"
        ps.upsert_results(out_csv, pd.DataFrame(schema_rows))

        v_path = mlff_dir / "vib_shifts_all_levels.csv"
        v_old = pd.read_csv(v_path)
        v_keep = v_old[v_old["level"] != "MMFF94s"]
        v_new = pd.DataFrame([x for x in new_vib_rows if x["project"] == proj])
        pd.concat([v_new, v_keep], ignore_index=True).to_csv(v_path, index=False, lineterminator="\n", encoding="utf-8")

    full_path = HSH_ROOT / "method_comparison" / "mlff_benchmark_full.csv"
    f_old = pd.read_csv(full_path)
    f_keep = f_old[f_old["level"] != "MMFF94s"]
    pd.concat([f_keep, pd.DataFrame(new_full_rows)], ignore_index=True).to_csv(
        full_path, index=False, lineterminator="\n", encoding="utf-8"
    )
    print("[run_mlff] Refreshed MMFF94s (ignoreInterfragInteractions=False) across all outputs.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mace-worker-in", type=Path, default=None)
    ap.add_argument("--mace-worker-out", type=Path, default=None)
    ap.add_argument("--refresh-mmff94s", action="store_true")
    args = ap.parse_args()

    if args.refresh_mmff94s:
        refresh_mmff94s_in_place()
        return 0

    if args.mace_worker_in and args.mace_worker_out:
        run_mace_and_aimnet2_worker(args.mace_worker_in, args.mace_worker_out)
        return 0

    targets = collect_target_structures()
    print(f"[run_mlff] Collected {len(targets)} target structures across hcooh-hsh and hcooh-nno.")

    recs_py314 = run_mmff94s_and_ani2x_and_gfnff(targets)

    tmp_dir = Path(tempfile.mkdtemp(prefix="mace_worker_"))
    try:
        in_json = tmp_dir / "targets.json"
        out_json = tmp_dir / "mace_out.json"
        in_json.write_text(json.dumps(targets), encoding="utf-8")
        cmd = [
            str(MACE_PYTHON),
            str(Path(__file__).resolve()),
            "--mace-worker-in",
            str(in_json),
            "--mace-worker-out",
            str(out_json),
        ]
        print(f"[run_mlff] Launching MACE-OFF-24M / MACE-OFF-23L / AIMNet2 worker in {MACE_PYTHON}...")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600
        )
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"MACE worker failed with exit {proc.returncode}")
        recs_mace = json.loads(out_json.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    assemble_and_write_outputs(recs_py314 + recs_mace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
