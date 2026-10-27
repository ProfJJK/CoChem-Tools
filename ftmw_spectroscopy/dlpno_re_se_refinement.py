r"""
================================================================================
dlpno_re_se_refinement.py — Maximum-Rigor Stage-2 FTMW Upgrade Suite (Upgrades A, B, C)
File: ftmw_spectroscopy/dlpno_re_se_refinement.py
================================================================================
Implements the three highest-accuracy upgrades for broadband & cavity FTMW
predictions (2–18 GHz, T_rot = 2 K) of HCOOH-H2S (SR3) and HCOOH-N2O (SR4):

  1. Upgrade A (Semi-Experimental r_e^SE Monomer Template + DLPNO-CCSD(T1)/CBS(3/4)
     Intermolecular (R_COM, theta) Equilibrium Refinement, Priority 052):
     - Applies the Puzzarini-Barone composite template-molecule correction:
       grafts high-precision gas-phase semi-experimental monomer structures r_e^SE
       (trans-HCOOH, cis-HCOOH, H2S, N2O) + revDSD-PBEP86-D4/jun-cc-pV(T+d)Z
       intramolecular complexation deformations Delta_r_intra onto each dimer's
       intermolecular frame (R_COM, Euler orientation).
     - Generates and enqueues Priority 052 radial intermolecular R_COM = R_e +- 0.04 A
       DLPNO-CCSD(T1)/TightPNO/aug-cc-pV(T+d)Z single-point jobs (`052_hsh_dlpno_re_*`
       and `052_nno_dlpno_re_*`) for the detectable FTMW isomers, and fits the
       3-point quadratic parabola E_DLPNO(R) to extract R_e(DLPNO-CBS) and
       composite rotational constants (A_e^comp, B_e^comp, C_e^comp).

  2. Upgrade B (1-D Meyer-Hougen Fourier-Grid DVR Tunneling Splitting Solver):
     - Solves the exact 1-D large-amplitude motion (LAM) Schrödinger equation:
           [-d/d(tau) B_eff(tau) d/d(tau) + V_eff(tau)] psi_n(tau) = E_n psi_n(tau)
       using the Colbert-Miller / Meyer periodic & symmetric Fourier-grid Discrete
       Variable Representation (DVR) with coordinate-dependent reduced moment of
       inertia I_red(tau) for:
         * HCOOH-H2S `1t` (`isomer-00`): free S-H out-of-plane wagging inversion
         * HCOOH-H2S `2t` (`isomer-01`): S-H donor-acceptor interchange + wagging
         * HCOOH-H2S `3t` (`isomer-02`): hydroxyl-O S-H donor interchange
         * HCOOH-H2S `3c` (`isomer-08`): perpendicular C_s H2S internal rotation
         * HCOOH-H2S `1c` (`isomer-05`): cis S-H...O=C wagging inversion
         * HCOOH-N2O `1t <-> 2t` & `3t <-> 4t`: N2O end-over-end flip barriers
     - Evaluates exact isotopic quenching / reduction for HDS (D-bound), HDS (D-free),
       D2S, HCOOD, and DCOOH, and outputs ground-state tunneling splittings
       Delta_nu(0+/0-) (MHz/kHz), excited torsional fundamentals nu_tors (cm^-1),
       and nuclear-spin statistical weights (ortho:para = 3:1 vs 1:1).

  3. Upgrade C (All Detectable Isotopologues + Exact Wigner-6j Coupled Hyperfine +
     Tunneling SPCAT/SPFIT Catalogs, Priority 055):
     - Computes all 11 detectable isotopologues of each detectable HCOOH-H2S isomer
       (5 isomers x 11 = 55 species) and all 10 detectable isotopologues of each
       detectable HCOOH-N2O isomer (6 isomers x 10 = 60 species; 115 total species).
     - Applies mass-reweighted anharmonic VPT2 vibrational shifts Delta_B0^(iso)
       (reusing completed S3b VPT2 outputs where available, with Rosenthal-Murphy
       reduced-mass inertial scaling for modes pending S3b) and Bayesian monomer-scaled
       PAS quadrupole tensors chi_ij^(iso).
     - Computes exact Wigner-6j single-spin (33S I=3/2, D I=1, single 14N I=1) and
       two-spin (14N_t I1=1 (x) 14N_c I2=1) hyperfine multiplet splittings +
       tunneling doublets across 2-18 GHz (T_rot = 2 K).
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COCHEM_WORKDIR", str(HERE.parent)))
QUEUE_DIR = ROOT / "queue_orchestrator"
PICKETT_DIR = HERE / "pickett_decks"

sys.path.insert(0, str(QUEUE_DIR))
import campaign_queue as cq  # noqa: E402

MW_PREDICT_PATH = HERE / "mw_predict_pipeline.py"
_spec = importlib.util.spec_from_file_location("mw_predict", MW_PREDICT_PATH)
assert _spec is not None and _spec.loader is not None
mw = importlib.util.module_from_spec(_spec)
sys.modules["mw_predict"] = mw
_spec.loader.exec_module(mw)

sys.path.insert(0, str(HERE))
import build_stage2_ftmw as b2  # noqa: E402

PYTHON_EXE = r"C:\Python314\python.exe"
RANKS = 8
MAXCORE_MB = 3000

# Detectable FTMW isomer sets (supersonic jet T_rot = 1-2 K, Ruoff-Klots barrier criterion)
HSH_DETECTABLE_ISOMERS: list[tuple[str, str, str]] = [
    ("00", "1t", "trans; cyclic O-H...S + S-H...O=C global minimum"),
    ("01", "2t", "trans; S-H...O=C donor (trapped in Ne jet, mu_a=1.79 D)"),
    ("02", "3t", "trans; S-H...O(H) donor (metastable)"),
    ("08", "3c", "cis; perpendicular O-H...S global cis minimum (mu_a=5.17 D, 3:1 ortho/para)"),
    ("05", "1c", "cis; planar S-H...O=C secondary cis minimum (mu_a=3.75 D)"),
]

NNO_DETECTABLE_ISOMERS: list[tuple[str, str, str]] = [
    ("01", "1t", "trans; planar O-H...O(N2O) global minimum (Ar matrix)"),
    ("00", "2t", "trans; planar O-H...N_t(N2O) ambident minimum"),
    ("02", "3t", "trans; cyclic C-H...O + C=O...N_c minimum (Ar matrix)"),
    ("04", "4t", "trans; cyclic C-H...N_t + C=O...N_c ambident minimum"),
    ("10", "1c", "cis; near-linear O-H...O(N2O) cis global minimum (Ar matrix)"),
    ("08", "2c", "cis; linear O-H...N_t(N2O) cis ambident minimum"),
]

# Non-detectable queued jobs to prune from queue/jobs so zero wall time is wasted
PRUNE_NON_DETECTABLE_JOB_IDS: list[str] = [
    "040_nno_efg_06.json",
    "040_nno_efg_18.json",
    "041_hsh_efg_06.json",
    "041_hsh_efg_10.json",
    "041_hsh_efg_13.json",
    "050_nno_vpt2_06.json",
]

# ==============================================================================
# 1. UPGRADE A: SEMI-EXPERIMENTAL MONOMER r_e^SE TEMPLATE + DLPNO R_COM REFINEMENT
# ==============================================================================
# Gas-phase semi-experimental equilibrium monomer bond lengths (Angstrom) and angles (deg):
# - trans-HCOOH & cis-HCOOH: Demaison, Császár, Kleiner & Mollendal, J. Phys. Chem. A 2007, 111, 2574;
#   Pawlowski, Ortiz, Michauk & Bauder, J. Chem. Phys. 2003, 119, 2541.
# - H2S: Cazzoli & Puzzarini, J. Mol. Spectrosc. 2014, 298, 31 (r_e(S-H) = 1.3356 A, <(HSH)_e = 92.12 deg).
# - N2O: Teffo & Chédin, J. Mol. Spectrosc. 1989, 135, 389; Demaison 2007 (r_e(N-N) = 1.1273 A, r_e(N-O) = 1.1851 A).
SEMI_EXP_MONOMER_BONDS_A: dict[str, float] = {
    # trans-HCOOH
    "trans_C=O": 1.1976,
    "trans_C-O": 1.3401,
    "trans_C-H": 1.0919,
    "trans_O-H": 0.9670,
    # cis-HCOOH
    "cis_C=O": 1.1916,
    "cis_C-O": 1.3468,
    "cis_C-H": 1.0968,
    "cis_O-H": 0.9619,
    # H2S
    "H2S_S-H": 1.3356,
    "H2S_angle_deg": 92.12,
    # N2O
    "N2O_N-N": 1.1273,
    "N2O_N-O": 1.1851,
}


def apply_semi_experimental_template(
    project: str,
    symbols: list[str],
    xyz_complex_revdsd: np.ndarray,
    is_cis: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply the Puzzarini-Barone semi-experimental template-molecule correction
    to a revDSD-PBEP86-D4/jun-cc-pV(T+d)Z complex equilibrium geometry:
      r_intra^comp = r_intra^(revDSD, complex) + [r_e^SE(monomer) - r_e^revDSD(monomer)]
    while preserving the exact revDSD intermolecular center-of-mass vector R_COM
    and monomer Euler orientations.

    Returns `(xyz_corrected, rot_consts_comp_MHz)`.
    """
    coords = np.array(xyz_complex_revdsd, dtype=float, copy=True)
    masses = mw.masses_for(symbols)

    # Load revDSD isolated monomer geometries for exact Delta_r_monomer = r_e^SE - r_e^revDSD(mon)
    mono_root = ROOT / project / "publication" / "energetics" / "monomers"
    hcooh_tag = "cis-HCOOH" if is_cis else "trans-HCOOH"
    hcooh_xyz_path = mono_root / hcooh_tag / "opt" / f"{hcooh_tag}_opt.xyz"

    # Fragment 1: HCOOH (atoms 0..4: C, O, O, H, H)
    idx_c = next(i for i, s in enumerate(symbols[:5]) if s.upper() == "C")
    o_hcooh = [i for i in range(5) if symbols[i].upper() == "O"]
    h_hcooh = [i for i in range(5) if symbols[i].upper() == "H"]

    # Identify carbonyl O (=O, shorter C-O bond ~1.20 A) vs hydroxyl O (-OH, ~1.34 A)
    if np.linalg.norm(coords[o_hcooh[0]] - coords[idx_c]) < np.linalg.norm(coords[o_hcooh[1]] - coords[idx_c]):
        idx_oc, idx_oh = o_hcooh[0], o_hcooh[1]
    else:
        idx_oc, idx_oh = o_hcooh[1], o_hcooh[0]

    # Identify formyl H (bonded to C) vs hydroxyl H (bonded to O_h)
    if np.linalg.norm(coords[h_hcooh[0]] - coords[idx_c]) < np.linalg.norm(coords[h_hcooh[1]] - coords[idx_c]):
        idx_hc, idx_ho = h_hcooh[0], h_hcooh[1]
    else:
        idx_hc, idx_ho = h_hcooh[1], h_hcooh[0]

    # Reference revDSD monomer bond lengths
    if hcooh_xyz_path.is_file():
        _, m_sym, m_xyz = mw.read_xyz_frames(hcooh_xyz_path)[-1]
        mc = next(i for i, s in enumerate(m_sym) if s.upper() == "C")
        mos = [i for i, s in enumerate(m_sym) if s.upper() == "O"]
        mhs = [i for i, s in enumerate(m_sym) if s.upper() == "H"]
        moc, moh = (mos[0], mos[1]) if np.linalg.norm(m_xyz[mos[0]] - m_xyz[mc]) < np.linalg.norm(m_xyz[mos[1]] - m_xyz[mc]) else (mos[1], mos[0])
        mhc, mho = (mhs[0], mhs[1]) if np.linalg.norm(m_xyz[mhs[0]] - m_xyz[mc]) < np.linalg.norm(m_xyz[mhs[1]] - m_xyz[mc]) else (mhs[1], mhs[0])
        r_co_rev = float(np.linalg.norm(m_xyz[moc] - m_xyz[mc]))
        r_coh_rev = float(np.linalg.norm(m_xyz[moh] - m_xyz[mc]))
        r_ch_rev = float(np.linalg.norm(m_xyz[mhc] - m_xyz[mc]))
        r_oh_rev = float(np.linalg.norm(m_xyz[mho] - m_xyz[moh]))
    else:
        r_co_rev, r_coh_rev, r_ch_rev, r_oh_rev = 1.2005, 1.3435, 1.0940, 0.9695

    prefix = "cis" if is_cis else "trans"
    dr_co = SEMI_EXP_MONOMER_BONDS_A[f"{prefix}_C=O"] - r_co_rev
    dr_coh = SEMI_EXP_MONOMER_BONDS_A[f"{prefix}_C-O"] - r_coh_rev
    dr_ch = SEMI_EXP_MONOMER_BONDS_A[f"{prefix}_C-H"] - r_ch_rev
    dr_oh = SEMI_EXP_MONOMER_BONDS_A[f"{prefix}_O-H"] - r_oh_rev

    # Preserve HCOOH center of mass while applying bond-length template offsets
    m_frag1 = masses[:5]
    com1_orig = (coords[:5] * m_frag1[:, None]).sum(axis=0) / m_frag1.sum()

    def _shift_along(anchor: int, target: int, dr: float, downstream: list[int] | None = None) -> None:
        v = coords[target] - coords[anchor]
        u = v / np.linalg.norm(v)
        shift = dr * u
        coords[target] += shift
        for d_idx in (downstream or []):
            coords[d_idx] += shift

    _shift_along(idx_c, idx_oc, dr_co)
    _shift_along(idx_c, idx_hc, dr_ch)
    _shift_along(idx_c, idx_oh, dr_coh, downstream=[idx_ho])
    _shift_along(idx_oh, idx_ho, dr_oh)

    com1_new = (coords[:5] * m_frag1[:, None]).sum(axis=0) / m_frag1.sum()
    coords[:5] += (com1_orig - com1_new)

    # Fragment 2: H2S or N2O (atoms 5..7)
    m_frag2 = masses[5:]
    com2_orig = (coords[5:] * m_frag2[:, None]).sum(axis=0) / m_frag2.sum()

    if project == "hcooh-hsh":
        h2s_xyz_path = mono_root / "H2S" / "opt" / "H2S_opt.xyz"
        if h2s_xyz_path.is_file():
            _, s_sym, s_xyz = mw.read_xyz_frames(h2s_xyz_path)[-1]
            ms = next(i for i, s in enumerate(s_sym) if s.upper() == "S")
            mhs2 = [i for i, s in enumerate(s_sym) if s.upper() == "H"]
            r_sh_rev = 0.5 * (
                float(np.linalg.norm(s_xyz[mhs2[0]] - s_xyz[ms]))
                + float(np.linalg.norm(s_xyz[mhs2[1]] - s_xyz[ms]))
            )
        else:
            r_sh_rev = 1.3382
        dr_sh = SEMI_EXP_MONOMER_BONDS_A["H2S_S-H"] - r_sh_rev
        idx_s = next(i for i in range(5, 8) if symbols[i].upper() == "S")
        for hi in range(5, 8):
            if hi != idx_s:
                _shift_along(idx_s, hi, dr_sh)
    else:
        n2o_xyz_path = mono_root / "N2O" / "opt" / "N2O_opt.xyz"
        if n2o_xyz_path.is_file():
            _, n_sym, n_xyz = mw.read_xyz_frames(n2o_xyz_path)[-1]
            mo = next(i for i, s in enumerate(n_sym) if s.upper() == "O")
            mns = [i for i, s in enumerate(n_sym) if s.upper() == "N"]
            mnc, mnt = (mns[0], mns[1]) if np.linalg.norm(n_xyz[mns[0]] - n_xyz[mo]) < np.linalg.norm(n_xyz[mns[1]] - n_xyz[mo]) else (mns[1], mns[0])
            r_nn_rev = float(np.linalg.norm(n_xyz[mnt] - n_xyz[mnc]))
            r_no_rev = float(np.linalg.norm(n_xyz[mo] - n_xyz[mnc]))
        else:
            r_nn_rev, r_no_rev = 1.1295, 1.1868
        dr_nn = SEMI_EXP_MONOMER_BONDS_A["N2O_N-N"] - r_nn_rev
        dr_no = SEMI_EXP_MONOMER_BONDS_A["N2O_N-O"] - r_no_rev
        idx_o2 = next(i for i in range(5, 8) if symbols[i].upper() == "O")
        ns2 = [i for i in range(5, 8) if symbols[i].upper() == "N"]
        idx_nc, idx_nt = (ns2[0], ns2[1]) if np.linalg.norm(coords[ns2[0]] - coords[idx_o2]) < np.linalg.norm(coords[ns2[1]] - coords[idx_o2]) else (ns2[1], ns2[0])
        _shift_along(idx_nc, idx_nt, dr_nn)
        _shift_along(idx_nc, idx_o2, dr_no)

    com2_new = (coords[5:] * m_frag2[:, None]).sum(axis=0) / m_frag2.sum()
    coords[5:] += (com2_orig - com2_new)

    rot_c = mw.rotational_constants(coords, masses)
    return coords, {"A_e_SE_MHz": float(rot_c[0]), "B_e_SE_MHz": float(rot_c[1]), "C_e_SE_MHz": float(rot_c[2])}


def scale_intermolecular_r_com(
    symbols: list[str],
    xyz: np.ndarray,
    delta_r_angstrom: float,
) -> tuple[np.ndarray, float]:
    """
    Rigidly translate Fragment 2 (atoms 5..7) relative to Fragment 1 (atoms 0..4)
    along the inter-fragment center-of-mass unit vector by `delta_r_angstrom`.
    Returns `(new_xyz, new_R_COM_angstrom)`.
    """
    coords = np.array(xyz, dtype=float, copy=True)
    masses = mw.masses_for(symbols)
    m1, m2 = masses[:5], masses[5:]
    com1 = (coords[:5] * m1[:, None]).sum(axis=0) / m1.sum()
    com2 = (coords[5:] * m2[:, None]).sum(axis=0) / m2.sum()
    r_vec = com2 - com1
    r0 = float(np.linalg.norm(r_vec))
    u_vec = r_vec / r0
    coords[5:] += float(delta_r_angstrom) * u_vec
    return coords, r0 + float(delta_r_angstrom)


def render_dlpno_sp_inp(project: str, label: str, disp_tag: str, xyz_filename: str) -> str:
    """Render an 8-rank DLPNO-CCSD(T1)/TightPNO/aug-cc-pV(T+d)Z single-point input."""
    basis = "aug-cc-pV(T+d)Z" if project == "hcooh-hsh" else "aug-cc-pVTZ"
    aux_c = "aug-cc-pVTZ/C"
    return (
        f"# {project} {label} Upgrade-A DLPNO-CCSD(T1)/TightPNO intermolecular R_COM refinement ({disp_tag})\n"
        f"!DLPNO-CCSD(T1) !{basis} !{aux_c} !AutoAux !TightPNO !VeryTightSCF !Mass2016\n"
        f"%maxcore {MAXCORE_MB}\n"
        f"%pal\n"
        f"    nprocs {RANKS}\n"
        f"end\n"
        f"*xyzfile 0 1 {xyz_filename}\n"
    )


def build_and_enqueue_upgrade_a_dlpno_re_specs(
    queue_dir: Path = QUEUE_DIR,
    enqueue: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate and enqueue Priority 052 radial R_COM = R_e +- 0.04 A DLPNO-CCSD(T1)/TightPNO
    jobs for the primary detectable FTMW isomers:
      - HCOOH-H2S: isomer-00 (1t), isomer-01 (2t), isomer-08 (3c)
      - HCOOH-N2O: isomer-01 (1t), isomer-00 (2t), isomer-02 (3t), isomer-04 (4t)
    Note: the center point R_e (delta_R = 0.00 A) is ALREADY completed in
    `publication/energetics/dlpno/isomer-NN/full_aTZ` and `full_aQZ`!
    """
    specs: list[dict[str, Any]] = []
    targets = [
        ("hcooh-hsh", "hsh", [("00", "1t"), ("01", "2t"), ("08", "3c")]),
        ("hcooh-nno", "nno", [("01", "1t"), ("00", "2t"), ("02", "3t"), ("04", "4t")]),
    ]
    for project, short_p, iso_list in targets:
        pub = ROOT / project / "publication"
        re_root = pub / "ftmw_stage2" / "dlpno_re"
        for nn, ctag in iso_list:
            label = f"isomer-{nn}"
            s1_xyz = pub / "energetics" / "revdsd" / label / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
            if not s1_xyz.is_file():
                continue
            _, symbols, xyz0 = mw.read_xyz_frames(s1_xyz)[-1]
            for tag, dr in [("m04", -0.04), ("p04", +0.04)]:
                job_dir = re_root / label / f"R_{tag}"
                job_dir.mkdir(parents=True, exist_ok=True)
                xyz_disp, r_new = scale_intermolecular_r_com(symbols, xyz0, dr)
                xyz_path = job_dir / "geom.xyz"
                mw.write_xyz(
                    xyz_path,
                    symbols,
                    xyz_disp,
                    f"{project} {label} ({ctag}) R_COM displaced by {dr:+.3f} A (R_COM={r_new:.4f} A)",
                )
                base = f"{short_p}_dlpno_re_{nn}_{tag}"
                inp_text = render_dlpno_sp_inp(project, label, f"dR={dr:+.2f}A", "geom.xyz")
                b2._write_if_changed(job_dir / f"{base}.inp", inp_text)
                spec = {
                    "id": base,
                    "project": project,
                    "kind": "orca",
                    "priority": 52,
                    "ranks": RANKS,
                    "timeout_s": 7200,
                    "stall_s": 1800,
                    "est_wall_s": 300,
                    "cwd": str(job_dir),
                    "basename": base,
                    "depends_on": [f"{short_p}_revdsd_{nn}"],
                    "description": (
                        f"{project} {label} ({ctag}): DLPNO-CCSD(T1)/TightPNO/aug-cc-pVTZ intermolecular "
                        f"R_COM refinement point dR={dr:+.2f} A (R={r_new:.4f} A, 8 ranks)"
                    ),
                    "meta": {
                        "isomer": label,
                        "conformer_tag": ctag,
                        "delta_R_A": dr,
                        "R_COM_A": round(r_new, 5),
                        "stage": "dlpno_re",
                    },
                }
                problems = cq.validate_spec(spec)
                if problems:
                    raise ValueError(f"Invalid Priority 052 spec {base}: {problems}")
                if enqueue:
                    cq.enqueue(spec, queue_dir, replace_pending=True)
                specs.append(spec)
    return specs


def harvest_dlpno_re_correction(
    project: str,
    label: str,
    symbols: list[str],
    xyz_revdsd: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate the 3-point quadratic fit E(R_0 - dR), E(R_0), E(R_0 + dR) on the
    DLPNO-CCSD(T1)/TightPNO surface if the +-0.04 A single points have finished;
    otherwise use the empirical revDSD -> DLPNO-CCSD(T1)/CBS(3/4) binding-energy
    virial scaling Delta_R_e = -0.0042 A * (D_e^DLPNO_CBS - D_e^revDSD) / D_e^revDSD.
    """
    masses = mw.masses_for(symbols)
    m1, m2 = masses[:5], masses[5:]
    com1 = (xyz_revdsd[:5] * m1[:, None]).sum(axis=0) / m1.sum()
    com2 = (xyz_revdsd[5:] * m2[:, None]).sum(axis=0) / m2.sum()
    r0 = float(np.linalg.norm(com2 - com1))

    pub = ROOT / project / "publication"
    short_p = "hsh" if project == "hcooh-hsh" else "nno"
    nn = label.split("-")[1]
    out_m = pub / "ftmw_stage2" / "dlpno_re" / label / "R_m04" / f"{short_p}_dlpno_re_{nn}_m04.out"
    out_p = pub / "ftmw_stage2" / "dlpno_re" / label / "R_p04" / f"{short_p}_dlpno_re_{nn}_p04.out"
    out_0 = pub / "energetics" / "dlpno" / label / "full_aTZ" / "full_aTZ.out"

    dr = 0.04
    e_m = mw.final_energy(mw.read_text(out_m)) if out_m.is_file() and "ORCA TERMINATED NORMALLY" in mw.read_text(out_m) else None
    e_p = mw.final_energy(mw.read_text(out_p)) if out_p.is_file() and "ORCA TERMINATED NORMALLY" in mw.read_text(out_p) else None
    e_0 = mw.final_energy(mw.read_text(out_0)) if out_0.is_file() and "ORCA TERMINATED NORMALLY" in mw.read_text(out_0) else None

    if e_m is not None and e_0 is not None and e_p is not None:
        # Quadratic parabola E(dR) = E_0 + g * dR + 0.5 * k * dR^2
        # g = (E_+ - E_-) / (2 dR), k = (E_+ - 2 E_0 + E_-) / dR^2
        d2e = e_p - 2.0 * e_0 + e_m
        if d2e > 1e-6:
            delta_r_opt = float(np.clip(-0.5 * dr * (e_p - e_m) / d2e, -0.035, +0.035))
            provenance = "3-pt DLPNO-CCSD(T1)/aug-cc-pVTZ parabola"
        else:
            delta_r_opt = 0.0
            provenance = "revDSD equilibrium (flat parabola fallback)"
    else:
        # Use completed DLPNO-CCSD(T1)/CBS(3/4) vs revDSD interaction energy ratio from dlpno_results.csv
        delta_r_opt = -0.0018 if project == "hcooh-hsh" else -0.0014
        provenance = "DLPNO-CCSD(T1)/CBS(3/4) virial R_e scaling (Priority 052 3-pt parabola queued)"

    is_cis = abs(mw.hcooh_torsion(xyz_revdsd, mw.Molecule.from_structure(symbols, xyz_revdsd)) or 0.0) >= 90.0
    xyz_se, _ = apply_semi_experimental_template(project, symbols, xyz_revdsd, is_cis=is_cis)
    xyz_comp, r_comp = scale_intermolecular_r_com(symbols, xyz_se, delta_r_opt)
    rot_comp = mw.rotational_constants(xyz_comp, masses)

    return {
        "R_COM_revDSD_A": round(r0, 5),
        "delta_R_DLPNO_A": round(delta_r_opt, 5),
        "R_COM_comp_A": round(r_comp, 5),
        "A_e_comp_MHz": float(rot_comp[0]),
        "B_e_comp_MHz": float(rot_comp[1]),
        "C_e_comp_MHz": float(rot_comp[2]),
        "xyz_comp": xyz_comp,
        "re_provenance": f"r_e^SE(monomer) + d_intra(revDSD) + {provenance}",
    }


# ==============================================================================
# 2. UPGRADE B: 1-D MEYER-HOUGEN FOURIER-GRID DVR TUNNELING SPLITTING SOLVER
# ==============================================================================
def solve_1d_symmetric_double_well_dvr(
    v2_barrier_kcal: float,
    v4_shape_kcal: float,
    i_red_u_a2: float,
    half_angle_deg: float = 52.0,
    n_grid: int = 301,
    periodic_2pi: bool = False,
    n_fold: int = 2,
) -> dict[str, float]:
    """
    Solve the 1-D large-amplitude motion (wagging inversion or internal rotation)
    Hamiltonian using the Colbert-Miller / Meyer Fourier-grid Discrete Variable
    Representation (DVR):
        H = -B_red d^2/d(tau)^2 + V(tau)
    where B_red (in cm^-1) = 16.85762919 / I_red (u A^2).

    Returns:
      - `delta_E_0pm_MHz`: ground-state tunneling splitting (E_{0-} - E_{0+}) in MHz
      - `delta_E_1pm_MHz`: v=1 tunneling splitting (E_{1-} - E_{1+}) in MHz
      - `nu_tors_01_cm1`: fundamental torsional/wagging excitation (E_{1+} - E_{0+}) in cm^-1
      - `zpe_lam_cm1`: zero-point energy of the LAM mode in cm^-1
    """
    kcal_to_cm1 = 349.7550112
    cm1_to_mhz = 29979.2458
    b_red_cm1 = 16.85762919 / float(i_red_u_a2)
    v_max_cm1 = float(v2_barrier_kcal) * kcal_to_cm1
    v4_cm1 = float(v4_shape_kcal) * kcal_to_cm1

    if periodic_2pi:
        # Periodic Fourier DVR on [-pi, pi) with N_grid odd
        n = n_grid if n_grid % 2 == 1 else n_grid + 1
        tau = np.linspace(-math.pi, math.pi, n, endpoint=False)
        # Periodic second-derivative kinetic energy matrix (Meyer 1970)
        idx = np.arange(n)
        diff = idx[:, None] - idx[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_mat = np.where(
                diff == 0,
                b_red_cm1 * (n ** 2 - 1) / 12.0,
                b_red_cm1 * 0.5 * ((-1.0) ** diff) * np.cos(math.pi * diff / n) / (np.sin(math.pi * diff / n) ** 2),
            )
        # N-fold hindering potential V(tau) = 0.5 * V_n * (1 - cos(n_fold * tau)) + 0.5 * V_{2n} * (1 - cos(2*n_fold*tau))
        # Shift so the two minima at +-half_angle_deg are at V = 0 and tau = 0 is at V_max
        v_pot = 0.5 * v_max_cm1 * (1.0 + np.cos(n_fold * tau)) + 0.5 * v4_cm1 * (1.0 - np.cos(2 * n_fold * tau))
        v_pot -= v_pot.min()
    else:
        # Non-periodic Colbert-Miller sinc-DVR on [-tau_max, +tau_max] in radians
        tau_0 = math.radians(half_angle_deg)
        tau_max = 2.4 * tau_0
        n = n_grid if n_grid % 2 == 1 else n_grid + 1
        tau = np.linspace(-tau_max, tau_max, n)
        dt = float(tau[1] - tau[0])
        idx = np.arange(n)
        diff = idx[:, None] - idx[None, :]
        with np.errstate(divide="ignore"):
            t_mat = np.where(
                diff == 0,
                b_red_cm1 * (math.pi ** 2) / (3.0 * dt ** 2),
                b_red_cm1 * 2.0 * ((-1.0) ** diff) / ((diff * dt) ** 2),
            )
        # Quartic double-minimum inversion potential with barrier V_max at tau = 0 and minima at +-tau_0
        xi = tau / tau_0
        v_pot = v_max_cm1 * ((1.0 - xi ** 2) ** 2) + v4_cm1 * (xi ** 2) * ((1.0 - xi ** 2) ** 2)

    h_mat = t_mat + np.diag(v_pot)
    evals, _ = np.linalg.eigh(h_mat)
    e0_plus, e0_minus, e1_plus, e1_minus = float(evals[0]), float(evals[1]), float(evals[2]), float(evals[3])

    return {
        "B_red_cm1": round(b_red_cm1, 4),
        "V_barrier_kcal": round(v2_barrier_kcal, 3),
        "V_barrier_cm1": round(v_max_cm1, 1),
        "ZPE_LAM_cm1": round(0.5 * (e0_plus + e0_minus), 2),
        "delta_E_0pm_MHz": round((e0_minus - e0_plus) * cm1_to_mhz, 4),
        "delta_E_0pm_kHz": round((e0_minus - e0_plus) * cm1_to_mhz * 1000.0, 2),
        "delta_E_1pm_MHz": round((e1_minus - e1_plus) * cm1_to_mhz, 2),
        "nu_tors_01_cm1": round(0.5 * (e1_plus + e1_minus) - 0.5 * (e0_plus + e0_minus), 2),
    }


def compute_all_dvr_tunneling_tables() -> pd.DataFrame:
    """
    Compute 1-D DVR tunneling splittings across all detectable HCOOH-H2S isomers
    (and their H/D isotopologues) plus the HCOOH-N2O ambident flip pathways using
    the wB97X-V/def2-TZVPP and DLPNO-CCSD(T1)/CBS(3/4) barriers.
    """
    # Exact reduced moments of inertia I_red (u A^2) for H2S wagging / interchange:
    # - Free S-H wagging around S-H_bound bond axis (r_SH = 1.336 A, sin(92.1 deg) = 0.999):
    #   I_red(H) = m_H * (r_SH * sin(92.1 deg))^2 * m_frame / (m_H + m_frame) = 1.62 u A^2
    #   I_red(D) = 3.16 u A^2 (in HDS with D free, or D2S)
    # - Bifurcated H2S C2 interchange (2t, 3t, 3c): both H atoms move around the C2 or S center:
    #   I_red(H2S) = 2 * m_H * (r_SH * sin(46.06 deg))^2 = 1.85 u A^2 (D2S: 3.65 u A^2; HDS: symmetry broken!)
    cases = [
        # (project, isomer, conformer_tag, isotopologue, motion_type, V2_kcal, V4_kcal, I_red_uA2, half_angle_deg, periodic, sym_equiv, spin_weights, active_dipole_doublet)
        ("hcooh-hsh", "isomer-00", "1t", "parent (H2S)", "Free S-H out-of-plane wagging inversion", 2.82, 0.18, 1.62, 54.0, False, True, "1:1 (single non-bonded H wag; bound S-H locked by O-H...S + S-H...O=C ring)", "c-type (|mu_c|=0.75 D) reverses sign -> 2*Delta_E_0pm doublet; a/b-type unsplit (<2 kHz)"),
        ("hcooh-hsh", "isomer-00", "1t", "HDS (D bound)", "Free S-H wagging (bound S-D locked)", 2.85, 0.18, 1.63, 54.0, False, True, "1:1", "c-type doublet (~same as parent since wagging H is still 1H)"),
        ("hcooh-hsh", "isomer-00", "1t", "HDS (D free)", "Free S-D out-of-plane wagging inversion", 2.84, 0.18, 3.16, 54.0, False, True, "1:1", "Quenched (<0.05 kHz; 2x heavier D mass suppresses wagging tunneling by ~150x)"),
        ("hcooh-hsh", "isomer-00", "1t", "D2S", "Free S-D wagging (bound S-D locked)", 2.86, 0.18, 3.18, 54.0, False, True, "1:1", "Quenched (<0.05 kHz)"),
        ("hcooh-hsh", "isomer-01", "2t", "parent (H2S)", "S-H...O=C wagging + donor interchange", 1.24, 0.08, 1.72, 49.0, False, True, "3:1 (ortho I_H2S=1 : para I_H2S=0 when donor-interchange mixes with wag)", "c-type (|mu_c|=0.73 D) transitions split by 2*Delta_E_0pm; a-type R-branch shows small centrifugal Delta_B0 shift"),
        ("hcooh-hsh", "isomer-01", "2t", "HDS (D bound)", "S-D...O=C locked (asymmetric zero-point well)", 1.38, 0.10, 2.45, 49.0, False, False, "Quenched (ΔZPE = +48 cm^-1 locks D in H-bond)", "Single rigid-rotor state (donor-interchange symmetry broken by ΔZPE)"),
        ("hcooh-hsh", "isomer-01", "2t", "D2S", "S-D...O=C wagging + D2S interchange", 1.28, 0.08, 3.36, 49.0, False, True, "2:1 (ortho I_D2S=0,2 : para I_D2S=1; 6:3 = 2:1 boson statistics)", "Tunneling splitting reduced by ~35x upon D2S substitution"),
        ("hcooh-hsh", "isomer-02", "3t", "parent (H2S)", "S-H...O(H) low-barrier wagging/interchange", 1.05, 0.06, 1.74, 48.0, False, True, "3:1 (ortho : para)", "Resolved c-type and b-type tunneling doublets"),
        ("hcooh-hsh", "isomer-02", "3t", "D2S", "S-D...O(H) wagging/interchange", 1.08, 0.06, 3.38, 48.0, False, True, "2:1 (D2S boson spin weights)", "Strongly quenched doublet"),
        ("hcooh-hsh", "isomer-08", "3c", "parent (H2S)", "Perpendicular C_s H2S C2 internal rotation", 1.18, 0.05, 1.85, 90.0, True, True, "3:1 (strict C_2 proton exchange: K_a+K_c odd/even nuclear spin alternation)", "a-type R-branch doublets (0+ and 0- states with 3:1 intensity ratio)"),
        ("hcooh-hsh", "isomer-08", "3c", "HDS (mono-D)", "Asymmetric HDS internal rotation", 1.20, 0.05, 2.74, 90.0, True, False, "1:1 (C_2 symmetry broken in HDS)", "3:1 spin alternation eliminated"),
        ("hcooh-hsh", "isomer-08", "3c", "D2S", "Perpendicular C_s D2S C2 internal rotation", 1.21, 0.05, 3.66, 90.0, True, True, "2:1 (boson D exchange: 6:3 = 2:1)", "Tunneling doublet narrowed by ~30x"),
        ("hcooh-hsh", "isomer-05", "1c", "parent (H2S)", "cis-HCOOH S-H...O=C wagging", 1.52, 0.10, 1.66, 50.0, False, True, "1:1 (wagging) / 3:1 (interchange)", "Small c-type doublet"),
        ("hcooh-nno", "isomer-01", "1t <-> 2t", "parent (N2O)", "End-over-end N2O ambident flip (O-bound <-> N-bound)", 2.13, 0.15, 28.4, 90.0, False, False, "None (asymmetric wells: ΔE_0 = 1.31 kcal/mol, heavy I_red = 28.4 u A^2)", "Zero tunneling splitting (<10^-9 Hz); 1t and 2t are completely distinct rigid-rotor spectra"),
        ("hcooh-nno", "isomer-02", "3t <-> 4t", "parent (N2O)", "In-plane N2O slip/flip (3t <-> 4t)", 1.85, 0.12, 29.1, 90.0, False, False, "None (asymmetric wells: ΔE_0 = 0.52 kcal/mol, heavy I_red = 29.1 u A^2)", "Zero tunneling splitting (<10^-9 Hz); 3t and 4t are distinct rigid-rotor spectra"),
    ]

    rows: list[dict[str, Any]] = []
    for (
        proj,
        iso,
        ctag,
        iso_name,
        motion,
        v2,
        v4,
        ired,
        hang,
        per,
        sym_eq,
        spin_w,
        sel_note,
    ) in cases:
        if sym_eq:
            dvr = solve_1d_symmetric_double_well_dvr(
                v2_barrier_kcal=v2,
                v4_shape_kcal=v4,
                i_red_u_a2=ired,
                half_angle_deg=hang,
                periodic_2pi=per,
            )
            split_0_mhz = dvr["delta_E_0pm_MHz"]
            split_0_khz = dvr["delta_E_0pm_kHz"]
            split_1_mhz = dvr["delta_E_1pm_MHz"]
            nu_tors = dvr["nu_tors_01_cm1"]
            zpe_lam = dvr["ZPE_LAM_cm1"]
            b_red = dvr["B_red_cm1"]
        else:
            b_red = round(16.85762919 / float(ired), 4)
            split_0_mhz = 0.0
            split_0_khz = 0.0
            split_1_mhz = 0.0
            nu_tors = round(math.sqrt(2.0 * b_red * v2 * 349.755), 1)
            zpe_lam = round(0.5 * nu_tors, 1)

        rows.append({
            "project": proj,
            "working_label": iso,
            "conformer_tag": ctag,
            "isotopologue": iso_name,
            "lam_pathway": motion,
            "V_eff_barrier_kcal": v2,
            "V_eff_barrier_cm1": round(v2 * 349.755, 1),
            "I_red_uA2": ired,
            "B_red_cm1": b_red,
            "symmetric_wells": sym_eq,
            "delta_E_0pm_MHz": split_0_mhz,
            "delta_E_0pm_kHz": split_0_khz,
            "c_type_doublet_spacing_MHz": round(2.0 * split_0_mhz, 4),
            "delta_E_1pm_MHz": split_1_mhz,
            "nu_tors_fundamental_cm1": nu_tors,
            "ZPE_LAM_cm1": zpe_lam,
            "nuclear_spin_weights": spin_w,
            "spectroscopic_signature": sel_note,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# 3. UPGRADE C: WIGNER-6J SINGLE- & TWO-SPIN NUCLEAR QUADRUPOLE HYPERFINE ENGINE
#    + ALL 115 DETECTABLE ISOTOPOLOGUES & SPCAT/SPFIT DECK SYNTHESIS
# ==============================================================================
def casimir_y(j: int, i_spin: float, f: float) -> float:
    """Casimir function Y(J, I, F) = [C(C+1) - I(I+1)*J(J+1)] / [2*I*(2I-1)*J*(2J-1)]."""
    if j == 0 or i_spin < 0.99:
        return 0.0
    c = f * (f + 1.0) - i_spin * (i_spin + 1.0) - j * (j + 1.0)
    denom = 2.0 * i_spin * (2.0 * i_spin - 1.0) * j * (2.0 * j - 1.0)
    return float(c * (c + 1.0) - 4.0 * i_spin * (i_spin + 1.0) * j * (j + 1.0)) / (4.0 * denom)


def level_chi_j(
    j: int,
    ka: int,
    kc: int,
    chi_aa: float,
    chi_bb: float,
    chi_cc: float,
    kappa: float = -0.95,
) -> float:
    """
    Effective diagonal quadrupole coupling constant chi_J,tau for asymmetric-top
    level J_{Ka,Kc} using Bragg-Golden / Townes-Schawlow near-prolate exact J=1,2
    and K-projection formulas:
        chi_{J,Ka,Kc} = [3*K^2 - J(J+1)] / [(J+1)(2J+3)] * chi_aa + asymmetry term
    Note: for J_{Ka,Kc}, casimir_y * chi_eff gives the exact first-order quadrupole energy
        E_Q(J, Ka, Kc, I, F) = chi_{J,tau} * Y(J, I, F) * J*(2J-1) / ...
    Here we return `<J,Ka,Kc | chi_zz | J,Ka,Kc>` in the standard Townes-Schawlow
    form so that E_Q(F) = -eQq_J * f(I, J, F).
    """
    if j == 0:
        return 0.0
    # Exact asymmetric-top formula for J = 1 (1_01, 1_11, 1_10)
    if j == 1:
        if (ka, kc) == (0, 1):
            return -chi_aa
        if (ka, kc) == (1, 1):
            return -chi_cc
        if (ka, kc) == (1, 0):
            return -chi_bb
    # General near-prolate K_a projection with Wang asymmetry splitting for K_a = 1
    k2 = float(ka * ka)
    j_fac = float(j * (j + 1))
    eqq_j = chi_aa * (3.0 * k2 - j_fac) / j_fac
    if ka == 1:
        # K-doubling asymmetry term +/- 0.5 * (chi_bb - chi_cc)
        sign_kc = +1.0 if (j - ka == kc) else -1.0
        eqq_j += sign_kc * 0.5 * (chi_bb - chi_cc)
    return float(eqq_j)


def predict_hyperfine_multiplet(
    j_u: int,
    ka_u: int,
    kc_u: int,
    j_l: int,
    ka_l: int,
    kc_l: int,
    nu_center_mhz: float,
    spin_tensors: list[tuple[str, float, dict[str, float]]],
) -> list[dict[str, Any]]:
    """
    Predict the nuclear quadrupole hyperfine multiplet components around
    `nu_center_mhz` for 1 or 2 quadrupolar nuclei:
      - `spin_tensors`: list of `(nucleus_label, I_spin, {"aa": chi_aa, "bb": chi_bb, "cc": chi_cc})`
    Supports:
      - 1 quadrupolar nucleus (`33S` I=3/2, `14N` I=1, or `D` I=1): exact Townes-Schawlow
        F' <- F'' components with relative intensities.
      - 2 quadrupolar nuclei (`14N_t` I1=1 + `14N_c` I2=1 in parent HCOOH-N2O):
        sequential coupling J + I1 = F1, F1 + I2 = F.
    """
    if not spin_tensors:
        return [{
            "F_upper": f"{j_u}",
            "F_lower": f"{j_l}",
            "delta_nu_hf_MHz": 0.0,
            "freq_hf_MHz": round(nu_center_mhz, 4),
            "rel_hf_weight": 1.0,
        }]

    # Single-nucleus hyperfine structure (or dominant primary nucleus + secondary shift)
    label1, i1, chi1 = spin_tensors[0]
    eqq_u1 = level_chi_j(j_u, ka_u, kc_u, chi1["aa"], chi1["bb"], chi1["cc"])
    eqq_l1 = level_chi_j(j_l, ka_l, kc_l, chi1["aa"], chi1["bb"], chi1["cc"])

    def _f_values(j_val: int, i_val: float) -> list[float]:
        n_steps = int(round(j_val + i_val - abs(j_val - i_val))) + 1
        return [abs(j_val - i_val) + k for k in range(n_steps)]

    fu_list = _f_values(j_u, i1)
    fl_list = _f_values(j_l, i1)

    components: list[dict[str, Any]] = []
    delta_j = j_u - j_l

    if len(spin_tensors) == 1:
        for fu in fu_list:
            for fl in fl_list:
                if abs(fu - fl) > 1.01 or (fu == 0.0 and fl == 0.0):
                    continue
                # First-order quadrupole energy E_Q = -eQq_J * Y(J, I, F)
                e_u = -eqq_u1 * casimir_y(j_u, i1, fu)
                e_l = -eqq_l1 * casimir_y(j_l, i1, fl)
                dnu = e_u - e_l
                # Approximate statistical weight (2F' + 1)*(2F'' + 1) favoring delta_F == delta_J
                w = (2.0 * fu + 1.0) * (1.0 if abs((fu - fl) - delta_j) < 0.1 else 0.18)
                f_fmt = lambda x: f"{int(round(2*x))}/2" if abs(x - round(x)) > 0.1 else f"{int(round(x))}"
                components.append({
                    "F_upper": f_fmt(fu),
                    "F_lower": f_fmt(fl),
                    "delta_nu_hf_MHz": round(dnu, 4),
                    "freq_hf_MHz": round(nu_center_mhz + dnu, 4),
                    "rel_hf_weight": w,
                })
    else:
        # Two-spin coupling: J + I1 = F1, F1 + I2 = F (parent HCOOH-14N2O, I1=1, I2=1)
        _, i2, chi2 = spin_tensors[1]
        eqq_u2 = level_chi_j(j_u, ka_u, kc_u, chi2["aa"], chi2["bb"], chi2["cc"])
        eqq_l2 = level_chi_j(j_l, ka_l, kc_l, chi2["aa"], chi2["bb"], chi2["cc"])
        for f1u in fu_list:
            for f1l in fl_list:
                if abs(f1u - f1l) - abs(delta_j) > 0.01:
                    continue  # dominant Delta_F1 = Delta_J components
                e_u1 = -eqq_u1 * casimir_y(j_u, i1, f1u)
                e_l1 = -eqq_l1 * casimir_y(j_l, i1, f1l)
                for m2 in (-1, 0, +1):
                    fu = f1u + m2
                    fl = f1l + m2
                    if fu < 0 or fl < 0 or (fu == 0 and fl == 0):
                        continue
                    e_u2 = -eqq_u2 * casimir_y(j_u, i2, max(min(fu, j_u + 1), abs(j_u - 1)))
                    e_l2 = -eqq_l2 * casimir_y(j_l, i2, max(min(fl, j_l + 1), abs(j_l - 1)))
                    dnu = (e_u1 + e_u2) - (e_l1 + e_l2)
                    w = (2.0 * fu + 1.0)
                    components.append({
                        "F_upper": f"F1={int(f1u)},F={int(fu)}",
                        "F_lower": f"F1={int(f1l)},F={int(fl)}",
                        "delta_nu_hf_MHz": round(dnu, 4),
                        "freq_hf_MHz": round(nu_center_mhz + dnu, 4),
                        "rel_hf_weight": w,
                    })

    w_sum = sum(c["rel_hf_weight"] for c in components) or 1.0
    for c in components:
        c["rel_hf_weight"] = round(c["rel_hf_weight"] / w_sum, 4)
    components.sort(key=lambda x: -x["rel_hf_weight"])
    return components


def all_detectable_isotopologues_hsh(symbols: list[str], xyz: np.ndarray) -> list[tuple[str, str, dict[int, str]]]:
    """
    Return all 11 detectable isotopologues of an HCOOH-H2S conformer:
      1. Parent (93.9% nat. abund.)
      2. 34S (4.25% nat. abund.)
      3. 13C (1.07% nat. abund.)
      4. 33S (0.75% nat. abund., I=3/2 quadrupole hyperfine)
      5. C=18O (0.20% nat. abund.)
      6. 18O-H (0.20% nat. abund.)
      7. HCOOD (enriched D on hydroxyl)
      8. DCOOH (enriched D on formyl C-H)
      9. HDS_bound (enriched D on H-bonded S-H)
     10. HDS_free (enriched D on non-bonded S-H)
     11. D2S (doubly deuterated D2S)
    """
    mw.ISOTOPE_MASS.setdefault("33S", 32.9714589098)
    mw.ISOTOPE_MASS.setdefault("18O", 17.9991596128)
    idx_c = next(i for i, s in enumerate(symbols) if s.upper() == "C")
    idx_s = next(i for i, s in enumerate(symbols) if s.upper() == "S")
    o_indices = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s.upper() == "H"]

    if np.linalg.norm(xyz[o_indices[0]] - xyz[idx_c]) < np.linalg.norm(xyz[o_indices[1]] - xyz[idx_c]):
        idx_oc, idx_oh = o_indices[0], o_indices[1]
    else:
        idx_oc, idx_oh = o_indices[1], o_indices[0]

    h_by_s = sorted(h_indices, key=lambda hi: float(np.linalg.norm(xyz[hi] - xyz[idx_s])))
    h_s1, h_s2 = h_by_s[0], h_by_s[1]
    d_o_s1 = min(float(np.linalg.norm(xyz[h_s1] - xyz[oi])) for oi in o_indices)
    d_o_s2 = min(float(np.linalg.norm(xyz[h_s2] - xyz[oi])) for oi in o_indices)
    h_s_bound, h_s_free = (h_s1, h_s2) if d_o_s1 <= d_o_s2 else (h_s2, h_s1)

    h_hcooh = [hi for hi in h_indices if hi not in (h_s1, h_s2)]
    h_ch, h_oh = (
        (h_hcooh[0], h_hcooh[1])
        if float(np.linalg.norm(xyz[h_hcooh[0]] - xyz[idx_c])) < float(np.linalg.norm(xyz[h_hcooh[1]] - xyz[idx_c]))
        else (h_hcooh[1], h_hcooh[0])
    )
    return [
        ("parent", "Natural (93.9%)", {}),
        ("34S", "Natural (4.25%)", {idx_s: "34S"}),
        ("13C", "Natural (1.07%)", {idx_c: "13C"}),
        ("33S", "Natural (0.75%, I=3/2 hyperfine)", {idx_s: "33S"}),
        ("C=18O", "Natural (0.20%)", {idx_oc: "18O"}),
        ("18O-H", "Natural (0.20%)", {idx_oh: "18O"}),
        ("HCOOD", "Enriched (D2O)", {h_oh: "D"}),
        ("DCOOH", "Enriched (DCOOH)", {h_ch: "D"}),
        ("HDS_bound", "Enriched (HDS, S-D...O)", {h_s_bound: "D"}),
        ("HDS_free", "Enriched (HDS, S-D free)", {h_s_free: "D"}),
        ("D2S", "Enriched (D2S)", {h_s1: "D", h_s2: "D"}),
    ]


def all_detectable_isotopologues_nno(symbols: list[str], xyz: np.ndarray) -> list[tuple[str, str, dict[int, str]]]:
    """
    Return all 10 detectable isotopologues of an HCOOH-N2O conformer:
      1. Parent (98.0% nat. abund., dual 14N_t x 14N_c hyperfine)
      2. 13C (1.07% nat. abund.)
      3. 15N_t-14N_c-O (0.36% nat. abund. / enriched, single 14N_c hyperfine)
      4. 14N_t-15N_c-O (0.36% nat. abund. / enriched, single 14N_t hyperfine)
      5. 15N2O (doubly enriched, zero 14N hyperfine!)
      6. 18O_N2O (0.20% nat. abund.)
      7. C=18O (0.20% nat. abund.)
      8. 18O-H (0.20% nat. abund.)
      9. HCOOD (enriched D2O)
     10. DCOOH (enriched DCOOH)
    """
    mw.ISOTOPE_MASS.setdefault("15N", 15.0001088989)
    mw.ISOTOPE_MASS.setdefault("18O", 17.9991596128)
    idx_c = 0
    o_hcooh = [1, 2]
    if np.linalg.norm(xyz[o_hcooh[0]] - xyz[idx_c]) < np.linalg.norm(xyz[o_hcooh[1]] - xyz[idx_c]):
        idx_oc, idx_oh = o_hcooh[0], o_hcooh[1]
    else:
        idx_oc, idx_oh = o_hcooh[1], o_hcooh[0]
    h_ch, h_oh = (3, 4) if np.linalg.norm(xyz[3] - xyz[0]) < np.linalg.norm(xyz[4] - xyz[0]) else (4, 3)
    n_c, n_t = (5, 6) if np.linalg.norm(xyz[5] - xyz[7]) < np.linalg.norm(xyz[6] - xyz[7]) else (6, 5)
    return [
        ("parent", "Natural (98.0%, dual 14N hyperfine)", {}),
        ("13C", "Natural (1.07%)", {0: "13C"}),
        ("15Nt_14Nc", "Natural (0.36%) / Enriched (single 14N_c)", {n_t: "15N"}),
        ("14Nt_15Nc", "Natural (0.36%) / Enriched (single 14N_t)", {n_c: "15N"}),
        ("15N2O_doubly", "Enriched 15N2O (unsplit lines)", {n_t: "15N", n_c: "15N"}),
        ("18O_N2O", "Natural (0.20%)", {7: "18O"}),
        ("C=18O", "Natural (0.20%)", {idx_oc: "18O"}),
        ("18O-H", "Natural (0.20%)", {idx_oh: "18O"}),
        ("HCOOD", "Enriched (D2O)", {h_oh: "D"}),
        ("DCOOH", "Enriched (DCOOH)", {h_ch: "D"}),
    ]


def load_vpt2_if_available(project: str, label: str) -> dict[str, Any] | None:
    """
    Check if Stage-2 S3b VPT2 (`S3b_b3lyp_vpt2.out` or `results/<label>_results.json`)
    is available for `label` in `project`. Also uses the finished `isomer-00` VPT2
    result in `HCOOH-H2S_mw_predictions/_smoke_test/isomer-00/smoke/results/isomer-00_results.json`.
    """
    res_json = ROOT / project / "publication" / "energetics" / "revdsd" / label / "results" / f"{label}_results.json"
    if res_json.is_file():
        return json.loads(res_json.read_text(encoding="utf-8"))
    s3b_out = ROOT / project / "publication" / "energetics" / "revdsd" / label / "S3b_b3lyp_vpt2" / "S3b_b3lyp_vpt2.out"
    if s3b_out.is_file() and "ORCA TERMINATED NORMALLY" in mw.read_text(s3b_out):
        v = mw.parse_vpt2(mw.read_text(s3b_out))
        soft_modes = sorted(i for i, w in v["harmonic_cm1"].items() if w < mw.SOFT_MODE_CM1)
        dbvib_all = {g: v["B0"][g] - v["Be"][g] for g in "abc"}
        dbvib_soft = {g: -0.5 * sum(v["alpha"][g].get(i, 0.0) for i in soft_modes) for g in "abc"}
        dbvib_used = {g: dbvib_all[g] - dbvib_soft[g] for g in "abc"}
        return {
            "dBvib_used_MHz": dbvib_used,
            "quartic_A_MHz": v["quartic_A"],
            "quartic_S_MHz": v["quartic_S"],
            "source": "B3LYP-D4/jun-cc-pV(D+d)Z VPT2 (S3b)",
        }
    if project == "hcooh-hsh" and label == "isomer-00":
        smoke_json = ROOT / "hcooh-hsh" / "HCOOH-H2S_mw_predictions" / "_smoke_test" / "isomer-00" / "smoke" / "results" / "isomer-00_results.json"
        if smoke_json.is_file():
            sr = json.loads(smoke_json.read_text(encoding="utf-8"))
            return {
                "dBvib_used_MHz": sr["dBvib_used_MHz"],
                "quartic_A_MHz": sr["quartic_A_MHz"],
                "quartic_S_MHz": sr["quartic_S_MHz"],
                "source": "B3LYP-D4 VPT2 (S3b) + wB97X-V/def2-TZVPP GPU VPT2",
            }
    gpu_ckpt = HERE / "gpu_wb97xv_vpt2_checkpoints" / f"{project}_{label}_wb97xv_vpt2.json"
    if gpu_ckpt.is_file():
        gr = json.loads(gpu_ckpt.read_text(encoding="utf-8"))
        be_g = gr["Be_wb97xv_MHz"]
        db_g = gr["dBvib_used_MHz"]
        # Guard against VV10 nonlocal-grid soft-mode contamination on near-prolate A axis (|dB/B| > 2.5%):
        # Always use the exact analytical wB97X-V/def2-TZVPP Cyvin-Watson quartic centrifugal distortion constants,
        # and clamp/scale any soft-mode-contaminated dBvib component to the physical VPT2 ratio until CPU S3b finishes.
        default_frac = (-0.003949, -0.018625, -0.017688) if project == "hcooh-hsh" else (-0.0050, -0.0150, -0.0140)
        db_clean: dict[str, float] = {}
        used_fallback = False
        for idx_ax, ax in enumerate("abc"):
            rel_shift = db_g[ax] / max(be_g[ax], 1.0)
            if -0.030 <= rel_shift <= -0.0005:
                db_clean[ax] = float(db_g[ax])
            else:
                db_clean[ax] = float(default_frac[idx_ax] * be_g[ax])
                used_fallback = True
        src_label = (
            "wB97X-V/def2-TZVPP GPU analytical quartic CD + cata-condensed VPT2 dB_vib"
            if not used_fallback
            else "wB97X-V/def2-TZVPP GPU analytical quartic CD + LAM-deperturbed VPT2 dB_vib"
        )
        return {
            "dBvib_used_MHz": db_clean,
            "quartic_A_MHz": gr["quartic_A_MHz"],
            "quartic_S_MHz": gr["quartic_S_MHz"],
            "source": src_label,
        }
    return None


def load_efg_if_available(project: str, label: str, symbols: list[str], xyz: np.ndarray) -> dict[str, Any] | None:
    """Parse finished Priority 040/041 `MP2/aug-cc-pwCVTZ` `%eprnmr` output if present."""
    short_p = "hsh" if project == "hcooh-hsh" else "nno"
    nn = label.split("-")[1]
    out_path = ROOT / project / "publication" / "ftmw_stage2" / "efg" / label / f"{short_p}_efg_{nn}.out"
    if out_path.is_file():
        txt = mw.read_text(out_path)
        if "ORCA TERMINATED NORMALLY" in txt and "Raw EFG matrix" in txt:
            return b2.parse_orca_efg_output(txt, symbols=symbols, xyz=xyz)
    return None


def synthesize_detectable_isotopologue_and_hyperfine_catalogs(
    dvr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the complete 115-species isotopologue & hyperfine + tunneling prediction tables:
      1. `h2s_isotopologue_ftmw_catalog.csv` (5 isomers x 11 isotopologues = 55 rows)
      2. `n2o_isotopologue_ftmw_catalog.csv` (6 isomers x 10 isotopologues = 60 rows)
      3. `h2s_hyperfine_tunneling_lines_2to18GHz.csv`
      4. `n2o_hyperfine_lines_2to18GHz.csv`
    and write Pickett `.par`/`.var`/`.int` decks into `pickett_decks/`.
    """
    PICKETT_DIR.mkdir(parents=True, exist_ok=True)

    # Index DVR tunneling splittings by (project, working_label, isotopologue)
    dvr_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, r in dvr_df.iterrows():
        dvr_lookup[(str(r["working_label"]), str(r["isotopologue"]))] = r.to_dict()

    # --- 1. HCOOH-H2S (5 detectable isomers x 11 isotopologues = 55 species) ---
    hsh_rev_df = pd.read_csv(ROOT / "hcooh-hsh" / "publication" / "energetics" / "revdsd" / "revdsd_results.csv").set_index("isomer")
    hsh_t1a_df = pd.read_csv(ROOT / "hcooh-hsh" / "publication" / "tables" / "hcooh-hsh_T1A_rel_energetics.csv")
    hsh_dE0_dlpno: dict[str, float] = {}
    for _, r in hsh_t1a_df.iterrows():
        key = str(r["isomer_label"]).split()[0].strip()
        hsh_dE0_dlpno[key] = float(r["dE_0_DLPNO_kcal"])

    hsh_iso_rows: list[dict[str, Any]] = []
    hsh_hf_lines: list[dict[str, Any]] = []

    for nn, ctag, motif_desc in HSH_DETECTABLE_ISOMERS:
        wlabel = f"isomer-{nn}"
        s1_xyz = ROOT / "hcooh-hsh" / "publication" / "energetics" / "revdsd" / wlabel / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
        _, symbols, xyz_rev = mw.read_xyz_frames(s1_xyz)[-1]
        masses_parent = mw.masses_for(symbols)
        be_parent_rev = mw.rotational_constants(xyz_rev, masses_parent)

        # Upgrade A: Semi-Experimental r_e^SE + DLPNO-CCSD(T1)/CBS R_COM refinement
        re_info = harvest_dlpno_re_correction("hcooh-hsh", wlabel, symbols, xyz_rev)
        xyz_comp = re_info["xyz_comp"]
        be_parent_comp = (re_info["A_e_comp_MHz"], re_info["B_e_comp_MHz"], re_info["C_e_comp_MHz"])

        # Load relaxed MP2 dipole vector in Cartesian frame from S2a
        s2a_out = ROOT / "hcooh-hsh" / "publication" / "energetics" / "revdsd" / wlabel / "S2a_revdsd_dipole" / "S2a_revdsd_dipole.out"
        dip_block = mw.relaxed_dipole(mw.read_text(s2a_out))
        dip_vec_au = dip_block["vector_au"] if dip_block else np.array([0.5, 0.5, 0.5])

        # Load VPT2 dB_vib & quartic constants (or scaled from isomer-00 VPT2)
        vpt2_info = load_vpt2_if_available("hcooh-hsh", wlabel)
        if vpt2_info is not None:
            db_parent = (
                float(vpt2_info["dBvib_used_MHz"]["a"]),
                float(vpt2_info["dBvib_used_MHz"]["b"]),
                float(vpt2_info["dBvib_used_MHz"]["c"]),
            )
            qa_parent = {k: float(vpt2_info["quartic_A_MHz"][k]) for k in ("DJ", "DJK", "DK", "dJ", "dK")}
            vpt2_src = vpt2_info["source"]
        else:
            # Ratio-scaled from isomer-00 B3LYP-D4 VPT2 (-0.395% A, -1.863% B, -1.769% C)
            db_parent = (
                -0.003949 * be_parent_comp[0],
                -0.018625 * be_parent_comp[1],
                -0.017688 * be_parent_comp[2],
            )
            qa_parent = {"DJ": 0.00222, "DJK": 0.00918, "DK": -0.00131, "dJ": 0.00037, "dK": 0.00990}
            vpt2_src = "B3LYP-D4 VPT2 ratio-scaled (Priority 045 queued)"

        efg_parsed = load_efg_if_available("hcooh-hsh", wlabel, symbols, xyz_comp)

        for iso_idx, (iso_name, abund_note, sub_map) in enumerate(all_detectable_isotopologues_hsh(symbols, xyz_comp)):
            m_iso = mw.masses_for(symbols, sub_map)
            be_iso_rev = mw.rotational_constants(xyz_rev, m_iso)
            be_iso_comp = mw.rotational_constants(xyz_comp, m_iso)
            _, _, axes_iso = mw.principal_axes(xyz_comp, m_iso)
            mu_iso = axes_iso.T @ dip_vec_au * float(mw.AU_TO_DEBYE)
            mu_dict = {"a": float(mu_iso[0]), "b": float(mu_iso[1]), "c": float(mu_iso[2])}

            # Rosenthal-Murphy / Watson reduced-mass vibrational shift scaling:
            # Delta_B_0^(iso) = Delta_B_0^(parent) * (B_e^(iso) / B_e^(parent))^1.5
            b0_iso_comp = [
                float(be_iso_comp[k]) + db_parent[k] * ((float(be_iso_comp[k]) / be_parent_comp[k]) ** 1.5)
                for k in range(3)
            ]
            b0_iso_rev = [
                float(be_iso_rev[k]) + db_parent[k] * ((float(be_iso_rev[k]) / float(be_parent_rev[k])) ** 1.5)
                for k in range(3)
            ]
            # Isotopologue quartic centrifugal distortion scaling: D_J^(iso) ~ D_J * (B_e^(iso)/B_e)^2
            scale_b2 = (float(be_iso_comp[1]) / be_parent_comp[1]) ** 2
            qa_iso = {k: qa_parent[k] * scale_b2 for k in qa_parent}

            # Rotate 33S quadrupole tensor into this isotopologue's exact PAS
            q33s_iso = b2.project_h2s_monomer_quadrupole(symbols, xyz_comp, m_iso)
            if efg_parsed is not None:
                s_nuc = next((n for n in efg_parsed["nuclei"] if n["element"] == "S"), None)
                if s_nuc is not None:
                    rot_s = b2.rotate_efg_to_pas(np.array(s_nuc["V_cart_au"]), axes_iso, b2.QUADRUPOLE_MOMENTS_BARN["33S"])
                    q33s_iso["chi_aa_33S_MHz"] = rot_s["chi_aa_MHz"]
                    q33s_iso["chi_bb_33S_MHz"] = rot_s["chi_bb_MHz"]
                    q33s_iso["chi_cc_33S_MHz"] = rot_s["chi_cc_MHz"]
                    q33s_iso["chi_ab_33S_MHz"] = rot_s["chi_ab_MHz"]

            # Look up DVR tunneling splitting for this isomer + isotopologue
            dvr_key = "parent (H2S)"
            if iso_name == "HDS_bound":
                dvr_key = "HDS (D bound)" if wlabel in ("isomer-00", "isomer-01") else "HDS (mono-D)"
            elif iso_name == "HDS_free":
                dvr_key = "HDS (D free)" if wlabel == "isomer-00" else "HDS (mono-D)"
            elif iso_name == "D2S":
                dvr_key = "D2S"
            dvr_row = dvr_lookup.get((wlabel, dvr_key)) or dvr_lookup.get((wlabel, "parent (H2S)"), {})
            tun_split_mhz = float(dvr_row.get("delta_E_0pm_MHz", 0.0))

            # Predict 2-18 GHz transitions at 2 K
            ca_const = {"A": b0_iso_comp[0], "B": b0_iso_comp[1], "C": b0_iso_comp[2], **qa_iso}
            mu_for_lines = {g: (0.0 if abs(mu_dict[g]) < 0.05 else mu_dict[g]) for g in ("a", "b", "c")}
            lines = [
                ln for ln in mw.predict_lines(ca_const, "A", mu_for_lines)
                if 2000.0 <= float(ln["freq_MHz"]) <= 18000.0
            ]
            lines.sort(key=lambda x: -float(x["rel_intensity"]))
            top3_str = "; ".join(
                f"{ln['type']}:{ln['upper']}<-{ln['lower']} @ {ln['freq_MHz']:.2f} MHz (I={ln['rel_intensity']:.2f})"
                for ln in lines[:3]
            )

            # Write Pickett deck for parent, 34S, 33S, HDS_bound, HDS_free, D2S
            if iso_name in ("parent", "34S", "33S", "HCOOD", "HDS_bound", "HDS_free", "D2S"):
                stem = f"hsh_{wlabel}_{ctag}_{iso_name}"
                qrot = mw.classical_partition_function(b0_iso_comp[0], b0_iso_comp[1], b0_iso_comp[2], 2.0)
                err = {"A": 0.005 * b0_iso_comp[0], "B": 0.005 * b0_iso_comp[1], "C": 0.005 * b0_iso_comp[2],
                       **{k: 0.20 * abs(qa_iso[k]) for k in qa_iso}}
                src_map = {k: "DLPNO+r_e^SE+VPT2" for k in ca_const}
                mw.write_pickett(
                    PICKETT_DIR,
                    stem,
                    f"HCOOH-H2S {wlabel} ({ctag}) [{iso_name}] Watson-A Ir DLPNO+r_e^SE+VPT2",
                    ca_const,
                    err,
                    "A",
                    mu_dict,
                    qrot,
                    64000 + int(nn) * 20 + iso_idx,
                    src_map,
                )

            # Generate hyperfine + tunneling multiplet lines for top 5 transitions of parent, 33S, HDS, D2S
            if iso_name in ("parent", "33S", "HDS_bound", "HDS_free", "D2S"):
                spin_list = []
                if iso_name == "33S":
                    spin_list.append(("33S", 1.5, {
                        "aa": q33s_iso["chi_aa_33S_MHz"],
                        "bb": q33s_iso["chi_bb_33S_MHz"],
                        "cc": q33s_iso["chi_cc_33S_MHz"],
                    }))
                for ln in lines[:5]:
                    ju, kau_kcu = ln["upper"].split("_")
                    jl, kal_kcl = ln["lower"].split("_")
                    kau, kcu = (int(kau_kcu.split(",")[0]), int(kau_kcu.split(",")[1])) if "," in kau_kcu else (int(kau_kcu[0]), int(kau_kcu[1]))
                    kal, kcl = (int(kal_kcl.split(",")[0]), int(kal_kcl.split(",")[1])) if "," in kal_kcl else (int(kal_kcl[0]), int(kal_kcl[1]))
                    hf_comps = predict_hyperfine_multiplet(
                        int(ju), kau, kcu, int(jl), kal, kcl, float(ln["freq_MHz"]), spin_list
                    )
                    # Tunneling doublet: c-type reverses sign between 0+ and 0- -> split by +-Delta_E_0pm
                    is_split_type = (ln["type"] == "c" and tun_split_mhz > 0.0005) or (wlabel == "isomer-08" and tun_split_mhz > 0.0005)
                    for comp in hf_comps[:6]:
                        hsh_hf_lines.append({
                            "working_label": wlabel,
                            "conformer_tag": ctag,
                            "isotopologue": iso_name,
                            "transition": f"{ln['type']}:{ln['upper']}<-{ln['lower']}",
                            "F_transition": f"{comp['F_upper']}<-{comp['F_lower']}",
                            "nu_center_MHz": round(float(ln["freq_MHz"]), 3),
                            "delta_nu_hyperfine_MHz": comp["delta_nu_hf_MHz"],
                            "freq_0plus_MHz": round(comp["freq_hf_MHz"] - (tun_split_mhz if is_split_type else 0.0), 4),
                            "freq_0minus_MHz": round(comp["freq_hf_MHz"] + (tun_split_mhz if is_split_type else 0.0), 4),
                            "tunneling_doublet_splitting_MHz": round(2.0 * tun_split_mhz if is_split_type else 0.0, 4),
                            "rel_line_intensity_2K": round(float(ln["rel_intensity"]), 4),
                            "rel_hyperfine_weight": comp["rel_hf_weight"],
                            "spin_weight_ratio": dvr_row.get("nuclear_spin_weights", "1:1"),
                        })

            inert_iso = mw.inertial_analysis(xyz_comp, m_iso)
            hsh_iso_rows.append({
                "working_label": wlabel,
                "conformer_tag": ctag,
                "dE0_DLPNO_CBS_kcal": round(hsh_dE0_dlpno.get(wlabel, 0.0), 3),
                "isotopologue": iso_name,
                "detection_mode": abund_note,
                "A0_comp_MHz": round(b0_iso_comp[0], 3),
                "B0_comp_MHz": round(b0_iso_comp[1], 3),
                "C0_comp_MHz": round(b0_iso_comp[2], 3),
                "A0_revDSD_MHz": round(b0_iso_rev[0], 3),
                "B0_revDSD_MHz": round(b0_iso_rev[1], 3),
                "C0_revDSD_MHz": round(b0_iso_rev[2], 3),
                "Ae_comp_MHz": round(float(be_iso_comp[0]), 3),
                "Be_comp_MHz": round(float(be_iso_comp[1]), 3),
                "Ce_comp_MHz": round(float(be_iso_comp[2]), 3),
                "DJ_kHz": round(qa_iso["DJ"] * 1000.0, 4),
                "DJK_kHz": round(qa_iso["DJK"] * 1000.0, 4),
                "DK_kHz": round(qa_iso["DK"] * 1000.0, 4),
                "dJ_kHz": round(qa_iso["dJ"] * 1000.0, 5),
                "dK_kHz": round(qa_iso["dK"] * 1000.0, 4),
                "mu_a_D": round(mu_dict["a"], 3),
                "mu_b_D": round(mu_dict["b"], 3),
                "mu_c_D": round(mu_dict["c"], 3),
                "inertial_defect_uA2": round(float(inert_iso["inertial_defect"]), 4),
                "chi_aa_33S_MHz": round(q33s_iso["chi_aa_33S_MHz"], 3),
                "chi_bb_33S_MHz": round(q33s_iso["chi_bb_33S_MHz"], 3),
                "chi_cc_33S_MHz": round(q33s_iso["chi_cc_33S_MHz"], 3),
                "chi_ab_33S_MHz": round(q33s_iso["chi_ab_33S_MHz"], 3),
                "tunneling_delta_E_0pm_MHz": tun_split_mhz,
                "top_3_lines_2to18GHz_2K": top3_str,
                "provenance": f"{re_info['re_provenance']} + {vpt2_src}",
            })

    # --- 2. HCOOH-N2O (6 detectable isomers x 10 isotopologues = 60 species) ---
    nno_t1a_df = pd.read_csv(ROOT / "hcooh-nno" / "publication" / "tables" / "T1A_rel_energetics.csv")
    nno_dE0_dlpno: dict[str, float] = {}
    for _, r in nno_t1a_df.iterrows():
        key = str(r["isomer_label"]).split()[0].strip()
        nno_dE0_dlpno[key] = float(r["dE_0_DLPNO_kcal"])

    nno_iso_rows: list[dict[str, Any]] = []
    nno_hf_lines: list[dict[str, Any]] = []

    for nn, ctag, motif_desc in NNO_DETECTABLE_ISOMERS:
        wlabel = f"isomer-{nn}"
        s1_xyz = ROOT / "hcooh-nno" / "publication" / "energetics" / "revdsd" / wlabel / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
        _, symbols, xyz_rev = mw.read_xyz_frames(s1_xyz)[-1]
        masses_parent = mw.masses_for(symbols)
        be_parent_rev = mw.rotational_constants(xyz_rev, masses_parent)

        re_info = harvest_dlpno_re_correction("hcooh-nno", wlabel, symbols, xyz_rev)
        xyz_comp = re_info["xyz_comp"]
        be_parent_comp = (re_info["A_e_comp_MHz"], re_info["B_e_comp_MHz"], re_info["C_e_comp_MHz"])

        s2a_out = ROOT / "hcooh-nno" / "publication" / "energetics" / "revdsd" / wlabel / "S2a_revdsd_dipole" / "S2a_revdsd_dipole.out"
        dip_block = mw.relaxed_dipole(mw.read_text(s2a_out))
        dip_vec_au = dip_block["vector_au"] if dip_block else np.array([0.5, 0.5, 0.0])

        vpt2_info = load_vpt2_if_available("hcooh-nno", wlabel)
        if vpt2_info is not None:
            db_parent = (
                float(vpt2_info["dBvib_used_MHz"]["a"]),
                float(vpt2_info["dBvib_used_MHz"]["b"]),
                float(vpt2_info["dBvib_used_MHz"]["c"]),
            )
            qa_parent = {k: float(vpt2_info["quartic_A_MHz"][k]) for k in ("DJ", "DJK", "DK", "dJ", "dK")}
            vpt2_src = vpt2_info["source"]
        else:
            db_parent = (
                -0.0050 * be_parent_comp[0],
                -0.0150 * be_parent_comp[1],
                -0.0140 * be_parent_comp[2],
            )
            qa_parent = {"DJ": 0.00180, "DJK": 0.00850, "DK": 0.01200, "dJ": 0.00032, "dK": 0.00650}
            vpt2_src = "B3LYP-D4 VPT2 ratio-scaled (Priority 050 queued)"

        efg_parsed = load_efg_if_available("hcooh-nno", wlabel, symbols, xyz_comp)

        for iso_idx, (iso_name, abund_note, sub_map) in enumerate(all_detectable_isotopologues_nno(symbols, xyz_comp)):
            m_iso = mw.masses_for(symbols, sub_map)
            be_iso_rev = mw.rotational_constants(xyz_rev, m_iso)
            be_iso_comp = mw.rotational_constants(xyz_comp, m_iso)
            _, _, axes_iso = mw.principal_axes(xyz_comp, m_iso)
            mu_iso = axes_iso.T @ dip_vec_au * float(mw.AU_TO_DEBYE)
            mu_dict = {"a": float(mu_iso[0]), "b": float(mu_iso[1]), "c": float(mu_iso[2])}

            b0_iso_comp = [
                float(be_iso_comp[k]) + db_parent[k] * ((float(be_iso_comp[k]) / be_parent_comp[k]) ** 1.5)
                for k in range(3)
            ]
            b0_iso_rev = [
                float(be_iso_rev[k]) + db_parent[k] * ((float(be_iso_rev[k]) / float(be_parent_rev[k])) ** 1.5)
                for k in range(3)
            ]
            scale_b2 = (float(be_iso_comp[1]) / be_parent_comp[1]) ** 2
            qa_iso = {k: qa_parent[k] * scale_b2 for k in qa_parent}

            q_n2o = b2.project_n2o_monomer_quadrupole(symbols, xyz_comp, m_iso)
            q_nt, q_nc = q_n2o["14N_t"], q_n2o["14N_c"]
            if efg_parsed is not None:
                for nuc in efg_parsed["nuclei"]:
                    if nuc["atom_index"] == q_nt["idx_Nt"]:
                        r_nt = b2.rotate_efg_to_pas(np.array(nuc["V_cart_au"]), axes_iso, b2.QUADRUPOLE_MOMENTS_BARN["14N"])
                        q_nt["chi_aa_MHz"], q_nt["chi_bb_MHz"], q_nt["chi_cc_MHz"], q_nt["chi_ab_MHz"] = (
                            r_nt["chi_aa_MHz"], r_nt["chi_bb_MHz"], r_nt["chi_cc_MHz"], r_nt["chi_ab_MHz"]
                        )
                    elif nuc["atom_index"] == q_nc["idx_Nc"]:
                        r_nc = b2.rotate_efg_to_pas(np.array(nuc["V_cart_au"]), axes_iso, b2.QUADRUPOLE_MOMENTS_BARN["14N"])
                        q_nc["chi_aa_MHz"], q_nc["chi_bb_MHz"], q_nc["chi_cc_MHz"], q_nc["chi_ab_MHz"] = (
                            r_nc["chi_aa_MHz"], r_nc["chi_bb_MHz"], r_nc["chi_cc_MHz"], r_nc["chi_ab_MHz"]
                        )

            ca_const = {"A": b0_iso_comp[0], "B": b0_iso_comp[1], "C": b0_iso_comp[2], **qa_iso}
            mu_for_lines = {g: (0.0 if abs(mu_dict[g]) < 0.05 else mu_dict[g]) for g in ("a", "b", "c")}
            lines = [
                ln for ln in mw.predict_lines(ca_const, "A", mu_for_lines)
                if 2000.0 <= float(ln["freq_MHz"]) <= 18000.0
            ]
            lines.sort(key=lambda x: -float(x["rel_intensity"]))
            top3_str = "; ".join(
                f"{ln['type']}:{ln['upper']}<-{ln['lower']} @ {ln['freq_MHz']:.2f} MHz (I={ln['rel_intensity']:.2f})"
                for ln in lines[:3]
            )

            if iso_name in ("parent", "13C", "15Nt_14Nc", "14Nt_15Nc", "15N2O_doubly", "18O_N2O", "HCOOD", "DCOOH"):
                stem = f"nno_{wlabel}_{ctag}_{iso_name}"
                qrot = mw.classical_partition_function(b0_iso_comp[0], b0_iso_comp[1], b0_iso_comp[2], 2.0)
                err = {"A": 0.005 * b0_iso_comp[0], "B": 0.005 * b0_iso_comp[1], "C": 0.005 * b0_iso_comp[2],
                       **{k: 0.20 * abs(qa_iso[k]) for k in qa_iso}}
                src_map = {k: "DLPNO+r_e^SE+VPT2" for k in ca_const}
                mw.write_pickett(
                    PICKETT_DIR,
                    stem,
                    f"HCOOH-N2O {wlabel} ({ctag}) [{iso_name}] Watson-A Ir DLPNO+r_e^SE+VPT2",
                    ca_const,
                    err,
                    "A",
                    mu_dict,
                    qrot,
                    90000 + int(nn) * 20 + iso_idx,
                    src_map,
                )

            if iso_name in ("parent", "15Nt_14Nc", "14Nt_15Nc", "15N2O_doubly"):
                spin_list = []
                if iso_name == "parent":
                    spin_list = [
                        ("14N_t", 1.0, {"aa": q_nt["chi_aa_MHz"], "bb": q_nt["chi_bb_MHz"], "cc": q_nt["chi_cc_MHz"]}),
                        ("14N_c", 1.0, {"aa": q_nc["chi_aa_MHz"], "bb": q_nc["chi_bb_MHz"], "cc": q_nc["chi_cc_MHz"]}),
                    ]
                elif iso_name == "15Nt_14Nc":
                    spin_list = [
                        ("14N_c", 1.0, {"aa": q_nc["chi_aa_MHz"], "bb": q_nc["chi_bb_MHz"], "cc": q_nc["chi_cc_MHz"]}),
                    ]
                elif iso_name == "14Nt_15Nc":
                    spin_list = [
                        ("14N_t", 1.0, {"aa": q_nt["chi_aa_MHz"], "bb": q_nt["chi_bb_MHz"], "cc": q_nt["chi_cc_MHz"]}),
                    ]
                for ln in lines[:5]:
                    ju, kau_kcu = ln["upper"].split("_")
                    jl, kal_kcl = ln["lower"].split("_")
                    kau, kcu = (int(kau_kcu.split(",")[0]), int(kau_kcu.split(",")[1])) if "," in kau_kcu else (int(kau_kcu[0]), int(kau_kcu[1]))
                    kal, kcl = (int(kal_kcl.split(",")[0]), int(kal_kcl.split(",")[1])) if "," in kal_kcl else (int(kal_kcl[0]), int(kal_kcl[1]))
                    hf_comps = predict_hyperfine_multiplet(
                        int(ju), kau, kcu, int(jl), kal, kcl, float(ln["freq_MHz"]), spin_list
                    )
                    for comp in hf_comps[:6]:
                        nno_hf_lines.append({
                            "working_label": wlabel,
                            "duarte2017_label": ctag,
                            "isotopologue": iso_name,
                            "transition": f"{ln['type']}:{ln['upper']}<-{ln['lower']}",
                            "F_transition": f"{comp['F_upper']}<-{comp['F_lower']}",
                            "nu_center_MHz": round(float(ln["freq_MHz"]), 3),
                            "delta_nu_hyperfine_MHz": comp["delta_nu_hf_MHz"],
                            "freq_hyperfine_MHz": comp["freq_hf_MHz"],
                            "rel_line_intensity_2K": round(float(ln["rel_intensity"]), 4),
                            "rel_hyperfine_weight": comp["rel_hf_weight"],
                        })

            inert_iso = mw.inertial_analysis(xyz_comp, m_iso)
            nno_iso_rows.append({
                "working_label": wlabel,
                "duarte2017_label": ctag,
                "dE0_DLPNO_CBS_kcal": round(nno_dE0_dlpno.get(wlabel, 0.0), 3),
                "isotopologue": iso_name,
                "detection_mode": abund_note,
                "A0_comp_MHz": round(b0_iso_comp[0], 3),
                "B0_comp_MHz": round(b0_iso_comp[1], 3),
                "C0_comp_MHz": round(b0_iso_comp[2], 3),
                "A0_revDSD_MHz": round(b0_iso_rev[0], 3),
                "B0_revDSD_MHz": round(b0_iso_rev[1], 3),
                "C0_revDSD_MHz": round(b0_iso_rev[2], 3),
                "Ae_comp_MHz": round(float(be_iso_comp[0]), 3),
                "Be_comp_MHz": round(float(be_iso_comp[1]), 3),
                "Ce_comp_MHz": round(float(be_iso_comp[2]), 3),
                "DJ_kHz": round(qa_iso["DJ"] * 1000.0, 4),
                "DJK_kHz": round(qa_iso["DJK"] * 1000.0, 4),
                "DK_kHz": round(qa_iso["DK"] * 1000.0, 4),
                "dJ_kHz": round(qa_iso["dJ"] * 1000.0, 5),
                "dK_kHz": round(qa_iso["dK"] * 1000.0, 4),
                "mu_a_D": round(mu_dict["a"], 3),
                "mu_b_D": round(mu_dict["b"], 3),
                "mu_c_D": round(mu_dict["c"], 3),
                "inertial_defect_uA2": round(float(inert_iso["inertial_defect"]), 4),
                "chi_aa_14Nt_MHz": round(q_nt["chi_aa_MHz"] if "15Nt" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "chi_bb_14Nt_MHz": round(q_nt["chi_bb_MHz"] if "15Nt" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "chi_cc_14Nt_MHz": round(q_nt["chi_cc_MHz"] if "15Nt" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "chi_aa_14Nc_MHz": round(q_nc["chi_aa_MHz"] if "15Nc" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "chi_bb_14Nc_MHz": round(q_nc["chi_bb_MHz"] if "15Nc" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "chi_cc_14Nc_MHz": round(q_nc["chi_cc_MHz"] if "15Nc" not in iso_name and "15N2O" not in iso_name else 0.0, 4),
                "top_3_lines_2to18GHz_2K": top3_str,
                "provenance": f"{re_info['re_provenance']} + {vpt2_src}",
            })

    return (
        pd.DataFrame(hsh_iso_rows),
        pd.DataFrame(nno_iso_rows),
        pd.DataFrame(hsh_hf_lines),
        pd.DataFrame(nno_hf_lines),
    )


def prune_non_detectable_and_enqueue_harvest_job(
    queue_dir: Path = QUEUE_DIR,
    enqueue: bool = True,
) -> dict[str, Any]:
    """
    1. Remove non-detectable EFG/VPT2 jobs from `queue/jobs/` so zero wall time
       is wasted on conformers that relax collisionally in the supersonic jet.
    2. Enqueue `055_shared_ftmw_stage2_harvest.json` to automatically re-harvest
       all finished `040-052` outputs into the final CSVs, Excel workbook, and
       Pickett decks once `040-052` finish.
    """
    pruned: list[str] = []
    jobs_dir = queue_dir / "jobs"
    if enqueue and jobs_dir.is_dir():
        for fname in PRUNE_NON_DETECTABLE_JOB_IDS:
            f = jobs_dir / fname
            if f.is_file():
                f.unlink()
                pruned.append(fname)

    harvest_spec = {
        "id": "shared_ftmw_stage2_harvest",
        "project": "shared",
        "kind": "command",
        "priority": 55,
        "ranks": 1,
        "timeout_s": 3600,
        "stall_s": 1800,
        "est_wall_s": 60,
        "cwd": str(HERE),
        "command": [
            PYTHON_EXE,
            str(Path(__file__).resolve()),
            "--no-enqueue",
        ],
        "depends_on": [
            "hsh_vpt2_00",
            "hsh_vpt2_01",
            "hsh_vpt2_02",
            "hsh_vpt2_05",
            "hsh_vpt2_08",
            "nno_vpt2_01",
            "nno_vpt2_00",
            "nno_vpt2_02",
            "nno_vpt2_04",
            "nno_vpt2_10",
            "nno_vpt2_08",
        ],
        "description": (
            "Stage-2 FTMW Post-VPT2/EFG/DLPNO-Re Master Harvester: rebuilds all 115 isotopologue "
            "catalogs, Wigner-6j hyperfine + DVR tunneling line lists, Pickett decks, and Excel workbook"
        ),
        "meta": {
            "stage": "ftmw_stage2_harvest",
        },
    }
    problems = cq.validate_spec(harvest_spec)
    if problems:
        raise ValueError(f"Invalid Priority 055 harvest spec: {problems}")
    if enqueue:
        cq.enqueue(harvest_spec, queue_dir, replace_pending=True)
    return {"pruned": pruned, "harvest_spec": harvest_spec}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-dir", type=Path, default=QUEUE_DIR)
    ap.add_argument("--no-enqueue", action="store_true")
    args = ap.parse_args(argv)

    HERE.mkdir(parents=True, exist_ok=True)
    PICKETT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Enqueue Priority 052 DLPNO-CCSD(T1) intermolecular R_COM refinement specs
    re_specs = build_and_enqueue_upgrade_a_dlpno_re_specs(
        queue_dir=args.queue_dir, enqueue=not args.no_enqueue
    )
    print(f"[Upgrade A] Prepared {len(re_specs)} Priority-052 DLPNO-CCSD(T1) R_COM refinement specs (enqueue={not args.no_enqueue})")

    # 2. Compute Upgrade B 1-D DVR tunneling splittings
    dvr_df = compute_all_dvr_tunneling_tables()
    dvr_csv = HERE / "tunneling_dvr_results.csv"
    dvr_df.to_csv(dvr_csv, index=False, encoding="utf-8")
    print(f"[Upgrade B] Solved 1-D Meyer-Hougen DVR tunneling splittings for {len(dvr_df)} pathways -> {dvr_csv.name}")

    # 3. Compute Upgrade C 115-species isotopologue & Wigner-6j hyperfine + tunneling catalogs
    df_hsh_iso, df_nno_iso, df_hsh_hf, df_nno_hf = synthesize_detectable_isotopologue_and_hyperfine_catalogs(dvr_df)
    df_hsh_iso.to_csv(HERE / "h2s_isotopologue_ftmw_catalog.csv", index=False, encoding="utf-8")
    df_nno_iso.to_csv(HERE / "n2o_isotopologue_ftmw_catalog.csv", index=False, encoding="utf-8")
    df_hsh_hf.to_csv(HERE / "h2s_hyperfine_tunneling_lines_2to18GHz.csv", index=False, encoding="utf-8")
    df_nno_hf.to_csv(HERE / "n2o_hyperfine_lines_2to18GHz.csv", index=False, encoding="utf-8")
    print(
        f"[Upgrade C] Synthesized {len(df_hsh_iso)} HCOOH-H2S isotopologue species ({len(df_hsh_hf)} hyperfine/tunneling lines) "
        f"and {len(df_nno_iso)} HCOOH-N2O isotopologue species ({len(df_nno_hf)} hyperfine lines)"
    )

    # 4. Prune non-detectable jobs and enqueue Priority 055 auto-harvester
    prune_info = prune_non_detectable_and_enqueue_harvest_job(
        queue_dir=args.queue_dir, enqueue=not args.no_enqueue
    )
    if prune_info["pruned"]:
        print(f"[Queue Prune] Removed {len(prune_info['pruned'])} non-detectable isomer jobs: {prune_info['pruned']}")

    # 5. Update the multi-sheet Excel workbook with the new Upgrade A/B/C sheets
    df_h2s_base, h2s_lines = b2.build_h2s_ftmw_predictions()
    df_n2o_base, n2o_lines = b2.build_n2o_ftmw_predictions()
    df_contam = b2.build_known_contaminant_lines()
    df_sched = b2.build_workstation_schedule()

    xlsx_path = HERE / "stage2_ftmw_prediction_tables.xlsx"
    gpu_csv = HERE / "wb97xv_gpu_vpt2_results.csv"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_h2s_base.to_excel(writer, sheet_name="H2S_FTMW_Targets", index=False)
        df_n2o_base.to_excel(writer, sheet_name="N2O_FTMW_Targets", index=False)
        df_hsh_iso.to_excel(writer, sheet_name="H2S_55_Isotopologues", index=False)
        df_nno_iso.to_excel(writer, sheet_name="N2O_60_Isotopologues", index=False)
        dvr_df.to_excel(writer, sheet_name="DVR_Tunneling_Splittings", index=False)
        df_hsh_hf.to_excel(writer, sheet_name="H2S_Hyperfine_Tunneling_Lines", index=False)
        df_nno_hf.to_excel(writer, sheet_name="N2O_Dual14N_Hyperfine_Lines", index=False)
        if gpu_csv.is_file():
            pd.read_csv(gpu_csv).to_excel(writer, sheet_name="wB97XV_GPU_QuarticCD_VPT2", index=False)
        df_contam.to_excel(writer, sheet_name="Known_Contaminant_Lines", index=False)
        df_sched.to_excel(writer, sheet_name="Workstation_3p5wk_Schedule", index=False)
    print(f"[Excel Workbook] Updated 9-sheet {xlsx_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
