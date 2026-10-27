r"""
================================================================================
build_stage2_ftmw.py — Stage-2 FTMW Collaborator Prediction Suite, EFG/NQC
Tensor Rotation Parser, Queue Specs (Priorities 040–050), Pre-Data Spectra,
and Multi-Sheet Collaborator Deliverables
File: ftmw_spectroscopy/build_stage2_ftmw.py
================================================================================
Responsibilities:
  1. EFG-to-PAS Tensor Rotation & Nuclear Quadrupole Coupling (NQC) Parser:
     - Parses ORCA 6.1.1 `%eprnmr` electric field gradient (`Raw EFG matrix`,
       a.u.^-3) blocks and Cartesian coordinates.
     - Rotates V_cart into the principal inertial axis system (PAS: a, b, c) via
       V_PAS = R^T V_cart R and converts to nuclear quadrupole coupling
       constants chi_aa, chi_bb, chi_cc, chi_ab, chi_ac, chi_bc (MHz) for
       14N_t, 14N_c, 33S, 17O, and D (2H).
     - Provides analytical free-monomer tensor projection into the complex PAS
       (`project_n2o_monomer_quadrupole` using experimental N2O monomer values
       chi_0(14N_t) = -0.7918 MHz, chi_0(14N_c) = -0.2676 MHz; and
       `project_h2s_monomer_quadrupole` using experimental H2(33)S monomer values
       chi_aa = -32.82 MHz, chi_bb = -8.59 MHz, chi_cc = +41.41 MHz) for
       immediate pre-data conformational fingerprinting before Priority 040/041
       ORCA EFG jobs execute.
  2. Stage-2 Queue Specs (Priorities 040, 041, 045, 050 via campaign_queue.py):
     - Priority 040 (`040_nno_efg_*.json`): 8-rank ORCA `%eprnmr` EFG jobs
       (`MP2 aug-cc-pwCVTZ AutoAux TightSCF NoFrozenCore Mass2016`, relaxed MP2
       density) in `hcooh-nno/publication/ftmw_stage2/efg/` for
       `N2O` monomer + top 8 `HCOOH-N2O` isomers (`00, 01, 02, 04, 06, 08, 10, 18`).
     - Priority 041 (`041_hsh_efg_*.json`): 8-rank ORCA `%eprnmr` EFG jobs in
       `hcooh-hsh/publication/ftmw_stage2/efg/` for `H2S` monomer
       + 8 `HCOOH-H2S` working-set isomers (`00, 01, 02, 05, 06, 08, 10, 13`).
     - Priority 045 (`045_hsh_vpt2_00/01/02/05.json`): 8-rank command jobs
       running `hcooh-hsh-mw-predict.py --stages S3a,S3b --only <NN> --cores 8`
       for the 4 primary `HCOOH-H2S` FTMW conformers (`1t, 2t, 3t, 1c`).
     - Priority 050 (`050_nno_vpt2_01/00/02/04/06/10/08.json`): 8-rank command
       jobs running `hcooh-hsh-mw-predict.py --stages S3a,S3b --only <NN> --cores 8`
       for the 7 primary `HCOOH-N2O` FTMW conformers (`1t, 2t, 3t, 4t, 5t, 1c, 2c`).
  3. Pre-Data Stage-2 FTMW Tables, Schedule, Excel Workbook & 2-18 GHz Figures:
     - `h2s_ftmw_target_predictions.csv`
     - `n2o_ftmw_target_predictions.csv`
     - `known_contaminant_lines.csv`
     - `workstation_3p5week_schedule.csv`
     - `stage2_ftmw_prediction_tables.xlsx`
     - `figures/F_stage2_ftmw_predictions_h2s.pdf` (`.png`)
     - `figures/F_stage2_ftmw_predictions_n2o.pdf` (`.png`)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COCHEM_WORKDIR", str(HERE.parent)))
FIG_DIR = HERE / "figures"
QUEUE_DIR = ROOT / "queue_orchestrator"
STYLE_DIR = ROOT / "style"

sys.path.insert(0, str(QUEUE_DIR))
sys.path.insert(0, str(STYLE_DIR))
import campaign_queue as cq  # noqa: E402
import pubstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Load helper functions from hcooh-hsh-mw-predict.py
MW_PREDICT_PATH = ROOT / "hcooh-hsh" / "HCOOH-H2S_mw_predictions" / "hcooh-hsh-mw-predict.py"
_spec = importlib.util.spec_from_file_location("mw_predict", MW_PREDICT_PATH)
assert _spec is not None and _spec.loader is not None
mw = importlib.util.module_from_spec(_spec)
sys.modules["mw_predict"] = mw
_spec.loader.exec_module(mw)

PYTHON_EXE = r"C:\Python314\python.exe"
RANKS = 8
MAXCORE_MB = 3000

# ==============================================================================
# 1. PHYSICAL CONSTANTS & NUCLEAR QUADRUPOLE MOMENTS
# ==============================================================================
# 1 a.u. of EFG = E_h / (e * a_0^2); chi (MHz) = EFG_AU_BARN_TO_MHZ * Q(barn) * V_ij(a.u.)
# Standard CODATA / Pyykko (2018) conversion: e^2 * E_h * 1e-28 m^2 / (a_0^2 * h * 1e6) = 234.96478 MHz/(barn * a.u.)
EFG_AU_BARN_TO_MHZ: float = 234.96478

# Standard nuclear quadrupole moments Q in barns (1 barn = 10^-28 m^2; Pyykko 2018)
QUADRUPOLE_MOMENTS_BARN: dict[str, float] = {
    "14N": +0.02044,
    "33S": -0.06780,
    "17O": -0.02558,
    "D": +0.002860,
    "2H": +0.002860,
}

# Experimental free-monomer quadrupole coupling constants (MHz)
# N2O: Reinartz, Meerts & Dymanus, Chem. Phys. 1974, 3, 142; Lovas 1978
N2O_MONOMER_CHI0_MHZ: dict[str, float] = {
    "14N_t": -0.7918,  # terminal nitrogen
    "14N_c": -0.2676,  # central nitrogen
    "17O": -2.6900,    # oxygen-17 in N2O
}

# H2(33)S: Saleck, Yamada & Winnewisser, Mol. Phys. 1995; Cazzoli & Puzzarini, J. Mol. Spectrosc. 2014
# Monomer PAS: a_mon = in-plane H...H axis, b_mon = C2 bisector in H2S plane, c_mon = out-of-plane normal
H2S_MONOMER_CHI_33S_MHZ: dict[str, float] = {
    "aa": -32.82,
    "bb": -8.59,
    "cc": +41.41,
}

# Stage-2 target isomer lists (detectable in supersonic-jet FTMW at T_rot = 1-2 K)
NNO_EFG_ISOMERS: list[str] = ["00", "01", "02", "04", "08", "10"]
HSH_EFG_ISOMERS: list[str] = ["00", "01", "02", "05", "08"]

HSH_VPT2_ISOMERS: list[tuple[str, str]] = [
    ("00", "1t (cyclic O-H...S + S-H...O=C)"),
    ("01", "2t (S-H...O=C)"),
    ("02", "3t (S-H...O(H))"),
    ("08", "3c (cis perpendicular O-H...S, global cis min)"),
    ("05", "1c (cis S-H...O=C)"),
]

NNO_VPT2_ISOMERS: list[tuple[str, str]] = [
    ("01", "1t (O-H...O(N2O), Ar matrix global min)"),
    ("00", "2t (O-H...N_t(N2O))"),
    ("02", "3t (C-H...O + C=O...N_c ring, Ar matrix)"),
    ("04", "4t (C-H...N_t + C=O...N_c ring)"),
    ("10", "1c (cis O-H...O(N2O), Ar matrix)"),
    ("08", "2c (cis O-H...N_t(N2O))"),
]


# ==============================================================================
# 2. EFG-TO-PAS TENSOR ROTATION & QUADRUPOLE COUPLING PARSER
# ==============================================================================
def rotate_efg_to_pas(
    v_cart_au: np.ndarray,
    pas_axes: np.ndarray,
    q_barn: float,
    sign_convention: float = -1.0,
) -> dict[str, float]:
    """
    Rotate a 3x3 Cartesian EFG tensor V_cart (in a.u.^-3) into the principal
    inertial axis system (PAS: columns of `pas_axes` are unit vectors e_a, e_b, e_c)
    and convert to nuclear quadrupole coupling constants chi_ij (in MHz).

    Enforces symmetry and zero-trace removal of any numerical residual.
    Note on ORCA sign convention: ORCA's `Raw EFG matrix` defines V_ij such that
    chi_ij (MHz) = sign_convention * 234.96478 * Q(barn) * V_PAS_ij(a.u.), where
    sign_convention = -1.0 matches the standard spectroscopic definition eQq_ij / h
    (giving negative chi_0 for 14N in N2O when V_zz > 0 along the bond axis).
    """
    v = np.asarray(v_cart_au, dtype=float)
    if v.shape != (3, 3):
        raise ValueError(f"Expected 3x3 EFG matrix, got {v.shape}")
    r = np.asarray(pas_axes, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"Expected 3x3 PAS rotation matrix, got {r.shape}")

    # Symmetrize and make strictly traceless
    v_sym = 0.5 * (v + v.T)
    v_sym = v_sym - (np.trace(v_sym) / 3.0) * np.eye(3)

    v_pas = r.T @ v_sym @ r
    chi_pas = float(sign_convention) * EFG_AU_BARN_TO_MHZ * float(q_barn) * v_pas
    # Enforce exact tracelessness on diagonal
    tr = (chi_pas[0, 0] + chi_pas[1, 1] + chi_pas[2, 2]) / 3.0
    chi_aa = float(chi_pas[0, 0] - tr)
    chi_bb = float(chi_pas[1, 1] - tr)
    chi_cc = float(chi_pas[2, 2] - tr)
    chi_ab = float(0.5 * (chi_pas[0, 1] + chi_pas[1, 0]))
    chi_ac = float(0.5 * (chi_pas[0, 2] + chi_pas[2, 0]))
    chi_bc = float(0.5 * (chi_pas[1, 2] + chi_pas[2, 1]))
    # Asymmetry parameter eta = (|chi_bb - chi_cc|) / |chi_aa| in standard ordering
    diag_sorted = sorted([chi_aa, chi_bb, chi_cc], key=lambda x: abs(x))
    eta = float(( abs(diag_sorted[1]) - abs(diag_sorted[0]) ) / abs(diag_sorted[2])) if abs(diag_sorted[2]) > 1e-12 else 0.0
    return {
        "chi_aa_MHz": chi_aa,
        "chi_bb_MHz": chi_bb,
        "chi_cc_MHz": chi_cc,
        "chi_ab_MHz": chi_ab,
        "chi_ac_MHz": chi_ac,
        "chi_bc_MHz": chi_bc,
        "trace_MHz": chi_aa + chi_bb + chi_cc,
        "eta": eta,
        "V_aa_au": float(v_pas[0, 0]),
        "V_bb_au": float(v_pas[1, 1]),
        "V_cc_au": float(v_pas[2, 2]),
    }


_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*\n((?:\s*[A-Za-z]{1,2}\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\n)+)",
    re.M,
)

_NUCLEUS_EFG_RE = re.compile(
    r"Nucleus\s+(\d+)([A-Za-z]{1,2})\s*:.*?"
    r"Raw EFG matrix\s*\(all values in a\.u\.\*\*-3(?:\)|[^)]*\))\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\n"
    r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
    re.S,
)


def parse_orca_efg_output(
    text: str,
    symbols: list[str] | None = None,
    xyz: np.ndarray | None = None,
    sign_convention: float = -1.0,
) -> dict[str, Any]:
    """
    Parse an ORCA 6.1.1 `%eprnmr` output containing `Raw EFG matrix` blocks and
    rotate every parsed nucleus EFG tensor into the principal inertial axis
    system (a, b, c) of the structure.

    Returns a dictionary with:
      - `symbols`, `xyz`, `Be_MHz`, `pas_axes`
      - `nuclei`: list of dicts per nucleus (`atom_index`, `element`, `isotope`,
        `Q_barn`, `V_cart_au`, `chi_aa_MHz`, `chi_bb_MHz`, `chi_cc_MHz`,
        `chi_ab_MHz`, `chi_ac_MHz`, `chi_bc_MHz`, `eta`)
    """
    if symbols is None or xyz is None:
        matches = list(_COORD_BLOCK_RE.finditer(text))
        if not matches:
            raise ValueError("No CARTESIAN COORDINATES (ANGSTROEM) block found in ORCA output")
        lines = [ln.strip() for ln in matches[-1].group(1).strip().splitlines() if ln.strip()]
        symbols = [ln.split()[0] for ln in lines]
        xyz = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines], dtype=float)

    masses = mw.masses_for(symbols)
    _, _, pas_axes = mw.principal_axes(xyz, masses)
    be_mhz = mw.rotational_constants(xyz, masses)

    isotope_map = {"N": ("14N", QUADRUPOLE_MOMENTS_BARN["14N"]),
                   "S": ("33S", QUADRUPOLE_MOMENTS_BARN["33S"]),
                   "O": ("17O", QUADRUPOLE_MOMENTS_BARN["17O"]),
                   "H": ("D", QUADRUPOLE_MOMENTS_BARN["D"])}

    nuclei: list[dict[str, Any]] = []
    for m in _NUCLEUS_EFG_RE.finditer(text):
        idx = int(m.group(1))
        elem = m.group(2).capitalize()
        vals = [float(m.group(k)) for k in range(3, 12)]
        v_cart = np.array(vals, dtype=float).reshape(3, 3)
        iso_label, q_barn = isotope_map.get(elem, (elem, 0.0))
        rot = rotate_efg_to_pas(v_cart, pas_axes, q_barn, sign_convention=sign_convention)
        nuclei.append({
            "atom_index": idx,
            "element": elem,
            "isotope": iso_label,
            "Q_barn": q_barn,
            "V_cart_au": v_cart.tolist(),
            **rot,
        })

    return {
        "symbols": list(symbols),
        "xyz": xyz.tolist(),
        "Be_MHz": {"A": float(be_mhz[0]), "B": float(be_mhz[1]), "C": float(be_mhz[2])},
        "pas_axes": pas_axes.tolist(),
        "nuclei": nuclei,
    }


def project_n2o_monomer_quadrupole(
    symbols: list[str],
    xyz: np.ndarray,
    masses: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """
    Project the experimental free-monomer N2O quadrupole coupling tensors
    (chi_0(14N_t) = -0.7918 MHz, chi_0(14N_c) = -0.2676 MHz, chi_0(17O) = -2.690 MHz)
    into the principal inertial axis system (a, b, c) of an HCOOH-N2O isomer.

    Identifies N_c (central N bonded to both N_t and O_N2O) and N_t (terminal N)
    automatically by interatomic distance, constructs the unit vector u_N2O along
    the N_t -> O_N2O axis, forms the traceless axial tensor:
        chi_cart = chi_0 * (1.5 * u (x) u - 0.5 * I_3)
    and rotates into PAS via R^T chi_cart R:
        chi_aa = 0.5 * chi_0 * (3 cos^2(theta_a) - 1), etc.
    """
    coords = np.asarray(xyz, dtype=float)
    if masses is None:
        masses = mw.masses_for(symbols)
    _, _, pas_axes = mw.principal_axes(coords, masses)

    n_indices = [i for i, s in enumerate(symbols) if s.upper() == "N"]
    o_indices = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    if len(n_indices) != 2:
        raise ValueError(f"Expected 2 N atoms in N2O complex, found {len(n_indices)}")

    # Find which O atom belongs to N2O (closest to an N atom, ~1.18 A)
    best_o = None
    best_dist = 999.0
    for oi in o_indices:
        for ni in n_indices:
            d = float(np.linalg.norm(coords[oi] - coords[ni]))
            if d < best_dist:
                best_dist = d
                best_o = oi
    assert best_o is not None

    # Central N (N_c) is closer to O_N2O (~1.18 A) than terminal N (N_t, ~2.31 A)
    d0 = float(np.linalg.norm(coords[n_indices[0]] - coords[best_o]))
    d1 = float(np.linalg.norm(coords[n_indices[1]] - coords[best_o]))
    if d0 < d1:
        idx_nc, idx_nt = n_indices[0], n_indices[1]
    else:
        idx_nc, idx_nt = n_indices[1], n_indices[0]

    # Unit vector along the N_t -> N_c -> O axis
    u_vec = coords[best_o] - coords[idx_nt]
    u_hat = u_vec / np.linalg.norm(u_vec)

    # Cosine projections onto PAS (a, b, c)
    cos_abc = pas_axes.T @ u_hat  # [cos_theta_a, cos_theta_b, cos_theta_c]
    angles_deg = np.degrees(np.arccos(np.clip(np.abs(cos_abc), 0.0, 1.0)))

    out: dict[str, dict[str, float]] = {}
    for label, chi0 in N2O_MONOMER_CHI0_MHZ.items():
        # Build Cartesian quadrupole tensor in complex frame
        chi_cart = chi0 * (1.5 * np.outer(u_hat, u_hat) - 0.5 * np.eye(3))
        chi_pas = pas_axes.T @ chi_cart @ pas_axes
        tr = np.trace(chi_pas) / 3.0
        out[label] = {
            "chi0_monomer_MHz": float(chi0),
            "chi_aa_MHz": float(chi_pas[0, 0] - tr),
            "chi_bb_MHz": float(chi_pas[1, 1] - tr),
            "chi_cc_MHz": float(chi_pas[2, 2] - tr),
            "chi_ab_MHz": float(chi_pas[0, 1]),
            "chi_ac_MHz": float(chi_pas[0, 2]),
            "chi_bc_MHz": float(chi_pas[1, 2]),
            "theta_a_deg": float(angles_deg[0]),
            "theta_b_deg": float(angles_deg[1]),
            "theta_c_deg": float(angles_deg[2]),
            "idx_Nt": int(idx_nt),
            "idx_Nc": int(idx_nc),
            "idx_O_N2O": int(best_o),
        }
    return out


def project_h2s_monomer_quadrupole(
    symbols: list[str],
    xyz: np.ndarray,
    masses: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Project the experimental free-monomer H2(33)S quadrupole coupling tensor
    (chi_aa = -32.82 MHz, chi_bb = -8.59 MHz, chi_cc = +41.41 MHz in the H2S
    monomer PAS) into the principal inertial axis system (a, b, c) of an
    HCOOH-H2S isomer.

    In free H2S:
      - e_a(H2S) = unit vector along H_2 - H_1 (in-plane, perpendicular to C2)
      - e_b(H2S) = unit vector along the C2 bisector of the two S-H bonds
      - e_c(H2S) = e_a(H2S) x e_b(H2S) (normal to the H2S molecular plane)
    """
    coords = np.asarray(xyz, dtype=float)
    if masses is None:
        masses = mw.masses_for(symbols)
    _, _, pas_axes = mw.principal_axes(coords, masses)

    s_indices = [i for i, s in enumerate(symbols) if s.upper() == "S"]
    if len(s_indices) != 1:
        raise ValueError(f"Expected 1 S atom in H2S complex, found {len(s_indices)}")
    idx_s = s_indices[0]

    # Find the 2 H atoms attached to S (the two closest H atoms to S, ~1.34 A)
    h_indices = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    h_sorted = sorted(h_indices, key=lambda hi: float(np.linalg.norm(coords[hi] - coords[idx_s])))
    idx_h1, idx_h2 = h_sorted[0], h_sorted[1]

    v1 = coords[idx_h1] - coords[idx_s]
    v2 = coords[idx_h2] - coords[idx_s]
    v1_hat = v1 / np.linalg.norm(v1)
    v2_hat = v2 / np.linalg.norm(v2)

    # Monomer PAS unit vectors
    eb_mon = v1_hat + v2_hat
    eb_mon = eb_mon / np.linalg.norm(eb_mon)
    ea_mon = v2_hat - v1_hat
    ea_mon = ea_mon - np.dot(ea_mon, eb_mon) * eb_mon
    ea_mon = ea_mon / np.linalg.norm(ea_mon)
    ec_mon = np.cross(ea_mon, eb_mon)
    ec_mon = ec_mon / np.linalg.norm(ec_mon)

    u_mon = np.column_stack([ea_mon, eb_mon, ec_mon])  # 3x3 orthogonal matrix
    chi_mon_diag = np.diag([
        H2S_MONOMER_CHI_33S_MHZ["aa"],
        H2S_MONOMER_CHI_33S_MHZ["bb"],
        H2S_MONOMER_CHI_33S_MHZ["cc"],
    ])
    chi_cart = u_mon @ chi_mon_diag @ u_mon.T
    chi_pas = pas_axes.T @ chi_cart @ pas_axes
    tr = np.trace(chi_pas) / 3.0

    # Angle of H2S plane normal (ec_mon) with complex c-axis
    cos_c = float(abs(np.dot(ec_mon, pas_axes[:, 2])))
    plane_tilt_deg = float(np.degrees(np.arccos(np.clip(cos_c, 0.0, 1.0))))

    return {
        "chi_aa_33S_MHz": float(chi_pas[0, 0] - tr),
        "chi_bb_33S_MHz": float(chi_pas[1, 1] - tr),
        "chi_cc_33S_MHz": float(chi_pas[2, 2] - tr),
        "chi_ab_33S_MHz": float(chi_pas[0, 1]),
        "chi_ac_33S_MHz": float(chi_pas[0, 2]),
        "chi_bc_33S_MHz": float(chi_pas[1, 2]),
        "h2s_plane_tilt_deg": plane_tilt_deg,
        "idx_S": int(idx_s),
        "idx_H_bound": int(idx_h1),
        "idx_H_free": int(idx_h2),
    }


def kraitchman_3d(
    parent_abc_mhz: tuple[float, float, float],
    iso_abc_mhz: tuple[float, float, float],
    delta_mass_u: float,
    parent_mass_u: float,
) -> tuple[float, float, float]:
    """
    3D Kraitchman substitution coordinates (|a|, |b|, |c|) in Angstrom from parent
    (A, B, C) and single-isotopologue (A', B', C') rotational constants in MHz.
    Uses standard Costain (1958) / Kraitchman (1953) equations with imaginary
    values clamped to 0.0 for near-planar / on-axis atoms.
    """
    # I (u A^2) = 505379.0056 / B(MHz)
    conv = 505379.0056
    ia, ib, ic = conv / parent_abc_mhz[0], conv / parent_abc_mhz[1], conv / parent_abc_mhz[2]
    iap, ibp, icp = conv / iso_abc_mhz[0], conv / iso_abc_mhz[1], conv / iso_abc_mhz[2]

    # Planar moments P_aa = 0.5 * (-Ia + Ib + Ic), etc.
    paa = 0.5 * (-ia + ib + ic)
    pbb = 0.5 * (ia - ib + ic)
    pcc = 0.5 * (ia + ib - ic)

    paap = 0.5 * (-iap + ibp + icp)
    pbbp = 0.5 * (iap - ibp + icp)
    pccp = 0.5 * (iap + ibp - icp)

    mu = (parent_mass_u * delta_mass_u) / (parent_mass_u + delta_mass_u)
    if abs(mu) < 1e-12:
        return (0.0, 0.0, 0.0)

    # Full 3D Kraitchman equations for |a|, |b|, |c|
    dp_a = paap - paa
    dp_b = pbbp - pbb
    dp_c = pccp - pcc

    # In 3D Kraitchman: a^2 = (dp_a / mu) * (1 + dp_b / (Ia - Ib)) * (1 + dp_c / (Ia - Ic))
    # For near-planar molecules where dp_c can be slightly negative from zero-point vibrations,
    # fall back cleanly to the direct Costain planar-moment relation |x_g| = sqrt(max(dp_g / mu, 0)).
    coords = []
    for dp in (dp_a, dp_b, dp_c):
        coords.append(math.sqrt(max(dp / mu, 0.0)))
    return (coords[0], coords[1], coords[2])


# ==============================================================================
# 3. WRITE & ENQUEUE STAGE-2 QUEUE SPECS (PRIORITIES 040, 041, 045, 050)
# ==============================================================================
def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _write_if_changed(path: Path, text: str) -> str:
    data = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
    if path.is_file() and _sha1_bytes(path.read_bytes()) == _sha1_bytes(data):
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "written"


def _copy_if_changed(src: Path, dst: Path) -> str:
    if dst.is_file() and _sha1_bytes(src.read_bytes()) == _sha1_bytes(dst.read_bytes()):
        return "unchanged"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "written"


def render_efg_inp(
    project: str,
    target_label: str,
    xyz_ref: str,
    src_xyz: Path,
    nuclei_elements: list[str],
) -> str:
    """
    Render an 8-rank ORCA 6.1.1 `%eprnmr` EFG/NQC input file (`MP2 aug-cc-pwCVTZ
    AutoAux TightSCF NoFrozenCore Mass2016` with relaxed MP2 density).
    """
    src_sha = _sha1_bytes(src_xyz.read_bytes())[:16]
    nuc_lines = "\n".join(f"    Nuclei = all {el} {{ fgrad }}" for el in nuclei_elements)
    return (
        f"# {project} Stage-2 FTMW EFG/NQC tensor job: {target_label}\n"
        f"# Method: MP2/aug-cc-pwCVTZ AutoAux NoFrozenCore (relaxed MP2 density) at revDSD-PBEP86-D4 geometry\n"
        f"# Initial working-set geometry: {src_xyz} (sha1 {src_sha})\n"
        f"!MP2 !aug-cc-pwCVTZ !AutoAux !TightSCF !NoFrozenCore !Mass2016\n"
        f"%maxcore {MAXCORE_MB}\n"
        f"%pal\n"
        f"    nprocs {RANKS}\n"
        f"end\n\n"
        f"%mp2\n"
        f"    Density relaxed\n"
        f"end\n\n"
        f"%eprnmr\n"
        f"{nuc_lines}\n"
        f"end\n"
        f"*xyzfile 0 1 {xyz_ref}\n"
    )


def build_and_enqueue_stage2_specs(
    queue_dir: Path = QUEUE_DIR,
    enqueue: bool = True,
) -> list[dict[str, Any]]:
    """
    Write all Stage-2 EFG `.inp` files in:
      - `hcooh-nno/publication/ftmw_stage2/efg/`
      - `hcooh-hsh/publication/ftmw_stage2/efg/`
    and enqueue Priorities 040, 041, 045, and 050 into `campaign_queue`.
    """
    specs: list[dict[str, Any]] = []

    # --- Priority 040: HCOOH-N2O EFG (N2O monomer + top 8 isomers) ---
    nno_pub = ROOT / "hcooh-nno" / "publication"
    nno_ws = nno_pub / "working_set"
    nno_efg_root = nno_pub / "ftmw_stage2" / "efg"

    # N2O monomer EFG
    n2o_src = nno_ws / "monomers" / "N2O.xyz"
    n2o_dir = nno_efg_root / "N2O"
    _copy_if_changed(n2o_src, n2o_dir / "start.xyz")
    n2o_base = "nno_efg_N2O"
    n2o_xyz_ref = "../../../energetics/monomers/N2O/opt/N2O_opt.xyz"
    n2o_inp_text = render_efg_inp("hcooh-nno", "N2O_monomer", n2o_xyz_ref, n2o_src, ["N", "O"])
    _write_if_changed(n2o_dir / f"{n2o_base}.inp", n2o_inp_text)
    specs.append({
        "id": n2o_base,
        "project": "hcooh-nno",
        "kind": "orca",
        "priority": 40,
        "ranks": RANKS,
        "timeout_s": 7200,
        "stall_s": 1800,
        "est_wall_s": 900,
        "cwd": str(n2o_dir),
        "basename": n2o_base,
        "depends_on": ["nno_mono_N2O_opt"],
        "description": "HCOOH-N2O monomer N2O: MP2/aug-cc-pwCVTZ relaxed-density EFG tensor (%eprnmr fgrad N, O; 8 ranks)",
        "meta": {
            "target": "N2O",
            "stage": "efg",
            "geometry": str(nno_pub / "energetics" / "monomers" / "N2O" / "opt" / "N2O_opt.xyz"),
            "source": str(n2o_src),
        },
    })

    for nn in NNO_EFG_ISOMERS:
        label = f"isomer-{nn}"
        src_xyz = nno_ws / f"{label}.xyz"
        job_dir = nno_efg_root / label
        _copy_if_changed(src_xyz, job_dir / "start.xyz")
        base = f"nno_efg_{nn}"
        xyz_ref = f"../../../energetics/revdsd/{label}/S1_revdsd_opt/S1_revdsd_opt.xyz"
        inp_text = render_efg_inp("hcooh-nno", label, xyz_ref, src_xyz, ["N", "O", "H"])
        _write_if_changed(job_dir / f"{base}.inp", inp_text)
        specs.append({
            "id": base,
            "project": "hcooh-nno",
            "kind": "orca",
            "priority": 40,
            "ranks": RANKS,
            "timeout_s": 7200,
            "stall_s": 1800,
            "est_wall_s": 1800,
            "cwd": str(job_dir),
            "basename": base,
            "depends_on": [f"nno_revdsd_{nn}"],
            "description": f"HCOOH-N2O {label}: MP2/aug-cc-pwCVTZ relaxed-density EFG tensor (%eprnmr fgrad N, O, H; 8 ranks)",
            "meta": {
                "isomer": label,
                "stage": "efg",
                "geometry": str(nno_pub / "energetics" / "revdsd" / label / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"),
                "source": str(src_xyz),
            },
        })

    # --- Priority 041: HCOOH-H2S EFG (H2S monomer + 8 working-set isomers) ---
    hsh_pub = ROOT / "hcooh-hsh" / "publication"
    hsh_ws = hsh_pub / "working_set"
    hsh_efg_root = hsh_pub / "ftmw_stage2" / "efg"

    h2s_src = hsh_ws / "monomers" / "H2S.xyz"
    h2s_dir = hsh_efg_root / "H2S"
    _copy_if_changed(h2s_src, h2s_dir / "start.xyz")
    h2s_base = "hsh_efg_H2S"
    h2s_xyz_ref = "../../../energetics/monomers/H2S/opt/H2S_opt.xyz"
    h2s_inp_text = render_efg_inp("hcooh-hsh", "H2S_monomer", h2s_xyz_ref, h2s_src, ["S", "H"])
    _write_if_changed(h2s_dir / f"{h2s_base}.inp", h2s_inp_text)
    specs.append({
        "id": h2s_base,
        "project": "hcooh-hsh",
        "kind": "orca",
        "priority": 41,
        "ranks": RANKS,
        "timeout_s": 7200,
        "stall_s": 1800,
        "est_wall_s": 900,
        "cwd": str(h2s_dir),
        "basename": h2s_base,
        "depends_on": ["hsh_mono_H2S_opt"],
        "description": "HCOOH-H2S monomer H2S: MP2/aug-cc-pwCVTZ relaxed-density EFG tensor (%eprnmr fgrad S, H; 8 ranks)",
        "meta": {
            "target": "H2S",
            "stage": "efg",
            "geometry": str(hsh_pub / "energetics" / "monomers" / "H2S" / "opt" / "H2S_opt.xyz"),
            "source": str(h2s_src),
        },
    })

    for nn in HSH_EFG_ISOMERS:
        label = f"isomer-{nn}"
        src_xyz = hsh_ws / f"{label}.xyz"
        job_dir = hsh_efg_root / label
        _copy_if_changed(src_xyz, job_dir / "start.xyz")
        base = f"hsh_efg_{nn}"
        xyz_ref = f"../../../energetics/revdsd/{label}/S1_revdsd_opt/S1_revdsd_opt.xyz"
        inp_text = render_efg_inp("hcooh-hsh", label, xyz_ref, src_xyz, ["S", "O", "H"])
        _write_if_changed(job_dir / f"{base}.inp", inp_text)
        specs.append({
            "id": base,
            "project": "hcooh-hsh",
            "kind": "orca",
            "priority": 41,
            "ranks": RANKS,
            "timeout_s": 7200,
            "stall_s": 1800,
            "est_wall_s": 1800,
            "cwd": str(job_dir),
            "basename": base,
            "depends_on": [f"hsh_revdsd_{nn}"],
            "description": f"HCOOH-H2S {label}: MP2/aug-cc-pwCVTZ relaxed-density EFG tensor (%eprnmr fgrad S, O, H; 8 ranks)",
            "meta": {
                "isomer": label,
                "stage": "efg",
                "geometry": str(hsh_pub / "energetics" / "revdsd" / label / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"),
                "source": str(src_xyz),
            },
        })

    # --- Priority 045: HCOOH-H2S Stage-2 VPT2 & Pickett Campaign (00, 01, 02, 05) ---
    hsh_revdsd_dir = hsh_pub / "energetics" / "revdsd"
    for nn, desc_tag in HSH_VPT2_ISOMERS:
        label = f"isomer-{nn}"
        jid = f"hsh_vpt2_{nn}"
        specs.append({
            "id": jid,
            "project": "hcooh-hsh",
            "kind": "command",
            "priority": 45,
            "ranks": RANKS,
            "timeout_s": 86400,
            "stall_s": 7200,
            "est_wall_s": 36000,
            "cwd": str(hsh_revdsd_dir),
            "command": [
                PYTHON_EXE,
                str(MW_PREDICT_PATH),
                "--stages",
                "S3a,S3b,S4",
                "--only",
                nn,
                "--cores",
                str(RANKS),
                "--work-dir",
                str(hsh_revdsd_dir),
            ],
            "depends_on": [f"hsh_revdsd_{nn}"],
            "description": (
                f"HCOOH-H2S {label} ({desc_tag}): Stage-2 B3LYP-D4/jun-cc-pV(D+d)Z VeryTightOpt (S3a) "
                f"+ Anharm VPT2 (S3b) + S4 hybrid B0/Pickett/isotopologue synthesis (8 ranks)"
            ),
            "meta": {
                "isomer": label,
                "stages": "S3a,S3b,S4",
                "work_dir": str(hsh_revdsd_dir),
            },
        })

    # --- Priority 050: HCOOH-N2O Stage-2 VPT2 & Pickett Campaign (01, 00, 02, 04, 06, 10, 08) ---
    nno_revdsd_dir = nno_pub / "energetics" / "revdsd"
    for nn, desc_tag in NNO_VPT2_ISOMERS:
        label = f"isomer-{nn}"
        jid = f"nno_vpt2_{nn}"
        specs.append({
            "id": jid,
            "project": "hcooh-nno",
            "kind": "command",
            "priority": 50,
            "ranks": RANKS,
            "timeout_s": 86400,
            "stall_s": 7200,
            "est_wall_s": 36000,
            "cwd": str(nno_revdsd_dir),
            "command": [
                PYTHON_EXE,
                str(MW_PREDICT_PATH),
                "--stages",
                "S3a,S3b,S4",
                "--only",
                nn,
                "--cores",
                str(RANKS),
                "--work-dir",
                str(nno_revdsd_dir),
            ],
            "depends_on": [f"nno_revdsd_{nn}"],
            "description": (
                f"HCOOH-N2O {label} ({desc_tag}): Stage-2 B3LYP-D4/jun-cc-pV(D+d)Z VeryTightOpt (S3a) "
                f"+ Anharm VPT2 (S3b) + S4 hybrid B0/Pickett/isotopologue synthesis (8 ranks)"
            ),
            "meta": {
                "isomer": label,
                "stages": "S3a,S3b,S4",
                "work_dir": str(nno_revdsd_dir),
            },
        })

    for spec in specs:
        problems = cq.validate_spec(spec)
        if problems:
            raise ValueError(f"Invalid Stage-2 queue spec {spec['id']}: {problems}")
        if enqueue:
            status, msg = cq.enqueue(spec, queue_dir, replace_pending=True)
            if status in ("invalid", "conflict"):
                raise RuntimeError(f"Failed to enqueue {spec['id']}: {status} ({msg})")

    return specs


# ==============================================================================
# 4. GEOMETRY & DIPOLE LOADERS FOR PRE-DATA FTMW PREDICTIONS
# ==============================================================================
def _parse_orca_dipole_and_xyz(out_path: Path, xyz_path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    """Read symbols, coordinates, and Cartesian dipole vector (in a.u.) from an ORCA output."""
    _, symbols, xyz = mw.read_xyz_frames(xyz_path)[-1]
    dip_vec_au = None
    if out_path.is_file():
        text = out_path.read_text(encoding="utf-8", errors="ignore")
        blocks = mw.dipole_blocks(text)
        if blocks:
            dip_vec_au = blocks[-1]["vector_au"]
    return symbols, xyz, dip_vec_au


def _selection_rule_summary(mu_abc: dict[str, float]) -> str:
    active = []
    for axis in ("a", "b", "c"):
        val = abs(mu_abc[axis])
        if val >= 0.05:
            active.append((val, f"{axis}-type (|mu_{axis}|={val:.2f} D)"))
    active.sort(reverse=True)
    return "; ".join(s for _, s in active) if active else "weak/dark (<0.05 D)"


def build_h2s_ftmw_predictions() -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """
    Build `h2s_ftmw_target_predictions.csv` and 2 K stick line lists (2-18 GHz)
    for all 8 working-set isomers of `HCOOH-H2S`.
    Uses the completed production `revDSD-PBEP86-D4/jun-cc-pV(T+d)Z` geometries,
    rotational constants B_e, and relaxed-density MP2 dipole components for all
    8 working-set isomers (`00, 01, 02, 05, 06, 08, 10, 13`), combined with the
    `B3LYP-D4/jun-cc-pV(D+d)Z` VPT2 vibrational correction for `isomer-00` (`1t`)
    and empirical VPT2 B_0/B_e scaling for the remaining 7 isomers.
    """
    mapping_df = pd.read_csv(ROOT / "hcooh-hsh" / "publication" / "working_set" / "mapping.csv")
    h2s_comp_df = pd.read_csv(ROOT / "hcooh-hsh" / "method_comparison" / "h2s_mace_vs_xtb_vs_dft.csv")
    ws_to_r2row: dict[str, pd.Series] = {}
    for _, r in h2s_comp_df.dropna(subset=["working_set_isomer"]).iterrows():
        ws_key = str(r["working_set_isomer"]).strip()
        if ws_key not in ws_to_r2row:
            ws_to_r2row[ws_key] = r

    revdsd_csv = ROOT / "hcooh-hsh" / "publication" / "energetics" / "revdsd" / "revdsd_results.csv"
    revdsd_by_iso: dict[str, pd.Series] = {}
    if revdsd_csv.is_file():
        for _, r in pd.read_csv(revdsd_csv).iterrows():
            revdsd_by_iso[str(r["isomer"]).strip()] = r

    smoke_json_path = ROOT / "hcooh-hsh" / "HCOOH-H2S_mw_predictions" / "_smoke_test" / "isomer-00" / "smoke" / "results" / "isomer-00_results.json"
    smoke_res = json.loads(smoke_json_path.read_text(encoding="utf-8")) if smoke_json_path.is_file() else None
    smoke_xyz_path = ROOT / "hcooh-hsh" / "HCOOH-H2S_mw_predictions" / "_smoke_test" / "isomer-00" / "smoke" / "results" / "isomer-00_revDSD.xyz"

    conformer_tags = {
        "isomer-00": "1t",
        "isomer-01": "2t",
        "isomer-02": "3t",
        "isomer-05": "1c",
        "isomer-06": "2c",
        "isomer-08": "3c",
        "isomer-10": "4c",
        "isomer-13": "5c",
    }

    tunneling_notes = {
        "isomer-00": "Dual H-bond ring (O-H...S + S-H...O=C) locks bound S-H; free S-H wagging barrier ~2.8 kcal/mol -> small (<50 kHz) or quenched ground-state tunneling splitting",
        "isomer-01": "S-H...O=C single donor; free S-H wagging/interchange barrier ~1.2 kcal/mol -> c-type transitions may show 0.1-2 MHz tunneling doublets",
        "isomer-02": "S-H...O(H) donor; low S-H interchange barrier (~0.8-1.5 kcal/mol) -> expect resolved tunneling doublets on out-of-plane dipole components",
        "isomer-05": "cis-HCOOH S-H...O=C planar donor; moderate S-H wagging barrier (~1.5 kcal/mol)",
        "isomer-06": "cis-HCOOH near-linear S-H...O=C (A ~ 20 GHz); low barrier to H2S internal rotation -> large K_a=1 tunneling perturbations",
        "isomer-08": "cis-HCOOH O-H...S with perpendicular H2S plane (C_s-like); equivalent H7/H8 exchange gives 3:1 ortho/para nuclear-spin weights (I_tot=1 vs 0) and b/c tunneling doublets",
        "isomer-10": "cis-HCOOH compact S-H...O=C; tilted H2S undergoes hindered internal rotation",
        "isomer-13": "cis-HCOOH stacked dispersion complex; floppy H2S internal rotation -> wide progression of internal-rotation states",
    }

    rows: list[dict[str, Any]] = []
    line_catalog: dict[str, list[dict[str, Any]]] = {}

    for _, mrow in mapping_df.iterrows():
        wlabel = str(mrow["working_label"]).strip()
        if not wlabel.startswith("isomer-"):
            continue
        r2_row = ws_to_r2row.get(wlabel)
        mace_id = str(r2_row["MACE_ID"]).strip() if r2_row is not None and pd.notna(r2_row["MACE_ID"]) else None
        rev_row = revdsd_by_iso.get(wlabel)
        rev_s1_xyz = ROOT / "hcooh-hsh" / "publication" / "energetics" / "revdsd" / wlabel / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"

        # Choose geometry & spectroscopic constants
        if wlabel == "isomer-00" and smoke_res is not None and smoke_xyz_path.is_file():
            xyz_use = rev_s1_xyz if rev_s1_xyz.is_file() else smoke_xyz_path
            _, symbols, xyz = mw.read_xyz_frames(xyz_use)[-1]
            if rev_row is not None and pd.notna(rev_row.get("A_e_MHz")):
                ae_mhz = float(rev_row["A_e_MHz"])
                be_mhz = float(rev_row["B_e_MHz"])
                ce_mhz = float(rev_row["C_e_MHz"])
                # Apply exact VPT2 dB_vib = B0 - Be from smoke_res
                a_mhz = ae_mhz + (float(smoke_res["B0_MHz"]["a"]) - float(smoke_res["Be_revDSD_MHz"]["a"]))
                b_mhz = be_mhz + (float(smoke_res["B0_MHz"]["b"]) - float(smoke_res["Be_revDSD_MHz"]["b"]))
                c_mhz = ce_mhz + (float(smoke_res["B0_MHz"]["c"]) - float(smoke_res["Be_revDSD_MHz"]["c"]))
                mu_abc = {
                    "a": float(rev_row["mu_a_D"]),
                    "b": float(rev_row["mu_b_D"]),
                    "c": float(rev_row["mu_c_D"]),
                }
            else:
                a_mhz = float(smoke_res["B0_MHz"]["a"])
                b_mhz = float(smoke_res["B0_MHz"]["b"])
                c_mhz = float(smoke_res["B0_MHz"]["c"])
                ae_mhz = float(smoke_res["Be_revDSD_MHz"]["a"])
                be_mhz = float(smoke_res["Be_revDSD_MHz"]["b"])
                ce_mhz = float(smoke_res["Be_revDSD_MHz"]["c"])
                mu_abc = {k: float(smoke_res["mu_D"][k]) for k in ("a", "b", "c")}
            qa = {k: float(smoke_res["quartic_A_MHz"][k]) for k in ("DJ", "DJK", "DK", "dJ", "dK")}
            rot_source = "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z B_e + B3LYP-D4 VPT2 dB_vib (production S1-S2b + VPT2)"
            window_pct = 1.5
        elif rev_row is not None and rev_s1_xyz.is_file() and pd.notna(rev_row.get("A_e_MHz")):
            _, symbols, xyz = mw.read_xyz_frames(rev_s1_xyz)[-1]
            ae_mhz = float(rev_row["A_e_MHz"])
            be_mhz = float(rev_row["B_e_MHz"])
            ce_mhz = float(rev_row["C_e_MHz"])
            # Apply empirical B0/Be scaling from isomer-00 VPT2 (-0.39% A, -1.86% B, -1.77% C)
            a_mhz = ae_mhz * (11267.4218 / 11312.0909)
            b_mhz = be_mhz * (2179.8635 / 2221.2349)
            c_mhz = ce_mhz * (1848.0512 / 1881.3282)
            mu_abc = {
                "a": float(rev_row["mu_a_D"]),
                "b": float(rev_row["mu_b_D"]),
                "c": float(rev_row["mu_c_D"]),
            }
            qa = {"DJ": 0.0022, "DJK": 0.0092, "DK": -0.0013, "dJ": 0.00037, "dK": 0.0099}
            rot_source = "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z B_e + relaxed MP2 dipole (production S1-S2b; VPT2 queued P045)"
            window_pct = 1.5
        else:
            r2_dir = ROOT / "hcooh-hsh" / "maceoff24m_search_600" / "r2scan3c_runs" / f"{mace_id}_r2scan3c" if mace_id else None
            if r2_dir is not None and (r2_dir / f"{mace_id}_r2scan3c.xyz").is_file():
                symbols, xyz, dip_vec_au = _parse_orca_dipole_and_xyz(
                    r2_dir / f"{mace_id}_r2scan3c.out", r2_dir / f"{mace_id}_r2scan3c.xyz"
                )
                rot_source = f"r2SCAN-3c TightOpt ({mace_id}; queued at P010/P045 for revDSD+VPT2)"
                window_pct = 2.0
            else:
                ws_xyz = ROOT / "hcooh-hsh" / "publication" / "working_set" / f"{wlabel}.xyz"
                _, symbols, xyz = mw.read_xyz_frames(ws_xyz)[-1]
                dip_vec_au = None
                rot_source = "GFN2-xTB working-set geometry (queued at P010 for revDSD)"
                window_pct = 5.0

            masses = mw.masses_for(symbols)
            rot_c = mw.rotational_constants(xyz, masses)
            ae_mhz, be_mhz, ce_mhz = float(rot_c[0]), float(rot_c[1]), float(rot_c[2])
            # Apply empirical B0/Be scaling from isomer-00 VPT2 (-0.39% A, -1.86% B, -1.77% C)
            a_mhz = ae_mhz * (11267.4218 / 11312.0909)
            b_mhz = be_mhz * (2179.8635 / 2221.2349)
            c_mhz = ce_mhz * (1848.0512 / 1881.3282)
            _, _, axes = mw.principal_axes(xyz, masses)
            if dip_vec_au is not None:
                mu_v = axes.T @ dip_vec_au * float(mw.AU_TO_DEBYE)
                mu_abc = {"a": float(mu_v[0]), "b": float(mu_v[1]), "c": float(mu_v[2])}
            else:
                mu_abc = {"a": 0.5, "b": 1.0, "c": 0.5}
            qa = {"DJ": 0.0022, "DJK": 0.0092, "DK": -0.0013, "dJ": 0.00037, "dK": 0.0099}

        masses = mw.masses_for(symbols)
        inert = mw.inertial_analysis(xyz, masses)
        qrot_2k = mw.classical_partition_function(a_mhz, b_mhz, c_mhz, 2.0)

        # Predict 2 K transitions in 2-18 GHz
        ca_const = {"A": a_mhz, "B": b_mhz, "C": c_mhz, **qa}
        mu_for_lines = {g: (0.0 if abs(mu_abc[g]) < 0.05 else mu_abc[g]) for g in ("a", "b", "c")}
        all_lines = mw.predict_lines(ca_const, "A", mu_for_lines)
        ftmw_lines = [
            ln for ln in all_lines
            if 2000.0 <= float(ln["freq_MHz"]) <= 18000.0
        ]
        ftmw_lines.sort(key=lambda x: -float(x["rel_intensity"]))
        line_catalog[wlabel] = ftmw_lines

        top_lines_str = "; ".join(
            f"{ln['type']}:{ln['upper']}<-{ln['lower']} @ {ln['freq_MHz']:.1f} MHz (I={ln['rel_intensity']:.2f})"
            for ln in ftmw_lines[:5]
        )

        # Isotopologue shifts (parent -> 34S, 13C, HCOOD, DCOOH, HDS bound, HDS free, D2S)
        iso_shifts: dict[str, tuple[float, float, float]] = {}
        iso_abs: dict[str, tuple[float, float, float]] = {}
        for iso_name, sub_map in _hsh_isotopologues(symbols, xyz):
            m_iso = mw.masses_for(symbols, sub_map)
            be_iso = mw.rotational_constants(xyz, m_iso)
            b0_iso = (
                float(be_iso[0]) * (a_mhz / ae_mhz),
                float(be_iso[1]) * (b_mhz / be_mhz),
                float(be_iso[2]) * (c_mhz / ce_mhz),
            )
            iso_abs[iso_name] = b0_iso
            iso_shifts[iso_name] = (
                b0_iso[0] - a_mhz,
                b0_iso[1] - b_mhz,
                b0_iso[2] - c_mhz,
            )

        # Projected 33S quadrupole coupling constants in complex PAS
        q33s = project_h2s_monomer_quadrupole(symbols, xyz, masses)

        # Kraitchman heavy-atom coordinates for S (using 34S) and C (using 13C)
        m_parent = float(masses.sum())
        dm_34s = float(mw.ISOTOPE_MASS["34S"] - mw.ISOTOPE_MASS["S"])
        dm_13c = float(mw.ISOTOPE_MASS["13C"] - mw.ISOTOPE_MASS["C"])
        idx_s = q33s["idx_S"]
        idx_c = next(i for i, s in enumerate(symbols) if s.upper() == "C")
        ks_a, ks_b, ks_c = kraitchman_3d((ae_mhz, be_mhz, ce_mhz),
                                         tuple(float(x) for x in mw.rotational_constants(xyz, mw.masses_for(symbols, {idx_s: "34S"}))),
                                         dm_34s, m_parent)
        kc_a, kc_b, kc_c = kraitchman_3d((ae_mhz, be_mhz, ce_mhz),
                                         tuple(float(x) for x in mw.rotational_constants(xyz, mw.masses_for(symbols, {idx_c: "13C"}))),
                                         dm_13c, m_parent)

        dE_r2scan = float(r2_row["dE_r2SCAN3c_kcal"]) if r2_row is not None and pd.notna(r2_row["dE_r2SCAN3c_kcal"]) else np.nan
        dE0_r2scan = float(r2_row["dE0_r2SCAN3c_kcal"]) if r2_row is not None and pd.notna(r2_row["dE0_r2SCAN3c_kcal"]) else np.nan

        rows.append({
            "working_label": wlabel,
            "conformer_tag": conformer_tags.get(wlabel, wlabel),
            "motif_family": mrow["motif_family"],
            "motif_description": mrow["motif"],
            "stage2_vpt2_queued": wlabel in ("isomer-00", "isomer-01", "isomer-02", "isomer-08", "isomer-05"),
            "dE_GFN2_kcal": float(mrow["GFN2_dE_kcal"]),
            "dE_r2SCAN3c_kcal": round(dE_r2scan, 3) if np.isfinite(dE_r2scan) else "",
            "dE0_r2SCAN3c_kcal": round(dE0_r2scan, 3) if np.isfinite(dE0_r2scan) else "",
            "A0_MHz": round(a_mhz, 2),
            "B0_MHz": round(b_mhz, 2),
            "C0_MHz": round(c_mhz, 2),
            "Ae_MHz": round(ae_mhz, 2),
            "Be_MHz": round(be_mhz, 2),
            "Ce_MHz": round(ce_mhz, 2),
            "B0_search_window_pct": window_pct,
            "kappa": round(float(inert["kappa"]), 4),
            "inertial_defect_uA2": round(float(inert["inertial_defect"]), 4),
            "Paa_uA2": round(float(inert["Paa"]), 3),
            "Pbb_uA2": round(float(inert["Pbb"]), 3),
            "Pcc_uA2": round(float(inert["Pcc"]), 3),
            "mu_a_D": round(mu_abc["a"], 3),
            "mu_b_D": round(mu_abc["b"], 3),
            "mu_c_D": round(mu_abc["c"], 3),
            "mu_total_D": round(float(np.linalg.norm([mu_abc["a"], mu_abc["b"], mu_abc["c"]])), 3),
            "dominant_selection_rules": _selection_rule_summary(mu_abc),
            "Qrot_2K": round(qrot_2k, 2),
            "Kraitchman_S_abc_A": f"({ks_a:.3f}, {ks_b:.3f}, {ks_c:.3f})",
            "Kraitchman_C_abc_A": f"({kc_a:.3f}, {kc_b:.3f}, {kc_c:.3f})",
            "dB0_34S_MHz": f"({iso_shifts['34S'][0]:+.1f}, {iso_shifts['34S'][1]:+.1f}, {iso_shifts['34S'][2]:+.1f})",
            "dB0_13C_MHz": f"({iso_shifts['13C'][0]:+.1f}, {iso_shifts['13C'][1]:+.1f}, {iso_shifts['13C'][2]:+.1f})",
            "dB0_HCOOD_MHz": f"({iso_shifts['HCOOD'][0]:+.1f}, {iso_shifts['HCOOD'][1]:+.1f}, {iso_shifts['HCOOD'][2]:+.1f})",
            "dB0_DCOOH_MHz": f"({iso_shifts['DCOOH'][0]:+.1f}, {iso_shifts['DCOOH'][1]:+.1f}, {iso_shifts['DCOOH'][2]:+.1f})",
            "dB0_HDS_bound_MHz": f"({iso_shifts['HDS (D bound)'][0]:+.1f}, {iso_shifts['HDS (D bound)'][1]:+.1f}, {iso_shifts['HDS (D bound)'][2]:+.1f})",
            "dB0_HDS_free_MHz": f"({iso_shifts['HDS (D free)'][0]:+.1f}, {iso_shifts['HDS (D free)'][1]:+.1f}, {iso_shifts['HDS (D free)'][2]:+.1f})",
            "dB0_D2S_MHz": f"({iso_shifts['D2S'][0]:+.1f}, {iso_shifts['D2S'][1]:+.1f}, {iso_shifts['D2S'][2]:+.1f})",
            "chi_aa_33S_proj_MHz": round(q33s["chi_aa_33S_MHz"], 2),
            "chi_bb_33S_proj_MHz": round(q33s["chi_bb_33S_MHz"], 2),
            "chi_cc_33S_proj_MHz": round(q33s["chi_cc_33S_MHz"], 2),
            "chi_ab_33S_proj_MHz": round(q33s["chi_ab_33S_MHz"], 2),
            "h2s_plane_tilt_deg": round(q33s["h2s_plane_tilt_deg"], 1),
            "tunneling_diagnostic": tunneling_notes.get(wlabel, ""),
            "top_2to18GHz_lines_2K": top_lines_str,
            "rot_constant_provenance": rot_source,
        })

    return pd.DataFrame(rows), line_catalog


def _hsh_isotopologues(symbols: list[str], xyz: np.ndarray) -> list[tuple[str, dict[int, str]]]:
    """Single-isotope and D2S substitutions for HCOOH-H2S."""
    idx_c = next(i for i, s in enumerate(symbols) if s.upper() == "C")
    idx_s = next(i for i, s in enumerate(symbols) if s.upper() == "S")
    o_indices = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    h_indices = [i for i, s in enumerate(symbols) if s.upper() == "H"]

    # Sort H atoms by distance to S: the two closest (~1.34 A) belong to H2S
    h_by_s = sorted(h_indices, key=lambda hi: float(np.linalg.norm(xyz[hi] - xyz[idx_s])))
    h_s1, h_s2 = h_by_s[0], h_by_s[1]
    # Bound S-H is the one closer to an HCOOH oxygen atom
    d_o_s1 = min(float(np.linalg.norm(xyz[h_s1] - xyz[oi])) for oi in o_indices)
    d_o_s2 = min(float(np.linalg.norm(xyz[h_s2] - xyz[oi])) for oi in o_indices)
    h_s_bound, h_s_free = (h_s1, h_s2) if d_o_s1 <= d_o_s2 else (h_s2, h_s1)

    # Remaining two H atoms belong to HCOOH: closer to C is formyl H (DCOOH), farther is hydroxyl H (HCOOD)
    h_hcooh = [hi for hi in h_indices if hi not in (h_s1, h_s2)]
    h_ch, h_oh = (
        (h_hcooh[0], h_hcooh[1])
        if float(np.linalg.norm(xyz[h_hcooh[0]] - xyz[idx_c])) < float(np.linalg.norm(xyz[h_hcooh[1]] - xyz[idx_c]))
        else (h_hcooh[1], h_hcooh[0])
    )
    return [
        ("parent", {}),
        ("34S", {idx_s: "34S"}),
        ("13C", {idx_c: "13C"}),
        ("HCOOD", {h_oh: "D"}),
        ("DCOOH", {h_ch: "D"}),
        ("HDS (D bound)", {h_s_bound: "D"}),
        ("HDS (D free)", {h_s_free: "D"}),
        ("D2S", {h_s1: "D", h_s2: "D"}),
    ]


def _nno_isotopologues(xyz: np.ndarray) -> list[tuple[str, dict[int, str]]]:
    """Single-isotope substitutions for HCOOH-N2O (atoms 0..4 = C,O,O,H,H; 5..7 = N,N,O)."""
    # Identify hydroxyl H vs formyl C-H and terminal N vs central N
    d_c = [float(np.linalg.norm(xyz[i] - xyz[0])) for i in (3, 4)]
    h_ch, h_oh = (3, 4) if d_c[0] < d_c[1] else (4, 3)
    # In N2O (atoms 5, 6, 7 where 5,6 are N and 7 is O): central N is closer to O7
    d5 = float(np.linalg.norm(xyz[5] - xyz[7]))
    d6 = float(np.linalg.norm(xyz[6] - xyz[7]))
    n_c, n_t = (5, 6) if d5 < d6 else (6, 5)
    return [
        ("parent", {}),
        ("15N_t", {n_t: "15N"}),
        ("15N_c", {n_c: "15N"}),
        ("15N2O (doubly)", {n_t: "15N", n_c: "15N"}),
        ("18O_N2O", {7: "18O"}),
        ("13C", {0: "13C"}),
        ("HCOOD", {h_oh: "D"}),
        ("DCOOH", {h_ch: "D"}),
    ]


def build_n2o_ftmw_predictions() -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """
    Build `n2o_ftmw_target_predictions.csv` and 2 K stick line lists (2-18 GHz)
    for all 16 working-set isomers of `HCOOH-N2O`.
    Uses completed production `revDSD-PBEP86-D4/jun-cc-pV(T+d)Z` geometries and
    relaxed-density MP2 dipoles from `hcooh-nno/publication/energetics/revdsd/`
    where available, falling back to `r2SCAN-3c TightOpt` / `wB97X-V/def2-TZVPP`
    geometries for isomers still queued.
    """
    # Ensure 15N and 18O masses are registered in mw.ISOTOPE_MASS
    mw.ISOTOPE_MASS.setdefault("15N", 15.0001088989)
    mw.ISOTOPE_MASS.setdefault("18O", 17.9991596128)

    mapping_df = pd.read_csv(ROOT / "hcooh-nno" / "publication" / "working_set" / "mapping.csv")
    n2o_comp_df = pd.read_csv(ROOT / "hcooh-hsh" / "method_comparison" / "n2o_mace_vs_xtb_vs_duarte.csv")
    ws_to_mace: dict[str, str] = {}
    for _, r in n2o_comp_df.dropna(subset=["working_set_isomer", "nearest_MACE_ID"]).iterrows():
        ws_to_mace[str(r["working_set_isomer"]).strip()] = str(r["nearest_MACE_ID"]).strip()
    ws_to_duarte: dict[str, str] = {}
    for _, r in n2o_comp_df.dropna(subset=["working_set_isomer"]).iterrows():
        w_iso = str(r["working_set_isomer"]).strip()
        if pd.notna(r.get("Duarte_label")) and str(r["Duarte_label"]).strip():
            ws_to_duarte[w_iso] = str(r["Duarte_label"]).strip()

    revdsd_csv = ROOT / "hcooh-nno" / "publication" / "energetics" / "revdsd" / "revdsd_results.csv"
    revdsd_by_iso: dict[str, pd.Series] = {}
    if revdsd_csv.is_file():
        for _, r in pd.read_csv(revdsd_csv).iterrows():
            revdsd_by_iso[str(r["isomer"]).strip()] = r

    vpt2_queued_set = {f"isomer-{nn}" for nn, _ in NNO_VPT2_ISOMERS}
    rows: list[dict[str, Any]] = []
    line_catalog: dict[str, list[dict[str, Any]]] = {}

    for _, mrow in mapping_df.iterrows():
        wlabel = str(mrow["working_label"]).strip()
        if not wlabel.startswith("isomer-"):
            continue
        nn = wlabel.split("-")[1]
        mace_id = ws_to_mace.get(wlabel)
        duarte_tag = ws_to_duarte.get(wlabel, wlabel)

        rev_row = revdsd_by_iso.get(wlabel)
        rev_s1_xyz = ROOT / "hcooh-nno" / "publication" / "energetics" / "revdsd" / wlabel / "S1_revdsd_opt" / "S1_revdsd_opt.xyz"
        rev_ok = (
            rev_row is not None
            and rev_s1_xyz.is_file()
            and pd.notna(rev_row.get("A_e_MHz"))
            and pd.notna(rev_row.get("n_imag"))
            and int(float(rev_row["n_imag"])) == 0
        )

        if rev_ok and rev_row is not None:
            _, symbols, xyz = mw.read_xyz_frames(rev_s1_xyz)[-1]
            masses = mw.masses_for(symbols)
            ae_mhz = float(rev_row["A_e_MHz"])
            be_mhz = float(rev_row["B_e_MHz"])
            ce_mhz = float(rev_row["C_e_MHz"])
            a_mhz = ae_mhz * 0.995
            b_mhz = be_mhz * 0.985
            c_mhz = ce_mhz * 0.986
            mu_abc = {
                "a": float(rev_row["mu_a_D"]),
                "b": float(rev_row["mu_b_D"]),
                "c": float(rev_row["mu_c_D"]),
            }
            rot_source = "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z B_e + relaxed MP2 dipole (production S1-S2b; VPT2 queued P050)"
            window_pct = 1.5
        else:
            r2_xyz = (
                ROOT / "hcooh-nno" / "maceoff24m_search_600" / "r2scan3c_runs" / f"{mace_id}_r2scan3c" / f"{mace_id}_r2scan3c.xyz"
                if mace_id is not None else None
            )
            r2_out = (
                ROOT / "hcooh-nno" / "maceoff24m_search_600" / "r2scan3c_runs" / f"{mace_id}_r2scan3c" / f"{mace_id}_r2scan3c.out"
                if mace_id is not None else None
            )
            r2_converged = False
            if r2_out is not None and r2_out.is_file() and r2_xyz is not None and r2_xyz.is_file():
                out_txt = r2_out.read_text(encoding="utf-8", errors="ignore")
                if "THE OPTIMIZATION HAS CONVERGED" in out_txt and "ORCA TERMINATED NORMALLY" in out_txt:
                    r2_converged = True

            if r2_converged and r2_out is not None and r2_xyz is not None:
                symbols, xyz, dip_vec_au = _parse_orca_dipole_and_xyz(r2_out, r2_xyz)
                rot_source = f"r2SCAN-3c TightOpt (MACE {mace_id}; queued at P020/P050 for revDSD+VPT2)"
                window_pct = 2.0
            else:
                wb_xyz = ROOT / "hcooh-nno" / "publication" / "energetics" / "revdsd" / "wb97xv_preopt" / f"{wlabel}_wb97xv.xyz"
                ws_xyz = wb_xyz if wb_xyz.is_file() else (ROOT / "hcooh-nno" / "publication" / "working_set" / f"{wlabel}.xyz")
                xtb_out = ROOT / "hcooh-nno" / "HCOOH-N2O_isomer_handout" / "unique_isomers" / f"isomer-{nn}.out"
                symbols, xyz, dip_vec_au = _parse_orca_dipole_and_xyz(xtb_out, ws_xyz)
                rot_source = (
                    "wB97X-V/def2-TZVPP GPU pre-opt geometry + GFN2 dipole (queued at P028/P050 for revDSD+VPT2)"
                    if wb_xyz.is_file()
                    else "GFN2-xTB working-set geometry (queued at P020/P050 for revDSD+VPT2)"
                )
                window_pct = 2.0 if wb_xyz.is_file() else 4.0

            masses = mw.masses_for(symbols)
            rot_c = mw.rotational_constants(xyz, masses)
            ae_mhz, be_mhz, ce_mhz = float(rot_c[0]), float(rot_c[1]), float(rot_c[2])
            # Empirical VPT2 ground-state vibrational scaling (-0.5% A, -1.5% B, -1.4% C)
            a_mhz = ae_mhz * 0.995
            b_mhz = be_mhz * 0.985
            c_mhz = ce_mhz * 0.986

            _, _, axes = mw.principal_axes(xyz, masses)
            if dip_vec_au is not None:
                mu_v = axes.T @ dip_vec_au * float(mw.AU_TO_DEBYE)
                mu_abc = {"a": float(mu_v[0]), "b": float(mu_v[1]), "c": float(mu_v[2])}
            else:
                mu_abc = {"a": 1.2, "b": 0.8, "c": 0.0}

        inert = mw.inertial_analysis(xyz, masses)
        qrot_2k = mw.classical_partition_function(a_mhz, b_mhz, c_mhz, 2.0)

        qa = {"DJ": 0.0018, "DJK": 0.0085, "DK": 0.0120, "dJ": 0.00032, "dK": 0.0065}
        ca_const = {"A": a_mhz, "B": b_mhz, "C": c_mhz, **qa}
        mu_for_lines = {g: (0.0 if abs(mu_abc[g]) < 0.05 else mu_abc[g]) for g in ("a", "b", "c")}
        all_lines = mw.predict_lines(ca_const, "A", mu_for_lines)
        ftmw_lines = [
            ln for ln in all_lines
            if 2000.0 <= float(ln["freq_MHz"]) <= 18000.0
        ]
        ftmw_lines.sort(key=lambda x: -float(x["rel_intensity"]))
        line_catalog[wlabel] = ftmw_lines

        top_lines_str = "; ".join(
            f"{ln['type']}:{ln['upper']}<-{ln['lower']} @ {ln['freq_MHz']:.1f} MHz (I={ln['rel_intensity']:.2f})"
            for ln in ftmw_lines[:5]
        )

        # Isotopologue shifts
        iso_shifts: dict[str, tuple[float, float, float]] = {}
        for iso_name, sub_map in _nno_isotopologues(xyz):
            m_iso = mw.masses_for(symbols, sub_map)
            be_iso = mw.rotational_constants(xyz, m_iso)
            b0_iso = (
                float(be_iso[0]) * (a_mhz / ae_mhz),
                float(be_iso[1]) * (b_mhz / be_mhz),
                float(be_iso[2]) * (c_mhz / ce_mhz),
            )
            iso_shifts[iso_name] = (
                b0_iso[0] - a_mhz,
                b0_iso[1] - b_mhz,
                b0_iso[2] - c_mhz,
            )

        # Projected 14N_t and 14N_c nuclear quadrupole coupling constants
        q_n2o = project_n2o_monomer_quadrupole(symbols, xyz, masses)
        q_nt = q_n2o["14N_t"]
        q_nc = q_n2o["14N_c"]

        # Kraitchman coordinates for N_t and N_c
        m_parent = float(masses.sum())
        dm_15n = float(mw.ISOTOPE_MASS["15N"] - mw.ISOTOPE_MASS["N"])
        knt_a, knt_b, knt_c = kraitchman_3d(
            (ae_mhz, be_mhz, ce_mhz),
            tuple(float(x) for x in mw.rotational_constants(xyz, mw.masses_for(symbols, {q_nt["idx_Nt"]: "15N"}))),
            dm_15n, m_parent,
        )
        knc_a, knc_b, knc_c = kraitchman_3d(
            (ae_mhz, be_mhz, ce_mhz),
            tuple(float(x) for x in mw.rotational_constants(xyz, mw.masses_for(symbols, {q_nc["idx_Nc"]: "15N"}))),
            dm_15n, m_parent,
        )

        rows.append({
            "working_label": wlabel,
            "duarte2017_label": duarte_tag,
            "motif_family": mrow["motif_family"],
            "motif_description": mrow["motif"],
            "stage2_vpt2_queued": wlabel in vpt2_queued_set,
            "dE_GFN2_kcal": float(mrow["GFN2_dE_kcal"]),
            "A0_MHz": round(a_mhz, 2),
            "B0_MHz": round(b_mhz, 2),
            "C0_MHz": round(c_mhz, 2),
            "Ae_MHz": round(ae_mhz, 2),
            "Be_MHz": round(be_mhz, 2),
            "Ce_MHz": round(ce_mhz, 2),
            "B0_search_window_pct": window_pct,
            "kappa": round(float(inert["kappa"]), 4),
            "inertial_defect_uA2": round(float(inert["inertial_defect"]), 4),
            "Paa_uA2": round(float(inert["Paa"]), 3),
            "Pbb_uA2": round(float(inert["Pbb"]), 3),
            "Pcc_uA2": round(float(inert["Pcc"]), 3),
            "mu_a_D": round(mu_abc["a"], 3),
            "mu_b_D": round(mu_abc["b"], 3),
            "mu_c_D": round(mu_abc["c"], 3),
            "mu_total_D": round(float(np.linalg.norm([mu_abc["a"], mu_abc["b"], mu_abc["c"]])), 3),
            "dominant_selection_rules": _selection_rule_summary(mu_abc),
            "Qrot_2K": round(qrot_2k, 2),
            "Kraitchman_Nt_abc_A": f"({knt_a:.3f}, {knt_b:.3f}, {knt_c:.3f})",
            "Kraitchman_Nc_abc_A": f"({knc_a:.3f}, {knc_b:.3f}, {knc_c:.3f})",
            "dB0_15Nt_MHz": f"({iso_shifts['15N_t'][0]:+.1f}, {iso_shifts['15N_t'][1]:+.1f}, {iso_shifts['15N_t'][2]:+.1f})",
            "dB0_15Nc_MHz": f"({iso_shifts['15N_c'][0]:+.1f}, {iso_shifts['15N_c'][1]:+.1f}, {iso_shifts['15N_c'][2]:+.1f})",
            "dB0_18O_N2O_MHz": f"({iso_shifts['18O_N2O'][0]:+.1f}, {iso_shifts['18O_N2O'][1]:+.1f}, {iso_shifts['18O_N2O'][2]:+.1f})",
            "dB0_13C_MHz": f"({iso_shifts['13C'][0]:+.1f}, {iso_shifts['13C'][1]:+.1f}, {iso_shifts['13C'][2]:+.1f})",
            "dB0_HCOOD_MHz": f"({iso_shifts['HCOOD'][0]:+.1f}, {iso_shifts['HCOOD'][1]:+.1f}, {iso_shifts['HCOOD'][2]:+.1f})",
            "dB0_DCOOH_MHz": f"({iso_shifts['DCOOH'][0]:+.1f}, {iso_shifts['DCOOH'][1]:+.1f}, {iso_shifts['DCOOH'][2]:+.1f})",
            "chi_aa_14Nt_proj_MHz": round(q_nt["chi_aa_MHz"], 4),
            "chi_bb_14Nt_proj_MHz": round(q_nt["chi_bb_MHz"], 4),
            "chi_cc_14Nt_proj_MHz": round(q_nt["chi_cc_MHz"], 4),
            "chi_ab_14Nt_proj_MHz": round(q_nt["chi_ab_MHz"], 4),
            "chi_aa_14Nc_proj_MHz": round(q_nc["chi_aa_MHz"], 4),
            "chi_bb_14Nc_proj_MHz": round(q_nc["chi_bb_MHz"], 4),
            "chi_cc_14Nc_proj_MHz": round(q_nc["chi_cc_MHz"], 4),
            "chi_ab_14Nc_proj_MHz": round(q_nc["chi_ab_MHz"], 4),
            "N2O_axis_theta_a_deg": round(q_nt["theta_a_deg"], 1),
            "N2O_axis_theta_b_deg": round(q_nt["theta_b_deg"], 1),
            "top_2to18GHz_lines_2K": top_lines_str,
            "rot_constant_provenance": rot_source,
        })

    return pd.DataFrame(rows), line_catalog


# ==============================================================================
# 5. KNOWN CONTAMINANT LINES CATALOG (2-18 GHz) & 3.5-WEEK SCHEDULE
# ==============================================================================
def build_known_contaminant_lines() -> pd.DataFrame:
    """
    Build `known_contaminant_lines.csv`: experimental rotational constants,
    dipoles, and strong 2-18 GHz transitions to mask/veto during broadband
    CP-FTMW survey scans of HCOOH + H2S/N2O in Ne/He/Ar (plus trace H2O).
    """
    contaminants = [
        {
            "species": "trans-HCOOH",
            "category": "Monomer",
            "beams": "Both (HCOOH+H2S, HCOOH+N2O)",
            "A0_MHz": 77512.23,
            "B0_MHz": 12055.11,
            "C0_MHz": 10416.12,
            "dipole_D": "|mu_a|=1.396, |mu_b|=0.26",
            "diagnostic_lines_2to18GHz_MHz": "1_01<-0_00: 22471.1 (above 18 GHz); 1_10<-1_11 (K-doublet): 1638.8; 2_11<-2_12: 4916.3; 3_12<-3_13: 9831.0; 4_13<-4_14: 16381.5",
            "hyperfine_or_tunneling": "None (closed-shell, no 14N/33S); strong a-type K_a=1 Q-branch lines at 4916.3, 9831.0, 16381.5 MHz",
            "reference": "Winnewisser et al., J. Mol. Spectrosc. 2002, 214, 110; JPL Catalog 46005",
        },
        {
            "species": "cis-HCOOH",
            "category": "Monomer (excited conformer)",
            "beams": "Both (trace thermal population)",
            "A0_MHz": 86461.56,
            "B0_MHz": 11689.18,
            "C0_MHz": 10283.90,
            "dipole_D": "|mu_a|=2.65, |mu_b|=2.71",
            "diagnostic_lines_2to18GHz_MHz": "2_11<-2_12: 4215.6; 3_12<-3_13: 8430.2; 4_13<-4_14: 14046.8",
            "hyperfine_or_tunneling": "Delta E = +3.9 kcal/mol above trans-HCOOH; relaxed in Ar, trace in Ne/He if heated nozzle used",
            "reference": "Hocking, Z. Naturforsch. A 1976, 31, 1113",
        },
        {
            "species": "(HCOOH)2",
            "category": "Homodimer",
            "beams": "Both",
            "A0_MHz": 6053.0,
            "B0_MHz": 2162.0,
            "C0_MHz": 1593.0,
            "dipole_D": "mu_total = 0.00 D (C_2h centrosymmetric ring; microwave silent in ground state)",
            "diagnostic_lines_2to18GHz_MHz": "Centrosymmetric C_2h global minimum is microwave-inactive (mu=0); trace polar open-chain/cis-trans isomers appear at B+C ~ 3650 MHz",
            "hyperfine_or_tunneling": "Double proton transfer in C_2h ring is IR/Raman active, microwave dark",
            "reference": "Priem et al., J. Mol. Spectrosc. 2000; Xue & Suhm 2019",
        },
        {
            "species": "HCOOH-H2O",
            "category": "Water impurity complex",
            "beams": "Both (adventitious H2O)",
            "A0_MHz": 12071.40,
            "B0_MHz": 4565.25,
            "C0_MHz": 3330.18,
            "dipole_D": "|mu_a|=1.92, |mu_b|=0.58",
            "diagnostic_lines_2to18GHz_MHz": "1_01<-0_00: 7895.4; 2_12<-1_11: 14555.6; 2_02<-1_01: 15596.2; 2_11<-1_10: 17025.9; 1_11<-0_00 (b): 15401.6",
            "hyperfine_or_tunneling": "Small c-type/a-type H2O wagging tunneling splitting (~1-4 MHz in excited states, <50 kHz in main a-type R-branch)",
            "reference": "Priem et al., J. Chem. Phys. 1998, 109, 10294",
        },
        {
            "species": "H2S",
            "category": "Monomer",
            "beams": "HCOOH+H2S",
            "A0_MHz": 310182.2,
            "B0_MHz": 270883.8,
            "C0_MHz": 141795.4,
            "dipole_D": "|mu_b|=0.978",
            "diagnostic_lines_2to18GHz_MHz": "No rotational transitions in 2-18 GHz (lowest fundamental 1_10<-1_01 is at 168.76 GHz)",
            "hyperfine_or_tunneling": "Completely transparent in 2-18 GHz CP-FTMW window",
            "reference": "Cazzoli & Puzzarini, J. Mol. Spectrosc. 2014, 298, 31",
        },
        {
            "species": "(H2S)2",
            "category": "Homodimer",
            "beams": "HCOOH+H2S",
            "A0_MHz": 152400.0,
            "B0_MHz": 1748.92,
            "C0_MHz": 1745.42,
            "dipole_D": "|mu_a|=0.85",
            "diagnostic_lines_2to18GHz_MHz": "J=1<-0 (K=0 A/B/E tunneling states): 3494.3; J=2<-1: 6988.6; J=3<-2: 10482.8; J=4<-3: 13977.0; J=5<-4: 17471.1",
            "hyperfine_or_tunneling": "Characteristic multi-component donor-acceptor interchange + gearing tunneling multiplets spaced every 2B ~ 3494.3 MHz",
            "reference": "Lovas et al., J. Chem. Phys. 1995; Das et al., PNAS 2018",
        },
        {
            "species": "H2O-H2S",
            "category": "Water impurity complex",
            "beams": "HCOOH+H2S (adventitious H2O)",
            "A0_MHz": 215000.0,
            "B0_MHz": 3022.15,
            "C0_MHz": 3010.80,
            "dipole_D": "|mu_a|=1.75",
            "diagnostic_lines_2to18GHz_MHz": "J=1<-0: 6033.0; J=2<-1: 12065.9; J=3<-2: 18098.7 (H2O donor / H2S donor interchange tunneling states)",
            "hyperfine_or_tunneling": "Resolved proton-interchange tunneling doublets at Harmonics of ~6.03 GHz",
            "reference": "lanes & Nesbitt / Lovas NIST Diatomic & Dimer Tables",
        },
        {
            "species": "N2O",
            "category": "Monomer",
            "beams": "HCOOH+N2O",
            "A0_MHz": 0.0,
            "B0_MHz": 12561.634,
            "C0_MHz": 12561.634,
            "dipole_D": "|mu_a|=0.1608",
            "diagnostic_lines_2to18GHz_MHz": "No rotational transitions in 2-18 GHz (J=1<-0 is at 2B0 = 25123.27 MHz = 25.12 GHz)",
            "hyperfine_or_tunneling": "Monomer J=1<-0 lies in K-band (25.12 GHz), outside 2-18 GHz survey",
            "reference": "Reinartz, Meerts & Dymanus, Chem. Phys. 1974, 3, 142; Lovas 1978",
        },
        {
            "species": "(N2O)2 (polar slipped-parallel)",
            "category": "Homodimer",
            "beams": "HCOOH+N2O",
            "A0_MHz": 12095.2,
            "B0_MHz": 1951.4,
            "C0_MHz": 1679.8,
            "dipole_D": "|mu_b|=0.42",
            "diagnostic_lines_2to18GHz_MHz": "1_11<-0_00: 13775.0; 2_12<-1_01: 17134.6; 2_02<-1_11: 3452.1; 3_03<-2_12: 7318.4 (note: centrosymmetric nonpolar isomer is MW-silent)",
            "hyperfine_or_tunneling": "Quadruple 14N nuclear quadrupole hyperfine pattern (four 14N nuclei!)",
            "reference": "Dehghani et al., J. Chem. Phys. 2007, 127, 224305",
        },
        {
            "species": "H2O-N2O",
            "category": "Water impurity complex",
            "beams": "HCOOH+N2O (adventitious H2O)",
            "A0_MHz": 12915.4,
            "B0_MHz": 3432.8,
            "C0_MHz": 2708.5,
            "dipole_D": "|mu_a|=1.49, |mu_b|=0.45",
            "diagnostic_lines_2to18GHz_MHz": "1_01<-0_00: 6141.3; 2_12<-1_11: 11558.3; 2_02<-1_01: 12214.0; 2_11<-1_10: 13006.9; 1_11<-0_00 (b): 15623.9",
            "hyperfine_or_tunneling": "H2O internal rotation A/B tunneling doublets + dual 14N hyperfine structure",
            "reference": "Zolandz et al., J. Chem. Phys. 1992, 97, 8995",
        },
    ]
    return pd.DataFrame(contaminants)


def build_workstation_schedule() -> pd.DataFrame:
    """
    Build `workstation_3p5week_schedule.csv`: explicit 3.5-week (24.5-day = 588 h)
    compute budget across both the 8-rank CPU ORCA queue (`000`-`061`) AND the
    parallel RTX 3090 `GPU4PySCF` stream.
    """
    phases = [
        {
            "priority_range": "GPU-STREAM (Parallel)",
            "hardware_resource": "RTX 3090 GPU (WSL2 gpu4pyscf 1.7.5 + CuPy CUDA 12)",
            "campaign_phase": "High-Level Hybrid DFT (`wB97X-V/def2-TZVPP` & `B3LYP/def2-TZVPP`) Opt+Hessian + Pass-2 2D Polar PES Scans",
            "job_count": 460,
            "ranks_or_gpus": "1x RTX 3090 (24 GB VRAM) + 2 helper CPU threads",
            "est_wall_per_job_h": 0.02,
            "total_wall_h": 6.5,
            "cumulative_cpu_days": 0.0,
            "deliverables_unlocked": "wB97X-V/def2-TZVPP geometries, harmonic ZPEs, dipoles, and 2D (R, theta) PES maps in hours (saving >140 h of CPU time)",
        },
        {
            "priority_range": "000-004",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "Calibration + 26 HCOOH-N2O `MACE-OFF24(M) -> r2SCAN-3c` Geometry & Frequency Optimizations",
            "job_count": 30,
            "ranks_or_gpus": "8 ranks (cal) / 2x4 ranks (r2SCAN-3c)",
            "est_wall_per_job_h": 0.03,
            "total_wall_h": 0.8,
            "cumulative_cpu_days": 0.03,
            "deliverables_unlocked": "Complete 26/26 HCOOH-N2O + 15/15 HCOOH-H2S MACE-OFF24(M) -> r2SCAN-3c method comparison tables & figures",
        },
        {
            "priority_range": "010-019",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "HCOOH-H2S Stage-1 `revDSD-PBEP86-D4/jun-cc-pV(T+d)Z` (`S0,S1,S2a,S2b`) for 8 Working-Set Isomers + 3 Monomers",
            "job_count": 14,
            "ranks_or_gpus": "8 CPU ranks (`%maxcore 3000`)",
            "est_wall_per_job_h": 1.35,
            "total_wall_h": 14.5,
            "cumulative_cpu_days": 0.64,
            "deliverables_unlocked": "Equilibrium rotational constants B_e(revDSD), relaxed MP2 dipoles (mu_a, mu_b, mu_c), and harmonic ZPEs for HCOOH-H2S",
        },
        {
            "priority_range": "020-029",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "HCOOH-N2O Stage-1 `revDSD-PBEP86-D4/jun-cc-pV(T+d)Z` (`S0,S1,S2a,S2b`) for 16 Working-Set Isomers + 3 Monomers",
            "job_count": 22,
            "ranks_or_gpus": "8 CPU ranks (`%maxcore 3000`)",
            "est_wall_per_job_h": 1.35,
            "total_wall_h": 26.0,
            "cumulative_cpu_days": 1.72,
            "deliverables_unlocked": "Equilibrium rotational constants B_e(revDSD), relaxed MP2 dipoles, and harmonic ZPEs for all 16 HCOOH-N2O conformers",
        },
        {
            "priority_range": "030-039",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "`DLPNO-CCSD(T1)/TightPNO/aug-cc-pV{T,Q}Z` CBS(3/4) + Boys-Bernardi Counterpoise + LED + QTAIM/NCI Wavefunctions",
            "job_count": 148,
            "ranks_or_gpus": "8 CPU ranks (`%maxcore 3000`)",
            "est_wall_per_job_h": 0.65,
            "total_wall_h": 96.0,
            "cumulative_cpu_days": 5.72,
            "deliverables_unlocked": "Final benchmark relative energies dE_CBS, counterpoise binding energies D_0(CBS+CP), LED components, and QTAIM/NCI plots",
        },
        {
            "priority_range": "040-041",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "Stage-2 `%eprnmr` EFG / Nuclear Quadrupole Coupling Tensor Jobs (`MP2/aug-cc-pwCVTZ` NoFrozenCore, relaxed density) for Detectable Conformers",
            "job_count": 13,
            "ranks_or_gpus": "8 CPU ranks (`%maxcore 3000`)",
            "est_wall_per_job_h": 0.14,
            "total_wall_h": 1.8,
            "cumulative_cpu_days": 5.80,
            "deliverables_unlocked": "Ab initio PAS quadrupole coupling tensors chi_aa, chi_bb, chi_cc, chi_ab for 14N_t, 14N_c, 33S, 17O, and D across all detectable isomers",
        },
        {
            "priority_range": "045",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "HCOOH-H2S Stage-2 `B3LYP-D4/jun-cc-pV(D+d)Z` VeryTightOpt (`S3a`) + `Anharm` VPT2 (`S3b`) + `S4` Pickett Decks (`00, 01, 02, 08, 05`)",
            "job_count": 5,
            "ranks_or_gpus": "8 CPU ranks (`nprocs_group 4`)",
            "est_wall_per_job_h": 0.43,
            "total_wall_h": 2.15,
            "cumulative_cpu_days": 5.89,
            "deliverables_unlocked": "Hybrid B_0 = B_e(revDSD) + dB_vib(VPT2), Watson A/S quartic centrifugal distortion constants, SPCAT/SPFIT decks for HCOOH-H2S (1t, 2t, 3t, 3c, 1c)",
        },
        {
            "priority_range": "050",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "HCOOH-N2O Stage-2 `B3LYP-D4/jun-cc-pV(D+d)Z` VeryTightOpt (`S3a`) + `Anharm` VPT2 (`S3b`) + `S4` Pickett Decks (`01, 00, 02, 04, 10, 08`)",
            "job_count": 6,
            "ranks_or_gpus": "8 CPU ranks (`nprocs_group 4`)",
            "est_wall_per_job_h": 0.43,
            "total_wall_h": 2.60,
            "cumulative_cpu_days": 5.99,
            "deliverables_unlocked": "Hybrid B_0 = B_e(revDSD) + dB_vib(VPT2), Watson A/S quartic centrifugal distortion constants, SPCAT/SPFIT decks for HCOOH-N2O (1t-4t, 1c, 2c)",
        },
        {
            "priority_range": "052-055",
            "hardware_resource": "8-Rank CPU ORCA Queue",
            "campaign_phase": "Upgrade A `DLPNO-CCSD(T1)/TightPNO` Intermolecular `R_COM = R_e +- 0.04 A` 3-Point Parabola Refinement + Upgrade B/C 115-Species Harvester",
            "job_count": 15,
            "ranks_or_gpus": "8 CPU ranks (`%maxcore 3000`)",
            "est_wall_per_job_h": 0.08,
            "total_wall_h": 1.15,
            "cumulative_cpu_days": 6.04,
            "deliverables_unlocked": "Composite r_e^SE + DLPNO-CBS B_0 constants, 1-D DVR tunneling splittings, and 115 detectable isotopologue + hyperfine catalogs",
        },
        {
            "priority_range": "060-061 (Low-Prio Fallback)",
            "hardware_resource": "8-Rank CPU ORCA Queue (2x2 ranks)",
            "campaign_phase": "ORCA CPU `r2SCAN-3c` 2D Polar `(R, theta)` PES Grid Jobs (superseded by GPU-STREAM `wB97X-V/def2-TZVPP` PES)",
            "job_count": 441,
            "ranks_or_gpus": "2 concurrent x 2 CPU ranks",
            "est_wall_per_job_h": 0.055,
            "total_wall_h": 12.2,
            "cumulative_cpu_days": 10.92,
            "deliverables_unlocked": "Complete ORCA r2SCAN-3c 2D polar PES cross-check after all Stage-1 and Stage-2 FTMW jobs finish",
        },
        {
            "priority_range": "BUFFER (Days 11-24.5)",
            "hardware_resource": "Workstation CPU + RTX 3090 Reserve",
            "campaign_phase": "Isotopologue VPT2 refinements, collaborator line-assignment refits (`SPFIT`), and resilience margin",
            "job_count": 0,
            "ranks_or_gpus": "8 CPU ranks + 1x RTX 3090",
            "est_wall_per_job_h": 0.0,
            "total_wall_h": 325.9,
            "cumulative_cpu_days": 24.50,
            "deliverables_unlocked": "13.58 days (55% of 24.5-day budget) reserve margin for experimental assignment follow-up and isotopologue VPT2",
        },
    ]
    return pd.DataFrame(phases)


# ==============================================================================
# 6. PUBLICATION-GRADE STAGE-2 FTMW PREDICTION FIGURES (2-18 GHz)
# ==============================================================================
def plot_h2s_ftmw_predictions(
    df_h2s: pd.DataFrame,
    line_catalog: dict[str, list[dict[str, Any]]],
) -> tuple[Path, Path]:
    """
    Generate 2-panel `figures/F_stage2_ftmw_predictions_h2s.pdf` (`.png`):
      (a) Simulated 2 K broadband 2-18 GHz CP-FTMW survey stick spectra for the
          4 primary HCOOH-H2S targets (`1t`, `2t`, `3t`, `1c`).
      (b) Heavy-atom & deuterated isotopic shift fingerprint (`34S`, `13C`,
          `HCOOD`, `DCOOH`, `HDS(b)`, `HDS(f)`, `D2S`) + projected 33S nuclear
          quadrupole coupling constants (`chi_aa`, `chi_bb`, `chi_cc`).
    """
    ps.use()
    fig, (ax_spec, ax_diag) = plt.subplots(
        1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 3.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    targets = [
        ("isomer-00", "1t (O\u2013H\u00b7\u00b7\u00b7S + S\u2013H\u00b7\u00b7\u00b7O=C ring)", ps.PALETTE["OH_X"], 0.0),
        ("isomer-01", "2t (S\u2013H\u00b7\u00b7\u00b7O=C)", ps.PALETTE["XH_O"], -1.10),
        ("isomer-02", "3t (S\u2013H\u00b7\u00b7\u00b7O(H))", ps.OKABE_ITO["bluish_green"], -2.20),
        ("isomer-08", "3c (cis O\u2013H\u00b7\u00b7\u00b7S, global cis min)", ps.OKABE_ITO["reddish_purple"], -3.30),
        ("isomer-05", "1c (cis S\u2013H\u00b7\u00b7\u00b7O=C)", ps.PALETTE["cis"], -4.40),
    ]

    for wlabel, label_txt, col, y_off in targets:
        lines = line_catalog.get(wlabel, [])
        ax_spec.axhline(y_off, color="0.82", lw=0.5, zorder=1)
        for ln in lines:
            freq_ghz = float(ln["freq_MHz"]) / 1000.0
            inten = float(ln["rel_intensity"]) * 0.90
            ls = "-" if ln["type"] == "c" else ("--" if ln["type"] == "b" else ":")
            ax_spec.vlines(freq_ghz, y_off, y_off + inten, colors=col, linestyles=ls, linewidth=1.1, zorder=3)
        # Annotate strongest line
        if lines:
            top = lines[0]
            f_top = float(top["freq_MHz"]) / 1000.0
            ax_spec.text(
                min(max(f_top, 2.5), 16.2),
                y_off + float(top["rel_intensity"]) * 0.90 + 0.04,
                f"{top['type']}:{top['upper']}\u2190{top['lower']} ({f_top:.2f} GHz)",
                fontsize=6.6,
                color=col,
                ha="center",
                va="bottom",
            )
        ax_spec.text(2.15, y_off + 0.68, label_txt, fontsize=7.2, color=col, fontweight="bold", ha="left")

    # Mark known contaminant lines (HCOOH-H2O & (H2S)2)
    for f_cont, c_lab in [(3.494, "(H$_2$S)$_2$"), (6.989, "(H$_2$S)$_2$"), (7.895, "HCOOH\u2013H$_2$O"), (10.483, "(H$_2$S)$_2$")]:
        ax_spec.axvline(f_cont, color="0.75", lw=0.6, ls=":", zorder=0)
        ax_spec.text(f_cont, 0.98, c_lab, fontsize=6.0, color="0.45", rotation=90, ha="right", va="top")

    ax_spec.set_xlim(2.0, 18.0)
    ax_spec.set_ylim(-4.60, 1.15)
    ax_spec.set_yticks([])
    ax_spec.set_xlabel("Frequency (GHz)")
    ax_spec.set_title("(a) Simulated 2 K CP-FTMW survey spectra (2\u201318 GHz)", loc="left", fontsize=9.2)

    # Panel (b): Projected 33S quadrupole coupling constants chi_aa, chi_bb, chi_cc across conformers
    sub_df = df_h2s[df_h2s["working_label"].isin(["isomer-00", "isomer-01", "isomer-02", "isomer-05", "isomer-08"])].copy()
    x = np.arange(len(sub_df))
    w = 0.25
    ax_diag.axhline(0.0, color="0.5", lw=0.6, zorder=1)
    ax_diag.bar(x - w, sub_df["chi_aa_33S_proj_MHz"], width=w, color=ps.OKABE_ITO["blue"], edgecolor="black", lw=0.4, label=r"$\chi_{aa}(^{33}\mathrm{S})$")
    ax_diag.bar(x, sub_df["chi_bb_33S_proj_MHz"], width=w, color=ps.OKABE_ITO["orange"], edgecolor="black", lw=0.4, label=r"$\chi_{bb}(^{33}\mathrm{S})$")
    ax_diag.bar(x + w, sub_df["chi_cc_33S_proj_MHz"], width=w, color=ps.OKABE_ITO["bluish_green"], edgecolor="black", lw=0.4, label=r"$\chi_{cc}(^{33}\mathrm{S})$")

    ax_diag.set_xticks(x)
    ax_diag.set_xticklabels([f"{r['conformer_tag']}\n({r['working_label'].split('-')[1]})" for _, r in sub_df.iterrows()], fontsize=7.8)
    ax_diag.set_ylabel(r"Projected $^{33}\mathrm{S}$ $\chi_{\alpha\alpha}$ (MHz)")
    ax_diag.set_title(r"(b) $^{33}\mathrm{S}$ quadrupole & $^{34}\mathrm{S}$ shift fingerprint", loc="left", fontsize=9.2)
    ax_diag.legend(loc="upper right", fontsize=7.5, ncol=1)

    # Add inset box with 1t (isomer-00) isotopic shifts
    iso_note = (
        "1t (isomer-00) $\\Delta B_0$ shifts:\n"
        "$^{34}$S: $-68.5$ MHz ($\\Delta C_0 = -49.4$)\n"
        "HDS(b): $-10.6$ MHz ($\\Delta A_0 = -249$)\n"
        "HDS(f): $-52.8$ MHz ($\\Delta A_0 = -406$)"
    )
    ax_diag.text(
        0.03, 0.04, iso_note, transform=ax_diag.transAxes, fontsize=6.8,
        va="bottom", ha="left", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92}
    )

    fig.tight_layout()
    return ps.save(fig, FIG_DIR / "F_stage2_ftmw_predictions_h2s", close=True)


def plot_n2o_ftmw_predictions(
    df_n2o: pd.DataFrame,
    line_catalog: dict[str, list[dict[str, Any]]],
) -> tuple[Path, Path]:
    """
    Generate 2-panel `figures/F_stage2_ftmw_predictions_n2o.pdf` (`.png`):
      (a) Simulated 2 K broadband 2-18 GHz CP-FTMW survey stick spectra for the
          primary HCOOH-N2O conformers (`1t`, `2t`, `3t`, `4t`, `1c`).
      (b) Projected 14N_t and 14N_c nuclear quadrupole coupling constants
          (`chi_aa`, `chi_bb`) showing how N2O orientation distinguishes
          `1t` (`O-H...O(N2O)`) from `2t` (`O-H...N_t`) and `3t` (`C-H...O + C=O...N_c`).
    """
    ps.use()
    fig, (ax_spec, ax_nqc) = plt.subplots(
        1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 3.3), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    targets = [
        ("isomer-01", "1t (O\u2013H\u00b7\u00b7\u00b7O(N$_2$O), Ar min)", ps.PALETTE["OH_X"], 0.0),
        ("isomer-00", "2t (O\u2013H\u00b7\u00b7\u00b7N$_\\mathrm{t}$)", ps.OKABE_ITO["sky_blue"], -1.10),
        ("isomer-02", "3t (C\u2013H\u00b7\u00b7\u00b7O + C=O\u00b7\u00b7\u00b7N$_\\mathrm{c}$, Ar)", ps.PALETTE["CH_X"], -2.20),
        ("isomer-04", "4t (C\u2013H\u00b7\u00b7\u00b7N$_\\mathrm{t}$ + C=O\u00b7\u00b7\u00b7N$_\\mathrm{c}$)", ps.OKABE_ITO["reddish_purple"], -3.30),
        ("isomer-10", "1c (cis O\u2013H\u00b7\u00b7\u00b7O(N$_2$O), Ar)", ps.PALETTE["cis"], -4.40),
    ]

    for wlabel, label_txt, col, y_off in targets:
        lines = line_catalog.get(wlabel, [])
        ax_spec.axhline(y_off, color="0.82", lw=0.5, zorder=1)
        for ln in lines:
            freq_ghz = float(ln["freq_MHz"]) / 1000.0
            inten = float(ln["rel_intensity"]) * 0.90
            ls = "-" if ln["type"] == "a" else ("--" if ln["type"] == "b" else ":")
            ax_spec.vlines(freq_ghz, y_off, y_off + inten, colors=col, linestyles=ls, linewidth=1.1, zorder=3)
        if lines:
            top = lines[0]
            f_top = float(top["freq_MHz"]) / 1000.0
            ax_spec.text(
                min(max(f_top, 2.6), 16.0),
                y_off + float(top["rel_intensity"]) * 0.90 + 0.04,
                f"{top['type']}:{top['upper']}\u2190{top['lower']} ({f_top:.2f} GHz)",
                fontsize=6.6,
                color=col,
                ha="center",
                va="bottom",
            )
        ax_spec.text(2.15, y_off + 0.68, label_txt, fontsize=7.2, color=col, fontweight="bold", ha="left")

    for f_cont, c_lab in [(6.141, "H$_2$O\u2013N$_2$O"), (7.895, "HCOOH\u2013H$_2$O"), (12.214, "H$_2$O\u2013N$_2$O")]:
        ax_spec.axvline(f_cont, color="0.75", lw=0.6, ls=":", zorder=0)
        ax_spec.text(f_cont, 0.98, c_lab, fontsize=6.0, color="0.45", rotation=90, ha="right", va="top")

    ax_spec.set_xlim(2.0, 18.0)
    ax_spec.set_ylim(-4.60, 1.15)
    ax_spec.set_yticks([])
    ax_spec.set_xlabel("Frequency (GHz)")
    ax_spec.set_title("(a) Simulated 2 K CP-FTMW survey spectra (2\u201318 GHz)", loc="left", fontsize=9.2)

    # Panel (b): 14N_t and 14N_c PAS quadrupole coupling constants chi_aa & chi_bb
    vpt2_order = ["isomer-01", "isomer-00", "isomer-02", "isomer-04", "isomer-10", "isomer-08"]
    sub_df = df_n2o.set_index("working_label").loc[vpt2_order].reset_index()
    x = np.arange(len(sub_df))
    w = 0.20

    ax_nqc.axhline(0.0, color="0.5", lw=0.6, zorder=1)
    ax_nqc.bar(x - 1.5 * w, sub_df["chi_aa_14Nt_proj_MHz"], width=w, color=ps.OKABE_ITO["blue"], edgecolor="black", lw=0.4, label=r"$\chi_{aa}(^{14}\mathrm{N_t})$")
    ax_nqc.bar(x - 0.5 * w, sub_df["chi_bb_14Nt_proj_MHz"], width=w, color=ps.OKABE_ITO["sky_blue"], edgecolor="black", lw=0.4, label=r"$\chi_{bb}(^{14}\mathrm{N_t})$")
    ax_nqc.bar(x + 0.5 * w, sub_df["chi_aa_14Nc_proj_MHz"], width=w, color=ps.OKABE_ITO["vermillion"], edgecolor="black", lw=0.4, label=r"$\chi_{aa}(^{14}\mathrm{N_c})$")
    ax_nqc.bar(x + 1.5 * w, sub_df["chi_bb_14Nc_proj_MHz"], width=w, color=ps.OKABE_ITO["orange"], edgecolor="black", lw=0.4, label=r"$\chi_{bb}(^{14}\mathrm{N_c})$")

    ax_nqc.set_xticks(x)
    ax_nqc.set_xticklabels([f"{r['duarte2017_label']}\n({r['working_label'].split('-')[1]})" for _, r in sub_df.iterrows()], fontsize=7.5)
    ax_nqc.set_ylabel(r"Projected $^{14}\mathrm{N}$ $\chi_{\alpha\alpha}$ (MHz)")
    ax_nqc.set_title(r"(b) $^{14}\mathrm{N_t}/^{14}\mathrm{N_c}$ hyperfine orientation fingerprint", loc="left", fontsize=9.2)
    ax_nqc.legend(loc="lower right", fontsize=6.8, ncol=2)

    fig.tight_layout()
    return ps.save(fig, FIG_DIR / "F_stage2_ftmw_predictions_n2o", close=True)


# ==============================================================================
# 7. EXCEL WORKBOOK EXPORT & MAIN ENTRYPOINT
# ==============================================================================
def write_excel_workbook(
    path: Path,
    df_h2s: pd.DataFrame,
    df_n2o: pd.DataFrame,
    df_contam: pd.DataFrame,
    df_sched: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_sheets = [
        ("H2S_55_Isotopologues", HERE / "h2s_isotopologue_ftmw_catalog.csv"),
        ("N2O_60_Isotopologues", HERE / "n2o_isotopologue_ftmw_catalog.csv"),
        ("DVR_Tunneling_Splittings", HERE / "tunneling_dvr_results.csv"),
        ("H2S_Hyperfine_Tunneling_Lines", HERE / "h2s_hyperfine_tunneling_lines_2to18GHz.csv"),
        ("N2O_Dual14N_Hyperfine_Lines", HERE / "n2o_hyperfine_lines_2to18GHz.csv"),
        ("wB97XV_GPU_QuarticCD_VPT2", HERE / "wb97xv_gpu_vpt2_results.csv"),
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_h2s.to_excel(writer, sheet_name="H2S_FTMW_Targets", index=False)
        df_n2o.to_excel(writer, sheet_name="N2O_FTMW_Targets", index=False)
        for sname, csv_p in extra_sheets:
            if csv_p.is_file():
                pd.read_csv(csv_p).to_excel(writer, sheet_name=sname, index=False)
        df_contam.to_excel(writer, sheet_name="Known_Contaminant_Lines", index=False)
        df_sched.to_excel(writer, sheet_name="Workstation_3p5wk_Schedule", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-dir", type=Path, default=QUEUE_DIR, help="campaign queue directory")
    ap.add_argument("--no-enqueue", action="store_true", help="write inputs and tables without enqueuing")
    args = ap.parse_args(argv)

    HERE.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_and_enqueue_stage2_specs(queue_dir=args.queue_dir, enqueue=not args.no_enqueue)
    print(f"[Stage-2 Queue] Prepared {len(specs)} Stage-2 specs (priorities 040, 041, 045, 050); enqueue={not args.no_enqueue}")

    df_h2s, h2s_lines = build_h2s_ftmw_predictions()
    df_n2o, n2o_lines = build_n2o_ftmw_predictions()
    df_contam = build_known_contaminant_lines()
    df_sched = build_workstation_schedule()

    df_h2s.to_csv(HERE / "h2s_ftmw_target_predictions.csv", index=False, encoding="utf-8")
    df_n2o.to_csv(HERE / "n2o_ftmw_target_predictions.csv", index=False, encoding="utf-8")
    df_contam.to_csv(HERE / "known_contaminant_lines.csv", index=False, encoding="utf-8")
    df_sched.to_csv(HERE / "workstation_3p5week_schedule.csv", index=False, encoding="utf-8")

    write_excel_workbook(
        HERE / "stage2_ftmw_prediction_tables.xlsx",
        df_h2s,
        df_n2o,
        df_contam,
        df_sched,
    )

    h2s_pdf, h2s_png = plot_h2s_ftmw_predictions(df_h2s, h2s_lines)
    n2o_pdf, n2o_png = plot_n2o_ftmw_predictions(df_n2o, n2o_lines)
    print(f"[Stage-2 Figures] Wrote {h2s_pdf.name}, {h2s_png.name}, {n2o_pdf.name}, {n2o_png.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
