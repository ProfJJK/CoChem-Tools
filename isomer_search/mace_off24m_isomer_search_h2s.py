#!/usr/bin/env python3
r"""
================================================================================
CoChem-GOAT / TOPOS: In-Process MACE-OFF24m Isomer Search (no ORCA ExtOpt)
File: hcooh-hsh-isomer-search-mace-off24m.py
================================================================================
1. PURPOSE:
   Finds, deduplicates, verifies and organizes the isomers of the HCOOH-H2S
   van der Waals / hydrogen-bonded complex (8 atoms) on the MACE-OFF24 (medium)
   potential ("MACE-OFF24m") inside ONE long-lived Python process. The model is
   loaded once and many structures are optimized together in single GPU calls.
   No subprocesses are started (no ORCA ExtOpt, no oet_client), at most
   MAX_THREADS (4) CPU threads are used, and nothing runs in parallel processes,
   per this host's stability rules. ORCA is used only afterwards: a ready-to-run
   input for the next level is written per isomer but NOT executed.

2. MECHANICS & SCIENTIFIC CHOICES:
   - Phase 0 (Model): MACE-OFF24m is loaded twice from the cached model file:
     float32 for the coarse batched search, float64 for refinement and Hessians.
     A self-check compares the batched GPU evaluation with MACE's own ASE
     calculator and stops the run if they disagree.
   - Phase 1 (Monomers): trans- and cis-HCOOH and H2S are optimized in float64.
     They provide the cis/trans starting geometries and the reference energies
     for binding energies. A frequency sanity check on trans-HCOOH guards the
     Hessian units and sign.
   - Phase 2 (Starts): the 4 user seed geometries plus N_STARTS_PER_CONFORMER
     random rigid-body placements of H2S around each HCOOH conformer (uniform
     random orientation and direction; closest contact drawn from
     CONTACT_SCALE_RANGE times the sum of vdW radii).
   - Phase 3 (Batched coarse search): all starts are optimized together with a
     vectorized FIRE optimizer on the GPU (float32, fmax COARSE_FMAX_EV_A).
     Graph edges are rebuilt on the GPU at every step. Dissociated or chemically
     changed structures (covalent bonds broken or formed) are discarded and
     near-identical minima are merged (same basin within COARSE_RMSD_TOL_A).
   - Phase 4 (Tight refinement): every coarse survivor is re-optimized one by
     one with ASE BFGS in float64 (fmax TIGHT_FMAX_EV_A, tighter than ORCA
     TightOpt), capped at MAX_SECONDS_PER_ISOMER.
   - Phase 5 (Deduplication, geometry first): within ENERGY_TOL_KCAL, two minima
     are the same isomer if they superimpose (permutation- and mirror-aware
     Kabsch RMSD <= RMSD_TOL_A) or, for the same HCOOH conformer, match in
     rotational constants A, B, C plus the planar moment Pcc. The H-bond motif
     is only a description: labels from distance cutoffs flip for borderline
     contacts, so they never decide identity.
   - Phase 6 (Verification): analytic MACE Hessian -> harmonic frequencies with
     translations/rotations projected out, ZPVE. An isomer with an imaginary
     mode is displaced along it in both directions, re-optimized, and the pool
     is deduplicated again (one round).
   - Phase 6b (Floppy families): minima of the same HCOOH conformer within
     FAMILY_ENERGY_TOL_KCAL and FAMILY_RMSD_TOL_A of each other (single
     linkage) are merged into one isomer. Such sets lie along a nearly flat
     coordinate (e.g. almost free H2S rotation about one contact), far below the
     zero-point level of that motion, so a spectrum shows one species. The
     lowest-energy member without imaginary modes represents the family.
   - Phase 7 (Inventory): sorted CSV, annotated multi-frame XYZ, and per isomer
     one XYZ file plus one ORCA input for the next level (written, not run).
   - Rebuild mode (--rebuild): re-runs Phases 5, 6b and 7 on the isomers saved
     by an earlier search in the current folder (no model, no optimization);
     the earlier files are kept in 'before_rebuild'.

3. EXPECTED OUTCOMES (SUCCESS vs. FAILURE):
   - Success: 'HCOOH-HSH_MACE-OFF24m-search_Inventory.csv',
     'HCOOH-HSH_MACE-OFF24m-search.finalensemble.xyz' and the folder
     'HCOOH-HSH_MACE-OFF24m-search_isomers', ordered from lowest to highest
     energy. The 'Hits' column counts how many independent starts reached each
     isomer; isomers reached only once suggest increasing N_STARTS_PER_CONFORMER.
     'Family_Size' > 1 marks a merged floppy family ('Family_Span_cm-1' = its
     energy spread).
   - Failure: the script stops with a clear message (model file missing,
     self-check mismatch, implausible monomer frequencies) before any long work.

Run it with the MACE environment, from the folder that should receive the results:
    C:\ORCA_6.1.1\orca-external-tools-2.0.0\.venv-maceoff\Scripts\python.exe hcooh-hsh-isomer-search-mace-off24m.py
Rebuild an existing inventory, from its results folder:
    C:\ORCA_6.1.1\orca-external-tools-2.0.0\.venv-maceoff\Scripts\python.exe ..\hcooh-hsh-isomer-search-mace-off24m.py --rebuild
================================================================================
"""

from __future__ import annotations

import os

# Host rule: at most 4-way parallelism. Must be set before numpy/torch load.
MAX_THREADS: int = 4
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = str(MAX_THREADS)

import argparse
import csv
import math
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.optimize import BFGS
from scipy.spatial.transform import Rotation

from mace.calculators.foundations_models import mace_off
from mace.calculators.mace import MACECalculator


# ==============================================================================
# 0. CENTRALIZED CONFIGURATION
# ==============================================================================
MOL_NAME: str = "HCOOH-HSH"
LEVEL_TAG: str = "MACE-OFF24m-search"
CHARGE: int = 0          # MACE-OFF has no charge/spin input; used for ORCA inputs only
MULTIPLICITY: int = 1

MODEL_FILE: Path = Path(
    r"C:\ORCA_6.1.1\orca-external-tools-2.0.0\src\oet\assets\mace-off\MACE-OFF24_medium.model"
)
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Search size and sampling
N_STARTS_PER_CONFORMER: int = 600         # random placements per HCOOH conformer (trans, cis)
RANDOM_SEED: int = 20261023
CONTACT_SCALE_RANGE: Tuple[float, float] = (0.95, 1.30)  # closest contact / sum of vdW radii

# Phase 3: batched coarse optimization (float32, GPU)
BATCH_SIZE: int = 300
COARSE_FMAX_EV_A: float = 0.01
COARSE_MAX_STEPS: int = 3000
FLYAWAY_COM_DIST_A: float = 10.0          # stop optimizing complexes that drift apart

# Phase 4: tight refinement (float64, ASE BFGS)
TIGHT_FMAX_EV_A: float = 0.0005           # ORCA TightOpt max gradient ~0.005 eV/A
TIGHT_MAX_STEPS: int = 3000
MAX_SECONDS_PER_ISOMER: float = 600.0     # hard cap per isomer (refinement)

# Structure validity
MAX_INTER_COM_DIST_A: float = 6.50        # dissociation guard: cull exploded dimers
BOND_FACTOR: float = 1.25                 # covalent bond if d < factor * (r_cov_i + r_cov_j)

# Deduplication thresholds
COARSE_ENERGY_TOL_KCAL: float = 0.30      # coarse minima are loosely converged
COARSE_RMSD_TOL_A: float = 0.20           # same basin; final dedup re-checks after refinement
ENERGY_TOL_KCAL: float = 0.02             # tightly converged float64 minima
RMSD_TOL_A: float = 0.10
ROT_REL_TOL: float = 0.001                # 0.1% on each of A, B, C
PCC_TOL_AMU_A2: float = 0.02
FINAL_MAXEN_KCAL: float = 8.0             # keep isomers within this window of the minimum

# Floppy families: near-degenerate minima along a nearly flat coordinate are one isomer
FAMILY_ENERGY_TOL_KCAL: float = 0.05      # ~17 cm-1
FAMILY_RMSD_TOL_A: float = 0.60           # to at least one family member (single linkage)

# Phase 6: frequencies
IMAG_FREQ_TOL_CM1: float = -10.0          # modes below this count as imaginary
SADDLE_DISPLACEMENT_A: float = 0.10       # largest atomic displacement along an imaginary mode

# Next-level ORCA inputs (written per isomer, NOT run; the original Phase 4 minus its Pickett file)
ORCA_NEXT_TAG: str = "r2scan3c"
ORCA_NEXT_NPROCS: int = 8                 # owner-run jobs; agents must stay <= 4
ORCA_NEXT_MAXCORE_MB: int = 1500
ORCA_NEXT_BLOCK: str = (
    "! r2SCAN-3c TightOpt Freq DefGrid3 TightSCF PrintBasis\n"
    "%geom\n  InHess XTB2\n  MaxIter 150\nend\n"
    "%elprop\n  Dipole true\n  Quadrupole true\nend\n"
)

# Physical constants (CODATA 2018)
EV_PER_HARTREE: float = 27.211386245988
EV_TO_KCAL: float = 23.060547830619
EV_TO_CM1: float = 8065.543937349
KCAL_TO_CM1: float = 349.7550112241
PLANCK_INERTIA_TO_MHZ: float = 505379.006  # amu * A^2 -> MHz
# sqrt(eV / (A^2 amu)) expressed as a wavenumber in cm^-1
HESSIAN_TO_CM1: float = math.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (
    2.0 * math.pi * 2.99792458e10
)

# Molecule definition: atom order C, O(H), H(O), O(=C), H(C) | S, H, H
SYMBOLS: List[str] = ["C", "O", "H", "O", "H", "S", "H", "H"]
FRAGMENT_1_INDICES: List[int] = [0, 1, 2, 3, 4]   # HCOOH
FRAGMENT_2_INDICES: List[int] = [5, 6, 7]         # H2S
MONOISOTOPIC_MASSES: Dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "O": 15.99491461957,
    "S": 31.97207117440,
}
VDW_RADII_A: Dict[str, float] = {"H": 1.20, "C": 1.70, "O": 1.52, "S": 1.80}  # Bondi

# Intermolecular hydrogen-bond definitions: (H index, donor index, label)
HB_DONORS: List[Tuple[int, int, str]] = [(2, 1, "O-H"), (4, 0, "C-H"), (6, 5, "S-H"), (7, 5, "S-H")]
HB_ACCEPTORS: List[Tuple[int, str]] = [(1, "O(H)"), (3, "O=C"), (5, "S")]
HB_MAX_DIST_A: Dict[str, float] = {"O": 2.6, "S": 3.0}  # H...acceptor, by acceptor element
HB_MIN_ANGLE_DEG: float = 110.0
HB_WEAK_MARGIN_A: float = 0.30  # contacts up to cutoff + margin are listed as "(weak)"

# Starting seed geometries (same as the GOAT scripts)
INITIAL_SEEDS: Dict[str, List[Tuple[float, float, float]]] = {
    "seed0": [
        (-4.6996660000, 3.4287870000, 0.0984920000),
        (-3.8766170000, 3.6098370000, 1.1442380000),
        (-3.0131560000, 3.2625300000, 0.8377610000),
        (-4.3168430000, 2.9272580000, -0.9446300000),
        (-5.7169270000, 3.7854600000, 0.3219000000),
        (-4.6656380000, 2.0040000000, -4.3409320000),
        (-4.1969600000, 2.2926680000, -3.1123150000),
        (-3.6167710000, 1.2285040000, -4.6496350000),
    ],
    "seed1": [
        (-4.6424554071, 3.3844103471, -0.0237193970),
        (-3.7172503334, 3.4679916277, 0.9449509700),
        (-2.9755288823, 2.8959694903, 0.6457720582),
        (-4.4811394328, 2.7028980730, -1.0187315863),
        (-5.5220015207, 4.0027523356, 0.2107309633),
        (-1.0857716844, 2.0315170895, -0.6837059128),
        (-1.9903704905, 1.5679816508, -1.5640944603),
        (-0.6736890368, 3.0034953930, -1.5123290909),
    ],
    "seed2": [
        (-4.8191453968, 3.3828376739, -0.0080259718),
        (-3.8605336928, 3.5382697915, 0.9173909159),
        (-3.0528224595, 3.1404543157, 0.5194609454),
        (-4.6128204202, 2.8327596459, -1.0738664873),
        (-5.7734396202, 3.8119425080, 0.3311378839),
        (-0.8072861521, 2.7324408970, -0.1956690237),
        (0.3517788247, 3.3689223110, 0.0331420864),
        (-0.3545808002, 2.1539400570, -1.3195163776),
    ],
    "seed4": [
        (-4.8098833526, 3.4517170456, 0.1033066321),
        (-3.7800784688, 3.5548195946, 0.9573611630),
        (-2.9971970994, 3.2233498872, 0.4660747772),
        (-4.6749256950, 3.0138009661, -1.0252349080),
        (-5.7458329015, 3.8077331644, 0.5585436581),
        (-1.5508180601, 2.4110528050, -1.8243470076),
        (-2.8748554774, 2.5921444052, -1.9996916978),
        (-1.4495944712, 1.6153927772, -2.9000419888),
    ],
}

# FIRE parameters (ASE defaults)
FIRE_DT0, FIRE_DTMAX, FIRE_MAXSTEP = 0.1, 1.0, 0.2
FIRE_NMIN, FIRE_FINC, FIRE_FDEC, FIRE_ASTART, FIRE_FA = 5, 1.1, 0.5, 0.1, 0.99


# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================
@dataclass
class Minimum:
    """One local minimum with its fingerprints and verification data."""

    coords: np.ndarray            # (8, 3) in Angstrom
    energy_ev: float
    origin: str                   # start that produced it (e.g. random-cis-0042, seed1)
    hits: int = 1                 # number of starts that ended in this minimum
    conformer: str = ""           # HCOOH conformer: trans / cis
    motif: str = ""               # intermolecular H-bond motif
    rot_mhz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kappa: float = 0.0            # Ray's asymmetry parameter
    pcc_amu_a2: float = 0.0       # planar moment Pcc
    freqs_cm1: Optional[np.ndarray] = None  # 3N-6 harmonic frequencies (negative = imaginary)
    modes: Optional[np.ndarray] = None      # Cartesian displacement vectors, (3N-6, N, 3)
    zpve_ev: float = 0.0
    label: str = ""
    binding_kcal: Optional[float] = None    # De vs relaxed monomers of the same HCOOH conformer
    family_size: int = 1                    # minima merged into this floppy family
    family_span_cm1: float = 0.0
    stored_imag: Optional[int] = None       # frequency summary read back by --rebuild
    stored_lowest_vib: Optional[float] = None

    @property
    def num_imag(self) -> int:
        if self.freqs_cm1 is None:
            return self.stored_imag or 0
        return int(np.sum(self.freqs_cm1 < IMAG_FREQ_TOL_CM1))

    @property
    def lowest_vib_cm1(self) -> Optional[float]:
        if self.freqs_cm1 is None:
            return self.stored_lowest_vib
        positive = self.freqs_cm1[self.freqs_cm1 > 0.0]
        return float(positive.min()) if positive.size else None


# ==============================================================================
# 2. GEOMETRY, FINGERPRINTS AND KABSCH ENGINE
# ==============================================================================
MASSES = np.array([MONOISOTOPIC_MASSES[s] for s in SYMBOLS])
COV_RADII = np.array([covalent_radii[atomic_numbers[s]] for s in SYMBOLS])


def center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()


def inter_com_distance(coords: np.ndarray) -> float:
    """Distance between the HCOOH and H2S centers of mass."""
    c1 = center_of_mass(coords[FRAGMENT_1_INDICES], MASSES[FRAGMENT_1_INDICES])
    c2 = center_of_mass(coords[FRAGMENT_2_INDICES], MASSES[FRAGMENT_2_INDICES])
    return float(np.linalg.norm(c1 - c2))


def bond_set(coords: np.ndarray) -> frozenset:
    """Covalent bonds as index pairs, from scaled covalent radii."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    limit = BOND_FACTOR * (COV_RADII[:, None] + COV_RADII[None, :])
    i, j = np.where(np.triu(d < limit, k=1))
    return frozenset(zip(i.tolist(), j.tolist()))


def dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def hcooh_conformer(coords: np.ndarray) -> str:
    """trans-HCOOH has the H-C-O-H dihedral near 180 deg, cis-HCOOH near 0 deg."""
    phi = dihedral_deg(coords[4], coords[0], coords[1], coords[2])
    return "trans" if abs(phi) > 90.0 else "cis"


def hbond_motif(coords: np.ndarray) -> str:
    """
    Sorted list of intermolecular D-H...A contacts. This is a description only
    (identity is decided by geometry); contacts just beyond the distance cutoff
    are listed as "(weak)" instead of disappearing.
    """
    frag = {i: 1 for i in FRAGMENT_1_INDICES} | {i: 2 for i in FRAGMENT_2_INDICES}
    strong, weak = set(), set()
    for h, donor, d_label in HB_DONORS:
        for acc, a_label in HB_ACCEPTORS:
            if frag[h] == frag[acc]:
                continue
            r_ha = np.linalg.norm(coords[acc] - coords[h])
            limit = HB_MAX_DIST_A[SYMBOLS[acc]]
            if r_ha > limit + HB_WEAK_MARGIN_A:
                continue
            v1, v2 = coords[donor] - coords[h], coords[acc] - coords[h]
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
            if angle >= HB_MIN_ANGLE_DEG:
                (strong if r_ha <= limit else weak).add(f"{d_label}...{a_label}")
    labels = sorted(strong) + [f"{lab} (weak)" for lab in sorted(weak - strong)]
    return " + ".join(labels) if labels else "no H-bond"


def rotational_fingerprint(coords: np.ndarray) -> Tuple[Tuple[float, float, float], float, float]:
    """Rotational constants A >= B >= C (MHz), Ray's kappa, and planar moment Pcc (amu A^2)."""
    rel = coords - center_of_mass(coords, MASSES)
    inertia = np.zeros((3, 3))
    for m, r in zip(MASSES, rel):
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    ia, ib, ic = np.sort(np.linalg.eigvalsh(inertia))
    a, b, c = (PLANCK_INERTIA_TO_MHZ / i if i > 1e-6 else 0.0 for i in (ia, ib, ic))
    kappa = (2.0 * b - a - c) / (a - c) if a - c > 1e-9 else 0.0
    pcc = 0.5 * (ia + ib - ic)
    return (a, b, c), kappa, pcc


def annotate(m: Minimum) -> Minimum:
    m.conformer = hcooh_conformer(m.coords)
    m.motif = hbond_motif(m.coords)
    m.rot_mhz, m.kappa, m.pcc_amu_a2 = rotational_fingerprint(m.coords)
    return m


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Minimum RMSD after optimal superposition, testing the H2S hydrogen swap
    (atoms 6 <-> 7) and the mirror image (enantiomers have identical spectra).
    """
    a = coords_a - center_of_mass(coords_a, MASSES)
    b = coords_b - center_of_mass(coords_b, MASSES)

    def _rmsd(p: np.ndarray, q: np.ndarray) -> float:
        u, _, vt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        return float(np.sqrt(np.mean(np.sum((p @ rot.T - q) ** 2, axis=1))))

    mirror = a * np.array([1.0, 1.0, -1.0])
    perm = [0, 1, 2, 3, 4, 5, 7, 6]
    return min(_rmsd(a, b), _rmsd(mirror, b), _rmsd(a[perm], b), _rmsd(mirror[perm], b))


def same_isomer(a: Minimum, b: Minimum, energy_tol_kcal: float, rmsd_tol_a: float, use_rot: bool) -> bool:
    """
    Geometry first: within the energy tolerance, minima that superimpose are the
    same isomer whatever their labels. Rotational constants plus Pcc are a
    secondary test, only within the same HCOOH conformer.
    """
    if abs(a.energy_ev - b.energy_ev) * EV_TO_KCAL > energy_tol_kcal:
        return False
    if kabsch_rmsd(a.coords, b.coords) <= rmsd_tol_a:
        return True
    if not use_rot or a.conformer != b.conformer:
        return False
    rel = max(abs(x - y) / y for x, y in zip(a.rot_mhz, b.rot_mhz))
    return rel <= ROT_REL_TOL and abs(a.pcc_amu_a2 - b.pcc_amu_a2) <= PCC_TOL_AMU_A2


def deduplicate(pool: List[Minimum], energy_tol_kcal: float, rmsd_tol_a: float, use_rot: bool, stage: str) -> List[Minimum]:
    """Greedy clustering from the lowest energy up; duplicates add their hits to the representative."""
    unique: List[Minimum] = []
    for cand in sorted(pool, key=lambda m: m.energy_ev):
        for rep in unique:
            if same_isomer(cand, rep, energy_tol_kcal, rmsd_tol_a, use_rot):
                rep.hits += cand.hits
                break
        else:
            unique.append(cand)
    print(f"  [DEDUP - {stage}] {len(pool)} minima -> {len(unique)} unique")
    return unique


def merge_families(isomers: List[Minimum]) -> List[Minimum]:
    """
    Single-linkage merge of near-degenerate minima of the same HCOOH conformer
    (within FAMILY_ENERGY_TOL_KCAL and FAMILY_RMSD_TOL_A of a family member).
    Such sets lie along a nearly flat coordinate (e.g. almost free H2S rotation
    about one contact), far below the zero-point level of that motion, so a
    spectrum shows one species. The lowest-energy member without imaginary
    modes represents the family; hits are summed.
    """
    parent = list(range(len(isomers)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(isomers):
        for j in range(i + 1, len(isomers)):
            b = isomers[j]
            if (a.conformer == b.conformer
                    and abs(a.energy_ev - b.energy_ev) * EV_TO_KCAL <= FAMILY_ENERGY_TOL_KCAL
                    and kabsch_rmsd(a.coords, b.coords) <= FAMILY_RMSD_TOL_A):
                parent[root(i)] = root(j)

    groups: Dict[int, List[Minimum]] = {}
    for i, m in enumerate(isomers):
        groups.setdefault(root(i), []).append(m)
    merged: List[Minimum] = []
    for members in groups.values():
        rep = min(members, key=lambda m: (m.num_imag > 0, m.energy_ev))
        if len(members) > 1:
            energies = [m.energy_ev for m in members]
            rep.hits = sum(m.hits for m in members)
            rep.family_size = sum(m.family_size for m in members)
            rep.family_span_cm1 = (max(energies) - min(energies)) * EV_TO_CM1
        merged.append(rep)
    print(f"  [FAMILIES] {len(isomers)} isomers -> {len(merged)} after merging near-degenerate floppy families")
    return merged


def is_valid_complex(coords: np.ndarray, reference_bonds: frozenset) -> Tuple[bool, str]:
    if inter_com_distance(coords) > MAX_INTER_COM_DIST_A:
        return False, "dissociated"
    if bond_set(coords) != reference_bonds:
        return False, "covalent bonds changed"
    return True, ""


# ==============================================================================
# 3. MACE-OFF24m: CALCULATORS AND BATCHED GPU EVALUATION
# ==============================================================================
def load_calculators() -> Tuple[MACECalculator, MACECalculator]:
    """MACE-OFF24m in float32 (coarse search) and float64 (refinement, Hessians)."""
    if not MODEL_FILE.is_file():
        raise SystemExit(f"CRITICAL ERROR: model file not found: {MODEL_FILE}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc32 = mace_off(model=str(MODEL_FILE), device=DEVICE, default_dtype="float32")
        calc64 = mace_off(model=str(MODEL_FILE), device=DEVICE, default_dtype="float64")
    return calc32, calc64


class BatchedMace:
    """
    Evaluates energies and forces of many same-size molecules in one model call.
    Graph edges are all atom pairs closer than r_max within each molecule (how
    MACE builds them for non-periodic systems), rebuilt on the GPU at every call,
    so no CPU neighbor lists are built inside the optimization loop.
    """

    def __init__(self, calc: MACECalculator, template_coords: np.ndarray) -> None:
        self.model = calc.models[0]
        self.dtype = next(self.model.parameters()).dtype
        self.device = next(self.model.parameters()).device
        self.r_max = float(calc.r_max)
        self.n_atoms = len(SYMBOLS)
        # One graph built by MACE itself supplies node attributes, head and cell
        template = calc._atoms_to_batch(Atoms(symbols=SYMBOLS, positions=template_coords))
        self.node_attrs_1 = template["node_attrs"].to(self.device, self.dtype)
        self.cell_1 = template["cell"].to(self.device, self.dtype)
        self.head_1 = template["head"].to(self.device) if "head" in template.keys else None
        ii, jj = np.where(~np.eye(self.n_atoms, dtype=bool))
        self.pair_i = torch.tensor(ii, device=self.device)
        self.pair_j = torch.tensor(jj, device=self.device)

    def energy_forces(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pos: (B, N, 3) in Angstrom -> energies (B,) in eV, forces (B, N, 3) in eV/A."""
        bsz, n = pos.shape[0], self.n_atoms
        flat = pos.reshape(-1, 3).detach().clone()
        offsets = (torch.arange(bsz, device=self.device) * n).repeat_interleave(len(self.pair_i))
        cand_i = self.pair_i.repeat(bsz) + offsets
        cand_j = self.pair_j.repeat(bsz) + offsets
        with torch.no_grad():
            keep = (flat[cand_j] - flat[cand_i]).norm(dim=1) < self.r_max
        edge_index = torch.stack([cand_i[keep], cand_j[keep]])
        zeros = torch.zeros((edge_index.shape[1], 3), dtype=self.dtype, device=self.device)
        data = {
            "positions": flat,
            "node_attrs": self.node_attrs_1.repeat(bsz, 1),
            "edge_index": edge_index,
            "shifts": zeros,
            "unit_shifts": zeros,
            "cell": self.cell_1.repeat(bsz, 1),
            "batch": torch.arange(bsz, device=self.device).repeat_interleave(n),
            "ptr": torch.arange(0, (bsz + 1) * n, n, device=self.device),
        }
        if self.head_1 is not None:
            data["head"] = self.head_1.repeat(bsz)
        out = self.model(data, training=False, compute_force=True, compute_stress=False)
        return out["energy"].detach(), out["forces"].detach().reshape(bsz, n, 3)

    def evaluate_chunked(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        energies, forces = [], []
        for k in range(0, len(coords), BATCH_SIZE):
            pos = torch.tensor(coords[k:k + BATCH_SIZE], dtype=self.dtype, device=self.device)
            e, f = self.energy_forces(pos)
            energies.append(e.cpu().numpy())
            forces.append(f.cpu().numpy())
        return np.concatenate(energies), np.concatenate(forces)


def self_check(ev: BatchedMace, calc: MACECalculator, coords: np.ndarray, tol_e: float, tol_f: float) -> None:
    """
    Batched evaluation of two different structures must reproduce MACE's own ASE
    calculator for each of them (catches API drift and cross-talk in the batch).
    """
    rng = np.random.default_rng(1)
    pair = np.stack([coords, coords + rng.normal(0.0, 0.03, coords.shape)])
    e_b, f_b = ev.evaluate_chunked(pair)
    worst_e = worst_f = 0.0
    for k in range(2):
        atoms = Atoms(symbols=SYMBOLS, positions=pair[k])
        atoms.calc = calc
        worst_e = max(worst_e, abs(e_b[k] - atoms.get_potential_energy()))
        worst_f = max(worst_f, float(np.abs(f_b[k] - atoms.get_forces()).max()))
    print(f"  [CHECK] batched vs MACECalculator ({ev.dtype}): max|dE| = {worst_e:.2e} eV, max|dF| = {worst_f:.2e} eV/A")
    if worst_e > tol_e or worst_f > tol_f:
        raise SystemExit("CRITICAL ERROR: batched MACE evaluation disagrees with MACECalculator; aborting.")


# ==============================================================================
# 4. OPTIMIZERS: BATCHED FIRE (GPU) AND TIGHT BFGS (ASE)
# ==============================================================================
def batched_fire(ev: BatchedMace, coords: np.ndarray, fmax: float, max_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized FIRE (ASE parameters) for a batch of molecules. Converged or
    drifting-apart structures stop moving; only active ones are evaluated.
    Returns final coordinates and a convergence mask.
    """
    r = torch.tensor(coords, dtype=ev.dtype, device=ev.device)
    bsz = r.shape[0]
    v = torch.zeros_like(r)
    dt = torch.full((bsz,), FIRE_DT0, dtype=ev.dtype, device=ev.device)
    alpha = torch.full((bsz,), FIRE_ASTART, dtype=ev.dtype, device=ev.device)
    n_pos = torch.zeros(bsz, dtype=torch.long, device=ev.device)
    active = torch.ones(bsz, dtype=torch.bool, device=ev.device)
    converged = torch.zeros(bsz, dtype=torch.bool, device=ev.device)
    masses = torch.tensor(MASSES, dtype=ev.dtype, device=ev.device)
    f1, f2 = FRAGMENT_1_INDICES, FRAGMENT_2_INDICES

    for step in range(max_steps):
        idx = torch.nonzero(active).squeeze(1)
        if idx.numel() == 0:
            break
        _, f = ev.energy_forces(r[idx])
        done = f.norm(dim=2).max(dim=1).values <= fmax
        converged[idx[done]] = True
        active[idx[done]] = False
        keep = ~done
        idx, f = idx[keep], f[keep]
        if idx.numel() == 0:
            break

        vi, dti, ai, ni = v[idx], dt[idx], alpha[idx], n_pos[idx]
        if step > 0:
            uphill = (f * vi).sum(dim=(1, 2)) <= 0.0
            f_norm = f.norm(dim=(1, 2)).clamp_min(1e-12)[:, None, None]
            v_norm = vi.norm(dim=(1, 2))[:, None, None]
            mixed = (1.0 - ai)[:, None, None] * vi + ai[:, None, None] * f / f_norm * v_norm
            vi = torch.where(uphill[:, None, None], torch.zeros_like(vi), mixed)
            grow = ~uphill & (ni > FIRE_NMIN)
            dti = torch.where(grow, (dti * FIRE_FINC).clamp(max=FIRE_DTMAX), dti)
            ai = torch.where(grow, ai * FIRE_FA, ai)
            ni = torch.where(uphill, torch.zeros_like(ni), ni + 1)
            dti = torch.where(uphill, dti * FIRE_FDEC, dti)
            ai = torch.where(uphill, torch.full_like(ai, FIRE_ASTART), ai)
        vi = vi + dti[:, None, None] * f
        dr = dti[:, None, None] * vi
        scale = (FIRE_MAXSTEP / dr.norm(dim=(1, 2)).clamp_min(1e-12)).clamp(max=1.0)
        r[idx] = r[idx] + dr * scale[:, None, None]
        v[idx], dt[idx], alpha[idx], n_pos[idx] = vi, dti, ai, ni

        # Complexes drifting apart will not come back: stop spending steps on them
        if step % 25 == 0:
            ri = r[idx]
            c1 = (ri[:, f1] * masses[f1][:, None]).sum(1) / masses[f1].sum()
            c2 = (ri[:, f2] * masses[f2][:, None]).sum(1) / masses[f2].sum()
            active[idx[(c1 - c2).norm(dim=1) > FLYAWAY_COM_DIST_A]] = False
        if step % 500 == 0 and step > 0:
            print(f"      step {step:5d}: {int(active.sum())} of {bsz} still optimizing")

    return r.cpu().numpy(), converged.cpu().numpy()


def bfgs_refine(coords: np.ndarray, calc: MACECalculator, symbols: List[str]) -> Tuple[np.ndarray, float, bool]:
    """Tight one-by-one refinement in float64 with a wall-time cap; returns coords, energy (eV), converged."""
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = calc
    opt = BFGS(atoms, maxstep=0.1, logfile=None)
    t_start, steps = time.time(), 0
    converged = False
    while steps < TIGHT_MAX_STEPS and time.time() - t_start < MAX_SECONDS_PER_ISOMER:
        opt.run(fmax=TIGHT_FMAX_EV_A, steps=50)
        steps += 50
        if np.linalg.norm(atoms.get_forces(), axis=1).max() <= TIGHT_FMAX_EV_A:
            converged = True
            break
    return atoms.get_positions().copy(), float(atoms.get_potential_energy()), converged


# ==============================================================================
# 5. HARMONIC ANALYSIS (ANALYTIC MACE HESSIAN)
# ==============================================================================
def harmonic_analysis(coords: np.ndarray, calc: MACECalculator, symbols: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Frequencies (cm^-1, negative = imaginary), Cartesian modes and ZPVE (eV)
    from the analytic Hessian, with translations and rotations projected out.
    """
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = calc
    n = len(atoms)
    hess = np.asarray(calc.get_hessian(atoms)).reshape(3 * n, 3 * n)
    hess = 0.5 * (hess + hess.T)
    masses = atoms.get_masses()  # standard atomic weights, as ORCA uses for frequencies
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
    hess_mw = hess * np.outer(inv_sqrt_m, inv_sqrt_m)

    # Translation/rotation vectors in mass-weighted coordinates
    rel = coords - center_of_mass(coords, masses)
    sqrt_m = np.sqrt(masses)
    tr = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = sqrt_m
        tr.append(t.ravel())
        axis = np.zeros(3)
        axis[k] = 1.0
        tr.append((np.cross(axis, rel) * sqrt_m[:, None]).ravel())
    u, s, _ = np.linalg.svd(np.array(tr).T, full_matrices=True)
    n_tr = int(np.sum(s > 1e-8 * s.max()))
    internal = u[:, n_tr:]  # orthonormal complement of the TR space

    evals, evecs = np.linalg.eigh(internal.T @ hess_mw @ internal)
    freqs = np.sign(evals) * HESSIAN_TO_CM1 * np.sqrt(np.abs(evals))
    modes = (internal @ evecs).T * inv_sqrt_m  # back to Cartesian displacements
    modes = modes.reshape(-1, n, 3)
    zpve_ev = 0.5 * float(np.sum(freqs[freqs > 0.0])) / EV_TO_CM1
    return freqs, modes, zpve_ev


# ==============================================================================
# 6. PHASES
# ==============================================================================
def phase1_monomers(calc64: MACECalculator) -> Dict[str, Tuple[np.ndarray, float]]:
    """Optimizes trans-HCOOH, cis-HCOOH and H2S; checks trans-HCOOH frequencies."""
    print("\n--- PHASE 1: MONOMERS (float64) ---")
    seed = np.array(INITIAL_SEEDS["seed0"])
    hcooh_symbols = [SYMBOLS[i] for i in FRAGMENT_1_INDICES]
    h2s_symbols = [SYMBOLS[i] for i in FRAGMENT_2_INDICES]
    monomers: Dict[str, Tuple[np.ndarray, float]] = {}

    trans_xyz, trans_e, ok_t = bfgs_refine(seed[FRAGMENT_1_INDICES], calc64, hcooh_symbols)
    # cis-HCOOH: rotate the hydroxyl H by 180 deg about the C-O(H) bond
    cis_start = trans_xyz.copy()
    axis = (cis_start[1] - cis_start[0]) / np.linalg.norm(cis_start[1] - cis_start[0])
    cis_start[2] = cis_start[1] + Rotation.from_rotvec(math.pi * axis).apply(cis_start[2] - cis_start[1])
    cis_xyz, cis_e, ok_c = bfgs_refine(cis_start, calc64, hcooh_symbols)
    h2s_xyz, h2s_e, ok_h = bfgs_refine(seed[FRAGMENT_2_INDICES], calc64, h2s_symbols)
    if not (ok_t and ok_c and ok_h):
        raise SystemExit("CRITICAL ERROR: a monomer optimization did not converge.")

    pad = np.zeros((3, 3))
    for name, xyz in (("trans", trans_xyz), ("cis", cis_xyz)):
        if hcooh_conformer(np.vstack([xyz, pad])) != name:
            raise SystemExit(f"CRITICAL ERROR: the {name}-HCOOH optimization changed conformer.")

    freqs, _, _ = harmonic_analysis(trans_xyz, calc64, hcooh_symbols)
    if freqs.min() < IMAG_FREQ_TOL_CM1 or not 3000.0 < freqs.max() < 4300.0:
        raise SystemExit(f"CRITICAL ERROR: implausible trans-HCOOH frequencies {np.round(freqs)}; check the Hessian.")

    monomers["trans"] = (trans_xyz, trans_e)
    monomers["cis"] = (cis_xyz, cis_e)
    monomers["H2S"] = (h2s_xyz, h2s_e)
    print(f"  trans-HCOOH {trans_e / EV_PER_HARTREE:.8f} Eh | cis-HCOOH {cis_e / EV_PER_HARTREE:.8f} Eh "
          f"(+{(cis_e - trans_e) * EV_TO_KCAL:.2f} kcal/mol) | H2S {h2s_e / EV_PER_HARTREE:.8f} Eh")
    print(f"  trans-HCOOH frequencies: {freqs.min():.0f} ... {freqs.max():.0f} cm-1 (sanity check passed)")
    return monomers


def place_h2s(hcooh: np.ndarray, h2s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random orientation and direction of H2S around HCOOH at a random closest-contact scale."""
    hc = hcooh - center_of_mass(hcooh, MASSES[FRAGMENT_1_INDICES])
    hs = Rotation.random(random_state=rng).apply(h2s - center_of_mass(h2s, MASSES[FRAGMENT_2_INDICES]))
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    target = rng.uniform(*CONTACT_SCALE_RANGE)
    r_sum = np.array([[VDW_RADII_A[SYMBOLS[i]] + VDW_RADII_A[SYMBOLS[j]] for j in FRAGMENT_2_INDICES]
                      for i in FRAGMENT_1_INDICES])
    previous = None
    for dist in np.arange(15.0, 0.0, -0.05):
        placed = hs + dist * direction
        ratio = (np.linalg.norm(hc[:, None, :] - placed[None, :, :], axis=-1) / r_sum).min()
        if ratio <= target:
            break
        previous = placed
    return np.vstack([hc, previous if previous is not None else placed])


def phase2_starts(monomers: Dict[str, Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, List[str]]:
    print("\n--- PHASE 2: STARTING STRUCTURES ---")
    rng = np.random.default_rng(RANDOM_SEED)
    starts = [np.array(xyz) for xyz in INITIAL_SEEDS.values()]
    origins = list(INITIAL_SEEDS.keys())
    for conf in ("trans", "cis"):
        for k in range(N_STARTS_PER_CONFORMER):
            starts.append(place_h2s(monomers[conf][0], monomers["H2S"][0], rng))
            origins.append(f"random-{conf}-{k:04d}")
    print(f"  {len(starts)} starts: {len(INITIAL_SEEDS)} seeds + {N_STARTS_PER_CONFORMER} random placements x 2 conformers")
    return np.array(starts), origins


def phase3_coarse(ev32: BatchedMace, starts: np.ndarray, origins: List[str], ref_bonds: frozenset) -> List[Minimum]:
    print(f"\n--- PHASE 3: BATCHED COARSE SEARCH (float32, {DEVICE}, fmax {COARSE_FMAX_EV_A} eV/A) ---")
    finals, conv = [], []
    for k in range(0, len(starts), BATCH_SIZE):
        t0 = time.time()
        xyz, ok = batched_fire(ev32, starts[k:k + BATCH_SIZE], COARSE_FMAX_EV_A, COARSE_MAX_STEPS)
        finals.append(xyz)
        conv.append(ok)
        print(f"  batch {k // BATCH_SIZE + 1}: {int(ok.sum())}/{len(ok)} converged in {time.time() - t0:.0f} s")
    finals_arr, conv_arr = np.concatenate(finals), np.concatenate(conv)
    energies, _ = ev32.evaluate_chunked(finals_arr)

    pool: List[Minimum] = []
    rejected: Dict[str, int] = {}
    for xyz, e, origin in zip(finals_arr, energies, origins):
        valid, reason = is_valid_complex(xyz, ref_bonds)
        if valid:
            pool.append(annotate(Minimum(coords=xyz, energy_ev=float(e), origin=origin)))
        else:
            rejected[reason] = rejected.get(reason, 0) + 1
    print(f"  {int(conv_arr.sum())}/{len(conv_arr)} converged; rejected: {rejected or 'none'}")
    return deduplicate(pool, COARSE_ENERGY_TOL_KCAL, COARSE_RMSD_TOL_A, use_rot=False, stage="coarse")


def phase4_refine(coarse: List[Minimum], calc64: MACECalculator, ref_bonds: frozenset) -> List[Minimum]:
    print(f"\n--- PHASE 4: TIGHT REFINEMENT (float64 BFGS, fmax {TIGHT_FMAX_EV_A} eV/A, "
          f"<= {MAX_SECONDS_PER_ISOMER / 60:.0f} min each) ---")
    refined: List[Minimum] = []
    t0 = time.time()
    for k, m in enumerate(coarse):
        xyz, e, ok = bfgs_refine(m.coords, calc64, SYMBOLS)
        valid, reason = is_valid_complex(xyz, ref_bonds)
        if not ok or not valid:
            print(f"  [SKIP] {m.origin}: {'not converged' if not ok else reason}")
            continue
        refined.append(annotate(Minimum(coords=xyz, energy_ev=e, origin=m.origin, hits=m.hits)))
        if (k + 1) % 25 == 0:
            print(f"  refined {k + 1}/{len(coarse)} ({time.time() - t0:.0f} s)")
    return deduplicate(refined, ENERGY_TOL_KCAL, RMSD_TOL_A, use_rot=True, stage="final")


def phase6_verify(isomers: List[Minimum], calc64: MACECalculator, ref_bonds: frozenset) -> List[Minimum]:
    """Frequencies for every isomer; one displacement-and-reoptimize round for saddle points."""
    print("\n--- PHASE 6: MINIMUM VERIFICATION (analytic Hessian) ---")
    for m in isomers:
        m.freqs_cm1, m.modes, m.zpve_ev = harmonic_analysis(m.coords, calc64, SYMBOLS)
    saddles = [m for m in isomers if m.num_imag > 0]
    if not saddles:
        print(f"  all {len(isomers)} isomers are true minima")
        return isomers

    print(f"  {len(saddles)} isomer(s) with imaginary modes: displacing along the lowest mode and re-optimizing")
    pool = [m for m in isomers if m.num_imag == 0]
    for m in saddles:
        mode = m.modes[0] / np.linalg.norm(m.modes[0], axis=1).max()
        products: List[Minimum] = []
        for sign in (1.0, -1.0):
            xyz, e, ok = bfgs_refine(m.coords + sign * SADDLE_DISPLACEMENT_A * mode, calc64, SYMBOLS)
            valid, _ = is_valid_complex(xyz, ref_bonds)
            if ok and valid:
                new = annotate(Minimum(coords=xyz, energy_ev=e, origin=f"{m.origin}-displaced", hits=0))
                new.freqs_cm1, new.modes, new.zpve_ev = harmonic_analysis(xyz, calc64, SYMBOLS)
                products.append(new)
        if products:
            products[0].hits += m.hits  # the starts that reached the saddle lead to its minima
            pool.extend(products)
        else:
            pool.append(m)  # could not be resolved: kept and flagged in the inventory
    isomers = deduplicate(pool, ENERGY_TOL_KCAL, RMSD_TOL_A, use_rot=True, stage="after saddle displacement")
    left = sum(1 for m in isomers if m.num_imag > 0)
    print(f"  {left} isomer(s) still have imaginary modes (flagged in the inventory)")
    return isomers


def phase7_export(isomers: List[Minimum], monomers: Optional[Dict[str, Tuple[np.ndarray, float]]], work_dir: Path) -> None:
    print("\n--- PHASE 7: ISOMER INVENTORY ---")
    if not isomers:
        raise SystemExit("No valid isomers survived Phases 3-6.")
    isomers = sorted(isomers, key=lambda m: m.energy_ev)
    e_min = isomers[0].energy_ev
    isomers = [m for m in isomers if (m.energy_ev - e_min) * EV_TO_KCAL <= FINAL_MAXEN_KCAL + 1e-9]
    e0_min = min(m.energy_ev + m.zpve_ev for m in isomers)
    for m in isomers:
        if m.binding_kcal is None and monomers is not None:
            m.binding_kcal = -(m.energy_ev - monomers[m.conformer][1] - monomers["H2S"][1]) * EV_TO_KCAL

    csv_path = work_dir / f"{MOL_NAME}_{LEVEL_TAG}_Inventory.csv"
    ens_path = work_dir / f"{MOL_NAME}_{LEVEL_TAG}.finalensemble.xyz"
    iso_dir = work_dir / f"{MOL_NAME}_{LEVEL_TAG}_isomers"
    iso_dir.mkdir(exist_ok=True)
    for old in list(iso_dir.glob("Isomer_*.xyz")) + list(iso_dir.glob("Isomer_*.inp")):
        old.unlink()

    rows = []
    for k, m in enumerate(isomers, start=1):
        m.label = f"Isomer_{k:02d}"
        rel_kcal = (m.energy_ev - e_min) * EV_TO_KCAL
        lowest = m.lowest_vib_cm1
        rows.append({
            "Isomer_ID": m.label,
            "Energy_Eh": f"{m.energy_ev / EV_PER_HARTREE:.8f}",
            "Rel_E_kcal_mol": f"{rel_kcal:.4f}",
            "Rel_E_cm-1": f"{rel_kcal * KCAL_TO_CM1:.2f}",
            "Rel_E0_kcal_mol": f"{(m.energy_ev + m.zpve_ev - e0_min) * EV_TO_KCAL:.4f}",
            "ZPVE_kcal_mol": f"{m.zpve_ev * EV_TO_KCAL:.4f}",
            "De_kcal_mol": f"{m.binding_kcal:.3f}" if m.binding_kcal is not None else "N/A",
            "HCOOH_conformer": m.conformer,
            "HBond_motif": m.motif,
            "Rot_A_MHz": f"{m.rot_mhz[0]:.3f}",
            "Rot_B_MHz": f"{m.rot_mhz[1]:.3f}",
            "Rot_C_MHz": f"{m.rot_mhz[2]:.3f}",
            "Kappa": f"{m.kappa:.5f}",
            "Pcc_amuA2": f"{m.pcc_amu_a2:.4f}",
            "Lowest_Vib_cm-1": f"{lowest:.1f}" if lowest is not None else "N/A",
            "Imag_Modes": m.num_imag,
            "Family_Size": m.family_size,
            "Family_Span_cm-1": f"{m.family_span_cm1:.1f}",
            "Hits": m.hits,
            "Found_By": m.origin,
        })
        comment = (f"[{m.label}] [{m.energy_ev / EV_PER_HARTREE:.8f} Eh] [{rel_kcal:.3f} kcal/mol] "
                   f"[{m.conformer}-HCOOH] [{m.motif}] "
                   f"[A={m.rot_mhz[0]:.2f} B={m.rot_mhz[1]:.2f} C={m.rot_mhz[2]:.2f} MHz] [hits {m.hits}]"
                   + (f" [family of {m.family_size}, span {m.family_span_cm1:.1f} cm-1]" if m.family_size > 1 else ""))
        coord_lines = "".join(f"  {s:<2s}  {x:16.10f}  {y:16.10f}  {z:16.10f}\n" for s, (x, y, z) in zip(SYMBOLS, m.coords))
        (iso_dir / f"{m.label}.xyz").write_text(f"{len(SYMBOLS)}\n{comment}\n{coord_lines}", encoding="utf-8")
        frag_lines = "".join(
            f"  {s}({1 if i in FRAGMENT_1_INDICES else 2})  {x:16.10f}  {y:16.10f}  {z:16.10f}\n"
            for i, (s, (x, y, z)) in enumerate(zip(SYMBOLS, m.coords))
        )
        next_base = f"{m.label}_{ORCA_NEXT_TAG}"
        (iso_dir / f"{next_base}.inp").write_text(
            f"# {m.label} from {LEVEL_TAG}: next-level refinement (owner-run; not started by the search)\n"
            f"%pal\n  nprocs {ORCA_NEXT_NPROCS}\nend\n%maxcore {ORCA_NEXT_MAXCORE_MB}\n\n"
            f"{ORCA_NEXT_BLOCK}"
            # No %output Pickettname: in parallel it hangs orca_prop_mpi on this PC (ORCA 6.1.1,
            # MS-MPI); rotational constants and dipole components are printed in the .out instead
            "%output\n  Print[P_AtCharges_M] 1\nend\n"
            f'%base "{next_base}"\n'
            f"* xyz {CHARGE} {MULTIPLICITY}\n{frag_lines}*\n"
            "%frag\n  Definition\n    1 { 0:4 } end\n    2 { 5:7 } end\n  end\nend\n",
            encoding="utf-8",
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(ens_path, "w", encoding="utf-8") as fh:
        for m in isomers:
            fh.write((iso_dir / f"{m.label}.xyz").read_text(encoding="utf-8"))

    print("----------------------------------------------------------------------------------------------")
    print(f"{'Isomer':<11}{'dE kcal/mol':>12}{'De kcal/mol':>13}  {'HCOOH':<6} {'A / B / C (MHz)':<27}{'Imag':>5}{'Fam':>4}{'Hits':>6}  Motif")
    print("----------------------------------------------------------------------------------------------")
    for m, row in zip(isomers, rows):
        de_text = f"{m.binding_kcal:13.3f}" if m.binding_kcal is not None else f"{'N/A':>13}"
        print(f"{m.label:<11}{float(row['Rel_E_kcal_mol']):>12.3f}{de_text}  {m.conformer:<6} "
              f"{m.rot_mhz[0]:8.1f} / {m.rot_mhz[1]:6.1f} / {m.rot_mhz[2]:6.1f} {m.num_imag:>5}{m.family_size:>4}{m.hits:>6}  {m.motif}")
    print("----------------------------------------------------------------------------------------------")
    print(f"  [EXPORT] CSV Table        : {csv_path}")
    print(f"  [EXPORT] Ensemble XYZ     : {ens_path}")
    print(f"  [EXPORT] Isomer XYZ + ORCA: {iso_dir}  (ORCA inputs are NOT run by this script)")


# ==============================================================================
# 7. REBUILD AN EXISTING INVENTORY (no model, no new search)
# ==============================================================================
def rebuild_inventory(work_dir: Path) -> None:
    """
    Re-runs the geometry-first deduplication, the floppy-family merge and the
    export on the isomers saved by an earlier search in work_dir. Frequencies,
    ZPVE and binding energies come from the saved CSV. The earlier files are
    moved to work_dir/before_rebuild once and always read from there, so a
    rebuild can be repeated.
    """
    print("======================================================================")
    print(f"REBUILD {LEVEL_TAG} INVENTORY: {work_dir}  (no model, no new search)")
    print("======================================================================")
    csv_name = f"{MOL_NAME}_{LEVEL_TAG}_Inventory.csv"
    iso_name = f"{MOL_NAME}_{LEVEL_TAG}_isomers"
    names = [csv_name, f"{MOL_NAME}_{LEVEL_TAG}.finalensemble.xyz", iso_name]
    raw_dir = work_dir / "before_rebuild"
    if raw_dir.exists():
        print(f"  reading the earlier search results kept in {raw_dir.name}")
    else:
        if not (work_dir / csv_name).is_file() or not (work_dir / iso_name).is_dir():
            raise SystemExit(f"CRITICAL ERROR: no saved inventory found in {work_dir}")
        raw_dir.mkdir()
        for name in names:
            if (work_dir / name).exists():
                shutil.move(str(work_dir / name), str(raw_dir / name))
        print(f"  earlier results moved to {raw_dir.name}")

    isomers: List[Minimum] = []
    with open(raw_dir / csv_name, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lines = (raw_dir / iso_name / f"{row['Isomer_ID']}.xyz").read_text(encoding="utf-8").splitlines()
            coords = np.array([[float(v) for v in line.split()[1:4]] for line in lines[2:2 + len(SYMBOLS)]])
            m = annotate(Minimum(coords=coords, energy_ev=float(row["Energy_Eh"]) * EV_PER_HARTREE,
                                 origin=row["Found_By"], hits=int(row["Hits"])))
            m.zpve_ev = float(row["ZPVE_kcal_mol"]) / EV_TO_KCAL
            m.binding_kcal = None if row["De_kcal_mol"] == "N/A" else float(row["De_kcal_mol"])
            m.stored_imag = int(row["Imag_Modes"])
            m.stored_lowest_vib = None if row["Lowest_Vib_cm-1"] == "N/A" else float(row["Lowest_Vib_cm-1"])
            m.family_size = int(row.get("Family_Size") or 1)
            m.family_span_cm1 = float(row.get("Family_Span_cm-1") or 0.0)
            isomers.append(m)
    print(f"  {len(isomers)} saved isomers read")
    isomers = deduplicate(isomers, ENERGY_TOL_KCAL, RMSD_TOL_A, use_rot=True, stage="geometry-first")
    isomers = merge_families(isomers)
    phase7_export(isomers, None, work_dir)
    print(f"\n[DONE] inventory rebuilt with {len(isomers)} isomers. No model was loaded and no ORCA job was run.")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="In-process MACE-OFF24m isomer search for HCOOH-H2S.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the inventory in the current folder from the isomers saved by an earlier "
        "search (geometry-first dedup + floppy-family merge; no model, no new search)",
    )
    if parser.parse_args().rebuild:
        rebuild_inventory(Path.cwd())
        return

    t_start = time.time()
    torch.set_num_threads(MAX_THREADS)
    try:
        torch.set_num_interop_threads(MAX_THREADS)
    except RuntimeError:
        pass  # only settable before the first parallel operation
    work_dir = Path.cwd()
    print("======================================================================")
    print(f"IN-PROCESS {LEVEL_TAG} ISOMER SEARCH: {MOL_NAME}  (device {DEVICE}, {MAX_THREADS} CPU threads)")
    print("======================================================================")

    print("\n--- PHASE 0: MODEL ---")
    calc32, calc64 = load_calculators()
    seed0 = np.array(INITIAL_SEEDS["seed0"])
    ev32, ev64 = BatchedMace(calc32, seed0), BatchedMace(calc64, seed0)
    self_check(ev64, calc64, seed0, tol_e=1e-6, tol_f=1e-5)
    # float32 total energies (~ -16000 eV) are only resolved to ~0.002 eV
    self_check(ev32, calc32, seed0, tol_e=1e-2, tol_f=1e-2)
    ref_bonds = bond_set(seed0)

    monomers = phase1_monomers(calc64)
    starts, origins = phase2_starts(monomers)
    coarse = phase3_coarse(ev32, starts, origins, ref_bonds)
    isomers = phase4_refine(coarse, calc64, ref_bonds)
    isomers = phase6_verify(isomers, calc64, ref_bonds)
    isomers = merge_families(isomers)
    phase7_export(isomers, monomers, work_dir)
    print(f"\n[DONE] {len(isomers)} isomers in {(time.time() - t_start) / 60:.1f} min. No ORCA job was run.")


if __name__ == "__main__":
    main()
