#!/usr/bin/env python3
"""
================================================================================
CoChem-GOAT / TOPOS: reusable GOAT isomer search, ORCA 6.1.1 + OPI 2.0 (GFN2-xTB)
File: isomer_search/goat_isomer_search.py
================================================================================
1. PURPOSE:
   Finds, deduplicates and organizes the low-energy isomers of a small
   molecule or van der Waals complex with ORCA's GOAT global optimizer at the
   GFN2-xTB level. Derived from the tested hcooh-hsh-goat-xtb.py (2026-10-24
   run, no crash); the chemistry and every safety behaviour are unchanged,
   only the molecule-specific parts moved out of the source:
   - seeds are read from .xyz files (no geometry in this script);
   - name, charge, multiplicity and energy window come from the command
     line or a job.toml file in the job folder;
   - fragments (monomers) are the connected components of the first seed's
     bond graph;
   - symmetry-equivalent atoms (terminal atoms of the same element on the
     same centre: the two H of H2S, the three H of a CH3 group) are found
     automatically and used in the RMSD comparison.
   The script STOPS once the isomers are collected and organized: no DFT,
   and every per-structure job is hard-capped at 10 minutes of wall time.

2. METHOD:
   ! NATIVE-GFN2-XTB: ORCA's own implementation of GFN2-xTB, so no separate
   xtb program is involved. Never put ExtOpt (external g-xTB) here: one
   cmd + python + xtb launch per gradient crashed the PC.

3. MECHANICS & SCIENTIFIC CHOICES:
   - Plain GOAT (not GOAT-EXPLORE): covalent bonds stay fixed, so every
     monomer stays intact and GOAT samples how they pack and their
     conformers. Every seed must have the bonding pattern of the first seed;
     a different molecule is refused (seeds/ mode) or run as its own group
     (folder mode), never silently mixed in.
   - No GFNUphill: GFN-FF exists only in the external xtb program (one launch
     per step). AutoWall true keeps the fragments from drifting apart.
   - Deduplication is deliberately conservative (dE <= 0.10 kcal/mol AND
     RMSD <= 0.125 A, the GOAT defaults). RMSD is minimized over rotations,
     reflections and the equivalent-atom permutations. Borderline pairs are
     NOT merged; they go to near_duplicates.csv for a human to judge.
   - Machine load (read before changing GOAT_NPROCS): ORCA runs every step by
     starting module programs (200-270 MB each). For a small molecule a step
     is ~0.02 s of chemistry and ~0.5 s of program start-up, so a parallel
     GOAT worker is a tight program-launch loop; 8 MPI workers crashed a
     desktop PC. GOAT therefore runs serially (nprocs 1, workers take turns,
     no mpiexec) with MiniPrint; every ORCA job runs alone at below-normal
     priority; a machine-wide lock file stops a second ORCA driver (another
     job folder, the microwave script) from starting while one is running.
     Measured: ~5.7 ORCA program starts/s, max 3 alive, 8 % CPU.

4. WORKFLOW:
   Phase 0  Pre-flight: seeds, topology, ORCA via OPI, one test gradient.
   Phase 1  Quench: TightOpt of each seed (keeps every seed basin).
   Phase 2  GOAT: one topology-preserving GOAT search per quenched seed.
   Phase 3  Pool: merge everything, reject broken or dissociated structures,
            remove exact duplicates.
   Phase 4  Characterize: TightOpt + NumFreq per unique candidate, one at a
            time, each killed after PER_ISOMER_TIMEOUT_S (10 min). A saddle
            point is pushed off along its imaginary mode both ways and
            re-optimized, so only true minima remain.
   Phase 5  Organize: final dedup, sort by energy, write isomer files, table,
            near-duplicate list and an audit of every structure's fate.

5. JOB FOLDER
   Which folder: a copy of this script placed in a job folder (next to
   .xyz files, seeds/ or job.toml) always works on its own folder; the
   master copy in isomer_search/ works on the current folder.
   Two layouts:
   a) FOLDER MODE (no seeds/ subfolder, no --seeds): every *.xyz in the job
      folder is a target. Targets are sorted into groups, one per molecule
      (same atoms and bonds; a file listing the atoms in another order is
      renumbered to match). Each group is searched on its own, one after
      another, with all its output in results/<group>/ (the group is named
      after its fragments, e.g. results/CH2O2-N2O/). results/run_log.txt
      and results/targets.csv show which file went where.
   b) SEEDS MODE: <job>/seeds/*.xyz (or --seeds files) are the seeds of ONE
      molecule; output goes into the job folder itself (layout below).
   In both, a file is single- or multi-frame; each frame is one seed named
   after its file (multi-frame: <stem>_f00, <stem>_f01, ...).
   <job>/job.toml          optional settings, e.g.
                               name = "HCOOH-H2S cis"
                               charge = 0
                               multiplicity = 1
                               window_kcal = 12.0      # GOAT MaxEn and final window
                               goat_min_iter = 3
                               goat_max_iter = 16
   Output, written into the job folder (folder mode: into results/<group>/):
   isomers/isomer-00.xyz ...      one file per unique isomer, 00 = lowest
   isomers/<name>_XTB2_isomers.xyz / .csv   all isomers, table
   isomers/near_duplicates.csv    pairs to compare by hand
   isomers/rejected_structures.xyz  broken/dissociated structures, TS + reason
   isomers/structure_audit.csv    every raw and characterized structure -> fate
   isomers/README.txt             how to read the files
   jobs/                          every ORCA job in its own folder
   run_log.txt, run_manifest.json, all_raw_structures.xyz

6. USAGE
   python goat_isomer_search.py                 (a copy inside the job folder) full run
   python isomer_search/goat_isomer_search.py   (from inside the job folder) full run
   ... --dry-run          write every ORCA input, run nothing
   ... --quick            setup test: first seed, 1 GOAT iteration, in quick_test/
   ... --seeds a.xyz b.xyz  seed files of one molecule (seeds mode)
   ... --name X --charge 0 --mult 1 --window 12   override job.toml
   ... --job-dir PATH     job folder other than the default (see 5.)
   ... --reuse-only       redo the analysis/export from finished jobs; never starts ORCA
   Re-running resumes: a job whose input is unchanged (the input embeds a
   fingerprint of its start geometry) and whose output says ORCA TERMINATED
   NORMALLY is reused instead of recomputed.

7. REQUIREMENTS:
   ORCA 6.1.1, Python >= 3.11 (tomllib), numpy, orca-pi 2.0 (import opi).
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from opi.core import Calculator
from opi.execution.core import Runner
from opi.input.blocks import BlockGeom, BlockGoat
from opi.input.simple_keywords import Goat, Opt, OutputControl, Sqm, Task
from opi.input.structures import XyzFile
from opi.lib.orca_binary import OrcaBinary
from opi.output.core import Output


# ==============================================================================
# 0. CENTRALIZED CONFIGURATION (defaults = the values of the tested run)
# ==============================================================================
TOOL_VERSION: str = "1.1 (2026-10-24, folder mode; from hcooh-hsh-goat-xtb.py)"
METHOD_TAG: str = "XTB2"
RESULTS_DIRNAME: str = "results"        # folder mode: results/<group>/ per molecule
ATOM_MATCH_MAX_STEPS: int = 200_000     # cap on the atom-order matching search

# Machine load (see "Machine load" above). GOAT_NPROCS > 1 turns every GOAT
# worker into a parallel mpiexec program-launch loop: keep it at 1.
GOAT_NPROCS: int = 1                    # serial GOAT: one worker at a time, no MPI
GOAT_NWORKERS: int = 4                  # minimum GOAT allows (one per temperature)
MAXCORE_MB: int = 1000                  # xTB needs very little memory
LOCK_FILE: Path = Path(tempfile.gettempdir()) / "orca_one_job_at_a_time.lock"

# GOAT search (per seed); iterations and MaxEn can be changed in job.toml
GOAT_TIMEOUT_S: int = 6 * 3600          # safety net for one (serial) GOAT run
GOAT_STALL_S: int = 900                 # no file written for 15 min = hung GOAT
STALL_CHECK_EVERY_S: int = 60           # how often the stall watchdog looks

# Per-structure jobs (quench, characterization): the 10-minute rule
PER_ISOMER_TIMEOUT_S: int = 600
OPT_MAXITER: int = 300                  # flat vdW surfaces need many steps

# Isomer organization
DEDUP_ENERGY_TOL_KCAL: float = 0.10     # GOAT EnDiff default
DEDUP_RMSD_TOL_A: float = 0.125         # GOAT RMSD default
NEAR_DUP_ENERGY_KCAL: float = 0.50      # flag pairs closer than this ...
NEAR_DUP_ROT_REL: float = 0.02          # ... with A/B/C within 2 % ...
NEAR_DUP_RMSD_A: float = 0.50           # ... or RMSD below 0.5 A
DUPLICATE_SEED_RMSD_A: float = 1e-3     # seeds closer than this are copies: skipped
DISSOCIATION_CONTACT_A: float = 4.5     # fragments farther apart than this = separated
BOND_SCALE: float = 1.25                # bond if d < 1.25 * (r_cov,i + r_cov,j)
IMAG_NOISE_CM1: float = 20.0            # |imag| below this is numerical noise
SADDLE_STEP_A: float = 0.25             # largest atom shift when stepping off a saddle point
MAX_PERMUTATIONS: int = 720             # cap on equivalent-atom permutations in the RMSD

# Physical constants
HARTREE_TO_KCAL: float = 627.5094740631
HARTREE_TO_CM1: float = 219474.63136320
ROT_CONST_MHZ_AMU_A2: float = 505379.0094            # B = 505379 / I(amu A^2)
ISOTOPE_MASSES: dict[str, float] = {                 # most abundant isotopes (AME2016)
    "H": 1.00782503223, "He": 4.00260325413, "Li": 7.0160034366, "Be": 9.012183065,
    "B": 11.00930536, "C": 12.0, "N": 14.00307400443, "O": 15.99491461957,
    "F": 18.99840316273, "Ne": 19.9924401762, "Na": 22.9897692820, "Mg": 23.985041697,
    "Al": 26.98153853, "Si": 27.97692653465, "P": 30.97376199842, "S": 31.9720711744,
    "Cl": 34.968852682, "Ar": 39.9623831237, "K": 38.9637064864, "Ca": 39.962590863,
    "Ge": 73.921177761, "As": 74.92159457, "Se": 79.9165218, "Br": 78.9183376,
    "Kr": 83.9114977282, "I": 126.9044719, "Xe": 131.9041550856,
}
COVALENT_RADII_A: dict[str, float] = {               # Cordero et al. 2008 (C: sp3)
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "I": 1.39, "Xe": 1.40,
}


# ==============================================================================
# 1. DATA STRUCTURES, LOGGING & PROCESS CONTROL
# ==============================================================================
@dataclass(eq=False)                    # identity equality: fields hold numpy arrays
class Structure3D:
    """One geometry with its energy and bookkeeping for the isomer table."""

    symbols: list[str]
    coords: np.ndarray                  # (N, 3) Angstrom
    energy_eh: float
    sources: list[str] = field(default_factory=list)
    n_raw: int = 1
    status: str = "not characterized"
    opt_converged: bool | None = None
    freqs_cm1: list[float] = field(default_factory=list)
    zpe_eh: float | None = None
    char_job: str = ""
    label: str = ""


@dataclass
class Seed:
    """One starting geometry read from a seed file."""

    name: str
    symbols: list[str]
    coords: np.ndarray
    path: Path
    frame: int
    note: str = ""                      # e.g. atoms renumbered to the group's order


@dataclass
class JobSettings:
    """Per-job settings: defaults, then job.toml, then the command line."""

    name: str
    charge: int = 0
    multiplicity: int = 1
    window_kcal: float = 12.0           # GOAT MaxEn and the final isomer window
    goat_min_iter: int = 3
    goat_max_iter: int = 16


@dataclass
class Topology:
    """
    Bond graph of the reference seed, its fragments (connected components)
    and the atom permutations that map the molecule onto itself (used to
    minimize the RMSD between two structures of the same molecule).
    """

    symbols: list[str]
    bonds: set[tuple[int, int]]
    fragments: list[list[int]]
    equivalent_groups: list[list[int]]
    permutations: list[np.ndarray]
    permutation_note: str = ""

    @classmethod
    def from_reference(cls, symbols: list[str], coords: np.ndarray) -> Topology:
        unknown = sorted(e for e in set(symbols) if e not in COVALENT_RADII_A or e not in ISOTOPE_MASSES)
        if unknown:
            raise SystemExit(f"Element(s) {', '.join(unknown)} missing from COVALENT_RADII_A/ISOTOPE_MASSES.")
        bonds = bonded_pairs(symbols, coords)
        n = len(symbols)
        neighbors: list[set[int]] = [set() for _ in range(n)]
        for i, j in bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)

        # Fragments: connected components of the bond graph, by lowest atom index
        fragments, seen = [], set()
        for start in range(n):
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                a = stack.pop()
                if a not in comp:
                    comp.add(a)
                    stack.extend(neighbors[a] - comp)
            seen |= comp
            fragments.append(sorted(comp))

        # Equivalent atoms: terminal (degree-1) atoms of one element on the same
        # centre. Swapping them is an exact symmetry of the bond graph.
        groups = []
        for centre in range(n):
            terminal: dict[str, list[int]] = {}
            for j in sorted(neighbors[centre]):
                if len(neighbors[j]) == 1:
                    terminal.setdefault(symbols[j], []).append(j)
            groups += [g for g in terminal.values() if len(g) >= 2]

        n_full = math.prod(math.factorial(len(g)) for g in groups)
        perms, note = [], ""
        if n_full <= MAX_PERMUTATIONS:
            for choice in itertools.product(*(itertools.permutations(g) for g in groups)):
                p = np.arange(n)
                for g, image in zip(groups, choice):
                    p[g] = image
                perms.append(p)
        else:                           # too many: identity + single swaps only
            perms.append(np.arange(n))
            for g in groups:
                for i, j in itertools.combinations(g, 2):
                    p = np.arange(n)
                    p[i], p[j] = j, i
                    perms.append(p)
            note = f"{n_full} permutations > cap {MAX_PERMUTATIONS}: identity + pair swaps only"
        return cls(symbols, bonds, fragments, groups, perms, note)

    def fragment_name(self, k: int) -> str:
        """Hill formula of fragment k (C, H first if carbon is present)."""
        counts: dict[str, int] = {}
        for i in self.fragments[k]:
            counts[self.symbols[i]] = counts.get(self.symbols[i], 0) + 1
        first = ["C", "H"] if "C" in counts else []
        order = [e for e in first if e in counts] + sorted(e for e in counts if e not in first)
        return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)

    def describe(self) -> list[str]:
        def atoms(idx: list[int]) -> str:
            return " ".join(f"{self.symbols[i]}{i}" for i in idx)

        lines = [f"fragment {k + 1}: {self.fragment_name(k):<8s} {atoms(f)}" for k, f in enumerate(self.fragments)]
        for g in self.equivalent_groups:
            centre = next(b if a == g[0] else a for a, b in self.bonds if g[0] in (a, b))   # terminal: one partner
            lines.append(f"equivalent atoms: {'/'.join(f'{self.symbols[i]}{i}' for i in g)} "
                         f"(on {self.symbols[centre]}{centre})")
        if not self.equivalent_groups:
            lines.append("equivalent atoms: none")
        lines.append(f"RMSD permutations: {len(self.permutations)}"
                     + (f" ({self.permutation_note})" if self.permutation_note else ""))
        return lines


_LOG_FILE: Path | None = None
_RUNNING: set[subprocess.Popen] = set()     # at most one ORCA job at a time
_REUSE_ONLY: bool = False                   # --reuse-only: never start ORCA


def log(msg: str = "") -> None:
    """Prints a line and appends it to run_log.txt."""
    print(msg, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def kill_tree(proc: subprocess.Popen) -> None:
    """Kills ORCA and every module program it started."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def kill_all_running() -> None:
    for proc in list(_RUNNING):
        kill_tree(proc)


def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists (never signals or kills it)."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                     # access denied: it exists
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                       # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_orca_lock(what: str) -> None:
    """
    Machine-wide 'one ORCA driver at a time' lock (a small JSON file in the
    temp folder, shared with the microwave-prediction script). A lock left by
    a crashed run (its PID no longer exists) is taken over.
    """
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            pid = int(info.get("pid", -1))
        except (OSError, ValueError):
            info, pid = {}, -1
        if pid > 0 and pid != os.getpid() and pid_alive(pid):
            raise SystemExit(
                f"Another ORCA driver is running (PID {pid}: {info.get('what', '?')}, since "
                f"{info.get('started', '?')}).\nOne ORCA job at a time: wait for it to finish. "
                f"If nothing is running, delete {LOCK_FILE}")
        LOCK_FILE.unlink(missing_ok=True)
    with open(LOCK_FILE, "x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "what": what, "started": datetime.now().isoformat(timespec="seconds")}, f)


def release_orca_lock() -> None:
    try:
        if int(json.loads(LOCK_FILE.read_text(encoding="utf-8")).get("pid", -1)) == os.getpid():
            LOCK_FILE.unlink()
    except (OSError, ValueError):
        pass


# ==============================================================================
# 2. GEOMETRY TOOLS (connectivity, rotational constants, RMSD, contacts)
# ==============================================================================
def bonded_pairs(symbols: list[str], coords: np.ndarray) -> set[tuple[int, int]]:
    """Covalent bonds from distances: d < BOND_SCALE * (r_i + r_j)."""
    pairs = set()
    for i, j in itertools.combinations(range(len(symbols)), 2):
        cutoff = BOND_SCALE * (COVALENT_RADII_A[symbols[i]] + COVALENT_RADII_A[symbols[j]])
        if np.linalg.norm(coords[i] - coords[j]) < cutoff:
            pairs.add((i, j))
    return pairs


def intermolecular_contacts(coords: np.ndarray, topo: Topology) -> list[tuple[float, int, int]]:
    """All atom pairs in different fragments sorted by distance: (d, i, j)."""
    contacts = []
    for fa, fb in itertools.combinations(topo.fragments, 2):
        contacts += [(float(np.linalg.norm(coords[i] - coords[j])), i, j) for i in fa for j in fb]
    return sorted(contacts)


def fragments_separated(coords: np.ndarray, topo: Topology) -> float | None:
    """
    None if every fragment is within DISSOCIATION_CONTACT_A of the rest of the
    complex (the fragment contact graph is connected); otherwise the largest
    gap (A) that breaks the complex apart.
    """
    nf = len(topo.fragments)
    if nf < 2:
        return None
    gap = np.zeros((nf, nf))
    for a, b in itertools.combinations(range(nf), 2):
        d = min(np.linalg.norm(coords[i] - coords[j]) for i in topo.fragments[a] for j in topo.fragments[b])
        gap[a, b] = gap[b, a] = d
    linked, frontier = {0}, [0]
    while frontier:
        a = frontier.pop()
        for b in range(nf):
            if b not in linked and gap[a, b] <= DISSOCIATION_CONTACT_A:
                linked.add(b)
                frontier.append(b)
    if len(linked) == nf:
        return None
    return float(min(gap[a, b] for a in linked for b in range(nf) if b not in linked))


def com_distance(symbols: list[str], coords: np.ndarray, topo: Topology) -> float | None:
    """Distance between the two fragments' centres of mass (two-fragment complexes only)."""
    if len(topo.fragments) != 2:
        return None

    def com(idx: list[int]) -> np.ndarray:
        m = np.array([ISOTOPE_MASSES[symbols[i]] for i in idx])
        return (coords[idx] * m[:, None]).sum(axis=0) / m.sum()

    return float(np.linalg.norm(com(topo.fragments[0]) - com(topo.fragments[1])))


def rotational_constants_mhz(symbols: list[str], coords: np.ndarray) -> tuple[float, float, float]:
    """A >= B >= C in MHz for the most abundant isotopologue."""
    m = np.array([ISOTOPE_MASSES[s] for s in symbols])
    r = coords - (coords * m[:, None]).sum(axis=0) / m.sum()
    inertia = np.zeros((3, 3))
    for mi, (x, y, z) in zip(m, r):
        inertia += mi * np.array([[y * y + z * z, -x * y, -x * z],
                                  [-x * y, x * x + z * z, -y * z],
                                  [-x * z, -y * z, x * x + y * y]])
    moments = np.sort(np.linalg.eigvalsh(inertia))
    a, b, c = (ROT_CONST_MHZ_AMU_A2 / mom if mom > 1e-8 else 0.0 for mom in moments)
    return float(a), float(b), float(c)


def dihedral_deg(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def best_rmsd(a: np.ndarray, b: np.ndarray, permutations: list[np.ndarray]) -> float:
    """
    Minimum RMSD (A) over rotations, reflections and equivalent-atom permutations.
    The orthogonal Procrustes optimum including reflections is
    RMSD^2 = (|P|^2 + |Q|^2 - 2 * sum(singular values of P^T Q)) / N.
    Mirror images count as the same isomer (identical energy and spectrum).
    """
    q = b - b.mean(axis=0)
    best = np.inf
    for perm in permutations:
        p = a[perm] - a[perm].mean(axis=0)
        s = np.linalg.svd(p.T @ q, compute_uv=False)
        msd = ((p * p).sum() + (q * q).sum() - 2.0 * s.sum()) / len(a)
        best = min(best, float(np.sqrt(max(msd, 0.0))))
    return best


def max_rel_rot_diff(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(abs(x - y) / max(abs(y), 1e-6) for x, y in zip(a, b))


def screen_structure(s: Structure3D, topo: Topology) -> str | None:
    """Returns a rejection reason, or None if every monomer is intact and the complex bound."""
    bonds = bonded_pairs(s.symbols, s.coords)
    if bonds != topo.bonds:
        made = sorted(bonds - topo.bonds)
        broken = sorted(topo.bonds - bonds)
        parts = [f"+{s.symbols[i]}{i}-{s.symbols[j]}{j}" for i, j in made]
        parts += [f"-{s.symbols[i]}{i}-{s.symbols[j]}{j}" for i, j in broken]
        return "topology changed " + " ".join(parts)
    gap = fragments_separated(s.coords, topo)
    if gap is not None:
        return f"dissociated (fragments {gap:.2f} A apart)"
    return None


DedupLink = tuple[Structure3D, Structure3D, float, float]     # (merged, kept, dE kcal/mol, RMSD A)


def deduplicate(items: list[Structure3D], permutations: list[np.ndarray],
                energy_tol_kcal: float = DEDUP_ENERGY_TOL_KCAL, rmsd_tol_a: float = DEDUP_RMSD_TOL_A,
                rot_rel_tol: float | None = None) -> tuple[list[Structure3D], list[DedupLink]]:
    """
    Conservative merge: same isomer only if dE <= energy_tol_kcal AND
    best_rmsd <= rmsd_tol_a (AND, if rot_rel_tol is given, every rotational
    constant within that relative difference). The lowest-energy member is
    kept and inherits the provenance (sources, n_raw) of everything merged
    into it. Returns the unique structures and one link per merged structure.
    """
    unique: list[Structure3D] = []
    links: list[DedupLink] = []
    for item in sorted(items, key=lambda x: x.energy_eh):
        for u in unique:
            de = abs(item.energy_eh - u.energy_eh) * HARTREE_TO_KCAL
            if de > energy_tol_kcal:
                continue
            if rot_rel_tol is not None and max_rel_rot_diff(
                    rotational_constants_mhz(item.symbols, item.coords),
                    rotational_constants_mhz(u.symbols, u.coords)) > rot_rel_tol:
                continue
            rmsd = best_rmsd(item.coords, u.coords, permutations)
            if rmsd <= rmsd_tol_a:
                u.sources.extend(src for src in item.sources if src not in u.sources)
                u.n_raw += item.n_raw
                links.append((item, u, de, rmsd))
                break
        else:
            unique.append(item)
    return unique, links


# ==============================================================================
# 3. FILE PARSERS & WRITERS
# ==============================================================================
_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?")


def read_xyz_frames(path: Path) -> list[tuple[str, list[str], np.ndarray]]:
    """Reads single- or multi-frame .xyz files: [(comment, symbols, coords)]."""
    frames = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if not head or not head[0].isdigit():
            i += 1
            continue
        n = int(head[0])
        if i + 2 + n > len(lines):
            break
        comment = lines[i + 1].strip()
        symbols, xyz = [], []
        for line in lines[i + 2: i + 2 + n]:
            parts = line.split()
            symbols.append(re.sub(r"[^A-Za-z]", "", parts[0]).capitalize())
            xyz.append([float(v) for v in parts[1:4]])
        frames.append((comment, symbols, np.array(xyz)))
        i += 2 + n
    return frames


def energy_from_comment(comment: str) -> float | None:
    """GOAT and ORCA .xyz comments hold the energy (Eh) as the first float."""
    m = _FLOAT_RE.search(comment)
    return float(m.group(0).replace("D", "E").replace("d", "e")) if m else None


def write_xyz(path: Path, structures: list[Structure3D], comments: list[str] | None = None) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, s in enumerate(structures):
            comment = comments[k] if comments else f"{s.energy_eh:.10f}"
            f.write(f"{len(s.symbols)}\n{comment}\n")
            for sym, (x, y, z) in zip(s.symbols, s.coords):
                f.write(f"{sym:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def final_energy(out_file: Path) -> float | None:
    hits = re.findall(r"FINAL SINGLE POINT ENERGY.*?(-?\d+\.\d+)\s*$",
                      out_file.read_text(encoding="utf-8", errors="ignore"), flags=re.M)
    return float(hits[-1]) if hits else None


def imaginary_mode(hess_file: Path) -> np.ndarray | None:
    """
    Cartesian displacement vector (N, 3) of the most negative frequency in an
    ORCA .hess file, or None if no frequency is below -IMAG_NOISE_CM1.
    $normal_modes is a 3N x 3N matrix printed in column blocks, each block
    a header of column indices followed by 3N rows "row v_1 ... v_k".
    """
    lines = [ln.strip() for ln in hess_file.read_text(encoding="utf-8", errors="ignore").splitlines()]
    k = lines.index("$vibrational_frequencies")
    n = int(lines[k + 1])
    freqs = [float(lines[k + 2 + i].split()[1]) for i in range(n)]
    mode = min(range(n), key=lambda i: freqs[i])
    if freqs[mode] > -IMAG_NOISE_CM1:
        return None
    k = lines.index("$normal_modes")
    rows, cols = (int(v) for v in lines[k + 1].split())
    modes = np.zeros((rows, cols))
    i = k + 2
    while i < len(lines) and not lines[i].startswith(("$", "#")):   # "#" comments end the block
        if not lines[i]:
            i += 1
            continue
        col_idx = [int(c) for c in lines[i].split()]
        for r in range(rows):
            parts = lines[i + 1 + r].split()
            modes[int(parts[0]), col_idx] = [float(v) for v in parts[1:]]
        i += 1 + rows
    return modes[:, mode].reshape(-1, 3)


def frequencies_and_zpe(out_file: Path) -> tuple[list[float], float | None]:
    """Vibrational frequencies (zero modes removed) and ZPE from the last block."""
    text = out_file.read_text(encoding="utf-8", errors="ignore")
    blocks = text.split("VIBRATIONAL FREQUENCIES")
    freqs: list[float] = []
    if len(blocks) > 1:
        block = blocks[-1].split("NORMAL MODES")[0]
        vals = [float(v) for v in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", block, flags=re.M)]
        freqs = [v for v in vals if abs(v) > 1e-3]
    zpe = re.findall(r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", text)
    return freqs, (float(zpe[-1]) if zpe else None)


def natural_key(path: Path) -> list:
    """Sort key so that seed2.xyz comes before seed10.xyz."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_seeds(job_dir: Path, seed_files: list[Path] | None) -> list[Seed]:
    """
    Every frame of every seed file becomes one seed, named after its file
    (<stem>, or <stem>_fNN for multi-frame files). Default: seeds/*.xyz.
    """
    files = seed_files or sorted((job_dir / "seeds").glob("*.xyz"), key=natural_key)
    if not files:
        raise SystemExit(f"No seed geometries: put .xyz files in {job_dir / 'seeds'} or pass --seeds.")
    seeds: list[Seed] = []
    for path in files:
        if not path.is_file():
            raise SystemExit(f"Seed file not found: {path}")
        frames = read_xyz_frames(path)
        if not frames:
            raise SystemExit(f"{path}: no xyz frame found")
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)
        for k, (_, symbols, coords) in enumerate(frames):
            name = stem if len(frames) == 1 else f"{stem}_f{k:02d}"
            seeds.append(Seed(name, symbols, coords, path.resolve(), k))
    names = [s.name for s in seeds]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise SystemExit(f"Seed names must be unique; repeated: {', '.join(repeated)}")
    return seeds


def default_job_dir() -> tuple[Path, str]:
    """
    A copy of this script inside a job folder (next to .xyz files, seeds/ or
    job.toml) works on its own folder, wherever it is started from; the
    master copy, in a folder without job files, works on the current folder.
    """
    here = Path(__file__).resolve().parent
    if (here / "job.toml").is_file() or (here / "seeds").is_dir() or any(here.glob("*.xyz")):
        return here, "the folder of this script copy"
    return Path.cwd(), "the current folder"


def atom_mapping(ref_symbols: list[str], ref_bonds: set[tuple[int, int]],
                 symbols: list[str], bonds: set[tuple[int, int]]) -> np.ndarray | None:
    """
    Atom order that turns a structure into the reference molecule:
    order[r] is the atom that plays reference atom r (same element, same
    bonds). None if the two are different molecules. Backtracking search
    over element, degree and bonds to the atoms already matched, capped at
    ATOM_MATCH_MAX_STEPS steps (a capped search stops the run, it never
    guesses).
    """
    n = len(ref_symbols)
    if len(symbols) != n or sorted(symbols) != sorted(ref_symbols) or len(bonds) != len(ref_bonds):
        return None

    def neighbor_sets(pairs: set[tuple[int, int]]) -> list[set[int]]:
        nb: list[set[int]] = [set() for _ in range(n)]
        for i, j in pairs:
            nb[i].add(j)
            nb[j].add(i)
        return nb

    rn, tn = neighbor_sets(ref_bonds), neighbor_sets(bonds)

    def signature(sym: list[str], nb: list[set[int]], i: int) -> tuple:
        return sym[i], len(nb[i]), tuple(sorted(sym[j] for j in nb[i]))

    cands = [[t for t in range(n) if signature(symbols, tn, t) == signature(ref_symbols, rn, r)] for r in range(n)]
    if any(not c for c in cands):
        return None
    # Match reference atoms breadth-first from the rarest kind, so each new
    # atom is bonded to one already matched (strong pruning)
    visit, seen = [], set()
    for root in sorted(range(n), key=lambda r: (len(cands[r]), r)):
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        while queue:
            r = queue.pop(0)
            visit.append(r)
            for x in sorted(rn[r] - seen):
                seen.add(x)
                queue.append(x)
    match, used, steps = [-1] * n, [False] * n, 0

    def extend(k: int) -> bool:
        nonlocal steps
        if k == n:
            return True
        r = visit[k]
        for t in cands[r]:
            steps += 1
            if steps > ATOM_MATCH_MAX_STEPS:
                raise SystemExit(f"Atom-order matching gave up after {ATOM_MATCH_MAX_STEPS} steps; "
                                 "list the atoms of every target in the same order.")
            if used[t] or any((match[x] in tn[t]) != (x in rn[r]) for x in visit[:k]):
                continue
            match[r], used[t] = t, True
            if extend(k + 1):
                return True
            match[r], used[t] = -1, False
        return False

    return np.array(match) if extend(0) else None


def group_targets(seeds: list[Seed]) -> list[list[Seed]]:
    """
    Folder mode: sorts the target structures into molecules. A structure
    joins the first group whose reference (the group's first structure) has
    the same atoms and bond graph; one that lists the atoms in another order
    is renumbered to the reference order. Anything else starts a new group,
    so a different molecule is searched separately, never mixed in.
    """
    groups: list[list[Seed]] = []
    group_bonds: list[set[tuple[int, int]]] = []
    for s in seeds:
        unknown = sorted(e for e in set(s.symbols) if e not in COVALENT_RADII_A or e not in ISOTOPE_MASSES)
        if unknown:
            raise SystemExit(f"{s.path.name}: element(s) {', '.join(unknown)} missing from "
                             "COVALENT_RADII_A/ISOTOPE_MASSES.")
        bonds = bonded_pairs(s.symbols, s.coords)
        for members, ref_bonds in zip(groups, group_bonds):
            ref = members[0]
            if s.symbols == ref.symbols and bonds == ref_bonds:
                members.append(s)
                break
            order = atom_mapping(ref.symbols, ref_bonds, s.symbols, bonds)
            if order is not None:
                s.symbols, s.coords = list(ref.symbols), s.coords[order]
                s.note = (f"atoms renumbered to the order of {ref.name} "
                          f"(new atom k = old atom {' '.join(str(int(i)) for i in order)})")
                members.append(s)
                break
        else:
            groups.append([s])
            group_bonds.append(bonds)
    return groups


def group_folder_names(topologies: list[Topology]) -> list[str]:
    """results/ subfolder per group: its fragment formulas, e.g. CH2O2-N2O (_2, _3 if repeated)."""
    names: list[str] = []
    for topo in topologies:
        base = "-".join(sorted(topo.fragment_name(k) for k in range(len(topo.fragments))))
        name, k = base, 2
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.append(name)
    return names


def load_settings(job_dir: Path, args: argparse.Namespace) -> JobSettings:
    """Defaults, then <job>/job.toml, then command-line overrides."""
    settings = JobSettings(name=job_dir.name)
    toml_file = job_dir / "job.toml"
    if toml_file.exists():
        data = tomllib.loads(toml_file.read_text(encoding="utf-8"))
        allowed = [f.name for f in fields(JobSettings)]
        unknown = sorted(set(data) - set(allowed))
        if unknown:
            raise SystemExit(f"{toml_file}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(allowed)}")
        for key, value in data.items():
            setattr(settings, key, value)
    for key, value in (("name", args.name), ("charge", args.charge), ("multiplicity", args.mult),
                       ("window_kcal", args.window)):
        if value is not None:
            setattr(settings, key, value)
    settings.charge, settings.multiplicity = int(settings.charge), int(settings.multiplicity)
    settings.window_kcal = float(settings.window_kcal)
    settings.goat_min_iter, settings.goat_max_iter = int(settings.goat_min_iter), int(settings.goat_max_iter)
    if settings.multiplicity < 1 or not (1 <= settings.goat_min_iter <= settings.goat_max_iter):
        raise SystemExit(f"Invalid settings: {settings}")
    return settings


# ==============================================================================
# 4. OPI 2.0 INPUT BUILDING & ORCA EXECUTION
# ==============================================================================
@dataclass
class Engine:
    """Everything needed to launch ORCA."""

    orca_bin: Path
    orca_version: str


def new_calculator(job_dir: Path, base: str, ncores: int) -> Calculator:
    """OPI Calculator preloaded with ! NATIVE-GFN2-XTB, cores and memory."""
    job_dir.mkdir(parents=True, exist_ok=True)
    calc = Calculator(basename=base, working_dir=job_dir, version_check=False)
    calc.json_via_input = False          # skip the GBW/property JSON export
    calc.input.ncores = ncores
    calc.input.memory = MAXCORE_MB
    calc.input.add_simple_keywords(Sqm.NATIVE_GFN2_XTB)   # in-process: no xtb.exe launches
    return calc


def write_input(calc: Calculator, start_xyz: Path, settings: JobSettings, strict: bool = True) -> bool:
    """
    Writes <base>.inp; returns True if it is identical to the one already there.
    The .inp only names start.xyz, so a fingerprint of the geometry is embedded
    as a comment: a changed start structure then forces a rerun on resume.
    """
    calc.structure = XyzFile(start_xyz, charge=settings.charge, multiplicity=settings.multiplicity, strict=strict)
    if start_xyz.exists():
        digest = hashlib.sha1(start_xyz.read_bytes()).hexdigest()[:16]
        calc.input.add_arbitrary_string(f"# start.xyz sha1 {digest}")
    inp = calc.working_dir / f"{calc.basename}.inp"
    old = inp.read_text(encoding="utf-8") if inp.exists() else None
    calc.write_input()
    return old == inp.read_text(encoding="utf-8")


def newest_mtime(folder: Path) -> float:
    """
    Newest modification time in a folder. os.scandir reads the times from the
    directory listing itself on Windows, so no file is opened, and files that
    GOAT deletes mid-scan are simply skipped.
    """
    newest = 0.0
    with os.scandir(folder) as entries:
        for entry in entries:
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                pass
    return newest


def run_orca(calc: Calculator, engine: Engine, timeout_s: int, unchanged_input: bool,
             stall_s: int | None = None) -> str:
    """
    Runs ORCA on <base>.inp in the job folder. Returns "reused", "ok",
    "failed", "timeout" or "stalled". ORCA is launched in its own process
    group at below-normal priority, so a timeout, Ctrl+C or any error kills
    the whole tree (ORCA and its module programs), which
    subprocess.run(timeout=...) would leave orphaned on Windows. With
    stall_s, a job whose folder has not changed for stall_s seconds is
    treated as hung and killed (checked every STALL_CHECK_EVERY_S).
    """
    output = Output(basename=calc.basename, working_dir=calc.working_dir)
    if unchanged_input and output.terminated_normally():
        return "reused"
    if _REUSE_ONLY:
        raise RuntimeError(f"--reuse-only: {calc.basename} has no finished output for its current input; "
                           "ORCA was not started.")
    env = os.environ.copy()
    env["PATH"] = str(engine.orca_bin.parent) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "1"         # serial jobs only: no thread pools either
    env["MKL_NUM_THREADS"] = "1"
    if os.name == "nt":
        extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    else:
        extra = {"start_new_session": True, "preexec_fn": lambda: os.nice(10)}
    out_file = calc.working_dir / f"{calc.basename}.out"
    with open(out_file, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(engine.orca_bin), f"{calc.basename}.inp"], cwd=calc.working_dir,
                                stdout=fh, stderr=subprocess.STDOUT, env=env, **extra)
        _RUNNING.add(proc)
        deadline = time.time() + timeout_s
        next_stall_check = time.time() + STALL_CHECK_EVERY_S
        try:
            while proc.poll() is None:
                time.sleep(2.0)
                now = time.time()
                if now > deadline:
                    kill_tree(proc)
                    return "timeout"
                if stall_s and now >= next_stall_check:
                    next_stall_check = now + STALL_CHECK_EVERY_S
                    if now - newest_mtime(calc.working_dir) > stall_s:
                        kill_tree(proc)
                        return "stalled"
        except BaseException:
            kill_tree(proc)                  # never leave ORCA running behind a crashed script
            raise
        finally:
            _RUNNING.discard(proc)
    return "ok" if output.terminated_normally() else "failed"


# ==============================================================================
# 5. PHASE 0: PRE-FLIGHT
# ==============================================================================
def resolve_engine() -> Engine:
    """Finds ORCA through OPI (config.toml, $OPI_ORCA or PATH) and checks its version."""
    runner = Runner()
    orca_bin = runner.get_orca_binary(OrcaBinary.ORCA)
    runner.check_version()               # raises if ORCA is missing or too old
    version = str(runner.get_version())
    log(f"[PRE-FLIGHT] ORCA {version}: {orca_bin}")
    if not version.startswith("6.1"):
        log(f"[PRE-FLIGHT] WARNING: written and tested for ORCA 6.1.1, found {version}.")
    return Engine(orca_bin=orca_bin, orca_version=version)


def engine_smoke_test(engine: Engine, jobs_dir: Path, seed_xyz: Path, settings: JobSettings) -> None:
    """One ORCA EnGrad on the first seed (a fraction of a second) before any long job."""
    calc = new_calculator(jobs_dir / "engine_test", "engine_test", ncores=1)
    calc.input.add_simple_keywords(Task.ENGRAD)
    shutil.copyfile(seed_xyz, calc.working_dir / "start.xyz")
    unchanged = write_input(calc, calc.working_dir / "start.xyz", settings)
    status = run_orca(calc, engine, timeout_s=120, unchanged_input=unchanged)
    energy = final_energy(calc.working_dir / "engine_test.out") if status in ("ok", "reused") else None
    if energy is None:
        raise RuntimeError(f"Engine test failed ({status}); see {calc.working_dir / 'engine_test.out'}")
    log(f"[PRE-FLIGHT] Engine test  : E({seed_xyz.stem}) = {energy:.8f} Eh ({status})")


# ==============================================================================
# 6. PHASES 1 & 2: SEED QUENCH + PER-SEED GOAT
# ==============================================================================
def quench_seed(seed: str, engine: Engine, jobs_dir: Path, settings: JobSettings,
                dry_run: bool) -> Structure3D | None:
    calc = new_calculator(jobs_dir / f"{seed}_quench", f"{seed}_quench", ncores=1)
    calc.input.add_simple_keywords(Opt.TIGHTOPT)
    calc.input.add_blocks(BlockGeom(maxiter=OPT_MAXITER))
    start = calc.working_dir / "start.xyz"
    shutil.copyfile(jobs_dir / "seeds" / f"{seed}.xyz", start)
    unchanged = write_input(calc, start, settings)
    out, xyz = calc.working_dir / f"{seed}_quench.out", calc.working_dir / f"{seed}_quench.xyz"
    if dry_run:                          # reuse a finished quench so the GOAT input is the real one
        if not (unchanged and xyz.exists() and Output(basename=f"{seed}_quench",
                                                      working_dir=calc.working_dir).terminated_normally()):
            return None
        status = "reused"
    else:
        t0 = time.time()
        status = run_orca(calc, engine, PER_ISOMER_TIMEOUT_S, unchanged)
        log(f"  [{seed}] quench {status} ({time.time() - t0:.0f} s)")
    energy = final_energy(out) if status in ("ok", "reused") and xyz.exists() else None
    if energy is None:
        log(f"  [{seed}] WARNING: quench failed; GOAT will start from the raw seed.")
        return None
    _, symbols, coords = read_xyz_frames(xyz)[-1]
    return Structure3D(symbols, coords, energy, sources=[f"{seed}_quench"])


def goat_seed(seed: str, start_xyz: Path, engine: Engine, jobs_dir: Path, settings: JobSettings,
              min_iter: int, max_iter: int, dry_run: bool) -> list[Structure3D]:
    # Serial GOAT: the workers take turns in one ORCA run, so ORCA launches one
    # module program at a time and never goes through mpiexec.
    calc = new_calculator(jobs_dir / f"{seed}_goat", f"{seed}_goat", ncores=GOAT_NPROCS)
    calc.input.add_simple_keywords(Goat.GOAT, OutputControl.MINIPRINT)   # MiniPrint: far less log I/O
    calc.input.add_blocks(BlockGoat(nworkers=GOAT_NWORKERS, minglobaliter=min_iter,
                                    maxglobaliter=max_iter, maxen=settings.window_kcal, autowall=True))
    # OPI 2.0 types RMSDMetric as an int, so these keywords go in verbatim
    calc.input.add_arbitrary_string("%goat\n    RMSDMetric eigenvalue\n    KeepWorkerData false\nend")
    start = calc.working_dir / "start.xyz"
    if start_xyz.exists():
        shutil.copyfile(start_xyz, start)
    unchanged = write_input(calc, start, settings, strict=not dry_run)
    if dry_run:
        return []

    t0 = time.time()
    log(f"  [{seed}] GOAT running ({GOAT_NWORKERS} workers in turn, {GOAT_NPROCS} core, "
        f"up to {max_iter} global iterations) ...")
    status = run_orca(calc, engine, GOAT_TIMEOUT_S, unchanged, stall_s=GOAT_STALL_S)
    log(f"  [{seed}] GOAT {status} ({(time.time() - t0) / 60:.1f} min)")
    ens = calc.working_dir / f"{seed}_goat.finalensemble.xyz"
    if status not in ("ok", "reused") or not ens.exists():
        partial = sorted(calc.working_dir.glob(f"{seed}_goat.finalensemble.globaliter.*.xyz"),
                         key=lambda p: int(p.suffixes[-2].lstrip(".")))
        if not partial:
            log(f"  [{seed}] WARNING: GOAT produced no ensemble; see {calc.working_dir / (seed + '_goat.out')}")
            return []
        ens = partial[-1]
        log(f"  [{seed}] using the last completed global iteration: {ens.name}")

    found = []
    for k, (comment, symbols, coords) in enumerate(read_xyz_frames(ens)):
        energy = energy_from_comment(comment)
        if energy is not None:
            found.append(Structure3D(symbols, coords, energy, sources=[f"{seed}_goat#{k:02d}"]))
    log(f"  [{seed}] GOAT ensemble: {len(found)} conformers")
    return found


# ==============================================================================
# 7. PHASE 4: PER-ISOMER CHARACTERIZATION (TightOpt + NumFreq, <= 10 min)
# ==============================================================================
def characterize(cand: Structure3D, tag: str, engine: Engine, jobs_dir: Path, settings: JobSettings) -> Structure3D:
    calc = new_calculator(jobs_dir / tag, tag, ncores=1)
    # 2 x 3N in-process gradients; no analytic native-xTB Hessian assumed
    calc.input.add_simple_keywords(Opt.TIGHTOPT, Task.NUMFREQ)
    calc.input.add_blocks(BlockGeom(maxiter=OPT_MAXITER))
    start = calc.working_dir / "start.xyz"
    write_xyz(start, [cand])
    unchanged = write_input(calc, start, settings)
    t0 = time.time()
    status = run_orca(calc, engine, PER_ISOMER_TIMEOUT_S, unchanged)
    cand.char_job = f"jobs/{tag}"
    out, xyz = calc.working_dir / f"{tag}.out", calc.working_dir / f"{tag}.xyz"

    if status == "timeout":
        cand.status = f"TIMEOUT (> {PER_ISOMER_TIMEOUT_S // 60} min): GOAT geometry kept"
    elif status == "failed" or not xyz.exists():
        cand.status = "FAILED: GOAT geometry kept"
    else:
        try:
            _, symbols, coords = read_xyz_frames(xyz)[-1]
            energy = final_energy(out)
            freqs, zpe = frequencies_and_zpe(out)
        except (OSError, IndexError, ValueError) as exc:
            cand.status = f"FAILED: unreadable output ({exc.__class__.__name__}): GOAT geometry kept"
        else:
            cand.symbols, cand.coords = symbols, coords
            cand.energy_eh = energy if energy is not None else cand.energy_eh
            cand.freqs_cm1, cand.zpe_eh = freqs, zpe
            cand.opt_converged = Output(basename=tag, working_dir=calc.working_dir).geometry_optimization_converged()
            imag = [f for f in freqs if f < 0.0]
            if not cand.opt_converged:
                cand.status = "opt NOT converged"
            elif not freqs:
                cand.status = "no frequencies found"
            elif not imag:
                cand.status = "minimum"
            elif all(f > -IMAG_NOISE_CM1 for f in imag):
                cand.status = f"minimum (tiny imag {min(imag):.0f} cm-1, numerical noise)"
            else:
                cand.status = f"SADDLE POINT ({len(imag)} imag, {min(imag):.0f} cm-1)"
    log(f"  [{tag}] {status:>7} ({time.time() - t0:5.0f} s)  {cand.status}")
    return cand


def follow_saddle(ts: Structure3D, engine: Engine, jobs_dir: Path, settings: JobSettings) -> list[Structure3D]:
    """
    A saddle point is a transition state, not an isomer. Step off it along
    its imaginary mode (largest atom shift SADDLE_STEP_A) in both directions
    and re-run TightOpt + NumFreq: each side relaxes into the minimum the
    transition state connects. Returns [] if the mode cannot be read.
    """
    base = Path(ts.char_job).name
    hess = jobs_dir / base / f"{base}.hess"
    try:
        mode = imaginary_mode(hess)
    except (OSError, ValueError, IndexError):
        mode = None
    if mode is None:
        log(f"  [{base}] could not read the imaginary mode; saddle point kept as is")
        return []
    step = SADDLE_STEP_A / np.linalg.norm(mode, axis=1).max()
    downhill = []
    for sign, side in ((1.0, "plus"), (-1.0, "minus")):
        start = Structure3D(list(ts.symbols), ts.coords + sign * step * mode, ts.energy_eh,
                            sources=ts.sources + [f"{base}-TS-{side}"])
        downhill.append(characterize(start, f"{base}_ts{side}", engine, jobs_dir, settings))
    return downhill


# ==============================================================================
# 8. PHASE 5: ORGANIZE & EXPORT
# ==============================================================================
def contact_label(coords: np.ndarray, symbols: list[str], topo: Topology, rank: int) -> str:
    contacts = intermolecular_contacts(coords, topo)
    if rank >= len(contacts):
        return ""
    d, i, j = contacts[rank]
    return f"{symbols[i]}{i}...{symbols[j]}{j} {d:.2f}"


def export_isomers(isomers: list[Structure3D], rejected: list[tuple[Structure3D, str]],
                   audit: list[list[str]], iso_dir: Path, topo: Topology, settings: JobSettings) -> Path:
    iso_dir.mkdir(parents=True, exist_ok=True)
    for old in itertools.chain(iso_dir.glob("isomer-*.xyz"), iso_dir.glob("Isomer_*.xyz")):
        old.unlink()
    emin = isomers[0].energy_eh
    stem = f"{re.sub(r'[^A-Za-z0-9_-]', '_', settings.name)}_{METHOD_TAG}_isomers"
    comments, rows = [], []
    for s in isomers:
        rel = (s.energy_eh - emin) * HARTREE_TO_KCAL
        a, b, c = rotational_constants_mhz(s.symbols, s.coords)
        lowest = min(s.freqs_cm1) if s.freqs_cm1 else None
        com = com_distance(s.symbols, s.coords, topo)
        comment = (f"{s.label} E={s.energy_eh:.8f} Eh dE={rel:.3f} kcal/mol "
                   f"A/B/C={a:.1f}/{b:.1f}/{c:.1f} MHz [{s.status}]")
        comments.append(comment)
        write_xyz(iso_dir / f"{s.label}.xyz", [s], [comment])
        rows.append({
            "Isomer": s.label,
            "Status": s.status,
            "Rel_E_kcal_mol": f"{rel:.3f}",
            "Rel_E_cm-1": f"{(s.energy_eh - emin) * HARTREE_TO_CM1:.1f}",
            "Energy_Eh": f"{s.energy_eh:.8f}",
            "ZPE_Eh": f"{s.zpe_eh:.6f}" if s.zpe_eh is not None else "",
            "A_MHz": f"{a:.2f}", "B_MHz": f"{b:.2f}", "C_MHz": f"{c:.2f}",
            "Closest_contact_A": contact_label(s.coords, s.symbols, topo, 0),
            "Second_contact_A": contact_label(s.coords, s.symbols, topo, 1),
            "COM_distance_A": f"{com:.3f}" if com is not None else "",
            "N_imag": sum(1 for f in s.freqs_cm1 if f < 0.0) if s.freqs_cm1 else "",
            "Lowest_freq_cm-1": f"{lowest:.1f}" if lowest is not None else "",
            "Opt_converged": "" if s.opt_converged is None else ("yes" if s.opt_converged else "NO"),
            "N_raw_structures": s.n_raw,
            "Found_by": " ".join(s.sources),
            "ORCA_job": s.char_job,
        })
    write_xyz(iso_dir / f"{stem}.xyz", isomers, comments)
    with open(iso_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Pairs the conservative dedup did NOT merge but a human should compare
    near = []
    for p, q in itertools.combinations(isomers, 2):
        de = abs(p.energy_eh - q.energy_eh) * HARTREE_TO_KCAL
        if de > NEAR_DUP_ENERGY_KCAL:
            continue
        rot = max_rel_rot_diff(rotational_constants_mhz(p.symbols, p.coords),
                               rotational_constants_mhz(q.symbols, q.coords))
        rmsd = best_rmsd(p.coords, q.coords, topo.permutations)
        if rot <= NEAR_DUP_ROT_REL or rmsd <= NEAR_DUP_RMSD_A:
            near.append([p.label, q.label, f"{de:.3f}", f"{100 * rot:.2f}", f"{rmsd:.3f}"])
    with open(iso_dir / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Isomer_A", "Isomer_B", "dE_kcal_mol", "max_dRot_pct", "RMSD_A"])
        writer.writerows(near)

    write_xyz(iso_dir / "rejected_structures.xyz", [s for s, _ in rejected],
              [f"{' '.join(s.sources)} E={s.energy_eh:.8f} REJECTED: {why}" for s, why in rejected])
    with open(iso_dir / "structure_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Structure", "Energy_Eh", "Status", "Fate"])
        writer.writerows(audit)

    frag_text = " | ".join(f"{topo.fragment_name(k)}: " + " ".join(f"{topo.symbols[i]}{i}" for i in frag)
                           for k, frag in enumerate(topo.fragments))
    equiv_text = ", ".join("/".join(f"{topo.symbols[i]}{i}" for i in g) for g in topo.equivalent_groups) or "none"
    (iso_dir / "README.txt").write_text(README_TEXT.format(
        mol=settings.name, method=METHOD_TAG, stem=stem, etol=DEDUP_ENERGY_TOL_KCAL, rtol=DEDUP_RMSD_TOL_A,
        ne=NEAR_DUP_ENERGY_KCAL, nr=100 * NEAR_DUP_ROT_REL, nd=NEAR_DUP_RMSD_A, win=settings.window_kcal,
        frags=frag_text, equiv=equiv_text), encoding="utf-8")

    log("")
    log(f"{'Isomer':<11}{'dE kcal/mol':>12}{'A / B / C (MHz)':>30}  {'closest contact (A)':<22} status")
    for s, row in zip(isomers, rows):
        abc = f"{float(row['A_MHz']):.0f} / {float(row['B_MHz']):.0f} / {float(row['C_MHz']):.0f}"
        log(f"{s.label:<11}{row['Rel_E_kcal_mol']:>12}{abc:>30}  {row['Closest_contact_A']:<22} {s.status}")
    log(f"\n  Near-duplicate pairs for manual review: {len(near)}")
    return iso_dir / f"{stem}.csv"


README_TEXT = """{mol} isomer set ({method}, GFN2-xTB)
==============================================================
isomer-NN.xyz              one structure per isomer, NN = energy rank (00 = lowest)
{stem}.xyz     all isomers in one file (open in Avogadro/Chemcraft
                           and step through the frames)
{stem}.csv     one row per isomer (see columns below)
near_duplicates.csv        pairs that MAY be the same isomer - check these
rejected_structures.xyz    structures removed automatically, with the reason
structure_audit.csv        every raw GOAT structure and every characterized
                           structure, and what became of it

How the set was made
  GOAT (topology-preserving) from each starting seed -> merged -> structures
  with broken monomers or separated fragments removed -> exact duplicates
  merged -> TightOpt + NumFreq on each -> any saddle point (transition
  state) pushed downhill both ways and re-optimized -> exact duplicates
  merged again. Transition states are listed in rejected_structures.xyz.
  "Exact duplicate" = dE <= {etol} kcal/mol AND RMSD <= {rtol} A, where the
  RMSD is minimized over rotation, mirror reflection and permutations of
  symmetry-equivalent atoms ({equiv}). Isomers above {win} kcal/mol were dropped.

Fragments (ORCA 0-based atom numbering)
  {frags}

Columns
  Status            minimum / SADDLE POINT / opt NOT converged / TIMEOUT
  Rel_E             energy above the lowest isomer (no ZPE correction)
  A/B/C_MHz         rotational constants, most abundant isotopologue
  *_contact_A       closest atom pairs between different fragments
  COM_distance_A    distance between the fragments' centres of mass (2 fragments)
  Found_by          which quench/GOAT runs found this isomer
  N_raw_structures  how many raw structures were merged into this one

Manual deduplication tips
  1. Start with near_duplicates.csv: pairs within {ne} kcal/mol whose
     rotational constants agree to {nr:.0f} % or whose RMSD is below {nd} A.
  2. Same isomer if the energies agree, A/B/C agree, and the same atoms make
     the closest contacts. Mirror images are the same isomer.
  3. A SADDLE POINT is not an isomer: it is a transition state between two.
     The script already follows them downhill; any left in the table could
     not be resolved automatically.
"""


# ==============================================================================
# 9. MAIN PIPELINE
# ==============================================================================
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reusable GOAT isomer search with ORCA 6.1.1 + OPI 2.0 (GFN2-xTB). "
                                            "Run a copy inside the job folder, or the master from inside it.")
    p.add_argument("--job-dir", type=Path, default=None,
                   help="job folder (default: the folder of a script copy, else the current folder)")
    p.add_argument("--seeds", type=Path, nargs="+",
                   help="seed .xyz files of one molecule (default: <job>/seeds/*.xyz, else every <job>/*.xyz)")
    p.add_argument("--name", help="molecule/job name (default: job.toml or the folder name)")
    p.add_argument("--charge", type=int, help="total charge (default 0)")
    p.add_argument("--mult", type=int, help="spin multiplicity (default 1)")
    p.add_argument("--window", type=float, help="energy window in kcal/mol (default 12)")
    p.add_argument("--quick", action="store_true", help="setup test: first seed, 1 GOAT iteration, in quick_test/")
    p.add_argument("--dry-run", action="store_true", help="write ORCA inputs, run nothing")
    p.add_argument("--reuse-only", action="store_true",
                   help="re-run the analysis from finished jobs only; stop instead of starting ORCA")
    return p.parse_args(argv)


@dataclass
class GroupRun:
    """One molecule to search: its settings, checked seeds, topology and run folder."""

    settings: JobSettings
    seeds: list[Seed]
    topo: Topology
    run_dir: Path
    min_iter: int
    max_iter: int
    skipped: dict[str, str] = field(default_factory=dict)     # seed copy -> the seed it copies

    @property
    def jobs_dir(self) -> Path:
        return self.run_dir / "jobs"


def main(argv: list[str] | None = None) -> int:
    global _REUSE_ONLY
    args = parse_args(argv)
    _REUSE_ONLY = args.reuse_only
    needs_lock = not (args.dry_run or args.reuse_only)
    job_dir, why = (args.job_dir, "--job-dir") if args.job_dir else default_job_dir()
    job_dir = job_dir.resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"Job folder not found: {job_dir}")
    settings = load_settings(job_dir, args)
    if args.seeds or any((job_dir / "seeds").glob("*.xyz")):
        return main_seeds_mode(args, settings, job_dir, needs_lock)
    return main_folder_mode(args, settings, job_dir, why, needs_lock)


def main_seeds_mode(args: argparse.Namespace, settings: JobSettings, job_dir: Path, needs_lock: bool) -> int:
    """seeds/ (or --seeds) = one molecule; output in the job folder itself."""
    seeds =collect_seeds(job_dir, [p.resolve() for p in args.seeds] if args.seeds else None)
    t_start = time.time()
    run = prepare_group(settings, seeds, job_dir, args.quick)
    engine = resolve_engine()
    if needs_lock:
        acquire_orca_lock(f"goat_isomer_search in {run.run_dir}")
    try:
        if not args.dry_run:
            engine_smoke_test(engine, run.jobs_dir, run.jobs_dir / "seeds" / f"{run.seeds[0].name}.xyz", settings)
        return run_pipeline(args, settings, run.topo, run.seeds, engine, run.run_dir, run.jobs_dir,
                            run.run_dir / "isomers", run.min_iter, run.max_iter, t_start)
    finally:
        if needs_lock:
            release_orca_lock()


def main_folder_mode(args: argparse.Namespace, settings: JobSettings, job_dir: Path, why: str,
                     needs_lock: bool) -> int:
    """
    Every *.xyz in the job folder is a target; one group (and one
    results/<group>/ folder) per molecule. Every group is checked before the
    first ORCA job starts; the groups then run one after another under one
    lock, and a failed group does not stop the next one.
    """
    global _LOG_FILE
    targets = sorted(job_dir.glob("*.xyz"), key=natural_key)
    if not targets:
        raise SystemExit(f"No targets: put .xyz files in {job_dir} (folder mode) or in "
                         f"{job_dir / 'seeds'} (one molecule), or pass --seeds.")
    groups = group_targets(collect_seeds(job_dir, targets))
    topologies = [Topology.from_reference(g[0].symbols, g[0].coords) for g in groups]
    folders = group_folder_names(topologies)
    results = job_dir / RESULTS_DIRNAME
    results.mkdir(exist_ok=True)
    overview = results / "run_log.txt"

    _LOG_FILE = overview
    log("=" * 78)
    log(f"{settings.name} GOAT isomer search, FOLDER MODE | GFN2-xTB | ORCA 6.1.1 + OPI 2.0 | "
        f"{datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 78)
    log(f"[FOLDER] Tool         : {Path(__file__).resolve()} v{TOOL_VERSION}")
    log(f"[FOLDER] Job folder   : {job_dir} ({why})")
    log(f"[FOLDER] Targets      : {len(targets)} file(s), {sum(len(g) for g in groups)} structure(s) "
        f"-> {len(groups)} group(s), searched one after another")
    for members, folder in zip(groups, folders):
        log(f"   {RESULTS_DIRNAME}/{folder}/ : {', '.join(f'{s.name} ({s.path.name})' for s in members)}")
        for s in members:
            if s.note:
                log(f"      {s.name}: {s.note}")

    runs: list[GroupRun] = []
    for members, folder in zip(groups, folders):
        name = settings.name if len(groups) == 1 else f"{settings.name} {folder}"
        runs.append(prepare_group(replace(settings, name=name), members, results / folder, args.quick))
    _LOG_FILE = overview
    with open(results / "targets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Target_file", "Frame", "Seed", "Group_folder", "Fate", "Note"])
        for run, members, folder in zip(runs, groups, folders):
            for s in members:
                fate = f"copy of {run.skipped[s.name]}: skipped" if s.name in run.skipped else "seed"
                writer.writerow([s.path.name, s.frame, s.name, f"{RESULTS_DIRNAME}/{folder}", fate, s.note])
    log(f"[FOLDER] Target list  : {results / 'targets.csv'}")

    engine = resolve_engine()
    if needs_lock:
        acquire_orca_lock(f"goat_isomer_search (folder mode) in {job_dir}")
    failed: list[str] = []
    try:
        for run, folder in zip(runs, folders):
            _LOG_FILE = overview
            log(f"\n[GROUP] {folder}: started {datetime.now():%Y-%m-%d %H:%M} -> {run.run_dir / 'run_log.txt'}")
            _LOG_FILE = run.run_dir / "run_log.txt"
            log(f"[PRE-FLIGHT] ORCA {engine.orca_version}: {engine.orca_bin}")
            try:
                if not args.dry_run:
                    engine_smoke_test(engine, run.jobs_dir, run.jobs_dir / "seeds" / f"{run.seeds[0].name}.xyz",
                                      run.settings)
                run_pipeline(args, run.settings, run.topo, run.seeds, engine, run.run_dir, run.jobs_dir,
                             run.run_dir / "isomers", run.min_iter, run.max_iter, time.time())
                outcome = "dry-run: inputs written" if args.dry_run else "finished"
            except Exception:
                kill_all_running()
                log("\n[ERROR] " + traceback.format_exc())
                failed.append(folder)
                outcome = "FAILED (see its run_log.txt); continuing with the next group"
            _LOG_FILE = overview
            log(f"[GROUP] {folder}: {outcome} {datetime.now():%Y-%m-%d %H:%M}")
    finally:
        if needs_lock:
            release_orca_lock()
    _LOG_FILE = overview
    log(f"\n[FOLDER DONE] {len(runs) - len(failed)}/{len(runs)} group(s) finished"
        + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def prepare_group(settings: JobSettings, seeds: list[Seed], run_dir: Path, quick: bool) -> GroupRun:
    """
    Pre-flight for one molecule (starts no ORCA job): topology from the first
    seed, every other seed checked against it, exact seed copies skipped,
    seeds written to jobs/seeds/. Logs to <run_dir>/run_log.txt.
    """
    global _LOG_FILE
    min_iter, max_iter = settings.goat_min_iter, settings.goat_max_iter
    if quick:
        seeds, min_iter, max_iter = seeds[:1], 1, 1
        run_dir = run_dir / "quick_test"
    jobs_dir = run_dir / "jobs"
    (jobs_dir / "seeds").mkdir(parents=True, exist_ok=True)
    _LOG_FILE = run_dir / "run_log.txt"

    log("=" * 78)
    log(f"{settings.name} GOAT isomer search | GFN2-xTB | ORCA 6.1.1 + OPI 2.0 | {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 78)
    log(f"[PRE-FLIGHT] Tool         : {Path(__file__).resolve()} v{TOOL_VERSION}")
    log(f"[PRE-FLIGHT] Run folder   : {run_dir}")
    log(f"[PRE-FLIGHT] Settings     : charge {settings.charge}, multiplicity {settings.multiplicity}, "
        f"window {settings.window_kcal} kcal/mol, GOAT {min_iter}-{max_iter} global iterations")
    if len(str(run_dir)) > 120:
        log("[PRE-FLIGHT] WARNING: long folder path; ORCA on Windows can fail beyond 260 characters.")

    # Topology from the first seed; every other seed must match it exactly
    ref = seeds[0]
    topo = Topology.from_reference(ref.symbols, ref.coords)
    for s in seeds[1:]:
        if s.symbols != ref.symbols:
            raise SystemExit(f"Seed {s.name} ({s.path.name}) has a different atom list/order than {ref.name}.")
        bonds = bonded_pairs(s.symbols, s.coords)
        if bonds != topo.bonds:
            diff = [f"+{s.symbols[i]}{i}-{s.symbols[j]}{j}" for i, j in sorted(bonds - topo.bonds)]
            diff += [f"-{s.symbols[i]}{i}-{s.symbols[j]}{j}" for i, j in sorted(topo.bonds - bonds)]
            raise SystemExit(f"Seed {s.name} does not have the bonding pattern of {ref.name}: {' '.join(diff)}")
    log(f"[PRE-FLIGHT] Topology (from {ref.name}, {len(ref.symbols)} atoms):")
    for line in topo.describe():
        log(f"               {line}")
    if len(topo.fragments) > 1 and fragments_separated(ref.coords, topo) is not None:
        log(f"[PRE-FLIGHT] WARNING: the fragments of {ref.name} are more than {DISSOCIATION_CONTACT_A} A apart.")

    for s in seeds:
        if s.note:
            log(f"[PRE-FLIGHT] {s.name}: {s.note}")

    # Exact copies among the seeds are skipped (they would repeat a whole GOAT run)
    kept: list[Seed] = []
    skipped: dict[str, str] = {}
    for s in seeds:
        twin = next((k for k in kept if best_rmsd(s.coords, k.coords, topo.permutations) <= DUPLICATE_SEED_RMSD_A),
                    None)
        if twin is not None:
            log(f"[PRE-FLIGHT] Seed {s.name} is a copy of {twin.name} "
                f"(RMSD {best_rmsd(s.coords, twin.coords, topo.permutations):.1e} A): skipped")
            skipped[s.name] = twin.name
        else:
            kept.append(s)
    log(f"[PRE-FLIGHT] Seeds        : {', '.join(f'{s.name} ({s.path.name})' for s in kept)}")
    log("[PRE-FLIGHT]                (serial: one ORCA job, one core at a time)")
    for s in kept:
        write_xyz(jobs_dir / "seeds" / f"{s.name}.xyz", [Structure3D(s.symbols, s.coords, 0.0)], [s.name])
    return GroupRun(settings, kept, topo, run_dir, min_iter, max_iter, skipped)


def run_pipeline(args: argparse.Namespace, settings: JobSettings, topo: Topology, seeds: list[Seed],
                 engine: Engine, run_dir: Path, jobs_dir: Path, iso_dir: Path,
                 min_iter: int, max_iter: int, t_start: float) -> int:
    names = [s.name for s in seeds]

    # Phase 1: quench seeds (one at a time, single-core, <= 10 min each)
    log("\nPHASE 1: SEED QUENCH (TightOpt)")
    quenched = [quench_seed(s, engine, jobs_dir, settings, args.dry_run) for s in names]

    # Phase 2: GOAT per seed (one at a time, serial GOAT)
    log("\nPHASE 2: GOAT SEARCH PER SEED")
    pool: list[Structure3D] = [q for q in quenched if q is not None]
    for name, q in zip(names, quenched):
        start = jobs_dir / f"{name}_quench" / f"{name}_quench.xyz"
        if q is None:
            start = jobs_dir / "seeds" / f"{name}.xyz"
        pool += goat_seed(name, start, engine, jobs_dir, settings, min_iter, max_iter, args.dry_run)
    if args.dry_run:
        log(f"\n[DRY-RUN] Inputs written under {jobs_dir}. Nothing was run.")
        return 0
    if not pool:
        raise RuntimeError("No structures came out of Phases 1-2; check the job .out files.")

    # Phase 3: screen + first dedup
    log("\nPHASE 3: POOL, SCREEN & DEDUPLICATE")
    rejected: list[tuple[Structure3D, str]] = []
    intact = []
    raw_fate: dict[int, str] = {}
    for s in pool:
        reason = screen_structure(s, topo)
        if reason:
            rejected.append((s, reason))
            raw_fate[id(s)] = f"rejected: {reason}"
        else:
            intact.append(s)
    candidates, links = deduplicate(intact, topo.permutations)
    cand_tag = {id(c): f"cand_{k + 1:03d}" for k, c in enumerate(candidates)}
    for c in candidates:
        raw_fate[id(c)] = f"characterized as {cand_tag[id(c)]}"
    for item, kept, de, rmsd in links:
        raw_fate[id(item)] = f"duplicate of {cand_tag[id(kept)]} (dE {de:.3f} kcal/mol, RMSD {rmsd:.3f} A)"
    audit = [["raw", s.sources[0], f"{s.energy_eh:.8f}", "", raw_fate[id(s)]] for s in pool]
    write_xyz(run_dir / "all_raw_structures.xyz", pool, [f"{' '.join(s.sources)} {s.energy_eh:.10f}" for s in pool])
    log(f"  raw: {len(pool)}  rejected: {len(rejected)}  unique candidates: {len(candidates)}")

    # Phase 4: characterize every candidate (one at a time, single-core, <= 10 min each)
    log(f"\nPHASE 4: CHARACTERIZATION (TightOpt + NumFreq, one at a time, "
        f"limit {PER_ISOMER_TIMEOUT_S // 60} min each)")
    characterized = [characterize(c, cand_tag[id(c)], engine, jobs_dir, settings) for c in candidates]
    every_char: list[Structure3D] = list(characterized)
    char_fate: dict[int, str] = {}

    # Phase 4b: follow every saddle point downhill to the minima it connects
    resolved: list[Structure3D] = []
    for s in characterized:
        if s.status.startswith("SADDLE POINT"):
            log(f"  [{Path(s.char_job).name}] saddle point: stepping off along the imaginary mode")
            downhill = follow_saddle(s, engine, jobs_dir, settings)
            if downhill:
                rejected.append((s, f"transition state, {s.status}; followed downhill in both directions"))
                char_fate[id(s)] = ("transition state; followed downhill to "
                                    + " and ".join(Path(d.char_job).name for d in downhill))
                every_char += downhill
                resolved += downhill
                continue
        resolved.append(s)

    # Phase 5: re-screen, final dedup, energy window, export
    log("\nPHASE 5: FINAL DEDUPLICATION & EXPORT")
    final_pool = []
    for s in resolved:
        reason = screen_structure(s, topo)
        if reason:
            rejected.append((s, reason + " (after TightOpt)"))
            char_fate[id(s)] = f"rejected: {reason} (after TightOpt)"
        else:
            final_pool.append(s)
    isomers, links = deduplicate(final_pool, topo.permutations)
    if not isomers:
        raise RuntimeError("Every structure was rejected; see rejected reasons in run_log.txt.")
    emin = isomers[0].energy_eh
    kept = [s for s in isomers if (s.energy_eh - emin) * HARTREE_TO_KCAL <= settings.window_kcal]
    for s in isomers:
        if s not in kept:
            rejected.append((s, f"above {settings.window_kcal} kcal/mol window"))
            char_fate[id(s)] = f"above {settings.window_kcal} kcal/mol window"
    for k, s in enumerate(kept):
        s.label = f"isomer-{k:02d}"
        char_fate[id(s)] = s.label
    for item, unique, de, rmsd in links:
        target = unique.label or char_fate.get(id(unique), "?")
        char_fate[id(item)] = (f"duplicate of {target} ({Path(unique.char_job).name}; "
                               f"dE {de:.3f} kcal/mol, RMSD {rmsd:.3f} A)")
    audit += [["characterized", Path(s.char_job).name, f"{s.energy_eh:.8f}", s.status, char_fate.get(id(s), "?")]
              for s in every_char]
    csv_path = export_isomers(kept, rejected, audit, iso_dir, topo, settings)

    elapsed = (time.time() - t_start) / 60
    manifest = {
        "tool": str(Path(__file__).resolve()), "tool_version": TOOL_VERSION,
        "molecule": settings.name, "charge": settings.charge, "multiplicity": settings.multiplicity,
        "method": "GFN2-xTB (! NATIVE-GFN2-XTB)", "orca_version": engine.orca_version,
        "orca_binary": str(engine.orca_bin), "opi_version": "2.0",
        "goat_nprocs": GOAT_NPROCS, "goat_workers": GOAT_NWORKERS, "parallel_jobs": 1,
        "seeds": {s.name: {"file": str(s.path), "frame": s.frame,
                           "sha1": hashlib.sha1((jobs_dir / "seeds" / f"{s.name}.xyz").read_bytes()).hexdigest(),
                           **({"note": s.note} if s.note else {})}
                  for s in seeds},
        "fragments": [[topo.fragment_name(k), f] for k, f in enumerate(topo.fragments)],
        "equivalent_atoms": topo.equivalent_groups,
        "goat_global_iter": [min_iter, max_iter], "goat_maxen_kcal": settings.window_kcal,
        "per_isomer_timeout_s": PER_ISOMER_TIMEOUT_S,
        "dedup": {"energy_kcal": DEDUP_ENERGY_TOL_KCAL, "rmsd_A": DEDUP_RMSD_TOL_A},
        "counts": {"raw": len(pool), "rejected": len(rejected), "candidates": len(candidates),
                   "isomers": len(kept)},
        "elapsed_min": round(elapsed, 1), "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"\n[DONE] {len(kept)} isomers in {elapsed:.1f} min -> {csv_path.parent}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:                        # main() has already released the lock
        kill_all_running()
        log("\n[ABORTED] Ctrl+C: all ORCA jobs were stopped. Re-run to resume.")
        sys.exit(130)
    except Exception:
        kill_all_running()
        log("\n[ERROR] " + traceback.format_exc())   # keep the reason in run_log.txt
        sys.exit(1)
