#!/usr/bin/env python3
"""
================================================================================
CoChem / TOPOS: microwave-spectrum predictions for the isomers of a molecule
or weakly bound complex (written for HCOOH-H2S; shared since v1.2)
ORCA 6.1.1 + OPI 2.0, revDSD-PBEP86-D4 + B3LYP-D4 hybrid scheme
File: ftmw_spectroscopy/mw_predict_pipeline.py   (index of shared tools: README.md)
================================================================================
1. PURPOSE:
   Turns every filled isomer file isomer-NN.xyz in the work folder into the
   numbers a microwave spectroscopist needs to find its lines: ground-state
   rotational constants B_0 (A, B, C), quartic centrifugal-distortion constants
   (Watson A and S reductions), dipole components mu_a, mu_b, mu_c, ready-made
   Pickett SPCAT/SPFIT decks (.par/.var/.int) and a predicted stick spectrum.
   Any molecule or complex of H, C, N, O, F, Si, P, S, Cl, Br, I: the lowest-
   numbered filled file defines it (bond graph, fragments, equivalent atoms),
   every other file must be the same molecule (any atom order: renumbered).
   --stages runs a subset (e.g. S0,S1,S2a,S2b for revDSD structures and
   energies only; S4 = the analysis, only if listed); a later full run
   reuses every finished stage.

2. METHOD (why a hybrid):
   B_0 = B_e(revDSD) + Delta B_vib(B3LYP)
   - B_e from a very tight revDSD-PBEP86-D4/2021/jun-cc-pV(T+d)Z equilibrium
     structure: double hybrids give the best structures of affordable methods.
   - Delta B_vib = B_0 - B_e and the distortion constants from B3LYP-D4/jun-
     cc-pV(D+d)Z VPT2 at the B3LYP minimum. ORCA 6.1.1 cannot run VPT2 for a
     double hybrid (VPT2 needs analytic Hessians; double hybrids have none;
     with NumFreq VPT2 is silently skipped), so the vibrational correction
     comes from the hybrid functional - the Barone-group practice, which also
     uses a double-zeta basis for it: the correction is a small difference
     (~1-2 % of B), far less basis-sensitive than B_e itself. jun-cc-pV(D+d)Z
     equals jun-cc-pVDZ on H, C, O and adds the tight d function on S (user
     decision 2026-10-25; ~10x cheaper VPT2 than jun-cc-pV(T+d)Z).
   - Dipoles from the relaxed revDSD density (%mp2 Density relaxed): the
     unrelaxed or SCF-only dipole of a double hybrid is not the right one.

3. STAGES (one isomer at a time, each in isomer-NN/<stage>/, resumable):
   S0   r2SCAN-3c TightOpt + Freq        pre-optimization, starting Hessian,
                                         first check that it is a minimum
   S1   revDSD very tight Opt            equilibrium structure -> B_e
   S2a  revDSD single point, relaxed density   dipole mu_a, mu_b, mu_c
   S2b  revDSD NumFreq                   harmonic frequencies, ZPE, minimum check
   S3a  B3LYP-D4 very tight Opt          B3LYP minimum (VPT2 needs its own minimum)
   S3b  B3LYP-D4 VPT2                    alpha constants, Delta B_vib, quartic constants
   S4   Python                           B_0, principal-axis dipoles with sign,
                                         inertial defect, isotopologues, Pickett
                                         decks, stick spectrum, provenance

4. MACHINE LOAD & SAFETY:
   - 8 MPI ranks per job (%pal nprocs 8, %maxcore 3000 = 24 GB of 64), one ORCA
     job at a time, below-normal priority. ORCA runs inside a Windows Job
     Object, so Ctrl+C, an error, a timeout or the death of this script kills
     ORCA, mpiexec, smpd and every MPI rank - and nothing else.
     Double-hybrid steps last minutes, so programs start rarely (unlike the
     tiny-molecule GOAT runs, where start-up storms crashed the PC).
   - A machine-wide lock file (shared with goat_isomer_search.py) refuses to
     start while another ORCA driver is running.
   - A stall watchdog kills a job whose folder has not changed for 60 min.
   - S3b asks ORCA for no Pickett file (%output Pickettname): in parallel it
     deadlocked VPT2 in the 2026-10-25 smoke test. Parallel displacement
     groups are off unless --vpt2-groups N is given.
   - --smoke uses 4 cores and small basis sets: run it first.

5. FILES (in the work folder: --work-dir, default the folder of this script):
   isomer-NN.xyz          input; an unfilled stub (comment starts with STUB, no
                          coordinates) is skipped with a message
   job.toml               optional: name = "HCOOH-N2O", charge = 0, multiplicity = 1
   isomer-NN/S0_r2scan3c ... S3b_b3lyp_vpt2/   ORCA jobs
   isomer-NN/results/     constants (.csv/.json), Pickett decks (A and S
                          reduction), isotopologues, stick spectrum, provenance
   mw_summary.csv         one row per isomer, plus the revDSD duplicate check
   mw_run_log.txt         everything printed

6. USAGE:
   python hcooh-hsh-mw-predict.py --dry-run      write every ORCA input, run nothing
   python hcooh-hsh-mw-predict.py --smoke        whole chain, small basis, 4 cores
   python hcooh-hsh-mw-predict.py                production, every filled isomer
   ... --work-dir D:\\path\\to\\folder            isomer files and outputs in that folder
   ... --stages S0,S1,S2a,S2b                     only these stages (others reused, never run)
   ... --only 00,02                               selected isomers (number as in the file name)
   ... --calibrate                                one timed revDSD gradient -> time estimate
   ... --analyze-only                             redo S4 from finished jobs, start no ORCA
   Re-running resumes: a job whose input is unchanged and whose output says
   ORCA TERMINATED NORMALLY is reused; an interrupted NumFreq restarts from its
   finished displacements. Exit code 0 = every requested stage of every filled
   isomer finished (a queue can rely on it), 2 = some did not.

7. NOT IN THIS TIER (stated in every results README, not silently skipped):
   counterpoise-corrected geometry, sextic distortion constants, tunnelling
   splittings and nuclear-spin statistical weights of large-amplitude motions
   (e.g. H2S internal rotation in HCOOH-H2S), 14N quadrupole hyperfine; VPT2
   alpha constants of the soft intermolecular modes are unreliable (flagged).

8. REQUIREMENTS: ORCA 6.1.1, Python >= 3.11, numpy, orca-pi 2.0.
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
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from opi.core import Calculator
from opi.execution.core import Runner
from opi.input.simple_keywords import (Approximation, AuxBasisSet, BasisSet, Dft, DispersionCorrection, Grid,
                                       Miscellaneous, Opt, Scf, SimpleKeyword, Task)
from opi.input.structures import XyzFile
from opi.lib.orca_binary import OrcaBinary
from opi.output.core import Output


# ==============================================================================
# 0. CENTRALIZED CONFIGURATION
# ==============================================================================
SCRIPT_VERSION: str = "1.2 (2026-10-25; shared: any molecule, --stages, --work-dir)"
HERE: Path = Path(__file__).resolve().parent
WORK_DIR: Path = HERE                   # --work-dir: the isomer files and every output live here
ISOMER_FILE_RE = re.compile(r"^isomer-(\d+)\.xyz$")
STAGE_KEYS: list[str] = ["S0", "S1", "S2a", "S2b", "S3a", "S3b"]

NPROCS: int = 8                         # production: 8 MPI ranks per ORCA job
SMOKE_NPROCS: int = 4                   # tests start at <= 4 cores
MAXCORE_MB: int = 3000                  # per rank: 8 x 3 GB = 24 GB of 64 GB
LOCK_FILE: Path = Path(tempfile.gettempdir()) / "orca_one_job_at_a_time.lock"
STALL_S: int = 3600                     # no file written for 60 min = hung job
STALL_CHECK_EVERY_S: int = 60
STAGE_TIMEOUT_H: dict[str, float] = {"S0": 3, "S1": 24, "S2a": 6, "S2b": 72, "S3a": 24, "S3b": 72, "CAL": 3}

# Very tight geometry convergence (method matrix F03 Sec. 4.4), Eh and bohr
GEOM_TOL: dict[str, float] = {"TolE": 1e-7, "TolRMSG": 3e-6, "TolMaxG": 1e-5, "TolRMSD": 5e-5, "TolMaxD": 1e-4}
GEOM_MAXITER: int = 200
NUMFREQ_INCREMENT_BOHR: float = 0.005   # two-sided (central) differences
VPT2_ANHARM_DISP: float = 0.05          # normal-coordinate displacement for the cubic/quartic force field
VPT2_HESSIAN_CUTOFF: float = 1e-12
Z_TOL: float = 1e-14                    # CP-SCF convergence for the VPT2 Hessians
# 2026-10-25 smoke test (ORCA 6.1.1, Windows, MS-MPI): with %output Pickettname, parallel VPT2 hung
# right after "writing data to file in Pickett format" (orca_prop_mpi ranks at 0 % CPU), with and without
# nprocs_group and with either MinimiseOrcaPrint. Every earlier VPT2 run on this PC was serial. S3b
# therefore writes no ORCA Pickett file: the script parses the .out and writes its own decks.
# Ranks per displaced-Hessian group: off by default; --vpt2-groups N opts in.
VPT2_NPROCS_GROUP: int | None = None
VPT2_MINIMISE_PRINT: str = "false"      # false: every sub-step is logged in the .out (diagnosable)

# Guards (method matrix F01 Sec. 16)
IMAG_NOISE_CM1: float = 15.0            # |imag| below this = numerical noise (flag, regrid)
SOFT_MODE_CM1: float = 100.0            # VPT2 alpha of modes below this are unreliable (flag)
DARK_DIPOLE_D: float = 0.1              # |mu_g| below this: g-type lines too weak to rely on
MASS_CHECK_REL: float = 1e-4            # our B_e vs ORCA's printed B_e (same masses?)
DFT_DUP = {"energy_kcal": 0.05, "rot_rel": 0.001, "rmsd_A": 0.125}   # matrix Stage-B rule

# Uncertainty windows written into the .var files [E]: SPCAT turns them into line errors
B0_WINDOW_REL: float = 0.015            # +-1.5 % on A, B, C (floppy vdW complex: 1-2 %)
QUARTIC_WINDOW_REL: float = 0.30        # +-30 % on the quartic constants

# Stick spectrum (built-in Watson engine; Pickett decks carry the same model)
SPEC_T_K: float = 2.0                   # supersonic-jet rotational temperature
SPEC_FMIN_MHZ, SPEC_FMAX_MHZ = 2000.0, 20000.0       # method matrix F01 Sec. 18: 2-20 GHz at 2 K
SPEC_JMAX: int = 15
SPEC_REL_CUTOFF: float = 1e-4           # keep lines above this fraction of the strongest

# Physical constants (CODATA 2018)
HARTREE_TO_KCAL: float = 627.5094740631
HARTREE_TO_CM1: float = 219474.63136320
CM1_TO_MHZ: float = 29979.2458
ROT_CONST_MHZ_AMU_A2: float = 6.62607015e-34 / (8 * math.pi ** 2 * 1.66053906660e-27 * 1e-20) / 1e6  # 505379.009
AU_TO_DEBYE: float = 2.541746473
K_B_MHZ_PER_K: float = 20836.61912                   # k/h in MHz/K
# Isotope masses (u) from Mendeleev 1.2.0 (CoChem Hub rule); the most abundant isotope of
# each element is the parent and matches ORCA's !Mass2016 (manual Sec. 4.6.1)
ISOTOPE_MASS: dict[str, float] = {
    "H": 1.007825031898, "D": 2.014101777844, "C": 12.0, "13C": 13.00335483534,
    "N": 14.00307400425, "15N": 15.00010889827, "O": 15.99491461926, "18O": 17.99915961214,
    "F": 18.99840316207, "Si": 27.97692653442, "29Si": 28.97649466434, "P": 30.97376199768,
    "S": 31.97207117354, "34S": 33.967867011, "Cl": 34.968852694, "37Cl": 36.965902573,
    "Br": 78.918337574, "81Br": 80.916288197, "I": 126.904472592,
}
# Isotopologues are generated from these: one single substitution per atom, plus the fully
# deuterated form of every group of equivalent terminal H atoms (D2S, CD3, ...)
RARE_ISOTOPE: dict[str, str] = {"H": "D", "C": "13C", "N": "15N", "O": "18O", "Si": "29Si", "S": "34S",
                                "Cl": "37Cl", "Br": "81Br"}
COVALENT_RADII_A: dict[str, float] = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11,
                                      "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39}   # Cordero 2008
BOND_SCALE: float = 1.25
HB_ACCEPTORS: set[str] = {"N", "O", "F", "S", "Cl", "Br"}
HB_CONTACT_A: float = 2.7               # H...acceptor in another fragment closer than this: H-bond donor


@dataclass(frozen=True)
class Level:
    """One complete set of model chemistries; 'smoke' swaps in small basis sets only."""

    name: str
    ncores: int
    dh_basis: tuple[SimpleKeyword, ...]     # revDSD: orbital, RI-MP2 (/C), RIJ (/J)
    hyb_basis: tuple[SimpleKeyword, ...]    # B3LYP: orbital, RIJ (/J)
    folder_suffix: str


PRODUCTION = Level("production", NPROCS,
                   (BasisSet.JUN_CC_PV_TPLUSD_Z, AuxBasisSet.AUG_CC_PVTZ_C, AuxBasisSet.DEF2_J),
                   (BasisSet.JUN_CC_PV_DPLUSD_Z, AuxBasisSet.DEF2_J), "")     # B3LYP S3a/S3b: (D+d)
SMOKE = Level("smoke", SMOKE_NPROCS,
              (BasisSet.DEF2_SVP, AuxBasisSet.DEF2_SVP_C, AuxBasisSet.DEF2_J),
              (BasisSet.DEF2_SVP, AuxBasisSet.DEF2_J), "smoke")

STAGES: list[tuple[str, str]] = [        # (key, job folder = ORCA basename)
    ("S0", "S0_r2scan3c"), ("S1", "S1_revdsd_opt"), ("S2a", "S2a_revdsd_dipole"),
    ("S2b", "S2b_revdsd_numfreq"), ("S3a", "S3a_b3lyp_opt"), ("S3b", "S3b_b3lyp_vpt2"),
]
STAGE_DIR: dict[str, str] = dict(STAGES)


# ==============================================================================
# 1. LOGGING, MACHINE-WIDE LOCK & PROCESS CONTROL
# ==============================================================================
_LOG_FILE: Path | None = None
_RUNNING: set[subprocess.Popen] = set()     # at most one ORCA job at a time
_NO_ORCA: bool = False                      # --analyze-only: never start ORCA


def log(msg: str = "") -> None:
    """Prints a line and appends it to mw_run_log.txt."""
    print(msg, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


_JOBS: dict[subprocess.Popen, int] = {}    # Windows Job Object of each running ORCA


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k


def attach_kill_job(proc: subprocess.Popen) -> None:
    """
    Windows: puts ORCA into a Job Object that kills every process in it when
    terminated or closed. Programs ORCA starts afterwards (mpiexec, smpd, MPI
    ranks, module programs) join the job, so a kill reaches all of them - in
    the 2026-10-25 smoke test a process-tree kill of orca.exe left mpiexec,
    smpd and four ranks behind. If this script dies, the job handle closes and
    ORCA goes with it. Only processes started by this ORCA are ever touched.
    """
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("IoInfo", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k = _kernel32()
    job = k.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.LimitFlags = 0x2000 | 0x800        # KILL_ON_JOB_CLOSE | BREAKAWAY_OK (never blocks MPI's own launches)
    if job and k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) \
            and k.AssignProcessToJobObject(job, int(proc._handle)):
        _JOBS[proc] = job
    else:
        log(f"  WARNING: no kill job for ORCA (Windows error {ctypes.get_last_error()}); tree kill only")
        if job:
            k.CloseHandle(job)


def close_kill_job(proc: subprocess.Popen) -> None:
    """Closes the job when ORCA is done; anything ORCA left behind is killed with it."""
    job = _JOBS.pop(proc, None)
    if job is not None:
        _kernel32().CloseHandle(job)


def kill_tree(proc: subprocess.Popen) -> None:
    """Kills ORCA, mpiexec, the MPI ranks and every module program they started."""
    job = _JOBS.get(proc)
    if job is not None:
        _kernel32().TerminateJobObject(job, 1)
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
    """Machine-wide 'one ORCA driver at a time' lock, shared with goat_isomer_search.py."""
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
# 2. ISOMER FILES: DISCOVERY, STUBS, THE MOLECULE
# ==============================================================================
@dataclass
class Isomer:
    """One isomer-NN.xyz file and everything derived from it."""

    label: str                          # file stem, e.g. "isomer-00"
    number: int
    path: Path
    symbols: list[str] = field(default_factory=list)
    coords: np.ndarray | None = None
    stub_reason: str = ""
    note: str = ""                      # e.g. atoms renumbered to the reference order
    flags: list[str] = field(default_factory=list)
    stage_status: dict[str, str] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    input_sha1: dict[str, str] = field(default_factory=dict)
    requested_done: bool = False        # every stage in --stages finished (ok or reused)


@dataclass
class JobSettings:
    """Optional <work-dir>/job.toml: name (titles, file stems), charge, multiplicity."""

    name: str = ""
    charge: int = 0
    multiplicity: int = 1


@dataclass
class Molecule:
    """
    Bond graph of the reference isomer (the lowest-numbered filled file), its
    fragments (connected components) and its groups of equivalent terminal
    atoms (same element on the same centre, e.g. the two H of H2S). Every
    other isomer must be the same molecule; a file listing its atoms in
    another order is renumbered to the reference order, so atom labels mean
    the same thing in every file and table.
    """

    symbols: list[str]
    bonds: set[tuple[int, int]]
    fragments: list[list[int]]
    equivalent_groups: list[list[int]]
    perms: list[np.ndarray]
    name: str

    @classmethod
    def from_structure(cls, symbols: list[str], coords: np.ndarray, name: str = "") -> Molecule:
        bonds = bonded_pairs(symbols, coords)
        nb = neighbor_sets(len(symbols), bonds)
        fragments, seen = [], set()
        for start in range(len(symbols)):
            if start not in seen:
                comp, stack = set(), [start]
                while stack:
                    a = stack.pop()
                    if a not in comp:
                        comp.add(a)
                        stack.extend(nb[a] - comp)
                seen |= comp
                fragments.append(sorted(comp))
        groups = []
        for centre in range(len(symbols)):
            terminal: dict[str, list[int]] = {}
            for j in sorted(nb[centre]):
                if len(nb[j]) == 1:
                    terminal.setdefault(symbols[j], []).append(j)
            groups += [g for g in terminal.values() if len(g) >= 2]
        perms = []
        for choice in itertools.islice(itertools.product(*(itertools.permutations(g) for g in groups)), 720):
            p = np.arange(len(symbols))
            for g, image in zip(groups, choice):
                p[g] = image
            perms.append(p)
        name = name or "-".join(hill_formula([symbols[i] for i in f]) for f in fragments)
        return cls(symbols, bonds, fragments, groups, perms, name)

    def fragment_of(self, i: int) -> int:
        return next(k for k, f in enumerate(self.fragments) if i in f)

    def partners(self, i: int) -> list[int]:
        return sorted(b if a == i else a for a, b in self.bonds if i in (a, b))


JOB: JobSettings = JobSettings()
MOL: Molecule | None = None             # set by discover_isomers()


def hill_formula(symbols: list[str]) -> str:
    counts = {s: symbols.count(s) for s in sorted(set(symbols))}
    first = [e for e in ("C", "H") if "C" in counts and e in counts]
    order = first + [e for e in counts if e not in first]
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def neighbor_sets(n: int, bonds: set[tuple[int, int]]) -> list[set[int]]:
    nb: list[set[int]] = [set() for _ in range(n)]
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    return nb


def atom_mapping(ref_symbols: list[str], ref_bonds: set[tuple[int, int]], symbols: list[str],
                 bonds: set[tuple[int, int]]) -> list[int] | None:
    """
    order[r] = the atom of the new structure that plays reference atom r (same
    element, same bonds), or None if the two are different molecules.
    Backtracking over element, degree and bonds to atoms already matched.
    """
    n = len(ref_symbols)
    if len(symbols) != n or sorted(symbols) != sorted(ref_symbols) or len(bonds) != len(ref_bonds):
        return None
    rn, tn = neighbor_sets(n, ref_bonds), neighbor_sets(n, bonds)
    sig = lambda sym, nb, i: (sym[i], len(nb[i]), tuple(sorted(sym[j] for j in nb[i])))   # noqa: E731
    cands = [[t for t in range(n) if sig(symbols, tn, t) == sig(ref_symbols, rn, r)] for r in range(n)]
    match, used = [-1] * n, [False] * n

    def extend(r: int) -> bool:
        if r == n:
            return True
        for t in cands[r]:
            if not used[t] and all((match[x] in tn[t]) == (x in rn[r]) for x in range(r)):
                match[r], used[t] = t, True
                if extend(r + 1):
                    return True
                match[r], used[t] = -1, False
        return False

    return match if extend(0) else None


def load_job_settings(folder: Path) -> JobSettings:
    toml_file = folder / "job.toml"
    if not toml_file.exists():
        return JobSettings()
    import tomllib
    data = tomllib.loads(toml_file.read_text(encoding="utf-8"))
    goat_keys = {"window_kcal", "goat_min_iter", "goat_max_iter"}      # _tools\goat_isomer_search.py: ignored here
    unknown = sorted(set(data) - {"name", "charge", "multiplicity"} - goat_keys)
    if unknown:
        raise SystemExit(f"{toml_file}: unknown key(s) {', '.join(unknown)}; allowed: name, charge, multiplicity "
                         f"(and the GOAT tool's {', '.join(sorted(goat_keys))}, ignored)")
    return JobSettings(str(data.get("name", "")), int(data.get("charge", 0)), int(data.get("multiplicity", 1)))


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
        try:
            for line in lines[i + 2: i + 2 + n]:
                parts = line.split()
                symbols.append(re.sub(r"[^A-Za-z]", "", parts[0]).capitalize())
                xyz.append([float(v) for v in parts[1:4]])
        except (IndexError, ValueError):
            break
        frames.append((comment, symbols, np.array(xyz)))
        i += 2 + n
    return frames


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def bonded_pairs(symbols: list[str], coords: np.ndarray) -> set[tuple[int, int]]:
    """Covalent bonds from distances: d < BOND_SCALE * (r_i + r_j)."""
    pairs = set()
    for i, j in itertools.combinations(range(len(symbols)), 2):
        cutoff = BOND_SCALE * (COVALENT_RADII_A[symbols[i]] + COVALENT_RADII_A[symbols[j]])
        if np.linalg.norm(coords[i] - coords[j]) < cutoff:
            pairs.add((i, j))
    return pairs


def discover_isomers(folder: Path, only: set[int] | None) -> list[Isomer]:
    """
    Every isomer-<digits>.xyz in the folder, sorted by number; stubs and
    unusable files marked. The lowest-numbered filled file (of all files, so
    --only never changes it) defines the molecule (global MOL); every other
    file must be the same molecule and is renumbered to its atom order.
    """
    global MOL
    found = []
    for path in folder.iterdir():
        m = ISOMER_FILE_RE.match(path.name)
        if not m or not path.is_file():
            continue
        iso = Isomer(label=path.stem, number=int(m.group(1)), path=path)
        frames = read_xyz_frames(path)
        if not frames:
            first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:2]
            iso.stub_reason = ("unfilled stub" if first and first[0].strip().upper().startswith("STUB")
                               else "no complete xyz frame")
        else:
            _, symbols, coords = frames[0]
            unknown = sorted({s for s in symbols if s not in ISOTOPE_MASS or s not in COVALENT_RADII_A})
            if unknown:
                iso.stub_reason = f"element(s) {', '.join(unknown)} not in ISOTOPE_MASS / COVALENT_RADII_A"
            else:
                iso.symbols, iso.coords = symbols, coords
        found.append(iso)
    found.sort(key=lambda x: x.number)
    filled = [iso for iso in found if not iso.stub_reason]
    if filled:
        ref = filled[0]
        MOL = Molecule.from_structure(ref.symbols, ref.coords, JOB.name)
        for iso in filled[1:]:
            bonds = bonded_pairs(iso.symbols, iso.coords)
            if iso.symbols == MOL.symbols and bonds == MOL.bonds:
                continue
            order = atom_mapping(MOL.symbols, MOL.bonds, iso.symbols, bonds)
            if order is None:
                iso.stub_reason = (f"not the same molecule as {ref.label} ({MOL.name}): different atoms or "
                                   "covalent bonds")
            else:
                iso.symbols, iso.coords = list(MOL.symbols), iso.coords[order]
                iso.note = f"atoms renumbered to the order of {ref.label} (new atom k = old atom {order})"
    return [iso for iso in found if only is None or iso.number in only]


# ==============================================================================
# 3. GEOMETRY & ROTATIONAL PHYSICS
# ==============================================================================
def masses_for(symbols: list[str], substitution: dict[int, str] | None = None) -> np.ndarray:
    """Atomic masses (u); the parent uses the most abundant isotopes, substitution maps atom -> isotope."""
    m = np.array([ISOTOPE_MASS[s] for s in symbols])
    for i, iso in (substitution or {}).items():
        m[i] = ISOTOPE_MASS[iso]
    return m


def hbond_donors(coords: np.ndarray, mol: Molecule) -> set[int]:
    """H atoms closer than HB_CONTACT_A to an acceptor atom (N, O, F, S, Cl, Br) of another fragment."""
    out = set()
    for h, s in enumerate(mol.symbols):
        if s == "H":
            k = mol.fragment_of(h)
            if any(mol.symbols[a] in HB_ACCEPTORS and mol.fragment_of(a) != k
                   and np.linalg.norm(coords[h] - coords[a]) < HB_CONTACT_A for a in range(len(mol.symbols))):
                out.add(h)
    return out


def isotopologues(mol: Molecule, coords: np.ndarray) -> list[tuple[str, dict[int, str]]]:
    """
    Parent; every single substitution by the common rare isotope (13C, 15N,
    18O, 34S, D, ...) of each atom - atom labels are the reference numbering,
    and an H that donates an H-bond in this structure is marked; the fully
    deuterated form of every group of equivalent terminal H (D2S, CD3, ...).
    """
    donors = hbond_donors(coords, mol)
    out: list[tuple[str, dict[int, str]]] = [("parent", {})]
    for i, s in enumerate(mol.symbols):
        if s in RARE_ISOTOPE:
            out.append((f"{RARE_ISOTOPE[s]}@{s}{i}" + (" (H-bond donor)" if i in donors else ""),
                        {i: RARE_ISOTOPE[s]}))
    for g in mol.equivalent_groups:
        if mol.symbols[g[0]] == "H":
            c = mol.partners(g[0])[0]
            out.append((f"D{len(g)}@{mol.symbols[c]}{c}", {i: "D" for i in g}))
    return out


def principal_axes(coords: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Centre of mass, principal moments Ia <= Ib <= Ic (u A^2) and the axis
    matrix V (columns a, b, c in the input frame). Sign convention: the
    component of largest magnitude of the a and b axis vectors is positive,
    and c = a x b (right-handed). Axis signs are a convention: only the sign of
    a dipole component RELATIVE to the atoms is physical, so the convention is
    stated next to every signed dipole.
    """
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    r = coords - com
    inertia = np.zeros((3, 3))
    for mi, ri in zip(masses, r):
        inertia += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    moments, vecs = np.linalg.eigh(inertia)
    for k in (0, 1):
        if vecs[np.argmax(np.abs(vecs[:, k])), k] < 0:
            vecs[:, k] *= -1.0
    vecs[:, 2] = np.cross(vecs[:, 0], vecs[:, 1])
    return com, moments, vecs


def rotational_constants(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """A >= B >= C in MHz."""
    _, moments, _ = principal_axes(coords, masses)
    return np.array([ROT_CONST_MHZ_AMU_A2 / m if m > 1e-8 else 0.0 for m in moments])


def inertial_analysis(coords: np.ndarray, masses: np.ndarray) -> dict[str, float]:
    """Inertial defect Delta = Ic - Ia - Ib and planar moments P_gg (u A^2); Ray's kappa."""
    _, (ia, ib, ic), _ = principal_axes(coords, masses)
    a, b, c = (ROT_CONST_MHZ_AMU_A2 / x for x in (ia, ib, ic))
    return {"Ia": ia, "Ib": ib, "Ic": ic, "inertial_defect": ic - ia - ib,
            "Paa": (ib + ic - ia) / 2, "Pbb": (ia + ic - ib) / 2, "Pcc": (ia + ib - ic) / 2,
            "kappa": (2 * b - a - c) / (a - c)}


def dihedral_deg(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def hcooh_torsion(coords: np.ndarray, mol: Molecule) -> float | None:
    """H-O-C=O dihedral of a formic-acid fragment (CH2O2 with C bonded to OH, =O and H), else None."""
    for frag in mol.fragments:
        if hill_formula([mol.symbols[i] for i in frag]) != "CH2O2":
            continue
        c = next(i for i in frag if mol.symbols[i] == "C")
        o_h = [o for o in mol.partners(c) if mol.symbols[o] == "O" and len(mol.partners(o)) == 2]
        o_c = [o for o in mol.partners(c) if mol.symbols[o] == "O" and len(mol.partners(o)) == 1]
        if len(o_h) == 1 and len(o_c) == 1:
            h_o = next(h for h in mol.partners(o_h[0]) if mol.symbols[h] == "H")
            return dihedral_deg(coords[h_o], coords[o_h[0]], coords[c], coords[o_c[0]])
    return None


def describe_geometry(coords: np.ndarray, symbols: list[str], mol: Molecule) -> dict[str, str]:
    """The two closest contacts between fragments; cis/trans of a formic-acid fragment if there is one."""
    pairs = sorted((float(np.linalg.norm(coords[i] - coords[j])), i, j)
                   for fa, fb in itertools.combinations(mol.fragments, 2) for i in fa for j in fb)
    tau = hcooh_torsion(coords, mol)
    return {"HCOOH_form": "" if tau is None else ("trans" if abs(tau) < 90.0 else "cis"),
            "H-O-C=O_deg": "" if tau is None else f"{tau:.1f}",
            "closest_contacts_A": "; ".join(f"{symbols[i]}{i}...{symbols[j]}{j} {d:.3f}" for d, i, j in pairs[:2])}


def best_rmsd(a: np.ndarray, b: np.ndarray, perms: list[np.ndarray]) -> float:
    """Minimum RMSD over rotation, reflection and the given atom permutations (A)."""
    q = b - b.mean(axis=0)
    best = np.inf
    for perm in perms:
        p = a[perm] - a[perm].mean(axis=0)
        s = np.linalg.svd(p.T @ q, compute_uv=False)
        best = min(best, float(np.sqrt(max(((p * p).sum() + (q * q).sum() - 2.0 * s.sum()) / len(a), 0.0))))
    return best


# ==============================================================================
# 4. ORCA INPUTS (OPI 2.0) & EXECUTION
# ==============================================================================
@dataclass
class Engine:
    orca_bin: Path
    orca_version: str


def resolve_engine(check_version: bool = True) -> Engine:
    runner = Runner()
    orca_bin = runner.get_orca_binary(OrcaBinary.ORCA)
    if not check_version and not os.environ.get("OPI_ORCA"):
        log(f"[PRE-FLIGHT] ORCA 6.1.1 (version check skipped in dry-run/analyze-only): {orca_bin}")
        return Engine(orca_bin, "6.1.1")
    runner.check_version()
    version = str(runner.get_version())
    log(f"[PRE-FLIGHT] ORCA {version}: {orca_bin}")
    if not version.startswith("6.1"):
        log(f"[PRE-FLIGHT] WARNING: written and tested for ORCA 6.1.1, found {version}.")
    return Engine(orca_bin, version)


def dh_keywords(level: Level) -> list[SimpleKeyword]:
    """revDSD-PBEP86-D4/2021 (D4 is part of the functional), RIJCOSX for the SCF, RI-MP2 (/C)."""
    return [Dft.REVDSD_PBEP86_D4_2021, *level.dh_basis, Approximation.RIJCOSX, Grid.DEFGRID3,
            Scf.VERYTIGHTSCF, Miscellaneous.MASS2016]


def hyb_keywords(level: Level) -> list[SimpleKeyword]:
    """B3LYP-D4 with ExtremeSCF: VPT2 differentiates Hessians numerically, so noise must be tiny."""
    return [Dft.B3LYP, DispersionCorrection.D4, *level.hyb_basis, Approximation.RIJCOSX, Grid.DEFGRID3,
            Scf.EXTREMESCF, Miscellaneous.MASS2016]


def geom_block(hess_name: str | None) -> str:
    lines = ["%geom"] + [f"    {k} {v:g}" for k, v in GEOM_TOL.items()] + [f"    MaxIter {GEOM_MAXITER}"]
    if hess_name:
        lines += ["    InHess Read", f'    InHessName "{hess_name}"']
    return "\n".join(lines + ["end"])


def new_calculator(job_dir: Path, base: str, level: Level) -> Calculator:
    job_dir.mkdir(parents=True, exist_ok=True)
    calc = Calculator(basename=base, working_dir=job_dir, version_check=False)
    calc.json_via_input = False
    calc.input.ncores = level.ncores
    calc.input.memory = MAXCORE_MB
    return calc


def build_stage(stage: str, job_dir: Path, level: Level, restart_numfreq: bool = False) -> Calculator:
    """The ORCA input of one stage; block keywords OPI does not type go in verbatim."""
    base = STAGE_DIR.get(stage, "calibrate")
    calc = new_calculator(job_dir, base, level)
    add = calc.input.add_simple_keywords
    if stage == "S0":
        add(Dft.R2SCAN_3C, Opt.TIGHTOPT, Task.FREQ, Miscellaneous.MASS2016)
        calc.input.add_arbitrary_string(f"%geom\n    MaxIter {GEOM_MAXITER}\nend")
    elif stage == "S1":
        add(*dh_keywords(level), Task.OPT)
        calc.input.add_arbitrary_string(geom_block("s0.hess"))
    elif stage == "S2a":
        add(*dh_keywords(level), Task.SP)
        calc.input.add_arbitrary_string("%mp2\n    Density relaxed\nend")
    elif stage == "S2b":
        add(*dh_keywords(level), Task.NUMFREQ)
        calc.input.add_arbitrary_string(
            f"%freq\n    CentralDiff true\n    Increment {NUMFREQ_INCREMENT_BOHR:g}\n"
            + ("    Restart true\n" if restart_numfreq else "") + "end")
    elif stage == "S3a":
        add(*hyb_keywords(level), Task.OPT)
        calc.input.add_arbitrary_string(geom_block("s0.hess"))
    elif stage == "S3b":
        add(*hyb_keywords(level), Miscellaneous.VPT2)
        calc.input.add_arbitrary_string(
            f"%vpt2\n    AnharmDisp {VPT2_ANHARM_DISP:g}\n    HessianCutoff {VPT2_HESSIAN_CUTOFF:g}\n"
            f"    MinimiseOrcaPrint {VPT2_MINIMISE_PRINT}\nend\n"
            f"%method\n    Z_Tol {Z_TOL:g}\nend")              # no %output Pickettname: it hung in parallel
        if VPT2_NPROCS_GROUP and level.ncores % VPT2_NPROCS_GROUP == 0 and level.ncores > VPT2_NPROCS_GROUP:
            # displaced Hessians in parallel groups (ORCA 6.1: also on Windows); each lasts minutes
            calc.input.ncores = None
            calc.input.add_arbitrary_string(f"%pal\n    nprocs {level.ncores}\n    nprocs_group {VPT2_NPROCS_GROUP}\nend")
    elif stage == "CAL":
        add(*dh_keywords(level), Task.ENGRAD)
    else:
        raise ValueError(stage)
    return calc


_IGNORE_FOR_RESUME = re.compile(r"(?im)^\s*restart\s+true\s*$\n?")


def write_input(calc: Calculator, start_xyz: Path, strict: bool = True) -> tuple[bool, str]:
    """
    Writes <base>.inp; returns (unchanged, sha1). A fingerprint of the start
    geometry is embedded as a comment, so a changed geometry forces a rerun;
    a 'Restart true' line alone does not count as a change.
    """
    calc.structure = XyzFile(start_xyz, charge=JOB.charge, multiplicity=JOB.multiplicity, strict=strict)
    if start_xyz.exists():
        calc.input.add_arbitrary_string(f"# start.xyz sha1 {hashlib.sha1(start_xyz.read_bytes()).hexdigest()[:16]}")
    inp = calc.working_dir / f"{calc.basename}.inp"
    old = inp.read_text(encoding="utf-8") if inp.exists() else None
    calc.write_input()
    new = inp.read_text(encoding="utf-8")
    unchanged = old is not None and _IGNORE_FOR_RESUME.sub("", old) == _IGNORE_FOR_RESUME.sub("", new)
    return unchanged, hashlib.sha1(new.encode()).hexdigest()[:16]


def newest_mtime(folder: Path) -> float:
    newest = 0.0
    with os.scandir(folder) as entries:
        for entry in entries:
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                pass
    return newest


def run_orca(calc: Calculator, engine: Engine, timeout_s: float, unchanged_input: bool) -> str:
    """
    Runs ORCA on <base>.inp; returns "reused", "ok", "failed", "timeout" or
    "stalled". Own process group at below-normal priority, so a timeout,
    Ctrl+C or any error kills ORCA, mpiexec and all ranks.
    """
    output = Output(basename=calc.basename, working_dir=calc.working_dir)
    if unchanged_input and output.terminated_normally():
        return "reused"
    if _NO_ORCA:
        return "missing"
    env = os.environ.copy()
    env["PATH"] = str(engine.orca_bin.parent) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "1"
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
        attach_kill_job(proc)
        deadline = time.time() + timeout_s
        next_check = time.time() + STALL_CHECK_EVERY_S
        try:
            while proc.poll() is None:
                time.sleep(5.0)
                now = time.time()
                if now > deadline:
                    kill_tree(proc)
                    return "timeout"
                if now >= next_check:
                    next_check = now + STALL_CHECK_EVERY_S
                    if now - newest_mtime(calc.working_dir) > STALL_S:
                        kill_tree(proc)
                        return "stalled"
        except BaseException:
            kill_tree(proc)
            raise
        finally:
            _RUNNING.discard(proc)
            close_kill_job(proc)
    return "ok" if output.terminated_normally() else "failed"


# ==============================================================================
# 5. PARSERS (ORCA 6.1.1 output text)
# ==============================================================================
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def final_energy(text: str) -> float | None:
    hits = re.findall(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text, flags=re.M)
    return float(hits[-1]) if hits else None


def opt_converged(text: str) -> bool:
    return "THE OPTIMIZATION HAS CONVERGED" in text


def frequencies_and_zpe(text: str) -> tuple[list[float], float | None]:
    """Vibrational frequencies (zero modes removed) and ZPE (Eh) from the last block."""
    blocks = text.split("VIBRATIONAL FREQUENCIES")
    freqs: list[float] = []
    if len(blocks) > 1:
        block = blocks[-1].split("NORMAL MODES")[0]
        vals = [float(v) for v in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", block, flags=re.M)]
        freqs = [v for v in vals if abs(v) > 1e-3]
    zpe = re.findall(r"Zero point energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", text)
    return freqs, (float(zpe[-1]) if zpe else None)


def dipole_blocks(text: str) -> list[dict]:
    """
    Every DIPOLE MOMENT block, in order: its Method and Level labels, the total
    vector (a.u., input Cartesian frame), magnitude (D) and the unsigned
    components along the rotational axes (D, order A, B, C) when printed.
    ORCA 6.1.1 prints, for a double hybrid with %mp2 Density relaxed, an SCF
    block, then 'Method : MP2 / Level : Unrelaxed density', then
    'Method : MP2 / Level : Relaxed density' (manual: the correct double-
    hybrid properties are the ones under the relaxed MP2 heading).
    """
    starts = [m.start() for m in re.finditer(r"^DIPOLE MOMENT\s*$", text, flags=re.M)] + [len(text)]
    out = []
    for begin, end in zip(starts, starts[1:]):
        chunk = text[begin: min(end, begin + 6000)]
        tot = re.search(r"Total Dipole Moment\s*:\s*(\S+)\s+(\S+)\s+(\S+)", chunk)
        if not tot:
            continue
        method = re.search(r"^Method\s*:\s*(.+?)\s*$", chunk, flags=re.M)
        level = re.search(r"^Level\s*:\s*(.+?)\s*$", chunk, flags=re.M)
        mag = re.search(r"Magnitude \(Debye\)\s*:\s*(\S+)", chunk)
        rot = re.search(r"x,y,z \[Debye\]:\s*(\S+)\s+(\S+)\s+(\S+)", chunk)
        out.append({"method": method.group(1) if method else "", "level": level.group(1) if level else "",
                    "vector_au": np.array([float(v) for v in tot.groups()]),
                    "magnitude_D": float(mag.group(1)) if mag else None,
                    "rot_axes_D": np.array([float(v) for v in rot.groups()]) if rot else None})
    return out


def relaxed_dipole(text: str) -> dict | None:
    """The last 'Method : MP2, Level : Relaxed density' block; None if there is none (never the SCF dipole)."""
    blocks = [b for b in dipole_blocks(text) if b["method"].upper() == "MP2" and "relaxed" in b["level"].lower()
              and "unrelaxed" not in b["level"].lower()]
    return blocks[-1] if blocks else None


def final_max_gradient(text: str) -> float | None:
    """MAX gradient (Eh/bohr) of the last optimization cycle (method matrix: report it)."""
    hits = re.findall(r"^\s*MAX gradient\s+(-?\d+\.\d+)\s+\d+\.\d+\s+(?:YES|NO)", text, flags=re.M)
    return float(hits[-1]) if hits else None


def softest_mode(hess_file: Path, masses: np.ndarray) -> tuple[float, float] | None:
    """
    Lowest vibrational frequency (cm-1) and its force constant (mdyn/A) from an
    ORCA .hess file: k = 4 pi^2 c^2 omega^2 mu, with the reduced mass mu from
    the Cartesian normal-mode vector (method matrix: report the softest force constant).
    """
    try:
        lines = [ln.strip() for ln in hess_file.read_text(encoding="utf-8", errors="ignore").splitlines()]
        k = lines.index("$vibrational_frequencies")
        n = int(lines[k + 1])
        freqs = [float(lines[k + 2 + i].split()[1]) for i in range(n)]
        vib = [i for i in range(n) if abs(freqs[i]) > 1e-3]
        mode = min(vib, key=lambda i: freqs[i])
        k = lines.index("$normal_modes")
        rows, cols = (int(v) for v in lines[k + 1].split())
        modes = np.zeros((rows, cols))
        i = k + 2
        while i < len(lines) and not lines[i].startswith(("$", "#")):
            if not lines[i]:
                i += 1
                continue
            idx = [int(c) for c in lines[i].split()]
            for r in range(rows):
                parts = lines[i + 1 + r].split()
                modes[int(parts[0]), idx] = [float(v) for v in parts[1:]]
            i += 1 + rows
    except (OSError, ValueError, IndexError):
        return None
    d2 = (modes[:, mode].reshape(-1, 3) ** 2).sum(axis=1)
    mu = float((masses * d2).sum() / d2.sum())
    return freqs[mode], 5.8915e-7 * mu * freqs[mode] ** 2 * (1 if freqs[mode] >= 0 else -1)


def numfreq_restart_files(job_dir: Path, base: str) -> list[Path]:
    """Finished-displacement files (<base>.res.NNNNN.<Type>) an interrupted NumFreq restarts from."""
    return sorted(job_dir.glob(f"{base}.res.*"))


def _block_after(text: str, header: str, length: int = 20000) -> str:
    i = text.rfind(header)
    return text[i: i + length] if i >= 0 else ""


def parse_vpt2(text: str) -> dict:
    """
    VPT2 section of an ORCA 6.1.1 output (layout checked on real 6.1.1 runs,
    tests/fixtures). Everything is printed in cm-1 and converted to MHz here.
    Axes x, y, z of the VPT2 tables are the principal axes (VPT2 rotates the
    molecule into its principal-axis frame); they are mapped to a, b, c by the
    size of B_e, so the mapping does not rely on ORCA's undocumented order.
    Returns B_e, B_0 (MHz, a/b/c), alpha_i per axis (MHz, B_0 = B_e - sum/2),
    harmonic and fundamental frequencies, quartic constants of both
    reductions (MHz), R5, R6, sigma, Fermi-resonance flags and the
    anharmonic ZPE (cm-1).
    """
    rot = _block_after(text, "Rotational constants B_e and B_0 [1/cm]", 2000)
    be = re.search(r"^B_e\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", rot, flags=re.M)
    b0 = re.search(r"^B_0\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", rot, flags=re.M)
    if not (be and b0):
        raise ValueError("no 'Rotational constants B_e and B_0' block (VPT2 did not run?)")
    be_xyz = np.array([float(v) for v in be.groups()]) * CM1_TO_MHZ
    b0_xyz = np.array([float(v) for v in b0.groups()]) * CM1_TO_MHZ
    order = list(np.argsort(-be_xyz))                   # x/y/z index for a, b, c
    axis_of = {"xyz"[order[k]]: "abc"[k] for k in range(3)}

    alpha: dict[str, dict[int, float]] = {"a": {}, "b": {}, "c": {}}
    tab = _block_after(text, "Vibrational-rotational constants [1/cm]")
    tab = tab.split("Rotational constants B_e and B_0")[0]
    for ax, i, *_, total in re.findall(r"^\s*([xyz])\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
                                       r"(-?\d+\.\d+)\s*$", tab, flags=re.M):
        alpha[axis_of[ax]][int(i)] = float(total) * CM1_TO_MHZ

    fund = _block_after(text, "Fundamental transitions [1/cm]", 4000).split("Zero-point")[0]
    harmonic, fundamental = {}, {}
    for i, w, v, _ in re.findall(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", fund, flags=re.M):
        harmonic[int(i)], fundamental[int(i)] = float(w), float(v)

    qtext = _block_after(text, "quartic centrifugal distortion parameters for Watson", 3000)
    s_part = qtext.split("s-reduced")[1].split("a-reduced")[0] if "s-reduced" in qtext else ""
    a_part = qtext.split("a-reduced")[1][:900] if "a-reduced" in qtext else ""

    def grab(part: str, label: str) -> float:
        m = re.search(rf"^\s*{label}\s+(-?\d+\.\d+e[-+]\d+)", part, flags=re.M)
        if not m:
            raise ValueError(f"quartic constant {label} not found")
        return float(m.group(1)) * CM1_TO_MHZ

    quartic_s = {k: grab(s_part, lab) for k, lab in (("DJ", "DJ"), ("DJK", "DJK"), ("DK", "DK"),
                                                       ("d1", "D1"), ("d2", "D2"))}
    quartic_a = {k: grab(a_part, lab) for k, lab in (("DJ", "DelJ"), ("DJK", "DelJK"), ("DK", "DelK"),
                                                       ("dJ", "delJ"), ("dK", "delK"))}
    extra = {k: grab(qtext, k) for k in ("R5", "R6")}
    # the first three lines of the a-reduced block are the unreduced D'_J, D'_K, D'_JK
    unreduced = {"DJ": grab(a_part, "DJ"), "DJK": grab(a_part, "DJK"), "DK": grab(a_part, "DK"),
                 "dJ": quartic_a["dJ"], "R5": extra["R5"], "R6": extra["R6"]}
    sigma = re.search(r"^\s*sigma\s+(-?\d+\.\d+e[-+]\d+)", qtext, flags=re.M)
    fermi = [ln.strip() for ln in re.findall(r"^\s*possible Type I+ resonance.*$", text, flags=re.M)]
    zpe = re.search(r"Zero-point ro-vibrational energy \[1/cm\][\s\S]*?Total:\s+(-?\d+\.\d+)", text)
    return {"Be": {g: float(be_xyz[order[k]]) for k, g in enumerate("abc")},
            "B0": {g: float(b0_xyz[order[k]]) for k, g in enumerate("abc")},
            "alpha": alpha, "harmonic_cm1": harmonic, "fundamental_cm1": fundamental,
            "quartic_A": quartic_a, "quartic_S": quartic_s, "unreduced": unreduced,
            "R5": extra["R5"], "R6": extra["R6"],
            "sigma": float(sigma.group(1)) if sigma else None, "fermi": fermi,
            "zpe_anharmonic_cm1": float(zpe.group(1)) if zpe else None,
            "axis_map": axis_of}


# ==============================================================================
# 6. BUILT-IN WATSON ENGINE (A or S reduction, Ir) - levels, lines, strengths
# ==============================================================================
def _jk_ops(J: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K values, P_z and molecule-fixed P_+ (lowers K: anomalous commutation) in |J,K>."""
    ks = np.arange(-J, J + 1)
    pp = np.zeros((len(ks), len(ks)))
    for i in range(1, len(ks)):
        k = ks[i]
        pp[i - 1, i] = math.sqrt(J * (J + 1) - k * (k - 1))
    return ks, np.diag(ks.astype(float)), pp


def watson_matrix(J: int, c: dict[str, float], reduction: str) -> np.ndarray:
    """
    Watson Hamiltonian (MHz) in |J,K>, Ir representation (z=a, x=b, y=c):
      A: H = H_rigid - DJ N^4 - DJK N^2 Nz^2 - DK Nz^4 - 2 dJ N^2 (Nx^2-Ny^2)
             - dK [Nz^2 (Nx^2-Ny^2) + (Nx^2-Ny^2) Nz^2]
      S: H = H_rigid - DJ N^4 - DJK N^2 Nz^2 - DK Nz^4 + d1 N^2 (N+^2+N-^2) + d2 (N+^4+N-^4)
    Keys of c: A, B, C and DJ, DJK, DK plus dJ, dK (A) or d1, d2 (S); MHz.
    """
    ks, _, pp = _jk_ops(J)
    x = J * (J + 1)
    n = len(ks)
    h = np.zeros((n, n))
    a, b, cc = c["A"], c["B"], c["C"]
    for i, k in enumerate(ks):
        h[i, i] = (0.5 * (b + cc) * x + (a - 0.5 * (b + cc)) * k * k
                   - c.get("DJ", 0.0) * x * x - c.get("DJK", 0.0) * x * k * k - c.get("DK", 0.0) * k ** 4)
        if i + 2 < n:
            s2 = pp[i, i + 1] * pp[i + 1, i + 2]          # <K|P+^2|K+2> = sqrt[...][...]
            if reduction == "A":
                off = 0.25 * (b - cc) - c.get("dJ", 0.0) * x - 0.5 * c.get("dK", 0.0) * (k * k + (k + 2) ** 2)
            else:
                off = 0.25 * (b - cc) + c.get("d1", 0.0) * x
            h[i, i + 2] = h[i + 2, i] = off * s2
        if reduction == "S" and i + 4 < n:
            s4 = pp[i, i + 1] * pp[i + 1, i + 2] * pp[i + 2, i + 3] * pp[i + 3, i + 4]
            h[i, i + 4] = h[i + 4, i] = c.get("d2", 0.0) * s4
    return h


def direction_cosines(jl: int, ju: int) -> dict[str, np.ndarray]:
    """
    Molecule-fixed direction-cosine operators between J=jl (columns) and
    J=ju (rows), normalized so that |<u|Phi_g|l>|^2 is the M- and space-
    axis-summed line strength. Phi_z from the closed form; Phi_+- from the
    commutators [P+-, Phi_z], which fixes every phase consistently with the
    Hamiltonian (checked: symmetric-top limits, sum rules, Ir = IIIr).
    """
    kl, _, ppl = _jk_ops(jl)
    ku, _, ppu = _jk_ops(ju)
    z = np.zeros((len(ku), len(kl)))
    for i, k in enumerate(ku):
        if abs(k) <= jl:
            j = k + jl
            if ju == jl + 1:
                z[i, j] = math.sqrt(((jl + 1) ** 2 - k * k) / (jl + 1))
            elif ju == jl and jl > 0:
                z[i, j] = k * math.sqrt((2 * jl + 1) / (jl * (jl + 1)))
    phip = ppu @ z - z @ ppl
    phim = -(ppu.T @ z - z @ ppl.T)
    return {"a": z, "b": (phip + phim) / 2, "c": (phip - phim) / 2j}      # Ir: z=a, x=b, y=c


def asym_label(J: int, index: int) -> tuple[int, int]:
    """Ka, Kc of the index-th level (energy order) of J."""
    return (index + 1) // 2, J - index // 2


def predict_lines(c: dict[str, float], reduction: str, mu_d: dict[str, float], t_k: float = SPEC_T_K,
                  fmin: float = SPEC_FMIN_MHZ, fmax: float = SPEC_FMAX_MHZ, jmax: int = SPEC_JMAX) -> list[dict]:
    """
    Stick spectrum: every R- and Q-branch line between fmin and fmax with
    relative intensity  nu * sum_g mu_g^2 S_g * (exp(-E_l/kT) - exp(-E_u/kT)).
    No nuclear-spin weights (H2S exchange gives 1:3 doublets: Phase E).
    """
    levels = {}
    for J in range(jmax + 2):
        e, v = np.linalg.eigh(watson_matrix(J, c, reduction))
        levels[J] = (e, v)
    kt = K_B_MHZ_PER_K * t_k
    lines = []
    for jl in range(jmax + 1):
        for ju in (jl, jl + 1):
            ops = direction_cosines(jl, ju)
            el, vl = levels[jl]
            eu, vu = levels[ju]
            amps = {g: (vu.conj().T @ ops[g] @ vl) for g in "abc"}
            for i in range(len(el)):
                for k in range(len(eu)):
                    nu = eu[k] - el[i]
                    if not (fmin <= nu <= fmax):
                        continue
                    s = {g: abs(amps[g][k, i]) ** 2 for g in "abc"}
                    strength = sum(mu_d[g] ** 2 * s[g] for g in "abc")
                    if strength < 1e-12:
                        continue
                    boltz = math.exp(-(el[i] - levels[0][0][0]) / kt) - math.exp(-(eu[k] - levels[0][0][0]) / kt)
                    kal, kcl = asym_label(jl, i)
                    kau, kcu = asym_label(ju, k)
                    kind = max("abc", key=lambda g: mu_d[g] ** 2 * s[g])
                    lines.append({"freq_MHz": nu, "upper": f"{ju}_{kau}{kcu}" if max(kau, kcu) < 10 else
                                  f"{ju}_{kau},{kcu}", "lower": f"{jl}_{kal}{kcl}" if max(kal, kcl) < 10 else
                                  f"{jl}_{kal},{kcl}", "type": kind, "S_" + kind: s[kind],
                                  "E_lower_cm1": (el[i] - levels[0][0][0]) / CM1_TO_MHZ,
                                  "intensity": nu * strength * boltz})
    if not lines:
        return []
    top = max(l["intensity"] for l in lines)
    keep = [dict(l, rel_intensity=l["intensity"] / top) for l in lines if l["intensity"] / top >= SPEC_REL_CUTOFF]
    return sorted(keep, key=lambda l: l["freq_MHz"])


def watson_reductions(u: dict[str, float], sigma: float) -> tuple[dict[str, float], dict[str, float]]:
    """
    Watson A and S reduced quartic constants (Ir) from the reduction-independent
    D'_J, D'_JK, D'_K, delta_J, R5, R6 printed by ORCA's VPT2, for a given
    sigma = (2A - B - C)/(B - C). ORCA reduces with sigma of its own B3LYP B_e;
    the decks combine the B3LYP force field with revDSD-based B_0, so the
    reduction is redone with the hybrid's sigma (for this near-prolate complex
    B - C is small and sigma shifts by several %). With ORCA's sigma these
    formulas reproduce ORCA's printed A and S sets (tests).
    """
    r5, r6 = u["R5"], u["R6"]
    qa = {"DJ": u["DJ"] - 2 * r6, "DJK": u["DJK"] + 12 * r6, "DK": u["DK"] - 10 * r6, "dJ": u["dJ"],
          "dK": -2 * r5 - 4 * sigma * r6}
    qs = {"DJ": u["DJ"] + r5 / sigma, "DJK": u["DJK"] - 6 * r5 / sigma, "DK": u["DK"] + 5 * r5 / sigma,
          "d1": -u["dJ"], "d2": r6 + r5 / (2 * sigma)}
    return qa, qs


def watson_relation_check(qa: dict[str, float], qs: dict[str, float], sigma: float) -> float:
    """
    Largest relative violation of the first-order A <-> S relations (Ir):
    DJ(A) = DJ(S) - 2 d2, DJK(A) = DJK(S) + 12 d2, DK(A) = DK(S) - 10 d2,
    dJ = -d1, dK = -4 sigma d2. Verified numerically for this Hamiltonian and
    on ORCA 6.1.1's own A/S sets; a large value means a mis-parsed constant.
    """
    d2 = qs["d2"]
    pairs = [(qa["DJ"], qs["DJ"] - 2 * d2), (qa["DJK"], qs["DJK"] + 12 * d2), (qa["DK"], qs["DK"] - 10 * d2),
             (qa["dJ"], -qs["d1"]), (qa["dK"], -4 * sigma * d2)]
    return max(abs(x - y) / max(abs(x), abs(y), 1e-12) for x, y in pairs)


# ==============================================================================
# 7. PICKETT DECKS (.par/.var/.int for SPFIT/SPCAT)
# ==============================================================================
# Pickett parameter IDs and the factor that turns a Watson constant into the
# Pickett coefficient. Sign trap (method matrix F01 Sec. 18): the quartic
# terms enter the Watson Hamiltonian with a minus sign, so Pickett gets -DJ,
# -DJK, -DK, -dJ, -dK; the S-reduction d1 and d2 enter with a plus sign.
PICKETT_IDS: dict[str, list[tuple[str, int, float]]] = {
    "A": [("A", 10000, 1.0), ("B", 20000, 1.0), ("C", 30000, 1.0), ("DJ", 200, -1.0), ("DJK", 1100, -1.0),
          ("DK", 2000, -1.0), ("dJ", 40100, -1.0), ("dK", 41000, -1.0)],
    "S": [("A", 10000, 1.0), ("B", 20000, 1.0), ("C", 30000, 1.0), ("DJ", 200, -1.0), ("DJK", 1100, -1.0),
          ("DK", 2000, -1.0), ("d1", 40100, 1.0), ("d2", 50000, 1.0)],
}


PICKETT_FLOATED: set[str] = {"A", "B", "C", "DJ", "DJK"}    # free in the SPFIT starting .par


def pickett_par_text(title: str, c: dict[str, float], err: dict[str, float] | None, reduction: str,
                     source: dict[str, str]) -> str:
    """
    .par/.var text. Line 2: NPAR NLINE NITR NXPAR THRESH ERRTST FRAC CAL.
    Option line: CHR SPIND NVIB KNMIN KNMAX IXX IAX WTPL WTMN VSYM EWT DIAG, with
    CHR = a/s (Watson reduction), IAX = 1 (Ir: quantization along a; every
    isomer here is a near-prolate top), WTPL = WTMN = 1 (no spin weights; the
    H2S 1:3 weights belong to the tunnelling model of Phase E).
    ERPAR (third column) is TRAP 3 of the method matrix: in the .par it is the
    a-priori constraint for SPFIT (1E+37 = free, 1E-37 = held at the value);
    in the .var it is the 1-sigma uncertainty SPCAT turns into line errors.
    Values in MHz; quartic terms carry the Pickett sign (see PICKETT_IDS).
    """
    lines = [title[:79],
             f"{len(PICKETT_IDS[reduction]):4d} {1000:5d} {10:4d} {0:4d}   0.0000E+000   1.0000E+006   1.0000E+000 1.0000000000",
             f"{reduction.lower()}   1   1   0   99   0   1   1   1   0   0   0"]
    for name, pid, factor in PICKETT_IDS[reduction]:
        value = factor * c[name]
        if err is None:
            erpar = 1e37 if name in PICKETT_FLOATED else 1e-37
        else:
            erpar = err[name]
        line = f"{pid:13d} {value: .15E} {erpar:.8E} /{'-' if factor < 0 else ''}{name} {source.get(name, '')}"
        lines.append(line[:80].rstrip())          # the source of every value (matrix F01 Sec. 18), <= 80 chars
    return "\n".join(lines) + "\n"


def pickett_int_text(title: str, mu_d: dict[str, float], partition_q: float, t_k: float, tag: int) -> str:
    """
    .int text. Line 2: FLAGS TAG QROT FBGN FEND STR0 STR1 FQLIM TEMP - the
    temperature is the ninth field (method matrix F01 Sec. 18); FQLIM in GHz.
    Dipoles signed, in Debye, in the documented form '1 x.xxx /mua'.
    """
    return (f"{title[:79]}\n"
            f" 0000 {tag:6d} {partition_q:12.4f}   0  20  -10.0000  -10.0000  40.0000 {t_k:7.2f}\n"
            f" 1 {mu_d['a']: .4f} /mua\n 2 {mu_d['b']: .4f} /mub\n 3 {mu_d['c']: .4f} /muc\n")


def read_pickett_constants(par_text: str, reduction: str) -> dict[str, float]:
    """Inverse of pickett_par_text (Pickett sign undone): used by the round-trip test."""
    by_id = {pid: (name, factor) for name, pid, factor in PICKETT_IDS[reduction]}
    out = {}
    for line in par_text.splitlines()[3:]:
        parts = line.split("/")[0].split()
        if len(parts) >= 2 and int(parts[0]) in by_id:
            name, factor = by_id[int(parts[0])]
            out[name] = float(parts[1].replace("D", "E")) / factor
    return out


def write_pickett(out_dir: Path, stem: str, title: str, c: dict[str, float], err: dict[str, float],
                  reduction: str, mu_d: dict[str, float], partition_q: float, tag: int,
                  source: dict[str, str]) -> list[Path]:
    """Writes <stem>.par (SPFIT start), <stem>.var (SPCAT input) and <stem>.int."""
    files = [out_dir / f"{stem}.par", out_dir / f"{stem}.var", out_dir / f"{stem}.int"]
    files[0].write_text(pickett_par_text(title, c, None, reduction, source), encoding="ascii", newline="\n")
    files[1].write_text(pickett_par_text(title, c, err, reduction, source), encoding="ascii", newline="\n")
    files[2].write_text(pickett_int_text(title, mu_d, partition_q, SPEC_T_K, tag), encoding="ascii", newline="\n")
    return files


def find_spcat() -> Path | None:
    """A native SPCAT executable: PATH, $COCHEM_BIN_DIR or spcat_path in the CoChem registry config."""
    name = "spcat.exe" if os.name == "nt" else "spcat"     # a Linux build cannot run on Windows
    hit = shutil.which(name)
    if hit:
        return Path(hit)
    bindir = os.environ.get("COCHEM_BIN_DIR")
    if bindir and (Path(bindir) / name).is_file():
        return Path(bindir) / name
    cfg = Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
    try:
        path = Path(json.loads(cfg.read_text(encoding="utf-8")).get("spcat_path") or "")
        if path.is_file() and (os.name != "nt" or path.suffix.lower() == ".exe"):
            return path
    except (OSError, ValueError):
        pass
    return None


def run_spcat(spcat: Path, var_file: Path, int_file: Path) -> Path | None:
    """Runs SPCAT on a copy of the deck (SPCAT names every file after its argument)."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in (var_file, int_file):
            shutil.copyfile(f, Path(tmp) / f"deck{f.suffix}")
        subprocess.run([str(spcat), "deck"], cwd=tmp, capture_output=True, timeout=300)
        cat = Path(tmp) / "deck.cat"
        if not cat.exists():
            return None
        target = var_file.with_suffix(".cat")
        shutil.copyfile(cat, target)
        return target


def classical_partition_function(a: float, b: float, c: float, t_k: float) -> float:
    """Asymmetric-top rotational partition function (classical; spin weights not included)."""
    kt = K_B_MHZ_PER_K * t_k
    return math.sqrt(math.pi * kt ** 3 / (a * b * c))


# ==============================================================================
# 8. ONE ISOMER: STAGES S0-S3 (ORCA) AND S4 (ANALYSIS)
# ==============================================================================
@dataclass
class Context:
    level: Level
    engine: Engine
    dry_run: bool
    analyze_only: bool
    stages: list[str] = field(default_factory=lambda: list(STAGE_KEYS))     # --stages (ORCA jobs)
    analyze: bool = True                                                     # S4 listed in --stages


def stage_dir(iso: Isomer, ctx: Context, stage: str) -> Path:
    root = WORK_DIR / iso.label
    if ctx.level.folder_suffix:
        root = root / ctx.level.folder_suffix
    return root / STAGE_DIR[stage]


def last_geometry(job_dir: Path, base: str) -> tuple[list[str], np.ndarray] | None:
    xyz = job_dir / f"{base}.xyz"
    frames = read_xyz_frames(xyz) if xyz.exists() else []
    return (frames[-1][1], frames[-1][2]) if frames else None


def finished_geometry(job_dir: Path, base: str) -> tuple[list[str], np.ndarray] | None:
    """
    Final geometry of a stage that ended normally, else None. A dry run chains
    through finished stages exactly like a real run, so its inputs are the real
    ones; with the input geometry instead, a dry run in a folder with finished
    stages rewrote their successors' inputs and forced a rerun (smoke test).
    """
    if "ORCA TERMINATED NORMALLY" not in read_text(job_dir / f"{base}.out"):
        return None
    return last_geometry(job_dir, base)


def stage_input_matches(iso: Isomer, ctx: Context, stage: str, start: tuple[list[str], np.ndarray]) -> bool:
    """
    True if the stage's existing .inp is exactly the input a run would write
    now. Rendered in a temporary folder, so an unrequested stage is never
    touched (a 'Restart true' line alone does not count as a difference).
    """
    job_dir, base = stage_dir(iso, ctx, stage), STAGE_DIR[stage]
    existing = job_dir / f"{base}.inp"
    if not existing.exists():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        write_xyz(Path(tmp) / "start.xyz", start[0], start[1], f"{iso.label} {stage} start")
        calc = build_stage(stage, Path(tmp), ctx.level)
        write_input(calc, Path(tmp) / "start.xyz")
        new = (Path(tmp) / f"{base}.inp").read_text(encoding="utf-8")
    return _IGNORE_FOR_RESUME.sub("", existing.read_text(encoding="utf-8")) == _IGNORE_FOR_RESUME.sub("", new)


def run_stage(iso: Isomer, ctx: Context, stage: str, start: tuple[list[str], np.ndarray],
              hess_from: Path | None = None) -> tuple[str, str]:
    """
    Writes the stage input, runs or reuses it; returns (status, output text).
    A stage left out of --stages is never written or run: it counts as
    'reused' only if it finished with exactly the input a run would write,
    otherwise it is 'not requested' and the chain stops before it.
    """
    job_dir = stage_dir(iso, ctx, stage)
    base = STAGE_DIR[stage]
    if stage not in ctx.stages:
        out = job_dir / f"{base}.out"
        done = "ORCA TERMINATED NORMALLY" in read_text(out) and stage_input_matches(iso, ctx, stage, start)
        status = "reused" if done else "not requested"
        iso.stage_status[stage] = status
        if not ctx.dry_run:
            log(f"  [{iso.label}] {stage:<4s} {status:>8s}  (not in --stages)")
        return status, read_text(out) if done else ""
    job_dir.mkdir(parents=True, exist_ok=True)
    start_xyz = job_dir / "start.xyz"
    write_xyz(start_xyz, start[0], start[1], f"{iso.label} {stage} start")
    if hess_from is not None and hess_from.exists():
        shutil.copyfile(hess_from, job_dir / "s0.hess")
    restart = (stage == "S2b" and not Output(basename=base, working_dir=job_dir).terminated_normally()
               and bool(numfreq_restart_files(job_dir, base)))
    calc = build_stage(stage, job_dir, ctx.level, restart_numfreq=restart)
    unchanged, sha = write_input(calc, start_xyz)
    iso.input_sha1[stage] = sha
    if ctx.dry_run:
        return "dry-run", ""
    t0 = time.time()
    status = run_orca(calc, ctx.engine, STAGE_TIMEOUT_H[stage] * 3600, unchanged)
    iso.timings_s[stage] = round(time.time() - t0, 1)
    iso.stage_status[stage] = status
    msg = f"  [{iso.label}] {stage:<4s} {status:>8s}  ({time.time() - t0:8.0f} s)"
    log(msg + ("   restarted from finished NumFreq displacements" if restart and status == "ok" else ""))
    return status, read_text(job_dir / f"{base}.out")


def process_isomer(iso: Isomer, ctx: Context) -> dict | None:
    """
    Runs S0-S3b in order and then S4. The chain stops at the first failure,
    or at an unfinished stage that --stages leaves out; iso.requested_done
    tells whether every requested stage finished (ok or reused).
    """
    log(f"\n=== {iso.label} ({iso.path.name}) ===")
    if iso.note:
        log(f"  [{iso.label}] {iso.note}")
    geo = (iso.symbols, iso.coords)
    ok = ("ok", "reused")
    last_requested = len(STAGE_KEYS) if ctx.analyze else max(STAGE_KEYS.index(s) for s in ctx.stages)

    def opt_status(status: str, text: str) -> str:
        return "not converged" if status in ok and not opt_converged(text) else status

    def stop(stage: str, status: str) -> bool:
        """True if the chain ends at this stage (and why is recorded)."""
        if ctx.dry_run or status in ok:
            return False
        if status == "not requested":
            if STAGE_KEYS.index(stage) < last_requested:
                iso.flags.append(f"{stage} is not finished and not in --stages: later stages cannot run")
            else:
                iso.requested_done = True        # everything requested lies before this stage and is done
            return True
        iso.flags.append(f"{stage} {status}: stopped")
        return True

    # S0: r2SCAN-3c pre-optimization + Hessian
    status, text = run_stage(iso, ctx, "S0", geo)
    s0 = stage_dir(iso, ctx, "S0")
    if stop("S0", opt_status(status, text)):
        return None
    if not ctx.dry_run:
        freqs, _ = frequencies_and_zpe(text)
        imag = [f for f in freqs if f < 0]
        if imag:
            iso.flags.append(f"S0 r2SCAN-3c: {len(imag)} imaginary mode(s), lowest {min(imag):.0f} cm-1 "
                             "(S2b decides at revDSD)")
        geo = last_geometry(s0, STAGE_DIR["S0"]) or geo
    else:
        geo = finished_geometry(s0, STAGE_DIR["S0"]) or geo
    hess0 = s0 / f"{STAGE_DIR['S0']}.hess"

    # S1: revDSD very tight optimization
    status, text = run_stage(iso, ctx, "S1", geo, hess_from=hess0)
    s1 = stage_dir(iso, ctx, "S1")
    if stop("S1", opt_status(status, text)):
        return None
    if not ctx.dry_run:
        geo = last_geometry(s1, STAGE_DIR["S1"]) or geo
        if bonded_pairs(*geo) != MOL.bonds:
            iso.flags.append("S1: covalent bonds changed during the revDSD optimization: stopped")
            return None
    else:
        geo = finished_geometry(s1, STAGE_DIR["S1"]) or geo
    geo_dh = geo

    # S2a: relaxed-density dipole; S2b: harmonic frequencies (both run before either is judged)
    status_a, _ = run_stage(iso, ctx, "S2a", geo_dh)
    status_b, text_b = run_stage(iso, ctx, "S2b", geo_dh)
    if stop("S2a", status_a) | stop("S2b", status_b):
        return None
    if not ctx.dry_run:
        freqs, _ = frequencies_and_zpe(text_b)
        imag = [f for f in freqs if f < 0]
        if imag and min(imag) < -IMAG_NOISE_CM1:
            write_imaginary_followups(iso, ctx, geo_dh)
            iso.flags.append(f"REJECTED: revDSD saddle point ({len(imag)} imaginary, lowest {min(imag):.0f} cm-1); "
                             "displaced geometries written to results/ for a new isomer file")
            return None
        if imag:
            iso.flags.append(f"S2b: tiny imaginary mode {min(imag):.0f} cm-1 (numerical noise; regrid if it matters)")

    # S3a: B3LYP very tight optimization; S3b: VPT2 at that minimum
    status, text = run_stage(iso, ctx, "S3a", geo_dh, hess_from=hess0)
    s3a = stage_dir(iso, ctx, "S3a")
    if stop("S3a", opt_status(status, text)):
        return None
    if not ctx.dry_run:
        geo_b3 = last_geometry(s3a, STAGE_DIR["S3a"]) or geo_dh
    else:
        geo_b3 = finished_geometry(s3a, STAGE_DIR["S3a"]) or geo_dh
    status, _ = run_stage(iso, ctx, "S3b", geo_b3)
    if ctx.dry_run:
        iso.requested_done = True
        return None
    if stop("S3b", status):
        return None
    if not ctx.analyze:                         # S4 not in --stages
        iso.requested_done = True
        return None
    res = analyze_isomer(iso, ctx)
    iso.requested_done = True
    return res


def write_imaginary_followups(iso: Isomer, ctx: Context, geo: tuple[list[str], np.ndarray]) -> None:
    """Writes the revDSD saddle point displaced +-0.25 A along its imaginary mode."""
    hess = stage_dir(iso, ctx, "S2b") / f"{STAGE_DIR['S2b']}.hess"
    try:
        lines = [ln.strip() for ln in hess.read_text(encoding="utf-8", errors="ignore").splitlines()]
        k = lines.index("$vibrational_frequencies")
        n = int(lines[k + 1])
        freqs = [float(lines[k + 2 + i].split()[1]) for i in range(n)]
        mode_idx = int(np.argmin(freqs))
        k = lines.index("$normal_modes")
        rows, cols = (int(v) for v in lines[k + 1].split())
        modes = np.zeros((rows, cols))
        i = k + 2
        while i < len(lines) and not lines[i].startswith(("$", "#")):
            if not lines[i]:
                i += 1
                continue
            idx = [int(c) for c in lines[i].split()]
            for r in range(rows):
                parts = lines[i + 1 + r].split()
                modes[int(parts[0]), idx] = [float(v) for v in parts[1:]]
            i += 1 + rows
        mode = modes[:, mode_idx].reshape(-1, 3)
    except (OSError, ValueError, IndexError):
        log(f"  [{iso.label}] could not read the imaginary mode from {hess}")
        return
    out = results_dir(iso, ctx)
    step = 0.25 / np.linalg.norm(mode, axis=1).max()
    for sign, side in ((1, "plus"), (-1, "minus")):
        write_xyz(out / f"{iso.label}_imag_follow_{side}.xyz", geo[0], geo[1] + sign * step * mode,
                  f"{iso.label} revDSD saddle point displaced {side} along the {freqs[mode_idx]:.0f} cm-1 mode")


def results_dir(iso: Isomer, ctx: Context) -> Path:
    d = WORK_DIR / iso.label / (ctx.level.folder_suffix or "") / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyze_isomer(iso: Isomer, ctx: Context) -> dict:
    """S4: hybrid B_0, signed principal-axis dipoles, guards, decks, spectrum, isotopologues, provenance."""
    out_dir = results_dir(iso, ctx)
    texts = {s: read_text(stage_dir(iso, ctx, s) / f"{STAGE_DIR[s]}.out") for s, _ in STAGES}
    res: dict = {"isomer": iso.label, "level": ctx.level.name}

    # revDSD equilibrium structure -> B_e (Mendeleev masses = !Mass2016)
    symbols, xyz = last_geometry(stage_dir(iso, ctx, "S1"), STAGE_DIR["S1"])
    masses = masses_for(symbols)
    be = rotational_constants(xyz, masses)
    res["Be_revDSD_MHz"] = dict(zip("abc", be.tolist()))
    res["inertia"] = inertial_analysis(xyz, masses)
    res["geometry"] = describe_geometry(xyz, symbols, MOL)
    res["E_revDSD_Eh"] = final_energy(texts["S1"])
    res["S1_final_max_gradient_Eh_bohr"] = final_max_gradient(texts["S1"])
    write_xyz(out_dir / f"{iso.label}_revDSD.xyz", symbols, xyz,
              f"{iso.label} revDSD-PBEP86-D4/2021/{ctx.level.dh_basis[0].keyword} equilibrium E={res['E_revDSD_Eh']}")

    # Dipole of the relaxed revDSD density, in the principal axes of the same geometry
    dip = relaxed_dipole(texts["S2a"])
    if dip is None:
        raise RuntimeError("S2a output has no 'Method : MP2 / Level : Relaxed density' dipole block")
    s2a_start = read_xyz_frames(stage_dir(iso, ctx, "S2a") / "start.xyz")[0][2]
    if np.abs(s2a_start - xyz).max() > 1e-6:
        raise RuntimeError("S2a geometry is not the S1 geometry: rerun S2a")
    _, _, axes = principal_axes(xyz, masses)
    mu_abc = axes.T @ dip["vector_au"] * AU_TO_DEBYE
    mu = {g: float(mu_abc[k]) for k, g in enumerate("abc")}
    res["mu_D"], res["mu_total_D"] = mu, float(np.linalg.norm(mu_abc))
    if dip["rot_axes_D"] is not None:
        dev = float(np.abs(np.abs(mu_abc) - np.abs(dip["rot_axes_D"])).max())
        res["mu_check_vs_orca_D"] = dev
        if dev > 0.01:
            iso.flags.append(f"dipole projection differs from ORCA's rotational-axis components by {dev:.3f} D")
    dark = [g for g in "abc" if abs(mu[g]) < DARK_DIPOLE_D]
    if dark:
        iso.flags.append(f"dark: |mu_{','.join(dark)}| < {DARK_DIPOLE_D} D -> those line types left out of the line list")

    # revDSD harmonic frequencies
    freqs, zpe = frequencies_and_zpe(texts["S2b"])
    res["revDSD_harmonic_cm1"], res["ZPE_revDSD_harm_Eh"] = freqs, zpe
    soft = softest_mode(stage_dir(iso, ctx, "S2b") / f"{STAGE_DIR['S2b']}.hess", masses)
    if soft:
        res["softest_mode_cm1"], res["softest_force_constant_mdyn_A"] = soft

    # B3LYP VPT2: vibrational correction and distortion constants
    sym3, xyz3 = last_geometry(stage_dir(iso, ctx, "S3a"), STAGE_DIR["S3a"])
    be_b3 = rotational_constants(xyz3, masses_for(sym3))
    res["E_B3LYP_Eh"] = final_energy(texts["S3a"])
    v = parse_vpt2(texts["S3b"])
    for k, g in enumerate("abc"):
        if abs(be_b3[k] - v["Be"][g]) > 0.5 + 1e-5 * be_b3[k]:       # ORCA prints B_e to 1e-5 cm-1 = 0.3 MHz
            iso.flags.append(f"B3LYP B_e({g}) {be_b3[k]:.1f} (ours) vs {v['Be'][g]:.1f} MHz (ORCA VPT2): masses differ?")
    soft_modes = sorted(i for i, w in v["harmonic_cm1"].items() if w < SOFT_MODE_CM1)
    dbvib_all = {g: v["B0"][g] - v["Be"][g] for g in "abc"}
    dbvib_soft = {g: -0.5 * sum(v["alpha"][g].get(i, 0.0) for i in soft_modes) for g in "abc"}
    dbvib = {g: dbvib_all[g] - dbvib_soft[g] for g in "abc"}
    b0 = {g: float(be[k] + dbvib[g]) for k, g in enumerate("abc")}
    window = {g: max(B0_WINDOW_REL * b0[g], abs(dbvib_soft[g])) for g in "abc"}
    res.update({"Be_B3LYP_MHz": dict(zip("abc", be_b3.tolist())), "dBvib_all_MHz": dbvib_all,
                "dBvib_soft_modes_MHz": dbvib_soft, "dBvib_used_MHz": dbvib, "B0_MHz": b0,
                "B0_all_modes_MHz": {g: float(be[k] + dbvib_all[g]) for k, g in enumerate("abc")},
                "B0_window_MHz": window, "soft_modes": {i: v["harmonic_cm1"][i] for i in soft_modes},
                "fermi": v["fermi"], "ZPE_B3LYP_anharm_cm1": v["zpe_anharmonic_cm1"]})
    if soft_modes:
        iso.flags.append(f"{len(soft_modes)} mode(s) below {SOFT_MODE_CM1:.0f} cm-1 excluded from dB_vib "
                         f"(matrix: no hybrid VPT2 for soft modes); their alpha sum is inside the window")
    if v["fermi"]:
        iso.flags.append(f"{len(v['fermi'])} possible Fermi resonance(s) in B3LYP VPT2 (not treated)")

    # Watson A and S constants: the B3LYP force field reduced with the hybrid's own sigma;
    # the S-reduced A, B, C follow from the first-order relations
    sigma = (2 * b0["a"] - b0["b"] - b0["c"]) / (b0["b"] - b0["c"])
    qa, qs = watson_reductions(v["unreduced"], sigma)
    ca = {"A": b0["a"], "B": b0["b"], "C": b0["c"], **qa}
    d2 = qs["d2"]
    cs = {"A": ca["A"] - 10 * d2, "B": ca["B"] + (8 * sigma + 4) * d2, "C": ca["C"] - (8 * sigma - 4) * d2, **qs}
    res.update({"sigma_hybrid": sigma, "sigma_orca_B3LYP": v["sigma"], "quartic_A_MHz": qa, "quartic_S_MHz": qs,
                "quartic_A_orca_MHz": v["quartic_A"], "quartic_S_orca_MHz": v["quartic_S"],
                "S_minus_A_rot_const_MHz": {"A": cs["A"] - ca["A"], "B": cs["B"] - ca["B"], "C": cs["C"] - ca["C"]}})
    if v["sigma"]:                      # the reduction formulas must reproduce ORCA's own sets with ORCA's sigma
        oa, os_ = watson_reductions(v["unreduced"], v["sigma"])
        dev = max(max(abs(oa[k] - v["quartic_A"][k]) / max(abs(v["quartic_A"][k]), 1e-12) for k in oa),
                  max(abs(os_[k] - v["quartic_S"][k]) / max(abs(v["quartic_S"][k]), 1e-12) for k in os_))
        res["orca_reduction_reproduced_max_rel_dev"] = dev
        if dev > 1e-4:
            iso.flags.append(f"ORCA's A/S quartic sets not reproduced from R5/R6 ({dev:.1e}): check the parse")
    res["kappa"] = (2 * ca["B"] - ca["A"] - ca["C"]) / (ca["A"] - ca["C"])
    if res["kappa"] > 0:
        iso.flags.append("near-oblate rotor: decks use Ir anyway; consider IIIr (IAX=3) for fitting")

    # Pickett decks
    src_b = "[D] Be(rDSD)+vib(B3LYP)"              # fits the 80-character parameter line
    sources = {"A": src_b, "B": src_b, "C": src_b} | {k: "[D] B3LYP-D4 VPT2" for k in ("DJ", "DJK", "DK", "dJ", "dK",
                                                                                     "d1", "d2")}
    qrot = classical_partition_function(ca["A"], ca["B"], ca["C"], SPEC_T_K)
    tag = round(masses.sum()) * 1000 + iso.number + 1            # JPL style: mass x 1000 + id
    for red, cc in (("A", ca), ("S", cs)):
        err = {k: (window[{"A": "a", "B": "b", "C": "c"}[k]] if k in "ABC" else QUARTIC_WINDOW_REL * abs(cc[k]))
               for k in cc}
        title = f"{MOL.name} {iso.label} Watson-{red} Ir revDSD/B3LYP {ctx.level.name} {datetime.now():%Y-%m-%d}"
        write_pickett(out_dir, f"{iso.label}_{red}", title, cc, err, red, mu, qrot, tag, sources)
    res["pickett"] = {"tag": tag, "QROT_classical_2K": qrot, "representation": "Ir (IAX=1)"}

    # Stick spectrum (built-in engine, dark types left out) and the A/S cross-check
    mu_lines = {g: (0.0 if g in dark else mu[g]) for g in "abc"}
    lines_a = predict_lines(ca, "A", mu_lines)
    lines_s = {(l["upper"], l["lower"]): l["freq_MHz"] for l in predict_lines(cs, "S", mu_lines)}
    common = [abs(l["freq_MHz"] - lines_s[(l["upper"], l["lower"])]) for l in lines_a
              if (l["upper"], l["lower"]) in lines_s]
    res["A_vs_S_max_line_diff_MHz"] = max(common) if common else None
    line_rel = max(window[g] / b0[g] for g in "abc")          # widest relative window of A, B, C [E]
    res["line_window_rel"] = line_rel
    with open(out_dir / f"{iso.label}_lines_{SPEC_T_K:g}K.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["freq_MHz", "window_MHz", "upper", "lower", "type", "rel_intensity", "E_lower_cm-1"])
        for l in sorted(lines_a, key=lambda x: -x["rel_intensity"]):
            w.writerow([f"{l['freq_MHz']:.3f}", f"{line_rel * l['freq_MHz']:.1f}", l["upper"], l["lower"],
                        l["type"], f"{l['rel_intensity']:.4f}", f"{l['E_lower_cm1']:.4f}"])
    res["n_lines"] = len(lines_a)
    spcat = find_spcat()
    res["spcat"] = str(spcat) if spcat else "not found (decks written; built-in engine line list provided)"
    if spcat:
        for red in ("A", "S"):
            run_spcat(spcat, out_dir / f"{iso.label}_{red}.var", out_dir / f"{iso.label}_{red}.int")

    # Isotopologues: B_e of the same revDSD structure scaled by the parent's B_0/B_e [E]
    rows = []
    for name, substitution in isotopologues(MOL, xyz):
        m_i = masses_for(symbols, substitution)
        be_i = rotational_constants(xyz, m_i)
        _, _, axes_i = principal_axes(xyz, m_i)
        mu_i = axes_i.T @ dip["vector_au"] * AU_TO_DEBYE
        b0_i = [be_i[k] * b0[g] / be[k] for k, g in enumerate("abc")]
        rows.append([name] + [f"{x:.2f}" for x in b0_i] + [f"{x:.2f}" for x in be_i]
                    + [f"{x:+.3f}" for x in mu_i] + [f"{inertial_analysis(xyz, m_i)['inertial_defect']:.4f}"])
    with open(out_dir / f"{iso.label}_isotopologues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isotopologue", "A0_MHz", "B0_MHz", "C0_MHz", "Ae_MHz", "Be_MHz", "Ce_MHz",
                    "mu_a_D", "mu_b_D", "mu_c_D", "inertial_defect_e_uA2"])
        w.writerows(rows)

    # Constants table, JSON, README
    with open(out_dir / f"{iso.label}_constants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "window_or_note", "source"])
        for k, g in enumerate("abc"):
            w.writerow([f"{g.upper()}_0 (MHz)", f"{b0[g]:.3f}", f"+-{window[g]:.1f}", src_b])
            w.writerow([f"{g.upper()}_e revDSD (MHz)", f"{be[k]:.3f}", "", "[D] S1"])
            w.writerow([f"dB_vib {g} used (MHz)", f"{dbvib[g]:.3f}", f"all modes {dbvib_all[g]:.3f}", "[D] S3b"])
        for red, cc, keys in (("A", ca, ("DJ", "DJK", "DK", "dJ", "dK")), ("S", cs, ("DJ", "DJK", "DK", "d1", "d2"))):
            for key in keys:
                w.writerow([f"{key} ({red}, kHz)", f"{1000 * cc[key]:.4f}", f"+-{100 * QUARTIC_WINDOW_REL:.0f} %",
                            "[D] B3LYP-D4 VPT2 (hybrid sigma)"])
        for g in "abc":
            w.writerow([f"mu_{g} (D, signed)", f"{mu[g]:+.3f}", "dark" if g in dark else "", "[D] S2a relaxed density"])
        for key in ("inertial_defect", "Pcc", "kappa"):
            w.writerow([key, f"{res['inertia'][key]:.5f}", "u A^2" if key != "kappa" else "", "[D] S1"])
    (out_dir / f"{iso.label}_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    write_results_readme(iso, ctx, res, out_dir)
    return res


def write_results_readme(iso: Isomer, ctx: Context, res: dict, out_dir: Path) -> None:
    b0, w, mu = res["B0_MHz"], res["B0_window_MHz"], res["mu_D"]
    floppy = [g for g in MOL.equivalent_groups if MOL.symbols[g[0]] == "H"]
    lam = ("  - large-amplitude motion of the equivalent H atoms ("
           + ", ".join("/".join(f"H{i}" for i in g) for g in floppy)
           + "): if their exchange barrier is low, expect\n"
           "    tunnelling doublets with nuclear-spin weights and shifted or missing lines of the dipole\n"
           "    component that changes sign. These decks are a single rigid-rotor state.\n") if floppy else ""
    text = f"""{MOL.name} {iso.label}: microwave predictions ({ctx.level.name} level)
{'=' * 72}
B_0 = B_e(revDSD-PBEP86-D4/2021) + dB_vib(B3LYP-D4 VPT2, modes >= {SOFT_MODE_CM1:.0f} cm-1)
  A_0 = {b0['a']:.1f} +- {w['a']:.1f} MHz    B_0 = {b0['b']:.1f} +- {w['b']:.1f} MHz    C_0 = {b0['c']:.1f} +- {w['c']:.1f} MHz
  mu_a = {mu['a']:+.2f} D   mu_b = {mu['b']:+.2f} D   mu_c = {mu['c']:+.2f} D   (relaxed revDSD density)
  kappa = {res['kappa']:.4f}   inertial defect (equilibrium) = {res['inertia']['inertial_defect']:.4f} u A^2

Files
  {iso.label}_A.par / .var / .int   Pickett deck, Watson A reduction, Ir representation
  {iso.label}_S.par / .var / .int   same, Watson S reduction
      .par = SPFIT starting deck (A, B, C, DJ, DJK free; the rest held at the prediction)
      .var = SPCAT input; third column = 1-sigma uncertainty -> SPCAT line errors
      .int = dipoles (signed, Debye), T = {SPEC_T_K:g} K as the ninth field of line 2
  {iso.label}_lines_{SPEC_T_K:g}K.csv     predicted lines {SPEC_FMIN_MHZ / 1000:g}-{SPEC_FMAX_MHZ / 1000:g} GHz, strongest first, with
                                  a search window (+-{100 * res['line_window_rel']:.1f} % of the frequency)
  {iso.label}_constants.csv       every constant with window and source tag
  {iso.label}_isotopologues.csv   every single 13C/15N/18O/34S/D substitution (atom labels as in
                                  the isomer file) and each fully deuterated equivalent-H group: B_e
                                  of the same structure scaled by the parent's B_0/B_e [E]
  {iso.label}_revDSD.xyz          the revDSD equilibrium structure
  {iso.label}_results.json        everything above, machine-readable
  provenance.json                 versions, keywords, thresholds, timings

Conventions
  Signed dipoles: principal axes of the revDSD structure with the a and b axis
  vectors chosen so that their largest Cartesian component (input frame) is
  positive, and c = a x b. Only signs relative to the atoms are physical.
  Pickett signs: the quartic constants enter as -DJ, -DJK, -DK, -dJ, -dK (A) and
  -DJ, -DJK, -DK, +d1, +d2 (S). Tags: [D] derived from these calculations, [E] estimate.
  The A deck uses B_0 as computed; the S deck's A, B, C follow from the first-order
  Watson relations: S - A = {res['S_minus_A_rot_const_MHz']['A']:+.4f} / {res['S_minus_A_rot_const_MHz']['B']:+.4f} / {res['S_minus_A_rot_const_MHz']['C']:+.4f} MHz (A / B / C), far inside the
  window; both decks predict the same lines (largest difference {res['A_vs_S_max_line_diff_MHz'] or 0:.4f} MHz below {SPEC_FMAX_MHZ / 1000:g} GHz).

Flags
""" + "".join(f"  - {f}\n" for f in iso.flags) + f"""
Not in this tier (follow-ups)
""" + lam + f"""  - counterpoise-corrected geometry; sextic distortion constants; VPT2 alphas of the
    soft intermolecular modes (excluded and folded into the window).
  - SPCAT: {res['spcat']}
"""
    (out_dir / "README.txt").write_text(text, encoding="utf-8")


# ==============================================================================
# 9. MAIN
# ==============================================================================
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="revDSD/B3LYP microwave predictions for the isomers of one molecule or "
                                            "complex (ORCA 6.1.1, OPI 2.0)")
    p.add_argument("--work-dir", type=Path, help="folder with the isomer-NN.xyz files; every output goes there "
                                                 "(default: the folder of this script)")
    p.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGE_KEYS)},S4 to run, e.g. S0,S1,S2a,S2b "
                                    "(default all; S4 = the analysis); other stages are only reused, never run or "
                                    "rewritten")
    p.add_argument("--dry-run", action="store_true", help="write every (requested) ORCA input, run nothing")
    p.add_argument("--smoke", action="store_true", help="whole chain with small basis sets on 4 cores (test)")
    p.add_argument("--calibrate", action="store_true", help="one timed revDSD gradient on the first isomer")
    p.add_argument("--analyze-only", action="store_true", help="redo S4 from finished jobs; never start ORCA")
    p.add_argument("--only", help="comma-separated isomer numbers, e.g. 00,2")
    p.add_argument("--cores", type=int, help=f"MPI ranks per job (default {NPROCS}; smoke {SMOKE_NPROCS})")
    p.add_argument("--vpt2-groups", type=int, metavar="N",
                   help="S3b: N ranks per parallel displaced-Hessian group (default off; hung in ORCA 6.1.1 on Windows)")
    return p.parse_args(argv)


def write_provenance(iso: Isomer, ctx: Context) -> None:
    """Versions, keywords as written, every threshold as a number, timings (method matrix F01 Sec. 20.2)."""
    stages = {}
    for stage, base in STAGES:
        inp = stage_dir(iso, ctx, stage) / f"{base}.inp"
        if inp.exists():
            text = inp.read_text(encoding="utf-8")
            stages[stage] = {"keywords": [ln[1:].strip() for ln in text.splitlines() if ln.startswith("!")],
                             "input_sha1": iso.input_sha1.get(stage, ""), "status": iso.stage_status.get(stage, ""),
                             "wall_s": iso.timings_s.get(stage)}
    import opi
    data = {
        "script": str(Path(__file__).resolve()), "script_version": SCRIPT_VERSION,
        "written": datetime.now().isoformat(timespec="seconds"), "level": ctx.level.name,
        "orca": {"version": ctx.engine.orca_version, "binary": str(ctx.engine.orca_bin)},
        "opi_version": getattr(opi, "__version__", "?"), "python": platform.python_version(),
        "numpy": np.__version__, "host": platform.node(), "nprocs": ctx.level.ncores, "maxcore_MB": MAXCORE_MB,
        "vpt2_nprocs_group": VPT2_NPROCS_GROUP,
        "thresholds": {"geom_bohr_Eh": GEOM_TOL, "geom_maxiter": GEOM_MAXITER, "scf_revDSD": "VeryTightSCF",
                       "scf_B3LYP": "ExtremeSCF", "grid": "DEFGRID3 (all DFT stages but S0)",
                       "numfreq_increment_bohr": NUMFREQ_INCREMENT_BOHR, "numfreq": "central differences",
                       "vpt2": {"AnharmDisp": VPT2_ANHARM_DISP, "HessianCutoff": VPT2_HESSIAN_CUTOFF, "Z_Tol": Z_TOL}},
        "correlation": "revDSD MP2 part: RI-MP2, frozen core (ORCA default), relaxed density for the dipole",
        "aux_basis": {"revDSD": [k.keyword for k in ctx.level.dh_basis[1:]],
                      "B3LYP": [k.keyword for k in ctx.level.hyb_basis[1:]]},
        "masses_u": ISOTOPE_MASS, "mass_source": "Mendeleev 1.2.0 (= ORCA !Mass2016 most abundant isotopes)",
        "rot_const_factor_MHz_uA2": ROT_CONST_MHZ_AMU_A2,
        "guards": {"imag_noise_cm1": IMAG_NOISE_CM1, "soft_mode_cm1": SOFT_MODE_CM1, "dark_D": DARK_DIPOLE_D},
        "windows": {"B0_rel": B0_WINDOW_REL, "quartic_rel": QUARTIC_WINDOW_REL},
        "stages": stages, "flags": iso.flags,
    }
    (results_dir(iso, ctx) / "provenance.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def calibrate(iso: Isomer, ctx: Context) -> int:
    """One timed revDSD gradient at the production level (method matrix F01 Sec. 20.1) -> stage estimates."""
    job = WORK_DIR / iso.label / (ctx.level.folder_suffix or "") / "CAL_revdsd_engrad"
    job.mkdir(parents=True, exist_ok=True)
    s0 = last_geometry(stage_dir(iso, ctx, "S0"), STAGE_DIR["S0"])
    symbols, xyz = s0 if s0 else (iso.symbols, iso.coords)
    write_xyz(job / "start.xyz", symbols, xyz, f"{iso.label} calibration geometry")
    calc = build_stage("CAL", job, ctx.level)
    unchanged, _ = write_input(calc, job / "start.xyz")
    if ctx.dry_run:
        log(f"[DRY-RUN] calibration input written: {job}")
        return 0
    t0 = time.time()
    status = run_orca(calc, ctx.engine, STAGE_TIMEOUT_H["CAL"] * 3600, unchanged)
    t = time.time() - t0
    log(f"[CALIBRATE] revDSD EnGrad on {iso.label}: {status}, {t:.0f} s with {ctx.level.ncores} ranks")
    if status == "ok":
        n_disp = 6 * len(symbols) - 6
        log(f"  S1  very tight opt  ~{30 * t / 3600:.1f}-{60 * t / 3600:.1f} h (30-60 gradients) [E]")
        log(f"  S2a relaxed dipole  ~{1.2 * t / 3600:.1f} h")
        log(f"  S2b NumFreq         ~{n_disp * t / 3600:.1f} h ({n_disp} displaced gradients)")
        log("  S0/S3 (r2SCAN-3c, B3LYP) are not covered by this calibration [E]")
    return 0 if status in ("ok", "reused") else 1


def revdsd_duplicates(isomers: list[Isomer], ctx: Context) -> list[str]:
    """
    Pairs of isomers whose revDSD (S1) structures meet the method matrix's
    Stage-B rule. Every isomer is in the reference atom order (discover_isomers),
    so the RMSD only has to allow the molecule's equivalent-atom permutations.
    """
    done = []
    for iso in isomers:
        job = stage_dir(iso, ctx, "S1")
        text = read_text(job / f"{STAGE_DIR['S1']}.out")
        geo = last_geometry(job, STAGE_DIR["S1"])
        if geo and opt_converged(text) and final_energy(text) is not None and bonded_pairs(*geo) == MOL.bonds:
            done.append((iso, final_energy(text), rotational_constants(geo[1], masses_for(geo[0])), geo[1]))
    notes = []
    for (i1, e1, r1, c1), (i2, e2, r2, c2) in itertools.combinations(done, 2):
        de = abs(e1 - e2) * HARTREE_TO_KCAL
        drot = float(np.max(np.abs(r1 - r2) / r2))
        rmsd = best_rmsd(c1, c2, MOL.perms)
        if de <= DFT_DUP["energy_kcal"] and drot <= DFT_DUP["rot_rel"] and rmsd <= DFT_DUP["rmsd_A"]:
            notes.append(f"{i1.label} and {i2.label} converge to the SAME revDSD structure "
                         f"(dE {de:.3f} kcal/mol, dRot {100 * drot:.3f} %, RMSD {rmsd:.3f} A)")
    return notes


def write_summary(isomers: list[Isomer], results: dict[str, dict], ctx: Context, dup_notes: list[str]) -> Path:
    path = WORK_DIR / f"mw_summary{'_' + ctx.level.folder_suffix if ctx.level.folder_suffix else ''}.csv"
    energies = {k: r["E_revDSD_Eh"] + (r["ZPE_revDSD_harm_Eh"] or 0.0) for k, r in results.items()
                if r.get("E_revDSD_Eh") is not None}
    e_min = min(energies.values()) if energies else None
    e_min_el = min((r["E_revDSD_Eh"] for r in results.values()), default=None)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["isomer", "status", "dE_e_kcal_mol", "dE_0_kcal_mol", "A0_MHz", "B0_MHz", "C0_MHz",
                    "window_B_MHz", "mu_a_D", "mu_b_D", "mu_c_D", "kappa", "inertial_defect_uA2", "HCOOH_form",
                    "closest_contacts_A", "flags"])
        for iso in isomers:
            r = results.get(iso.label)
            if iso.stub_reason:
                w.writerow([iso.label, f"skipped: {iso.stub_reason}"] + [""] * 14)
            elif r is None:
                state = (f"requested stages done ({','.join(ctx.stages + ['S4'] * ctx.analyze)})" if iso.requested_done
                         else "not finished")
                w.writerow([iso.label, state] + [""] * 13 + ["; ".join(iso.flags)])
            else:
                w.writerow([iso.label, "done", f"{(r['E_revDSD_Eh'] - e_min_el) * HARTREE_TO_KCAL:.3f}",
                            f"{(energies[iso.label] - e_min) * HARTREE_TO_KCAL:.3f}",
                            *(f"{r['B0_MHz'][g]:.2f}" for g in "abc"), f"{r['B0_window_MHz']['b']:.1f}",
                            *(f"{r['mu_D'][g]:+.3f}" for g in "abc"), f"{r['kappa']:.4f}",
                            f"{r['inertia']['inertial_defect']:.4f}", r["geometry"]["HCOOH_form"],
                            r["geometry"]["closest_contacts_A"], "; ".join(iso.flags)])
        for note in dup_notes:
            w.writerow(["DUPLICATE CHECK", note] + [""] * 14)
    return path


def main(argv: list[str] | None = None) -> int:
    global _LOG_FILE, _NO_ORCA, VPT2_NPROCS_GROUP, WORK_DIR, JOB
    args = parse_args(argv)
    level = SMOKE if args.smoke else PRODUCTION
    if args.cores:
        level = Level(level.name, args.cores, level.dh_basis, level.hyb_basis, level.folder_suffix)
    if args.vpt2_groups:
        VPT2_NPROCS_GROUP = args.vpt2_groups
    names = STAGE_KEYS + ["S4"] if not args.stages else [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in names if s not in STAGE_KEYS + ["S4"]]
    if bad or not names:
        raise SystemExit(f"--stages: unknown stage(s) {', '.join(bad) or '(none given)'}; "
                         f"choose from {','.join(STAGE_KEYS)},S4")
    stages = [s for s in STAGE_KEYS if s in names]             # run order, whatever order was typed
    analyze = "S4" in names
    names = stages + ["S4"] * analyze
    WORK_DIR = (args.work_dir or HERE).resolve()
    if not WORK_DIR.is_dir():
        raise SystemExit(f"--work-dir {WORK_DIR} does not exist")
    JOB = load_job_settings(WORK_DIR)
    _NO_ORCA = args.analyze_only
    _LOG_FILE = WORK_DIR / f"mw_run_log{'_smoke' if args.smoke else ''}.txt"
    only = {int(x) for x in args.only.split(",")} if args.only else None

    log("=" * 78)
    log(f"Microwave predictions | revDSD + B3LYP hybrid | ORCA 6.1.1 + OPI 2.0 | {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 78)
    mode = "dry-run" if args.dry_run else "analyze-only" if args.analyze_only else "calibrate" if args.calibrate else "run"
    log(f"[PRE-FLIGHT] Script {Path(__file__).resolve()} v{SCRIPT_VERSION}")
    log(f"[PRE-FLIGHT] Level {level.name}, {level.ncores} ranks x {MAXCORE_MB} MB, mode {mode}, "
        f"stages {','.join(names)}, folder {WORK_DIR}")
    isomers = discover_isomers(WORK_DIR, only)
    if not isomers:
        raise SystemExit(f"No isomer-<number>.xyz files in {WORK_DIR}.")
    for iso in isomers:
        if iso.stub_reason:
            log(f"[PRE-FLIGHT] {iso.path.name}: skipped ({iso.stub_reason})")
    filled = [iso for iso in isomers if not iso.stub_reason]
    if MOL is not None:
        log(f"[PRE-FLIGHT] Molecule {MOL.name}: {len(MOL.symbols)} atoms, charge {JOB.charge}, multiplicity "
            f"{JOB.multiplicity}; equivalent atoms "
            + (", ".join("/".join(f"{MOL.symbols[i]}{i}" for i in g) for g in MOL.equivalent_groups) or "none"))
    log(f"[PRE-FLIGHT] Isomers to process: {', '.join(i.label for i in filled) or 'none'}")
    if not filled:
        log("Nothing to do: fill the isomer-NN.xyz stubs with the unique isomers first.")
        return 0
    needs_lock = not (args.dry_run or args.analyze_only)
    engine = resolve_engine(check_version=needs_lock)
    ctx = Context(level, engine, args.dry_run, args.analyze_only, stages, analyze)
    if needs_lock:
        acquire_orca_lock(f"hcooh-hsh-mw-predict ({mode}, {level.name}) in {WORK_DIR}")
    try:
        if args.calibrate:
            return calibrate(filled[0], ctx)
        results: dict[str, dict] = {}
        for iso in filled:
            try:
                res = process_isomer(iso, ctx)
            except (RuntimeError, ValueError, OSError) as exc:
                iso.flags.append(f"analysis error: {exc}")
                log(f"  [{iso.label}] ERROR: {exc}")
                res = None
            if res is not None:
                results[iso.label] = res
                log(f"  [{iso.label}] B0 = {res['B0_MHz']['a']:.1f} / {res['B0_MHz']['b']:.1f} / "
                    f"{res['B0_MHz']['c']:.1f} MHz, mu = {res['mu_D']['a']:+.2f} / {res['mu_D']['b']:+.2f} / "
                    f"{res['mu_D']['c']:+.2f} D, {res['n_lines']} lines")
            for flag in iso.flags:
                log(f"  [{iso.label}] flag: {flag}")
            if not args.dry_run:
                write_provenance(iso, ctx)
        if args.dry_run:
            log(f"\n[DRY-RUN] Inputs of stages {','.join(stages) or '(none)'} written for {len(filled)} isomer(s). "
                "Nothing was run.")
            return 0
        dups = revdsd_duplicates(filled, ctx)
        for note in dups:
            log(f"[DUPLICATE CHECK] {note}")
        summary = write_summary(isomers, results, ctx, dups)
        done = sum(1 for iso in filled if iso.requested_done)
        what = "complete" if names == STAGE_KEYS + ["S4"] else f"done with stages {','.join(names)}"
        log(f"\n[DONE] {done} of {len(filled)} isomer(s) {what} -> {summary}")
        return 0 if done == len(filled) else 2
    finally:
        if needs_lock:
            release_orca_lock()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        kill_all_running()
        release_orca_lock()
        log("\n[ABORTED] Ctrl+C: all ORCA jobs were stopped. Re-run to resume.")
        sys.exit(130)
    except Exception:
        kill_all_running()
        release_orca_lock()
        log("\n[ERROR] " + traceback.format_exc())
        sys.exit(1)
