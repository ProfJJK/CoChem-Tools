#!/usr/bin/env python3
"""
GPU4PySCF wB97X-V/def2-TZVPP Driver for HCOOH-H2S and HCOOH-N2O Campaigns
=========================================================================
Runs inside WSL2 Ubuntu or Linux (`~/.venvs/gpu4pyscf/bin/python`) on an
NVIDIA CUDA GPU (e.g. RTX 3090 24 GB VRAM) without touching the ORCA CPU lock.

Modes:
  1. `--mode cal-bench`:
     Reproduces the ORCA `cal_grid_wb97xv` constrained optimization (`{ B 0 5 C }`,
     `{ A 3 0 5 C }`) at `wB97X-V/def2-TZVPP` on GPU and compares wall time,
     energy, and geometry RMSD against the 2362.0 s (39.4 min) ORCA CPU baseline.
  2. `--mode saddle-audit`:
     Computes `wB97X-V/def2-TZVPP` analytic Hessians, harmonic vibrational
     frequencies, and stationary-point optimizations (TS or minimum) across:
       - HCOOH-H2S GFN2-xTB GOAT saddle points (`trans:cand_002`, `cis:cand_006`,
         `cis:cand_010`, `cis:cand_010_tsminus`)
       - HCOOH-N2O GFN2-xTB GOAT saddle points (`cand_015`, `cand_054`)
       - HCOOH-H2S special MACE-OFF24(M) / planar structures (`Isomer_04` C-H...S
         collapse, `Isomer_08` / `isomer-10` planar Cs, `Isomer_09` / `isomer-05`
         planar Cs)
  3. `--mode preopt-revdsd`:
     Pre-optimizes all 4 monomers (`trans-HCOOH`, `cis-HCOOH`, `H2S`, `N2O`),
     all 8 `HCOOH-H2S` working-set isomers (`00, 01, 02, 05, 06, 08, 10, 13`),
     and all 16 `HCOOH-N2O` working-set isomers (`00..18`) at `wB97X-V/def2-TZVPP`
     on the RTX 3090 GPU, verifies 0 imaginary frequencies via analytic Hessian
     (automatically following any imaginary out-of-plane mode if a planar input
     was trapped on a Cs saddle), backs up the original GFN2-xTB `.xyz` files into
     `gfn2_backup/`, and updates `revdsd/isomer-NN.xyz` so `revDSD` converges in
     minimal cycles.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HARTREE_TO_KCAL = 627.5094740631
BOHR_TO_ANG = 0.529177210903
PLANCK_CONVERSION_MHZ_AMU_A2 = 505379.006

MONOISOTOPIC_MASS: dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.00000000000,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
}


def ensure_wsl_cuda_env() -> None:
    """Ensure LD_LIBRARY_PATH includes the venv's nvidia/*/lib dirs; re-spawn via subprocess if needed."""
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
    res = subprocess.run([sys.executable] + sys.argv, env=env)
    raise SystemExit(res.returncode)


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


def read_multi_xyz(path: Path) -> list[tuple[str, list[str], list[list[float]]]]:
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    frames = []
    i = 0
    while i < len(raw_lines):
        if not raw_lines[i].strip():
            i += 1
            continue
        n = int(raw_lines[i].strip().split()[0])
        comment = raw_lines[i + 1].strip() if (i + 1) < len(raw_lines) else ""
        elems = []
        coords = []
        for k in range(n):
            parts = raw_lines[i + 2 + k].split()
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames.append((comment, elems, coords))
        i += 2 + n
    return frames


def write_xyz(path: Path, elems: list[str], coords: Any, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(elems)}", comment]
    for el, (x, y, z) in zip(elems, coords):
        lines.append(f"{el:<2s}  {float(x):15.10f}  {float(y):15.10f}  {float(z):15.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_rot_constants_and_pcc(elems: list[str], coords_arr: Any) -> tuple[float, float, float, float]:
    import numpy as np

    coords = np.asarray(coords_arr, dtype=float)
    masses = np.array([MONOISOTOPIC_MASS[e] for e in elems], dtype=float)
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
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
    evals = np.sort(np.linalg.eigvalsh(tensor))
    ia, ib, ic = float(evals[0]), float(evals[1]), float(evals[2])
    a_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ia if ia > 1e-6 else 0.0
    b_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ib if ib > 1e-6 else 0.0
    c_mhz = PLANCK_CONVERSION_MHZ_AMU_A2 / ic if ic > 1e-6 else 0.0
    pcc = 0.5 * (ia + ib - ic)
    return a_mhz, b_mhz, c_mhz, pcc


def kabsch_rmsd(coords_a: Any, coords_b: Any) -> float:
    import numpy as np

    p = np.asarray(coords_a, dtype=float)
    q = np.asarray(coords_b, dtype=float)
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


def build_mf(elems: list[str], coords: Any, fine_grid: bool = False):
    from pyscf import gto
    from gpu4pyscf.dft import rks

    atom_spec = [(el, tuple(float(v) for v in xyz)) for el, xyz in zip(elems, coords)]
    mol = gto.M(atom=atom_spec, basis="def2-tzvpp", unit="Angstrom", verbose=0)
    mf = rks.RKS(mol, xc="wb97x-v").density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = (99, 590) if fine_grid else (75, 302)
    mf.nlcgrids.atom_grid = (50, 194)
    mf.conv_tol = 1e-9 if fine_grid else 1e-8
    return mol, mf


def eval_hessian_and_freqs(elems: list[str], coords: Any, fine_grid: bool = False) -> dict[str, Any]:
    import numpy as np
    from pyscf.hessian import thermo

    t0 = time.time()
    mol, mf = build_mf(elems, coords, fine_grid=fine_grid)
    e_eh = float(mf.kernel())
    gobj = mf.nuc_grad_method()
    grad = np.asarray(gobj.kernel(), dtype=float)
    max_g = float(np.max(np.abs(grad)))
    rms_g = float(np.sqrt(np.mean(grad * grad)))

    hobj = mf.Hessian()
    hobj.auxbasis_response = 1
    hess = hobj.kernel()
    info = thermo.harmonic_analysis(mol, hess)
    freqs_raw = info["freq_wavenumber"]
    freqs_signed: list[float] = []
    for f in freqs_raw:
        if np.iscomplexobj(f) and abs(f.imag) > abs(f.real):
            freqs_signed.append(-float(abs(f.imag)))
        else:
            freqs_signed.append(float(np.real(f)))
    freqs_arr = np.array(freqs_signed, dtype=float)
    n_imag = int(np.sum(freqs_arr < -5.0))
    lowest_cm1 = float(freqs_arr[0]) if len(freqs_arr) > 0 else 0.0
    second_cm1 = float(freqs_arr[1]) if len(freqs_arr) > 1 else 0.0
    zpe_kcal = float(np.sum(freqs_arr[freqs_arr > 0.0]) * 0.5 * 0.002859144)
    norm_modes = np.asarray(info["norm_mode"], dtype=float)
    return {
        "E_Eh": e_eh,
        "max_grad_Eh_Bohr": max_g,
        "rms_grad_Eh_Bohr": rms_g,
        "freqs_cm1": freqs_arr,
        "n_imag": n_imag,
        "lowest_freq_cm1": lowest_cm1,
        "second_freq_cm1": second_cm1,
        "zpe_kcal_mol": zpe_kcal,
        "norm_modes": norm_modes,
        "hess": hess,
        "wall_s": time.time() - t0,
    }


def optimize_wb97xv(
    elems: list[str],
    coords: Any,
    *,
    constraints_txt: str | None = None,
    transition: bool = False,
    maxsteps: int = 80,
) -> dict[str, Any]:
    import numpy as np
    from pyscf.geomopt import geometric_solver

    t0 = time.time()
    mol, mf = build_mf(elems, coords, fine_grid=False)
    params: dict[str, Any] = {
        "maxsteps": maxsteps,
        "convergence_energy": 1e-6,
        "convergence_grms": 1e-4,
        "convergence_gmax": 3e-4,
        "convergence_drms": 1.2e-3,
        "convergence_dmax": 1.8e-3,
    }
    if transition:
        params["transition"] = True
        params["hessian"] = "first"

    tmp_c = None
    if constraints_txt:
        fd, tmp_c = tempfile.mkstemp(prefix="geom_constr_", suffix=".txt")
        os.close(fd)
        Path(tmp_c).write_text(constraints_txt, encoding="utf-8")
        params["constraints"] = tmp_c
        params["coordsys"] = "dlc"

    try:
        mol_opt = geometric_solver.optimize(mf, **params)
    finally:
        if tmp_c and os.path.exists(tmp_c):
            os.remove(tmp_c)

    opt_coords = np.asarray(mol_opt.atom_coords(unit="Angstrom"), dtype=float)
    mol_sp, mf_sp = build_mf(elems, opt_coords, fine_grid=True)
    e_final = float(mf_sp.kernel())
    return {
        "coords": opt_coords,
        "E_Eh": e_final,
        "opt_wall_s": time.time() - t0,
    }


def run_cal_bench(orca_root: Path) -> int:
    cal_dir = orca_root / "_publication" / "queue" / "calibration" / "cal_grid_wb97xv"
    _, elems, start_coords = read_xyz(cal_dir / "start.xyz")
    _, _, orca_coords = read_xyz(cal_dir / "cal_grid_wb97xv.xyz")

    # 1. Measure SP + Gradient + Analytical Hessian at the ORCA converged geometry
    h_orca = eval_hessian_and_freqs(elems, orca_coords, fine_grid=True)

    # 2. ORCA constraints: { B 0 5 C } and { A 3 0 5 C } (1-based in geomeTRIC: distance 1 6, angle 4 1 6)
    c_txt = "$freeze\ndistance 1 6\nangle 4 1 6\n"
    res = optimize_wb97xv(elems, start_coords, constraints_txt=c_txt, maxsteps=45)
    rmsd_vs_orca = kabsch_rmsd(res["coords"], orca_coords)

    # Read ORCA final energy and wall time
    orca_json = json.loads(
        (orca_root / "_publication" / "queue" / "done" / "002_cal_grid_wb97xv.json").read_text(encoding="utf-8")
    )
    orca_wall_s = float(orca_json["wall_s"])
    orca_out_txt = (cal_dir / "cal_grid_wb97xv.out").read_text(encoding="utf-8", errors="ignore")
    orca_e_hits = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", orca_out_txt)
    orca_e = float(orca_e_hits[-1]) if orca_e_hits else float("nan")

    speedup = orca_wall_s / res["opt_wall_s"]
    summary = {
        "benchmark": "cal_grid_wb97xv (constrained wB97X-V/def2-TZVPP Opt, 8 atoms)",
        "orca_cpu_ranks": 2,
        "orca_wall_s": orca_wall_s,
        "gpu4pyscf_rtx3090_opt_wall_s": round(res["opt_wall_s"], 2),
        "gpu4pyscf_rtx3090_sp_grad_hess_wall_s": round(h_orca["wall_s"], 2),
        "speedup_factor": round(speedup, 2),
        "E_orca_Eh": orca_e,
        "E_gpu4pyscf_at_orca_geom_Eh": round(h_orca["E_Eh"], 8),
        "dE_gpu_minus_orca_at_same_geom_kcal_mol": round((h_orca["E_Eh"] - orca_e) * HARTREE_TO_KCAL, 4),
        "E_gpu4pyscf_opt_Eh": round(res["E_Eh"], 8),
        "rmsd_gpu_vs_orca_A": round(rmsd_vs_orca, 5),
    }
    out_json = orca_root / "hcooh-hsh" / "method_comparison" / "gpu4pyscf_calibration_benchmark.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== GPU4PySCF Calibration Benchmark (cal_grid_wb97xv) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def run_saddle_audit(orca_root: Path) -> int:
    import numpy as np

    hsh_root = orca_root / "hcooh-hsh"
    nno_root = orca_root / "hcooh-nno"

    # Reference global minimum energies at wB97X-V/def2-TZVPP (from r2SCAN-3c Isomer_01 geometries)
    _, hsh01_el, hsh01_co = read_xyz(
        hsh_root / "maceoff24m_search_600" / "r2scan3c_runs" / "Isomer_01_r2scan3c" / "Isomer_01_r2scan3c.xyz"
    )
    hsh_gm_opt = optimize_wb97xv(hsh01_el, hsh01_co, maxsteps=40)
    e_ref_hsh = hsh_gm_opt["E_Eh"]

    _, nno01_el, nno01_co = read_xyz(
        nno_root / "maceoff24m_search_600" / "r2scan3c_runs" / "Isomer_01_r2scan3c" / "Isomer_01_r2scan3c.xyz"
    )
    nno_gm_opt = optimize_wb97xv(nno01_el, nno01_co, maxsteps=40)
    e_ref_nno = nno_gm_opt["E_Eh"]

    candidates: list[dict[str, Any]] = []

    # 1. HCOOH-H2S GFN2-xTB transition states (4 frames)
    hsh_ts_df = csv.DictReader(
        (hsh_root / "merged_isomers" / "transition_states.csv").read_text(encoding="utf-8-sig").splitlines()
    )
    hsh_ts_frames = read_multi_xyz(hsh_root / "merged_isomers" / "transition_states.xyz")
    for row, (cmt, el, co) in zip(hsh_ts_df, hsh_ts_frames):
        candidates.append(
            {
                "system": "HCOOH-H2S",
                "id": f"GFN2_TS_{row['Run']}_{row['Job']}",
                "source_level": "GFN2-xTB GOAT TS",
                "description": f"{row['Side_plus_relaxes_to']} <-> {row['Side_minus_relaxes_to']} ({row['Status']})",
                "elems": el,
                "coords": co,
                "ref_E": e_ref_hsh,
                "is_ts_candidate": True,
            }
        )

    # 2. HCOOH-N2O GFN2-xTB transition states (2 frames)
    nno_ts_df = csv.DictReader(
        (nno_root / "HCOOH-N2O_isomer_handout" / "non_unique_isomers" / "transition_states.csv")
        .read_text(encoding="utf-8-sig")
        .splitlines()
    )
    nno_ts_frames = read_multi_xyz(
        nno_root / "HCOOH-N2O_isomer_handout" / "non_unique_isomers" / "transition_states.xyz"
    )
    for row, (cmt, el, co) in zip(nno_ts_df, nno_ts_frames):
        candidates.append(
            {
                "system": "HCOOH-N2O",
                "id": f"GFN2_TS_{row['Transition_state']}",
                "source_level": "GFN2-xTB GOAT TS",
                "description": f"{row['Downhill_plus']} <-> {row['Downhill_minus']} ({row['Imag_freq_cm-1']} cm^-1)",
                "elems": el,
                "coords": co,
                "ref_E": e_ref_nno,
                "is_ts_candidate": True,
            }
        )

    # 3. HCOOH-H2S MACE-OFF24(M) / planar special structures
    for mid, desc in [
        ("Isomer_04", "MACE C-H...S floppy family (collapses -> Isomer_02 at DFT)"),
        ("Isomer_08", "MACE planar cis S-H...O=C + S-H...O(H) (relaxes non-planar -> isomer-13)"),
        ("Isomer_09", "MACE planar cis S-H...O=C (relaxes non-planar -> isomer-05)"),
    ]:
        _, el, co = read_xyz(
            hsh_root / "maceoff24m_search_600" / "HCOOH-HSH_MACE-OFF24m-search_isomers" / f"{mid}.xyz"
        )
        candidates.append(
            {
                "system": "HCOOH-H2S",
                "id": f"MACE_{mid}",
                "source_level": "MACE-OFF24(M)",
                "description": desc,
                "elems": el,
                "coords": co,
                "ref_E": e_ref_hsh,
                "is_ts_candidate": False,
            }
        )

    audit_rows: list[dict[str, Any]] = []
    for cand in candidates:
        t0 = time.time()
        print(f"[SADDLE-AUDIT] Evaluating {cand['system']} {cand['id']}...", flush=True)
        # Step A: Analytic Hessian at the input candidate geometry
        h_init = eval_hessian_and_freqs(cand["elems"], cand["coords"], fine_grid=False)
        a_i, b_i, c_i, pcc_i = compute_rot_constants_and_pcc(cand["elems"], cand["coords"])

        # Step B: Relax on wB97X-V/def2-TZVPP surface (if planar with imag mode, displace 0.12 A along mode 0 before opt)
        trial_coords = np.asarray(cand["coords"], dtype=float).copy()
        if not cand["is_ts_candidate"] and h_init["n_imag"] > 0 and pcc_i < 0.05:
            mode0 = h_init["norm_modes"][0]
            step = 0.12 / max(float(np.linalg.norm(mode0, axis=1).max()), 1e-6)
            trial_coords = trial_coords + step * mode0

        opt_res = optimize_wb97xv(cand["elems"], trial_coords, transition=False, maxsteps=50)
        h_opt = eval_hessian_and_freqs(cand["elems"], opt_res["coords"], fine_grid=True)
        a_f, b_f, c_f, pcc_f = compute_rot_constants_and_pcc(cand["elems"], opt_res["coords"])
        rmsd_relax = kabsch_rmsd(cand["coords"], opt_res["coords"])

        row_out = {
            "system": cand["system"],
            "candidate_id": cand["id"],
            "source_level": cand["source_level"],
            "description": cand["description"],
            "init_dE_wb97xv_kcal": round((h_init["E_Eh"] - cand["ref_E"]) * HARTREE_TO_KCAL, 3),
            "init_max_grad_Eh_Bohr": round(h_init["max_grad_Eh_Bohr"], 6),
            "init_n_imag_wb97xv": h_init["n_imag"],
            "init_lowest_freq_cm1": round(h_init["lowest_freq_cm1"], 2),
            "init_second_freq_cm1": round(h_init["second_freq_cm1"], 2),
            "init_Pcc_amuA2": round(pcc_i, 4),
            "relaxed_dE_wb97xv_kcal": round((h_opt["E_Eh"] - cand["ref_E"]) * HARTREE_TO_KCAL, 3),
            "relaxed_n_imag_wb97xv": h_opt["n_imag"],
            "relaxed_lowest_freq_cm1": round(h_opt["lowest_freq_cm1"], 2),
            "relaxed_A_MHz": round(a_f, 2),
            "relaxed_B_MHz": round(b_f, 2),
            "relaxed_C_MHz": round(c_f, 2),
            "relaxed_Pcc_amuA2": round(pcc_f, 4),
            "relax_RMSD_A": round(rmsd_relax, 4),
            "wall_s": round(time.time() - t0, 2),
        }
        audit_rows.append(row_out)
        print(
            f"  -> {cand['id']}: init n_imag={h_init['n_imag']} ({h_init['lowest_freq_cm1']:.1f} cm^-1) | "
            f"relaxed dE={row_out['relaxed_dE_wb97xv_kcal']:.2f} kcal/mol, n_imag={h_opt['n_imag']} "
            f"({h_opt['lowest_freq_cm1']:.1f} cm^-1), RMSD={rmsd_relax:.3f} A ({row_out['wall_s']:.1f} s)",
            flush=True,
        )

    out_csv = hsh_root / "method_comparison" / "wb97xv_gpu_saddle_audit.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    shutil.copyfile(out_csv, orca_root / "_publication" / "wb97xv_gpu_saddle_audit.csv")
    print(f"[SADDLE-AUDIT] Wrote {len(audit_rows)} rows to {out_csv}", flush=True)
    return 0


def run_preopt_revdsd(orca_root: Path) -> int:
    import numpy as np

    hsh_root = orca_root / "hcooh-hsh"
    nno_root = orca_root / "hcooh-nno"

    targets: list[dict[str, Any]] = []

    # 1. Monomers (shared + project-specific)
    for proj_name, proj_root, mono_list in [
        ("hcooh-hsh", hsh_root, ["trans-HCOOH", "cis-HCOOH", "H2S"]),
        ("hcooh-nno", nno_root, ["trans-HCOOH", "cis-HCOOH", "N2O"]),
    ]:
        for mname in mono_list:
            ws_m = proj_root / "publication" / "working_set" / "monomers" / f"{mname}.xyz"
            opt_start = proj_root / "publication" / "energetics" / "monomers" / mname / "opt" / "start.xyz"
            targets.append(
                {
                    "project": proj_name,
                    "kind": "monomer",
                    "label": mname,
                    "src_xyz": ws_m,
                    "dst_xyz_list": [opt_start],
                    "backup_dir": proj_root / "publication" / "energetics" / "monomers" / "gfn2_backup",
                    "preopt_dir": proj_root / "publication" / "energetics" / "monomers" / "wb97xv_preopt",
                }
            )

    # 2. HCOOH-H2S 8 working-set isomers (00, 01, 02, 05, 06, 08, 10, 13)
    hsh_isos = ["00", "01", "02", "05", "06", "08", "10", "13"]
    for nn in hsh_isos:
        lbl = f"isomer-{nn}"
        rev_xyz = hsh_root / "publication" / "energetics" / "revdsd" / f"{lbl}.xyz"
        targets.append(
            {
                "project": "hcooh-hsh",
                "kind": "dimer",
                "label": lbl,
                "src_xyz": rev_xyz,
                "dst_xyz_list": [rev_xyz],
                "backup_dir": hsh_root / "publication" / "energetics" / "revdsd" / "gfn2_backup",
                "preopt_dir": hsh_root / "publication" / "energetics" / "revdsd" / "wb97xv_preopt",
            }
        )

    # 3. HCOOH-N2O 16 working-set isomers
    nno_isos = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "12", "13", "15", "16", "18"]
    for nn in nno_isos:
        lbl = f"isomer-{nn}"
        rev_xyz = nno_root / "publication" / "energetics" / "revdsd" / f"{lbl}.xyz"
        targets.append(
            {
                "project": "hcooh-nno",
                "kind": "dimer",
                "label": lbl,
                "src_xyz": rev_xyz,
                "dst_xyz_list": [rev_xyz],
                "backup_dir": nno_root / "publication" / "energetics" / "revdsd" / "gfn2_backup",
                "preopt_dir": nno_root / "publication" / "energetics" / "revdsd" / "wb97xv_preopt",
            }
        )

    # Cache monomer optimizations so trans-HCOOH and cis-HCOOH are only computed once
    mono_cache: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for tgt in targets:
        t0 = time.time()
        lbl = tgt["label"]
        proj = tgt["project"]
        src_xyz: Path = tgt["src_xyz"]
        bak_dir: Path = tgt["backup_dir"]
        pre_dir: Path = tgt["preopt_dir"]
        bak_dir.mkdir(parents=True, exist_ok=True)
        pre_dir.mkdir(parents=True, exist_ok=True)

        bak_file = bak_dir / f"{lbl}.xyz"
        if not bak_file.exists():
            shutil.copyfile(src_xyz, bak_file)

        # Always read from the original backup so reruns are idempotent
        cmt, elems, init_coords_list = read_xyz(bak_file)
        init_coords = np.asarray(init_coords_list, dtype=float)
        a0, b0, c0, pcc0 = compute_rot_constants_and_pcc(elems, init_coords)

        if tgt["kind"] == "monomer" and lbl in mono_cache:
            cached = mono_cache[lbl]
            opt_coords = cached["opt_coords"]
            h_res = cached["h_res"]
            n_displaced = cached["n_displaced"]
        else:
            print(f"[PREOPT-REVDSD] Optimizing {proj} {lbl} ({len(elems)} atoms)...", flush=True)
            opt_res = optimize_wb97xv(elems, init_coords, maxsteps=60)
            opt_coords = opt_res["coords"]
            h_res = eval_hessian_and_freqs(elems, opt_coords, fine_grid=True)
            n_displaced = 0

            # If a planar stationary point has an imaginary out-of-plane mode (< -5 cm^-1), follow it once
            if tgt["kind"] == "dimer" and h_res["n_imag"] > 0:
                n_displaced = 1
                mode0 = h_res["norm_modes"][0]
                step = 0.15 / max(float(np.linalg.norm(mode0, axis=1).max()), 1e-6)
                disp_coords = opt_coords + step * mode0
                print(
                    f"  -> {proj} {lbl} had {h_res['n_imag']} imag mode ({h_res['lowest_freq_cm1']:.1f} cm^-1); "
                    f"displacing along mode 0 and re-optimizing...",
                    flush=True,
                )
                opt_res2 = optimize_wb97xv(elems, disp_coords, maxsteps=60)
                h_res2 = eval_hessian_and_freqs(elems, opt_res2["coords"], fine_grid=True)
                if h_res2["E_Eh"] <= h_res["E_Eh"] + 1e-5 and h_res2["n_imag"] < h_res["n_imag"]:
                    opt_coords = opt_res2["coords"]
                    h_res = h_res2

            if tgt["kind"] == "monomer":
                mono_cache[lbl] = {
                    "opt_coords": opt_coords,
                    "h_res": h_res,
                    "n_displaced": n_displaced,
                }

        a1, b1, c1, pcc1 = compute_rot_constants_and_pcc(elems, opt_coords)
        rmsd_shift = kabsch_rmsd(init_coords, opt_coords)

        new_comment = (
            f"{proj} {lbl} pre-optimized at wB97X-V/def2-TZVPP (GPU4PySCF RTX 3090) "
            f"E={h_res['E_Eh']:.10f} Eh | n_imag={h_res['n_imag']}"
        )
        write_xyz(pre_dir / f"{lbl}_wb97xv.xyz", elems, opt_coords, new_comment)
        for dst_p in tgt["dst_xyz_list"]:
            write_xyz(dst_p, elems, opt_coords, new_comment)

        wall_s = time.time() - t0
        row = {
            "project": proj,
            "kind": tgt["kind"],
            "label": lbl,
            "E_wb97xv_Eh": round(h_res["E_Eh"], 8),
            "ZPE_wb97xv_kcal_mol": round(h_res["zpe_kcal_mol"], 4),
            "max_grad_Eh_Bohr": round(h_res["max_grad_Eh_Bohr"], 6),
            "n_imag_wb97xv": h_res["n_imag"],
            "lowest_freq_cm1": round(h_res["lowest_freq_cm1"], 2),
            "displaced_from_saddle": n_displaced,
            "RMSD_GFN2_to_wb97xv_A": round(rmsd_shift, 4),
            "A_GFN2_MHz": round(a0, 2),
            "B_GFN2_MHz": round(b0, 2),
            "C_GFN2_MHz": round(c0, 2),
            "A_wb97xv_MHz": round(a1, 2),
            "B_wb97xv_MHz": round(b1, 2),
            "C_wb97xv_MHz": round(c1, 2),
            "Pcc_GFN2_amuA2": round(pcc0, 4),
            "Pcc_wb97xv_amuA2": round(pcc1, 4),
            "wall_s": round(wall_s, 2),
        }
        summary_rows.append(row)
        print(
            f"  -> {proj} {lbl}: E={h_res['E_Eh']:.6f} Eh | RMSD(GFN2->wB97X-V)={rmsd_shift:.3f} A | "
            f"B: {b0:.1f} -> {b1:.1f} MHz | n_imag={h_res['n_imag']} ({h_res['lowest_freq_cm1']:.1f} cm^-1) "
            f"[{wall_s:.1f} s]",
            flush=True,
        )

    out_csv = hsh_root / "method_comparison" / "wb97xv_gpu_revdsd_preopt.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    shutil.copyfile(out_csv, orca_root / "_publication" / "wb97xv_gpu_revdsd_preopt.csv")
    print(f"[PREOPT-REVDSD] Wrote {len(summary_rows)} rows to {out_csv}", flush=True)
    return 0


def run_fix_nno_10(orca_root: Path) -> int:
    import numpy as np

    rev_dir = orca_root / "hcooh-nno" / "publication" / "energetics" / "revdsd"
    disp_p = rev_dir / "isomer-10_planar_cs_backup" / "results" / "isomer-10_imag_follow_plus.xyz"
    if not disp_p.exists():
        disp_p = rev_dir / "isomer-10.xyz"
    _, elems, coords = read_xyz(disp_p)
    a0, b0, c0, pcc0 = compute_rot_constants_and_pcc(elems, coords)
    print(f"[FIX-NNO-10] Displaced start: A={a0:.2f}, B={b0:.2f}, C={c0:.2f} MHz, Pcc={pcc0:.4f} amu*A^2", flush=True)

    opt = optimize_wb97xv(elems, coords, maxsteps=60)
    h = eval_hessian_and_freqs(elems, opt["coords"], fine_grid=True)
    opt_coords = opt["coords"]
    if h["n_imag"] > 0:
        mode0 = h["norm_modes"][0]
        step = 0.20 / max(float(np.linalg.norm(mode0, axis=1).max()), 1e-6)
        disp2 = opt_coords + step * mode0
        opt2 = optimize_wb97xv(elems, disp2, maxsteps=60)
        h2 = eval_hessian_and_freqs(elems, opt2["coords"], fine_grid=True)
        if h2["n_imag"] < h["n_imag"]:
            opt_coords = opt2["coords"]
            h = h2

    a1, b1, c1, pcc1 = compute_rot_constants_and_pcc(elems, opt_coords)
    print(
        f"[FIX-NNO-10] wB97X-V opt: E={h['E_Eh']:.10f} Eh | n_imag={h['n_imag']} "
        f"| lowest={h['lowest_freq_cm1']:.2f} cm^-1 | A={a1:.2f}, B={b1:.2f}, C={c1:.2f} MHz | Pcc={pcc1:.4f} amu*A^2",
        flush=True,
    )
    # Note: if wB97X-V is nearly planar (Pcc < 0.2) whereas revDSD has a -18 cm^-1 out-of-plane imaginary mode,
    # apply a 0.25 A out-of-plane displacement along the softest out-of-plane mode of wB97X-V or keep the revDSD
    # displaced coordinates if S0_r2scan3c needs finite out-of-plane symmetry breaking.
    comment = (
        f"hcooh-nno isomer-10 symmetry-broken non-planar C1 start "
        f"(wB97X-V/def2-TZVPP E={h['E_Eh']:.10f} Eh, n_imag={h['n_imag']}, lowest={h['lowest_freq_cm1']:.2f} cm^-1)"
    )
    write_xyz(rev_dir / "wb97xv_preopt" / "isomer-10_wb97xv_nonplanar.xyz", elems, opt_coords, comment)
    # Also check out-of-plane span of opt_coords vs disp_p
    return 0


def main() -> int:
    ensure_wsl_cuda_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("cal-bench", "saddle-audit", "preopt-revdsd", "fix-nno-10", "all"),
        default="all",
    )
    ap.add_argument("--orca-root", type=Path, default=Path(os.environ.get("COCHEM_WORKDIR", ".")))
    args = ap.parse_args()

    if args.mode == "fix-nno-10":
        return run_fix_nno_10(args.orca_root)
    if args.mode in ("cal-bench", "all"):
        run_cal_bench(args.orca_root)
    if args.mode in ("saddle-audit", "all"):
        run_saddle_audit(args.orca_root)
    if args.mode in ("preopt-revdsd", "all"):
        run_preopt_revdsd(args.orca_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
