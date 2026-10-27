#!/usr/bin/env python3
"""
================================================================================
merge_isomer_sets.py -- HCOOH-H2S: one isomer set from several GOAT runs (plan B1)
================================================================================
1. PURPOSE:
   Pools every characterized structure (jobs/cand_* : TightOpt + NumFreq at
   GFN2-xTB) of the GOAT runs started from trans-, cis- and gauche-HCOOH,
   removes duplicates with the method matrix's Stage-B rule, relabels the
   unique minima isomer-00, -01, ... by energy (isomer-00 = lowest, the
   numbering used everywhere afterwards) and writes an audit log in which
   every raw GOAT structure and every characterized structure appears once,
   with its fate. Runs no ORCA; reads the finished run folders only.

2. DEDUPLICATION (method matrix F03 App. A.1, Stage B):
   same isomer only if |dE| <= 0.05 kcal/mol AND every rotational constant
   within 0.1 % AND RMSD <= 0.125 A, with the RMSD minimized over rotation,
   mirror reflection and the H6/H7 swap of H2S. This is stricter than the GOAT
   tool's own rule (0.10 kcal/mol, RMSD only), so the merge can only split
   what the runs merged, never merge more. Pairs that are close but not merged
   are listed in near_duplicates.csv for the students.

3. FATE OF EVERY STRUCTURE:
   raw GOAT structure -> the characterization job that started from it, or a
   duplicate of one (matched to the job's start geometry, so the result does
   not depend on any run's sorting); characterized structure -> isomer-NN,
   duplicate of isomer-NN, transition state (with the minima it connects),
   rejected (broken/dissociated after optimization, failed, not converged).

4. OUTPUT (merged_isomers/ next to this script):
   isomer-NN.xyz, all_unique_isomers.xyz, unique_isomer_energy_table.csv,
   duplicates.csv/.xyz, transition_states.csv/.xyz, near_duplicates.csv,
   rejected.csv/.xyz, run_isomer_map.csv, merge_audit.csv, merge_report.txt

5. USAGE:
   python merge_isomer_sets.py                       the three runs (all must be finished)
   python merge_isomer_sets.py --run trans=hcooh-hsh-goat-xtb --run cis=cis-hcooh-goat
   python merge_isomer_sets.py --out other_folder
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goat_isomer_search as gis                     # geometry tools, parsers, dedup (no ORCA is started)

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = [("trans", "hcooh-hsh-goat-xtb"), ("cis", "cis-hcooh-goat"), ("gauche", "gauche-hcooh-goat")]
STAGE_B = {"energy_kcal": 0.05, "rot_rel": 0.001, "rmsd_A": 0.125}           # method matrix F03 App. A.1
RUN_DEDUP = {"energy_kcal": gis.DEDUP_ENERGY_TOL_KCAL, "rmsd_A": gis.DEDUP_RMSD_TOL_A}   # what the runs used
NEAR_DUP = {"energy_kcal": 0.50, "rot_rel": 0.02, "rmsd_A": 0.50}
WINDOW_KCAL = 12.0
SAME_GEOMETRY_A = 1e-6                               # raw structure == job start geometry


@dataclass(eq=False)
class Characterized:
    """One cand_* job of one run, with everything the merge needs."""

    run: str
    tag: str
    s: gis.Structure3D
    status: str
    start: np.ndarray
    start_e: float
    raw_sources: list[str] = field(default_factory=list)
    fate: str = ""


def hcooh_form(c: np.ndarray) -> str:
    """trans-HCOOH: H2-O1-C0=O3 near 0 deg (the stable monomer); cis: near 180 deg."""
    return "trans" if abs(gis.dihedral_deg(c[2], c[1], c[0], c[3])) < 90.0 else "cis"


def characterize_from_job(run: str, job: Path) -> Characterized:
    """Reads a finished cand_* job exactly the way goat_isomer_search.characterize() judged it."""
    tag = job.name
    start_frames = gis.read_xyz_frames(job / "start.xyz")
    _, symbols, start = start_frames[0]
    start_e = gis.energy_from_comment(start_frames[0][0]) or 0.0
    out, xyz = job / f"{tag}.out", job / f"{tag}.xyz"
    text = out.read_text(encoding="utf-8", errors="ignore") if out.exists() else ""
    if "ORCA TERMINATED NORMALLY" not in text or not xyz.exists():
        s = gis.Structure3D(symbols, start, start_e, sources=[f"{run}:{tag}"], char_job=f"{run}/jobs/{tag}",
                            status="FAILED or unfinished: start geometry kept")
        return Characterized(run, tag, s, s.status, start, start_e)
    _, symbols, coords = gis.read_xyz_frames(xyz)[-1]
    energy = gis.final_energy(out)
    freqs, zpe = gis.frequencies_and_zpe(out)
    s = gis.Structure3D(symbols, coords, energy if energy is not None else start_e, sources=[f"{run}:{tag}"],
                        freqs_cm1=freqs, zpe_eh=zpe, char_job=f"{run}/jobs/{tag}")
    s.opt_converged = "THE OPTIMIZATION HAS CONVERGED" in text
    imag = [f for f in freqs if f < 0.0]
    if not s.opt_converged:
        status = "opt NOT converged"
    elif not freqs:
        status = "no frequencies found"
    elif not imag:
        status = "minimum"
    elif all(f > -gis.IMAG_NOISE_CM1 for f in imag):
        status = f"minimum (tiny imag {min(imag):.0f} cm-1, numerical noise)"
    else:
        status = f"SADDLE POINT ({len(imag)} imag, {min(imag):.0f} cm-1)"
    s.status = status
    return Characterized(run, tag, s, status, start, start_e)


def load_run(run: str, folder: Path, topo: gis.Topology | None) -> tuple[list[Characterized], list[list[str]],
                                                                          gis.Topology]:
    """All characterized jobs of one run plus the fate of each raw GOAT structure."""
    if not (folder / "run_manifest.json").exists() or not (folder / "all_raw_structures.xyz").exists():
        raise SystemExit(f"Run '{run}' ({folder}) is not finished (no run_manifest.json / all_raw_structures.xyz).")
    jobs = sorted((p for p in (folder / "jobs").glob("cand_*") if p.is_dir()), key=gis.natural_key)
    chars = [characterize_from_job(run, j) for j in jobs]
    if topo is None:
        seed = sorted((folder / "jobs" / "seeds").glob("*.xyz"), key=gis.natural_key)[0]
        _, sym, xyz = gis.read_xyz_frames(seed)[0]
        topo = gis.Topology.from_reference(sym, xyz)
    for c in chars:
        if c.s.symbols != topo.symbols:
            raise SystemExit(f"{run}:{c.tag} has a different atom list/order: runs cannot be merged.")

    # raw GOAT structures -> the job that started from them, or the job they were merged into
    # (same rule the run applied: dE <= 0.10 kcal/mol AND RMSD <= 0.125 A against the job's start)
    starts = [c for c in chars if not c.tag.endswith(("_tsplus", "_tsminus"))]
    raw_rows = []
    for comment, sym, xyz in gis.read_xyz_frames(folder / "all_raw_structures.xyz"):
        parts = comment.split()
        raw_tag, energy = parts[0], float(parts[-1])
        raw = gis.Structure3D(sym, xyz, energy)
        reason = gis.screen_structure(raw, topo)
        fate = f"rejected by the run: {reason}" if reason else ""
        if not fate:
            for c in starts:
                if np.abs(c.start - xyz).max() <= SAME_GEOMETRY_A:
                    fate = f"characterized as {run}:{c.tag}"
                    c.raw_sources.append(raw_tag)
                    break
        if not fate:
            for c in sorted(starts, key=lambda x: x.start_e):      # the run compared in this order too
                de = abs(energy - c.start_e) * gis.HARTREE_TO_KCAL
                rmsd = gis.best_rmsd(xyz, c.start, topo.permutations)
                if de <= RUN_DEDUP["energy_kcal"] and rmsd <= RUN_DEDUP["rmsd_A"]:
                    fate = f"duplicate of {run}:{c.tag} in the run (dE {de:.3f} kcal/mol, RMSD {rmsd:.3f} A)"
                    c.raw_sources.append(raw_tag)
                    break
        raw_rows.append([run, "raw GOAT", raw_tag, f"{energy:.8f}", "", fate or "UNMATCHED (check by hand)"])
    return chars, raw_rows, topo


def rot(s: gis.Structure3D) -> tuple[float, float, float]:
    return gis.rotational_constants_mhz(s.symbols, s.coords)


def pair_metrics(a: gis.Structure3D, b: gis.Structure3D, topo: gis.Topology) -> tuple[float, float, float]:
    """dE (kcal/mol), largest relative rotational-constant difference, best RMSD (A)."""
    return (abs(a.energy_eh - b.energy_eh) * gis.HARTREE_TO_KCAL, gis.max_rel_rot_diff(rot(a), rot(b)),
            gis.best_rmsd(a.coords, b.coords, topo.permutations))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def contact(s: gis.Structure3D, topo: gis.Topology, rank: int) -> str:
    return gis.contact_label(s.coords, s.symbols, topo, rank)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge the HCOOH-H2S GOAT runs into one isomer set (Stage-B rule)")
    p.add_argument("--run", action="append", help="name=folder (repeatable); default: trans, cis, gauche")
    p.add_argument("--out", type=Path, default=HERE / "merged_isomers")
    args = p.parse_args(argv)
    runs = [tuple(r.split("=", 1)) for r in args.run] if args.run else DEFAULT_RUNS
    out = args.out if args.out.is_absolute() else HERE / args.out
    out.mkdir(parents=True, exist_ok=True)

    topo = None
    chars: list[Characterized] = []
    audit: list[list[str]] = []
    for name, folder in runs:
        c, raw_rows, topo = load_run(name, (HERE / folder).resolve(), topo)
        chars += c
        audit += raw_rows

    # Classify every characterized structure
    by_key = {(c.run, c.tag): c for c in chars}
    minima, ts_list, rejected = [], [], []
    for c in chars:
        reason = gis.screen_structure(c.s, topo)
        if reason:
            c.fate = f"rejected: {reason} (after TightOpt)"
            rejected.append(c)
        elif c.status.startswith("SADDLE POINT"):
            ts_list.append(c)
        elif c.status.startswith("minimum"):
            minima.append(c)
        else:
            c.fate = f"rejected: {c.status}"
            rejected.append(c)

    # Stage-B dedup on copies (deduplicate() extends the kept member's provenance)
    work = {id(c): copy.deepcopy(c.s) for c in minima}
    back = {id(work[id(c)]): c for c in minima}
    for c in minima:
        work[id(c)].sources = [f"{c.run}:{c.tag}"] + [f"{c.run}:{r}" for r in c.raw_sources]
        work[id(c)].n_raw = max(1, len(c.raw_sources))
    unique, links = gis.deduplicate(list(work.values()), topo.permutations, STAGE_B["energy_kcal"],
                                    STAGE_B["rmsd_A"], rot_rel_tol=STAGE_B["rot_rel"])
    emin = unique[0].energy_eh
    kept = [u for u in unique if (u.energy_eh - emin) * gis.HARTREE_TO_KCAL <= WINDOW_KCAL]
    for k, u in enumerate(kept):
        u.label = f"isomer-{k:02d}"
        back[id(u)].fate = u.label
    for u in unique:
        if u not in kept:
            back[id(u)].fate = f"above {WINDOW_KCAL} kcal/mol window"
    dup_rows = []
    for item, u, de, rmsd in links:
        target = u.label or back[id(u)].fate
        drot = gis.max_rel_rot_diff(rot(item), rot(u))
        back[id(item)].fate = f"duplicate of {target}"
        dup_rows.append([target, back[id(item)].run, back[id(item)].tag, f"{item.energy_eh:.8f}", f"{de:.4f}",
                         f"{100 * drot:.4f}", f"{rmsd:.4f}", f"{back[id(u)].run}:{back[id(u)].tag}"])

    # Transition states: the minima their two followed sides relaxed into
    ts_rows = []
    for c in ts_list:                    # label first, so an end that is itself a saddle reads correctly
        c.fate = f"transition state {c.run}:{c.tag}"
    ends_of = {}
    for c in ts_list:
        sides = [by_key.get((c.run, f"{c.tag}_ts{side}")) for side in ("plus", "minus")]
        ends_of[id(c)] = [s.fate.replace("duplicate of ", "") if s else "not followed" for s in sides]
    for c in ts_list:
        ends = ends_of[id(c)]
        c.fate = "transition state between " + " and ".join(ends)
        ts_rows.append([c.run, c.tag, f"{c.s.energy_eh:.8f}", f"{(c.s.energy_eh - emin) * gis.HARTREE_TO_KCAL:.3f}",
                        c.status, ends[0], ends[1], hcooh_form(c.s.coords)])

    # Exports
    for old in out.glob("isomer-*.xyz"):
        old.unlink()
    rows, comments = [], []
    for u in kept:
        c0 = back[id(u)]
        a, b, cc = rot(u)
        de = (u.energy_eh - emin) * gis.HARTREE_TO_KCAL
        z0 = (kept[0].zpe_eh or 0.0)
        runs_found = sorted({s.split(":")[0] for s in u.sources})
        comment = (f"{u.label} E={u.energy_eh:.8f} Eh dE={de:.3f} kcal/mol A/B/C={a:.1f}/{b:.1f}/{cc:.1f} MHz "
                   f"{hcooh_form(u.coords)}-HCOOH GFN2-xTB [{c0.run}:{c0.tag}]")
        comments.append(comment)
        gis.write_xyz(out / f"{u.label}.xyz", [u], [comment])
        com = gis.com_distance(u.symbols, u.coords, topo)
        rows.append([u.label, f"{de:.3f}", f"{de / gis.HARTREE_TO_KCAL * gis.HARTREE_TO_CM1:.1f}",
                     f"{(de + ((u.zpe_eh or 0.0) - z0) * gis.HARTREE_TO_KCAL):.3f}", f"{u.energy_eh:.8f}",
                     f"{u.zpe_eh:.6f}" if u.zpe_eh is not None else "", f"{a:.2f}", f"{b:.2f}", f"{cc:.2f}",
                     hcooh_form(u.coords), contact(u, topo, 0), contact(u, topo, 1),
                     f"{com:.3f}" if com is not None else "", f"{min(u.freqs_cm1):.1f}" if u.freqs_cm1 else "",
                     u.n_raw, " ".join(runs_found), f"{c0.run}:{c0.tag}"])
    gis.write_xyz(out / "all_unique_isomers.xyz", kept, comments)
    write_csv(out / "unique_isomer_energy_table.csv",
              ["Isomer", "dE_kcal_mol", "dE_cm-1", "dE_ZPE_kcal_mol", "E_Eh", "ZPE_Eh", "A_MHz", "B_MHz", "C_MHz",
               "HCOOH_form", "Closest_contact_A", "Second_contact_A", "COM_distance_A", "Lowest_freq_cm-1",
               "Times_found", "Found_in_runs", "Representative_job"], rows)
    write_csv(out / "duplicates.csv", ["Duplicate_of", "Run", "Job", "E_Eh", "dE_kcal_mol", "max_dRot_pct", "RMSD_A",
                                       "Kept_job"], dup_rows)
    dups = [back[id(item)] for item, *_ in links]
    gis.write_xyz(out / "duplicates.xyz", [c.s for c in dups], [f"{c.run}:{c.tag} {c.fate} E={c.s.energy_eh:.8f}"
                                                                 for c in dups])
    write_csv(out / "transition_states.csv", ["Run", "Job", "E_Eh", "dE_kcal_mol", "Status", "Side_plus_relaxes_to",
                                              "Side_minus_relaxes_to", "HCOOH_form"], ts_rows)
    gis.write_xyz(out / "transition_states.xyz", [c.s for c in ts_list], [f"{c.run}:{c.tag} {c.fate}" for c in ts_list])
    write_csv(out / "rejected.csv", ["Run", "Job", "E_Eh", "Reason"],
              [[c.run, c.tag, f"{c.s.energy_eh:.8f}", c.fate] for c in rejected])
    gis.write_xyz(out / "rejected.xyz", [c.s for c in rejected], [f"{c.run}:{c.tag} {c.fate}" for c in rejected])

    near = []
    for x, y in itertools.combinations(kept, 2):
        de, drot, rmsd = pair_metrics(x, y, topo)
        if de <= NEAR_DUP["energy_kcal"] and (drot <= NEAR_DUP["rot_rel"] or rmsd <= NEAR_DUP["rmsd_A"]):
            near.append([x.label, y.label, f"{de:.3f}", f"{100 * drot:.2f}", f"{rmsd:.3f}"])
    write_csv(out / "near_duplicates.csv", ["Isomer_A", "Isomer_B", "dE_kcal_mol", "max_dRot_pct", "RMSD_A"], near)

    # Each run's own isomer list -> merged label (labels Isomer_NN or isomer-NN)
    map_rows = []
    for name, folder in runs:
        for f in sorted((HERE / folder / "isomers").glob("*.xyz"), key=gis.natural_key):
            if not re.match(r"(?i)^isomer[-_]\d+\.xyz$", f.name):
                continue
            _, sym, xyz = gis.read_xyz_frames(f)[0]
            job = next((c for c in chars if c.run == name and np.abs(c.s.coords - xyz).max() <= 1e-6), None)
            map_rows.append([name, f.stem, job.fate.replace("duplicate of ", "") if job else "no matching job",
                             job.tag if job else ""])
    write_csv(out / "run_isomer_map.csv", ["Run", "Run_isomer", "Merged_isomer", "Job"], map_rows)

    audit += [[c.run, "characterized", c.tag, f"{c.s.energy_eh:.8f}", c.status, c.fate] for c in chars]
    write_csv(out / "merge_audit.csv", ["Run", "Stage", "Structure", "E_Eh", "Status", "Fate"], audit)

    report = [f"HCOOH-H2S merged isomer set ({datetime.now():%Y-%m-%d %H:%M}), GFN2-xTB",
              "Runs: " + ", ".join(f"{n} = {f}" for n, f in runs),
              f"Stage-B rule: |dE| <= {STAGE_B['energy_kcal']} kcal/mol AND dA,dB,dC <= {100 * STAGE_B['rot_rel']:g} % "
              f"AND RMSD <= {STAGE_B['rmsd_A']} A (rotation, reflection, H6/H7 swap)",
              f"Characterized structures: {len(chars)}  minima: {len(minima)}  transition states: {len(ts_list)}  "
              f"rejected: {len(rejected)}", f"Unique minima: {len(kept)} (+{len(unique) - len(kept)} above window)", ""]
    report += [f"  {r[0]}  dE {r[1]:>7s} kcal/mol  A/B/C {float(r[6]):8.1f} {float(r[7]):7.1f} {float(r[8]):7.1f} MHz  "
               f"{r[9]:<5s}  {r[10]:<18s} found {r[14]}x in {r[15]}" for r in rows]
    report += ["", f"Transition states: {len(ts_rows)}"] + [f"  {r[0]}:{r[1]}  dE {r[3]} kcal/mol  -> {r[5]} / {r[6]}"
                                                            for r in ts_rows]
    report += ["", f"Near-duplicate pairs (not merged): {len(near)}"] + [f"  {' '.join(r)}" for r in near]
    unmatched = [r for r in audit if r[5].startswith("UNMATCHED")]
    report += ["", f"Raw GOAT structures: {sum(1 for r in audit if r[1] == 'raw GOAT')} "
                   f"(unmatched: {len(unmatched)})"]
    (out / "merge_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\n-> {out}")
    return 0 if not unmatched else 2


if __name__ == "__main__":
    sys.exit(main())
