#!/usr/bin/env python3
"""
Generate publication-grade comparison figures comparing isomer search wall-clock time
vs. unique isomers discovered for the custom in-process batched GPU MACE-OFF24(M)
engine vs. ORCA 6.1.1 GFN2-xTB GOAT across both binary complexes:
  1. HCOOH...H2S (Project SR3)
  2. HCOOH...N2O (Project SR4)
  3. Combined 2x2 cross-complex benchmark figure
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Publication typography (ACS J. Phys. Chem. A standard)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe Okabe-Ito / ACS palette
COL_MACE_FAM = "#0072B2"      # Deep royal blue (MACE-OFF24M conformational families)
COL_MACE_RAW = "#56B4E9"      # Sky blue (MACE-OFF24M pre-family tight minima)
COL_MACE_DFT = "#009E73"      # Bluish green (MACE -> DFT verified minima)
COL_GOAT_RAW = "#D55E00"      # Vermilion/orange-red (GFN2-xTB GOAT working-set / raw)
COL_GOAT_DFT = "#CC79A7"      # Reddish purple (GFN2-xTB GOAT -> DFT unique basins)
COL_GRAY     = "#555555"


def get_h2s_search_data() -> dict:
    """
    Exact empirical timing and isomer discovery trajectory for HCOOH...H2S (SR3)
    from maceoff24m_search_600/search_run.log, rebuild_run.log, and GOAT run_log.txt.
    """
    # MACE-OFF24(M): 1,204 starts (4 seeds + 600 trans + 600 cis)
    # Hits per final family (out of 1204 starts): [299, 185, 103, 17, 183, 65, 206, 66, 80]
    # Phase 0+1: 0.25 min (monomers)
    # Phase 3 coarse batches (300 starts each):
    #   Batch 1 (trans 1..300): +161s -> t = 2.93 min (all 4 trans families hit: 4 families, 8 tight sub-basins)
    #   Batch 2 (trans 301..600): +157s -> t = 5.55 min (4 trans families saturated)
    #   Batch 3 (cis 1..300): +166s -> t = 8.32 min (all 5 cis families hit: 9/9 families total, 13 tight sub-basins!)
    #   Batch 4 (cis 301..600): +173s -> t = 11.20 min (9/9 families saturated across 1,200 random starts)
    #   Batch 5 (4 seeds): +28s -> t = 11.67 min (365 coarse basins -> 9/9 families)
    # Phase 4 tight float64 BFGS refinement (365 basins -> 15 tight minima): +1155s -> t = 30.92 min
    # Phase 6+6b analytic MACE Hessian + saddle follow + hierarchical dedup: +41s -> t = 31.60 min
    mace_time_min = np.array([0.0, 0.25, 2.93, 5.55, 8.32, 11.20, 11.67, 30.92, 31.60])
    mace_families = np.array([0,   0,    4,    4,    9,    9,     9,     9,     9])
    mace_tight_13 = np.array([0,   0,    7,    8,    13,   13,    13,    13,    13])
    mace_dft_ver  = np.array([0,   0,    3,    3,    8,    8,     8,     8,     8])

    # ORCA 6.1.1 GFN2-xTB GOAT:
    # Trans seeds (hcooh-hsh-goat-xtb/run_log.txt):
    #   seed0: 36.8 min -> finds Isomer_01 (00), Isomer_02 (01) [2 trans]
    #   seed1: +51.9 min (88.7 min) -> finds Isomer_03 (02) [3 trans]
    #   seed2: +42.2 min (130.9 min) -> [3 trans]
    #   seed4: +52.1 min (183.0 min) -> [3 trans]
    #   Phase 4 TightOpt+NumFreq + saddle follow: +15.2 min (198.2 min) -> 3 trans minima
    # Cis seed (cis-hcooh-goat/run_log.txt):
    #   start_cis quench + GOAT (25.2 min) + Phase 4 + saddle follow (16.7 min) = +41.9 min (240.1 min)
    #   Adds 5 working-set cis conformers (8 total working-set; 7 genuine DFT basins; misses MACE Isomer_06)
    # Gauche seed (gauche-hcooh-goat/run_log.txt):
    #   start_gauche quench + GOAT (46.8 min) + Phase 4 (4.4 min) = +51.2 min (291.3 min)
    #   Relaxes back to the 3 trans minima (0 new isomers)
    goat_time_min = np.array([0.0, 36.8, 88.7, 130.9, 183.0, 198.2, 223.4, 240.1, 291.3])
    goat_ws_count = np.array([0,   2,    3,    3,     3,     3,     8,     8,     8])
    goat_dft_ver  = np.array([0,   2,    3,    3,     3,     3,     7,     7,     7])
    goat_labels   = [
        "", "seed0\n(trans)", "seed1\n(trans)", "seed2\n(trans)",
        "seed4\n(trans)", "trans\nNumFreq", "start_cis\nGOAT", "cis\nNumFreq", "start_gauche\n(no new)"
    ]

    return {
        "mace_time_min": mace_time_min,
        "mace_families": mace_families,
        "mace_tight_13": mace_tight_13,
        "mace_dft_ver": mace_dft_ver,
        "goat_time_min": goat_time_min,
        "goat_ws_count": goat_ws_count,
        "goat_dft_ver": goat_dft_ver,
        "goat_labels": goat_labels,
    }


def get_n2o_search_data() -> dict:
    """
    Exact empirical timing and isomer discovery trajectory for HCOOH...N2O (SR4)
    from maceoff24m_search_600/search_run.log and results/CH2O2-N2O/run_log.txt.
    """
    # MACE-OFF24(M): 1,209 starts (9 seeds + 600 trans + 600 cis)
    # Phase 0+1: 0.30 min (monomers)
    # Phase 3 coarse batches (4 x 300 + 1 x 9 = 889s = 14.82 min):
    #   Batch 1 (trans 1..300): +192s -> t = 3.50 min (12 trans families hit, 16 raw minima)
    #   Batch 2 (trans 301..600): +197s -> t = 6.78 min (all 16 trans families hit, 22 raw minima)
    #   Batch 3 (cis 1..300): +216s -> t = 10.38 min (16 trans + 9 cis = 25 families hit, 34 raw minima)
    #   Batch 4 (cis 301..600): +230s -> t = 14.22 min (all 28 families / 26 bound hit, 38 raw minima)
    #   Batch 5 (9 seeds): +54s -> t = 15.12 min (440 coarse basins)
    # Phase 4 tight float64 BFGS refinement (440 basins -> 42 tight minima): +2147s -> t = 50.90 min
    # Phase 6+6b analytic MACE Hessian + 7 saddle displacements + family merge: +84s -> t = 52.30 min
    #   -> 38 verified minima -> 28 conformational families (26 bound) -> 18 r2SCAN-3c verified minima
    mace_time_min = np.array([0.0, 0.30, 3.50, 6.78, 10.38, 14.22, 15.12, 50.90, 52.30])
    mace_raw_38   = np.array([0,   0,    16,   22,   34,    38,    38,    38,    38])
    mace_fam_28   = np.array([0,   0,    12,   16,   25,    28,    28,    28,    28])
    mace_dft_18   = np.array([0,   0,    8,    10,   16,    18,    18,    18,    18])

    # ORCA 6.1.1 GFN2-xTB GOAT (results/CH2O2-N2O/run_log.txt):
    # Phase 1 seed quench (9 seeds): 334 s = 5.57 min
    # Phase 2 GOAT per seed (cumulative wall time & cumulative working-set / DFT basins found):
    #   s0_student (+39.8m -> 45.4m): finds iso-00,01,02,03,04,05,06 (7 working-set; 5 unique DFT)
    #   s1_t_OH-N  (+37.6m -> 83.0m): +0 new (7 working-set; 5 unique DFT)
    #   s2_t_OH-O  (+41.5m -> 124.5m): +iso-07 (8 working-set; 5 unique DFT [07 collapses to 05])
    #   s3_t_CH-O  (+43.9m -> 168.4m): +0 new (8 working-set; 5 unique DFT)
    #   s4_t_stack (+34.4m -> 202.8m): +0 new (8 working-set; 5 unique DFT)
    #   s5_c_OH-N  (+43.0m -> 245.8m): +iso-08,09,10,16 (12 working-set; 9 unique DFT)
    #   s6_c_OH-O  (+49.8m -> 295.6m): +0 new (12 working-set; 9 unique DFT)
    #   s7_c_CH-O  (+81.6m -> 377.2m): +iso-13,18 (14 working-set; 11 unique DFT)
    #   s8_c_stack (+38.6m -> 415.8m): +iso-12,15 (16 working-set; 11 unique DFT [12/13->09, 15 dup])
    # Phase 4 TightOpt+NumFreq (58 candidates + 4 TS follow-ups = +73.5m -> 489.3 min):
    #   21 raw GOAT outputs -> 16 working-set conformers -> 11 unique DFT basins
    goat_time_min = np.array([
        0.0, 5.6, 45.4, 83.0, 124.5, 168.4, 202.8, 245.8, 295.6, 377.2, 415.8, 489.3
    ])
    goat_raw_21   = np.array([0, 0, 7, 7, 8, 8, 8, 14, 14, 17, 21, 21])
    goat_ws_16    = np.array([0, 0, 7, 7, 8, 8, 8, 12, 12, 14, 16, 16])
    goat_dft_11   = np.array([0, 0, 5, 5, 5, 5, 5,  9,  9, 11, 11, 11])

    return {
        "mace_time_min": mace_time_min,
        "mace_raw_38": mace_raw_38,
        "mace_fam_28": mace_fam_28,
        "mace_dft_18": mace_dft_18,
        "goat_time_min": goat_time_min,
        "goat_raw_21": goat_raw_21,
        "goat_ws_16": goat_ws_16,
        "goat_dft_11": goat_dft_11,
    }


def draw_h2s_panels(ax_curve: plt.Axes, ax_bar: plt.Axes, d: dict) -> None:
    """Render Panel (a) discovery curve and Panel (b) metric comparison for HCOOH...H2S."""
    # --- Panel (a): Discovery Curve ---
    ax_curve.axvspan(0.25, 11.67, color="#EBF5FB", alpha=0.75, zorder=0,
                     label="MACE Phase 3: Batched float32 FIRE (1,204 starts)")
    ax_curve.axvspan(11.67, 31.60, color="#EAFAF1", alpha=0.65, zorder=0,
                     label="MACE Phases 4–6: float64 BFGS + Analytic Hessian")

    ax_curve.step(d["mace_time_min"], d["mace_tight_13"], where="post",
                  color=COL_MACE_RAW, linestyle="--", linewidth=1.8,
                  label="MACE-OFF24(M): Tight Minima (13 raw)")
    ax_curve.plot(d["mace_time_min"], d["mace_families"], "-o",
                  color=COL_MACE_FAM, linewidth=2.4, markersize=5.5,
                  label="MACE-OFF24(M): Unique Families (9, 31.6 min)")
    ax_curve.plot(d["mace_time_min"], d["mace_dft_ver"], "-s",
                  color=COL_MACE_DFT, linewidth=2.0, markersize=4.8,
                  label="MACE-OFF24(M) → r²SCAN-3c Minima (8/8 = 100%)")

    ax_curve.plot(d["goat_time_min"], d["goat_ws_count"], "-^",
                  color=COL_GOAT_RAW, linewidth=2.2, markersize=5.5,
                  label="ORCA GFN2-xTB GOAT: Working Set (8, 291.3 min)")
    ax_curve.plot(d["goat_time_min"], d["goat_dft_ver"], ":d",
                  color=COL_GOAT_DFT, linewidth=2.0, markersize=5.0,
                  label="ORCA GFN2-xTB GOAT → DFT Minima (7/8, misses Iso_06)")

    # Key milestone callouts
    ax_curve.annotate(
        "All 9 families hit\nin 8.3 min (Batch 3)",
        xy=(8.32, 9.0), xytext=(22.0, 11.4),
        arrowprops=dict(arrowstyle="->", color=COL_MACE_FAM, lw=1.2),
        fontsize=8.0, fontweight="bold", color=COL_MACE_FAM,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL_MACE_FAM, alpha=0.94)
    )
    ax_curve.annotate(
        "MACE Done (31.6 min)\n1,204 starts → 9 families\n(9.2× faster than GOAT)",
        xy=(31.6, 9.0), xytext=(48.0, 5.6),
        arrowprops=dict(arrowstyle="->", color="#004D7A", lw=1.3),
        fontsize=8.0, fontweight="bold", color="#004D7A",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F0F8FF", ec="#004D7A", alpha=0.95)
    )
    ax_curve.annotate(
        "trans GOAT only (198.2 min)\nTrapped in 3 trans minima\n(0 cis conformers found)",
        xy=(198.2, 3.0), xytext=(105.0, 0.7),
        arrowprops=dict(arrowstyle="->", color=COL_GOAT_RAW, lw=1.2),
        fontsize=7.8, color="#903000",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8F5", ec=COL_GOAT_RAW, alpha=0.94)
    )
    ax_curve.annotate(
        "cis + gauche GOAT\n(291.3 min total)\n7 DFT minima (misses Iso_06)",
        xy=(291.3, 7.0), xytext=(202.0, 4.2),
        arrowprops=dict(arrowstyle="->", color=COL_GOAT_DFT, lw=1.2),
        fontsize=7.8, color="#7A295C",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL_GOAT_DFT, alpha=0.94)
    )

    ax_curve.set_title("(a) HCOOH···H₂S: Isomer Discovery vs. Search Wall-Clock Time")
    ax_curve.set_xlabel("Cumulative Wall-Clock Time (minutes)")
    ax_curve.set_ylabel("Cumulative Unique Isomers Discovered")
    ax_curve.set_xlim(-5, 310)
    ax_curve.set_ylim(-0.4, 16.5)
    ax_curve.grid(True, linestyle=":", alpha=0.45)
    ax_curve.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.9)

    # --- Panel (b): Grouped Horizontal Metric Summary ---
    categories = [
        "Verified DFT Minima\n(r²SCAN-3c, n_imag=0)",
        "Deduplicated Families /\nWorking-Set Conformers",
        "Raw Stationary Points\n(Pre-Family Merge)",
        "Search Speed\n(Conformers / Hour)",
    ]
    mace_vals = [8, 9, 13, 9.0 / (31.6 / 60.0)]      # 17.1 conformers/hr
    goat_vals = [7, 8, 14, 8.0 / (291.3 / 60.0)]     # 1.65 conformers/hr

    y = np.arange(len(categories))
    height = 0.35
    rects1 = ax_bar.barh(y + height/2, mace_vals, height, color=COL_MACE_FAM,
                         edgecolor="black", linewidth=0.7,
                         label="Batched GPU MACE-OFF24(M) (1,204 starts, 31.6 min)")
    rects2 = ax_bar.barh(y - height/2, goat_vals, height, color=COL_GOAT_RAW,
                         edgecolor="black", linewidth=0.7,
                         label="ORCA 6.1.1 GFN2-xTB GOAT (6 seeds, 291.3 min)")

    for r in rects1:
        w = r.get_width()
        txt = f"{w:.1f}/h" if w > 15 else f"{int(round(w))}"
        ax_bar.text(w + 0.35, r.get_y() + r.get_height()/2, txt,
                    va="center", ha="left", fontsize=8.3, fontweight="bold", color=COL_MACE_FAM)
    for r in rects2:
        w = r.get_width()
        txt = f"{w:.2f}/h" if w < 3 else f"{int(round(w))}"
        ax_bar.text(w + 0.35, r.get_y() + r.get_height()/2, txt,
                    va="center", ha="left", fontsize=8.3, fontweight="bold", color=COL_GOAT_RAW)

    # Summary box inside bar chart (placed cleanly on right without touching bars)
    ax_bar.text(
        0.98, 0.04,
        "Sampling & Accuracy Summary (HCOOH···H₂S):\n"
        "• MACE-OFF24(M): 1,204 starts in 31.6 min (9.2× faster)\n"
        "• Discovers MACE-only cis Isomer_06 missed by GOAT\n"
        "• Global min B_e error vs. revDSD: +0.89% (MACE)\n"
        "  vs. +13.77% over-binding in GFN2-xTB GOAT",
        transform=ax_bar.transAxes, ha="right", va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#888888", alpha=0.96)
    )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(categories)
    ax_bar.set_xlabel("Count or Discovery Rate (Isomers / Hour)")
    ax_bar.set_title("(b) HCOOH···H₂S: Completeness & Throughput Metrics")
    ax_bar.set_xlim(0, 31.0)
    ax_bar.set_ylim(-0.55, 4.15)
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_bar.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.9)


def draw_n2o_panels(ax_curve: plt.Axes, ax_bar: plt.Axes, d: dict) -> None:
    """Render Panel (a) discovery curve and Panel (b) metric comparison for HCOOH...N2O."""
    # --- Panel (a): Discovery Curve ---
    ax_curve.axvspan(0.30, 15.12, color="#EBF5FB", alpha=0.75, zorder=0,
                     label="MACE Phase 3: Batched float32 FIRE (1,209 starts)")
    ax_curve.axvspan(15.12, 52.30, color="#EAFAF1", alpha=0.65, zorder=0,
                     label="MACE Phases 4–6: float64 BFGS + Analytic Hessian")

    ax_curve.step(d["mace_time_min"], d["mace_raw_38"], where="post",
                  color=COL_MACE_RAW, linestyle="--", linewidth=1.8,
                  label="MACE-OFF24(M): Verified Minima (38 raw)")
    ax_curve.plot(d["mace_time_min"], d["mace_fam_28"], "-o",
                  color=COL_MACE_FAM, linewidth=2.4, markersize=5.5,
                  label="MACE-OFF24(M): Unique Families (28 [26 bound], 52.3 min)")
    ax_curve.plot(d["mace_time_min"], d["mace_dft_18"], "-s",
                  color=COL_MACE_DFT, linewidth=2.0, markersize=4.8,
                  label="MACE-OFF24(M) → r²SCAN-3c Minima (18 verified)")

    ax_curve.plot(d["goat_time_min"], d["goat_raw_21"], "--^",
                  color="#E69F00", linewidth=1.6, markersize=4.5,
                  label="ORCA GFN2-xTB GOAT: Raw Ensemble (21, incl. 1 saddle)")
    ax_curve.plot(d["goat_time_min"], d["goat_ws_16"], "-^",
                  color=COL_GOAT_RAW, linewidth=2.2, markersize=5.5,
                  label="ORCA GFN2-xTB GOAT: Working Set (16, 489.3 min)")
    ax_curve.plot(d["goat_time_min"], d["goat_dft_11"], ":d",
                  color=COL_GOAT_DFT, linewidth=2.0, markersize=5.0,
                  label="ORCA GFN2-xTB GOAT → DFT Minima (11 unique basins)")

    # Key milestone callouts
    ax_curve.annotate(
        "All 28 MACE families hit\nin 14.2 min (Batch 4)",
        xy=(14.22, 28.0), xytext=(38.0, 34.0),
        arrowprops=dict(arrowstyle="->", color=COL_MACE_FAM, lw=1.2),
        fontsize=8.0, fontweight="bold", color=COL_MACE_FAM,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL_MACE_FAM, alpha=0.94)
    )
    ax_curve.annotate(
        "MACE Done (52.3 min)\n28 families / 18 DFT minima\n(9.4× faster than GOAT;\ncorrectly ranks 1t global min)",
        xy=(52.3, 28.0), xytext=(72.0, 17.5),
        arrowprops=dict(arrowstyle="->", color="#004D7A", lw=1.3),
        fontsize=7.9, fontweight="bold", color="#004D7A",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F0F8FF", ec="#004D7A", alpha=0.95)
    )
    ax_curve.annotate(
        "trans GOAT seeds s0–s4 (202.8 min)\nPlateaus at 8 trans candidates\n(5 unique DFT basins)",
        xy=(202.8, 8.0), xytext=(110.0, 1.2),
        arrowprops=dict(arrowstyle="->", color=COL_GOAT_RAW, lw=1.2),
        fontsize=7.8, color="#903000",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8F5", ec=COL_GOAT_RAW, alpha=0.94)
    )
    ax_curve.annotate(
        "9-Seed GOAT Done (489.3 min = 8.16 h)\n16 working-set → 11 DFT basins\n(GFN2 inverts 1t/2t stability)",
        xy=(489.3, 16.0), xytext=(295.0, 5.2),
        arrowprops=dict(arrowstyle="->", color=COL_GOAT_RAW, lw=1.2),
        fontsize=7.8, color="#7A295C",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL_GOAT_DFT, alpha=0.94)
    )

    ax_curve.set_title("(a) HCOOH···N₂O: Isomer Discovery vs. Search Wall-Clock Time")
    ax_curve.set_xlabel("Cumulative Wall-Clock Time (minutes)")
    ax_curve.set_ylabel("Cumulative Unique Isomers Discovered")
    ax_curve.set_xlim(-10, 515)
    ax_curve.set_ylim(-1.0, 48.0)
    ax_curve.grid(True, linestyle=":", alpha=0.45)
    ax_curve.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.8)

    # --- Panel (b): Grouped Horizontal Metric Summary ---
    categories = [
        "Verified DFT Minima\n(r²SCAN-3c, n_imag=0)",
        "Deduplicated Families /\nWorking-Set Conformers",
        "Raw Stationary Points\n(Pre-Family Merge)",
        "Search Speed\n(Conformers / Hour)",
    ]
    mace_vals = [18, 28, 38, 28.0 / (52.3 / 60.0)]    # 32.1 conformers/hr
    goat_vals = [11, 16, 21, 16.0 / (489.3 / 60.0)]   # 1.96 conformers/hr

    y = np.arange(len(categories))
    height = 0.35
    rects1 = ax_bar.barh(y + height/2, mace_vals, height, color=COL_MACE_FAM,
                         edgecolor="black", linewidth=0.7,
                         label="Batched GPU MACE-OFF24(M) (1,209 starts, 52.3 min)")
    rects2 = ax_bar.barh(y - height/2, goat_vals, height, color=COL_GOAT_RAW,
                         edgecolor="black", linewidth=0.7,
                         label="ORCA 6.1.1 GFN2-xTB GOAT (9 seeds, 489.3 min)")

    for r in rects1:
        w = r.get_width()
        txt = f"{w:.1f}/h" if (r.get_y() > 2.5) else f"{int(round(w))}"
        ax_bar.text(w + 0.6, r.get_y() + r.get_height()/2, txt,
                    va="center", ha="left", fontsize=8.3, fontweight="bold", color=COL_MACE_FAM)
    for r in rects2:
        w = r.get_width()
        txt = f"{w:.2f}/h" if (r.get_y() > 2.5) else f"{int(round(w))}"
        ax_bar.text(w + 0.6, r.get_y() + r.get_height()/2, txt,
                    va="center", ha="left", fontsize=8.3, fontweight="bold", color=COL_GOAT_RAW)

    ax_bar.text(
        0.98, 0.04,
        "Sampling & Accuracy Summary (HCOOH···N₂O):\n"
        "• MACE-OFF24(M): 1,209 starts in 52.3 min (9.4× faster)\n"
        "• +64% more verified r²SCAN-3c minima (18 vs. 11)\n"
        "• Resolves GFN2-xTB 1t/2t global-minimum inversion\n"
        "  (places cyclic O-bonded 1t at 0.00 kcal/mol)",
        transform=ax_bar.transAxes, ha="right", va="bottom",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#888888", alpha=0.96)
    )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(categories)
    ax_bar.set_xlabel("Count or Discovery Rate (Isomers / Hour)")
    ax_bar.set_title("(b) HCOOH···N₂O: Completeness & Throughput Metrics")
    ax_bar.set_xlim(0, 66.0)
    ax_bar.set_ylim(-0.55, 4.15)
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax_bar.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.9)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "figures")
    args = ap.parse_args()

    d_h2s = get_h2s_search_data()
    d_n2o = get_n2o_search_data()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Individual Figure for HCOOH...H2S
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_h2s_panels(ax1a, ax1b, d_h2s)
    fig1.tight_layout(pad=1.4)
    target1 = args.out_dir / "F_mace_vs_goat_search_h2s"
    fig1.savefig(str(target1) + ".pdf", bbox_inches="tight")
    fig1.savefig(str(target1) + ".png", dpi=300, bbox_inches="tight")
    print(f"[WROTE] {target1}.pdf / .png")
    plt.close(fig1)

    # 2. Individual Figure for HCOOH...N2O
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_n2o_panels(ax2a, ax2b, d_n2o)
    fig2.tight_layout(pad=1.4)
    target2 = args.out_dir / "F_mace_vs_goat_search_n2o"
    fig2.savefig(str(target2) + ".pdf", bbox_inches="tight")
    fig2.savefig(str(target2) + ".png", dpi=300, bbox_inches="tight")
    print(f"[WROTE] {target2}.pdf / .png")
    plt.close(fig2)

    # 3. Combined 2x2 Figure for Cross-Complex Method Comparison
    fig3, axes = plt.subplots(2, 2, figsize=(13.6, 9.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_h2s_panels(axes[0, 0], axes[0, 1], d_h2s)
    draw_n2o_panels(axes[1, 0], axes[1, 1], d_n2o)
    axes[1, 0].set_title("(c) HCOOH···N₂O: Isomer Discovery vs. Search Wall-Clock Time")
    axes[1, 1].set_title("(d) HCOOH···N₂O: Completeness & Throughput Metrics")
    fig3.tight_layout(pad=1.5)
    comb_target = args.out_dir / "F_mace_vs_goat_search_combined"
    fig3.savefig(str(comb_target) + ".pdf", bbox_inches="tight")
    fig3.savefig(str(comb_target) + ".png", dpi=300, bbox_inches="tight")
    print(f"[WROTE] {comb_target}.pdf / .png")
    plt.close(fig3)


if __name__ == "__main__":
    main()
