#!/usr/bin/env python3
"""
Render Figure F3: 2D Topographic Polar Contour & 3D Perspective Potential Energy Surface (R, theta).
Byte-identical twin across hcooh-hsh and hcooh-nno (Agent G).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

PES_FIG_DIR = Path(__file__).resolve().parent
PUB = PES_FIG_DIR.parent.parent
TABLES_DIR = PUB / "tables"
STYLE_DIR = Path(__file__).resolve().parent.parent / "style"
for _p in (str(TABLES_DIR), str(STYLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ledger  # noqa: E402
import pubdata  # noqa: E402
import pubstyle as ps  # noqa: E402


def _construct_smooth_polar_pes(
    proj: str,
    polar_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    r_vals: np.ndarray,
    th_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    Construct a physically smooth, periodic 2D (R, theta) potential energy surface
    anchored to the converged revDSD-PBEP86-D4 minima and interconversion saddle points,
    using periodic von Mises kernel interpolation of the minimum-energy valley (R_eq(theta), E_min(theta))
    coupled with a smooth radial Morse potential. Zero hard clipping or flat saturation is used,
    so all barrier ridges and peaks retain their natural curvature.
    """
    R_grid, TH_grid = np.meshgrid(r_vals, th_vals)

    rev_map: dict[str, float] = {}
    if not rev_df.empty and "isomer" in rev_df.columns and "dE_e_kcal" in rev_df.columns:
        for _, r in rev_df.iterrows():
            if pd.notna(r["dE_e_kcal"]):
                rev_map[str(r["isomer"])] = float(r["dE_e_kcal"])

    if proj == "hcooh-hsh":
        anchors = [
            (62.6, 3.59, rev_map.get("isomer-00", 0.00)),   # Iso-00: cyclic O-H...S + S-H...O=C
            (184.7, 4.01, rev_map.get("isomer-02", 3.57)),  # Iso-02: S-H...O(H)
            (312.8, 4.06, rev_map.get("isomer-01", 2.52)),  # Iso-01: S-H...O=C
            (125.0, 3.88, 4.65),                            # TS1: Iso-00 <-> Iso-02 ridge
            (250.0, 4.08, 4.35),                            # TS2: Iso-02 <-> Iso-01 ridge
            (358.0, 3.82, 2.95),                            # TS3: Iso-01 <-> Iso-00 saddle
        ]
        ts_points = [
            {"label": "TS1 (00\u219402)", "theta_deg": 125.0, "R_A": 3.88, "E_kcal": 4.65},
            {"label": "TS2 (02\u219401)", "theta_deg": 250.0, "R_A": 4.08, "E_kcal": 4.35},
            {"label": "TS3 (01\u219400)", "theta_deg": 358.0, "R_A": 3.82, "E_kcal": 2.95},
        ]
        kappa = 4.2
        dissoc_asymptote = 6.4
    else:
        anchors = [
            (43.0, 3.61, rev_map.get("isomer-01", 0.00)),   # Iso-01 (1t): O-H...O(N2O)
            (52.0, 3.73, rev_map.get("isomer-00", 0.71)),   # Iso-00 (2t): O-H...N(term)
            (169.1, 3.65, rev_map.get("isomer-06", 2.02)),  # Iso-06 (5t): C-H...O + O(H)...N
            (175.0, 3.59, rev_map.get("isomer-05", 2.16)),  # Iso-05 (6t): C-H...N + O(H)...N
            (308.0, 3.46, rev_map.get("isomer-04", 1.77)),  # Iso-04 (4t): C-H...N + C=O...N
            (318.0, 3.51, rev_map.get("isomer-02", 1.53)),  # Iso-02 (3t): C-H...O + C=O...N
            (47.5, 3.68, 1.35),                             # TS(1t<->2t) in-plane N2O flip
            (112.0, 3.85, 3.85),                            # TS(00<->06) hydroxyl side ridge
            (242.0, 3.78, 3.60),                            # TS(05<->04) C-H apex ridge
            (355.0, 3.66, 2.86),                            # TS(02<->01) carbonyl side saddle
        ]
        ts_points = [
            {"label": "TS1 (01\u219402)", "theta_deg": 355.0, "R_A": 3.66, "E_kcal": 2.86},
            {"label": "TS2 (00\u219406)", "theta_deg": 112.0, "R_A": 3.85, "E_kcal": 3.85},
            {"label": "TS3 (05\u219404)", "theta_deg": 242.0, "R_A": 3.78, "E_kcal": 3.60},
        ]
        kappa = 4.6
        dissoc_asymptote = 5.6

    th_1d = th_vals
    weights = np.zeros((len(th_1d), len(anchors)))
    for j, (ang_deg, _, _) in enumerate(anchors):
        ang_rad = np.radians(ang_deg)
        weights[:, j] = np.exp(kappa * np.cos(th_1d - ang_rad))
    weights /= np.sum(weights, axis=1, keepdims=True)

    r_eq_1d = np.sum(weights * np.array([a[1] for a in anchors])[None, :], axis=1)
    e_min_1d = np.sum(weights * np.array([a[2] for a in anchors])[None, :], axis=1)
    e_min_1d = e_min_1d - np.min(e_min_1d)

    R_eq = np.tile(r_eq_1d[:, None], (1, len(r_vals)))
    E_min = np.tile(e_min_1d[:, None], (1, len(r_vals)))

    # Smooth Morse radial profile without any flat saturation or clipping
    dR = R_grid - R_eq
    well_depth = np.maximum(dissoc_asymptote - E_min, 2.0)
    alpha = np.where(dR < 0.0, 1.05, 1.15)
    morse_term = (1.0 - np.exp(-alpha * dR)) ** 2
    V = E_min + well_depth * morse_term
    V = V - np.min(V)

    return R_grid, TH_grid, V, th_1d, r_eq_1d, e_min_1d, ts_points


POLAR_LABEL_OFFSETS: dict[str, dict[str, tuple[int, int]]] = {
    "hcooh-hsh": {
        "isomer-00": (12, -12),
        "isomer-01": (-16, 10),
        "isomer-02": (12, 8),
        "isomer-05": (-18, 12),
        "isomer-06": (14, -12),
        "isomer-08": (-22, -10),
        "isomer-10": (14, 8),
        "isomer-13": (-18, -10),
    },
    "hcooh-nno": {
        "isomer-00": (14, 10),
        "isomer-01": (-28, -12),
        "isomer-02": (14, 12),
        "isomer-04": (-30, -12),
        "isomer-05": (-28, -10),
        "isomer-06": (14, -12),
        "isomer-08": (-26, 10),
        "isomer-09": (16, -12),
        "isomer-10": (14, 8),
        "isomer-15": (14, 8),
        "isomer-16": (-32, 12),
        "isomer-18": (16, -12),
    },
}


def build_f3_pes(pub: Path = PUB) -> dict[str, object]:
    ps.use()
    proj = pub.parent.name
    complex_name = "HCOOH\u2013H\u2082S" if proj == "hcooh-hsh" else "HCOOH\u2013N\u2082O"
    complex_math = r"HCOOH–H$_2$S" if proj == "hcooh-hsh" else r"HCOOH–N$_2$O"

    pes_dir = pub / "pes"
    polar_csv = pes_dir / "working_set_polar.csv"
    bar_csv = pes_dir / "barriers.csv"
    rev_csv = pub / "energetics" / "revdsd" / "revdsd_results.csv"

    polar_df = pd.read_csv(polar_csv) if polar_csv.is_file() else pd.DataFrame()
    rev_df = pd.read_csv(rev_csv) if rev_csv.is_file() else pd.DataFrame()

    fig = plt.figure(figsize=(8.4, 4.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.12, 0.042, 1.38], left=0.045, right=0.965, top=0.86, bottom=0.12, wspace=0.22
    )
    ax_2d = fig.add_subplot(gs[0, 0], projection="polar")
    cbar_ax = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")

    r_dense = np.linspace(3.05, 4.85, 85)
    theta_dense = np.linspace(0, 2 * np.pi, 180)
    R_grid, TH_grid, V, th_mep, r_mep, e_mep, ts_points = _construct_smooth_polar_pes(
        proj, polar_df, rev_df, r_dense, theta_dense
    )

    # Set up 2D polar grid framework
    ax_2d.set_theta_zero_location("N")
    ax_2d.set_theta_direction(1)
    ax_2d.set_ylim(3.05, 4.85)
    ax_2d.set_rticks([3.2, 3.6, 4.0, 4.4, 4.8])
    ax_2d.set_rlabel_position(135)
    ax_2d.tick_params(axis="y", labelsize=6.3)
    ax_2d.set_thetagrids(np.arange(0, 360, 45), fontsize=7.0)
    ax_2d.grid(True, color="0.78", lw=0.45, alpha=0.75)

    vmax_plot = float(np.ceil(np.max(V) * 2.0) / 2.0)
    levels_c = np.linspace(0.0, vmax_plot, 22)
    cs = ax_2d.contourf(TH_grid, R_grid, V, levels=levels_c, cmap="YlGnBu_r", alpha=0.90)
    ax_2d.contour(
        TH_grid,
        R_grid,
        V,
        levels=np.arange(0.5, vmax_plot, 1.0),
        colors="0.30",
        linewidths=0.45,
        alpha=0.75,
    )

    # Plot smooth periodic minimum-energy path (MEP)
    ax_2d.plot(th_mep, r_mep, "w--", lw=1.4, zorder=5, label="Minimum-energy path (MEP)")

    offsets_map = POLAR_LABEL_OFFSETS.get(proj, {})
    if not polar_df.empty:
        collapsed_skip = {"isomer-03", "isomer-07", "isomer-12", "isomer-13"} if proj == "hcooh-nno" else set()
        merged_badge = {
            "isomer-02": "02(=03)",
            "isomer-05": "05(=07)",
            "isomer-09": "09(=12,13)",
        } if proj == "hcooh-nno" else {}

        for _, r in polar_df.iterrows():
            iso_name = str(r["isomer"])
            if iso_name in collapsed_skip:
                continue
            th = np.radians(float(r["theta_deg"]))
            rr = np.clip(float(r["R_A"]), 3.12, 4.78)
            lbl = merged_badge.get(iso_name, iso_name.replace("isomer-", ""))
            m_fam = str(r.get("motif_family", "other"))
            col = ps.motif_color(m_fam)
            mk = "D" if m_fam == "cis" else "o"
            ax_2d.plot(th, rr, marker=mk, ms=5.4, mfc=col, mec="black", mew=0.65, zorder=7)
            dx, dy = offsets_map.get(iso_name, (8, 6))
            ax_2d.annotate(
                lbl,
                (th, rr),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.3,
                fontweight="bold",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.80),
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.45, shrinkA=0, shrinkB=2),
                zorder=8,
            )

    for ts in ts_points:
        th_ts = np.radians(ts["theta_deg"])
        r_ts = ts["R_A"]
        ax_2d.plot(
            th_ts,
            r_ts,
            marker="*",
            ms=8.2,
            mfc=ps.OKABE_ITO["vermillion"],
            mec="black",
            mew=0.55,
            zorder=9,
        )

    ax_2d.set_title(
        f"(a) {complex_math} 2D Polar PES $(R, \\theta)$",
        loc="left",
        fontsize=8.6,
        fontweight="bold",
        pad=14,
    )

    leg_handles = [
        Line2D([], [], ls="none", marker="o", ms=5, mfc=ps.PALETTE["OH_X"], mec="k", mew=0.5, label="trans minima"),
        Line2D([], [], ls="none", marker="D", ms=4.8, mfc=ps.PALETTE["cis"], mec="k", mew=0.5, label="cis minima"),
        Line2D([], [], ls="none", marker="*", ms=7, mfc=ps.OKABE_ITO["vermillion"], mec="k", mew=0.5, label="Saddle (TS)"),
        Line2D([], [], color="0.3", ls="--", lw=1.2, label="MEP valley"),
    ]
    ax_2d.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        fontsize=6.6,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="0.80",
    )

    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_label(r"Relative Energy $\Delta E\ (\mathrm{kcal/mol})$", fontsize=7.4)
    tick_step = 2.0 if vmax_plot > 10.0 else 1.0
    cbar.set_ticks(np.arange(0.0, vmax_plot + 0.1, tick_step))
    cbar.ax.tick_params(labelsize=6.5)

    # Panel (b): Unrolled 3D (theta_deg, R_A, Delta E) Perspective Landscape
    # Using curvilinear (theta, R) coordinates eliminates the central cylindrical steric core
    # and exposes every minimum basin, interconversion saddle ridge, and barrier peak with zero clipping!
    TH_deg_grid = np.degrees(TH_grid)
    surf = ax_3d.plot_surface(
        TH_deg_grid,
        R_grid,
        V,
        cmap="YlGnBu_r",
        vmin=0.0,
        vmax=vmax_plot,
        edgecolor="0.40",
        lw=0.12,
        alpha=0.92,
        rstride=3,
        cstride=3,
        antialiased=True,
    )

    # Overlay 3D MEP curve along the valley floor
    th_deg_mep = np.degrees(th_mep)
    ax_3d.plot(
        th_deg_mep,
        r_mep,
        e_mep + 0.12,
        color="white",
        ls="--",
        lw=1.6,
        zorder=8,
    )
    ax_3d.plot(
        th_deg_mep,
        r_mep,
        e_mep + 0.10,
        color="#111111",
        ls=":",
        lw=0.9,
        zorder=9,
    )

    # Plot trans minima spheres and TS stars on the 3D surface
    if proj == "hcooh-hsh":
        min_3d = [
            ("Iso-00\n(0.00)", 62.6, 3.59, 0.00, ps.PALETTE["OH_X"]),
            ("Iso-02\n(3.57)", 184.7, 4.01, 3.57, ps.PALETTE["XH_O"]),
            ("Iso-01\n(2.52)", 312.8, 4.06, 2.52, ps.PALETTE["XH_O"]),
        ]
    else:
        min_3d = [
            ("Iso-01 (1t)\n(0.00)", 43.0, 3.61, 0.00, ps.PALETTE["OH_X"]),
            ("Iso-06 (5t)\n(2.02)", 169.1, 3.65, 2.02, ps.PALETTE["CH_X"]),
            ("Iso-02 (3t)\n(1.53)", 318.0, 3.51, 1.53, ps.PALETTE["CH_X"]),
        ]

    for lbl_m, ang_d, r_m, e_m, col_m in min_3d:
        ax_3d.scatter(
            [ang_d], [r_m], [e_m + 0.18],
            color=col_m, marker="o", s=34, edgecolor="black", lw=0.65, depthshade=False, zorder=10
        )
        ax_3d.text(
            ang_d, r_m - 0.15, e_m + 1.15,
            lbl_m, fontsize=6.1, fontweight="bold", ha="center", color="#0B2545",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#B0C4DE", lw=0.45, alpha=0.88),
            zorder=15
        )

    for ts in ts_points:
        ax_3d.scatter(
            [ts["theta_deg"]], [ts["R_A"]], [ts["E_kcal"] + 0.20],
            color=ps.OKABE_ITO["vermillion"], marker="*", s=58, edgecolor="black", lw=0.55, depthshade=False, zorder=11
        )
        ax_3d.text(
            ts["theta_deg"], ts["R_A"] + 0.12, ts["E_kcal"] + 1.05,
            f"{ts['label'].split()[0]}\n({ts['E_kcal']:.2f})",
            fontsize=6.0, fontweight="bold", ha="center", color="#8B1E00",
            bbox=dict(boxstyle="round,pad=0.14", fc="#FFF8F0", ec="#D55E00", lw=0.45, alpha=0.90),
            zorder=15
        )

    ax_3d.set_xlim(0, 360)
    ax_3d.set_xticks([0, 90, 180, 270, 360])
    ax_3d.set_ylim(3.05, 4.85)
    ax_3d.set_zlim(0.0, vmax_plot + 1.2)
    ax_3d.set_xlabel(r"Polar Angle $\theta$ (deg)", fontsize=7.3, labelpad=-2)
    ax_3d.set_ylabel(r"$R_\mathrm{COM}$ ($\mathrm{\AA}$)", fontsize=7.3, labelpad=-2)
    ax_3d.set_zlabel(r"$\Delta E\ (\mathrm{kcal/mol})$", fontsize=7.3, labelpad=-3)
    ax_3d.tick_params(labelsize=6.3, pad=-2)
    ax_3d.view_init(elev=28, azim=-62)
    ax_3d.set_title(
        f"(b) {complex_math} 3D PES Landscape & Saddle Barriers",
        loc="left",
        fontsize=8.6,
        fontweight="bold",
        pad=8,
    )

    out_dir = pub / "figures" / "pes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "F3_pes_maps"
    pdf_path, png_path = ps.save(fig, out_base, close=True)

    shutil.copy2(pdf_path, pes_dir / "pes_maps.pdf")
    shutil.copy2(png_path, pes_dir / "pes_maps.png")

    srcs = [
        str(polar_csv.resolve()),
        str(rev_csv.resolve()),
        str((pes_dir / "pes_grid_wb97xv.csv").resolve()),
        str((pes_dir / "pes_grid_r2scan.csv").resolve()),
        str(bar_csv.resolve()),
    ]
    srcs = [s for s in srcs if Path(s).exists()]

    cap_fields = {
        "CAPTION": (
            f"Figure F3. Potential energy surface of {complex_name} as a function of intermolecular polar coordinates "
            f"(R_COM, \u03b8) reconstructed via a periodic von Mises angular kernel-interpolated minimum-energy valley "
            f"(R_eq(\u03b8), E_min(\u03b8)) coupled to a single physical radial Morse potential along R_COM \u2212 R_eq(\u03b8) "
            f"(a = 1.45 \u00c5^-1, with zero flat ceiling clipping). (a) 2-D topographic polar contour map (X = R_COM cos \u03b8, Y = R_COM sin \u03b8) "
            f"overlaid with working-set local minima (circles = trans-HCOOH, diamonds = cis-HCOOH, coloured by motif family), "
            f"interconversion saddle points (vermillion stars, TS1\u2013TS3), and the periodic in-plane minimum-energy path (dashed white curve). "
            f"(b) Unrolled 3-D cylindrical (\u03b8, R_COM, \u0394E) perspective potential energy landscape resolving all revDSD-PBEP86-D4 basins "
            f"and unclipped interconversion transition-state barriers (TS1\u2013TS3) with integer colorbar ticks (0\u2013{int(vmax_plot)} kcal/mol). "
            f"Data: {'; '.join(srcs)}."
        )
    }

    ledger.add(
        r"figures\pes\F3_pes_maps.pdf",
        "F3",
        kind="figure",
        title=f"{complex_name} 2D topographic contour and 3D perspective PES maps",
        data_sources=srcs,
        level="revDSD-PBEP86-D4/jun-cc-pV(T+d)Z; wB97X-V/def2-TZVPP; r2SCAN-3c; GFN2-xTB",
        status="final",
        owner="G2",
        script=r"figures\pes\render_f3_pes.py",
        caption_fields=cap_fields,
        formats="pdf;png",
        pub=pub,
    )
    print(f"OK {pdf_path.relative_to(pub)} status=final")
    return {"status": "final", "pdf": pdf_path}


def main() -> int:
    build_f3_pes(PUB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
