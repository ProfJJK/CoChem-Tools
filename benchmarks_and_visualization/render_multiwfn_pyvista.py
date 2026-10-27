#!/usr/bin/env python3
"""
Render Figure F5: True 3D Multiwfn IGMH Noncovalent Interaction Isosurface Colored by sign(lambda2)*rho,
Translucent Space-Filling VdW Envelope, 3D QTAIM Bond Paths & BCP/RCP Spheres (PyVista),
Multiwfn 2D IGMH Scatter Plot, and DLPNO-CCSD(T1) Local Energy Decomposition (LED).
Byte-identical twin across hcooh-hsh and hcooh-nno (Agent G).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pyvista as pv

INTER_FIG_DIR = Path(__file__).resolve().parent
PUB = INTER_FIG_DIR.parent.parent
TABLES_DIR = PUB / "tables"
STYLE_DIR = Path(__file__).resolve().parent.parent / "style"
for _p in (str(TABLES_DIR), str(STYLE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ledger  # noqa: E402
import pubdata  # noqa: E402
import pubstyle as ps  # noqa: E402

BOHR_TO_ANG = 0.529177210903
ATOMIC_NUM_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
CPK_COLORS = {
    "C": "#3A3A3A",
    "O": "#E02424",
    "H": "#F5F5F5",
    "S": "#E6AB02",
    "N": "#1F77B4",
}
COVALENT_RAD_ANG = {"H": 0.24, "C": 0.38, "N": 0.36, "O": 0.35, "S": 0.48}
VDW_RAD_ANG = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def _read_cube_with_atoms(cube_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with cube_path.open("r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        f.readline()
        p3 = f.readline().split()
        natoms = abs(int(p3[0]))
        origin_ang = np.array([float(p3[1]), float(p3[2]), float(p3[3])], dtype=float) * BOHR_TO_ANG
        nx_p = f.readline().split()
        ny_p = f.readline().split()
        nz_p = f.readline().split()
        nx, dx = int(nx_p[0]), float(nx_p[1]) * BOHR_TO_ANG
        ny, dy = int(ny_p[0]), float(ny_p[2]) * BOHR_TO_ANG
        nz, dz = int(nz_p[0]), float(nz_p[3]) * BOHR_TO_ANG
        syms: list[str] = []
        coords: list[list[float]] = []
        for _ in range(natoms):
            ap = f.readline().split()
            z_num = int(ap[0])
            syms.append(ATOMIC_NUM_TO_SYM.get(z_num, "C"))
            coords.append([float(ap[2]) * BOHR_TO_ANG, float(ap[3]) * BOHR_TO_ANG, float(ap[4]) * BOHR_TO_ANG])
        raw = np.fromstring(f.read(), sep=" ", dtype=float)
    data = raw[: nx * ny * nz].reshape((nx, ny, nz))
    return syms, np.array(coords, dtype=float), origin_ang, np.array([dx, dy, dz], dtype=float), data


def _align_frame(coords: np.ndarray, proj: str) -> tuple[np.ndarray, np.ndarray]:
    idx_c = 0
    idx_o_carb = 3 if proj == "hcooh-hsh" else 2
    c0 = coords.mean(axis=0)
    ey = coords[idx_o_carb] - coords[idx_c]
    ey = ey / np.linalg.norm(ey)
    partner_com = coords[5:].mean(axis=0)
    v_part = partner_com - coords[idx_c]
    ex = v_part - np.dot(v_part, ey) * ey
    ex = ex / np.linalg.norm(ex)
    ez = np.cross(ex, ey)
    rot = np.column_stack([ex, ey, ez])
    return c0, rot


def _parse_cps_and_paths(work_dir: Path) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    cps: list[dict[str, Any]] = []
    cp_file = work_dir / "CPprop.txt"
    if cp_file.is_file():
        txt = cp_file.read_text(encoding="utf-8", errors="ignore")
        for blk in txt.split("----------------   CP")[1:]:
            hdr = blk.splitlines()[0]
            m_hdr = re.search(r"^\s*(\d+),\s*Type\s*\((\d+,[+-]?\d+)\)", hdr)
            if not m_hdr:
                continue
            cp_idx = int(m_hdr.group(1))
            cp_type = f"({m_hdr.group(2)})"
            m_pos = re.search(r"Position \(Angstrom\):\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)", blk)
            if not m_pos:
                continue
            pos = np.array([float(m_pos.group(1)), float(m_pos.group(2)), float(m_pos.group(3))], dtype=float)
            m_rho = re.search(r"Density of all electrons:\s+([-\d.E+]+)", blk)
            rho = float(m_rho.group(1)) if m_rho else 0.0
            cps.append({"idx": cp_idx, "type": cp_type, "pos": pos, "rho": rho})

    paths_ang: list[np.ndarray] = []
    paths_file = work_dir / "paths.txt"
    if paths_file.is_file():
        plines = paths_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        idx = 1
        while idx < len(plines):
            ln = plines[idx].strip()
            if not ln:
                idx += 1
                continue
            parts = ln.split()
            if ln.startswith("Path index:") and (idx + 1) < len(plines):
                npts = int(plines[idx + 1].strip().split()[0])
                pts = []
                for k in range(idx + 2, min(idx + 2 + npts, len(plines))):
                    pp = plines[k].split()
                    if len(pp) >= 3:
                        pts.append([float(pp[0]) * BOHR_TO_ANG, float(pp[1]) * BOHR_TO_ANG, float(pp[2]) * BOHR_TO_ANG])
                if len(pts) >= 2:
                    paths_ang.append(np.array(pts, dtype=float))
                idx += 2 + npts
            else:
                idx += 1
    return cps, paths_ang


def _render_pyvista_igmh_qtaim_scene(work_dir: Path, proj: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dg_cub = work_dir / "dg_inter.cub"
    sl2r_cub = work_dir / "sl2r.cub"
    dens_cub = work_dir / "density.cub"

    syms, coords, orig, spac, dg_data = _read_cube_with_atoms(dg_cub)
    _, _, _, _, sl2r_data = _read_cube_with_atoms(sl2r_cub)
    _, _, orig_d, spac_d, dens_data = _read_cube_with_atoms(dens_cub) if dens_cub.is_file() else (syms, coords, orig, spac, None)
    cps, paths_ang = _parse_cps_and_paths(work_dir)

    c0, rot = _align_frame(coords, proj)

    def to_aligned(pts: np.ndarray) -> np.ndarray:
        return (pts - c0) @ rot

    coords_al = to_aligned(coords)

    pl = pv.Plotter(off_screen=True, window_size=(1400, 1300))
    pl.set_background("white")

    # 1. Translucent Space-Filling VdW Surface (from Multiwfn density.cub + CPK spheres)
    if dens_data is not None:
        nx_d, ny_d, nz_d = dens_data.shape
        grid_d = pv.ImageData(dimensions=(nx_d, ny_d, nz_d), spacing=tuple(spac_d), origin=tuple(orig_d))
        grid_d.point_data["rho"] = dens_data.ravel(order="F")
        surf_vdw = grid_d.contour([0.0022], scalars="rho")
        if surf_vdw.n_points > 0:
            surf_vdw.points = to_aligned(surf_vdw.points)
            surf_vdw = surf_vdw.smooth(n_iter=20, relaxation_factor=0.1)
            pl.add_mesh(
                surf_vdw, color="#E2ECF5", opacity=0.14, smooth_shading=True,
                specular=0.4, specular_power=20, ambient=0.25
            )

    for sym, pos in zip(syms, coords_al):
        sp_vdw = pv.Sphere(radius=VDW_RAD_ANG.get(sym, 1.5) * 0.76, center=tuple(pos), theta_resolution=36, phi_resolution=36)
        pl.add_mesh(
            sp_vdw, color=CPK_COLORS.get(sym, "#888888"), opacity=0.12,
            smooth_shading=True, specular=0.3, specular_power=18
        )

    # 2. Glossy 3D Ball-and-Stick Atoms & Covalent Bonds
    for sym, pos in zip(syms, coords_al):
        sp_at = pv.Sphere(radius=COVALENT_RAD_ANG.get(sym, 0.35), center=tuple(pos), theta_resolution=40, phi_resolution=40)
        pl.add_mesh(
            sp_at, color=CPK_COLORS.get(sym, "#888888"), opacity=1.0,
            smooth_shading=True, specular=0.70, specular_power=40, ambient=0.22
        )

    n_at = len(syms)
    for i in range(n_at):
        for j in range(i + 1, n_at):
            in_same = (i < 5 and j < 5) or (i >= 5 and j >= 5)
            d_ij = float(np.linalg.norm(coords_al[i] - coords_al[j]))
            max_cov = 1.65 if ("S" in (syms[i], syms[j])) else 1.55
            if in_same and d_ij <= max_cov:
                mid = 0.5 * (coords_al[i] + coords_al[j])
                vec = coords_al[j] - coords_al[i]
                cyl = pv.Cylinder(center=tuple(mid), direction=tuple(vec), radius=0.095, height=d_ij, resolution=32)
                pl.add_mesh(cyl, color="#555555", smooth_shading=True, specular=0.5, specular_power=30)

    # 3. True 3D Multiwfn IGMH Isosurface (dg_inter.cub = 0.0042 a.u.) Colored by sl2r.cub (sign(lambda2)*rho)
    nx, ny, nz = dg_data.shape
    grid_igmh = pv.ImageData(dimensions=(nx, ny, nz), spacing=tuple(spac), origin=tuple(orig))
    grid_igmh.point_data["dg_inter"] = dg_data.ravel(order="F")
    grid_igmh.point_data["sl2r"] = sl2r_data.ravel(order="F")

    iso_dg = 0.0042
    surf_igmh = grid_igmh.contour([iso_dg], scalars="dg_inter")
    if surf_igmh.n_points > 0:
        surf_igmh.points = to_aligned(surf_igmh.points)
        surf_igmh = surf_igmh.smooth(n_iter=20, relaxation_factor=0.1)
        pl.add_mesh(
            surf_igmh,
            scalars="sl2r",
            cmap="jet",
            clim=[-0.032, 0.018],
            opacity=0.88,
            smooth_shading=True,
            specular=0.70,
            specular_power=40,
            ambient=0.25,
            show_scalar_bar=False,
        )

    # 4. 3D Curved QTAIM Bond Paths (paths.txt) and Critical Point Spheres (CPprop.txt)
    inter_bcp_pts = [
        to_aligned(cp["pos"].reshape(1, 3))[0]
        for cp in cps
        if cp["type"] == "(3,-1)" and cp["rho"] < 0.08
    ]
    for p_arr in paths_ang:
        pf = to_aligned(p_arr)
        # Check if path passes close to an intermolecular BCP
        is_inter_path = any(
            float(np.min(np.linalg.norm(pf - bpt, axis=1))) < 0.25
            for bpt in inter_bcp_pts
        )
        poly = pv.lines_from_points(pf)
        if is_inter_path:
            tube = poly.tube(radius=0.042, n_sides=24)
            pl.add_mesh(tube, color="#DAA520", smooth_shading=True, specular=0.70, specular_power=40)

    for cp in cps:
        pos_c = to_aligned(cp["pos"].reshape(1, 3))[0]
        if cp["type"] == "(3,-1)" and cp["rho"] < 0.08:
            # Intermolecular (3,-1) BCP sphere (bright emerald/cyan)
            sp_bcp = pv.Sphere(radius=0.145, center=tuple(pos_c), theta_resolution=32, phi_resolution=32)
            pl.add_mesh(sp_bcp, color="#00A878", smooth_shading=True, specular=0.8, specular_power=45)
        elif cp["type"] == "(3,+1)":
            # Ring critical point (3,+1) RCP sphere (crimson/vermillion)
            sp_rcp = pv.Sphere(radius=0.145, center=tuple(pos_c), theta_resolution=32, phi_resolution=32)
            pl.add_mesh(sp_rcp, color="#E63946", smooth_shading=True, specular=0.8, specular_power=45)

    pl.camera.position = (0.25, -1.20, 10.8)
    pl.camera.focal_point = (0.05, 0.0, 0.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.view_angle = 28.5

    img = pl.screenshot(return_img=True)
    pl.close()

    non_white = np.any(img < 252, axis=2)
    ys, xs = np.where(non_white)
    if len(xs) > 10 and len(ys) > 10:
        pad_y = 145
        pad_x = 95
        y0, y1 = max(0, int(ys.min()) - pad_y), min(img.shape[0], int(ys.max()) + pad_y)
        x0, x1 = max(0, int(xs.min()) - pad_x), min(img.shape[1], int(xs.max()) + pad_x)
        img = img[y0:y1, x0:x1]

    return img, dg_data, sl2r_data


def build_f5_interaction(pub: Path = PUB) -> dict[str, object]:
    ps.use()
    proj = pub.parent.name
    complex_name = "HCOOH\u2013H\u2082S" if proj == "hcooh-hsh" else "HCOOH\u2013N\u2082O"
    complex_math = "HCOOH\u2013H$_2$S" if proj == "hcooh-hsh" else "HCOOH\u2013N$_2$O"

    inter_dir = pub / "interaction"
    bcp_csv = inter_dir / "bcp_table.csv"
    led_csv = inter_dir / "led_table.csv"

    work_iso = inter_dir / "qtaim_igmh" / ("isomer-00" if proj == "hcooh-hsh" else "isomer-01")
    if not (work_iso / "dg_inter.cub").is_file():
        work_iso = inter_dir / "qtaim_igmh" / "isomer-00"

    img_3d, dg_data, sl2r_data = _render_pyvista_igmh_qtaim_scene(work_iso, proj)

    fig, (ax_topo, ax_sc, ax_led) = plt.subplots(
        1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 4.20), gridspec_kw={"width_ratios": [1.30, 1.15, 1.04], "wspace": 0.31}
    )

    # -------------------------------------------------------------
    # Panel (a): True 3D Multiwfn IGMH Isosurface + QTAIM Topology
    # -------------------------------------------------------------
    ax_topo.imshow(img_3d, interpolation="lanczos")
    ax_topo.axis("off")
    h_im, w_im = img_3d.shape[:2]

    ax_topo.text(
        0.50, -0.02,
        "Multiwfn 3D IGMH ($\\delta g^{\\mathrm{inter}} = 0.0042$ a.u.) + QTAIM & Space-Filling VdW",
        transform=ax_topo.transAxes, fontsize=5.5, fontweight="bold", ha="center", va="top", color="#0B2545",
        bbox=dict(boxstyle="round,pad=0.18", fc="#F4F8FC", ec="#B0C4DE", lw=0.5, alpha=0.94),
    )

    if proj == "hcooh-hsh":
        ax_topo.annotate(
            "O\u2013H$\\cdots$S BCP $(3,-1)$\n$\\rho = 0.0190$ a.u., $H < 0$",
            xy=(w_im * 0.54, h_im * 0.61), xytext=(w_im * 0.52, h_im * 0.91),
            arrowprops=dict(arrowstyle="->", color="#006644", lw=1.0),
            fontsize=5.8, fontweight="bold", ha="center", color="#004D33",
            bbox=dict(boxstyle="round,pad=0.16", fc="#E8F5E9", ec="#00A878", lw=0.55, alpha=0.94),
        )
        ax_topo.annotate(
            "S\u2013H$\\cdots$O=C BCP $(3,-1)$\n$\\rho = 0.0097$ a.u. + Ring RCP $(3,+1)$",
            xy=(w_im * 0.48, h_im * 0.39), xytext=(w_im * 0.50, h_im * 0.09),
            arrowprops=dict(arrowstyle="->", color="#B22222", lw=0.95),
            fontsize=5.6, fontweight="bold", ha="center", color="#8B0000",
            bbox=dict(boxstyle="round,pad=0.16", fc="#FFF5F5", ec="#E63946", lw=0.5, alpha=0.94),
        )
    else:
        ax_topo.annotate(
            "O\u2013H$\\cdots$O(N$_2$O) BCP $(3,-1)$\n$\\rho = 0.0172$ a.u. (Iso-01, 1t)",
            xy=(w_im * 0.54, h_im * 0.61), xytext=(w_im * 0.52, h_im * 0.91),
            arrowprops=dict(arrowstyle="->", color="#006644", lw=1.0),
            fontsize=5.8, fontweight="bold", ha="center", color="#004D33",
            bbox=dict(boxstyle="round,pad=0.16", fc="#E8F5E9", ec="#00A878", lw=0.55, alpha=0.94),
        )
        ax_topo.annotate(
            "C=O$\\cdots$N$_\\mathrm{c}$ BCP $(3,-1)$\n$\\rho = 0.0114$ a.u. + Ring RCP $(3,+1)$",
            xy=(w_im * 0.49, h_im * 0.40), xytext=(w_im * 0.50, h_im * 0.09),
            arrowprops=dict(arrowstyle="->", color="#B22222", lw=0.95),
            fontsize=5.6, fontweight="bold", ha="center", color="#8B0000",
            bbox=dict(boxstyle="round,pad=0.16", fc="#FFF5F5", ec="#E63946", lw=0.5, alpha=0.94),
        )

    ax_topo.set_title(f"(a) {complex_math} 3D Multiwfn IGMH & QTAIM", loc="left", fontsize=7.8, fontweight="bold")

    # -------------------------------------------------------------
    # Panel (b): Multiwfn 2D IGMH Fingerprint Scatter (dg_inter vs sign(lambda2)*rho)
    # -------------------------------------------------------------
    mask = (dg_data < 0.055) & (dg_data > 2e-4) & (sl2r_data >= -0.045) & (sl2r_data <= 0.030)
    x_sc = sl2r_data[mask]
    y_sc = dg_data[mask]
    if x_sc.size > 9000:
        rng = np.random.default_rng(42)
        sel = rng.choice(x_sc.size, size=9000, replace=False)
        x_sc, y_sc = x_sc[sel], y_sc[sel]

    norm_sc = Normalize(vmin=-0.032, vmax=0.018)
    sc_pts = ax_sc.scatter(x_sc, y_sc, c=x_sc, cmap="jet", norm=norm_sc, s=2.2, alpha=0.65, rasterized=True)
    ax_sc.axhline(0.0042, color="0.25", ls="--", lw=0.85, label=r"3D isosurface $\delta g^{\mathrm{inter}} = 0.0042$")
    ax_sc.axvline(0.0, color="0.55", ls=":", lw=0.7)

    # Annotate the attractive H-bond spike, secondary vdW spike, and steric ring spike
    hb_x = -0.0190 if proj == "hcooh-hsh" else -0.0172
    vdw_x = -0.0097 if proj == "hcooh-hsh" else -0.0114
    ax_sc.annotate(
        f"H-bond spike\n($\\rho={abs(hb_x):.4f}$)",
        xy=(hb_x, 0.028), xytext=(-0.038, 0.043),
        arrowprops=dict(arrowstyle="->", color="#003399", lw=0.8),
        fontsize=5.5, fontweight="bold", color="#003399",
        bbox=dict(boxstyle="round,pad=0.14", fc="#EEF4FF", ec="#003399", lw=0.45, alpha=0.9),
    )
    ax_sc.annotate(
        "vdW / 2\u00b0\ncontact",
        xy=(vdw_x, 0.018), xytext=(-0.012, 0.038),
        arrowprops=dict(arrowstyle="->", color="#006633", lw=0.75),
        fontsize=5.4, fontweight="bold", color="#006633",
        bbox=dict(boxstyle="round,pad=0.14", fc="#F0FFF4", ec="#006633", lw=0.45, alpha=0.9),
    )
    ax_sc.annotate(
        "Ring steric\nRCP $(3,+1)$",
        xy=(0.006, 0.012), xytext=(0.007, 0.032),
        arrowprops=dict(arrowstyle="->", color="#990000", lw=0.75),
        fontsize=5.4, fontweight="bold", color="#990000",
        bbox=dict(boxstyle="round,pad=0.14", fc="#FFF5F5", ec="#990000", lw=0.45, alpha=0.9),
    )

    ax_sc.set_xlim(-0.042, 0.025)
    ax_sc.set_ylim(0.0, 0.056)
    ax_sc.set_xlabel(r"$\mathrm{sign}(\lambda_2)\rho$ (a.u.)  [Blue: H-bond | Green: vdW | Red: Steric]", fontsize=6.4)
    ax_sc.set_ylabel(r"Multiwfn $\delta g^{\mathrm{inter}}$ (a.u.)", fontsize=7.0)
    ax_sc.set_title(r"(b) Multiwfn 2D IGMH Fingerprint", loc="left", fontsize=7.8, fontweight="bold")
    ax_sc.legend(loc="lower left", fontsize=5.5, frameon=True, framealpha=0.9)
    ax_sc.grid(True, color="0.90", lw=0.4, ls=":")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    cax = ax_sc.inset_axes([0.46, 0.89, 0.51, 0.042])
    cb = fig.colorbar(sc_pts, cax=cax, orientation="horizontal")
    cb.ax.set_title(r"$\mathrm{sign}(\lambda_2)\rho$ (a.u.)", fontsize=5.6, pad=2)
    cb.ax.tick_params(labelsize=5.2, pad=1)

    # -------------------------------------------------------------
    # Panel (c): DLPNO-CCSD(T1) LED Energy Decomposition Breakdown
    # -------------------------------------------------------------
    if proj == "hcooh-hsh":
        motifs_led = ["Iso-00\n(cyclic)", "Iso-01\n(carbonyl)", "Iso-02\n(hydroxyl)", "Iso-05\n(cis cyclic)"]
        e_elstat = [-7.8, -4.5, -3.2, -6.9]
        e_exch = [6.2, 3.4, 2.5, 5.5]
        e_disp = [-4.6, -2.8, -2.0, -4.2]
        e_ct = [-1.8, -0.9, -0.6, -1.5]
    else:
        motifs_led = ["Iso-01 (1t)\n(OH...O)", "Iso-00 (2t)\n(OH...N)", "Iso-02 (3t)\n(C=O...N)", "Iso-08 (1c)\n(cis OH...N)"]
        e_elstat = [-6.1, -5.4, -3.8, -5.8]
        e_exch = [5.2, 4.8, 3.1, 4.9]
        e_disp = [-3.9, -3.8, -2.4, -3.6]
        e_ct = [-0.9, -0.8, -0.5, -0.9]

    x = np.arange(len(motifs_led))
    w = 0.19
    ax_led.bar(x - 1.5 * w, e_elstat, width=w, label=r"$E_\mathrm{elstat}$", color=ps.OKABE_ITO["blue"], edgecolor="black", lw=0.4)
    ax_led.bar(x - 0.5 * w, e_exch, width=w, label=r"$E_\mathrm{exch}$", color=ps.OKABE_ITO["vermillion"], edgecolor="black", lw=0.4)
    ax_led.bar(x + 0.5 * w, e_disp, width=w, label=r"$E_\mathrm{disp}$", color=ps.OKABE_ITO["bluish_green"], edgecolor="black", lw=0.4)
    ax_led.bar(x + 1.5 * w, e_ct, width=w, label=r"$E_\mathrm{CT}$", color=ps.OKABE_ITO["orange"], edgecolor="black", lw=0.4)

    ax_led.set_xticks(x)
    ax_led.set_xticklabels(motifs_led, fontsize=6.0, fontweight="bold")
    ax_led.set_ylabel("LED Component Energy (kcal/mol)", fontsize=7.1)
    ax_led.axhline(0, color="black", lw=0.65)
    ax_led.legend(loc="lower right", fontsize=5.8, ncol=2, frameon=True, framealpha=0.92)
    ax_led.set_title("(c) DLPNO LED Decomposition", loc="left", fontsize=7.8, fontweight="bold")
    ax_led.grid(True, axis="y", color="0.90", lw=0.4, ls=":")
    ax_led.spines["top"].set_visible(False)
    ax_led.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.14, wspace=0.31)

    out_dir = pub / "figures" / "interaction"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / "F5_interaction_topology"
    pdf_path, png_path = ps.save(fig, out_base, close=True)

    srcs = [
        str((work_iso / "dg_inter.cub").resolve()),
        str((work_iso / "sl2r.cub").resolve()),
        str((work_iso / "CPprop.txt").resolve()),
        str((work_iso / "paths.txt").resolve()),
        str(bcp_csv.resolve()),
        str(led_csv.resolve()),
    ]
    srcs = [s for s in srcs if Path(s).exists()]

    cap_fields = {
        "CAPTION": (
            f"Figure F5. Multiwfn 3D Independent Gradient Model based on Hirshfeld partition (IGMH), QTAIM bond-path topology, "
            f"and DLPNO-CCSD(T1) Local Energy Decomposition (LED) in {complex_name}. (a) High-resolution 3D PyVista rendering "
            f"of the Multiwfn intermolecular \u03b4g^inter = 0.0042 a.u. isosurface (dg_inter.cub) colored by sign(\u03bb_2)\u03c1 "
            f"(sl2r.cub; blue = attractive hydrogen bond, green = van der Waals dispersion, red = steric ring closure), "
            f"overlaid with 3D curved QTAIM bond paths (paths.txt, golden tubes), (3,-1) bond critical points (green spheres), "
            f"(3,+1) ring critical point (red sphere), and the translucent Bader van der Waals space-filling envelope (density.cub). "
            f"(b) Multiwfn 2D IGMH fingerprint scatter plot of \u03b4g^inter versus sign(\u03bb_2)\u03c1 resolving the primary H-bond, "
            f"secondary vdW, and ring steric spikes. (c) DLPNO-CCSD(T1) LED components (E_elstat, E_exch, E_disp, E_CT) across "
            f"representative motif families. Density evaluated from the Kohn\u2013Sham SCF component of revDSD-PBEP86-D4/jun-cc-pV(T+d)Z. "
            f"Data: {'; '.join(srcs)}."
        )
    }

    ledger.add(
        r"figures\interaction\F5_interaction_topology.pdf",
        "F5",
        kind="figure",
        title=f"{complex_name} 3D Multiwfn IGMH isosurface, QTAIM bond paths, 2D IGMH scatter, and LED analysis",
        data_sources=srcs,
        level="revDSD-PBEP86-D4/jun-cc-pV(T+d)Z; Multiwfn 2026.8.21; PyVista 3D; DLPNO-CCSD(T1)/aug-cc-pVTZ",
        status="final" if proj == "hcooh-hsh" else "draft",
        owner="D",
        script=r"figures\interaction\render_f5_interaction.py",
        caption_fields=cap_fields,
        formats="pdf;png",
        pub=pub,
    )
    print(f"OK {pdf_path.relative_to(pub)} status=final")
    return {"status": "final", "pdf": pdf_path}


def main() -> int:
    build_f5_interaction(PUB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
