"""
================================================================================
pubstyle -- shared figure style, palette, plot helpers and results schema
File: style/pubstyle.py
Campaign: HCOOH-H2S / HCOOH-N2O publication campaign (Oct 26 - Nov 8, 2026)
Owner: F0 Foundation ("Style+Sets"). Read-only for every other agent.
================================================================================
One importable module so that every figure of both projects (manuscripts,
poster, colloquium slides) has the same fonts, line widths, colours and export.

Quick start
-----------
    import sys
    sys.path.insert(0, "style")
    import pubstyle as ps

    df = ps.read_results(r"...\\energetics\\tables\\energy_results.csv")   # schema.md
    motifs = ps.load_motif_map(r"...\\publication\\working_set\\mapping.csv")
    fig, ax = ps.ladder(df, [ps.LEVELS["gfn2"], ps.LEVELS["revdsd"], ps.LEVELS["dlpno_cbs"]],
                        show_zpe=True, grey_first=True, motif_map=motifs)
    ps.save(fig, r"...\\figures\\energy_ladder")      # -> energy_ladder.pdf + .png (300 dpi)

Contents
--------
use()                    apply the rcParams (serif/STIX, 9-11 pt, thin axes, inward ticks,
                         no top/right spines, TrueType fonts embedded in PDF/PS)
figsize(), subplots()    single (3.35 in) / double (7.0 in) column figure sizes
PALETTE, DISPLAY_NAMES   colour-blind-safe (Okabe-Ito) project + motif-family colours
motif_key(), motif_color(), project_key(), color_for(), display_name(), load_motif_map()
save(fig, path)          <path>.pdf (vector) + <path>.png (300 dpi); returns both paths
ladder(df, levels)       energy ladder (layout="levels" or layout="isomers")
compare(df, x, y)        computed-vs-literature plot with 1:1 line and optional +-% band
polar_pes(R, theta, E)   polar (R, theta) PES heat map; shared colour scale via shared_scale()
SCHEMA_COLUMNS, check_results(), read_results(), upsert_results()   (see schema.md)

Backend: if no backend was chosen by the caller (matplotlib.use, MPLBACKEND,
matplotlibrc, IPython/Jupyter/Spyder) and the process has no display or is not
interactive (no console on stdin+stdout: an agent's shell, a batch job), the Agg
backend is selected at import. A backend the caller already set is never changed.
Set PUBSTYLE_HEADLESS=1 to force Agg.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

__version__ = "1.0 (2026-10-25)"


# ==============================================================================
# 0. BACKEND (headless-safe; never overrides a backend the caller set)
# ==============================================================================
def _explicit_backend() -> str | None:
    """The backend chosen so far, or None if matplotlib would still auto-select one."""
    getter = getattr(matplotlib.rcParams, "_get_backend_or_none", None)
    if callable(getter):
        try:
            return getter()
        except Exception:                       # pragma: no cover - private API changed
            pass
    try:
        value = dict.__getitem__(matplotlib.rcParams, "backend")
    except Exception:                           # pragma: no cover
        return None
    from matplotlib import rcsetup
    return None if value is getattr(rcsetup, "_auto_backend_sentinel", object()) else value


def _should_force_agg(explicit_backend: str | None, platform: str, environ: Mapping[str, str],
                      stdin_isatty: bool, interactive: bool) -> bool:
    """Pure decision function (unit-tested): True -> select the Agg backend.
    `stdin_isatty` means "a person at a console" (stdin AND stdout are terminals, see
    _console_attached): an agent's shell or a batch job captures stdout -> Agg."""
    if explicit_backend:                        # caller / matplotlibrc / IPython already chose
        return False
    if environ.get("MPLBACKEND"):
        return False
    if environ.get("PUBSTYLE_HEADLESS", "").strip() not in ("", "0"):
        return True
    if platform.startswith(("linux", "freebsd", "openbsd", "netbsd", "sunos", "aix")):
        if not (environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")):
            return True                         # no X/Wayland display
    return not (stdin_isatty or interactive)    # batch job / agent shell: no one to show to


def _console_attached() -> bool:
    """True only if stdin and stdout are both terminals (an agent's shell on Windows gives
    stdin a pseudo-terminal but captures stdout, so it counts as headless)."""
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty()
                    and sys.stdout is not None and sys.stdout.isatty())
    except Exception:
        return False


BACKEND_FORCED_TO_AGG: bool = False
if _should_force_agg(_explicit_backend(), sys.platform, os.environ, _console_attached(),
                     hasattr(sys, "ps1") or "IPython" in sys.modules or bool(sys.flags.interactive)):
    matplotlib.use("Agg")
    BACKEND_FORCED_TO_AGG = True

import matplotlib.pyplot as plt                 # noqa: E402  (after the backend decision)
import pandas as pd                             # noqa: E402
from matplotlib import patheffects              # noqa: E402
from matplotlib.colors import Normalize         # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402


# ==============================================================================
# 1. CONSTANTS: sizes, palette, levels, uncertainty labels
# ==============================================================================
SINGLE_COLUMN_IN: float = 3.35                  # ACS/AIP single column
DOUBLE_COLUMN_IN: float = 7.0                   # double column
DPI: int = 300
HARTREE_TO_KCAL: float = 627.509474             # 1 Eh in kcal/mol (schema.md)

# Okabe & Ito (2008) colour-universal-design palette; grey for "other".
OKABE_ITO: dict[str, str] = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9", "bluish_green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
    "grey": "#999999",
}

# Fixed palette, identical in both projects: project keys and motif-family keys.
# Projects and motif families are never coloured in the same plot (ladder: motifs;
# compare(group="project"): projects), so XH_O (an H2S-only family) may share H2S's orange.
PALETTE: dict[str, str] = {
    # projects
    "H2S": OKABE_ITO["orange"],                 # HCOOH-H2S (hcooh-hsh)
    "N2O": OKABE_ITO["sky_blue"],               # HCOOH-N2O (hcooh-nno)
    # motif families (short ASCII keys)
    "OH_X": OKABE_ITO["blue"],                  # HCOOH O-H donor to the partner (incl. cyclic rings)
    "XH_O": OKABE_ITO["orange"],                # partner X-H donor to an HCOOH O (S-H...O; H2S only)
    "CH_X": OKABE_ITO["bluish_green"],          # HCOOH C-H...X (with or without C=O...N)
    "side_on": OKABE_ITO["reddish_purple"],     # side-on / stacked, no conventional H-bond
    "cis": OKABE_ITO["vermillion"],             # every complex of cis-HCOOH
    "other": OKABE_ITO["grey"],                 # unknown / unassigned
    "monomer": OKABE_ITO["black"],              # monomer rows (trans-/cis-HCOOH, H2S, N2O)
}
MOTIF_KEYS: tuple[str, ...] = ("OH_X", "XH_O", "CH_X", "side_on", "cis", "other")
FAMILY_KEYS: tuple[str, ...] = MOTIF_KEYS + ("monomer",)     # valid mapping.csv motif_family values
PROJECT_KEYS: tuple[str, ...] = ("H2S", "N2O")
PROJECTS: dict[str, str] = {"hcooh-hsh": "H2S", "hcooh-nno": "N2O"}   # schema "project" -> key

DISPLAY_NAMES: dict[str, str] = {
    "H2S": "HCOOH\u2013H$_2$S",
    "N2O": "HCOOH\u2013N$_2$O",
    "OH_X": "O\u2013H\u00b7\u00b7\u00b7X hydrogen bond",
    "XH_O": "X\u2013H\u00b7\u00b7\u00b7O (partner donor)",
    "CH_X": "C\u2013H\u00b7\u00b7\u00b7X",
    "side_on": "side-on/stacked",
    "cis": "cis-HCOOH variants",
    "other": "other",
    "monomer": "monomer",
}

GREY_SEARCH: str = "#9E9E9E"                    # greyed "search level" column / markers
GREY_LIGHT: str = "#D4D4D4"
CYCLE: list[str] = [OKABE_ITO[k] for k in ("blue", "orange", "bluish_green", "vermillion",
                                           "reddish_purple", "sky_blue", "yellow", "black")]
LEVEL_MARKERS: tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X")

# Canonical level strings: every agent writes exactly these into the "level" column.
LEVELS: dict[str, str] = {
    "gfn2": "GFN2-xTB",
    "r2scan3c": "r2SCAN-3c",
    "wb97xv": "wB97X-V/def2-TZVPP",
    "revdsd": "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z",
    "dlpno_atz": "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVTZ",
    "dlpno_aqz": "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVQZ",
    "dlpno_cbs": "DLPNO-CCSD(T1)/TightPNO/CBS(aTZ,aQZ)",
}
LEVEL_SHORT: dict[str, str] = {                 # axis/tick labels
    "GFN2-xTB": "GFN2-xTB",
    "r2SCAN-3c": "r$^2$SCAN-3c",
    "wB97X-V/def2-TZVPP": "\u03c9B97X-V",
    "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z": "revDSD-PBEP86-D4",
    "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVTZ": "DLPNO-CCSD(T1)/aTZ",
    "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVQZ": "DLPNO-CCSD(T1)/aQZ",
    "DLPNO-CCSD(T1)/TightPNO/CBS(aTZ,aQZ)": "DLPNO-CCSD(T1)/CBS",
}
# ladder column ticks: line-broken so three levels fit a 3.35 in single column at 9 pt
LEVEL_TICK: dict[str, str] = {
    "revDSD-PBEP86-D4/jun-cc-pV(T+d)Z": "revDSD-\nPBEP86-D4",
    "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVTZ": "DLPNO-\nCCSD(T1)/\naTZ",
    "DLPNO-CCSD(T1)/TightPNO/aug-cc-pVQZ": "DLPNO-\nCCSD(T1)/\naQZ",
    "DLPNO-CCSD(T1)/TightPNO/CBS(aTZ,aQZ)": "DLPNO-\nCCSD(T1)/\nCBS",
}

# Uncertainty labels (method-matrix style; plan.md Section 1, week-10 guide for GFN2-xTB).
UNCERTAINTY: dict[str, str] = {
    "geometry_Be": "\u00b10.5\u20132 % [E]",            # revDSD B_e
    "dE": "\u00b10.2\u20130.5 kcal/mol [E]",            # DLPNO-CCSD(T1)/CBS relative energies
    "De": "\u00b10.3\u20130.5 kcal/mol [E]",            # binding energies, CP
    "pes": "\u00b11\u20132 kcal/mol [E]",               # PES maps; barriers are lower bounds
    "gfn2": "\u00b11\u20132 kcal/mol (search level)",   # GFN2-xTB energies
    "interaction": "qualitative",                       # QTAIM / NCI / LED
}

STYLE_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "figure.titlesize": 11.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "legend.title_fontsize": 9.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.formatter.use_mathtext": True,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    "figure.figsize": (SINGLE_COLUMN_IN, 2.6),
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,                         # embedded TrueType (print shops, journals)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "image.cmap": "viridis",
}


# ==============================================================================
# 2. STYLE
# ==============================================================================
def use(rc: Mapping[str, Any] | None = None) -> None:
    """Apply the campaign rcParams (idempotent). `rc` overrides single entries."""
    matplotlib.rcParams.update(STYLE_RC)
    if rc:
        matplotlib.rcParams.update(dict(rc))


def figsize(width: str | float = "single", aspect: float = 0.75,
            height: float | None = None) -> tuple[float, float]:
    """(w, h) in inches: width "single" (3.35 in), "double" (7.0 in) or a number."""
    if width == "single":
        w = SINGLE_COLUMN_IN
    elif width == "double":
        w = DOUBLE_COLUMN_IN
    else:
        w = float(width)
    return (w, float(height) if height is not None else w * float(aspect))


def subplots(width: str | float = "single", aspect: float = 0.75, height: float | None = None,
             **kwargs):
    """plt.subplots with the campaign style and a column-width figure size."""
    use()
    return plt.subplots(figsize=figsize(width, aspect, height), **kwargs)


# ==============================================================================
# 3. PALETTE LOOKUPS (never raise on unknown motifs)
# ==============================================================================
def _norm_text(s: Any) -> str:
    s = str(s).strip().lower()
    for a, b in (("\u00b7", "."), ("\u2026", "..."), ("\u22ef", "..."), ("\u2013", "-"),
                 ("\u2014", "-"), ("\u2212", "-"), ("$", ""), ("_", "")):
        s = s.replace(a, b)
    s = re.sub(r"\.{2,}", "...", s)
    return re.sub(r"\s+", " ", s)


_MOTIF_ALIASES: dict[str, str] = {}
for _k in FAMILY_KEYS:
    _MOTIF_ALIASES[_norm_text(_k)] = _k
    _MOTIF_ALIASES[_norm_text(DISPLAY_NAMES[_k])] = _k
_MOTIF_ALIASES.update({
    _norm_text(a): k for a, k in (
        ("OH...X", "OH_X"), ("O-H...X", "OH_X"), ("OH...X hydrogen bond", "OH_X"),
        ("XH...O", "XH_O"), ("X-H...O", "XH_O"), ("SH...O", "XH_O"), ("S-H...O", "XH_O"),
        ("CH...X", "CH_X"), ("C-H...X", "CH_X"), ("side-on", "side_on"), ("side on", "side_on"),
        ("stacked", "side_on"), ("sideon", "side_on"), ("cis-HCOOH", "cis"), ("cis variants", "cis"),
    )})


def _motif_key_or_none(name: Any) -> str | None:
    if name is None:
        return None
    if isinstance(name, float) and math.isnan(name):
        return None
    text = _norm_text(name)
    if not text or text in ("nan", "none"):
        return None
    return _MOTIF_ALIASES.get(text)


def motif_key(name: Any) -> str:
    """Normalize a motif name (key, display name, ASCII variant) to a family key; unknown -> 'other'."""
    return _motif_key_or_none(name) or "other"


def motif_color(key: Any) -> str:
    """Colour of a motif family. Never raises: unknown, empty or NaN -> PALETTE['other']."""
    return PALETTE[motif_key(key)]


def project_key(name: Any) -> str | None:
    """'hcooh-hsh'/'H2S'/'HCOOH-H2S' -> 'H2S'; 'hcooh-nno'/'N2O'/'HCOOH-N2O' -> 'N2O'; else None."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None
    t = _norm_text(name).replace(" ", "")
    if t in PROJECTS:
        return PROJECTS[t]
    aliases = {"h2s": "H2S", "hcooh-h2s": "H2S", "hcoohh2s": "H2S", "hcooh-hsh": "H2S",
               "n2o": "N2O", "hcooh-n2o": "N2O", "hcoohn2o": "N2O", "hcooh-nno": "N2O"}
    return aliases.get(t)


def color_for(value: Any) -> str:
    """Palette colour of a project or motif key/name (projects first); unknown -> 'other'."""
    p = project_key(value)
    return PALETTE[p] if p else motif_color(value)


def display_name(value: Any) -> str:
    """Legend text for a project or motif key; anything else is returned as str()."""
    p = project_key(value)
    if p:
        return DISPLAY_NAMES[p]
    k = _motif_key_or_none(value)
    return DISPLAY_NAMES[k] if k else str(value)


def load_motif_map(mapping_csv: str | os.PathLike) -> dict[str, str]:
    """{working_label: motif_family} from a working_set mapping.csv (UTF-8)."""
    m = pd.read_csv(mapping_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for col in ("working_label", "motif_family"):
        if col not in m.columns:
            raise ValueError(f"{mapping_csv}: column '{col}' missing (see working_set README.md)")
    return {r.working_label: motif_key(r.motif_family) for r in m.itertuples(index=False)}


# ==============================================================================
# 4. EXPORT
# ==============================================================================
def save(fig, path_without_ext: str | os.PathLike, *, dpi: int = DPI, tight: bool = True,
         close: bool = False) -> tuple[Path, Path]:
    """
    Write <path>.pdf (vector, TrueType fonts embedded) and <path>.png (dpi, default 300).
    Creates the parent folder. A trailing .pdf/.png in the path is ignored.
    tight=True (default) trims to the drawn content; tight=False keeps the exact figure size
    (e.g. a 7.0 in wide figure gives a 2100 px wide PNG at 300 dpi).
    The PDF carries no creation date, so re-running a script reproduces the same file.
    Returns (pdf_path, png_path).
    """
    p = Path(path_without_ext)
    if p.suffix.lower() in (".pdf", ".png"):
        p = p.with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = Path(str(p) + ".pdf"), Path(str(p) + ".png")
    # tight=False must override STYLE_RC's savefig.bbox = "tight" (bbox_inches=None would fall back to it)
    kw: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if not tight:
        rc["savefig.bbox"] = "standard"
    with matplotlib.rc_context(rc):
        fig.savefig(pdf, format="pdf",
                    metadata={"CreationDate": None, "Creator": "pubstyle.py (HCOOH publication campaign)"},
                    **kw)
        fig.savefig(png, format="png", dpi=dpi, **kw)
    if close:
        plt.close(fig)
    return pdf, png


# ==============================================================================
# 5. ENERGY LADDER
# ==============================================================================
def _natural_key(s: Any) -> list:
    """'isomer-2' < 'isomer-10'; tuples so that numbers and text never compare directly."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", str(s)) if t]


def _spread_labels(y: Sequence[float], gap: float) -> np.ndarray:
    """1-D label de-overlap: clusters of labels closer than `gap` are spread evenly
    around the mean of their original positions (deterministic)."""
    y = np.asarray(y, float)
    if y.size == 0:
        return y.copy()
    clusters = [[int(i)] for i in np.argsort(y, kind="stable")]

    def place(c: list[int]) -> list[float]:
        centre, m = float(np.mean(y[c])), len(c)
        return [centre + (k - (m - 1) / 2.0) * gap for k in range(m)]

    changed = True
    while changed:
        changed = False
        merged = [clusters[0]]
        for c in clusters[1:]:
            if place(c)[0] - place(merged[-1])[-1] < gap - 1e-12:
                merged[-1] = merged[-1] + c
                changed = True
            else:
                merged.append(c)
        clusters = merged
    pos = np.empty_like(y)
    for c in clusters:
        for i, v in zip(c, place(c)):
            pos[i] = v
    return pos


def _points_to_data_dy(ax, points: float) -> float:
    height_px = max(ax.get_window_extent().height, 1.0)
    y0, y1 = ax.get_ylim()
    return abs(y1 - y0) * (points * ax.figure.dpi / 72.0) / height_px


def _level_tick(level: str, level_labels: Mapping[str, str] | None) -> str:
    if level_labels and level in level_labels:
        return level_labels[level]
    if level in LEVEL_TICK:
        return LEVEL_TICK[level]
    if level in LEVEL_SHORT:
        return LEVEL_SHORT[level]
    return str(level).replace("/", "/\n")


def _require(df, cols: Iterable[str | None], what: str) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise ValueError(f"{what}: column(s) missing from the DataFrame: {', '.join(missing)}")


def ladder(df: pd.DataFrame, levels: Sequence[str] | None = None, *, layout: str = "levels",
           isomer_col: str = "isomer", level_col: str = "level", dE_col: str = "dE_e_kcal",
           dE0_col: str = "dE_0_kcal", motif_col: str = "motif", project_col: str = "project",
           project: str | None = None, isomers: Sequence[str] | None = None,
           motif_map: Mapping[str, str] | None = None, show_zpe: bool = False,
           grey_first: bool = False, search_label: str = "search level",
           level_labels: Mapping[str, str] | None = None,
           isomer_labels: Mapping[str, str] | Callable[[str], str] | None = None,
           labels: str = "last", bar_width: float = 0.56, connectors: bool = True,
           legend: str | None = "above", ylabel: str | None = None, title: str | None = None,
           ax=None, figsize: tuple[float, float] | None = None):
    """
    Energy ladder from a long results DataFrame (schema.md: one row per isomer and level).

    layout="levels" (default): one column per level along x in the order given; each isomer
        a short horizontal bar at its dE (kcal/mol), coloured by motif family; thin
        connectors join the same isomer across adjacent levels; isomer labels to the right
        of the last column (labels="last" | "first" | "both" | "none"), spread apart with
        leader lines when they would overlap.
    layout="isomers": isomers along x, one marker series (shape) per level, dE on y, a thin
        line per isomer joining its levels; marker face = motif colour.
    show_zpe=True: dE_0 drawn as open symbols beside each bar (levels) / at each marker (isomers).
    grey_first=True: the first level is drawn grey and marked `search_label` (week-11 guide:
        keep GFN2-xTB as the grey "search level").
    Motif per isomer: motif_map[isomer] if given (e.g. load_motif_map(mapping.csv)), else the
        first recognisable value of `motif_col` (then of a "motif_family" column) -- a family key
        ("OH_X", "CH_X", "side_on", "cis", ...) or its display name --, else "other".
    project: keep only rows of that project ("hcooh-hsh"/"H2S", "hcooh-nno"/"N2O"; a schema CSV
        may hold both projects).
    Missing values (empty cells / NaN) are skipped. Duplicate (isomer, level) rows raise.
    Returns (fig, ax).
    """
    use()
    if layout not in ("levels", "isomers"):
        raise ValueError("layout must be 'levels' or 'isomers'")
    if labels not in ("last", "first", "both", "none"):
        raise ValueError("labels must be 'last', 'first', 'both' or 'none'")
    _require(df, [isomer_col, level_col, dE_col] + ([dE0_col] if show_zpe else []), "ladder")
    data = df.copy()
    if project is not None:
        _require(data, [project_col], "ladder(project=...)")
        want = project_key(project)
        if want is not None:                    # "hcooh-hsh", "H2S", "HCOOH-H2S" all select H2S
            data = data[data[project_col].map(project_key) == want]
        else:
            data = data[data[project_col] == project]
    if levels is None:
        levels = list(dict.fromkeys(data[level_col].tolist()))
    levels = list(levels)
    if not levels:
        raise ValueError("ladder: no levels to plot")
    data = data[data[level_col].isin(levels)]
    dup = data.duplicated([isomer_col, level_col], keep=False)
    if dup.any():
        pairs = sorted({(str(a), str(b)) for a, b in
                        data.loc[dup, [isomer_col, level_col]].itertuples(index=False)})
        raise ValueError(f"ladder: duplicate rows for (isomer, level): {pairs[:5]}")
    for col in [dE_col] + ([dE0_col] if show_zpe else []):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    def pivot(col: str) -> pd.DataFrame:
        return data.pivot(index=isomer_col, columns=level_col, values=col).reindex(columns=levels)

    E = pivot(dE_col)
    E0 = pivot(dE0_col) if show_zpe else None
    if isomers is None:
        isomers = sorted(E.index.tolist(), key=_natural_key)
    isomers = [i for i in isomers if i in E.index]
    E = E.reindex(isomers)
    if E0 is not None:
        E0 = E0.reindex(isomers)
    arr = E.to_numpy(float)
    vals = arr[np.isfinite(arr)]
    if E0 is not None:
        arr0 = E0.to_numpy(float)
        vals = np.concatenate([vals, arr0[np.isfinite(arr0)]])
    if vals.size == 0:
        raise ValueError("ladder: no finite dE values for the requested levels")

    motifs: dict[str, str] = {}
    for iso in isomers:
        m = motif_map.get(iso) if motif_map is not None else None
        for col in (motif_col, "motif_family"):
            if m is not None or col not in data.columns:
                continue
            cands = [v for v in data.loc[data[isomer_col] == iso, col].tolist()
                     if _motif_key_or_none(v) is not None]
            m = cands[0] if cands else None
        motifs[iso] = motif_key(m)
    colors = {iso: PALETTE[motifs[iso]] for iso in isomers}

    def lab(iso: str) -> str:
        if isomer_labels is None:
            return str(iso)
        if callable(isomer_labels):
            return str(isomer_labels(iso))
        return str(isomer_labels.get(iso, iso))

    if ax is None:
        if figsize is None:
            figsize = (SINGLE_COLUMN_IN, 3.3) if layout == "levels" else (SINGLE_COLUMN_IN, 2.8)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    lo, hi = min(0.0, float(vals.min())), float(vals.max())
    pad = 0.06 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    n_lev = len(levels)
    zpe_dx = 0.09

    if layout == "levels":
        w = float(bar_width)
        for j, lev in enumerate(levels):
            grey = grey_first and j == 0
            for iso in isomers:
                v = E.at[iso, lev]
                if not np.isfinite(v):
                    continue
                col = GREY_SEARCH if grey else colors[iso]
                ax.hlines(v, j - w / 2, j + w / 2, colors=col, linewidth=2.0, zorder=3,
                          capstyle="butt")
                if E0 is not None and np.isfinite(E0.at[iso, lev]):
                    ax.plot(j + w / 2 + zpe_dx, E0.at[iso, lev], ls="none", marker="o", ms=3.2,
                            mfc="white", mec=col, mew=0.8, zorder=4, clip_on=False)
            if connectors and j < n_lev - 1:
                for iso in isomers:
                    v, vn = E.at[iso, lev], E.at[iso, levels[j + 1]]
                    if np.isfinite(v) and np.isfinite(vn):
                        ax.plot([j + w / 2, j + 1 - w / 2], [v, vn], lw=0.5, alpha=0.7, zorder=2,
                                color=GREY_SEARCH if grey else colors[iso], solid_capstyle="butt")
        # isomer labels, spread apart with leader lines where they would overlap
        gap = _points_to_data_dy(ax, 1.15 * float(matplotlib.rcParams["font.size"]))
        sides = {"last": ["right"], "first": ["left"], "both": ["left", "right"], "none": []}[labels]
        for side in sides:
            by_col: dict[int, list[tuple[str, float]]] = {}
            for iso in isomers:
                idx = [j for j, lev in enumerate(levels) if np.isfinite(E.at[iso, lev])]
                if not idx:
                    continue
                j = idx[-1] if side == "right" else idx[0]
                by_col.setdefault(j, []).append((iso, float(E.at[iso, levels[j]])))
            for j, items in by_col.items():
                pos = _spread_labels([v for _, v in items], gap)
                if side == "right":
                    x_end = j + w / 2 + (2 * zpe_dx if E0 is not None else 0.0)
                    x_txt, ha, x_lead = x_end + 0.10, "left", x_end + 0.07
                else:
                    x_end = j - w / 2
                    x_txt, ha, x_lead = x_end - 0.10, "right", x_end - 0.07
                for (iso, v), yp in zip(items, pos):
                    if abs(yp - v) > 1e-9:
                        ax.plot([x_end, x_lead], [v, yp], lw=0.4, color="0.45", zorder=2, clip_on=False)
                    ax.text(x_txt, yp, lab(iso), ha=ha, va="center", clip_on=False, zorder=5)
        left = -0.5 - (0.85 if labels in ("first", "both") else 0.0)
        right = n_lev - 1 + 0.5 + (0.85 if labels in ("last", "both") else 0.0)
        ax.set_xlim(left, right)
        ax.set_xticks(range(n_lev))
        ticks = [_level_tick(lev, level_labels) for lev in levels]
        if grey_first:                          # "(search\nlevel)": keeps the first tick narrow
            ticks[0] = f"{ticks[0]}\n(" + search_label.replace(" ", "\n", 1) + ")"
        ax.set_xticklabels(ticks)
        if grey_first:
            ax.get_xticklabels()[0].set_color(GREY_SEARCH)
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    else:  # layout == "isomers"
        xs = np.arange(len(isomers), dtype=float)
        offsets = (np.arange(n_lev) - (n_lev - 1) / 2.0) * min(0.14, 0.6 / max(n_lev, 1))
        for i, iso in enumerate(isomers):
            pts = [(xs[i] + offsets[j], E.at[iso, lev]) for j, lev in enumerate(levels)
                   if np.isfinite(E.at[iso, lev])]
            if len(pts) > 1:
                px, py = zip(*pts)
                ax.plot(px, py, color=colors[iso], lw=0.6, alpha=0.85, zorder=2)
        for j, lev in enumerate(levels):
            grey = grey_first and j == 0
            mk = LEVEL_MARKERS[j % len(LEVEL_MARKERS)]
            for i, iso in enumerate(isomers):
                v = E.at[iso, lev]
                if np.isfinite(v):
                    ax.plot(xs[i] + offsets[j], v, ls="none", marker=mk, ms=4.3, zorder=3,
                            mfc=GREY_LIGHT if grey else colors[iso],
                            mec=GREY_SEARCH if grey else "black", mew=0.5)
                if E0 is not None and np.isfinite(E0.at[iso, lev]):
                    ax.plot(xs[i] + offsets[j], E0.at[iso, lev], ls="none", marker=mk, ms=4.3,
                            mfc="none", mec=GREY_SEARCH if grey else colors[iso], mew=0.8, zorder=4)
        many = len(isomers) > 6
        ax.set_xticks(xs)
        ax.set_xticklabels([lab(i) for i in isomers], rotation=45 if many else 0,
                           ha="right" if many else "center", rotation_mode="anchor" if many else "default")
        ax.set_xlim(-0.6, len(isomers) - 0.4)
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel if ylabel is not None else r"$\Delta E$ (kcal/mol)")
    if title:
        ax.set_title(title)
    if legend:
        handles: list[Line2D] = []
        if layout == "isomers":
            for j, lev in enumerate(levels):
                grey = grey_first and j == 0
                text = _level_tick(lev, level_labels).replace("\n", "")
                handles.append(Line2D([], [], ls="none", marker=LEVEL_MARKERS[j % len(LEVEL_MARKERS)],
                                      ms=4.3, mfc=GREY_LIGHT if grey else "white",
                                      mec=GREY_SEARCH if grey else "black", mew=0.6,
                                      label=f"{text} ({search_label})" if grey else text))
        elif grey_first:
            handles.append(Line2D([], [], color=GREY_SEARCH, lw=2.0, label=search_label))
        for k in FAMILY_KEYS:
            if k in motifs.values():
                if layout == "levels":
                    handles.append(Line2D([], [], color=PALETTE[k], lw=2.0, label=DISPLAY_NAMES[k]))
                else:
                    handles.append(Line2D([], [], ls="none", marker="s", ms=5, mfc=PALETTE[k],
                                          mec="none", label=DISPLAY_NAMES[k]))
        if show_zpe:
            handles.append(Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec="0.2",
                                  mew=0.8, label=r"$\Delta E_0$ (with ZPE)"))
        if legend == "above":
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                      borderaxespad=0.0, handlelength=1.4)
        else:
            ax.legend(handles=handles, loc=legend)
    return fig, ax


# ==============================================================================
# 6. COMPUTED vs LITERATURE
# ==============================================================================
def compare(df: pd.DataFrame, x: str, y: str, *, label: str | None = None, group: str | None = None,
            band_pct: float | None = None, xerr: str | None = None, yerr: str | None = None,
            log: bool = False, xlabel: str | None = None, ylabel: str | None = None,
            title: str | None = None, show_stats: bool = True,
            colors: Mapping[str, str] | None = None, legend: str | None = "lower right",
            ax=None, figsize: tuple[float, float] | None = None):
    """
    Computed (y) vs reference/literature (x) with a 1:1 line (geometry parameters,
    rotational constants, ...).
    band_pct: shaded +-band_pct % band around y = x (e.g. 2 for the B_e +-0.5-2 % class).
    label: column with point labels; group: column for colours (motif-family or project keys
    get their PALETTE colour, anything else the Okabe-Ito cycle; `colors` overrides).
    log=True: both axes logarithmic (rotational constants A >> B, C).
    show_stats: mean absolute % deviation 100*|y-x|/|x| (MAPD), its maximum and n in the corner
        (rows with x = 0 are left out of the statistics, not of the plot).
    Rows with a non-numeric x or y are skipped. Returns (fig, ax).
    """
    use()
    _require(df, [x, y, label, group, xerr, yerr], "compare")
    d = df.copy()
    d[x] = pd.to_numeric(d[x], errors="coerce")
    d[y] = pd.to_numeric(d[y], errors="coerce")
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]
    if d.empty:
        raise ValueError("compare: no rows with finite x and y")
    if log and ((d[x] <= 0).any() or (d[y] <= 0).any()):
        raise ValueError("compare: log=True needs positive x and y")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (SINGLE_COLUMN_IN, SINGLE_COLUMN_IN))
    else:
        fig = ax.figure
    lo = float(min(d[x].min(), d[y].min()))
    hi = float(max(d[x].max(), d[y].max()))
    if log:
        lim = (lo / 1.15, hi * 1.15)
        xx = np.geomspace(*lim, 200)
    else:
        span = (hi - lo) or abs(hi) or 1.0
        lim = (lo - 0.06 * span, hi + 0.06 * span)
        xx = np.linspace(*lim, 200)
    if band_pct:
        p = float(band_pct) / 100.0
        ax.fill_between(xx, xx * (1 - p), xx * (1 + p), color="#E4E4E4", lw=0, zorder=0,
                        label=f"\u00b1{band_pct:g} %")
    ax.plot(xx, xx, color="0.35", lw=0.7, ls="--", zorder=1, label="1:1")
    if group:
        names = list(dict.fromkeys(d[group].astype(str).tolist()))
        known = all(project_key(n) or _motif_key_or_none(n) for n in names)
        gcol = {n: (colors or {}).get(n) or (color_for(n) if known else CYCLE[k % len(CYCLE)])
                for k, n in enumerate(names)}
        series = [(d[d[group].astype(str) == n], gcol[n], display_name(n) if known else n)
                  for n in names]
    else:
        series = [(d, CYCLE[0], None)]
    for sub, col, text in series:
        if xerr or yerr:
            ax.errorbar(sub[x], sub[y],
                        xerr=pd.to_numeric(sub[xerr], errors="coerce") if xerr else None,
                        yerr=pd.to_numeric(sub[yerr], errors="coerce") if yerr else None,
                        fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, zorder=2)
        ax.scatter(sub[x], sub[y], s=18, color=col, edgecolors="black", linewidths=0.4, zorder=3,
                   label=text)
    if label:
        for xv, yv, tv in zip(d[x], d[y], d[label]):
            ax.annotate(str(tv), (xv, yv), xytext=(3, 3), textcoords="offset points",
                        ha="left", va="bottom", zorder=5)
    if show_stats:
        nz = d[d[x] != 0]
        if len(nz):
            dev = 100.0 * (nz[y] - nz[x]) / nz[x].abs()
            ax.text(0.03, 0.97, f"MAPD {dev.abs().mean():.2f} %, max {dev.abs().max():.2f} % "
                    f"(n = {len(nz)})", transform=ax.transAxes, ha="left", va="top")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    if title:
        ax.set_title(title)
    if legend and (group or band_pct):
        ax.legend(loc=legend)
    return fig, ax


# ==============================================================================
# 7. POLAR PES MAP
# ==============================================================================
@dataclass(frozen=True)
class PESScale:
    """Shared colour scale for PES maps (pass the same object for both complexes)."""

    vmin: float = 0.0
    vmax: float = 10.0
    step: float | None = None                   # contour spacing; None -> automatic
    cmap: str = "viridis"

    @property
    def contour_step(self) -> float:
        return float(self.step) if self.step else _nice_step(self.vmax - self.vmin)

    @property
    def contour_levels(self) -> np.ndarray:
        s = self.contour_step
        return np.arange(self.vmin + s, self.vmax + 1e-9 * max(1.0, abs(self.vmax)), s)


def _nice_step(span: float) -> float:
    if not np.isfinite(span) or span <= 0:
        return 1.0
    raw = span / 8.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw - 1e-12:
            return m * mag
    return 10.0 * mag                           # pragma: no cover


def shared_scale(*E_arrays, vmax: float | None = None, step: float | None = None,
                 cmap: str = "viridis") -> PESScale:
    """One PESScale covering several maps (NaN-safe): vmin = min(0, data), vmax rounded up."""
    vals = [np.asarray(a, float).ravel() for a in E_arrays]
    allv = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([])
    if allv.size == 0:
        raise ValueError("shared_scale: no finite values")
    vmin = min(0.0, float(allv.min()))
    if vmax is None:
        st = float(step) if step else _nice_step(float(allv.max()) - vmin)
        vmax = vmin + math.ceil((float(allv.max()) - vmin) / st - 1e-9) * st
    return PESScale(vmin=vmin, vmax=float(vmax), step=step, cmap=cmap)


def _centres_to_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def polar_pes(R: Sequence[float], theta: Sequence[float], E, minima: Sequence | None = None, *,
              vmin: float | None = None, vmax: float | None = None,
              scale: PESScale | tuple[float, float] | None = None, cmap: str | None = None,
              contours: bool = True, contour_step: float | None = None,
              contour_levels: Sequence[float] | None = None, contour_labels: bool = False,
              relative: bool = False, cbar: bool = True, cbar_label: str = "\u0394E (kcal/mol)",
              theta_zero: str = "N", theta_direction: int = 1, r_label: str = "R (\u00c5)",
              rlim: tuple[float, float] | None = None, rlabel_position: float = 22.5,
              nan_color: str = "#D9D9D9", title: str | None = None, ax=None,
              figsize: tuple[float, float] | None = None):
    """
    Polar heat map of a relaxed (R, theta) PES.
    R: radial grid (Angstrom, 1-D, n_R); theta: angular grid (degrees, 0-360, 1-D, n_theta);
    E: 2-D array (n_R, n_theta) [or (n_theta, n_R): transposed automatically] in kcal/mol
       relative to the global minimum (relative=True subtracts nanmin(E) first).
       NaN = unconverged/skipped point: drawn in `nan_color`, ignored by the contours.
    minima: [(R, theta_deg, label), ...] drawn as white stars with labels.
    Colour scale: vmin/vmax, or scale=PESScale (shared_scale(E_h2s, E_n2o)) or a
       (vmin, vmax) tuple, so both complexes share one scale; explicit vmin/vmax win.
       Values above vmax are clipped (the colour bar shows an extension).
    theta_zero="N": theta = 0 at the top (gallery orientation: carbonyl up), counterclockwise.
    Returns (fig, ax).
    """
    use()
    R = np.asarray(R, float).ravel()
    T = np.asarray(theta, float).ravel()
    Z = np.array(E, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"polar_pes: E must be 2-D, got shape {Z.shape}")
    if Z.shape == (R.size, T.size):
        pass
    elif Z.shape == (T.size, R.size):
        Z = Z.T
    else:
        raise ValueError(f"polar_pes: E shape {Z.shape} does not match (len(R), len(theta)) = "
                         f"({R.size}, {T.size})")
    if not np.isfinite(Z).any():
        raise ValueError("polar_pes: E has no finite values")
    if relative:
        Z = Z - np.nanmin(Z)
    ro, to = np.argsort(R), np.argsort(T)
    R, T, Z = R[ro], T[to], Z[np.ix_(ro, to)]

    # colour scale
    s_vmin = s_vmax = s_step = None
    s_cmap = None
    if isinstance(scale, PESScale):
        s_vmin, s_vmax, s_cmap, s_step = scale.vmin, scale.vmax, scale.cmap, scale.step
    elif scale is not None:
        s_vmin, s_vmax = (float(v) for v in scale)
    if vmin is None:
        vmin = s_vmin if s_vmin is not None else min(0.0, float(np.nanmin(Z)))
    if vmax is None:
        vmax = s_vmax if s_vmax is not None else float(np.nanmax(Z))
    vmin, vmax = float(vmin), float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0
    cm = matplotlib.colormaps[cmap or s_cmap or "viridis"].with_extremes(bad=nan_color)

    # cell edges; theta cells subdivided so they follow the arcs
    periodic = False
    if T.size >= 3:
        step = float(np.median(np.diff(T)))
        periodic = abs((T[-1] - T[0] + step) - 360.0) < 0.5 * step
    t_edges = _centres_to_edges(T)
    sub = max(1, int(math.ceil(float(np.max(np.diff(t_edges))) / 3.0)))
    t_fine = np.concatenate([np.linspace(t_edges[i], t_edges[i + 1], sub + 1)[:-1]
                             for i in range(t_edges.size - 1)] + [t_edges[-1:]])
    Z_fine = np.ma.masked_invalid(np.repeat(Z, sub, axis=1))
    r_edges = _centres_to_edges(R)
    r_edges[0] = max(r_edges[0], 0.0)

    if ax is None:
        fig = plt.figure(figsize=figsize or (SINGLE_COLUMN_IN, SINGLE_COLUMN_IN * 0.9))
        ax = fig.add_subplot(projection="polar")
    else:
        fig = ax.figure
        if getattr(ax, "name", "") != "polar":
            raise ValueError("polar_pes: ax must be a polar axes (projection='polar')")
    ax.set_theta_zero_location(theta_zero)
    ax.set_theta_direction(theta_direction)
    mesh = ax.pcolormesh(np.radians(t_fine), r_edges, Z_fine, cmap=cm, norm=Normalize(vmin, vmax),
                         shading="flat", zorder=1)
    mesh.set_edgecolor("face")
    mesh.set_linewidth(0.1)

    if contours and np.isfinite(Z).sum() >= 4 and R.size >= 2 and T.size >= 2:
        if contour_levels is not None:
            lv = np.asarray(contour_levels, float)
        else:
            st = float(contour_step or s_step or _nice_step(vmax - vmin))
            lv = np.arange(vmin + st, vmax + 1e-9 * max(1.0, abs(vmax)), st)
        lv = lv[(lv > np.nanmin(Z)) & (lv < np.nanmax(Z))]
        if lv.size:
            Tc, Zc = T, Z
            if periodic:
                Tc = np.append(T, T[0] + 360.0)
                Zc = np.hstack([Z, Z[:, :1]])
            cs = ax.contour(np.radians(Tc), R, np.ma.masked_invalid(Zc), levels=lv, colors="white",
                            linewidths=0.4, alpha=0.8, zorder=2)
            if contour_labels:
                ax.clabel(cs, fmt="%g", fontsize=7)
    if minima:
        for m in minima:
            r_m, t_m = float(m[0]), float(m[1])
            text = str(m[2]) if len(m) > 2 and m[2] is not None else ""
            ax.plot(np.radians(t_m), r_m, ls="none", marker="*", ms=7.5, mfc="white", mec="black",
                    mew=0.6, zorder=5, clip_on=False)
            if text:
                ax.annotate(text, (np.radians(t_m), r_m), xytext=(4, 3), textcoords="offset points",
                            ha="left", va="bottom", zorder=6,
                            path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")])
    r_lo, r_hi = rlim if rlim is not None else (0.0, float(r_edges[-1]))
    ax.set_ylim(r_lo, r_hi)
    from matplotlib.ticker import MaxNLocator   # radial ticks only where there are data
    t_lo = max(float(r_lo), float(r_edges[0]))
    rt = MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(t_lo, r_hi)
    rt = rt[(rt >= t_lo - 1e-9) & (rt <= r_hi + 1e-9)]
    if rt.size:
        ax.set_rticks(rt)
    ax.set_rlabel_position(rlabel_position)
    ax.set_thetagrids(np.arange(0, 360, 45))
    ax.grid(True, color="0.55", lw=0.3, alpha=0.6)
    ax.set_axisbelow(False)
    if "polar" in ax.spines:
        ax.spines["polar"].set_linewidth(0.7)
    ax.tick_params(axis="both", pad=1.5)
    ax.text(np.radians(rlabel_position), r_hi * 1.16, r_label, ha="center", va="center")
    if cbar:
        over, under = np.nanmax(Z) > vmax, np.nanmin(Z) < vmin
        extend = "both" if over and under else "max" if over else "min" if under else "neither"
        cb = fig.colorbar(mesh, ax=ax, pad=0.12, fraction=0.05, shrink=0.85, extend=extend)
        cb.set_label(cbar_label)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(direction="in", width=0.6)
    if title:
        ax.set_title(title, pad=12)
    return fig, ax


# ==============================================================================
# 8. RESULTS SCHEMA (machine-readable twin of schema.md)
# ==============================================================================
SCHEMA_VERSION: str = "1.0 (2026-10-25)"
SCHEMA_DOC: Path = Path(__file__).with_name("schema.md")
SCHEMA_COLUMNS: list[str] = [
    "isomer", "project", "level",
    "E_e_Eh", "ZPE_Eh", "dE_e_kcal", "dE_0_kcal",
    "D_e_raw_kcal", "D_e_CP_kcal", "D_e_halfCP_kcal",
    "A_e_MHz", "B_e_MHz", "C_e_MHz",
    "max_gradient_Eh_bohr", "softest_freq_cm1", "n_imag",
    "uncertainty_label", "provenance", "computed_on",
]
SCHEMA_TEXT_COLUMNS: tuple[str, ...] = ("isomer", "project", "level", "uncertainty_label",
                                        "provenance", "computed_on")
SCHEMA_NUMERIC_COLUMNS: tuple[str, ...] = tuple(c for c in SCHEMA_COLUMNS if c not in SCHEMA_TEXT_COLUMNS)
SCHEMA_KEY: tuple[str, str, str] = ("project", "isomer", "level")
SCHEMA_PROJECTS: tuple[str, ...] = ("hcooh-hsh", "hcooh-nno")
MONOMER_LABELS: tuple[str, ...] = ("trans-HCOOH", "cis-HCOOH", "H2S", "N2O")
_ISOMER_RE = re.compile(r"^isomer-\d{2}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][0-9:.+\-Z]+)?$")


class SchemaError(ValueError):
    """A results table does not follow schema.md."""


def _blank(v: Any) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return True
    return str(v).strip() in ("", "nan", "NaN", "None")


def check_results(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """
    Validate a results table against schema.md. Always: every column of SCHEMA_COLUMNS
    present (extra columns allowed), otherwise SchemaError naming the missing columns.
    strict=True also checks values: project in SCHEMA_PROJECTS, isomer 'isomer-NN' or a
    monomer label, level/provenance non-empty, numeric columns numeric or empty, n_imag a
    non-negative integer, computed_on an ISO date, no duplicate (project, isomer, level).
    Returns df unchanged (for chaining).
    """
    if not hasattr(df, "columns"):
        raise TypeError("check_results expects a pandas DataFrame")
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"results table is missing {len(missing)} schema column(s): "
                          f"{', '.join(missing)}  (required, in order: {', '.join(SCHEMA_COLUMNS)}; "
                          f"see {SCHEMA_DOC})")
    if not strict:
        return df
    problems: list[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=2):     # row 1 = CSV header
        r = dict(zip(df.columns, row))
        if str(r["project"]) not in SCHEMA_PROJECTS:
            problems.append(f"row {i}: project {r['project']!r} not in {SCHEMA_PROJECTS}")
        iso = str(r["isomer"])
        if not (_ISOMER_RE.match(iso) or iso in MONOMER_LABELS):
            problems.append(f"row {i}: isomer {iso!r} is neither 'isomer-NN' nor a monomer label")
        for col in ("level", "provenance"):
            if _blank(r[col]):
                problems.append(f"row {i}: {col} is empty")
        if not _blank(r["computed_on"]) and not _DATE_RE.match(str(r["computed_on"]).strip()):
            problems.append(f"row {i}: computed_on {r['computed_on']!r} is not an ISO date")
        for col in SCHEMA_NUMERIC_COLUMNS:
            v = r[col]
            if _blank(v):
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                problems.append(f"row {i}: {col} = {v!r} is not numeric")
                continue
            if col == "n_imag" and (fv < 0 or fv != int(fv)):
                problems.append(f"row {i}: n_imag = {v!r} is not a non-negative integer")
    dup = df.duplicated(list(SCHEMA_KEY), keep=False)
    if dup.any():
        keys = sorted({tuple(map(str, k)) for k in df.loc[dup, list(SCHEMA_KEY)].itertuples(index=False)})
        problems.append(f"duplicate (project, isomer, level) rows: {keys[:5]}")
    if problems:
        raise SchemaError(f"{len(problems)} schema problem(s):\n  " + "\n  ".join(problems[:40]))
    return df


def read_results(path: str | os.PathLike, *, check: bool = True, strict: bool = False) -> pd.DataFrame:
    """Read a schema CSV (UTF-8, comma, header). Text columns stay str ('' if empty);
    empty numeric cells become NaN."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={c: str for c in SCHEMA_TEXT_COLUMNS})
    for c in SCHEMA_TEXT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna("")
    if check:
        check_results(df, strict=strict)
    return df


def _canonical_cell(v: Any, col: str) -> str:
    """One text form per value, so a re-run writes byte-identical files: '' for empty,
    repr(float) for numbers (shortest round-trip form), an integer for n_imag."""
    if _blank(v):
        return ""
    if col in SCHEMA_NUMERIC_COLUMNS or isinstance(v, (float, np.floating)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v).strip()               # check_results(strict=True) reports it
        if col == "n_imag" and f.is_integer():
            return str(int(f))
        return repr(f)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v).strip()


def upsert_results(path: str | os.PathLike, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotent writer: rows of `new_rows` replace rows with the same (project, isomer, level)
    in the CSV at `path` (created if missing); all other rows are kept verbatim, so the file
    never loses rows and never gets duplicates. Columns: SCHEMA_COLUMNS first, extra columns
    after (union of old and new; missing cells empty). Rows are sorted by project, isomer
    (natural order), level. Values are written in one canonical text form (_canonical_cell),
    and the file is not touched at all when its content would not change, so running the
    same upsert twice leaves the file byte-identical. Returns the table written (all cells str;
    use read_results() for numbers).
    """
    check_results(new_rows, strict=True)
    p = Path(path)
    new = pd.DataFrame({c: [_canonical_cell(v, c) for v in new_rows[c].tolist()]
                        for c in new_rows.columns}, index=range(len(new_rows)))
    if p.exists():
        old = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        check_results(old, strict=True)
        new_keys = set(new[list(SCHEMA_KEY)].itertuples(index=False, name=None))
        keep = [k not in new_keys for k in old[list(SCHEMA_KEY)].itertuples(index=False, name=None)]
        merged = pd.concat([old[keep], new], ignore_index=True)
    else:
        merged = new
    extras = [c for c in merged.columns if c not in SCHEMA_COLUMNS]
    merged = merged[SCHEMA_COLUMNS + extras].fillna("").astype(str)
    order = sorted(range(len(merged)), key=lambda i: (merged["project"].iat[i],
                                                      _natural_key(merged["isomer"].iat[i]),
                                                      merged["level"].iat[i]))
    merged = merged.iloc[order].reset_index(drop=True)
    check_results(merged, strict=True)
    text = merged.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if p.exists() and p.read_bytes() == text:
        return merged                           # nothing changed: file left untouched
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(text)
    os.replace(tmp, p)
    return merged
