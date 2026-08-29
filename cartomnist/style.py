"""Sheet style: the figures are drawn as chart sheets, not as plots.

Two deliberate choices:

1. **Analytic charts use a validated categorical palette** (blue / orange / aqua
   for the three regimes), assigned in fixed slot order and always accompanied
   by direct labels — those three hues sit below 3:1 against the paper surface,
   so identity is never carried by colour alone.

2. **Plates are drawn like map sheets** — neatline border, engraved serif
   titles, a scale bar in millimetres of skin, a compass rose wherever a
   bearing is drawn, and a source-diagram inset. This is not decoration. The
   argument of the paper is that a benchmark should look like a chart, so the
   figures are built to the standard they are arguing for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
PAPER = "#f5f0e6"          # aged chart paper
PAPER_DEEP = "#ebe3d3"     # inset / panel fill
INK = "#14202b"            # engraving ink
INK_2 = "#54606b"          # secondary text
INK_3 = "#8b939b"          # muted / grid
RULE = "#c9bfa8"

# validated categorical slots 1-3 (all-pairs safe): the three regimes
REGIME_COLOR = {
    "naive": "#2a78d6",
    "generalized": "#eb6834",
    "native224": "#1baf7a",
}
REGIME_LABEL = {
    "naive": "naive resize 28²",
    "generalized": "generalized 28²",
    "native224": "native 224²",
}

# structure symbology — slots 1-5, only ever used in small multiples where each
# panel is titled, so the all-pairs cap does not apply
STRUCTURE_COLOR = {
    "pigment_network": "#2a78d6",
    "dots_globules": "#eb6834",
    "streaks": "#1baf7a",
    "vessels": "#eda100",
    "blue_white_veil": "#e87ba4",
}
STRUCTURE_GLYPH = {
    "pigment_network": "mesh",
    "dots_globules": "dots",
    "streaks": "streaks",
    "vessels": "vessels",
    "blue_white_veil": "veil",
}

GOOD, WARN, BAD = "#1baf7a", "#eda100", "#e34948"

# Physical scale: a dermoscopic field is ~20 mm across at 224 px.
FIELD_MM = 20.0


# --------------------------------------------------------------------------
def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
    })


def smallcaps(s: str) -> str:
    """Letter-spaced upper case, the closest matplotlib gets to engraved caps."""
    return " ".join(s.upper())


# --------------------------------------------------------------------------
# Sheet furniture
# --------------------------------------------------------------------------
def neatline(fig, pad: float = 0.012) -> None:
    """The double border of a printed chart sheet."""
    for i, (p, lw) in enumerate(((pad, 1.6), (pad + 0.008, 0.7))):
        fig.add_artist(Rectangle((p, p), 1 - 2 * p, 1 - 2 * p, transform=fig.transFigure,
                                 fill=False, edgecolor=INK if i == 0 else RULE,
                                 linewidth=lw, zorder=1000))


def sheet_title(fig, title: str, subtitle: str = "", plate: str = "") -> None:
    """A long, unwrapped subtitle set at fig-fraction x=0.5 renders as one
    line and can run into the corner plate label or off the frame entirely
    -- the same failure caption() already guards against. Wrap it the same
    way here.

    The title's vertical position used to be a fixed figure-fraction
    (0.985), which put the text a fixed FRACTION of the figure's height
    below the top edge. The neatline border is likewise placed a fixed
    fraction in from the edge. But the 15pt title's own height is fixed in
    POINTS, not in figure-fraction -- so on a short figure (small
    get_size_inches height), that fixed fraction of headroom is fewer
    absolute points than the glyphs need, and the bold title clips into the
    border above it. Compute the gap in points, then convert to this
    figure's own fraction, so short and tall figures get the same real
    clearance."""
    import textwrap
    fig_h_in = fig.get_size_inches()[1]
    top_gap_in = 24.0 / 72.0
    y = 1 - top_gap_in / fig_h_in
    if plate:
        fig.text(0.028, y, smallcaps(plate), fontsize=7.5, color=INK_3,
                 va="top", family="serif")
    fig.text(0.5, y, title, fontsize=15, color=INK, ha="center", va="top",
             family="serif", fontweight="bold")
    if subtitle:
        fontsize = 9
        inches = fig.get_size_inches()[0] * 0.86
        chars = max(40, int(inches / (fontsize * 0.0075)))
        wrapped = "\n".join(textwrap.wrap(subtitle, chars))
        title_gap_in = 27.0 / 72.0
        fig.text(0.5, y - title_gap_in / fig_h_in, wrapped, fontsize=fontsize,
                 color=INK_2, ha="center", va="top", family="serif",
                 style="italic", linespacing=1.4)


def caption(fig, text: str, y: float = 0.022, width_frac: float = 0.84,
            fontsize: float = 7.6) -> None:
    """Footer note, hard-wrapped to stay inside the neatline.

    matplotlib's `wrap=True` measures against the figure width, not against the
    drawn frame, so a centred caption happily runs out through the border. Wrap
    it explicitly instead, estimating the character budget from the figure's
    physical width.
    """
    import textwrap
    inches = fig.get_size_inches()[0] * width_frac
    chars = max(40, int(inches / (fontsize * 0.0075)))
    fig.text(0.5, y, "\n".join(textwrap.wrap(text, chars)), fontsize=fontsize,
             color=INK_2, ha="center", va="bottom", family="serif",
             linespacing=1.5)


def structure_key(fig, entries, y: float = 0.905, fontsize: float = 7.4) -> None:
    """Horizontal symbology key, drawn in the sheet header.

    Belongs to the figure rather than to a panel: a legend tucked inside one
    image panel covers the very data it is explaining.
    """
    n = len(entries)
    span = min(0.62, 0.1 * n)
    x0 = 0.5 - span / 2
    step = span / max(1, n - 1) if n > 1 else 0
    for i, (label, color) in enumerate(entries):
        x = x0 + i * step
        fig.add_artist(Rectangle((x - 0.011, y - 0.004), 0.009, 0.009,
                                 transform=fig.transFigure, facecolor=color,
                                 edgecolor=INK, lw=0.4, zorder=900))
        fig.text(x + 0.002, y, label, fontsize=fontsize, color=INK, va="center",
                 ha="left", family="serif")


def scalebar(ax, image_px: int, field_mm: float = FIELD_MM,
             bar_mm: float = 5.0, loc=(0.06, 0.06), color: str = "#ffffff") -> None:
    """A scale bar in millimetres of skin — the unit a clinician thinks in.

    Every plate that shows an image carries one. A benchmark that ships images
    with no physical scale is asking its users to reason about resolution
    without telling them what a pixel is.
    """
    px = image_px * (bar_mm / field_mm)
    x0 = loc[0] * image_px
    y0 = (1 - loc[1]) * image_px
    h = max(1.5, image_px * 0.012)
    n = 4
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * px / n, y0 - h), px / n, h,
                               facecolor=color if i % 2 else INK,
                               edgecolor=INK, linewidth=0.5, zorder=50))
    ax.text(x0 + px / 2, y0 - h * 2.1, f"{bar_mm:g} mm", ha="center", va="bottom",
            fontsize=6.5, color=color, zorder=51,
            path_effects=_halo())


def _halo(lw: float = 1.6, fg: str = INK):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground=fg)]


def compass(ax, size_px: float, loc=(0.88, 0.14), color: str = "#ffffff") -> None:
    """Compass rose. Present only where the plate draws bearings, because a
    compass on a sheet with no orientation information is a lie."""
    cx, cy = loc[0] * size_px, (1 - loc[1]) * size_px
    r = size_px * 0.055
    for ang, ln in ((90, 1.0), (270, 0.65), (0, 0.65), (180, 0.65)):
        a = np.radians(ang)
        ax.plot([cx, cx + r * ln * np.cos(a)], [cy, cy - r * ln * np.sin(a)],
                color=color, lw=1.1, zorder=50, path_effects=_halo(2.0))
    ax.add_patch(plt.Circle((cx, cy), r * 0.18, facecolor=color, edgecolor=INK,
                            lw=0.5, zorder=51))
    ax.text(cx, cy - r * 1.35, "N", ha="center", va="bottom", fontsize=6.5,
            color=color, zorder=51, path_effects=_halo())


def imshow_plate(ax, img: np.ndarray, title: str = "", subtitle: str = "",
                 border: str = INK, interpolation: str = "nearest"):
    """Show an image inside a ruled panel with an engraved caption."""
    ax.imshow(np.clip(img, 0, 1), interpolation=interpolation)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(border); s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=9, color=INK, pad=4)
    if subtitle:
        ax.text(0.5, -0.045, subtitle, transform=ax.transAxes, ha="center",
                va="top", fontsize=7, color=INK_2, style="italic")
    return ax


def pixel_grid(ax, n: int, extent_px: int, color: str = "#ffffff",
               alpha: float = 0.28) -> None:
    """Draw the target-grid cell boundaries over a native image.

    This is the single most useful thing to look at: it shows, literally, how
    much skin falls inside one benchmark pixel.
    """
    step = extent_px / n
    for i in range(1, n):
        ax.axhline(i * step - 0.5, color=color, lw=0.35, alpha=alpha, zorder=20)
        ax.axvline(i * step - 0.5, color=color, lw=0.35, alpha=alpha, zorder=20)


def direct_label(ax, x, y, text, color, dx=6, dy=0, fontsize=8.5, weight="bold"):
    """The relief rule: these hues are below 3:1 on paper, so every series that
    uses one carries a visible text label in the same colour family."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=fontsize, fontweight=weight,
                va="center", ha="left", family="serif")


def direct_labels_declutter(ax, entries, min_gap_frac=0.07, **kwargs) -> None:
    """direct_label for a group of series that share one edge of the axes.

    Two series can legitimately end at nearly the same value on real data
    even when a hand-picked smoke-test fixture never happens to produce
    that -- and a set of direct_label calls with dy=0 then stacks their
    text on top of each other. Sort by y and push apart anything closer
    than min_gap_frac of the axis range before labelling, so the group is
    readable regardless of how close the underlying values land.

    entries: sequence of (x, y, text, color).
    """
    ylo, yhi = ax.get_ylim()
    min_gap = min_gap_frac * (yhi - ylo)
    placed = sorted(entries, key=lambda e: e[1])
    for i in range(1, len(placed)):
        x1, y1, t1, c1 = placed[i]
        y0 = placed[i - 1][1]
        if y1 - y0 < min_gap:
            placed[i] = (x1, y0 + min_gap, t1, c1)
    for x, y, text, color in placed:
        direct_label(ax, x, y, text, color, **kwargs)


def legend_swatch(ax, entries: Sequence[tuple], loc=(0.02, 0.98), fontsize=7.5,
                  title: str = ""):
    """Hand-drawn legend box in the chart-sheet idiom."""
    x, y = loc
    if title:
        ax.text(x, y, smallcaps(title), transform=ax.transAxes, fontsize=6.8,
                color=INK_3, va="top", ha="left")
        y -= 0.055
    for label, color in entries:
        ax.add_patch(Rectangle((x, y - 0.028), 0.035, 0.024,
                               transform=ax.transAxes, facecolor=color,
                               edgecolor=INK, lw=0.4, zorder=40, clip_on=False))
        ax.text(x + 0.05, y - 0.016, label, transform=ax.transAxes,
                fontsize=fontsize, color=INK, va="center", ha="left", zorder=40)
        y -= 0.052


def save(fig, path: Path | str, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


# --------------------------------------------------------------------------
def structure_cmap(name: str):
    """One-hue sequential ramp from paper to the structure's colour."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        f"carto_{name}", [PAPER_DEEP, STRUCTURE_COLOR[name]], N=256)


def confidence_cmap():
    """Zones of confidence: red (unsurveyed) -> paper -> green (surveyed).

    Diverging, with a genuinely neutral midpoint, because the certificate has a
    meaningful zero: 0.5 is exactly at the detectability floor.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "carto_conf", [BAD, "#f0efec", GOOD], N=256)
