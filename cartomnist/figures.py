"""Every figure in the deliverable.

Each function returns the Path it wrote. Plates that show imagery show *real*
imagery — the point of the paper is what happens to actual dermoscopic pixels,
and a bar chart cannot make that argument.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Rectangle

from . import style as S
from .contracts import OPERATORS, NAIVE_RESIZE
from .data import CLASS_SHORT
from .filterbanks import STRUCTURE_NAMES, scale_periods


# ==========================================================================
# shared rendering helpers
# ==========================================================================
def render_symbol_plate(tensor: np.ndarray, spec, upscale: int = 12,
                        structures: Sequence[str] = tuple(STRUCTURE_NAMES),
                        draw_bearings: bool = True) -> tuple:
    """Render the symbol channels as a map: colour = family, tick = bearing.

    Returns (rgb_canvas, bearing_segments) where segments is a list of
    (x0, y0, x1, y1, colour, alpha) in canvas pixel coordinates.
    """
    t = np.asarray(tensor)
    n = t.shape[-1]
    size = n * upscale
    canvas = np.ones((size, size, 3), dtype=np.float32) * np.array(
        [int(S.PAPER_DEEP[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)

    segs = []
    for sname in structures:
        idxs = spec.groups[sname]
        d_idx = [i for i in idxs if spec.names[i].endswith("::density")]
        if not d_idx:
            continue
        d = np.asarray(t[d_idx[0]], dtype=np.float32)
        col = np.array([int(S.STRUCTURE_COLOR[sname][i:i + 2], 16) / 255
                        for i in (1, 3, 5)], dtype=np.float32)
        big = np.kron(d, np.ones((upscale, upscale), dtype=np.float32))
        a = np.clip(big, 0, 1)[..., None] * 0.55
        canvas = canvas * (1 - a) + col[None, None, :] * a

        if not draw_bearings:
            continue
        c_idx = [i for i in idxs if spec.names[i].endswith("::cos2theta")]
        s_idx = [i for i in idxs if spec.names[i].endswith("::sin2theta")]
        if not (c_idx and s_idx):
            continue
        c2 = np.asarray(t[c_idx[0]]) * 2.0 - 1.0
        s2 = np.asarray(t[s_idx[0]]) * 2.0 - 1.0
        mag = np.hypot(c2, s2)
        theta = 0.5 * np.arctan2(s2, c2)
        for i in range(n):
            for j in range(n):
                if mag[i, j] < 0.08 or d[i, j] < 0.18:
                    continue
                L = upscale * 0.42 * min(1.0, 0.4 + d[i, j])
                cx, cy = (j + 0.5) * upscale, (i + 0.5) * upscale
                dx, dy = L * np.cos(theta[i, j]), L * np.sin(theta[i, j])
                segs.append((cx - dx, cy - dy, cx + dx, cy + dy,
                             S.STRUCTURE_COLOR[sname],
                             float(np.clip(0.35 + 0.65 * mag[i, j], 0, 1))))
    return np.clip(canvas, 0, 1), segs


def draw_symbol_plate(ax, tensor, spec, upscale: int = 12, title: str = "",
                      subtitle: str = "", draw_bearings: bool = True):
    canvas, segs = render_symbol_plate(tensor, spec, upscale,
                                       draw_bearings=draw_bearings)
    S.imshow_plate(ax, canvas, title, subtitle)
    for x0, y0, x1, y1, col, alpha in segs:
        ax.plot([x0, x1], [y0, y1], color=col, lw=1.0, alpha=alpha,
                solid_capstyle="round", zorder=10)
    return ax


def upsample_nn(img_chw: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(img_chw)
    if a.ndim == 3 and a.shape[0] <= 32:
        a = np.transpose(a[:3], (1, 2, 0))
    return np.kron(a, np.ones((factor, factor, 1), dtype=np.float32))


def _to_hwc(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a[..., :3], 0, 1)


# ==========================================================================
# PLATE I — the legend
# ==========================================================================
def _glyph(ax, kind: str):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    c, c2 = S.INK, S.INK_3
    if kind == "selection":
        for i, x in enumerate(np.linspace(0.12, 0.88, 7)):
            keep = i in (0, 2, 5)
            ax.add_patch(Circle((x, 0.5), 0.055, facecolor=c if keep else "none",
                                edgecolor=c if keep else c2,
                                lw=1.0, ls="-" if keep else ":"))
    elif kind == "simplification":
        t = np.linspace(0, 1, 40)
        ax.plot(t, 0.72 + 0.09 * np.sin(t * 22) * np.exp(-t), color=c2, lw=0.9)
        ax.plot([0, 0.35, 0.68, 1.0], [0.3, 0.38, 0.26, 0.32], color=c, lw=1.6)
        ax.plot([0, 0.35, 0.68, 1.0], [0.3, 0.38, 0.26, 0.32], "o", color=c, ms=3)
    elif kind == "aggregation":
        rng = np.random.default_rng(3)
        for _ in range(11):
            x, y = rng.uniform(0.08, 0.42), rng.uniform(0.25, 0.75)
            ax.add_patch(Rectangle((x, y), 0.055, 0.045, facecolor=c2, edgecolor="none"))
        ax.add_patch(Polygon([[0.58, 0.28], [0.92, 0.33], [0.88, 0.74], [0.6, 0.7]],
                             facecolor=c, alpha=0.85, edgecolor=c))
        ax.annotate("", xy=(0.55, 0.5), xytext=(0.46, 0.5),
                    arrowprops=dict(arrowstyle="->", color=c2, lw=1.0))
    elif kind == "displacement":
        t = np.linspace(0.05, 0.95, 50)
        ax.plot(t, 0.42 + 0.05 * np.sin(t * 6), color="#2a78d6", lw=2.2)
        ax.plot(t, 0.42 + 0.05 * np.sin(t * 6), color=c2, lw=0.7, ls=":")
        ax.plot(t, 0.60 + 0.05 * np.sin(t * 6), color=c, lw=1.6)
        ax.annotate("", xy=(0.5, 0.585), xytext=(0.5, 0.475),
                    arrowprops=dict(arrowstyle="->", color=c2, lw=0.9))
    elif kind == "exaggeration":
        ax.plot([0.08, 0.42], [0.5, 0.5], color=c2, lw=0.6)
        ax.plot([0.58, 0.92], [0.5, 0.5], color=c, lw=4.5,
                solid_capstyle="butt")
        ax.text(0.25, 0.24, "true", ha="center", fontsize=6, color=c2)
        ax.text(0.75, 0.24, "drawn", ha="center", fontsize=6, color=c)
    elif kind == "symbolization":
        rng = np.random.default_rng(7)
        for _ in range(60):
            x, y = rng.uniform(0.06, 0.4), rng.uniform(0.2, 0.8)
            ax.plot([x, x + 0.02], [y, y + 0.015], color=c2, lw=0.4)
        ax.add_patch(Rectangle((0.58, 0.28), 0.34, 0.44, facecolor="none",
                               edgecolor=c, lw=0.8))
        for xx in np.linspace(0.62, 0.88, 4):
            ax.plot([xx, xx], [0.34, 0.66], color="#2a78d6", lw=1.4)
        ax.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                    arrowprops=dict(arrowstyle="->", color=c2, lw=1.0))
    elif kind == "source_diagram":
        cm = S.confidence_cmap()
        rng = np.random.default_rng(11)
        g = rng.random((5, 5)) * 0.8 + 0.1
        ax.imshow(g, cmap=cm, extent=(0.1, 0.9, 0.15, 0.85), vmin=0, vmax=1,
                  aspect="auto")
        ax.add_patch(Rectangle((0.1, 0.15), 0.8, 0.7, facecolor="none",
                               edgecolor=c, lw=1.0))
    elif kind == "naive":
        rng = np.random.default_rng(5)
        for _ in range(70):
            x, y = rng.uniform(0.06, 0.42), rng.uniform(0.2, 0.8)
            ax.plot([x, x + 0.02], [y, y + 0.015], color=c2, lw=0.4)
        ax.add_patch(Rectangle((0.58, 0.28), 0.34, 0.44, facecolor="#cfc7b3",
                               edgecolor=c, lw=0.8))
        ax.text(0.75, 0.5, "?", ha="center", va="center", fontsize=15, color=c)
        ax.annotate("", xy=(0.55, 0.5), xytext=(0.45, 0.5),
                    arrowprops=dict(arrowstyle="->", color=c2, lw=1.0))


def _wrap(text: str, width_in: float, fontsize: float) -> list:
    import textwrap
    chars = max(20, int(width_in / (fontsize * 0.0072)))
    return textwrap.wrap(text, chars) or [""]


def fig_legend(outdir: Path) -> Path:
    """Plate I, laid out by hand.

    matplotlib's `wrap=True` measures against the *figure*, not the axes it is
    drawn in, so contract text happily runs out through the side of its own
    card. Everything here is therefore wrapped explicitly and the card heights
    are derived from the resulting line counts, so no card can ever overflow
    however long a contract clause gets.
    """
    S.use_style()
    items = list(OPERATORS) + [NAIVE_RESIZE]

    FIG_W = 15.0
    MARGIN, GAP = 0.5, 0.45
    HEADER, FOOTER = 1.15, 0.85
    ROW_GAP = 0.3
    GLYPH_W, PAD = 1.0, 0.18
    FS_T, FS_A, FS_K, FS_B, FS_C = 9.0, 7.2, 6.4, 6.9, 6.2
    LH = 1.55 / 72.0                      # line height, inches per point

    card_w = (FIG_W - 2 * MARGIN - GAP) / 2
    text_w = card_w - GLYPH_W - 2 * PAD

    # ---- wrap everything first, then size the cards from the line counts
    blocks = []
    for c in items:
        b = {
            "c": c,
            "analogue": _wrap(c.cartographic_analogue, text_w, FS_A),
            "preserves": [_wrap("· " + x, text_w, FS_B) for x in c.preserves],
            "sacrifices": [_wrap("· " + x, text_w, FS_B) for x in c.sacrifices],
            "invariant": _wrap("Invariant: " + c.invariant, text_w, FS_C),
        }
        h = PAD * 2
        h += FS_T * LH * 1.9
        h += len(b["analogue"]) * FS_A * LH + 0.10
        h += FS_K * LH * 1.5
        h += sum(len(w) for w in b["preserves"]) * FS_B * LH + 0.07
        h += FS_K * LH * 1.5
        h += sum(len(w) for w in b["sacrifices"]) * FS_B * LH + 0.10
        h += len(b["invariant"]) * FS_C * LH
        b["h"] = h
        blocks.append(b)

    n_rows = (len(blocks) + 1) // 2
    row_h = [max(blocks[2 * r]["h"],
                 blocks[2 * r + 1]["h"] if 2 * r + 1 < len(blocks) else 0)
             for r in range(n_rows)]
    FIG_H = HEADER + FOOTER + sum(row_h) + ROW_GAP * (n_rows - 1)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y_cursor = FIG_H - HEADER

    for r in range(n_rows):
        h = row_h[r]
        y_cursor -= h
        for col in range(2):
            k = 2 * r + col
            if k >= len(blocks):
                continue
            b, c = blocks[k], blocks[2 * r + col]["c"]
            x = MARGIN + col * (card_w + GAP)
            is_naive = c.operator == "naive_resize"

            ax = fig.add_axes([x / FIG_W, y_cursor / FIG_H,
                               card_w / FIG_W, h / FIG_H])
            ax.set_xlim(0, card_w); ax.set_ylim(0, h)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(S.BAD if is_naive else S.RULE)
                sp.set_linewidth(1.0)
            ax.set_facecolor("#f2e5e3" if is_naive else S.PAPER_DEEP)

            gax = fig.add_axes([(x + PAD) / FIG_W,
                                (y_cursor + h / 2 - GLYPH_W / 2) / FIG_H,
                                GLYPH_W * 0.86 / FIG_W, GLYPH_W * 0.86 / FIG_H])
            gax.patch.set_alpha(0.0)
            _glyph(gax, "naive" if is_naive else c.operator)

            tx = GLYPH_W + PAD
            ty = h - PAD

            def put(lines, fs, color, weight="normal", style="normal", lead=1.0):
                nonlocal ty
                for ln in lines:
                    ty -= fs * LH * lead
                    ax.text(tx, ty, ln, fontsize=fs, color=color, va="baseline",
                            ha="left", fontweight=weight, style=style)

            put([S.smallcaps(c.operator.replace("_", " "))], FS_T,
                S.BAD if is_naive else S.INK, weight="bold", lead=1.5)
            ty -= 0.04
            put(b["analogue"], FS_A, S.INK_2, style="italic")
            ty -= 0.08
            put(["PRESERVES"], FS_K, S.GOOD, weight="bold", lead=1.45)
            for w in b["preserves"]:
                put(w, FS_B, S.INK)
            ty -= 0.06
            put(["SACRIFICES"], FS_K, S.BAD, weight="bold", lead=1.45)
            for w in b["sacrifices"]:
                put(w, FS_B, S.INK)
            ty -= 0.07
            put(b["invariant"], FS_C, S.INK_3, style="italic")
        y_cursor -= ROW_GAP

    S.sheet_title(fig, "The Legend",
                  "Named generalization operators and their fidelity contracts. The final "
                  "card is the baseline every MedMNIST-style benchmark ships without one.",
                  plate="Figure 2")
    S.caption(fig, "Vocabulary after Ratajski (1967), Brassel & Weibel (1988), "
                   "McMaster & Shea (1992). Source-diagram practice after IHO S-4 / S-57. "
                   "Every invariant named here is checked by tests/test_operators.py.",
              y=0.020)
    S.neatline(fig)
    return S.save(fig, outdir / "fig01_legend.png")


# ==========================================================================
# PLATE II — the three regimes, side by side, on real lesions
# ==========================================================================
def fig_three_regimes(outdir: Path, native: np.ndarray, naive: np.ndarray,
                      gen: np.ndarray, spec, labels: np.ndarray,
                      idx: Sequence[int], target: int = 28) -> Path:
    S.use_style()
    n = len(idx)
    fig = plt.figure(figsize=(14.2, 2.55 * n + 2.5))
    top = 1.0 - 1.55 / (2.55 * n + 2.5)
    gs = GridSpec(n, 5, figure=fig, hspace=0.16, wspace=0.06,
                  left=0.062, right=0.975, top=top, bottom=0.085)

    heads = ["native 224²  (the survey)", "naive resize 28²", "generalized 28²  (base)",
             "generalized 28²  (symbols)", "source diagram"]
    subs = ["what the camera recorded", "what the benchmark ships",
            "aggregation only", "symbolization + bearings", "zones of confidence"]

    for r, i in enumerate(idx):
        img = _to_hwc(native[i])
        ax = fig.add_subplot(gs[r, 0])
        S.imshow_plate(ax, img, heads[0] if r == 0 else "",
                       subs[0] if r == 0 else "", interpolation="bilinear")
        S.pixel_grid(ax, target, img.shape[0])
        S.scalebar(ax, img.shape[0])
        S.compass(ax, img.shape[0])
        ax.set_ylabel(CLASS_SHORT[int(labels[i])], fontsize=9, color=S.INK,
                      rotation=0, ha="right", va="center", labelpad=14,
                      fontweight="bold")

        ax = fig.add_subplot(gs[r, 1])
        S.imshow_plate(ax, upsample_nn(naive[i], 12), heads[1] if r == 0 else "",
                       subs[1] if r == 0 else "")
        ax = fig.add_subplot(gs[r, 2])
        S.imshow_plate(ax, upsample_nn(gen[i][:3], 12), heads[2] if r == 0 else "",
                       subs[2] if r == 0 else "")
        ax = fig.add_subplot(gs[r, 3])
        draw_symbol_plate(ax, gen[i], spec, 12, heads[3] if r == 0 else "",
                          subs[3] if r == 0 else "")

        ax = fig.add_subplot(gs[r, 4])
        cert_idx = [j for j in spec.groups["source_diagram"]
                    if spec.names[j].endswith("::certificate")]
        cert = np.asarray(gen[i][cert_idx[0]])
        ax.imshow(np.kron(cert, np.ones((12, 12))), cmap=S.confidence_cmap(),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(S.INK); s.set_linewidth(1.0)
        if r == 0:
            ax.set_title(heads[4], fontsize=9, color=S.INK, pad=4)
            ax.text(0.5, -0.045, subs[4], transform=ax.transAxes, ha="center",
                    va="top", fontsize=7, color=S.INK_2, style="italic")

    S.sheet_title(fig, "Three Ways to Publish the Same Lesion at 28×28",
                  "Column 1 is the source survey with the 28×28 cell boundaries drawn on it: "
                  "each benchmark pixel is ~0.71 mm of skin.",
                  plate="Figure 3")
    S.structure_key(fig, [(k.replace("_", " "), v)
                          for k, v in S.STRUCTURE_COLOR.items()], y=top + 0.022)
    S.caption(fig, "Naive resize discards every structure whose period is below "
                   "2 benchmark pixels. Generalization measures those structures at native "
                   "resolution and draws them as symbols instead — then states, per cell, "
                   "whether the measurement was supportable.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig02_three_regimes.png")


# ==========================================================================
# PLATE III — the survey sheet for one lesion
# ==========================================================================
def fig_symbol_atlas(outdir: Path, native_img: np.ndarray, tensor: np.ndarray,
                     intermediates: Dict, spec, label: str,
                     target: int = 28) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(14.6, 7.4))
    gs = GridSpec(2, 6, figure=fig, hspace=0.2, wspace=0.07,
                  left=0.03, right=0.975, top=0.865, bottom=0.10)

    img = _to_hwc(native_img)
    ax = fig.add_subplot(gs[:, 0])
    S.imshow_plate(ax, img, "the survey", f"native 224² · {label}",
                   interpolation="bilinear")
    S.scalebar(ax, img.shape[0])
    S.compass(ax, img.shape[0])
    S.pixel_grid(ax, target, img.shape[0])

    surv = intermediates["survey"]
    periods = scale_periods(img.shape[0])
    for k, sname in enumerate(STRUCTURE_NAMES):
        ax = fig.add_subplot(gs[0, k + 1])
        d = np.asarray(surv[sname]["density"])
        ax.imshow(d, cmap=S.structure_cmap(sname), vmin=0, vmax=1,
                  interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(S.STRUCTURE_COLOR[sname]); s.set_linewidth(1.2)
        ax.set_title(sname.replace("_", " "), fontsize=8.4,
                     color=S.STRUCTURE_COLOR[sname], pad=4)
        ax.text(0.5, -0.05, f"period ≈ {periods[sname]:.0f} px "
                            f"({periods[sname] / img.shape[0] * S.FIELD_MM:.2f} mm)",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.6,
                color=S.INK_2)

        axb = fig.add_subplot(gs[1, k + 1])
        d_idx = [i for i in spec.groups[sname] if spec.names[i].endswith("::density")]
        sym = np.asarray(tensor[d_idx[0]])
        axb.imshow(np.kron(sym, np.ones((12, 12))), cmap=S.structure_cmap(sname),
                   vmin=0, vmax=1, interpolation="nearest")
        axb.set_xticks([]); axb.set_yticks([]); axb.grid(False)
        for s in axb.spines.values():
            s.set_visible(True); s.set_color(S.INK); s.set_linewidth(1.0)
        nyq = periods[sname] / (2 * img.shape[0] / target)
        verdict = "survives resize" if nyq >= 1 else "sub-Nyquist — symbolized"
        axb.text(0.5, -0.05, f"{verdict}\n(p/2Δ = {nyq:.2f})", transform=axb.transAxes,
                 ha="center", va="top", fontsize=6.6,
                 color=S.GOOD if nyq >= 1 else S.BAD)

    fig.text(0.032, 0.545, S.smallcaps("native response"), fontsize=6.8,
             color=S.INK_3, rotation=90, va="center")
    fig.text(0.032, 0.28, S.smallcaps("28² symbol"), fontsize=6.8,
             color=S.INK_3, rotation=90, va="center")

    S.sheet_title(fig, "Survey Sheet: From Instrument Response to Symbol",
                  "Top row: what the fixed filter banks measure at native resolution. "
                  "Bottom row: the same measurement aggregated, displaced and exaggerated onto the 28×28 sheet.",
                  plate="Figure 4")
    S.caption(fig, "p/2Δ is the structure's period divided by twice the benchmark cell width — "
                   "the Nyquist ratio. Below 1, no resampling scheme can represent the "
                   "structure; it must be symbolized or lost.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig03_symbol_atlas.png")


# ==========================================================================
# PLATE IV — source diagrams
# ==========================================================================
def fig_source_diagram(outdir: Path, native: np.ndarray, gen: np.ndarray, spec,
                       labels: np.ndarray, idx: Sequence[int],
                       nyquist: Dict[str, Dict[str, float]],
                       target: int = 28) -> Path:
    S.use_style()
    n = len(idx)
    fig = plt.figure(figsize=(13.4, 3.05 * 2 + 4.0))
    gs = GridSpec(4, n, figure=fig, hspace=0.30, wspace=0.07,
                  left=0.04, right=0.975, top=0.875, bottom=0.075,
                  height_ratios=[1, 1, 0.10, 0.95])
    cert_idx = [j for j in spec.groups["source_diagram"]
                if spec.names[j].endswith("::certificate")][0]
    cm = S.confidence_cmap()

    for k, i in enumerate(idx):
        img = _to_hwc(native[i])
        ax = fig.add_subplot(gs[0, k])
        S.imshow_plate(ax, img, CLASS_SHORT[int(labels[i])], "",
                       interpolation="bilinear")
        S.scalebar(ax, img.shape[0])

        cert = np.asarray(gen[i][cert_idx])
        big = np.kron(cert, np.ones((img.shape[0] // target,) * 2))
        ax.imshow(big, cmap=cm, vmin=0, vmax=1, alpha=0.55,
                  extent=(-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5),
                  zorder=5)
        S.pixel_grid(ax, target, img.shape[0], alpha=0.18)

        axb = fig.add_subplot(gs[1, k])
        im = axb.imshow(np.kron(cert, np.ones((12, 12))), cmap=cm, vmin=0, vmax=1,
                        interpolation="nearest")
        axb.set_xticks([]); axb.set_yticks([]); axb.grid(False)
        for sp in axb.spines.values():
            sp.set_visible(True); sp.set_color(S.INK); sp.set_linewidth(1.0)
        axb.text(0.5, -0.05, f"mean confidence {cert.mean():.2f} · "
                             f"{100 * (cert < 0.5).mean():.0f}% below floor",
                 transform=axb.transAxes, ha="center", va="top", fontsize=7,
                 color=S.INK_2)

    cax = fig.add_subplot(gs[2, 1:max(2, n - 1)] if n > 2 else gs[2, :])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["unsurveyed\n(symbol is extrapolation)", "at the floor",
                       "surveyed\n(symbol is supported)"])
    cb.ax.tick_params(labelsize=6.6, colors=S.INK_2, length=0)
    cb.outline.set_edgecolor(S.INK)

    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    rows = [(k, v["period_native_px"], v["target_cell_px"], v["nyquist_ratio"],
             v["survives_naive_resize"]) for k, v in nyquist.items()]
    ax.text(0.5, 1.02, S.smallcaps("Nyquist audit of the 224 → 28 reduction"),
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4,
            color=S.INK, fontweight="bold")
    hdr = ["structure", "period (native px)", "cell width (px)", "p / 2Δ",
           "survives naive resize?"]
    xs = [0.07, 0.32, 0.50, 0.65, 0.80]
    for x, h in zip(xs, hdr):
        ax.text(x, 0.86, h, transform=ax.transAxes, fontsize=7.2, color=S.INK_3,
                ha="left" if x < 0.3 else "center", va="top")
    ax.plot([0.05, 0.95], [0.815, 0.815], transform=ax.transAxes, color=S.INK, lw=0.9)
    y = 0.74
    for name, p, cell, ratio, ok in rows:
        ax.text(xs[0], y, name.replace("_", " "), transform=ax.transAxes,
                fontsize=7.4, color=S.STRUCTURE_COLOR.get(name, S.INK),
                va="top", fontweight="bold")
        for x, v in zip(xs[1:4], (f"{p:.1f}", f"{cell:.1f}", f"{ratio:.2f}")):
            ax.text(x, y, v, transform=ax.transAxes, fontsize=7.4, color=S.INK,
                    ha="center", va="top")
        ax.text(xs[4], y, "yes" if ok else "NO — lost", transform=ax.transAxes,
                fontsize=7.4, ha="center", va="top",
                color=S.GOOD if ok else S.BAD, fontweight="bold")
        y -= 0.135

    S.sheet_title(fig, "The Source Diagram",
                  "Per-cell zones of confidence, computed from the native image alone — "
                  "no labels, available at test time.",
                  plate="Figure 5")
    S.caption(fig, "A navigator has been able to read this inset since the nineteenth century. "
                   "It is the same object as the abstention model: where the survey did not "
                   "support the symbol, the chart says so instead of guessing.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig04_source_diagram.png")


# ==========================================================================
# PLATE V — the Mercator result
# ==========================================================================
def fig_mercator(outdir: Path, mres: List, bin_labels: Sequence[str],
                 dyn_range: Dict[str, Dict[str, Dict[str, float]]],
                 native: np.ndarray, ita: np.ndarray,
                 example_idx: Dict[str, int]) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.28,
                  left=0.062, right=0.975, top=0.87, bottom=0.135,
                  height_ratios=[1.15, 1])

    regimes = ["naive", "generalized_ef", "generalized_abs"]
    rlabel = {"naive": "naive resize", "generalized_ef": "generalized (equal fidelity)",
              "generalized_abs": "generalized (absolute scale)"}
    rcolor = {"naive": "#2a78d6", "generalized_ef": "#eb6834",
              "generalized_abs": "#1baf7a"}

    # ---- panel A: retention vs ITA stratum, averaged over structures
    ax = fig.add_subplot(gs[0, :2])
    labs = list(bin_labels)
    x = np.arange(len(labs))
    for r in regimes:
        vals, sems = [], []
        for lab in labs:
            v = [m.per_bin_mean.get(lab, np.nan) for m in mres if m.regime == r]
            s = [m.per_bin_sem.get(lab, np.nan) for m in mres if m.regime == r]
            vals.append(np.nanmean(v) if len(v) else np.nan)
            sems.append(np.nanmean(s) if len(s) else np.nan)
        vals, sems = np.asarray(vals), np.asarray(sems)
        ax.plot(x, vals, "-o", color=rcolor[r], lw=2.0, ms=6,
                markeredgecolor=S.PAPER, markeredgewidth=1.2, zorder=5)
        ax.fill_between(x, vals - sems, vals + sems, color=rcolor[r], alpha=0.14,
                        lw=0)
        if np.isfinite(vals).any():
            j = int(np.flatnonzero(np.isfinite(vals))[-1])
            S.direct_label(ax, x[j], vals[j], rlabel[r], rcolor[r])
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(" (", "\n(") for lab in labs], fontsize=7.4)
    ax.set_ylabel("structure retention  (R² of native map)", fontsize=8.5)
    ax.set_xlabel("individual typology angle — darker skin ← → lighter skin", fontsize=8.5)
    ax.set_title("A · How much of the native structure survives, by pigmentation",
                 loc="left", fontsize=9.6)
    ax.set_xlim(-0.35, len(labs) - 0.35 + 1.3)

    # ---- panel B: distortion spread per regime (the Mercator number)
    ax = fig.add_subplot(gs[0, 2])
    spreads = []
    for r in regimes:
        v = [m.spread for m in mres if m.regime == r and np.isfinite(m.spread)]
        spreads.append(float(np.mean(v)) if v else np.nan)
    bars = ax.barh(np.arange(len(regimes)), spreads,
                   color=[rcolor[r] for r in regimes], height=0.52,
                   edgecolor=S.PAPER, linewidth=2)
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_yticklabels([rlabel[r].replace(" (", "\n(") for r in regimes], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("retention spread across ITA strata\n(max − min; 0 = equal fidelity)",
                  fontsize=8)
    ax.set_title("B · The distortion", loc="left", fontsize=9.6)
    for i, (b, v) in enumerate(zip(bars, spreads)):
        if np.isfinite(v):
            ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8,
                    color=S.INK, fontweight="bold")
    ax.grid(axis="y", visible=False)

    # ---- panel C/D: real images at the extremes
    for k, (key, title) in enumerate((("dark", "lowest ITA in the test split"),
                                      ("light", "highest ITA in the test split"))):
        i = example_idx.get(key)
        ax = fig.add_subplot(gs[1, k])
        if i is None:
            ax.axis("off"); continue
        img = _to_hwc(native[i])
        S.imshow_plate(ax, img, title, f"ITA = {ita[i]:.0f}°",
                       interpolation="bilinear")
        S.scalebar(ax, img.shape[0])
        S.pixel_grid(ax, 28, img.shape[0])

    # ---- panel E: symbol dynamic range
    ax = fig.add_subplot(gs[1, 2])
    for r, key in (("generalized_ef", "equal fidelity"),
                   ("generalized_abs", "absolute scale")):
        d = dyn_range.get(r, {})
        vals = []
        for lab in labs:
            v = [d[s][lab] for s in d if lab in d[s]]
            vals.append(np.mean(v) if v else np.nan)
        ax.plot(x, vals, "-o", color=rcolor[r], lw=2.0, ms=5,
                markeredgecolor=S.PAPER, markeredgewidth=1.1)
        if np.isfinite(vals).any():
            j = int(np.flatnonzero(np.isfinite(vals))[-1])
            S.direct_label(ax, x[j], vals[j], key, rcolor[r], fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels([f"q{i+1}" for i in range(len(labs))],
                                         fontsize=7.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("symbol channel p95 (usable range)", fontsize=8)
    ax.set_xlabel("ITA stratum", fontsize=8)
    ax.set_title("C · Why", loc="left", fontsize=9.6)
    ax.set_xlim(-0.3, len(labs) - 0.3 + 1.1)

    S.sheet_title(fig, "The Mercator Result",
                  "A uniform rule applied to a non-uniform world produces non-uniform error — "
                  "and says nothing about it.",
                  plate="Figure 6")
    S.caption(fig, "Retention is measured without a classifier: a ridge probe fitted on the "
                   "training split reconstructs the native structure map from each regime, and "
                   "is scored per test image. Equal fidelity is a design decision inside the "
                   "framework, not an automatic property of it — the third arm shows the "
                   "distortion returning when the normalisation is made global.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig05_mercator.png")


# ==========================================================================
# PLATE VI — reliability diagrams
# ==========================================================================
def fig_reliability(outdir: Path, rel: Dict[str, Dict], evals: Dict[str, object]) -> Path:
    S.use_style()
    keys = [k for k in ("naive", "generalized", "native224") if k in rel]
    fig = plt.figure(figsize=(4.5 * len(keys), 6.3))
    gs = GridSpec(2, len(keys), figure=fig, hspace=0.1, wspace=0.22,
                  left=0.07, right=0.975, top=0.80, bottom=0.16,
                  height_ratios=[3, 1])

    for k, key in enumerate(keys):
        r = rel[key]
        col = S.REGIME_COLOR[key]
        ev = evals[key]
        ax = fig.add_subplot(gs[0, k])
        ax.plot([0, 1], [0, 1], color=S.INK_3, lw=1.0, ls=(0, (4, 3)), zorder=1)
        m = r["counts"] > 0
        ax.bar(r["bin_centers"][m], r["accuracy"][m],
               width=(r["bin_edges"][1] - r["bin_edges"][0]) * 0.86,
               color=col, edgecolor=S.PAPER, linewidth=2, zorder=3)
        ax.plot(r["confidence"][m], r["accuracy"][m], "-o", color=S.INK, lw=1.2,
                ms=4, zorder=5, markeredgecolor=S.PAPER, markeredgewidth=1)
        for c, a, cf in zip(r["bin_centers"][m], r["accuracy"][m], r["confidence"][m]):
            ax.plot([cf, cf], [a, cf], color=S.BAD, lw=1.0, alpha=0.7, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{S.REGIME_LABEL[key]}", loc="left", fontsize=9.8, color=col,
                     fontweight="bold")
        ax.set_xticklabels([])
        if k == 0:
            ax.set_ylabel("observed accuracy", fontsize=8.5)
        ax.text(0.04, 0.955, f"ECE {getattr(ev, 'ece'):.4f}\n"
                             f"(uncal. {getattr(ev, 'ece_uncalibrated'):.4f})\n"
                             f"MCE {getattr(ev, 'mce'):.3f}\n"
                             f"Brier {getattr(ev, 'brier'):.3f}\n"
                             f"T = {getattr(ev, 'temperature'):.2f}",
                transform=ax.transAxes, fontsize=7.2, va="top", color=S.INK,
                bbox=dict(facecolor=S.PAPER, edgecolor=S.RULE, boxstyle="square,pad=0.4"))

        axh = fig.add_subplot(gs[1, k])
        axh.bar(r["bin_centers"], r["counts"],
                width=(r["bin_edges"][1] - r["bin_edges"][0]) * 0.86,
                color=S.INK_3, edgecolor=S.PAPER, linewidth=1.5)
        axh.set_xlim(0, 1)
        axh.set_xlabel("predicted confidence", fontsize=8.5)
        if k == 0:
            axh.set_ylabel("n", fontsize=8.5)

    S.sheet_title(fig, "Reliability Diagrams",
                  "Ship one with every benchmark. Red rungs are the calibration gap; "
                  "temperature fitted on validation only.",
                  plate="Figure 7")
    S.caption(fig, "A model that is accurate and badly calibrated is a chart with correct "
                   "soundings and no datum. Both the pre- and post-temperature ECE are "
                   "reported so the scaling cannot hide a miscalibrated model.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig06_reliability.png")


# ==========================================================================
# PLATE VII — Töpfer
# ==========================================================================
def fig_topfer(outdir: Path, law: Dict) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(12.4, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24, left=0.07, right=0.975,
                  top=0.74, bottom=0.20)

    curves = law["curves"]
    ax = fig.add_subplot(gs[0])
    ramp = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"]
    res = sorted(curves)
    for i, r in enumerate(res):
        c = ramp[min(i, len(ramp) - 1)]
        ks, aucs = curves[r]["k"], curves[r]["auc"]
        ax.plot(ks, aucs, "-o", color=c, lw=2.0, ms=5,
                markeredgecolor=S.PAPER, markeredgewidth=1.1)
        S.direct_label(ax, ks[-1], aucs[-1], f"{r}²", c)
        kstar = law["kstar"].get(r)
        if kstar is not None and kstar in ks:
            ax.plot([kstar], [aucs[ks.index(kstar)]], "*", color=S.INK, ms=13,
                    zorder=8)
    ax.set_xlabel("K — symbolized structure families retained", fontsize=8.5)
    ax.set_ylabel("test macro AUC", fontsize=8.5)
    ax.set_title("A · Diminishing returns at each scale", loc="left", fontsize=9.8)
    ax.set_xlim(-0.3, max(max(c["k"]) for c in curves.values()) + 1.3)

    ax = fig.add_subplot(gs[1])
    rs = np.array(sorted(law["kstar"]), dtype=float)
    ks = np.array([law["kstar"][int(r)] for r in rs], dtype=float)
    if len(rs) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no resolution produced a usable K*\n"
                          "(every symbol budget scored within noise of the others)",
                ha="center", va="center", fontsize=9, color=S.INK_2)
        S.sheet_title(fig, "The Radical Law, Applied to Benchmarks", "", plate="Figure 8")
        S.neatline(fig)
        return S.save(fig, outdir / "fig07_topfer.png")
    ax.plot(rs, ks, "o", color="#eb6834", ms=9, markeredgecolor=S.PAPER,
            markeredgewidth=1.4, zorder=6)
    S.direct_label(ax, rs[-1], ks[-1], "measured K*", "#eb6834")

    tc = law["topfer_curve"]
    if tc:
        rr = np.linspace(min(rs), max(rs), 100)
        anchor_r, anchor_k = float(rs[-1]), float(ks[-1])
        ax.plot(rr, anchor_k * np.sqrt(rr / anchor_r), color="#2a78d6", lw=2.0,
                zorder=4)
        S.direct_label(ax, rr[0], anchor_k * np.sqrt(rr[0] / anchor_r),
                       "Töpfer  n ∝ √r", "#2a78d6", dx=6, dy=-12)
    rr = np.linspace(min(rs), max(rs), 100)
    if np.isfinite(law["alpha"]):
        ax.plot(rr, np.exp(law["log_intercept"]) * rr ** law["alpha"],
                color=S.INK, lw=1.2, ls=(0, (5, 3)), zorder=5)
        ax.text(0.03, 0.94, f"fitted  K* ∝ r^{law['alpha']:.2f}\n"
                            f"Töpfer predicts  r^0.50",
                transform=ax.transAxes, fontsize=8, va="top", color=S.INK,
                bbox=dict(facecolor=S.PAPER, edgecolor=S.RULE,
                          boxstyle="square,pad=0.4"))
    else:
        # K* = 0 at too many resolutions for a log fit; say so on the plate
        # rather than drawing nothing and letting the reader assume a null.
        if np.isfinite(law.get("alpha_shifted", np.nan)):
            ax.plot(rr, np.exp(law["log_intercept_shifted"])
                    * rr ** law["alpha_shifted"] - 1.0,
                    color=S.INK, lw=1.2, ls=(0, (5, 3)), zorder=5)
        ax.text(0.03, 0.94, "unshifted power law undefined\n"
                            f"(K* > 0 at only "
                            f"{law.get('n_resolutions_with_positive_kstar', 0)} "
                            "resolution(s))\nshifted fit on K*+1 shown dashed",
                transform=ax.transAxes, fontsize=7.6, va="top", color=S.BAD,
                bbox=dict(facecolor=S.PAPER, edgecolor=S.BAD,
                          boxstyle="square,pad=0.4"))
    ax.set_xlabel("benchmark grid resolution r (px)", fontsize=8.5)
    ax.set_ylabel("K*  — symbol budget", fontsize=8.5)
    ax.set_title("B · A scale law for benchmark design", loc="left", fontsize=9.8)

    S.sheet_title(fig, "The Radical Law, Applied to Benchmarks",
                  "Töpfer & Pillewizer (1966) computed how many features a map should keep "
                  "when its scale drops. The same question has an answer for benchmarks.",
                  plate="Figure 8")
    S.caption(fig, f"Star marks K*, the first K within {law['frac_of_max']:.0%} of the best AUC at "
                   "that scale. If K* tracks √r, a benchmark designer can compute the symbol budget "
                   "for a chosen resolution rather than guessing it. A steeper exponent means low-"
                   "resolution grids saturate on symbols faster than cartographic practice predicts.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig07_topfer.png")


# ==========================================================================
# PLATE VIII — the certificate as an abstention model
# ==========================================================================
def fig_certificate(outdir: Path, curves: Dict[str, Dict]) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(11.6, 5.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24, left=0.07, right=0.975,
                  top=0.74, bottom=0.19)
    colors = {"source certificate": "#eb6834", "softmax confidence": "#2a78d6",
              "deep ensemble disagreement": "#1baf7a",
              "MC-dropout uncertainty": "#9b59b6", "random": S.INK_3}

    ax = fig.add_subplot(gs[0])
    for name, c in curves.items():
        col = colors.get(name, S.INK)
        ax.plot(c["coverage"], c["balanced_risk"], lw=2.0, color=col,
                ls=(0, (4, 3)) if name == "random" else "-")

    # Labels are stacked in a fixed-position block in the top-left corner,
    # ranked by each curve's value at the leftmost coverage point, rather
    # than pinned to that value's actual height: once more than two or three
    # trust signals are compared, curves can start anywhere in the axes
    # (including near the bottom edge), and offsetting a label from its own
    # data point can push it off the plot into the caption below. A fixed
    # block avoids both the overlap and the off-plot cases regardless of how
    # the curves happen to be arranged.
    ylo, yhi = ax.get_ylim()
    y0 = yhi - 0.06 * (yhi - ylo)
    step = 0.075 * (yhi - ylo)
    x0 = ax.get_xlim()[0] + 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ranked = sorted(curves.items(), key=lambda kv: -kv[1]["balanced_risk"][0])
    for rank, (name, c) in enumerate(ranked):
        col = colors.get(name, S.INK)
        ax.annotate(name, xy=(x0, y0 - rank * step), color=col, fontsize=7.6,
                   fontweight="bold", va="center", ha="left", family="serif")
    ax.set_xlabel("coverage — fraction of the test set kept", fontsize=8.5)
    ax.set_ylabel("balanced risk on the kept fraction", fontsize=8.5)
    ax.set_title("A · Coverage–risk", loc="left", fontsize=9.8)

    ax = fig.add_subplot(gs[1])
    names = list(curves)
    vals = [curves[n]["balanced_aurc"] for n in names]
    ax.barh(np.arange(len(names)), vals,
            color=[colors.get(n, S.INK) for n in names], height=0.5,
            edgecolor=S.PAPER, linewidth=2)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8, fontweight="bold",
                color=S.INK)
    ax.set_xlabel("area under the balanced coverage–risk curve (lower is better)",
                  fontsize=8)
    ax.set_title("B · Summary", loc="left", fontsize=9.8)

    S.sheet_title(fig, "The Certificate Is the Abstention Model",
                  "Ranking by the source diagram — computed from the image alone, before any "
                  "prediction — against ranking by the model's own confidence.",
                  plate="Figure 9")
    S.caption(fig, "The certificate is available for an unlabelled image and does not depend "
                   "on the classifier, so it can be published with the dataset rather than "
                   "refitted per model. Where it beats softmax confidence, the dataset itself "
                   "knew which images it could not support.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig08_certificate.png")


# ==========================================================================
# PLATE IX — rare classes and the headline recovery
# ==========================================================================
def fig_rare_class(outdir: Path, evals: Dict[str, object], rare: Sequence[int],
                   class_counts: Dict[str, int]) -> Path:
    S.use_style()
    keys = [k for k in ("naive", "generalized", "native224") if k in evals]
    fig = plt.figure(figsize=(13.2, 5.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22, left=0.06, right=0.975,
                  top=0.74, bottom=0.16, width_ratios=[1.7, 1])

    ax = fig.add_subplot(gs[0])
    nC = len(getattr(evals[keys[0]], "per_class_recall"))
    x = np.arange(nC)
    w = 0.8 / len(keys)
    for j, key in enumerate(keys):
        v = np.asarray(getattr(evals[key], "per_class_recall"), dtype=float)
        ax.bar(x + (j - (len(keys) - 1) / 2) * w, v, width=w * 0.9,
               color=S.REGIME_COLOR[key], edgecolor=S.PAPER, linewidth=2,
               zorder=3)
    for c in rare:
        ax.axvspan(c - 0.45, c + 0.45, color="#e3d9c2", alpha=0.5, zorder=0)
    # Labels sit at fixed axes-fraction positions in the top-right margin
    # rather than pinned to the last class's bar height: a rare class can
    # legitimately have near-zero recall in every regime, and a label
    # offset from that near-zero point would land on the x-tick labels
    # below the axes instead of beside the bar.
    for j, key in enumerate(keys):
        ax.text(0.985, 0.95 - j * 0.075, S.REGIME_LABEL[key],
                transform=ax.transAxes, color=S.REGIME_COLOR[key],
                fontsize=7.6, fontweight="bold", ha="right", va="center",
                family="serif")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASS_SHORT[i]}\nn={class_counts.get(CLASS_SHORT[i], 0)}"
                        for i in range(nC)], fontsize=7.6)
    ax.set_ylabel("per-class recall (test)", fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.6, nC - 0.4 + 1.6)
    ax.set_title("A · Rare classes are where resolution was supposed to matter "
                 "(shaded = rare in train)", loc="left", fontsize=9.6)

    ax = fig.add_subplot(gs[1])
    metrics = [("rare_recall", "rare-class recall"),
               ("balanced_accuracy", "balanced accuracy"),
               ("macro_auc", "macro AUC"),
               ("ita_gap", "ITA gap (lower better)")]
    y = np.arange(len(metrics))
    w = 0.8 / len(keys)
    for j, key in enumerate(keys):
        vals = [float(getattr(evals[key], m)) for m, _ in metrics]
        ax.barh(y + (j - (len(keys) - 1) / 2) * w, vals, height=w * 0.9,
                color=S.REGIME_COLOR[key], edgecolor=S.PAPER, linewidth=2)
        for yy, v in zip(y + (j - (len(keys) - 1) / 2) * w, vals):
            if np.isfinite(v):
                ax.text(v, yy, f" {v:.3f}", va="center", fontsize=6.8, color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in metrics], fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title("B · Headline", loc="left", fontsize=9.6)

    # recovery fraction annotation
    if all(k in evals for k in ("naive", "generalized", "native224")):
        n_, g_, t_ = (float(getattr(evals[k], "rare_recall"))
                      for k in ("naive", "generalized", "native224"))
        if np.isfinite([n_, g_, t_]).all() and abs(t_ - n_) > 1e-6:
            frac = (g_ - n_) / (t_ - n_)
            ax.text(0.02, -0.16, f"Generalized 28² recovers {frac:+.0%} of the "
                                 f"rare-class recall gap that 224² recovers.",
                    transform=ax.transAxes, fontsize=8.6, color=S.INK,
                    fontweight="bold")

    S.sheet_title(fig, "Was the Resolution Problem Ever About Pixels?",
                  "If a 28×28 sheet with a legend recovers what 224×224 recovers, the "
                  "benchmark never needed to be bigger.",
                  plate="Figure 10")
    S.neatline(fig)
    return S.save(fig, outdir / "fig09_rare_class.png")


# ==========================================================================
# PLATE X — ablations / the shortcut check
# ==========================================================================
def fig_ablation(outdir: Path, ab: Dict[str, Dict[str, float]]) -> Path:
    S.use_style()
    names = list(ab)
    fig = plt.figure(figsize=(12.0, 0.52 * len(names) + 3.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.18, left=0.30, right=0.975,
                  top=0.80, bottom=0.14)

    for k, (metric, lab) in enumerate((("macro_auc", "macro AUC"),
                                       ("rare_recall", "rare-class recall"))):
        ax = fig.add_subplot(gs[k])
        vals = [ab[n].get(metric, np.nan) for n in names]
        base = ab.get("generalized (full)", {}).get(metric, np.nan)
        cols = []
        for n, v in zip(names, vals):
            if n.startswith("generalized (full)"):
                cols.append("#eb6834")
            elif "shuffled" in n or "shortcut" in n:
                cols.append(S.BAD)
            else:
                cols.append(S.INK_3)
        ax.barh(np.arange(len(names)), vals, color=cols, height=0.55,
                edgecolor=S.PAPER, linewidth=2)
        if np.isfinite(base):
            ax.axvline(base, color="#eb6834", lw=1.0, ls=(0, (4, 3)), zorder=5)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names if k == 0 else [""] * len(names), fontsize=7.6)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab, fontsize=8.5)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=7, color=S.INK)
        ax.set_title(("A · " if k == 0 else "B · ") + lab, loc="left", fontsize=9.6)

    S.sheet_title(fig, "Ablations, Including the One That Could Sink This",
                  "Dashed line = the full generalized sheet. Red rows are the shortcut tests: "
                  "if performance survives destroying the symbols' alignment to the image, "
                  "the model was reading a shortcut, not evidence.",
                  plate="Figure 11")
    S.caption(fig, "Dropping the symbols at test time should hurt. Shuffling the symbol "
                   "channels across images should hurt at least as much — if it does not, the "
                   "symbols were acting as a per-image fingerprint rather than as measurement.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig10_ablation.png")


# ==========================================================================
# PLATE XI — ISIC attribute validation
# ==========================================================================
def fig_isic(outdir: Path, res: Dict) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(10.4, 5.0))
    ax = fig.add_subplot(111)
    if not res.get("available"):
        ax.axis("off")
        ax.text(0.5, 0.6, "ISIC 2018 Task 2 attribute masks not available",
                ha="center", fontsize=12, color=S.INK, fontweight="bold")
        ax.text(0.5, 0.44, str(res.get("reason", "")), ha="center", fontsize=8.5,
                color=S.INK_2, wrap=True)
        ax.text(0.5, 0.26, "The symbol channels are therefore reported as "
                           "measured-but-unvalidated.\nThis is a caveat, not a "
                           "detail: a filter bank that fires on compression\n"
                           "artefacts would still win an ablation.",
                ha="center", fontsize=8.5, color=S.BAD, style="italic")
    else:
        attrs = list(res["attributes"])
        y = np.arange(len(attrs))
        px = [res["attributes"][a].get("pixel_auc_mean", np.nan) for a in attrs]
        sem = [res["attributes"][a].get("pixel_auc_sem", 0.0) for a in attrs]
        im = [res["attributes"][a].get("image_auc", np.nan) for a in attrs]
        ax.barh(y - 0.19, px, xerr=sem, height=0.34,
                color=[S.STRUCTURE_COLOR[a] for a in attrs],
                edgecolor=S.PAPER, linewidth=2,
                error_kw=dict(ecolor=S.INK, lw=0.9, capsize=2.5))
        ax.barh(y + 0.19, im, height=0.34,
                color=[S.STRUCTURE_COLOR[a] for a in attrs], alpha=0.45,
                edgecolor=S.PAPER, linewidth=2)
        ax.axvline(0.5, color=S.BAD, lw=1.2, ls=(0, (4, 3)))
        ax.text(0.5, len(attrs) - 0.35, " chance", color=S.BAD, fontsize=7.6,
                va="center")
        for yy, v in zip(y - 0.19, px):
            if np.isfinite(v):
                ax.text(v, yy, f"  {v:.3f}  pixel", va="center", fontsize=7.4,
                        color=S.INK, fontweight="bold")
        for yy, v in zip(y + 0.19, im):
            if np.isfinite(v):
                ax.text(v, yy, f"  {v:.3f}  image", va="center", fontsize=7.4,
                        color=S.INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([a.replace("_", " ") for a in attrs], fontsize=9)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("AUC against expert annotation", fontsize=8.5)
        ax.text(0.0, -0.16, f"n = {res['n_images']} ISIC 2018 images · uncovered "
                            f"ISIC attributes: {', '.join(res['uncovered_attributes'])}",
                transform=ax.transAxes, fontsize=7.6, color=S.INK_2)

    S.sheet_title(fig, "Do the Symbols Track What Dermatologists Annotate?",
                  "Pixel AUC: does the symbol rank annotated cells above unannotated ones "
                  "within an image. Image AUC: does symbol density separate present from absent.",
                  plate="Figure 12")
    S.neatline(fig)
    return S.save(fig, outdir / "fig11_isic_validation.png")


# ==========================================================================
# PLATE XII — the standard, as one sheet
# ==========================================================================
def fig_headline(outdir: Path, evals: Dict[str, object], law: Dict,
                 merc: Dict[str, Dict[str, float]], cert: Dict[str, Dict],
                 nyquist: Dict) -> Path:
    S.use_style()
    fig = plt.figure(figsize=(13.0, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.3, left=0.06,
                  right=0.97, top=0.82, bottom=0.10)

    def stat(ax, value, label, sub="", color=S.INK):
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=S.PAPER_DEEP, edgecolor=S.RULE, lw=1.0))
        ax.text(0.5, 0.66, value, transform=ax.transAxes, ha="center",
                va="center", fontsize=26, color=color, fontweight="bold")
        ax.text(0.5, 0.32, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.4, color=S.INK)
        if sub:
            ax.text(0.5, 0.14, sub, transform=ax.transAxes, ha="center",
                    va="center", fontsize=7, color=S.INK_2, style="italic")

    # 1 — gap recovery
    try:
        n_, g_, t_ = (float(getattr(evals[k], "rare_recall"))
                      for k in ("naive", "generalized", "native224"))
        trail = f"naive {n_:.3f} → gen {g_:.3f} → 224 {t_:.3f}"
        if abs(t_ - n_) > 1e-6 and np.isfinite([n_, g_, t_]).all():
            frac = (g_ - n_) / (t_ - n_)
            stat(fig.add_subplot(gs[0, 0]), f"{frac:.0%}",
                 "of the rare-class gap that 224² buys,\n"
                 "recovered at 28² by generalization", trail, "#eb6834")
        else:
            # 224 bought nothing over naive, so there is no gap to recover a
            # fraction of. Saying so is the honest reading; printing "nan%" or
            # silently dividing by ~0 is not.
            stat(fig.add_subplot(gs[0, 0]), "no gap",
                 "224² bought no rare-class recall over naive 28²,\n"
                 "so the recovery fraction is undefined", trail, S.INK_2)
    except Exception:                                   # noqa: BLE001
        stat(fig.add_subplot(gs[0, 0]), "—", "rare-class gap recovery")

    # 2 — sub-Nyquist structures
    lost = sum(1 for v in nyquist.values() if not v["survives_naive_resize"])
    stat(fig.add_subplot(gs[0, 1]), f"{lost}/{len(nyquist)}",
         "diagnostic structures that cannot survive\n224² → 28² resampling at any quality",
         "and are symbolized instead", S.BAD)

    # 3 — ECE
    try:
        e = float(getattr(evals["generalized"], "ece"))
        e0 = float(getattr(evals["naive"], "ece"))
        stat(fig.add_subplot(gs[0, 2]), f"{e:.3f}",
             "expected calibration error, generalized 28²",
             f"naive 28² = {e0:.3f}", "#eb6834")
    except Exception:                                   # noqa: BLE001
        stat(fig.add_subplot(gs[0, 2]), "—", "ECE")

    # 4 — Töpfer alpha
    a = law.get("alpha", float("nan"))
    stat(fig.add_subplot(gs[1, 0]), f"r^{a:.2f}" if np.isfinite(a) else "—",
         "measured symbol-budget scale law",
         "Töpfer's radical law predicts r^0.50", "#2a78d6")

    # 5 — Mercator spread
    try:
        sn = merc["naive"]["mean_spread"]
        sg = merc["generalized_ef"]["mean_spread"]
        stat(fig.add_subplot(gs[1, 1]), f"{sn:.3f} → {sg:.3f}",
             "retention spread across ITA strata\nnaive → generalized",
             "0 = equal fidelity by construction", "#1baf7a")
    except Exception:                                   # noqa: BLE001
        stat(fig.add_subplot(gs[1, 1]), "—", "ITA distortion spread")

    # 6 — certificate
    try:
        c = cert["source certificate"]["balanced_aurc"]
        s = cert["softmax confidence"]["balanced_aurc"]
        stat(fig.add_subplot(gs[1, 2]), f"{c:.3f}",
             "AURC of the label-free source certificate",
             f"model's own confidence = {s:.3f}", "#eb6834")
    except Exception:                                   # noqa: BLE001
        stat(fig.add_subplot(gs[1, 2]), "—", "certificate AURC")

    S.sheet_title(fig, "The Field Has Been Shipping Maps Without Legends",
                  "A medical imaging benchmark should ship (a) named generalization operators "
                  "with fidelity contracts, (b) a scale bar in physical units, and "
                  "(c) a reliability diagram.",
                  plate="Figure 1")
    S.caption(fig, "Every existing MedMNIST-style dataset is auditable against this standard.")
    S.neatline(fig)
    return S.save(fig, outdir / "fig00_headline.png")
