"""The source diagram.

A nautical chart carries an inset showing which waters were surveyed, how
densely, and when. It is not decoration: it tells a navigator where the chart
may be trusted and where soundings are two centuries old and half a mile apart.

This module computes the same object for a medical image. For every cell of the
target grid and every diagnostic structure, it answers one question from the
NATIVE image alone:

    Did the acquisition locally clear the sampling and contrast floor that this
    structure needs in order to be measurable at all?

The answer is expressed as two log-ratio *margins* in stops. Positive means the
survey was adequate; negative means the symbol drawn in that cell is an
extrapolation and should be read as such. `certificate = min(margins)` is the
per-cell zone-of-confidence, and its image-level aggregate is what the
abstention experiment thresholds on.

Crucially this never touches the labels, so it cannot leak, and it is available
at test time for an unlabelled image.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from .filterbanks import STRUCTURES, scale_periods
from .operators import aggregate


# --------------------------------------------------------------------------
def _bandpass(L: np.ndarray, period: float) -> np.ndarray:
    """Difference-of-Gaussians isolating the octave around `period`."""
    lo = ndi.gaussian_filter(L, period / 3.0)
    hi = ndi.gaussian_filter(L, period * 1.2)
    return lo - hi


def _local_rms(x: np.ndarray, win: float) -> np.ndarray:
    m = ndi.uniform_filter(x, size=int(max(3, win)))
    m2 = ndi.uniform_filter(x * x, size=int(max(3, win)))
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def estimate_noise_floor(L: np.ndarray) -> float:
    """Robust sensor/compression noise estimate: MAD of the finest Laplacian.

    Uses the classic 1.4826 * MAD / sqrt(2) estimator on a 3x3 Laplacian, which
    is dominated by noise rather than by anatomy at that scale.
    """
    lap = ndi.laplace(L)
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(1.4826 * mad / np.sqrt(6.0) + 1e-6)


# --------------------------------------------------------------------------
def source_diagram(rgb: np.ndarray, target: int,
                   native_size: int | None = None) -> Dict[str, np.ndarray]:
    """Per-cell survey adequacy for every structure.

    Returns
    -------
    dict with
        "sampling"    (T, target, target) margin in stops, per structure
        "contrast"    (T, target, target) margin in stops, per structure
        "certificate" (target, target)    min over both margins and structures,
                                          squashed to [0, 1] (0.5 == exactly at
                                          the floor)
        "nyquist"     (T,) scalar per structure: is the structure resolvable at
                      all on a `target` grid by naive resampling? (>0 == yes)
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if native_size is None:
        native_size = rgb.shape[0]
    L = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)[..., 0] / 100.0

    periods = scale_periods(native_size)
    noise = estimate_noise_floor(L)
    cell_px = native_size / float(target)

    samp, cont, nyq = [], [], []
    for spec in STRUCTURES:
        p = periods[spec.name]
        band = _bandpass(L, p)

        # --- sampling margin: is there measurable energy in the structure's
        #     octave, above the sensor noise floor?
        energy = _local_rms(band, win=max(3.0, p * 2.0))
        s_margin = np.log2((energy + 1e-9) / noise)

        # --- contrast margin: does that energy clear the structure's minimum
        #     detectable contrast, in L* units?
        c_margin = np.log2((energy + 1e-9) / spec.contrast_floor)

        samp.append(aggregate(s_margin, target))
        cont.append(aggregate(c_margin, target))

        # --- Nyquist bookkeeping for the naive-resize arm: a structure of
        #     period p survives resampling to `target` only if p >= 2 * cell_px
        nyq.append(np.log2(p / (2.0 * cell_px)))

    samp = np.stack(samp, 0).astype(np.float32)
    cont = np.stack(cont, 0).astype(np.float32)
    worst = np.minimum(samp, cont).min(axis=0)
    certificate = 1.0 / (1.0 + np.exp(-worst))          # 0.5 == exactly at floor

    return {
        "sampling": samp,
        "contrast": cont,
        "certificate": certificate.astype(np.float32),
        "nyquist": np.asarray(nyq, dtype=np.float32),
    }


def image_certificate(cert: np.ndarray, symbol_density: np.ndarray) -> float:
    """Collapse the per-cell certificate to one number for abstention.

    Weighted by where symbols were actually drawn: a chart is untrustworthy in
    proportion to how much of what it asserts sits over unsurveyed water. Empty
    background being unsurveyed does not matter.
    """
    w = np.clip(symbol_density, 0, None)
    tot = w.sum()
    if tot < 1e-8:
        return float(cert.mean())
    return float((cert * w).sum() / tot)


def nyquist_table(target: int, native_size: int = 224) -> Dict[str, Dict[str, float]]:
    """The table that makes the whole argument concrete.

    For each structure: its period in native pixels, the target cell footprint,
    and whether naive resampling can represent it at all.
    """
    periods = scale_periods(native_size)
    cell = native_size / float(target)
    out = {}
    for spec in STRUCTURES:
        p = periods[spec.name]
        out[spec.name] = {
            "period_native_px": round(float(p), 2),
            "target_cell_px": round(float(cell), 2),
            "nyquist_ratio": round(float(p / (2.0 * cell)), 3),
            "survives_naive_resize": bool(p >= 2.0 * cell),
        }
    return out
