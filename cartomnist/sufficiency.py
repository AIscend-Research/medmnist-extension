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

from typing import Dict

import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from .filterbanks import STRUCTURES, scale_periods
from .operators import aggregate

# Margins are in stops (log2). This is the width of the transition band when
# the margins are squashed into the [0, 1] certificate.
CERT_TEMPERATURE_STOPS = 3.0


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
    """Robust global sensor/compression noise estimate, in L*/100 units."""
    lap = ndi.laplace(L)
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(1.4826 * mad / np.sqrt(6.0) + 1e-6)


def local_noise_floor(L: np.ndarray, win: int = 9, quiet: int = 25) -> np.ndarray:
    """Per-pixel noise floor, estimated from the quietest nearby neighbourhood.

    A plain local |Laplacian| average is inflated wherever anatomy exists, which
    would make the source diagram say "noisy" when it means "detailed". Taking a
    minimum filter over the local averages picks the quietest patch in the
    vicinity, which is the standard presence-robust way to read a noise floor
    off a single image.
    """
    lap = np.abs(ndi.laplace(L))
    local = ndi.uniform_filter(lap, size=int(max(3, win)))
    floor = ndi.minimum_filter(local, size=int(max(5, quiet)))
    # E|x| = sigma*sqrt(2/pi) for Gaussian noise; /sqrt(6) is the 4-neighbour
    # Laplacian's noise gain
    return np.maximum(floor / 0.7979 / np.sqrt(6.0), 1e-6).astype(np.float32)


# --------------------------------------------------------------------------
_LSTAR_LUT: np.ndarray | None = None


def _lstar_lut() -> np.ndarray:
    """L*/100 for each of the 256 sRGB grey codes."""
    global _LSTAR_LUT
    if _LSTAR_LUT is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        y = lin  # D65 grey: Y == linear value
        f = np.where(y > 0.008856, np.cbrt(y), 7.787 * y + 16.0 / 116.0)
        _LSTAR_LUT = ((116.0 * f - 16.0) / 100.0).astype(np.float32)
    return _LSTAR_LUT


def quantization_step(L: np.ndarray) -> np.ndarray:
    """L*/100 resolution of one 8-bit code, at each pixel's local luminance.

    This is the quietly decisive quantity for the equity argument, and it is
    pure physics rather than modelling. sRGB encoding is roughly perceptually
    uniform but not exactly: in the dark end, one code value spans a *larger*
    step in L*. So on darker skin, an 8-bit image can represent fewer distinct
    contrast levels for the same structure — before any resizing, before any
    model. A benchmark that never states this is asserting a fidelity it does
    not have.
    """
    lut = _lstar_lut()
    step = np.diff(lut, append=lut[-1] + (lut[-1] - lut[-2]))
    code = np.clip((L * 100.0), 0, 100)
    # invert L* -> code via the LUT, then read the step there
    idx = np.searchsorted(lut, code / 100.0).clip(0, 255)
    return np.maximum(step[idx], 1e-6).astype(np.float32)


# --------------------------------------------------------------------------
def source_diagram(rgb: np.ndarray, target: int,
                   native_size: int | None = None) -> Dict[str, np.ndarray]:
    """Per-cell survey adequacy for every structure.

    Both margins ask about the **acquisition**, never about the anatomy. That
    distinction is the whole design. Measuring band-limited energy at each
    structure's scale — the obvious first implementation — conflates "this
    structure is not here" with "this structure could not have been seen here",
    so a perfectly exposed lesion with no vessels reads as unsurveyed. A source
    diagram that fires on absence is worse than none: it would tell a navigator
    the chart is untrustworthy wherever the sea floor happens to be flat.

    Returns
    -------
    dict with
        "sampling"    (T, t, t) noise margin in stops, per structure
        "contrast"    (T, t, t) quantisation margin in stops, per structure
        "certificate" (t, t)    min over both margins and all structures,
                                squashed to [0, 1] (0.5 == exactly at the floor)
        "nyquist"     (T,) per structure: is it resolvable at all on a `target`
                      grid by naive resampling? (>0 == yes)
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    if native_size is None:
        native_size = rgb.shape[0]
    L = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)[..., 0] / 100.0

    periods = scale_periods(native_size)
    noise = local_noise_floor(L)
    qstep = quantization_step(L)
    # clipped pixels carry no recoverable modulation at all
    clipped = ndi.uniform_filter(
        ((rgb.max(-1) > 0.995) | (rgb.min(-1) < 0.005)).astype(np.float32),
        size=9)
    cell_px = native_size / float(target)

    samp, cont, nyq = [], [], []
    for spec in STRUCTURES:
        p = periods[spec.name]

        # --- noise margin: could a structure at this structure's minimum
        #     interesting contrast be told apart from the noise here? Averaging
        #     over the structure's own footprint buys sqrt(N) of SNR, which is
        #     why a large diffuse veil survives noise that would bury a dot.
        n_avg = max(1.0, (p / 2.0) ** 2)
        s_margin = np.full_like(
            L, 0.0) + np.log2(spec.contrast_floor * np.sqrt(n_avg) / noise)

        # --- quantisation margin: can 8-bit encoding even represent that
        #     contrast at this local luminance? Clipping destroys it outright.
        c_margin = np.log2(spec.contrast_floor / qstep) - 6.0 * clipped

        samp.append(aggregate(s_margin, target))
        cont.append(aggregate(c_margin, target))

        # --- Nyquist bookkeeping for the naive-resize arm: a structure of
        #     period p survives resampling to `target` only if p >= 2 * cell_px
        nyq.append(np.log2(p / (2.0 * cell_px)))

    samp = np.stack(samp, 0).astype(np.float32)
    cont = np.stack(cont, 0).astype(np.float32)
    # Zones of confidence: the *worst* structure governs the cell, as on a
    # chart where one shoal decides the safe draught. Squashed with a 3-stop
    # temperature rather than 1: a single stop below the floor is a caution,
    # not a cliff, and a hard sigmoid saturates the whole map to 0 or 1 and
    # destroys the spatial information the inset exists to convey.
    worst = np.minimum(samp, cont).min(axis=0)
    certificate = 1.0 / (1.0 + np.exp(-worst / CERT_TEMPERATURE_STOPS))

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
