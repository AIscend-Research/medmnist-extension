"""Pan-sharpening baseline: the remote-sensing analogue of this project's own
symbolize-then-fuse move, made literal.

`generalize.py`'s pipeline measures fine structure at native resolution and
injects it into the coarser target sheet — structurally the same move as
remote-sensing pan-sharpening, where a high-resolution panchromatic band is
fused into coarser multispectral bands. This module makes that comparison
literal: an actual classical fusion algorithm (Brovey or IHS) stands in for
cartographic symbolization, at the same final resolution, so the comparison
is a citable baseline rather than only an analogy.

Nothing here reads a label. `pansharpen_resize` returns (3, target, target)
float32 in [0, 1], drop-in comparable with `generalize.naive_resize` and
`baselines.naive_resize_interp`.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import hsv2rgb, rgb2hsv

from . import operators as ops
from .parallel import concat_arrays, map_chunks

_METHODS = ("brovey", "ihs")


def to_pan_band(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luminance — the "panchromatic" band.

    Gonzalez & Woods, *Digital Image Processing*.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]).astype(np.float32)


def _bicubic_resize(rgb: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    resized = img.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def brovey_fuse(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """Classic Brovey transform: `fused_i = ms_i * pan / mean_c(ms)`.

    Gillespie, Kahle & Walker (1987), "Color enhancement of highly
    correlated images."
    """
    ms = np.clip(np.asarray(ms, dtype=np.float32), 0.0, 1.0)
    mean_ms = ms.mean(axis=-1)
    ratio = pan / np.maximum(mean_ms, 1e-6)
    fused = ms * ratio[..., None]
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def ihs_fuse(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """IHS fusion: replace intensity (V) with the panchromatic band,
    histogram-matched to V's mean/std to limit spectral distortion.

    Chavez, Sides & Anderson (1991), "Comparison of three different methods
    to merge multiresolution and multispectral data."
    """
    ms = np.clip(np.asarray(ms, dtype=np.float32), 0.0, 1.0)
    hsv = rgb2hsv(ms).astype(np.float32)
    v = hsv[..., 2]
    v_mean, v_std = float(v.mean()), float(v.std()) + 1e-6
    p_mean, p_std = float(pan.mean()), float(pan.std()) + 1e-6
    matched = (pan - p_mean) / p_std * v_std + v_mean
    hsv[..., 2] = np.clip(matched, 0.0, 1.0)
    return np.clip(hsv2rgb(hsv), 0.0, 1.0).astype(np.float32)


def pansharpen_resize(rgb_native: np.ndarray, target: int = 28,
                      method: str = "brovey") -> np.ndarray:
    """Pan-sharpen the native image, then pool it down to `target` exactly
    like every other arm — an apples-to-apples comparison against
    `generalize.naive_resize`, which is area-mean pooling with no fusion at
    all.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    rgb = np.asarray(rgb_native, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    native = rgb.shape[0]

    ms_low = ops.aggregate(rgb, target)              # the naive-resize baseline
    ms_up = _bicubic_resize(ms_low, native)           # back to native resolution
    pan = to_pan_band(rgb)

    fused = brovey_fuse(ms_up, pan) if method == "brovey" else ihs_fuse(ms_up, pan)
    out = ops.aggregate(fused, target)
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


def pansharpen_resize_batch(rgbs, target: int = 28, method: str = "brovey",
                            n_jobs: int = 0) -> np.ndarray:
    """Batch version, parallelised the same way as `generalize.generalize_batch`."""
    def _run(chunk):
        return np.stack([pansharpen_resize(chunk[i], target, method)
                         for i in range(len(chunk))])

    return concat_arrays(map_chunks(rgbs, _run, n_jobs))
