"""Baselines for the comparison the generalized regime has to beat.

`generalize.naive_resize` is area-mean pooling, which is what MedMNIST-style
benchmarks actually ship (see `data.load_derma28`, verified against the
official arrays). The obvious first question a reviewer asks is whether the
naive-vs-generalized gap is really about symbolization, or just about that
particular choice of resampling kernel. This module answers it: two smarter,
still-symbol-free resamplers, trained and evaluated the same way as every
other arm.

Nothing here is symbolized. All three functions return (3, target, target)
float32 in [0, 1], drop-in comparable with `generalize.naive_resize`.
"""
from __future__ import annotations

import numpy as np

from .parallel import concat_arrays, map_chunks

_PIL_METHODS = {"bicubic": "BICUBIC", "lanczos": "LANCZOS"}


def naive_resize_interp(rgb: np.ndarray, target: int = 28,
                        method: str = "bicubic") -> np.ndarray:
    """A single-image resize with a smarter kernel than area-mean pooling.

    `method` is "bicubic" or "lanczos" — the two resamplers a reviewer would
    reach for first as "the obvious better alternative" to area-mean pooling.
    Unlike `operators.aggregate`, these are NOT guaranteed to conserve the
    areal integral; that is exactly the point of comparing them.
    """
    if method not in _PIL_METHODS:
        raise ValueError(f"method must be one of {list(_PIL_METHODS)}, got {method!r}")
    from PIL import Image

    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    img = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))
    resample = getattr(Image, _PIL_METHODS[method])
    resized = img.resize((target, target), resample=resample)
    out = np.asarray(resized, dtype=np.float32) / 255.0
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


def naive_resize_interp_batch(rgbs, target: int = 28, method: str = "bicubic",
                              n_jobs: int = 0) -> np.ndarray:
    """Batch version, parallelised the same way as `generalize.generalize_batch`."""
    def _run(chunk):
        return np.stack([naive_resize_interp(chunk[i], target, method)
                         for i in range(len(chunk))])

    return concat_arrays(map_chunks(rgbs, _run, n_jobs))
