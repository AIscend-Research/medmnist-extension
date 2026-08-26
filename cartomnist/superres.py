"""Super-resolution-then-downsample baseline.

The most obvious alternative explanation for any naive-vs-generalized gap:
maybe the problem was never that a 28x28 grid can't hold enough pixels, and a
generic image prior applied to the SHIPPED 28x28 benchmark image would already
recover most of what symbolization buys. This module builds that arm: take
the naive-resized 28x28 RGB image only (never the true native image — that
would defeat the point of the control), upsample it back to native resolution
with a classical super-resolution proxy, and train the same ResNet-18 the
`native224` arm uses on the result.

Real-ESRGAN (Wang et al., 2021) was considered for this baseline instead of a
classical upsampler and deliberately left out: `basicsr`/`realesrgan` are
known to break on newer torchvision (they import a private
`torchvision.transforms.functional_tensor` module that was removed), which
conflicts with this project's exact-pin-verified-against-the-full-suite
dependency policy (see pyproject.toml), and Real-ESRGAN's pretrained weights
need a download the Kaggle notebook is designed to run without (see
`data.load_derma224`'s offline-Kaggle docstring). The classical arm below
answers the same question — "does a generic upsampling prior already close
the gap?" — without the dependency risk.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .parallel import concat_arrays, map_chunks


def classical_sr_upsample(rgb28: np.ndarray, target: int = 224) -> np.ndarray:
    """Lanczos upsample of a (3, h, w) naive-resized image, then unsharp-mask.

    Unsharp masking after a high-quality interpolation kernel is the standard
    "classical" reference point in the super-resolution literature (e.g. Dong
    et al., 2014, use bicubic upsampling as SRCNN's baseline) — it recovers
    edge contrast a plain interpolation smooths away, without hallucinating
    texture the way a learned generative prior would.

    Takes the already-downsampled (3, h, w) image in [0, 1], not an HWC native
    image — unlike `generalize.naive_resize` / `baselines.naive_resize_interp`,
    which both *produce* the 28x28 image, this function's whole job is to be a
    pure function of that 28x28 image and nothing else.
    """
    rgb28 = np.asarray(rgb28, dtype=np.float32)
    if rgb28.max() > 1.5:
        rgb28 = rgb28 / 255.0
    rgb28 = np.clip(rgb28, 0.0, 1.0)
    hwc = np.transpose(rgb28, (1, 2, 0))
    img = Image.fromarray((hwc * 255.0 + 0.5).astype(np.uint8))
    up = img.resize((target, target), resample=Image.LANCZOS)
    sharp = up.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=0))
    out = np.asarray(sharp, dtype=np.float32) / 255.0
    return np.transpose(out, (2, 0, 1)).astype(np.float32)


def classical_sr_upsample_batch(rgb28s, target: int = 224,
                                n_jobs: int = 0) -> np.ndarray:
    """Batch version, parallelised the same way as `generalize.generalize_batch`."""
    def _run(chunk):
        return np.stack([classical_sr_upsample(chunk[i], target)
                         for i in range(len(chunk))])

    return concat_arrays(map_chunks(rgb28s, _run, n_jobs))
