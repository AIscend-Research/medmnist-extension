"""Tests for the SR-then-downsample baseline (P0): does a generic upsampling
prior recover naive resizing's losses without symbolization?"""
from __future__ import annotations

import numpy as np
from PIL import Image

from cartomnist.generalize import naive_resize
from cartomnist.superres import classical_sr_upsample, classical_sr_upsample_batch
from tests.test_operators import synth


def test_shape_and_range():
    naive28 = naive_resize(synth(), target=28)
    out = classical_sr_upsample(naive28, target=224)
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_batch_matches_single():
    naive28s = np.stack([naive_resize(synth(seed=i), target=28) for i in range(4)])
    batch = classical_sr_upsample_batch(naive28s, target=224)
    single = np.stack([classical_sr_upsample(naive28s[i], target=224)
                       for i in range(4)])
    assert np.allclose(batch, single)


def test_differs_from_plain_upsample_with_no_sharpening():
    """The whole point of this baseline over plain interpolation: if unsharp
    masking changed nothing, this "SR" arm would just be `naive_resize_interp`
    with extra steps."""
    naive28 = naive_resize(synth(), target=28)
    sr = classical_sr_upsample(naive28, target=224)

    hwc = np.transpose(naive28, (1, 2, 0))
    img = Image.fromarray((np.clip(hwc, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    plain = np.transpose(
        np.asarray(img.resize((224, 224), resample=Image.LANCZOS),
                   dtype=np.float32) / 255.0,
        (2, 0, 1))
    assert not np.allclose(sr, plain, atol=1e-3)


def test_deterministic():
    """A pure function of the 28x28 input: same input, same output every time."""
    naive28 = naive_resize(synth(), target=28)
    a = classical_sr_upsample(naive28, target=224)
    b = classical_sr_upsample(naive28, target=224)
    assert np.array_equal(a, b)
