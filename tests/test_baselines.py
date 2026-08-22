"""Tests for the smarter-interpolation baselines (Phase 3: does the naive-vs-
generalized gap survive a better resize kernel, or is it just area-mean's fault)."""
from __future__ import annotations

import numpy as np
import pytest

from cartomnist.baselines import naive_resize_interp, naive_resize_interp_batch
from cartomnist.generalize import naive_resize
from tests.test_operators import synth


@pytest.mark.parametrize("method", ["bicubic", "lanczos"])
def test_shape_and_range(method):
    img = synth()
    out = naive_resize_interp(img, target=28, method=method)
    assert out.shape == (3, 28, 28)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        naive_resize_interp(synth(), target=28, method="nearest")


def test_batch_matches_single():
    imgs = np.stack([synth(seed=i) for i in range(4)])
    batch = naive_resize_interp_batch(imgs, target=28, method="bicubic")
    single = np.stack([naive_resize_interp(imgs[i], target=28, method="bicubic")
                       for i in range(4)])
    assert np.allclose(batch, single)


def test_differs_from_area_mean_pooling():
    """The whole point of this baseline: if it were numerically identical to
    `naive_resize`, it would not be answering the reviewer's question."""
    img = synth()
    area = naive_resize(img, target=28)
    bicubic = naive_resize_interp(img, target=28, method="bicubic")
    assert not np.allclose(area, bicubic, atol=1e-3)


def test_bicubic_and_lanczos_differ_from_each_other():
    img = synth()
    bicubic = naive_resize_interp(img, target=28, method="bicubic")
    lanczos = naive_resize_interp(img, target=28, method="lanczos")
    assert not np.allclose(bicubic, lanczos, atol=1e-4)
