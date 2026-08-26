"""Tests for the pan-sharpening baseline (P1): a literal, citable
remote-sensing fusion algorithm standing in for cartographic symbolization."""
from __future__ import annotations

import numpy as np
import pytest

from cartomnist.generalize import naive_resize
from cartomnist.pansharpen import (pansharpen_resize, pansharpen_resize_batch,
                                   to_pan_band)
from tests.test_operators import synth


@pytest.mark.parametrize("method", ["brovey", "ihs"])
def test_shape_and_range(method):
    img = synth()
    out = pansharpen_resize(img, target=28, method=method)
    assert out.shape == (3, 28, 28)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        pansharpen_resize(synth(), target=28, method="pca")


def test_batch_matches_single():
    imgs = np.stack([synth(seed=i) for i in range(4)])
    batch = pansharpen_resize_batch(imgs, target=28, method="brovey")
    single = np.stack([pansharpen_resize(imgs[i], target=28, method="brovey")
                       for i in range(4)])
    assert np.allclose(batch, single)


def test_differs_from_area_mean_pooling():
    """The whole point of this baseline: fusing panchromatic detail in before
    the final pool should change the result versus plain area-mean pooling."""
    img = synth()
    area = naive_resize(img, target=28)
    fused = pansharpen_resize(img, target=28, method="brovey")
    assert not np.allclose(area, fused, atol=1e-3)


def test_brovey_and_ihs_differ_from_each_other():
    img = synth()
    brovey = pansharpen_resize(img, target=28, method="brovey")
    ihs = pansharpen_resize(img, target=28, method="ihs")
    assert not np.allclose(brovey, ihs, atol=1e-4)


def test_pan_band_matches_luminance_of_grey_input():
    grey = np.full((16, 16, 3), 0.4, dtype=np.float32)
    pan = to_pan_band(grey)
    assert pan.shape == (16, 16)
    assert np.allclose(pan, 0.4, atol=1e-5)
