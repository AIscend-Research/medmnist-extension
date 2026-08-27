"""Tests for the learned/attention-based downsampler baseline (P1): is fixed
area-mean pooling itself costing accuracy, independent of symbolization?"""
from __future__ import annotations

import numpy as np
import torch

from cartomnist import operators as ops
from cartomnist.learned_pool import LearnedDownsample
from tests.test_operators import synth


def _to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))[None]).float()


def test_invalid_mode_raises():
    try:
        LearnedDownsample(channels=3, factor=8, mode="nearest")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_linear_mode_matches_area_mean_at_init():
    img = synth()
    x = _to_tensor(img)
    ds = LearnedDownsample(channels=3, factor=8, mode="linear")
    with torch.no_grad():
        out = ds(x)[0].numpy()
    expected = np.transpose(ops.aggregate(img, 28), (2, 0, 1))
    assert np.allclose(out, expected, atol=1e-5)


def test_attention_mode_shape():
    img = synth()
    x = _to_tensor(img)
    ds = LearnedDownsample(channels=3, factor=8, mode="attention")
    out = ds(x)
    assert out.shape == (1, 3, 28, 28)
    assert torch.isfinite(out).all()


def test_attention_weights_sum_to_one_per_block():
    """The whole point of softmax normalisation: the operator should still
    conserve total intensity when the input is uniform (weighted average of a
    constant is that constant), the same sanity check `aggregate` satisfies
    for a constant field."""
    x = torch.ones(1, 3, 16, 16) * 0.37
    ds = LearnedDownsample(channels=3, factor=4, mode="attention")
    out = ds(x)
    assert torch.allclose(out, torch.full_like(out, 0.37), atol=1e-5)


def test_gradient_flows_through_both_modes():
    img = synth()
    x = _to_tensor(img)

    linear = LearnedDownsample(channels=3, factor=8, mode="linear")
    linear(x).sum().backward()
    assert linear.pool.weight.grad is not None
    assert torch.isfinite(linear.pool.weight.grad).all()

    attn = LearnedDownsample(channels=3, factor=8, mode="attention")
    attn(x).sum().backward()
    assert attn.score.weight.grad is not None
    assert torch.isfinite(attn.score.weight.grad).all()
