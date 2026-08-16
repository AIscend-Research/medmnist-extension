"""Tests for the training layer.

The important one is `test_augmentation_is_equivariant`. Symbol channels carry
orientation, so flipping a tensor without transforming cos2θ/sin2θ trains the
model on charts whose compass rose disagrees with their roads. That bug is
completely silent: the loss still goes down, the orientation channels just
quietly degrade to noise. The test pins it by checking the only thing that can
be checked from outside — that generalizing a transformed image equals
transforming a generalized image.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cartomnist.generalize import channel_spec, generalize
from cartomnist.train import EquivariantAug
from tests.test_operators import synth


class FixedRNG:
    """Deterministic stand-in that replays a scripted sequence of draws."""

    def __init__(self, randoms, ints):
        self._r, self._i = list(randoms), list(ints)

    def random(self):
        return self._r.pop(0)

    def integers(self, lo, hi):
        return self._i.pop(0)


def _apply(x, hflip=False, vflip=False, k=0):
    spec = channel_spec()
    cos_i = [i for i, n in enumerate(spec.names) if n.endswith("::cos2theta")]
    sin_i = [i for i, n in enumerate(spec.names) if n.endswith("::sin2theta")]
    aug = EquivariantAug(cos_i, sin_i)
    rng = FixedRNG([0.0 if hflip else 1.0, 0.0 if vflip else 1.0], [k])
    return aug(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)), rng).numpy()


@pytest.mark.parametrize("kind", ["hflip", "vflip", "rot90", "rot270"])
def test_augmentation_is_equivariant(kind):
    img = synth(seed=2)
    t = generalize(img, target=28)

    if kind == "hflip":
        img2, got = img[:, ::-1], _apply(t, hflip=True)
    elif kind == "vflip":
        img2, got = img[::-1, :], _apply(t, vflip=True)
    elif kind == "rot90":
        img2, got = np.rot90(img, 1, axes=(0, 1)), _apply(t, k=1)
    else:
        img2, got = np.rot90(img, 3, axes=(0, 1)), _apply(t, k=3)

    want = generalize(np.ascontiguousarray(img2), target=28)

    # Density channels must match closely; orientation channels are compared
    # only where a bearing was actually drawn, since the simplification
    # operator thresholds coherence and the threshold can flip on ties.
    spec = channel_spec()
    dens = [i for i in range(spec.n) if not (
        spec.names[i].endswith("::cos2theta") or spec.names[i].endswith("::sin2theta"))]
    assert np.abs(got[dens] - want[dens]).mean() < 0.02, \
        float(np.abs(got[dens] - want[dens]).mean())

    ori = [i for i in range(spec.n) if spec.names[i].endswith("::cos2theta")
           or spec.names[i].endswith("::sin2theta")]
    drawn = (np.abs(got[ori] - 0.5) > 0.02) & (np.abs(want[ori] - 0.5) > 0.02)
    if drawn.sum() > 20:
        err = np.abs(got[ori][drawn] - want[ori][drawn]).mean()
        assert err < 0.06, float(err)


def test_undrawn_bearing_survives_augmentation():
    """The simplification contract says 0 means 'no bearing drawn'. Encoded
    that is 0.5, and every augmentation rule must map 0.5 to 0.5."""
    spec = channel_spec()
    cos_i = [i for i, n in enumerate(spec.names) if n.endswith("::cos2theta")]
    sin_i = [i for i, n in enumerate(spec.names) if n.endswith("::sin2theta")]
    x = np.full((spec.n, 8, 8), 0.5, dtype=np.float32)
    for h, v, k in ((True, False, 0), (False, True, 0), (False, False, 1),
                    (True, True, 3)):
        got = _apply(x, hflip=h, vflip=v, k=k)
        assert np.allclose(got[cos_i], 0.5) and np.allclose(got[sin_i], 0.5)


def test_smallcnn_runs_at_every_sweep_resolution():
    from cartomnist.models import SmallCNN
    spec = channel_spec()
    m = SmallCNN(spec.n, 7)
    for r in (7, 14, 28, 56):
        out = m(torch.zeros(2, spec.n, r, r))
        assert out.shape == (2, 7)


def test_class_weights_favour_rare_classes():
    from cartomnist.train import _class_weights
    y = np.array([0] * 100 + [1] * 10)
    w = _class_weights(y, 2).numpy()
    assert w[1] > w[0]


def test_temperature_scaling_reduces_nll_on_overconfident_logits():
    from cartomnist.metrics import fit_temperature, reliability, softmax
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 800)
    logits = rng.normal(0, 1, (800, 3))
    logits[np.arange(800), y] += 1.0
    logits *= 6.0                                  # deliberately overconfident
    T = fit_temperature(logits, y)
    assert T > 1.2
    assert reliability(softmax(logits / T), y)["ece"] < \
           reliability(softmax(logits), y)["ece"]
