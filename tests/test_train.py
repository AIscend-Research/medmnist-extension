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

    # Orientation is compared only where a bearing is *well posed*: on the
    # interior, and where the pooled resultant is clearly above the coherence
    # floor in both sheets. A cell whose resultant is near zero has no single
    # bearing to preserve — the synthetic mesh is a separable sin·sin pattern,
    # so it is genuinely bi-directional and its circular mean flips between two
    # equally good modes on numerical noise. Testing those cells would measure
    # tie-breaking, not the double-angle algebra this test exists to pin.
    interior = np.zeros(got.shape[-2:], dtype=bool)
    interior[2:-2, 2:-2] = True

    checked = 0
    for sname in ("pigment_network", "streaks", "vessels"):
        ci = [i for i in spec.groups[sname] if spec.names[i].endswith("::cos2theta")][0]
        si = [i for i in spec.groups[sname] if spec.names[i].endswith("::sin2theta")][0]

        def decode(a):
            return a[ci] * 2 - 1, a[si] * 2 - 1

        gc, gs = decode(got)
        wc, ws = decode(want)
        m = (np.hypot(gc, gs) > 0.35) & (np.hypot(wc, ws) > 0.35) & interior
        if m.sum() < 10:
            continue
        checked += int(m.sum())
        # angular error on the projective line, via the double-angle vectors
        dot = (gc[m] * wc[m] + gs[m] * ws[m]) / (
            np.hypot(gc[m], gs[m]) * np.hypot(wc[m], ws[m]))
        ang = np.degrees(np.arccos(np.clip(dot, -1, 1))) / 2.0    # back to θ
        assert np.median(ang) < 5.0, (sname, float(np.median(ang)))

    assert checked > 20, f"too few well-posed bearings to test ({checked})"


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


def test_mc_dropout_infer_varies_across_passes():
    from torch.utils.data import DataLoader, TensorDataset

    from cartomnist.models import SmallCNN
    from cartomnist.train import mc_dropout_infer

    torch.manual_seed(0)
    m = SmallCNN(3, 4, width=8, p_drop=0.5)
    x = torch.rand(16, 3, 8, 8)
    y = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    probs = mc_dropout_infer(m, loader, torch.device("cpu"), T=10, amp=False)
    assert probs.shape == (10, 16, 4)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    # dropout must actually be active during sampling, or every pass would be
    # identical and the whole point of the signal would be vacuous.
    assert probs.var(axis=0).sum() > 0


def test_train_eval_mc_dropout_is_opt_in():
    from cartomnist.config import Config
    from cartomnist.train import train_eval

    rng = np.random.default_rng(0)
    x_tr = rng.random((12, 3, 8, 8)).astype(np.float32)
    y_tr = rng.integers(0, 3, 12)
    x_va = rng.random((6, 3, 8, 8)).astype(np.float32)
    y_va = rng.integers(0, 3, 6)
    x_te = rng.random((6, 3, 8, 8)).astype(np.float32)
    y_te = rng.integers(0, 3, 6)

    cfg = Config()
    cfg.epochs_small = 1
    cfg.batch_size = 4
    cfg.amp = False

    default_result = train_eval(x_tr, y_tr, x_va, y_va, x_te, y_te,
                                regime="naive", n_classes=3, cfg=cfg, seed=0,
                                verbose=False)
    assert default_result.mc_dropout_probs is None

    mc_result = train_eval(x_tr, y_tr, x_va, y_va, x_te, y_te,
                           regime="naive", n_classes=3, cfg=cfg, seed=0,
                           verbose=False, mc_dropout_samples=5)
    assert mc_result.mc_dropout_probs is not None
    assert mc_result.mc_dropout_probs.shape == (5, 6, 3)


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
