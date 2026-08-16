"""Invariant tests. Each one checks a clause of a published fidelity contract."""
from __future__ import annotations

import numpy as np
import pytest

from cartomnist import operators as ops
from cartomnist.generalize import channel_spec, generalize, generalize_multi
from cartomnist.sufficiency import nyquist_table, source_diagram


def synth(n=224, seed=0):
    """A synthetic 'lesion' with a periodic mesh, blobs and a ridge."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    base = 0.62 + 0.06 * np.sin(3 * xx) - 0.25 * np.exp(
        -(((xx - .5) ** 2 + (yy - .5) ** 2) / 0.05))
    mesh = 0.05 * np.sin(2 * np.pi * xx * n / 6) * np.sin(2 * np.pi * yy * n / 6)
    blobs = np.zeros_like(base)
    for _ in range(40):
        cy, cx = rng.uniform(0.25, 0.75, 2)
        blobs -= 0.18 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / 2e-5))
    g = np.clip(base + mesh + blobs, 0.05, 0.98)
    img = np.stack([g * 1.05, g * 0.82, g * 0.75], -1)
    return np.clip(img + rng.normal(0, 0.004, img.shape), 0, 1).astype(np.float32)


# ---- AGGREGATION conserves the areal integral -----------------------------
def test_aggregation_conserves_mass():
    x = np.random.default_rng(0).random((224, 224)).astype(np.float32)
    out = ops.aggregate(x, 28)
    assert out.shape == (28, 28)
    assert np.isclose(out.mean(), x.mean(), atol=1e-6)


def test_aggregation_handles_non_divisible():
    x = np.random.default_rng(0).random((225, 225)).astype(np.float32)
    assert ops.aggregate(x, 28).shape == (28, 28)


# ---- EXAGGERATION is strictly monotone ------------------------------------
def test_exaggeration_preserves_rank():
    x = np.random.default_rng(0).random(5000).astype(np.float32)
    y = ops.exaggerate(x, 0.55)
    assert np.array_equal(np.argsort(x), np.argsort(y))
    assert y.min() >= 0 and y.max() <= 1


# ---- DISPLACEMENT: identity when uncongested, order-preserving always ------
def test_displacement_identity_when_uncongested():
    s = np.random.default_rng(0).random((5, 8, 8)).astype(np.float32) * 0.15
    assert np.allclose(ops.displace(s, 0.6), s)


def test_displacement_preserves_within_cell_order():
    s = np.random.default_rng(1).random((5, 8, 8)).astype(np.float32) * 1.5
    out = ops.displace(s, 0.6)
    assert np.array_equal(np.argsort(s, axis=0), np.argsort(out, axis=0))
    assert out.max() <= s.max() + 1e-6


# ---- SIMPLIFICATION zeroes exactly below the coherence floor --------------
def test_simplification_zeroes_below_tau():
    rng = np.random.default_rng(0)
    c, s = rng.normal(size=(2, 16, 16)).astype(np.float32)
    coh = rng.random((16, 16)).astype(np.float32)
    co, so = ops.simplify_orientation(c, s, coh, 0.3)
    below = coh < 0.3
    assert np.all(co[below] == 0) and np.all(so[below] == 0)
    assert np.allclose(co[~below], c[~below])


# ---- orientation pooling happens on the projective line -------------------
def test_orientation_pooling_is_double_angle():
    """theta = 0 and theta = pi are the same bearing; their mean must be too."""
    theta = np.zeros((4, 4), np.float32)
    theta[:, 2:] = np.pi - 1e-6
    w = np.ones((4, 4), np.float32)
    c, s = ops.aggregate_orientation(theta, w, 1)
    assert np.isclose(float(c.ravel()[0]), 1.0, atol=1e-3)
    assert abs(float(s.ravel()[0])) < 1e-3


# ---- Töpfer ---------------------------------------------------------------
def test_topfer_radical_law():
    # halving resolution keeps 1/sqrt(2) of the features
    assert np.isclose(ops.topfer_n(100, 224, 112), 100 / np.sqrt(2), rtol=1e-6)


# ---- pipeline shape / range ----------------------------------------------
def test_generalize_shape_and_range():
    img = synth()
    spec = channel_spec()
    t = generalize(img, target=28)
    assert t.shape == (spec.n, 28, 28)
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert np.isfinite(t).all()
    # channels 0-2 are exactly the naive area-resize
    from cartomnist.generalize import naive_resize
    assert np.allclose(t[:3], naive_resize(img, 28), atol=1e-6)


def test_generalize_multi_matches_single():
    img = synth(seed=3)
    multi = generalize_multi(img, [14, 28])
    for r in (14, 28):
        assert np.allclose(multi[r], generalize(img, target=r), atol=1e-6)


def test_symbols_respond_to_structure():
    """The mesh symbol must be stronger on an image that has a mesh."""
    from cartomnist.generalize import channel_spec
    spec = channel_spec()
    ch = [i for i in spec.groups["pigment_network"]
          if spec.names[i].endswith("::density")][0]
    img = synth()
    flat = np.clip(np.stack([np.full((224, 224), 0.6)] * 3, -1)
                   + np.random.default_rng(0).normal(0, 0.004, (224, 224, 3)),
                   0, 1).astype(np.float32)
    on_structure = generalize(img, 28)[ch].mean()
    on_noise = generalize(flat, 28)[ch].mean()
    assert on_structure > 3 * on_noise, (on_structure, on_noise)


def test_symbols_do_not_amplify_pure_noise():
    """A featureless noisy image must not produce confident symbols.

    Guards the failure mode that plain per-image percentile normalisation has:
    it stretches noise to full range, drawing a confident pigment-network
    symbol over nothing — and would manufacture the equity result, since the
    lowest-contrast images would get amplified the most.
    """
    spec = channel_spec()
    dens = [i for i in spec.idx("symbols") if spec.names[i].endswith("::density")]
    rng = np.random.default_rng(0)
    for sigma in (0.002, 0.01, 0.04):
        flat = np.clip(0.6 + rng.normal(0, sigma, (224, 224, 3)), 0, 1
                       ).astype(np.float32)
        t = generalize(flat, 28)
        assert t[dens].mean() < 0.12, (sigma, float(t[dens].mean()))


# ---- source diagram -------------------------------------------------------
def test_source_diagram_shapes_and_bounds():
    sd = source_diagram(synth(), 28)
    assert sd["certificate"].shape == (28, 28)
    assert 0.0 <= sd["certificate"].min() and sd["certificate"].max() <= 1.0
    assert sd["sampling"].shape[0] == sd["contrast"].shape[0] == 5


def test_nyquist_audit_flags_subpixel_structures():
    tbl = nyquist_table(28, 224)
    # cell width is 8 native px, so anything with period < 16 px is unrepresentable
    assert tbl["pigment_network"]["survives_naive_resize"] is False
    assert tbl["dots_globules"]["survives_naive_resize"] is False
    assert tbl["blue_white_veil"]["survives_naive_resize"] is True


def test_source_diagram_is_label_free():
    """Certificate must be reproducible from the image alone."""
    img = synth(seed=5)
    a = source_diagram(img, 28)["certificate"]
    b = source_diagram(img.copy(), 28)["certificate"]
    assert np.array_equal(a, b)


# ---- the FFT Gabor path must agree with plain spatial convolution ---------
def test_gabor_fft_matches_spatial_convolution():
    """The bank is evaluated by FFT for speed. Pin it against the obvious
    implementation, because a mis-centred kernel changes the energy barely at
    all while corrupting the bearing — which is exactly the bug this caught."""
    from scipy import ndimage as ndi

    from cartomnist.filterbanks import _gabor_bank, _gabor_energy_fft

    rng = np.random.default_rng(0)
    img = ndi.gaussian_filter(rng.standard_normal((160, 160)), 1.2).astype(np.float32)
    period = 6.0

    got, thetas = _gabor_energy_fft(img, period)
    for k, (theta, kr, ki) in enumerate(_gabor_bank(period)):
        assert np.isclose(thetas[k], theta)
        want = np.hypot(ndi.convolve(img, kr, mode="constant"),
                        ndi.convolve(img, ki, mode="constant"))
        # interior only: the FFT path is zero-padded linear convolution, which
        # matches mode="constant" everywhere except within a kernel radius of
        # the frame
        m = slice(20, -20)
        rel = np.abs(got[k][m, m] - want[m, m]).max() / (want[m, m].max() + 1e-9)
        assert rel < 1e-4, (k, float(rel))
