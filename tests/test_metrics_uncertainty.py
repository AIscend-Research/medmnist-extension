"""Tests for the general-purpose uncertainty signals the source-diagram
certificate is scored against (selective-prediction extension, P1)."""
from __future__ import annotations

import numpy as np

from cartomnist.metrics import ensemble_uncertainty, mc_dropout_uncertainty


def test_mc_dropout_uncertainty_shape_and_nonnegative():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (10, 20, 4))
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    out = mc_dropout_uncertainty(probs)
    assert out.shape == (20,)
    assert (out >= 0).all()


def test_mc_dropout_uncertainty_zero_when_passes_identical():
    probs = np.tile(np.array([[0.7, 0.2, 0.1]]), (5, 6, 1))
    out = mc_dropout_uncertainty(probs)
    assert np.allclose(out, 0.0)


def test_ensemble_uncertainty_shape_and_nonnegative():
    rng = np.random.default_rng(0)
    seeds = [rng.dirichlet(np.ones(4), size=15) for _ in range(3)]
    out = ensemble_uncertainty(seeds)
    assert out.shape == (15,)
    assert (out >= -1e-8).all()


def test_ensemble_uncertainty_zero_when_members_agree():
    member = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    out = ensemble_uncertainty([member, member, member])
    assert np.allclose(out, 0.0, atol=1e-6)


def test_ensemble_uncertainty_positive_when_members_disagree():
    a = np.array([[0.9, 0.05, 0.05]])
    b = np.array([[0.05, 0.9, 0.05]])
    out = ensemble_uncertainty([a, b])
    assert out[0] > 0.1
