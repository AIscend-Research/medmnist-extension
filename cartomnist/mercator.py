"""The Mercator test: is the "neutral" transform neutral?

Mercator preserves angles everywhere and area nowhere, and the areas it inflates
are not randomly distributed — they are the high latitudes. The projection is not
biased because someone chose to favour Greenland; it is biased because a
uniform rule applied to a non-uniform world produces non-uniform error, and the
map does not say so.

Naive downsampling is the same kind of object. It applies one low-pass filter to
every image. But the amplitude of a dermoscopic structure in L* is not
independent of constitutive pigmentation: on darker skin the same structure
presents at lower luminance contrast, so more of it falls below the post-resize
noise floor. A uniform rule, a non-uniform loss.

This module measures that directly, without a classifier in the loop, so the
result cannot be explained away by label distribution or by training dynamics:

    retention(image, structure) = R^2 of a probe that tries to reconstruct the
    NATIVE-resolution structure density map from what the regime kept.

Then it asks whether retention varies with ITA, for three regimes:

    naive                    expected: falls with decreasing ITA (the Mercator
                             result)
    generalized (equal-fid)  expected: flat by construction
    generalized (absolute)   expected: the distortion returns — showing that
                             equal fidelity is a design choice inside the
                             framework, not an automatic property of it
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .filterbanks import STRUCTURE_NAMES


# --------------------------------------------------------------------------
def _patch_features(x: np.ndarray, k: int = 3) -> np.ndarray:
    """(N, C, H, W) -> (N*H*W, C*k*k + 1): each cell's k x k neighbourhood.

    The probe is deliberately *local and translation-invariant* rather than a
    whole-image regression. A whole-image ridge has C*H*W features (14k at 18
    channels) against N images, which at any realistic N is wildly
    under-determined — the regularisation, not the representation, would decide
    the answer. The local form has C*k*k features (162) against N*H*W samples
    (millions), so the fit is over-determined by four orders of magnitude and
    the number it returns is about the representation.

    It also asks the right question: can the structure content of *this cell* be
    recovered from what the regime kept *near this cell*.
    """
    x = np.asarray(x, dtype=np.float32)
    n, c, h, w = x.shape
    pad = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="edge")
    feats = np.empty((n, c * k * k, h, w), dtype=np.float32)
    i = 0
    for dy in range(k):
        for dx in range(k):
            feats[:, i * c:(i + 1) * c] = xp[:, :, dy:dy + h, dx:dx + w]
            i += 1
    flat = feats.transpose(0, 2, 3, 1).reshape(n * h * w, c * k * k)
    return np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float32)], axis=1)


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = X.shape[1]
    A = X.T.astype(np.float64) @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T.astype(np.float64) @ Y.astype(np.float64))


def _standardize_rows(Y: np.ndarray) -> np.ndarray:
    """Per-image zero-mean, unit-variance.

    This makes retention a statement about *spatial layout*, not amplitude,
    which is the only way the comparison can be fair. Equal-fidelity
    normalisation deliberately discards per-image amplitude — scoring against a
    raw-amplitude target would therefore penalise it for doing exactly what it
    was designed to do, and would make the experiment circular in the other
    direction. Layout is what a symbol is supposed to carry, so layout is what
    is measured.
    """
    m = Y.mean(axis=1, keepdims=True)
    s = Y.std(axis=1, keepdims=True) + 1e-8
    return (Y - m) / s


def _per_row_r2(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """R^2 per image. Y is already row-standardised, so ss_tot == n per row and
    a probe that predicts the global mean map scores ~0."""
    ss_res = ((Y - P) ** 2).sum(axis=1)
    ss_tot = (Y ** 2).sum(axis=1) + 1e-9
    return 1.0 - ss_res / ss_tot


def retention(regime_train: np.ndarray, regime_test: np.ndarray,
              target_train: np.ndarray, target_test: np.ndarray,
              lam: float = 10.0, n_fit_images: int = 400,
              batch: int = 256, seed: int = 0) -> np.ndarray:
    """Per-image R^2 of reconstructing the native structure layout from a regime.

    The target is the *un-normalised* native structure map (see
    `generalize.raw_structure_maps`), row-standardised. The probe is fitted on
    train and evaluated per test image, so no regime can win by memorising the
    test set, and none can win by having produced the target itself.
    """
    rng = np.random.default_rng(seed)
    tr = np.asarray(regime_train, dtype=np.float32)
    n_fit = min(n_fit_images, len(tr))
    sel = np.sort(rng.choice(len(tr), n_fit, replace=False))

    Xtr = _patch_features(tr[sel])
    Ytr = _standardize_rows(np.asarray(target_train, np.float32)[sel]
                            .reshape(n_fit, -1)).reshape(-1)
    W = _ridge_fit(Xtr, Ytr[:, None], lam)
    del Xtr

    te = np.asarray(regime_test, dtype=np.float32)
    Yte = _standardize_rows(np.asarray(target_test, np.float32)
                            .reshape(len(te), -1))
    cells = te.shape[-1] * te.shape[-2]
    out = np.empty(len(te), dtype=np.float32)
    for a in range(0, len(te), batch):
        b = min(a + batch, len(te))
        pred = (_patch_features(te[a:b]) @ W).reshape(b - a, cells)
        out[a:b] = _per_row_r2(Yte[a:b], pred)
    return out


# --------------------------------------------------------------------------
@dataclass
class MercatorResult:
    regime: str
    structure: str
    per_bin_mean: Dict[str, float]
    per_bin_sem: Dict[str, float]
    spread: float                # max - min across bins: the distortion
    slope_per_10deg: float       # OLS of retention on ITA
    slope_p: float               # permutation p-value for slope != 0
    n_per_bin: Dict[str, int]

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _slope_and_p(ita: np.ndarray, ret: np.ndarray, n_perm: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    ok = np.isfinite(ita) & np.isfinite(ret)
    if ok.sum() < 20:
        return float("nan"), float("nan")
    a, b = ita[ok].astype(np.float64), ret[ok].astype(np.float64)
    a = (a - a.mean())
    slope = float((a @ (b - b.mean())) / (a @ a + 1e-12))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    bc = b - b.mean()
    for i in range(n_perm):
        null[i] = (a @ rng.permutation(bc)) / (a @ a + 1e-12)
    p = float((np.abs(null) >= abs(slope)).mean())
    return slope * 10.0, p       # per 10 degrees of ITA


def analyse(regimes: Dict[str, Dict[str, np.ndarray]],
            targets_train: Dict[str, np.ndarray],
            targets_test: Dict[str, np.ndarray],
            ita_test: np.ndarray, bin_idx: np.ndarray, bin_labels: Sequence[str],
            structures: Sequence[str] = tuple(STRUCTURE_NAMES),
            seed: int = 0) -> List[MercatorResult]:
    """Run the full comparison.

    Parameters
    ----------
    regimes : {"naive": {"train": X, "test": X}, "generalized_ef": {...}, ...}
        The representations being compared. Each value is (N, C, t, t).
    targets_* : {structure: (N, t, t)}
        The native-resolution structure density maps, area-pooled to the target
        grid — i.e. the ground truth of "what was actually there", measured
        before any regime touched the image.
    """
    out: List[MercatorResult] = []
    for rname, arrs in regimes.items():
        for s in structures:
            r = retention(arrs["train"], arrs["test"],
                          targets_train[s], targets_test[s])
            per_mean, per_sem, per_n = {}, {}, {}
            for b, lab in enumerate(bin_labels):
                m = bin_idx == b
                if m.sum() < 10:
                    continue
                v = r[m]
                per_mean[lab] = float(np.nanmean(v))
                per_sem[lab] = float(np.nanstd(v) / np.sqrt(max(1, np.isfinite(v).sum())))
                per_n[lab] = int(m.sum())
            vals = [v for v in per_mean.values() if np.isfinite(v)]
            spread = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
            slope, p = _slope_and_p(ita_test, r, seed=seed)
            out.append(MercatorResult(rname, s, per_mean, per_sem, spread,
                                      slope, p, per_n))
    return out


# --------------------------------------------------------------------------
def symbol_dynamic_range(tensors: np.ndarray, spec, ita: np.ndarray,
                         bin_idx: np.ndarray, bin_labels: Sequence[str],
                         structures: Sequence[str] = tuple(STRUCTURE_NAMES)
                         ) -> Dict[str, Dict[str, float]]:
    """How much of the [0,1] symbol range each ITA stratum actually gets to use.

    This is the mechanism behind the retention result, stated in the most
    concrete possible terms: if darker-skinned subjects' symbols live in the
    bottom fifth of the channel, the network has fewer usable levels for them,
    and no amount of training fixes it. Under equal-fidelity normalisation these
    numbers should be indistinguishable across strata.
    """
    out: Dict[str, Dict[str, float]] = {}
    for s in structures:
        d_idx = [i for i in spec.groups[s] if spec.names[i].endswith("::density")]
        if not d_idx:
            continue
        d = np.asarray(tensors[:, d_idx[0]], dtype=np.float32).reshape(len(tensors), -1)
        p95 = np.percentile(d, 95, axis=1)
        row = {}
        for b, lab in enumerate(bin_labels):
            m = bin_idx == b
            if m.sum() < 10:
                continue
            row[lab] = float(np.mean(p95[m]))
        out[s] = row
    return out


def summarise(results: Sequence[MercatorResult]) -> Dict[str, Dict[str, float]]:
    """Headline numbers: mean distortion spread per regime, and how many
    structures show a statistically detectable ITA slope."""
    agg: Dict[str, Dict[str, float]] = {}
    for r in results:
        a = agg.setdefault(r.regime, {"mean_spread": 0.0, "n": 0,
                                      "n_significant_slopes": 0,
                                      "max_spread": 0.0})
        if np.isfinite(r.spread):
            a["mean_spread"] += r.spread
            a["max_spread"] = max(a["max_spread"], r.spread)
            a["n"] += 1
        if np.isfinite(r.slope_p) and r.slope_p < 0.05:
            a["n_significant_slopes"] += 1
    for a in agg.values():
        a["mean_spread"] = a["mean_spread"] / max(1, a["n"])
    return agg
