"""Töpfer's radical law, as a scale law for benchmark design.

Töpfer & Pillewizer (1966) asked how many features a map should retain when its
scale is reduced, and answered

    n_target = n_source * sqrt(M_source / M_target)

with M the scale denominator. Written in linear resolution r (M ∝ 1/r) that is
n_target = n_source * sqrt(r_target / r_source): halve the resolution, keep
~71% of the features.

The analogue here: how many symbolized structure families are worth carrying at
a given grid resolution before the returns vanish? If the empirical K*(r) tracks
sqrt(r), a benchmark designer can *compute* the symbol budget for a chosen
resolution instead of guessing it — which is the practical deliverable of this
section.

The measurement is deliberately conservative: K*(r) is the smallest K whose mean
test macro-AUC reaches `frac` of the best AUC achieved at that resolution, so
the curve is defined by diminishing returns rather than by a significance test
that would be underpowered at these sample sizes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .filterbanks import STRUCTURE_NAMES
from .generalize import channel_spec
from .metrics import macro_auc, softmax
from .operators import topfer_n


# --------------------------------------------------------------------------
def structure_importance(x: np.ndarray, y: np.ndarray, spec,
                         structures: Sequence[str] = tuple(STRUCTURE_NAMES),
                         seed: int = 0) -> Dict[str, float]:
    """Rank structure families by a cheap linear probe.

    One multinomial logistic regression per family, on that family's channels
    only, scored by macro one-vs-rest AUC on a held-out third. Cheap, and
    critically it is fitted on TRAIN ONLY, so the selection order used by the
    sweep never sees the test split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    flat = np.asarray(x, dtype=np.float32).reshape(n, x.shape[1], -1)
    out: Dict[str, float] = {}
    for s in structures:
        idx = spec.groups[s]
        # mean + std per channel is enough signal for a ranking and keeps the
        # probe at a few dozen features rather than a few thousand
        f = np.concatenate([flat[:, idx].mean(-1), flat[:, idx].std(-1)], axis=1)
        try:
            xa, xb, ya, yb = train_test_split(f, y, test_size=0.33,
                                              random_state=seed, stratify=y)
            sc = StandardScaler().fit(xa)
            clf = LogisticRegression(max_iter=400, C=1.0, multi_class="auto")
            clf.fit(sc.transform(xa), ya)
            p = clf.predict_proba(sc.transform(xb))
            out[s] = float(macro_auc(p, yb))
        except Exception:                              # noqa: BLE001
            out[s] = 0.0
    return out


# --------------------------------------------------------------------------
@dataclass
class TopferPoint:
    resolution: int
    k: int
    macro_auc: float
    balanced_accuracy: float
    n_channels: int


def run_sweep(tensors_by_res: Dict[int, Dict[str, np.ndarray]],
              labels: Dict[str, np.ndarray],
              order: Sequence[str], cfg, n_classes: int,
              with_source_diagram: bool = True) -> List[TopferPoint]:
    """Train one model per (resolution, K) cell of the sweep.

    `tensors_by_res[r]["train"|"val"|"test"]` are the generalized tensors at
    resolution r, all carrying the FULL channel set; K is applied by channel
    selection rather than by regenerating, so every cell of the sweep sees
    numerically identical symbol measurements.
    """
    from .metrics import per_class_recall
    from .train import EquivariantAug, train_eval

    full = channel_spec(STRUCTURE_NAMES, with_source_diagram)
    points: List[TopferPoint] = []

    for r in sorted(tensors_by_res):
        t = tensors_by_res[r]
        for k in cfg.topfer_k_values:
            keep = list(order[:k])
            chans = full.idx("rgb", *keep)
            if with_source_diagram:
                chans = sorted(set(chans) | set(full.groups["source_diagram"]))
            cos_i = [chans.index(i) for i, nm in enumerate(full.names)
                     if "cos2theta" in nm and i in chans]
            sin_i = [chans.index(i) for i, nm in enumerate(full.names)
                     if "sin2theta" in nm and i in chans]
            aug = EquivariantAug(cos_i, sin_i)

            aucs, baccs = [], []
            for seed in cfg.topfer_seeds:
                res = train_eval(
                    t["train"], labels["train"], t["val"], labels["val"],
                    t["test"], labels["test"],
                    regime=f"topfer_r{r}_k{k}", n_classes=n_classes, cfg=cfg,
                    channels=chans, aug=aug, seed=seed,
                    epochs=cfg.topfer_epochs, verbose=False)
                p = softmax(res.test_logits)
                aucs.append(macro_auc(p, res.test_y))
                baccs.append(float(np.nanmean(
                    per_class_recall(p, res.test_y, n_classes))))
            points.append(TopferPoint(r, k, float(np.mean(aucs)),
                                      float(np.mean(baccs)), len(chans)))
            print(f"  Töpfer r={r:>2} K={k}: AUC {points[-1].macro_auc:.4f}")
    return points


def fit_scale_law(points: Sequence[TopferPoint], frac: float = 0.99,
                  source_resolution: int = 224) -> Dict[str, object]:
    """Extract K*(r) and compare it with Töpfer's prediction.

    Returns the empirical knee per resolution, a fitted power law K* ∝ r^alpha
    (Töpfer predicts alpha = 0.5), and the anchored Töpfer curve.
    """
    by_r: Dict[int, List[TopferPoint]] = {}
    for p in points:
        by_r.setdefault(p.resolution, []).append(p)

    kstar: Dict[int, int] = {}
    curves: Dict[int, Dict[str, List[float]]] = {}
    for r, ps in by_r.items():
        ps = sorted(ps, key=lambda p: p.k)
        aucs = np.array([p.macro_auc for p in ps])
        ks = np.array([p.k for p in ps])
        curves[r] = {"k": ks.tolist(), "auc": aucs.tolist()}
        if not np.isfinite(aucs).any():
            continue
        thresh = float(np.nanmax(aucs)) * frac
        ok = np.flatnonzero(aucs >= thresh)
        kstar[r] = int(ks[ok[0]]) if len(ok) else int(ks[int(np.nanargmax(aucs))])

    rs = np.array(sorted(kstar), dtype=float)
    ks = np.array([kstar[int(r)] for r in rs], dtype=float)

    def _loglog(x, y):
        A = np.vstack([np.log(x), np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        return float(coef[0]), float(coef[1])

    # The power-law fit is on log K*, so resolutions where no symbol budget paid
    # for itself (K* = 0) cannot enter it. Those are reported rather than
    # silently dropped, and a shifted fit on log(K* + 1) is always defined so
    # the plate still has a curve to draw.
    alpha = intercept = float("nan")
    pos = ks > 0
    diagnostic = ""
    if pos.sum() >= 2:
        alpha, intercept = _loglog(rs[pos], ks[pos])
    else:
        diagnostic = (
            f"Only {int(pos.sum())} of {len(rs)} resolutions had K* > 0, so the "
            "unshifted power law is undefined. Either the symbols bought nothing "
            "at these resolutions, or the sweep is underpowered — check that the "
            "K=0 arm is not already at ceiling.")
    alpha_shifted, intercept_shifted = (_loglog(rs, ks + 1.0) if len(rs) >= 2
                                        else (float("nan"), float("nan")))

    # anchor Töpfer at the largest measured resolution
    topfer_curve = {}
    if len(rs):
        r_anchor, k_anchor = float(rs[-1]), float(ks[-1])
        for r in rs:
            topfer_curve[int(r)] = topfer_n(k_anchor, r_anchor, float(r))

    return {"kstar": {int(k): int(v) for k, v in kstar.items()},
            "curves": {int(k): v for k, v in curves.items()},
            "alpha": alpha, "log_intercept": intercept,
            "alpha_shifted": alpha_shifted,
            "log_intercept_shifted": intercept_shifted,
            "n_resolutions_with_positive_kstar": int(pos.sum()),
            "diagnostic": diagnostic,
            "topfer_alpha": 0.5, "topfer_curve": topfer_curve,
            "frac_of_max": frac, "source_resolution": source_resolution,
            "interpretation": (
                "alpha near 0.5 means the symbol budget for a benchmark at "
                "resolution r can be read off Töpfer's radical law; alpha "
                "well above 0.5 means low-resolution grids saturate on symbols "
                "faster than cartographic practice would predict.")}
