"""Metrics, with calibration treated as a first-class result.

The thesis of this project is that a benchmark should ship a reliability
diagram, so calibration is not an afterthought here: every evaluation returns
temperature-scaled probabilities (temperature fitted on the validation split
only), the reliability bins needed to draw the diagram, and the coverage-risk
curve that the sufficiency certificate is scored against.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

# np.trapz was renamed in NumPy 2.0; Kaggle images are not always on 2.x.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 200) -> float:
    """Golden-section search on validation NLL. One parameter, no autograd."""
    def nll(T: float) -> float:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
        if b - a < 1e-4:
            break
    return float((a + b) / 2.0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """Equal-width confidence bins on the top-1 probability.

    Returns bin centres, per-bin accuracy, per-bin mean confidence and counts —
    everything the diagram needs, plus ECE / MCE / Brier.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)

    acc = np.full(n_bins, np.nan)
    avg_conf = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            acc[b] = correct[m].mean()
            avg_conf[b] = conf[m].mean()

    valid = counts > 0
    w = counts[valid] / counts.sum()
    gap = np.abs(acc[valid] - avg_conf[valid])
    ece = float((w * gap).sum())
    mce = float(gap.max()) if gap.size else float("nan")

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    return {"bin_centers": (edges[:-1] + edges[1:]) / 2, "bin_edges": edges,
            "accuracy": acc, "confidence": avg_conf, "counts": counts,
            "ece": ece, "mce": mce, "brier": brier}


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def macro_auc(probs: np.ndarray, y: np.ndarray) -> float:
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    try:
        if len(present) == probs.shape[1]:
            return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        p = probs[:, present]
        p = p / p.sum(axis=1, keepdims=True)
        remap = {c: i for i, c in enumerate(present)}
        return float(roc_auc_score(np.array([remap[v] for v in y]), p,
                                   multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def per_class_recall(probs: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    pred = probs.argmax(axis=1)
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y == c
        if m.sum():
            out[c] = float((pred[m] == c).mean())
    return out


def per_class_auc(probs: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = (y == c).astype(int)
        if 0 < m.sum() < len(m):
            out[c] = float(roc_auc_score(m, probs[:, c]))
    return out


# --------------------------------------------------------------------------
# Abstention / coverage-risk
# --------------------------------------------------------------------------
def coverage_risk(score: np.ndarray, probs: np.ndarray, y: np.ndarray,
                  n_points: int = 40) -> Dict[str, np.ndarray]:
    """Risk (1 - balanced accuracy) as a function of coverage.

    `score` is the *trust* signal — higher means keep. Sweeping the abstention
    threshold gives the curve; the area under it (lower is better) is the
    single number reported.
    """
    order = np.argsort(-np.asarray(score, dtype=np.float64))
    pred = probs.argmax(axis=1)
    n = len(y)
    covs, risks, bal = [], [], []
    for frac in np.linspace(0.1, 1.0, n_points):
        k = max(2, int(round(frac * n)))
        sel = order[:k]
        yy, pp = y[sel], pred[sel]
        covs.append(k / n)
        risks.append(float((yy != pp).mean()))
        rec = [float((pp[yy == c] == c).mean()) for c in np.unique(yy)]
        bal.append(1.0 - float(np.mean(rec)))
    covs, risks, bal = map(np.asarray, (covs, risks, bal))
    return {"coverage": covs, "risk": risks, "balanced_risk": bal,
            "aurc": float(_trapz(risks, covs) / (covs[-1] - covs[0])),
            "balanced_aurc": float(_trapz(bal, covs) / (covs[-1] - covs[0]))}


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(fn, probs: np.ndarray, y: np.ndarray, n: int = 200,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    base = fn(probs, y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        v = fn(probs[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return base, float("nan"), float("nan")
    return base, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------
@dataclass
class Evaluation:
    regime: str
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_auc: float
    macro_auc_lo: float
    macro_auc_hi: float
    rare_recall: float                       # macro recall over rare classes
    rare_auc: float
    ece: float
    ece_uncalibrated: float
    mce: float
    brier: float
    temperature: float
    per_class_recall: List[float] = field(default_factory=list)
    per_class_auc: List[float] = field(default_factory=list)
    ita_balanced_accuracy: Dict[str, float] = field(default_factory=dict)
    ita_auc: Dict[str, float] = field(default_factory=dict)
    ita_gap: float = float("nan")
    ita_ece: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(logits: np.ndarray, y: np.ndarray, *, regime: str, n_classes: int,
             rare_classes: Sequence[int],
             val_logits: Optional[np.ndarray] = None,
             val_y: Optional[np.ndarray] = None,
             ita_bin: Optional[np.ndarray] = None,
             ita_labels: Optional[Sequence[str]] = None,
             n_bins: int = 15, temperature_scale: bool = True) -> tuple[Evaluation, Dict]:
    """Full evaluation. Returns (Evaluation, extras-for-plotting)."""
    raw_probs = softmax(logits)
    T = 1.0
    if temperature_scale and val_logits is not None and val_y is not None:
        T = fit_temperature(val_logits, val_y)
    probs = softmax(logits / T)

    rel = reliability(probs, y, n_bins)
    rel_raw = reliability(raw_probs, y, n_bins)
    pcr = per_class_recall(probs, y, n_classes)
    pca = per_class_auc(probs, y, n_classes)
    auc, lo, hi = bootstrap_ci(macro_auc, probs, y)

    rare = [c for c in rare_classes if np.isfinite(pcr[c])]
    ita_bacc: Dict[str, float] = {}
    ita_auc: Dict[str, float] = {}
    ita_ece: Dict[str, float] = {}
    if ita_bin is not None and ita_labels is not None:
        for b, lab in enumerate(ita_labels):
            m = ita_bin == b
            if m.sum() < 25:
                continue
            r = per_class_recall(probs[m], y[m], n_classes)
            ita_bacc[lab] = float(np.nanmean(r))
            ita_auc[lab] = macro_auc(probs[m], y[m])
            ita_ece[lab] = reliability(probs[m], y[m], n_bins)["ece"]
    gaps = [v for v in ita_bacc.values() if np.isfinite(v)]
    ita_gap = float(max(gaps) - min(gaps)) if len(gaps) >= 2 else float("nan")

    ev = Evaluation(
        regime=regime, n=int(len(y)),
        accuracy=float((probs.argmax(1) == y).mean()),
        balanced_accuracy=float(np.nanmean(pcr)),
        macro_auc=auc, macro_auc_lo=lo, macro_auc_hi=hi,
        rare_recall=float(np.nanmean([pcr[c] for c in rare])) if rare else float("nan"),
        rare_auc=float(np.nanmean([pca[c] for c in rare])) if rare else float("nan"),
        ece=rel["ece"], ece_uncalibrated=rel_raw["ece"], mce=rel["mce"],
        brier=rel["brier"], temperature=float(T),
        per_class_recall=[float(v) for v in pcr],
        per_class_auc=[float(v) for v in pca],
        ita_balanced_accuracy=ita_bacc, ita_auc=ita_auc, ita_gap=ita_gap,
        ita_ece=ita_ece,
    )
    return ev, {"probs": probs, "raw_probs": raw_probs,
                "reliability": rel, "reliability_raw": rel_raw}
