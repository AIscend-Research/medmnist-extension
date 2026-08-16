"""The generalization operators themselves.

Each function implements exactly one contract from `cartomnist.contracts`, and
each contract's `invariant` field names a property that `tests/` actually
checks. Keep them pure: numpy in, numpy out, no config lookups.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------
# AGGREGATION — area-mean pooling, conserves the areal integral
# --------------------------------------------------------------------------
def aggregate(x: np.ndarray, target: int) -> np.ndarray:
    """Area-mean pool an (H, W) or (H, W, C) map onto a `target` x `target` grid.

    Contract invariant: mean(out) == mean(x) exactly when H % target == 0.
    """
    x = np.asarray(x, dtype=np.float32)
    single = x.ndim == 2
    if single:
        x = x[..., None]
    H, W, C = x.shape
    if H % target or W % target:
        # pad by edge replication to the next multiple, then pool
        ph, pw = (-H) % target, (-W) % target
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")
        H, W = x.shape[:2]
    fh, fw = H // target, W // target
    out = x.reshape(target, fh, target, fw, C).mean(axis=(1, 3))
    return out[..., 0] if single else out


def aggregate_orientation(theta: np.ndarray, weight: np.ndarray,
                          target: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool an orientation field correctly, i.e. in double-angle space.

    Orientations live on the projective line: 0 and pi are the same bearing, so
    averaging them arithmetically gives pi/2, which is a right angle wrong. The
    resultant length doubles as the pooled coherence.
    """
    c = aggregate(weight * np.cos(2.0 * theta), target)
    s = aggregate(weight * np.sin(2.0 * theta), target)
    w = aggregate(weight, target) + 1e-12
    cbar, sbar = c / w, s / w
    return cbar.astype(np.float32), sbar.astype(np.float32)


# --------------------------------------------------------------------------
# SIMPLIFICATION — do not draw a bearing you cannot place
# --------------------------------------------------------------------------
def simplify_orientation(cos2t: np.ndarray, sin2t: np.ndarray,
                         coherence: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero the orientation channels wherever coherence is below `tau`.

    Contract invariant: out == 0 exactly on {coherence < tau}.
    """
    keep = (coherence >= tau).astype(np.float32)
    return (cos2t * keep).astype(np.float32), (sin2t * keep).astype(np.float32)


# --------------------------------------------------------------------------
# EXAGGERATION — draw the sub-pixel road wider than it is
# --------------------------------------------------------------------------
def exaggerate(x: np.ndarray, gamma: float) -> np.ndarray:
    """Strictly monotone amplitude expansion of the faint end.

    Contract invariant: strictly increasing on [0, 1] for gamma > 0, so rank
    order — and therefore every rank-based statistic — is preserved exactly.
    """
    return np.power(np.clip(x, 0.0, 1.0), float(gamma)).astype(np.float32)


# --------------------------------------------------------------------------
# DISPLACEMENT — resolve symbol congestion without reordering
# --------------------------------------------------------------------------
def displace(symbols: np.ndarray, lam: float) -> np.ndarray:
    """Attenuate co-located symbols in congested cells.

    `symbols` is (T, H, W) — T structure-density planes on the target grid.
    Where the per-cell total exceeds 1 (the cell is "full"), every symbol is
    scaled by the same per-cell factor, so no symbol is erased and the
    within-cell ordering is untouched. Where the cell is not congested the
    operator is exactly the identity.

    Contract invariant: argsort(symbols[:, i, j]) is unchanged for all (i, j),
    and displace(x)[.., i, j] == x[.., i, j] wherever sum_t x[t, i, j] <= 1.
    """
    total = symbols.sum(axis=0)
    congestion = np.clip(total - 1.0, 0.0, None)
    factor = 1.0 / (1.0 + float(lam) * congestion)
    return (symbols * factor[None]).astype(np.float32)


# --------------------------------------------------------------------------
# SELECTION — Töpfer's radical law, and the fixed top-K variant
# --------------------------------------------------------------------------
def topfer_n(n_source: float, scale_source: float, scale_target: float) -> float:
    """Töpfer & Pillewizer's radical law.

    n_f = n_a * sqrt(M_a / M_f), where M is the scale *denominator*: a 1:1,000,000
    chart has M = 1e6. Working in linear resolution r instead (M ∝ 1/r), the law
    becomes n_target = n_source * sqrt(r_target / r_source).
    """
    return float(n_source) * float(np.sqrt(scale_target / scale_source))


def select(names: Sequence[str], keep: Sequence[str]) -> List[int]:
    """Indices of the retained structure families, in canonical order."""
    keep = set(keep)
    return [i for i, n in enumerate(names) if n in keep]


def rank_structures(importance: Dict[str, float]) -> List[str]:
    """Structure families, most informative first. Ties broken by name so the
    ordering is deterministic and reproducible across runs."""
    return [k for k, _ in sorted(importance.items(), key=lambda kv: (-kv[1], kv[0]))]
