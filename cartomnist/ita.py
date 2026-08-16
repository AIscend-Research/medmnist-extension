"""Individual Typology Angle — the axis along which "neutral" resizing is not neutral.

ITA = arctan((L* - 50) / b*) * 180/pi, computed on *unaffected perilesional
skin* rather than on the lesion itself. Del Bino & Bernerd (2013) give the
canonical bin edges, which map onto but are not identical to Fitzpatrick
phototype.

Caveat worth stating in the paper: dermoscopic images are contact-illuminated
and often polarised, so ITA measured from them is a proxy for constitutive
pigmentation, not a colorimetric ground truth. It is used here as a
*stratification variable* — for detecting non-uniform distortion across the
population — not as a clinical measurement. Kinyanjui et al. (2020) use it the
same way on ISIC/HAM10000, which is the comparison point.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from skimage.color import rgb2lab

# Del Bino & Bernerd (2013) bin edges, in degrees, from dark to very light.
FITZ_EDGES = [-np.inf, -30.0, 10.0, 28.0, 41.0, 55.0, np.inf]
FITZ_LABELS = ["dark", "brown", "tan", "intermediate", "light", "very light"]


def _perilesional_mask(shape: Tuple[int, int], border_frac: float = 0.18) -> np.ndarray:
    """Ring of pixels outside the central lesion disc.

    Dermoscopic framing centres the lesion, so the outer frame is skin. Using a
    ring rather than the corners avoids the circular vignette of contact scopes.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)
    return (r > (1.0 - border_frac * 2)) & (r < 1.0)


def image_ita(rgb: np.ndarray) -> float:
    """ITA in degrees for one image. NaN if no usable skin was found."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    lab = rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    m = _perilesional_mask(L.shape)
    # drop hair, vignette, ink marks and specular highlights before averaging
    m &= (L > 25) & (L < 96) & (b > 1.0) & (np.abs(a) < 45)
    if m.sum() < 50:
        m = (L > 25) & (L < 96) & (b > 1.0)
        if m.sum() < 50:
            return float("nan")

    # median is the right statistic here: robust to residual hair and bubbles
    Lm, bm = float(np.median(L[m])), float(np.median(b[m]))
    return float(np.degrees(np.arctan2(Lm - 50.0, bm)))


def batch_ita(rgbs, n_jobs: int = 0) -> np.ndarray:
    from .parallel import concat_arrays, map_chunks

    def _run(chunk):
        return np.array([image_ita(chunk[i]) for i in range(len(chunk))],
                        dtype=np.float32)

    return concat_arrays(map_chunks(rgbs, _run, n_jobs))


# --------------------------------------------------------------------------
def fitzpatrick_bins(ita: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Canonical Del Bino bins. Empty bins are dropped, and the surviving
    labels are returned so plots never show a phantom category."""
    ita = np.asarray(ita, dtype=np.float32)
    idx = np.digitize(np.nan_to_num(ita, nan=-999), FITZ_EDGES) - 1
    idx = np.clip(idx, 0, len(FITZ_LABELS) - 1)
    idx[~np.isfinite(ita)] = -1
    return idx.astype(np.int32), list(FITZ_LABELS)


def tertile_bins(ita: np.ndarray, n: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Population-relative bins.

    HAM10000 is overwhelmingly light-skinned, so the canonical bins leave the
    darker categories with a handful of images and unusable confidence
    intervals. Quantile bins keep every stratum populated; both are reported,
    and the paper should show both, because choosing only the flattering one is
    exactly the move Harley warns about.
    """
    ita = np.asarray(ita, dtype=np.float32)
    ok = np.isfinite(ita)
    qs = np.quantile(ita[ok], np.linspace(0, 1, n + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.digitize(ita, qs) - 1
    idx = np.clip(idx, 0, n - 1)
    idx[~ok] = -1
    labels = [f"ITA q{i+1} [{qs[i]:.0f}°,{qs[i+1]:.0f}°)" for i in range(n)]
    labels[0] = f"ITA q1 (darkest, <{qs[1]:.0f}°)"
    labels[-1] = f"ITA q{n} (lightest, ≥{qs[-2]:.0f}°)"
    return idx.astype(np.int32), labels


def make_bins(ita: np.ndarray, mode: str = "fitzpatrick"):
    if mode == "tertile":
        return tertile_bins(ita)
    idx, labels = fitzpatrick_bins(ita)
    # drop bins with too few members to support a statistic, merging downward
    counts = np.bincount(idx[idx >= 0], minlength=len(labels))
    keep = [i for i, c in enumerate(counts) if c >= 30]
    if len(keep) < 2:
        return tertile_bins(ita)
    remap = {old: new for new, old in enumerate(keep)}
    new_idx = np.full_like(idx, -1)
    for old, new in remap.items():
        new_idx[idx == old] = new
    return new_idx, [f"{labels[i]} (n={counts[i]})" for i in keep]


def summary(ita: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(ita)
    return {
        "n": int(ok.sum()),
        "n_missing": int((~ok).sum()),
        "mean": float(np.mean(ita[ok])),
        "median": float(np.median(ita[ok])),
        "p05": float(np.percentile(ita[ok], 5)),
        "p95": float(np.percentile(ita[ok], 95)),
    }
