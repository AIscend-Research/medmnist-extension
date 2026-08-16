"""Data access and the cached construction of the three 28x28 regimes.

Regimes under comparison
------------------------
    naive        MedMNIST as shipped: RGB area-resized 224 -> 28.  3 channels.
    generalized  cartographic generalization of the SAME 224 source.  18 ch.
    native224    the 224 source itself, upper bound on what pixels can buy.

All three come from one 224 source array, so nothing differs but the
generalization policy.

Offline Kaggle
--------------
`load_derma224` tries, in order: an explicit npz path, a Kaggle input dataset
directory, the medmnist package cache, then a download. Set
`CARTO_DATA_DIR=/kaggle/input/<your-medmnist-dataset>` to run with internet off.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

CLASS_NAMES = [
    "actinic keratosis / IEC",     # akiec
    "basal cell carcinoma",        # bcc
    "benign keratosis",            # bkl
    "dermatofibroma",              # df
    "melanoma",                    # mel
    "melanocytic nevus",           # nv
    "vascular lesion",             # vasc
]
CLASS_SHORT = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# --------------------------------------------------------------------------
def _candidate_paths(name: str) -> list[Path]:
    cands: list[Path] = []
    env = os.environ.get("CARTO_DATA_DIR")
    if env:
        cands += [Path(env) / name, Path(env)]
    cands += [
        Path("/kaggle/input"),
        Path.home() / ".medmnist",
        Path.cwd() / "data",
    ]
    return cands


def _find_npz(filename: str) -> Optional[Path]:
    for base in _candidate_paths(filename):
        if base.is_file() and base.name == filename:
            return base
        if base.is_dir():
            direct = base / filename
            if direct.exists():
                return direct
            hits = sorted(base.rglob(filename))
            if hits:
                return hits[0]
    return None


def load_derma224(download: bool = True) -> Dict[str, np.ndarray]:
    """Return {split_images, split_labels} for DermaMNIST at 224x224."""
    fn = "dermamnist_224.npz"
    path = _find_npz(fn)

    if path is None and download:
        try:
            import medmnist
            from medmnist import INFO
            info = INFO["dermamnist"]
            cls = getattr(medmnist, info["python_class"])
            for split in ("train", "val", "test"):
                cls(split=split, download=True, size=224)
            path = _find_npz(fn)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                f"Could not obtain {fn}. With internet disabled, add the "
                f"MedMNIST 224 dataset as a Kaggle input and set "
                f"CARTO_DATA_DIR to its directory. Original error: {exc}"
            ) from exc
    if path is None:
        raise FileNotFoundError(f"{fn} not found; see load_derma224 docstring.")

    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    for split in ("train", "val", "test"):
        out[f"{split}_labels"] = out[f"{split}_labels"].astype(np.int64).ravel()
    return out


def load_derma28(download: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """The official 28x28 MedMNIST arrays, used only as a sanity check that our
    area-pooled `naive` regime reproduces the shipped benchmark."""
    path = _find_npz("dermamnist.npz")
    if path is None and download:
        try:
            import medmnist
            from medmnist import INFO
            cls = getattr(medmnist, INFO["dermamnist"]["python_class"])
            for split in ("train", "val", "test"):
                cls(split=split, download=True)
            path = _find_npz("dermamnist.npz")
        except Exception:                              # noqa: BLE001
            return None
    if path is None:
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------
def stratified_subsample(labels: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Class-proportional subsample that never empties a rare class.

    Every class keeps at least min(30, n_c) examples; the remainder is drawn
    proportionally. Without the floor, FAST mode would delete `df` and `vasc`
    outright and the rare-class experiment would be vacuous.
    """
    if k <= 0 or k >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    floors = {c: min(30, int((labels == c).sum())) for c in classes}
    budget = k - sum(floors.values())
    chosen = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_extra = 0
        if budget > 0:
            share = (len(idx) - floors[c]) / max(1, len(labels) - sum(floors.values()))
            n_extra = int(round(budget * share))
        take = min(len(idx), floors[c] + max(0, n_extra))
        chosen.append(idx[:take])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return np.sort(out[:k]) if len(out) > k else np.sort(out)


# --------------------------------------------------------------------------
class DermaSource:
    """Holds the 224 source arrays and the index subsets for each split."""

    def __init__(self, cfg, raw: Optional[Dict[str, np.ndarray]] = None):
        """`raw` lets the smoke test inject a synthetic dataset with the same
        keys, so the whole pipeline can be exercised without a download."""
        self.cfg = cfg
        raw = raw if raw is not None else load_derma224()
        self.splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        caps = {"train": cfg.max_train, "val": cfg.max_val, "test": cfg.max_test}
        for split in ("train", "val", "test"):
            x = raw[f"{split}_images"]
            y = raw[f"{split}_labels"]
            idx = stratified_subsample(y, caps[split], seed=1234)
            self.splits[split] = (x[idx], y[idx])
        self.n_classes = int(max(y.max() for _, y in self.splits.values()) + 1)

    def images(self, split: str) -> np.ndarray:
        return self.splits[split][0]

    def labels(self, split: str) -> np.ndarray:
        return self.splits[split][1]

    def class_counts(self, split: str = "train") -> Dict[str, int]:
        y = self.labels(split)
        return {CLASS_SHORT[i]: int((y == i).sum()) for i in range(self.n_classes)}

    def rare_classes(self, threshold: float = 0.05) -> list[int]:
        y = self.labels("train")
        share = np.bincount(y, minlength=self.n_classes) / len(y)
        return [int(i) for i in np.flatnonzero(share < threshold)]


# --------------------------------------------------------------------------
def cached(path: Path, build, force: bool = False) -> np.ndarray:
    """Memoise an expensive array build to .npy on disk."""
    path = Path(path)
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = build()
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)
    return np.load(path, mmap_mode="r")
