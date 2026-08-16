"""Do the symbols track what dermatologists actually annotate?

This is the caveat the proposal names first, and it deserves a real answer:
symbolized channels are only as good as the filter banks, and a filter bank that
fires on compression artefacts would still look impressive in an ablation.

ISIC 2018 Task 2 ships expert binary masks for five dermoscopic attributes on
2,594 images, which is exactly the instrument needed. Three of them map onto our
vocabulary:

    ISIC "pigment_network"   -> pigment_network
    ISIC "globules"          -> dots_globules
    ISIC "streaks"           -> streaks

(ISIC's `negative_network` and `milia_like_cyst` have no counterpart in this
filter bank; they are reported as uncovered vocabulary rather than quietly
dropped, because "which features survive" is the whole argument.)

Two scores are reported per attribute:

    pixel AUC   — does the symbol channel rank annotated cells above
                  unannotated ones, within each image? Computed per image and
                  averaged, so large lesions do not dominate.
    image AUC   — does the image-level symbol density separate images where the
                  attribute is present from images where it is absent?

If pixel AUC sits near 0.5 for a structure, that structure's symbol is not
measuring what its name claims, and the paper must say so.

The module degrades gracefully: if the ISIC data is not present it returns
`{"available": False, ...}` and the pipeline continues.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ATTR_MAP = {
    "pigment_network": "pigment_network",
    "globules": "dots_globules",
    "streaks": "streaks",
}
UNCOVERED = ["negative_network", "milia_like_cyst"]


# --------------------------------------------------------------------------
def find_isic_root() -> Optional[Path]:
    """Locate an ISIC 2018 Task 2 directory tree."""
    cands: List[Path] = []
    env = os.environ.get("CARTO_ISIC_DIR")
    if env:
        cands.append(Path(env))
    cands += [Path("/kaggle/input"), Path.cwd() / "data"]
    for base in cands:
        if not base.exists():
            continue
        if (base / "ISIC2018_Task1-2_Training_Input").exists():
            return base
        for hit in base.glob("*/ISIC2018_Task1-2_Training_Input"):
            return hit.parent
        for hit in base.glob("*/*/ISIC2018_Task1-2_Training_Input"):
            return hit.parent
    return None


def _image_dir(root: Path) -> Path:
    return root / "ISIC2018_Task1-2_Training_Input"


def _mask_dir(root: Path) -> Path:
    for name in ("ISIC2018_Task2_Training_GroundTruth_v3",
                 "ISIC2018_Task2_Training_GroundTruth"):
        if (root / name).exists():
            return root / name
    return root / "ISIC2018_Task2_Training_GroundTruth_v3"


def list_cases(root: Path, limit: int = 300) -> List[str]:
    ids = sorted(re.sub(r"\.jpg$", "", p.name)
                 for p in _image_dir(root).glob("ISIC_*.jpg"))
    return ids[:limit]


# --------------------------------------------------------------------------
def validate(target: int = 28, native: int = 224, limit: int = 300,
             equal_fidelity: bool = True, seed: int = 0) -> Dict:
    """Run the validation. Returns a JSON-serialisable dict."""
    root = find_isic_root()
    if root is None:
        return {"available": False,
                "reason": "ISIC 2018 Task 2 not found; set CARTO_ISIC_DIR or "
                          "add the dataset as a Kaggle input.",
                "uncovered_attributes": UNCOVERED}

    from PIL import Image
    from sklearn.metrics import roc_auc_score

    from .generalize import channel_spec, generalize
    from .filterbanks import STRUCTURE_NAMES

    spec = channel_spec()
    ids = list_cases(root, limit)
    if not ids:
        return {"available": False, "reason": f"no images under {root}",
                "uncovered_attributes": UNCOVERED}

    pixel_scores: Dict[str, List[float]] = {v: [] for v in ATTR_MAP.values()}
    img_density: Dict[str, List[float]] = {v: [] for v in ATTR_MAP.values()}
    img_present: Dict[str, List[int]] = {v: [] for v in ATTR_MAP.values()}
    n_used = 0

    for iid in ids:
        ip = _image_dir(root) / f"{iid}.jpg"
        if not ip.exists():
            continue
        img = np.asarray(Image.open(ip).convert("RGB").resize(
            (native, native), Image.BILINEAR), dtype=np.float32) / 255.0
        t = generalize(img, target=target, equal_fidelity=equal_fidelity)
        n_used += 1

        for attr, ours in ATTR_MAP.items():
            mp = _mask_dir(root) / f"{iid}_attribute_{attr}.png"
            if not mp.exists():
                continue
            m = np.asarray(Image.open(mp).convert("L").resize(
                (target, target), Image.BILINEAR), dtype=np.float32) / 255.0
            gt = (m > 0.2).astype(np.int32).ravel()

            ch = [i for i in spec.groups[ours] if spec.names[i].endswith("::density")]
            d = np.asarray(t[ch[0]]).ravel()

            img_density[ours].append(float(np.percentile(d, 95)))
            img_present[ours].append(int(gt.sum() > 0))
            if 0 < gt.sum() < len(gt):
                pixel_scores[ours].append(float(roc_auc_score(gt, d)))

    out: Dict[str, object] = {"available": True, "root": str(root),
                              "n_images": n_used,
                              "uncovered_attributes": UNCOVERED,
                              "attributes": {}}
    for ours in ATTR_MAP.values():
        px = pixel_scores[ours]
        entry: Dict[str, object] = {
            "n_annotated_images": len(px),
            "pixel_auc_mean": float(np.mean(px)) if px else float("nan"),
            "pixel_auc_sem": (float(np.std(px) / np.sqrt(len(px))) if px
                              else float("nan")),
            "pixel_auc_frac_above_chance": (float(np.mean(np.asarray(px) > 0.5))
                                            if px else float("nan")),
        }
        y = np.asarray(img_present[ours])
        d = np.asarray(img_density[ours])
        if len(y) > 10 and 0 < y.sum() < len(y):
            entry["image_auc"] = float(roc_auc_score(y, d))
            entry["prevalence"] = float(y.mean())
        out["attributes"][ours] = entry           # type: ignore[index]

    covered = [v for v in ATTR_MAP.values()]
    out["verdict"] = (
        "Symbol channels are validated against expert annotation for "
        f"{', '.join(covered)}. Vessels and blue-white veil have no ISIC Task 2 "
        "counterpart and remain unvalidated — they are reported in the paper as "
        "measured-but-unvalidated channels."
    )
    return out
