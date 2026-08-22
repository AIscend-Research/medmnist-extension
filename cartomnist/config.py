"""Global configuration for the cartographic generalization pipeline.

Everything the experiment needs to be reproducible lives here. The notebook
overrides a handful of fields; nothing else in the package reads globals.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------
# Where things live
# --------------------------------------------------------------------------
def _default_root() -> Path:
    for cand in ("/kaggle/working", os.environ.get("CARTO_ROOT", "")):
        if cand and Path(cand).exists():
            return Path(cand)
    return Path.cwd()


@dataclass
class Paths:
    root: Path = field(default_factory=_default_root)

    @property
    def out(self) -> Path:
        return self.root / "carto_out"

    @property
    def figures(self) -> Path:
        return self.out / "figures"

    @property
    def results(self) -> Path:
        return self.out / "results"

    @property
    def cache(self) -> Path:
        """Scratch for multi-GB intermediate tensors.

        Deliberately OUTSIDE `out/` so the downloadable bundle never contains
        them. On Kaggle it goes to /kaggle/temp, which is not persisted into the
        notebook's output — so a committed run uploads the report, not 4 GB of
        cached float32.
        """
        kaggle_tmp = Path("/kaggle/temp")
        if kaggle_tmp.exists():
            return kaggle_tmp / "carto_cache"
        return self.root / "carto_cache"

    def mkdirs(self) -> "Paths":
        for p in (self.out, self.figures, self.results, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------
@dataclass
class Config:
    # ---- data
    dataset: str = "dermamnist"
    native_size: int = 224          # the "1:10,000 survey"
    target_size: int = 28           # the "1:1,000,000 chart"
    n_classes: int = 7

    # subsampling (FAST mode on Kaggle's 30h/week GPU budget)
    max_train: int = 0              # 0 == use all
    max_val: int = 0
    max_test: int = 0

    # ---- generalization
    equal_fidelity: bool = True     # per-image robust normalisation of symbols
    exaggeration_gamma: float = 0.55
    displacement_lambda: float = 0.6
    coherence_floor: float = 0.15   # below this, orientation is not drawn
    selection_k: int = 0            # 0 == keep all structures (Töpfer sweep overrides)

    # ---- training
    epochs_small: int = 25
    epochs_native: int = 8
    batch_size: int = 256
    batch_size_native: int = 64
    lr: float = 3e-3
    lr_native: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0    # keep 0 — we care about calibration
    class_weighted_loss: bool = True
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    amp: bool = True

    # ---- Töpfer sweep
    topfer_resolutions: List[int] = field(default_factory=lambda: [7, 14, 28, 56])
    topfer_k_values: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    # 35, not 12 or 18: the sweep trains on pipeline.py's `n_top` subsample
    # (<=1500 images in fast mode), roughly half the main `generalized`
    # regime's training set, which is HALF as many optimizer steps per epoch
    # (confirmed: 5 batches/epoch vs 11 at batch_size=256). 18 epochs still
    # left several real-data cells stuck (confirmed via the plateau warning
    # firing on r=14/r=28, K=2/K=5) with the fitted alpha coming out NaN.
    # OneCycleLR is also parameterized by the *total* epoch count, so
    # underestimating it anneals the LR to ~0 before the model has used
    # enough steps at a useful LR, not just "fewer steps" in a linear sense.
    # 35 is verified directly: the exact r=28,K=5 cell on real cached data
    # reaches AUC 0.780 at 35 epochs, matching the main regime's 0.783.
    topfer_epochs: int = 35
    topfer_seeds: List[int] = field(default_factory=lambda: [0])

    # ---- calibration / abstention
    n_reliability_bins: int = 15
    temperature_scale: bool = True

    # ---- ITA stratification
    ita_bins: str = "fitzpatrick"   # "fitzpatrick" | "tertile"

    # ---- misc
    n_jobs: int = 0                 # 0 == os.cpu_count()
    fast: bool = False
    run_isic_validation: bool = True
    device: str = "auto"

    paths: Paths = field(default_factory=lambda: Paths().mkdirs())

    # ------------------------------------------------------------------
    def apply_fast(self) -> "Config":
        """Shrink the experiment so the whole notebook fits in ~40 GPU-minutes.

        epochs_small=25 (the class default — no separate fast-mode discount
        on this one), not 12 or 18: on real DermaMNIST-224 the 18-channel
        `generalized` regime's training loss sits stuck near ln(7) (chance)
        for the first ~12 epochs before breaking free — the wider stem input
        needs more optimizer steps than the 3-channel `naive` regime does at
        the same learning rate. 12 epochs silently produced a collapsed,
        near-chance `generalized` model (confirmed: test AUC 0.51, rare
        recall 0.00). 18 looked sufficient on a single seed, but a 3-seed,
        fully-reproducible run (fixed noise_ceiling determinism + CPU, where
        this codebase is confirmed bit-exact seed-to-seed) showed 1 of 3
        seeds still landing in the stuck zone at 18 (loss 1.98 -> 1.94, a
        1.9% improvement — the plateau warning below caught it). A separate
        seed=0 diagnostic converged cleanly all the way through 25 epochs,
        which is also already the class default used for the non-fast run —
        so fast mode gets no special discount on this number anymore.
        train_eval's plateau warning is the backstop if a future change
        reintroduces this on a harder dataset or a still-insufficient budget.
        """
        self.fast = True
        self.max_train, self.max_val, self.max_test = 3000, 800, 1500
        self.epochs_small, self.epochs_native = 25, 3
        self.seeds = [0]
        self.topfer_resolutions = [7, 14, 28]
        self.topfer_k_values = [0, 2, 5]
        # Not shrunk below the class default (35): apply_fast's own n_top
        # subsample (<=1500 images) is exactly the case that default is
        # verified against — see the class-level comment. A wrong scale-law
        # fit is a worse outcome than a slower preview.
        self.topfer_epochs = 35
        return self

    def to_json(self, path: Path | None = None) -> str:
        d = {k: v for k, v in asdict(self).items() if k != "paths"}
        d["paths_root"] = str(self.paths.root)
        s = json.dumps(d, indent=2, default=str)
        if path is not None:
            Path(path).write_text(s)
        return s


CFG = Config()
