#!/usr/bin/env python
"""Generate notebooks/kaggle_cartomnist.ipynb.

Kept as a generator rather than a hand-edited .ipynb so the notebook stays
diffable and cannot drift from the package API.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/AIscend-Research/medmnist-extension.git"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(True)}


CELLS = [
    md(r"""
# Stop resizing medical images. Generalize them like a map.

**Cartographic generalization for MedMNIST — a full, end-to-end run.**

A mapmaker going from 1:10,000 to 1:1,000,000 faces exactly the problem a
224×224 → 28×28 medical benchmark faces: the features that matter fall below the
resolvable width of the medium. Cartographers never solved this by resizing the
map. Over two centuries they developed **generalization**: a vocabulary of named
operators — selection, simplification, aggregation, displacement, exaggeration,
symbolization — each with an explicit contract about what is preserved and what
is sacrificed. A road half a millimetre wide at scale is drawn two millimetres
wide anyway, because the map's purpose is navigation, not photometry. And a
nautical chart prints a **source diagram**: an inset showing which waters were
surveyed, how densely, and when, so a navigator can see at a glance where the
chart cannot be trusted.

This notebook implements that discipline for DermaMNIST and runs the experiment
that tests it.

### What gets compared

| regime | what it is | channels |
|---|---|---|
| `naive` | MedMNIST as shipped — RGB area-resized 224→28 | 3 |
| `generalized` | the same 224 source, generalized like a map | 18 |
| `native224` | the 224 source itself — the pixel upper bound | 3 |

The `naive` arm is literally channels 0–2 of the generalized tensor, so the
**only** difference between the two low-resolution arms is the presence of the
symbol and source-diagram channels. Not an interpolation kernel, not a colour
round-trip, not a different crop.

### What comes out

- a **benchmark table** with a reliability diagram and ECE for every regime
- the **Nyquist audit**: which diagnostic structures resizing cannot keep, at all
- the **Töpfer sweep**: how many symbolized structures a grid of resolution *r*
  can pay for, versus the radical law's √r prediction
- the **Mercator result**: whether the "neutral" transform distorts uniformly
  across skin tone — measured without a classifier in the loop
- the **source diagram** scored as an abstention model
- **ablations**, including the shortcut test that could sink the whole thing
- twelve chart-sheet plates, built from real dermoscopic imagery
- one zip containing **only** the analysis, results and figures
"""),

    md(r"""
---
## 1 · Settings

`FAST = True` fits the whole run in roughly 30–50 GPU-minutes (a range that's
provisional until confirmed on an actual GPU run — a single-seed local CPU
run took ~48 minutes end to end; the epoch budgets behind it were raised for
reliability, see `Config.apply_fast`). Set it to `False` for the full-data
run (a few hours). Everything else is left at defaults that are recorded
verbatim into `results/config.json`.
"""),

    code(r"""
FAST = True          # False = full data, 3 seeds, 4 resolutions in the sweep
USE_GPU = True       # falls back to CPU automatically if none is attached
RUN_ISIC = True      # validates symbols against ISIC 2018 Task 2 masks if present

# Add these as Kaggle inputs to run with internet DISABLED:
#   MedMNIST v2 (needs dermamnist_224.npz)  -> set MEDMNIST_INPUT below
#   ISIC 2018 Task 1-2, single dataset with both the training images and the
#   Task 2 masks                            -> set ISIC_INPUT below
MEDMNIST_INPUT = None    # e.g. "/kaggle/input/medmnist2d"
ISIC_INPUT     = None    # e.g. "/kaggle/input/isic2018-task2"

# Or, if the images and the Task 2 masks come from two separate Kaggle
# datasets, set these instead and the bootstrap cell below will symlink them
# together into a directory isic.py can read as one root.
ISIC_IMAGES_INPUT = None   # dataset containing ISIC2018_Task1-2_Training_Input
ISIC_MASKS_INPUT  = None   # dataset containing ISIC2018_Task2_Training_GroundTruth_v3
"""),

    md(r"""
---
## 2 · Environment

Installs the package and its dependencies. On Kaggle with internet **off**, add
this repository as a Kaggle *dataset* or *utility script* and the import below
will pick it up from `/kaggle/input` without any download.
"""),

    code(r"""
import os, subprocess, sys
from pathlib import Path

REPO = "%s"

def _sh(*a):
    print("$", " ".join(a)); subprocess.run(a, check=False)

# --- make `cartomnist` importable -----------------------------------------
def _bootstrap():
    try:
        import cartomnist  # noqa: F401
        return "already importable"
    except ImportError:
        pass
    # 1. a sibling checkout (running this notebook from inside the repo)
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "cartomnist" / "__init__.py").exists():
            sys.path.insert(0, str(p)); return f"local checkout at {p}"
    # 2. the repo added as a Kaggle input
    for p in Path("/kaggle/input").glob("*/**/cartomnist/__init__.py") \
             if Path("/kaggle/input").exists() else []:
        sys.path.insert(0, str(p.parent.parent)); return f"kaggle input {p.parent.parent}"
    # 3. clone it
    dst = Path("/kaggle/working/medmnist-extension")
    if not dst.exists():
        _sh("git", "clone", "--depth", "1", REPO, str(dst))
    if dst.exists():
        sys.path.insert(0, str(dst)); return f"cloned to {dst}"
    raise RuntimeError("could not locate the cartomnist package")

_sh(sys.executable, "-m", "pip", "install", "-q",
    "scikit-image==0.26.0", "medmnist==3.0.2")
print("bootstrap:", _bootstrap())

import cartomnist
print("cartomnist", cartomnist.__version__)

if MEDMNIST_INPUT: os.environ["CARTO_DATA_DIR"] = MEDMNIST_INPUT

def _find_isic_subdir(base: str, candidates: list[str]) -> Path:
    # Fail loudly rather than silently skipping: a wrong/typo'd Kaggle input
    # slug or dataset layout must stop the run, not quietly disable ISIC
    # validation 2000 seconds in.
    base_path = Path(base)
    if not base_path.exists():
        raise RuntimeError(
            f"ISIC input path {base!r} does not exist. Is that dataset actually "
            f"attached to this kernel (Add Input on the right)? Check the slug."
        )
    for name in candidates:
        if (base_path / name).exists():
            return base_path / name
        hit = next(base_path.glob(f"**/{name}"), None)
        if hit is not None:
            return hit
    # case-insensitive fallback, in case Kaggle's unzip renamed casing
    wanted = {c.lower() for c in candidates}
    hit = next((p for p in base_path.glob("**/*")
                if p.is_dir() and p.name.lower() in wanted), None)
    if hit is not None:
        return hit
    top = sorted(p.name for p in base_path.iterdir())
    raise RuntimeError(
        f"Could not find any of {candidates} anywhere under {base}. "
        f"Top-level contents of {base}: {top}. Fix ISIC_IMAGES_INPUT / "
        f"ISIC_MASKS_INPUT above, or the expected folder name."
    )

if ISIC_IMAGES_INPUT and ISIC_MASKS_INPUT:
    merged = Path("/kaggle/working/isic_combined")
    merged.mkdir(parents=True, exist_ok=True)
    for src_base, candidates in [
        (ISIC_IMAGES_INPUT, ["ISIC2018_Task1-2_Training_Input"]),
        (ISIC_MASKS_INPUT, ["ISIC2018_Task2_Training_GroundTruth_v3",
                             "ISIC2018_Task2_Training_GroundTruth"]),
    ]:
        src = _find_isic_subdir(src_base, candidates)
        dst = merged / src.name
        if not dst.exists():
            dst.symlink_to(src)
    os.environ["CARTO_ISIC_DIR"] = str(merged)
elif ISIC_INPUT:
    if not Path(ISIC_INPUT).exists():
        raise RuntimeError(
            f"ISIC_INPUT={ISIC_INPUT!r} does not exist. Is that dataset "
            f"attached to this kernel? Check the slug."
        )
    os.environ["CARTO_ISIC_DIR"] = ISIC_INPUT

import torch
has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
device_name = ("cuda: " + torch.cuda.get_device_name(0) if torch.cuda.is_available()
               else "mps" if has_mps else "cpu")
print("torch", torch.__version__, "| device that Config(device=\"auto\") will pick:",
      device_name)
""" % REPO),

    md(r"""
---
## 3 · The legend, before any training

Before a single model is fitted, two things can already be printed, and both are
things a benchmark ought to ship: the **fidelity contracts** of every operator
that touches the pixels, and the **Nyquist audit** of the reduction itself.

The audit is the whole argument in one table. At 224→28 each benchmark pixel
covers 8 native pixels, so any structure whose period is under 16 native pixels
is below the target grid's Nyquist limit. No resampling kernel, at any quality
setting, can represent it. Symbolization does not beat the sampling theorem — it
moves the measurement *upstream* of the reduction, which is a different thing.
"""),

    code(r"""
from cartomnist import nyquist_table, legend_dict
from cartomnist.filterbanks import STRUCTURES
import pandas as pd

pd.set_option("display.width", 160)
audit = pd.DataFrame(nyquist_table(28, 224)).T
audit.index.name = "structure"
display(audit)

print("\nStructure vocabulary (fixed filter banks, no training):")
for s in STRUCTURES:
    print(f"  {s.name:<18} period ≈ {s.period_px_at_224:>4.1f} px "
          f"({s.period_px_at_224/224*20:.2f} mm)   {s.note}")

print("\nOperators in the legend:")
for c in legend_dict()["generalized"]:
    print(f"  · {c['operator']:<16} {c['cartographic_analogue'][:88]}…")
"""),

    md(r"""
---
## 4 · Look at one lesion before running anything

Worth spending thirty seconds on. The left panel is the native survey with the
28×28 cell boundaries drawn over it — that grid is what the benchmark keeps. The
right panels are the symbolized structures measured *before* the reduction.
"""),

    code(r"""
import numpy as np, matplotlib.pyplot as plt
from cartomnist.data import load_derma224, CLASS_SHORT
from cartomnist.generalize import generalize, channel_spec
from cartomnist import figures as F, style as S

raw = load_derma224()
spec = channel_spec()
i = int(np.flatnonzero(raw["test_labels"] == 4)[0])          # a melanoma
img = raw["test_images"][i]
tensor, inter = generalize(img, target=28, return_intermediates=True)

S.use_style()
F.fig_symbol_atlas(Path("."), img, tensor, inter, spec,
                   CLASS_SHORT[int(raw["test_labels"][i])], 28)
display(__import__("IPython").display.Image("fig03_symbol_atlas.png", width=1100))
"""),

    md(r"""
---
## 5 · The full run

This does everything: generalization of all three splits, ITA estimation, three
training regimes, temperature scaling, ablations including the shortcut test,
the classifier-free Mercator analysis, the Töpfer resolution sweep, the ISIC
validation if the data is present, all twelve plates, and the bundle.

Progress is printed with elapsed time per stage.
"""),

    code(r"""
from cartomnist import Config, run_all

cfg = Config()
cfg.paths.root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
cfg.paths.mkdirs()
if FAST:
    cfg.apply_fast()
cfg.device = "auto" if USE_GPU else "cpu"
cfg.run_isic_validation = RUN_ISIC

print("output   ->", cfg.paths.out)
print("scratch  ->", cfg.paths.cache, "(never included in the bundle)")

out = run_all(cfg)

# ISIC inputs were explicitly configured above -> validation must actually
# have run. A quiet "available: False" here means hours of GPU time produced
# a report with Plate XI missing and nobody would notice until it's too late.
_isic_wired = bool(ISIC_INPUT or (ISIC_IMAGES_INPUT and ISIC_MASKS_INPUT))
if RUN_ISIC and _isic_wired:
    isic_res = out["isic_validation"]
    if not isic_res.get("available"):
        raise RuntimeError(
            "ISIC inputs were configured but validation did not run: "
            f"{isic_res.get('reason')!r}. Fix the inputs and rerun rather than "
            "shipping a report missing Plate XI."
        )
"""),

    md(r"""
---
## 6 · Headline results
"""),

    code(r"""
import pandas as pd
rows = []
for k, e in out["evaluations"].items():
    rows.append({
        "regime": k, "params": int(e["extra"].get("n_params", 0)),
        "macro AUC": e["macro_auc"], "95% CI":
            f"[{e['macro_auc_lo']:.3f}, {e['macro_auc_hi']:.3f}]",
        "bal. acc": e["balanced_accuracy"], "rare recall": e["rare_recall"],
        "ECE": e["ece"], "ECE (uncal.)": e["ece_uncalibrated"],
        "Brier": e["brier"], "ITA gap": e["ita_gap"],
    })
display(pd.DataFrame(rows).set_index("regime").round(4))

import math

gr = out.get("gap_recovery", {})
f = gr.get("fraction_of_224_gap_recovered_at_28", float("nan"))
print(f"\nGeneralized 28² recovers {f:.1%} of the rare-class recall gap that 224² recovers.")

law = out["topfer"]["law"]
if math.isnan(law["alpha"]):
    print(f"Töpfer scale law: undefined this run — {law['diagnostic']}")
else:
    print(f"Töpfer scale law: measured K* ∝ r^{law['alpha']:.3f} "
          f"(radical law predicts r^0.50)")
for r, s in out["mercator"]["summary"].items():
    print(f"Mercator · {r:<18} retention spread across ITA strata = {s['mean_spread']:.4f}")
full = out["ablations"]["generalized (full)"]["macro_auc"]
sc   = out["ablations"]["symbols shuffled across images (shortcut test)"]["macro_auc"]
print(f"\nShortcut test: shuffling symbols across images costs {full - sc:+.4f} AUC "
      f"({'evidence, not a shortcut' if full - sc > 0.01 else 'WARNING — investigate'})")
"""),

    md(r"""
---
## 7 · The plates

All twelve, inline.
"""),

    code(r"""
from IPython.display import Image, display, Markdown
titles = {
 "fig00_headline": "Frontispiece — the standard, as one sheet",
 "fig01_legend": "Plate I — the legend",
 "fig02_three_regimes": "Plate II — three ways to publish the same lesion at 28×28",
 "fig03_symbol_atlas": "Plate III — survey sheet: instrument response to symbol",
 "fig04_source_diagram": "Plate IV — the source diagram and the Nyquist audit",
 "fig05_mercator": "Plate V — the Mercator result",
 "fig06_reliability": "Plate VI — reliability diagrams",
 "fig07_topfer": "Plate VII — the radical law as a benchmark scale law",
 "fig08_certificate": "Plate VIII — the certificate as an abstention model",
 "fig09_rare_class": "Plate IX — rare-class recovery",
 "fig10_ablation": "Plate X — ablations, including the shortcut test",
 "fig11_isic_validation": "Plate XI — do the symbols track expert annotation?",
}
for p in sorted(cfg.paths.figures.glob("fig*.png")):
    display(Markdown(f"### {titles.get(p.stem, p.stem)}"))
    display(Image(str(p), width=1150))
"""),

    md(r"""
---
## 8 · Download only the analysis

`run_all` already wrote `cartomnist_report.zip`. It contains the standalone HTML
report, every plate, and every results table as CSV plus JSON — and **nothing
else**. Cached tensors live in a separate scratch directory (on Kaggle, in
`/kaggle/temp`, which is never persisted), so they are neither zipped nor
uploaded when the notebook is committed.

The cell below prints the manifest and gives you a direct download link.
"""),

    code(r"""
import zipfile, shutil
from IPython.display import FileLink, display

zpath = cfg.paths.root / "cartomnist_report.zip"

# Belt and braces: make sure nothing heavy is sitting in the output directory.
for junk in cfg.paths.root.glob("*.npy"):
    junk.unlink()
if (cfg.paths.root / "carto_cache").exists():
    shutil.rmtree(cfg.paths.root / "carto_cache", ignore_errors=True)

with zipfile.ZipFile(zpath) as z:
    names = z.namelist()
    total = sum(i.file_size for i in z.infolist())
print(f"{zpath.name} — {zpath.stat().st_size/1e6:.1f} MB compressed, "
      f"{total/1e6:.1f} MB expanded, {len(names)} files\n")
for n in sorted(names):
    print("  ", n)

print("\nDownload:")
display(FileLink(str(zpath)))
"""),

    md(r"""
---
## 9 · What this proposes

A medical imaging benchmark should ship:

1. **Named generalization operators with fidelity contracts** — what was
   preserved, what was sacrificed, and a checkable invariant for each.
2. **A scale bar in physical units** — millimetres of tissue per pixel, so a
   reader can tell whether a structure is representable *before* training
   anything on it.
3. **A reliability diagram** — with its ECE, reported both before and after
   calibration.

Every existing MedMNIST-style dataset is auditable against that standard today.
The source diagram *is* the abstention model, shipped with the data instead of
refitted per model. And the equity result is not a bias audit bolted on
afterwards — it falls out of the framework, because a generalization policy is
precisely a decision about whose features survive.

### Caveats owned up front

- **The symbols are only as good as the filter banks.** Plate XI validates three
  of the five against ISIC 2018 Task 2 expert masks. Vessels and blue-white veil
  have no ISIC counterpart and are reported as measured-but-unvalidated.
- **The model could learn the symbols as a new shortcut.** Plate X tests this
  directly: dropping the symbols at test time, and — the harder test — shuffling
  them across images to destroy their alignment with the picture.
- **Equal fidelity is conditional.** Symbol normalisation is anchored to a
  measured noise floor, so where an image genuinely cannot support a measurement
  the symbol collapses toward zero rather than amplifying noise. That is the
  honest behaviour, and the source diagram is what makes the remaining
  distortion legible instead of hidden.
- **ITA from dermoscopy is a proxy.** Contact-illuminated, often polarised
  images give a stratification variable, not a colorimetric ground truth.
"""),
]


def main() -> None:
    # Deterministic, positional cell ids (not random) so regenerating the
    # notebook produces byte-identical output — the whole point of keeping
    # this as a generator. nbformat's validator already warns that a missing
    # id "will become a hard error in future nbformat versions".
    for i, cell in enumerate(CELLS):
        cell["id"] = f"cell-{i:02d}"

    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_cartomnist.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
