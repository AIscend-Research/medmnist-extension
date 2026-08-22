# cartomnist

**Stop resizing medical images. Generalize them like a map — and ship a
reliability diagram with every benchmark.**

A mapmaker going from 1:10,000 to 1:1,000,000 faces exactly the problem a
224×224 → 28×28 medical benchmark faces: the features that matter fall below the
resolvable width of the medium. Cartographers never solved this by resizing the
map. They developed **generalization** — a vocabulary of named operators, each
with an explicit contract about what is preserved and what is sacrificed. A road
half a millimetre wide at scale is drawn two millimetres wide anyway, because
the map's purpose is navigation, not photometry. Töpfer's radical law (1966) even
gives a formula for how many features should survive a change of scale. And
nautical charts print a **source diagram**: an inset showing which waters were
surveyed, how densely, and when, so a navigator can see where the chart cannot
be trusted. That is a sufficiency certificate, and sailors have used one since
the nineteenth century.

This repository implements that discipline for DermaMNIST and runs the
experiment that tests whether it works.

---

## The claim

Take the diagnostic structures dermatology has already validated — pigment
network, dots/globules, streaks, vessels, blue-white veil — and extract them at
**native resolution** with cheap hand-designed filter banks (no training). Then
produce a 28×28 *generalized* image: RGB is downsampled as before, but the
sub-Nyquist structures are **symbolized** into extra low-resolution channels of
presence, density and orientation — the way a road narrower than a pixel is
drawn as a symbol rather than dropped. Add a **source diagram** channel stating,
per region, whether the local sampling and contrast floor met what each
structure needs.

If a 28×28 sheet with a legend recovers most of the rare-class and ITA gap that
224×224 recovers, then the "resolution problem" was never about pixels. The
benchmark didn't need to be bigger. It needed a legend.

---

## Quick start

### Kaggle (the intended path)

Open [`notebooks/kaggle_cartomnist.ipynb`](notebooks/kaggle_cartomnist.ipynb),
attach a GPU, run all. `FAST = True` fits the whole thing in roughly
~30–50 GPU-minutes (epoch budgets were raised from an earlier, faster-but-
unreliable setting — see `Config.apply_fast`'s docstring; a single-seed local
CPU run took ~48 minutes end to end, and a real GPU should meaningfully
speed up the ResNet-18 stage specifically — treat this range as provisional
until confirmed on an actual GPU run); `FAST = False` runs full data with
three seeds.

The notebook ends by handing you `cartomnist_report.zip` — a standalone HTML
report, twelve plates, and every results table as CSV and JSON. Cached tensors
are written to `/kaggle/temp`, which Kaggle does not persist, so committing the
notebook uploads the analysis and not four gigabytes of float32.

To run with **internet disabled**, add the MedMNIST v2 dataset (needs
`dermamnist_224.npz`) and optionally ISIC 2018 Task 1-2 as Kaggle inputs, then
set `MEDMNIST_INPUT` / `ISIC_INPUT` in the settings cell.

### Local

```bash
pip install -e ".[torch,data,dev]"
python -m pytest tests/ -q          # 24 invariant + equivariance tests
python scripts/smoke_test.py        # full pipeline on synthetic data, CPU, no download
python scripts/run_all.py --fast    # the real thing
```

---

## What it produces

| output | what it is |
|---|---|
| `report.html` | standalone, self-contained; embeds every plate |
| `results/benchmark_table.csv` | AUC with bootstrap CI, balanced acc, rare recall, ECE (pre- and post-calibration), Brier, ITA gap |
| `results/nyquist_audit.csv` | which structures resizing cannot keep, at all |
| `results/mercator_retention.csv` | classifier-free structure retention per ITA stratum |
| `results/topfer_sweep.csv` | AUC across (resolution × symbol budget) |
| `results/ablations.csv` | including the shortcut test |
| `results/legend.json` | machine-readable fidelity contracts |
| `figures/*.png` | twelve chart sheets, built from real dermoscopic imagery |

---

## The experiments

**Three regimes.** `naive` (MedMNIST as shipped), `generalized` (18 channels),
`native224` (ResNet-18 upper bound). The `naive` arm is literally channels 0–2
of the generalized tensor, so the only difference between the low-resolution
arms is the presence of the symbol and source-diagram channels — not an
interpolation kernel, not a colour round-trip, not a different crop.

**The Nyquist audit.** At 224→28 each benchmark pixel covers 8 native pixels, so
any structure with a period under 16 native pixels is below the target grid's
Nyquist limit. Four of the five are. Symbolization does not beat the sampling
theorem; it moves the measurement upstream of the reduction, which is a
different thing.

**Töpfer.** Sweep (resolution × number of symbolized families) and fit
K\* ∝ r^α. The radical law predicts α = 0.5. If it holds, a benchmark designer
can *compute* the symbol budget for a chosen resolution instead of guessing.

**Mercator.** Naive downsampling applies one low-pass filter to every image, but
structure amplitude in L\* is not independent of constitutive pigmentation — a
uniform rule, a non-uniform loss. Retention is measured **without a classifier**:
a local ridge probe fitted on train reconstructs the native structure map from
each regime, scored per test image, stratified by ITA. Three arms: naive,
generalized with equal-fidelity normalisation, and generalized with a global
absolute scale — the third exists to show that equal fidelity is a *design
decision inside the framework*, not an automatic property of it.

**The certificate as abstention.** The source diagram is label-free and
available at test time for an unlabelled image, so it can be published *with the
dataset* rather than refitted per model. It is scored against the model's own
softmax confidence on a coverage–risk curve.

---

## Caveats, owned up front

- **The symbols are only as good as the filter banks.** `cartomnist/isic.py`
  validates three of the five against ISIC 2018 Task 2 expert attribute masks
  (pixel AUC within image, and image-level AUC). Vessels and blue-white veil
  have no ISIC counterpart and are reported as *measured-but-unvalidated* rather
  than quietly assumed.
- **The model could learn the symbols as a new shortcut.** Two tests, both in
  the report: dropping the symbols at test time, and — the harder one —
  shuffling them across images to destroy their alignment with the picture. If
  the second doesn't hurt, the symbols were a per-image fingerprint and the
  result is an artefact.
- **Equal fidelity is conditional.** Symbol normalisation is anchored to a
  *measured noise floor* (`filterbanks.noise_ceiling`), so where an image
  genuinely cannot support a measurement the symbol collapses toward zero rather
  than amplifying noise into a confident false structure. Plain per-image
  percentile normalisation does the opposite, and would have manufactured the
  equity result by amplifying the lowest-contrast images the most. The claim is
  therefore *equal fidelity wherever the measurement is supportable*, with the
  source diagram making the remainder legible.
- **ITA from dermoscopy is a proxy.** Contact-illuminated, often polarised
  images give a stratification variable, not a colorimetric ground truth.
  HAM10000 is also heavily skewed light, so both canonical Del Bino bins and
  population quantile bins are reported — showing only the flattering one is
  exactly the move Harley warns about.
- **Orientation channels are not augmentation-invariant.** Flipping a chart
  flips the bearing of every road on it. `train.EquivariantAug` transforms
  cos2θ/sin2θ correctly and `tests/test_train.py` pins it, because that bug is
  completely silent — the loss still goes down, the bearings just quietly become
  noise.

---

## Repository layout

```
cartomnist/
  contracts.py    the legend — fidelity contracts, machine-readable
  filterbanks.py  the five instruments + the measured noise null
  operators.py    selection, simplification, aggregation, displacement,
                  exaggeration; each with a checkable invariant
  sufficiency.py  the source diagram / zones of confidence
  generalize.py   the pipeline; channel spec lives here and nowhere else
  ita.py          individual typology angle + stratification
  mercator.py     classifier-free retention analysis
  topfer.py       the radical law as a benchmark scale law
  isic.py         validation against ISIC 2018 Task 2 masks
  metrics.py      calibration as a first-class result
  models.py       SmallCNN (all low-res arms) + ResNet-18 (224 arm)
  train.py        training, and the equivariant augmenter
  style.py        chart-sheet styling; validated categorical palette
  figures.py      the twelve plates
  report.py       HTML + CSV + JSON + the bundle
  pipeline.py     run_all()
```

---

## References

Töpfer & Pillewizer (1966), *The principles of selection*; Brassel & Weibel
(1988); McMaster & Shea (1992), *Generalization in Digital Cartography*; Douglas
& Peucker (1973); Harley (1989), *Deconstructing the Map*; IHO S-4 / S-57 source
diagrams and CATZOC; Argenziano et al. (1998), the 7-point checklist; Del Bino &
Bernerd (2013) for ITA; Tschandl et al. (2018) HAM10000; Codella et al. (2018)
ISIC; Kinyanjui et al. (2020) for ITA-stratified skin-lesion analysis; Yang et
al. (2023), MedMNIST v2; Guo et al. (2017) for calibration and temperature
scaling.
