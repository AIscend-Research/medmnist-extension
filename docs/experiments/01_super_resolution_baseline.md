# P0 — Super-resolution-then-downsample baseline

## Question

Could a *generic* image prior — not this project's symbolization — already
recover what area-mean pooling destroys when 224² is reduced to 28²? If a
plain upsampling trick closes the `naive` → `native224` gap, the paper's
argument that the fix has to be cartographic (measure structure at native
resolution, symbolize it) is undermined.

## Implementation

- `cartomnist/superres.py`: `classical_sr_upsample(rgb28, target=224)` —
  Lanczos upsample of the **already naive-resized 28×28 RGB image**
  (never the true native image) followed by an unsharp-mask sharpening pass.
  This is the standard "classical SR baseline" move in the super-resolution
  literature (Dong et al., 2014, SRCNN, use bicubic upsampling as their own
  baseline).
- Batched via `classical_sr_upsample_batch`, same `parallel.map_chunks`
  pattern as every other batch operator in the codebase.
- Wired into `pipeline.run_all()` as a new arm, `sr_classical`, trained with
  `regime="native224"` — i.e. the same ResNet-18 and native-resolution
  hyperparameters (`epochs_native`, `batch_size_native`, `lr_native`) that
  the real `native224` arm uses. This is deliberate: the question is whether
  a *generic prior applied to the 28x28 product* can reach native224-like
  performance, not whether a small CNN can.
- `contracts.SR_UPSAMPLE`: states the invariant this baseline is built to
  satisfy — output is a deterministic function of the 28×28 naive-resized
  image alone, so a measured recovery of the gap can only be credited to the
  upsampling prior, never to smuggled native-resolution access.
- **Real-ESRGAN was deliberately excluded.** `basicsr`/`realesrgan` are
  known to import a private `torchvision.transforms.functional_tensor`
  module that newer torchvision removed, which conflicts with this
  project's exact-pin-verified-against-the-full-suite dependency policy, and
  Real-ESRGAN's pretrained weights need a download the project's Kaggle
  notebook is designed to run without. This is recorded in the module
  docstring so a future contributor doesn't have to rediscover it.
- Tests: `tests/test_superres.py` — shape/range, determinism, and that
  unsharp-masking measurably changes the result versus plain Lanczos
  interpolation (otherwise this arm would just be `baselines.naive_resize_interp`
  with extra steps).

## Results (real DermaMNIST-224, full run: 3 seeds, no subsampling)

| regime | macro AUC | balanced acc | rare recall | ECE |
|---|---|---|---|---|
| naive | 0.9003 | 0.5787 | 0.6017 | 0.0366 |
| generalized | 0.8582 | 0.4829 | 0.6067 | 0.0652 |
| native224 | 0.9272 | 0.6805 | 0.7173 | 0.0306 |
| **sr_classical** | **0.9177** | **0.6242** | **0.6306** | **0.0389** |

## Interpretation

`sr_classical` (0.918 AUC) lands close to `native224` (0.927) and clearly
above both `naive` (0.900) and `generalized` (0.858) on this run. Read at
face value, this says a purely classical upsampling prior recovers most of
the naive→native224 gap without any symbolization — a real challenge to the
paper's framing.

**One important confound, stated plainly rather than glossed over:**
`sr_classical` trains a ResNet-18 (matching `native224`'s architecture and
hyperparameters), while `naive` and `generalized` train the deliberately
small `SmallCNN` (see `models.py`'s own docstring: "the claim is about
preprocessing, not capacity"). Part of `sr_classical`'s edge over
`naive`/`generalized` is very plausibly architecture/capacity, not the SR
preprocessing itself — the comparison to `naive`/`generalized` is **not**
apples-to-apples the way `pansharpen_brovey`/`ihs` (same `SmallCNN`, same
epoch budget) are. The clean comparison this baseline actually answers is
`sr_classical` vs. `native224`: both ResNet-18, both native-resolution
input, differing only in whether that input is the true 224² acquisition or
a classically-upsampled 28² product. On that comparison, `sr_classical`
(0.918) trails `native224` (0.927) by a modest but real margin — a generic
prior gets close, but does not fully close the gap to genuine native-resolution
acquisition. Disentangling the architecture confound from the naive/generalized
comparison (e.g., a ResNet-18 trained directly on the naive 28² image, no SR)
would be a natural follow-up if this baseline needs to inform the paper's
claims more precisely.
