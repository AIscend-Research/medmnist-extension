# P1 — Pan-sharpening baseline (remote-sensing literature)

## Question

This project's own pipeline (`generalize.py`) measures fine structure at
native resolution and injects it into the coarser target sheet —
structurally the same move as remote-sensing pan-sharpening, where a
high-resolution panchromatic band is fused into coarser multispectral
bands. Does an actual, citable, off-the-shelf fusion algorithm from that
literature close any of the naive/generalized gap on its own, without
cartographic symbolization? This turns an analogy in the paper's framing
into a real, comparable baseline.

## Implementation

- `cartomnist/pansharpen.py`:
  - `to_pan_band(rgb)`: ITU-R BT.601 luminance — the "panchromatic" band
    (Gonzalez & Woods, *Digital Image Processing*).
  - `brovey_fuse(ms, pan)`: classic Brovey transform,
    `fused_i = ms_i · pan / mean_c(ms)` (Gillespie, Kahle & Walker, 1987).
  - `ihs_fuse(ms, pan)`: RGB→HSV, replace intensity (V) with the
    panchromatic band histogram-matched to V's mean/std (standard practice
    to limit spectral distortion), HSV→RGB (Chavez, Sides & Anderson, 1991).
  - `pansharpen_resize(rgb_native, target, method)`: area-mean pool the
    native image to `target` (= the naive-resize baseline), bicubic-upsample
    that back to native resolution, fuse it with the native luminance via
    Brovey or IHS, then area-mean pool the **fused** image down to `target`
    — an apples-to-apples final shape versus `generalize.naive_resize`, with
    fusion as the only added step.
- Wired into `pipeline.run_all()` as two arms, `pansharpen_brovey` and
  `pansharpen_ihs`, both trained with the same `SmallCNN` and epoch budget
  as `naive` (regime `"pansharpen"`, which only needs to not equal
  `"native224"` to route to the small-CNN hyperparameters) — a genuinely
  fair, same-architecture comparison, unlike the SR baseline (see
  `01_super_resolution_baseline.md`'s confound note).
- `contracts.PAN_SHARPEN`: names this project's own symbolize-then-fuse move
  as the same structural pattern, and states the known failure mode this
  baseline inherits from the literature — Brovey/IHS fusion distorts color
  in saturated regions because both assume a PAN–MS correlation a lesion
  photograph need not actually have.
- Tests: `tests/test_pansharpen.py` — shape/range, invalid-method rejection,
  batch-equals-single, numerically distinct from plain area-mean pooling and
  from each other, and a luminance sanity check on a flat grey input.

## Results (real DermaMNIST-224, full run: 3 seeds, no subsampling)

| regime | macro AUC | balanced acc | rare recall | ECE |
|---|---|---|---|---|
| naive | 0.9003 | 0.5787 | 0.6017 | 0.0366 |
| **pansharpen_brovey** | **0.9016** | **0.5896** | **0.6118** | **0.0379** |
| **pansharpen_ihs** | **0.8985** | **0.5757** | **0.6017** | **0.0342** |

Same `SmallCNN`, same epoch budget as `naive` for all three rows — this is
the fairest of the four new comparisons.

## Interpretation

Both fusion methods land within noise of plain `naive` resizing: Brovey
edges it out by 0.13 points of AUC and a bit more on balanced accuracy and
rare recall; IHS is statistically indistinguishable, slightly below on two
of four metrics. **Neither classical remote-sensing fusion algorithm
meaningfully outperforms plain area-mean pooling on this task.**

This is informative in the direction the `PAN_SHARPEN` fidelity contract
predicted: Brovey and IHS both assume a strong, well-behaved correlation
between the injected high-resolution band and the existing low-resolution
color bands — true by construction for a satellite's PAN and MS sensors
imaging the same scene at the same time, but not something a dermoscopy
image's luminance channel is guaranteed to have with respect to its own
color content. The literal, off-the-shelf remote-sensing move does not
transfer for free; whatever gain the project's own symbolize-then-fuse
pipeline achieves is doing something more specific than generic band
fusion. That is exactly the citable comparison this baseline exists to
provide — the analogy in the paper's framing holds structurally, but the
generic implementation of it does not solve the problem here.
