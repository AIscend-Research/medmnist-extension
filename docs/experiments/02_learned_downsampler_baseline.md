# P1 — Learned/attention-based downsampler baseline

## Question

Is fixed area-mean pooling itself (`operators.aggregate`) leaving accuracy
on the table, independent of symbolization? If a pooling layer trained
jointly with the classifier beats plain averaging by a wide margin, part of
the naive/generalized gap could be about *how* pixels are pooled, not
*what* gets symbolized before pooling.

## Implementation

- `cartomnist/learned_pool.py`: `LearnedDownsample(channels, factor, mode)`,
  a depthwise `Conv2d(kernel=stride=factor, groups=channels, bias=False)`.
  - `mode="linear"`: weight initialized to exactly `1/factor**2` — bit-for-bit
    equal to `operators.aggregate` before any gradient step (checked by
    `tests/test_learned_pool.py::test_linear_mode_matches_area_mean_at_init`).
    Any measured gap against the fixed `naive` baseline is then attributable
    to what training changes, not to an initialization advantage.
  - `mode="attention"`: a small scoring conv produces one score per pixel,
    softmax-normalised within each `factor × factor` block via
    `F.unfold`/weighted sum — a strict generalization of uniform area-mean
    weights. Area-mean pooling is the special case where the score network
    is identically zero.
- `models.LearnedDownsampleCNN`: `LearnedDownsample → SmallCNN`, composed and
  trained end-to-end. Registered in `models.build()` under regimes
  `"learned_pool_linear"` / `"learned_pool_attention"`.
- `train.train_eval`'s native-vs-small hyperparameter switch was broadened
  to recognize these two regimes as native-resolution consumers (they train
  directly on `src.images(split)`, not a precomputed 28×28 tensor — the
  whole point is end-to-end learning from the real 224² input).
- `contracts.LEARNED_POOL`: states what's given up relative to
  `contracts.AGGREGATION` — the areal-integral invariant area-mean
  guarantees by construction, and determinism (the exact pooling rule now
  depends on the training run, not just the pixels).
- Tests: `tests/test_learned_pool.py` — invalid-mode rejection, the
  linear-mode-matches-area-mean-at-init invariant, attention-mode shape and
  the "uniform input → uniform output" sanity check, and gradient-flow
  checks for both modes.

## Results (real DermaMNIST-224, full run: 3 seeds, no subsampling)

| regime | macro AUC | balanced acc | rare recall | ECE |
|---|---|---|---|---|
| naive | 0.9003 | 0.5787 | 0.6017 | 0.0366 |
| **learned_pool_linear** | **0.8966** | **0.5659** | **0.5670** | **0.0510** |
| **learned_pool_attention** | **0.8969** | **0.5783** | **0.5936** | **0.0496** |

This is the cleanest comparison of the four new baselines: `naive`,
`learned_pool_linear`, and `learned_pool_attention` all train the same
`SmallCNN` classifier on the same epoch budget, differing only in the
pooling step feeding it (fixed area-mean vs. two trained variants).

## Interpretation

Both learned variants land essentially at parity with `naive` — linear mode
0.4 points of AUC below, attention mode 0.3 points below, well within the
kind of run-to-run variation this codebase's own comments describe for
small-CNN regimes. Neither the linear pooling layer (starting from
area-mean) nor the attention pooling layer found a meaningfully better
summary of each 8×8 block than the fixed uniform average, at this epoch
budget and this task.

Read plainly: **fixed area-mean pooling does not appear to be the
bottleneck.** This is evidence *against* the "it's just a bad pooling rule"
alternative explanation, and consistent with the paper's framing that the
naive/generalized gap is about what survives resampling at all
(symbolization), not about the linear-combination rule used to resample it.
A caveat worth carrying forward: this was one training run per mode (no
seed averaging for the learned-pool arms' own internal stochasticity beyond
the 3 outer seeds already reflected above), and the attention variant's
score network was not tuned beyond a single reasonable default (3×3 conv,
no extra regularization) — a wider search over pooling-layer capacity or
training schedule could still move this number.
