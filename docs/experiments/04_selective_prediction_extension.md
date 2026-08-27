# P1 — Selective prediction / active-learning extension to the abstention comparison

## Question

The project already compares its label-free source-diagram certificate
against softmax confidence for test-time abstention (`pipeline.py`'s
`certificate_vs_confidence`, Plate VIII). Is the certificate actually better
than *general-purpose* uncertainty estimation from the selective-prediction
/ active-learning literature — MC-dropout and deep ensembles — or does it
just beat the weakest baseline (plain softmax)?

## Implementation

- `metrics.mc_dropout_uncertainty(mc_probs)`: predictive variance across `T`
  stochastic dropout passes, summed over classes (Gal & Ghahramani, 2016,
  "Dropout as a Bayesian Approximation").
- `metrics.ensemble_uncertainty(probs_per_seed)`: mean KL divergence of each
  ensemble member from the ensemble mean — an epistemic-disagreement score
  (Lakshminarayanan, Pritzel & Blundell, 2017, "Simple and Scalable
  Predictive Uncertainty Estimation using Deep Ensembles").
- `train._enable_dropout` / `train.mc_dropout_infer`: puts only `nn.Dropout`
  layers back in train mode (GroupNorm and everything else stays in eval
  mode) and runs `T` stochastic forward passes, returning `(T, N, C)`
  softmax probabilities.
- `train.train_eval` gained an opt-in `mc_dropout_samples: int = 0`
  parameter and `TrainResult.mc_dropout_probs`. Default behavior for every
  existing call site (ablations, the shortcut test, every other regime) is
  unchanged; only the `generalized` arm's first seed pays the extra
  `T`-pass cost (`cfg.mc_dropout_samples`, default 20).
- `pipeline.py`'s abstention section now adds two more curves to the
  existing `certificate vs. confidence vs. random` comparison:
  `"deep ensemble disagreement"` (from all `cfg.seeds`' softmax outputs) and
  `"MC-dropout uncertainty"` (from the stashed `mc_dropout_probs`), both
  scored the same way via `metrics.coverage_risk`/AURC.
- `figures.fig_certificate` (Plate VIII) picks up both new curves
  automatically (the plotting loop is generic over curve names); its
  on-curve labels were re-laid-out (stacked vertically by value, not pinned
  independently) since 5 curves overlapping was unreadable with the
  original 3-curve label placement.
- Tests: `tests/test_metrics_uncertainty.py` (shape, non-negativity,
  zero-when-identical / zero-when-agreeing edge cases for both functions)
  and additions to `tests/test_train.py` (`mc_dropout_infer` actually varies
  pass-to-pass; `mc_dropout_samples=0` leaves `TrainResult.mc_dropout_probs`
  as `None`).

## A caveat worth recording (from the fast-mode preview, no longer live)

An earlier fast-mode preview run (`cfg.seeds = [0]`) briefly produced a
"deep ensemble disagreement" number, but it was not meaningful:
`ensemble_uncertainty` computes the KL divergence of each seed's predictions
from the mean across seeds, and with exactly one seed, that mean *is* that
seed's own prediction, so the KL divergence is identically zero for every
test image everywhere. That preview number only looked plausible by
coincidence of where ties landed in the sort. The results below are from
the full run (`cfg.seeds = [0, 1, 2]`), where this is no longer an issue.

## Results (real DermaMNIST-224, full run: 3 seeds, no subsampling)

Balanced AURC (lower is better):

| trust signal | balanced AURC |
|---|---|
| **softmax confidence** | **0.5115** |
| random | 0.5173 |
| deep ensemble disagreement | 0.5427 |
| source certificate | 0.5450 |
| MC-dropout uncertainty | 0.5617 |

(`n_seeds = 3` confirmed in `results.json`'s `evaluations.generalized.extra`.)

## Interpretation

This is the one result across all four experiments that most directly
challenges an existing claim in the codebase, and it should be stated
plainly rather than softened: **on the full run, plain softmax confidence
is the best of the five trust signals, and the source certificate does not
beat it.** The certificate does still clearly beat MC-dropout uncertainty,
and edges out deep-ensemble disagreement, but it sits behind both softmax
confidence and even the random baseline. This is the opposite of the
ranking `report.py`'s own headline tile asserts ("AURC of the label-free
source certificate" implicitly framed as the good number to report against
confidence) — the shipped `report.html` from this run will show the same
reversal, since `figures.fig_certificate` and the report both read directly
from this same `certificate_vs_confidence` dict.

A few honest caveats before treating this as a settled result:

- This is a **single full run**, not repeated across independent
  experiment-level trials — the certificate/confidence/ensemble/MC-dropout
  ranking could itself have run-to-run variance the same way the
  `naive`/`generalized` benchmark numbers do (see the P0 doc's note on the
  known seed-sensitivity of the 18-channel `generalized` arm, which is the
  same model every one of these five signals is scored against here).
- MC-dropout being the worst signal in both the preview and full runs is a
  consistent pattern, not a fluke of one run — worth investigating whether
  `SmallCNN`'s `p_drop=0.15` is simply too low to produce a useful spread
  across the 20 stochastic passes on this task.
- The result does not mean the certificate is uninformative — it clearly
  separates from MC-dropout and roughly matches ensemble disagreement — only
  that the specific "certificate beats confidence" claim does not hold on
  this real-data, full-seed run the way it appeared to in whatever run the
  existing report copy was written against.

This is a finding for you or the repo's maintainers to look at directly —
it's outside the scope of what the four new baselines were meant to test
(the abstention *comparison* is the new experiment; the certificate and
softmax confidence signals themselves are pre-existing code), but it's a
real result from the extension, not an artifact of my implementation of
`mc_dropout_uncertainty`/`ensemble_uncertainty`, which are both covered by
passing unit tests independent of this specific dataset.
