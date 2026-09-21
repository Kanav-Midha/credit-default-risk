# Methodology

The decisions behind the numbers, and what would invalidate them.

## Validation strategy

Five-fold stratified cross-validation on `application_train`, producing
out-of-fold (OOF) predictions for all 307,511 rows. Every reported metric is
computed on those OOF predictions.

**Why stratified.** At an 8.1% positive rate, an unstratified 5-way split can
hand one fold a noticeably different default rate. Fold scores then measure
fold composition as much as model quality, and their mean becomes unstable
across seeds.

**Why not a single holdout.** A 20% holdout of this dataset contains ~5,000
defaults. The standard error on AUC at that size is large enough to hide the
differences between model versions that this project needs to detect. OOF
predictions use every row for evaluation while never evaluating a model on data
it trained on.

**Known limitation.** The dataset carries no reliable application timestamp, so
this is a random split, not a temporal one. A production scorecard must be
validated out-of-time, because a random split assumes the future looks like the
past and therefore overstates real-world performance. This is the single
largest gap between this project and a deployable model, and it is a property
of the dataset rather than a shortcut taken here.

## Leakage controls

- `SK_ID_CURR` is excluded from the feature matrix. Any correlation between an
  administrative key and the target is an artefact of how the data was
  assembled, and a tree will happily memorise it.
- No target encoding, and no feature computed from the target anywhere in the
  pipeline. Target statistics computed over the full training set leak into
  every row, inflating validation scores by several points while adding nothing
  on unseen data.
- All child-table aggregates are functions of a single applicant's own history,
  so nothing crosses between applicants.
- Feature engineering is fitted independently of the fold split because it
  contains no fitted state — only row-wise arithmetic and per-applicant
  aggregation. Were an imputer or scaler added, it would have to move inside
  the fold loop.

## Metric choices

| Metric | Why it is here |
|---|---|
| ROC-AUC | Comparable to the public leaderboard, so the result can be sanity-checked against a known benchmark |
| Gini | `2 × AUC − 1`; the form used in credit risk reporting |
| PR-AUC | Degrades visibly when the rare class is hard, which ROC-AUC can mask |
| KS statistic | Standard regulatory measure of scorecard strength |
| Brier score, ECE | Calibration — required because the decision layer consumes probabilities as probabilities |
| Lift at 10% | How a portfolio manager reads model value |

Accuracy is deliberately absent. Predicting "nobody defaults" scores 91.9%.

## Why the positive class is not reweighted

`is_unbalance` and `scale_pos_weight` are left unset. Two reasons:

1. Reweighting does not improve ranking. AUC depends only on the ordering of
   scores, and class weights largely rescale them.
2. Reweighting destroys calibration. A model trained with the positive class
   upweighted outputs probabilities inflated by roughly the weight ratio. The
   expected-loss calculation would then be wrong by that factor.

Imbalance is handled at the decision layer instead, by choosing a threshold
from the cost of each error.

## Business assumptions

The cost model uses two constants, defined in `config.py`:

| Constant | Value | Meaning |
|---|---|---|
| `LGD` | 0.65 | Fraction of exposure lost when a loan defaults |
| `PROFIT_MARGIN` | 0.12 | Profit on a performing loan, as a fraction of credit |

**These are assumptions, not measurements.** The dataset contains no recovery
or pricing data, so neither can be derived from it. The values are plausible
for unsecured consumer lending but would differ by product, geography and
vintage.

Because the headline savings figure depends on them, `make evaluate` re-optimises
the threshold across a 5×5 grid of both and writes the result to
`reports/sensitivity.csv`. The claim being made is therefore not "the model saves
X" but "the model saves something positive across every plausible assumption in
this range, and X under the central one". If any cell in that grid showed a
loss, the headline number would be an artefact of the assumption rather than a
property of the model.

## What would invalidate the results

- **A temporal split showing materially worse performance.** Likely, and the
  honest expectation. Random-split performance is an upper bound.
- **Calibration drifting under a changed population.** The threshold is derived
  from probabilities; if those become miscalibrated, the cost-optimal cutoff
  moves and the policy silently degrades.
- **Different LGD or margin in a real portfolio.** The sensitivity grid covers
  0.45–0.85 and 0.08–0.20. Outside that range the conclusion has not been
  tested.
