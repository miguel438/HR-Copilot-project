# Statistical fit model - training report

Generated 2026-09-06T19:19:47Z from `training/fit_labels.csv`.

## Dataset

- 294 (requisition, candidate) pairs
- 21 candidates, 14 requisitions
- bands: no-fit 244, partial 24, strong 26
- reviewer overrode the LLM prefill on 2% of rows
- one annotator, so there is no inter-annotator agreement to quote

**How the labels were reviewed, stated plainly.** The reviewer was pointed at the ~29
rows where the LLM's score was ambiguous, and several of those were flagged specifically
because the Evaluator had contradicted its own job-family rule. The overrides then went
in that direction, and `title_overlap` - the job-family feature - is the coefficient that
moved most as a result. No model prediction was ever shown during labelling, so this is
not leakage; but the labels are not a blind sample either, and a reader should know that
the review was steered toward job-family consistency before reading the coefficients.

**Synthetic augmentation, stated plainly.** 154 of the 294 pairs above come from 11 synthetic candidates added purely to grow and rebalance the training set (see training/make_synthetic_resumes.py and training/build_pairs_synthetic.py) - they were never real applicants and never reached the production Qdrant collection or the live app. Each was written as a deliberately unambiguous fit for one job family, which is why this batch is easier than the original 140 rows: the near-zero override rate on it reflects that design, not a claim that the Evaluator has gotten better at the genuinely hard cases the original corpus contains. Two rows from this batch are worth reading anyway, because they show the LLM diverging from the keyword features in the way the whole model design depends on: a senior B2C product manager scored a partial fit against a B2B-specific requisition despite a perfect title match (skill_overlap 0.0, correctly downgraded on domain), and a sales candidate scored a strong fit despite a low literal skill_overlap (0.33) because the Evaluator read her CV's own phrasing as equivalent to the requisition's wording.

## Cross-validation

Both are grouped, never random: the same CV appears in every requisition's rows, so a
random split leaks a candidate across folds. Scaling is fitted inside each fold.

| split | folds | MAE (pts) | fold MAE range | band accuracy | Spearman |
|---|---|---|---|---|---|
| leave-one-candidate-out | 21 | 0.77 | 0.09-1.96 | 89% | 0.66 |
| leave-one-requisition-out | 14 | 0.77 | 0.09-1.50 | 90% | 0.65 |

Leave-one-requisition-out is the number to quote when asked whether this generalises:
the effective sample size on the role side is the number of requisitions, not the number
of pairs. Neither split is fully clean - a held-out candidate's fold still shares
requisitions with training - and a properly blocked double-holdout is not viable at this
size.

## Baselines

Both metrics, because they disagree: on a set that is mostly no-fit, always predicting
no-fit scores badly on MAE and very well on accuracy. Quoting whichever one flatters the
model is the easiest way to overstate it.

| baseline | MAE (pts) | band accuracy |
|---|---|---|
| skill_overlap alone | 0.84 | 86% |
| similarity alone | 1.19 | 81% |
| predict the majority band | 1.20 | 83% |
| predict the mean | 2.00 | - |
| **the model (leave-one-candidate-out)** | **0.77** | **89%** |

**Verdict: the model beats every baseline on both MAE and band accuracy.**

The majority-band baseline looks strong because most pairs are obvious no-fits. That is
why per-band error below matters more than the overall number.

**Read that verdict carefully.** The margin over *skill_overlap alone* is +0.07 points of MAE on 294 rows, far inside the fold-to-fold spread of 0.09-1.96. The honest claim is that the five features are *no worse* than the single best one, not that they are measurably better. What the extra features buy is interpretability - a coefficient table that says why a candidate scored what they did - rather than accuracy.

### The number that matters most

Of 26 candidates labelled a strong fit, the model recovers **50%** (13 missed).
That is the figure to quote about usefulness, not overall accuracy: a screening aid
that under-rates good candidates fails at the job, and with most pairs being obvious
no-fits the headline accuracy barely moves when it happens. It is also why the
estimate is advisory in the report and never touches SCORE_THRESHOLD.

## Error by true band

| true band | n | MAE (pts) |
|---|---|---|
| 0 | 244 | 0.52 |
| 1 | 24 | 1.73 |
| 2 | 26 | 2.19 |

## Confusion (leave-one-candidate-out)

| true \ predicted | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 231 | 13 | 0 |
| 1 | 5 | 18 | 1 |
| 2 | 1 | 12 | 13 |

## Coefficients

Ridge, alpha 10.0. Units are points out of 10 per standard deviation of
the feature, so they are directly comparable with each other.

| feature | coefficient | permutation importance |
|---|---|---|
| skill_overlap | +1.219 | 0.603 |
| years_surplus | -0.017 | -0.000 |
| years_shortfall | -0.404 | 0.073 |
| title_overlap | +1.014 | 0.453 |
| similarity | +0.286 | 0.074 |
| _intercept_ | +1.204 | |

## Largest residuals

The three pairs the model gets most wrong out of fold. Worth reading before trusting it.

| requisition | candidate | true | predicted |
|---|---|---|---|
| req-t09-sales-ae | Emma Garcia | 9.0 | 1.7 |
| req-t05-data-analyst-senior | Noah Kim | 0.0 | 5.9 |
| req-t07-product-manager | Olivia Martinez | 9.0 | 3.5 |

## Deliberately hard pairs

6 pairs where the honest answer is arguable. Out-of-fold MAE **2.02 pts** against 0.77 overall - the gap is the honest
measure of how much of the headline number comes from easy negatives.

| requisition | candidate | true | predicted |
|---|---|---|---|
| req-t01-backend-senior | James Anderson | 9.0 | 6.2 |
| req-t04-data-analyst | Noah Kim | 9.0 | 8.8 |
| req-t06-devops | Alice Chen | 5.0 | 4.4 |
| req-t10-platform-adjacent | Alice Chen | 9.0 | 4.5 |
| req-t10-platform-adjacent | Ryan Patel | 9.0 | 6.0 |
| req-t14-fullstack | Liam Wong | 5.0 | 4.1 |
