# Leave-one-state-out diagnostic: protocol

Frozen October 17, 2026, before the first run. Committed separately so the commit history shows it came first.

## Question

How much ranking skill and calibration do the two served model recipes lose in a state they were never fitted or calibrated on? The current evaluation answers this only with the geographic holdouts, which hold 1 to 4 positives per hazard and are already inspected.

## Data

`data/training.csv`, rows with `role == 'development'` and `target_date < '2026-01-01'`: 18 counties in 12 states. The script drops every other row when it loads the file and asserts that no 2026 date and no geographic-validation or geographic-test county remains. The January to June 2026 final test and the six holdout states are never read.

## Recipes

Fixed hyperparameters, exactly as in `scripts/train.py`. No search.

- Flood: standardized, class-balanced logistic regression on the eight base features. This is the recipe of the served frozen baseline, refitted on the expanded data.
- Storm: class-balanced LightGBM on the enhanced features, the served `lightgbm_enhanced` recipe.

Both get sigmoid calibration on logit probabilities, as served.

## Periods and arms

Fit on 2024, calibrate on January to June 2025, score July to December 2025.

- Seen-state arm: fit and calibrate on all 12 states; score every state's July to December 2025 rows.
- Left-out arm: for each state, fit and calibrate on the other 11 states only; score that state's July to December 2025 rows. A fold whose calibration rows lack a class is reported as failed and its rows are dropped from both arms.

Both arms score the same rows, so the comparison is paired.

## Metrics

Pooled over all scored rows, per arm and hazard: rows, positives, average precision, Brier score, expected positives (sum of probabilities) against observed positives, and recall and false alerts per 100 county-days at the served thresholds (flood 0.0306398, storm 0.0756798). Per-state average precision and positives are listed as well.

Uncertainty: 300 paired calendar-week block bootstrap resamples (seed 42) of the left-out minus seen-state difference in average precision and Brier score, the same method as `scripts/train.py`. The intervals are descriptive.

## Known limits, stated before the run

- July to December 2025 was the selection period and the served thresholds were chosen on it, so threshold metrics in the seen-state arm are optimistic. Average precision and Brier score do not depend on the threshold.
- 12 states give 12 folds of very different size and event count; states with few positives make per-state numbers noisy.
- Counties in neighbouring states share weather regimes, so leaving one state out still understates the gap for a new region.

## Decision rule

Diagnostic only. Whatever the result, no model, threshold, calibration or serving scope changes. The result goes into `docs/EVALUATION.md` and `docs/MODEL_CARD.md` as a measured statement of geographic transfer, with this protocol linked. Output: `models/spatial-cv.json`.
