# Geographic generalization study: frozen protocol

Frozen 2026-10-18 before any cohort observation is acquired and before any model is fitted. This is a bounded retrospective experiment on a research prototype, not field evidence.

## Frozen question

The [hybrid MLP experiment](MLP_HYBRID_RESULTS.md) reported that a 6-hour perceptron with a ceiling-based threshold beat the linear extrapolation on temporal event recall and cleared ten of thirteen promotion gates. Its comparison was not matched: the network used the ceiling rule while every comparator used the stricter row rule, so the network was scored at a more permissive operating point than its baselines. This experiment tests whether the advantage survives a matched comparison, on a third cohort whose outcomes no prior experiment has inspected.

Two hypotheses are frozen together.

- **H1, matched operating points.** Under an identical threshold rule, the multilayer perceptron does not beat the damped-trend and linear baselines or the boosted models on event recall in both final partitions. If it does, the earlier recall claim holds up.
- **H2, geographic ranking.** In a larger reserved geographic partition, the perceptron's average-precision advantage over the linear baseline excludes zero under paired HUC4 resampling.

Both hypotheses are falsifiable by the frozen gates below. The target is unchanged: hourly instantaneous discharge above the gauge's 2000-2015 daily-mean 95th percentile after 24 consecutive accepted hourly readings at or below it, which is high flow, not verified flooding.

## Cohort

Every station identifier registered by any prior experiment in this repository is excluded before screening. This cohort takes one development gauge per HUC2 region and, where a region has a second eligible gauge in a different HUC4 group, one geographic gauge from it, which spreads coverage across more regions than either earlier cohort. Gauge order inside a region is the first SHA256 ordering of `geo31` plus the station ID.

All eligible CAMELS HUC4 groups have already appeared in an earlier reservation, so this cohort is fresh at the station level only. No watershed-polygon screen was applied, and no regional-climate independence is claimed. That limitation is the main threat to H2 and is reported with the result.

## Data and partitions

Official USGS OGC v1 approved instantaneous discharge (parameter 00060, statistic 00011), right-closed hourly bins, decisions at 00, 06, 12 and 18 UTC using observations only through issue time minus two hours, with 169 consecutive accepted hours and 24 quiet hours required for eligibility. Features are the frozen shift set.

Fit on development gauges in 2020-2022, calibrate on 2023, select on 2024 development plus geographic-validation rows, evaluate once in 2025. Labels crossing a calendar-year boundary are purged. Final-year acquisition begins only after selection is locked, and no 2025 row enters fitting, calibration or selection.

## Candidates and matched rules

Every candidate family, including the physical baselines and both boosted models, is scored under both threshold rules. This is the change that makes the comparison fair, and it is the reason for the new experiment rather than an extension of the last one.

- **Rules.** The row rule maximizes row true positives under one false alert per 100 validation decisions. The ceiling rule takes the lowest validation threshold whose pooled false-alert rate stays at or below 1.5 per 100 and whose every region stays at or below three, matching the promotion gate's own ceilings.
- **Candidates.** Flow proximity, linear and damped trend projections; logistic regression; LightGBM; XGBoost; and four perceptrons (direct, damped-offset hybrid, hybrid with trailing exceedance history, and hybrid with a balanced loss), each receiving both rules.

Networks reuse the hybrid experiment's architectures and training configuration unchanged, so any difference is attributable to the cohort and the matched rules.

## Evaluation and promotion gates

Report average precision, Brier score, reliability bins, precision, decision recall, unique-event recall, false alerts per 100 decisions and per station-day, false-warning episodes, first-warning lead, and per-HUC2 and per-station results, with event coverage and the reason each uncounted onset dropped.

Promotion requires all of: at least 20 eligible events in each final partition and 50 in total at the evaluated horizon; a paired HUC4 cluster bootstrap 95% lower bound above zero on the average-precision difference against the selected baseline; an event-recall gain of at least five percentage points in both partitions with a pooled paired lower bound above zero; a Brier score no worse in either partition; no final HUC2 above three false alerts per 100 decisions with a pooled burden at most 1.5; and no loss of median lead on jointly detected events. At least one horizon must pass every statistical gate. Integration additionally requires historical input-availability evidence and a site-to-gauge applicability assessment, neither of which a retrospective experiment can supply.

Selection picks the strongest non-baseline and the strongest physical baseline per horizon by mean 2024 validation average precision, breaking ties by lower mean Brier then by name. A failed gate is a result. Criteria are never relaxed after inspecting the final year.

The matched comparison is reported separately for every architecture at both rules, so a reader can see whether any apparent advantage is an operating-point artifact.

## Reproduction and interpretation

Run the `freeze`, `development`, then worker fitting commands, lock selection, then `test` acquisition and the one final evaluation. Sources, dataset hashes, artifacts, predictions, coverage and machine-readable gate decisions stay under `experiments/sprint-20261017-mlp-geo`. Production models, safety rules and the served hazard index are unchanged by this study.
