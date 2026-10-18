# Hybrid MLP onset experiment: frozen protocol

Frozen 2026-10-17 before any cohort observation is acquired and before any model is fitted. This is a bounded retrospective experiment on a research prototype, not field evidence.

## Frozen question and scope

The [MLP onset experiment](MLP_ONSET_RESULTS.md) found that a plain perceptron ranked hourly high-flow onset best at 24 hours in both partitions and warned earlier, then failed promotion because a six-hour damped trend projection detected more events at the frozen one-percent alert budget. This experiment asks two separate questions that the earlier design could not separate: does giving the network the damped projection as a predictive offset improve it, and does selecting the alert threshold against unique events rather than row true positives recover the detections?

The target is identical to the earlier hourly studies: hourly instantaneous discharge above the gauge's 2000-2015 daily-mean 95th percentile after 24 consecutive accepted hourly readings at or below it. This is high flow, not verified flooding, road inundation or a crew-safety outcome.

## Cohort, independence and roles

The earlier cohort's 2025 outcomes are inspected, so this experiment draws a second cohort from the same CAMELS registry after excluding every station identifier any prior experiment registered. The two cohorts are disjoint. Gauges are grouped by HUC4 and ordered inside each HUC2 region by the first SHA256 ordering of `hyb29` plus the station ID; regions contribute at most two gauges from distinct HUC4 groups. Within the cohort, one HUC4 group per region is reserved for the geographic partition before any outcome is seen.

This cohort is fresh at the station level. It shares HUC2 regions with earlier studies, and its HUC4 groups were available only because earlier selections did not use them, so no regional-climate independence is claimed.

Exclusions for missing or unacceptable instantaneous discharge are recorded, never replaced after viewing outcomes.

## Data and partitions

Official USGS OGC v1 approved instantaneous discharge (parameter 00060, statistic 00011), right-closed hourly bins, decisions at 00, 06, 12 and 18 UTC using observations only through issue time minus two hours, 169 consecutive accepted hourly values and 24 quiet hours required for eligibility. Features are the frozen shift set recorded in the [earlier protocol](MLP_ONSET_PROTOCOL.md).

Fit on development gauges in 2020-2022. Fit probability calibrators on development gauges in 2023. Select models and thresholds on 2024 development plus geographic-validation rows, weighting the two partitions' average precision equally. Purge label-crossing decisions at calendar-year boundaries. Evaluate once in 2025 on the development partition (temporal test) and the reserved geographic partition (geographic test). No 2025 row enters any fitting or selection decision, and final-year acquisition starts only after selection is locked.

## Candidates

Every candidate fits identical rows and is calibrated by one-variable logistic regression on 2023.

- **Physical baselines.** Flow proximity, a linear six-hour trend projection, and a damped six-hour trend projection.
- **Statistical and boosted comparators.** Logistic regression, LightGBM and XGBoost on the full feature set, at the earlier experiment's frozen configuration.
- **MLP direct.** The two-layer 64-unit SiLU perceptron from the earlier experiment, unchanged, as the reproduction control.
- **MLP hybrid.** A network over the same features whose logit is added to the standardized damped projection before calibration, so the network learns a correction to the physical extrapolation instead of rediscovering flow persistence.
- **MLP hybrid prior.** The hybrid network plus four trailing exceedance-history features derived from the row sequence: whether discharge at the two-hour cutoff is above the reference, the fraction of the trailing 30 and 90 days above it, and a capped count of days since the last exceedance.

Each MLP is fitted once and receives both threshold rules, so the architecture question and the operating-point question are separated rather than confounded.

## Threshold rules

Both rules see only pooled 2024 validation rows and both respect a budget of one false alert per 100 decisions, counting tied scores together.

- **Row rule.** Maximize row-level true positives under the budget. This is the earlier experiment's rule and the reproduction control.
- **Event rule.** Maximize the number of unique validation events detected under the same budget, breaking ties by fewer false alerts and then by higher threshold.

The promotion gate measures event recall, so the event rule is the hypothesis under test and the row rule is its control. No rule may use final-year outcomes.

## Evaluation and promotion gates

Report average precision, Brier score, reliability bins, precision, decision recall, unique-event recall, false alerts per 100 decisions and per station-day, false-warning episodes, first-warning lead, and per-HUC2 and per-station results, plus event coverage with the reason each uncounted onset was dropped.

Promotion requires all of: at least 20 eligible events in each final partition and 50 in total at the evaluated horizon; a paired HUC4 cluster bootstrap 95% lower bound above zero on the average-precision difference against the selected baseline; an event-recall gain of at least five percentage points in both partitions with a pooled paired lower bound above zero; a Brier score no worse in either partition; no final HUC2 above three false alerts per 100 decisions with a pooled burden at most 1.5; and no loss of median lead on jointly detected events. At least one horizon must pass every statistical gate. Operational integration additionally requires historical input-availability evidence and a site-to-gauge applicability assessment, neither of which this retrospective experiment can supply.

A failed gate is a result, not permission to relax the criterion or substitute a candidate after inspecting the final year. Selection picks the strongest non-baseline candidate and the strongest physical baseline per horizon by mean 2024 temporal and geographic validation average precision, breaking ties by lower mean Brier and then by name, exactly as before.

## Reproduction and interpretation

Run the `freeze`, `development`, then worker fitting commands, lock selection, then `test` acquisition and the one final evaluation. Sources, dataset hashes, candidate artifacts, predictions, coverage and machine-readable gate decisions stay under `experiments/sprint-20261017-mlp-hybrid`. Production models, safety rules and the served hazard index are unchanged by this study.

Primary sources: USGS OGC documentation at `https://api.waterdata.usgs.gov/docs/ogcapi` and the continuous collection schema. The offset formulation follows the historical-transition study's use of a physical prior as a training starting point rather than a feature.
