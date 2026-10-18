# MLP hourly onset experiment: frozen protocol

Frozen 2026-10-17 before any cohort observation is acquired and before any model is fitted. This is a bounded retrospective experiment on a research prototype, not field evidence.

## Frozen question and scope

Does a multilayer perceptron predict hourly high-flow onset better than the six-hour damped-trend projection, LightGBM and XGBoost, and does it detect more emerging events at a matched false-alert budget with useful lead time? The earlier response experiment found that a small pooled neural mixture detected 61 measured high-flow events against LightGBM's 44 on a daily target, then failed promotion on regional false alerts and an average-precision interval that included zero. That daily result does not transfer to an hourly target, so this experiment retests the architecture question on the hourly pipeline rather than assuming the advantage reproduces.

The target is identical to the [hourly high-flow onset experiment](SHIFT_HYDROLOGY_PROTOCOL.md): hourly instantaneous discharge above the gauge's 2000-2015 daily-mean 95th percentile after 24 consecutive accepted hourly readings at or below it. This is high flow, not verified flooding, road inundation or a crew-safety outcome. Equality is not an exceedance.

## Cohort, independence and roles

The cohort is drawn from the full CAMELS metadata registry and excludes every station identifier recorded by a prior experiment's registry, plus every identifier in the shift experiment's cohort. Every eligible CAMELS HUC4 has already appeared in an earlier reservation, so this cohort is fresh at the station level but not at the drainage or regional-climate level. That limitation travels with every geographic claim here.

Screening requires a drainage area between 20 and 1,200 square miles, a known elevation and an eight-digit HUC. Gauges are grouped by HUC4 and ordered inside each HUC2 region by the first SHA256 ordering of `mlp17` plus the station ID. Regions contribute at most two gauges. Within the cohort, up to two HUC4 groups per region are reserved for the geographic partition before any outcome is seen, and the remaining gauges form the development partition. Development and geographic partitions therefore share regions but not HUC4 groups.

Exclusions for missing or unacceptable instantaneous discharge are recorded, never replaced after viewing outcomes.

## Data, timing and features

Official USGS OGC v1 `continuous` discharge, parameter 00060, instantaneous statistic 00011, cubic feet per second, approved values with a null qualifier. Ambiguous series, duplicate timestamps, unexpected units and pagination anomalies are rejected. Compressed responses, exact URLs, retrieval times and checksums are retained.

Right-closed hourly bins retain the last accepted reading in the preceding hour without interpolation. Decisions fall at 00, 06, 12 and 18 UTC and use observations only through issue time minus two hours. Eligibility requires 169 consecutive accepted hourly values and 24 quiet hours through that same cutoff. Labels cover `(issue, issue + horizon]` and require every following hour observed.

Features are the frozen shift set: log1p discharge divided by the historical reference at lags 2, 8, 14, 26, 50, 74 and 170 hours; multiscale rises at 6, 12, 24, 48, 72 and 168 hours; six-hour acceleration; 24 and 72 hour ranges; annual phase; latitude; longitude; log drainage area; and elevation. These are causal hydrological representations, not a measured soil-water balance. The two-hour observation delay is an explicit assumption: approved USGS records are retrospective and no as-issued replay was available for this cohort.

## Partitions

Fit on development gauges in 2021-2022. Fit probability calibrators on development gauges in 2023. Select models and alert thresholds on development plus geographic 2024 rows, weighting the two partitions' average precision equally. Purge decisions whose 24-hour labels cross a calendar-year boundary. Evaluate once in 2025, separately on the development partition (temporal test) and the reserved geographic partition (geographic test).

Development acquisition covers 2021 through 2024. Final-year acquisition for 2025 starts only after model, calibrator and threshold hashes are locked. No 2025 row enters any fitting, calibration, early-stopping or selection decision.

## Candidates

Every candidate fits identical rows, is calibrated by one-variable logistic regression on 2023, and receives one threshold chosen on pooled 2024 validation rows at a budget of one false alert per 100 decisions, including tied scores.

- **Physical baselines.** Flow proximity (`q_2`), a linear six-hour trend projection, and a damped six-hour trend projection. These are the strong cheap comparators the shift experiment selected.
- **Gradient boosting.** LightGBM and XGBoost on the full feature set, fixed at the shift experiment's frozen configuration. No hyperparameter search.
- **MLP direct.** A two-layer 64-unit SiLU perceptron with dropout 0.2 on the full feature set.
- **MLP residual.** An input projection into 64 units followed by two residual blocks, each a 64-unit SiLU pair with dropout and an additive skip connection.
- **MLP multitask.** A shared 64-unit trunk with one output head per horizon, trained jointly across the three horizons and evaluated per horizon.

Networks train on CPU with two threads, deterministic algorithms, a fixed seed, AdamW, gradient clipping and early stopping on the 2023 development year. No tuning search is run: every architectural and optimization setting is fixed in code before fitting. A negative calibrator slope fails that candidate rather than reversing its interpretation.

## Evaluation and promotion gates

Report average precision, Brier score, probability reliability bins, precision, decision recall, event recall on distinct onsets, false alerts per 100 decisions and per station-day, false-warning episodes separated by more than 24 hours, first-warning lead, and per-HUC2 and per-station results. Report event coverage: every certified onset, how many have an eligible prior decision, and losses attributed to missing history, the quiet interval, the observation buffer or the year boundary.

Promotion requires all of: at least 20 eligible events in each final partition and 50 in total at the evaluated horizon; a paired HUC4 cluster bootstrap 95% lower bound above zero on the average-precision difference; an event-recall gain of at least five percentage points in both partitions with a pooled paired lower bound above zero; a Brier score no worse in either partition; no final HUC2 above three false alerts per 100 decisions with a pooled burden at most 1.5; and no loss of median lead on jointly detected events. At least one horizon must pass every statistical gate. Operational integration additionally requires historical input-availability evidence and a site-to-gauge applicability assessment, neither of which this retrospective experiment can supply.

A failed gate is a result, not permission to relax the criterion or substitute a different candidate after inspecting the final year.

## Reproduction and interpretation

Run the `freeze`, `development`, then worker fitting commands, lock selection, then `test` acquisition and the one final evaluation. Sources, dataset hashes, candidate artifacts, predictions, coverage and machine-readable gate decisions stay under `experiments/sprint-20261017-mlp-onset`. Production models, safety rules and the served hazard index are unchanged by this study.

Primary sources: USGS OGC documentation at `https://api.waterdata.usgs.gov/docs/ogcapi` and the continuous collection schema. The architecture follows the response experiment's pooled network, which is a plain SiLU perceptron rather than an attention model.
