# Connected-river development pilot

Freeze this protocol before collecting upstream discharge or fitting transport models. This pilot tests whether a connected upstream hydrograph adds useful information under PlanIT's existing daily observation delay. It cannot establish geographic transfer. Keep all thirteen metadata-eligible, uninspected downstream watersheds unopened.

## Population and feasibility

The metadata screen covers all 62 existing development gauges and 31 new basins whose simplified NLDI polygons have zero positive-area intersection with the 215 previously reserved gauges and with earlier accepted new basins. The new basins are distinct catchments, not necessarily new climate regions. Navigation uses upstream tributaries within 500 km, never nearest-neighbor proximity.

Require an eight-digit USGS stream gauge with daily mean discharge parameter 00060 and statistic 00003. Its catalog must begin no later than January 1, 2000, end no earlier than September 25, 2024 and contain at least 7,000 records. Require upstream drainage area between 5% and 95% of downstream area and an indexed location within 1 km of USGS coordinates. Pick the largest qualifying area ratio, breaking ties by identifier. These are discovery filters, not proof of complete or approved observations. NLDI indexes whole local catchments and does not establish precise hydraulic connectivity or travel time.

Only four development pairs qualify. All four enter the pilot, with no replacement based on outcomes. No independent geographic model claim is permitted. The thirteen potential test pairs are reserved, not a validated test set. Do not request their discharge in this study.

## Target, timing and features

Reuse the existing onset training table and its archived ECMWF HRES forecasts. This is not a GFS experiment. At issue time 00:00 UTC, predict a new downstream daily high-flow event on local D+1, D+2 or D+3 after seven quiet days. Use the existing onset labels and completeness filters without modification. The reference is the approved 2000 through 2015 daily discharge 95th percentile. A percentile exceedance is not verified flooding.

Both gauges' latest predictor observation is local D-3. Calendar lags, not row offsets, determine D-3, D-4 and D-7. Normalize each gauge by its own pre-2016 reference and apply log1p. Raw upstream features are those three lags, D-3 minus D-4, and the drainage-area ratio.

The proposed transport innovation measures the additional trajectory predicted from upstream flow. For each pair, fit separate standardized ridge regressions with alpha 10 on complete reference-period days. Local inputs are normalized downstream flow at t, t-1 and t-4, plus seasonal sine and cosine. Augmented inputs add the matching upstream flows. Targets are normalized downstream flow at t+4, t+5 and t+6. Require all inputs and targets to fall before 2016, and at least 2,000 complete rows per lead. At an issue date, t is D-3. Each innovation is augmented prediction minus local prediction. Fit no coefficient using later flow. This is statistical error correction, not a hydraulic routing model or a measured propagation delay.

Approved observations may contain retrospective revisions. D-3 observation availability and D-1 forecast availability remain assumptions without historical publication timestamps. Do not describe the resulting pilot as an operational hindcast.

## Models and evaluation

Use identical eligible decision rows for all candidates, dropping rows missing any required upstream predictor or innovation. Reuse the prior training table's weather, hydrology and snow features. Compare forecast rainfall ranking, local hydrology LightGBM, weather plus local hydrology and snow LightGBM, that combined baseline plus raw upstream features, and the combined baseline plus transport innovations. Add logistic regression on the combined baseline as a simple comparator.

LightGBM uses 200 trees, learning rate .04, maximum depth 3, seven leaves, seed 42, deterministic column-wise fitting and two threads. Logistic regression uses standardized inputs, C=1 and maximum 2,000 iterations. No tuning, architecture search or seed search is allowed. Transport fitting uses only the reference period; classifier fitting uses 2017 through 2020. Fit probability calibration on 2021 using logistic regression on raw scores. Set 0.5%, 1% and 2% false-warning thresholds on 2021 as well. Report calibration and threshold reuse as a limitation.

Evaluate once on 2022, excluding January 1 through 7 and December 25 through 31 as in the prior onset study. This period and these downstream gauges have been inspected in previous studies, so every result is exploratory development evidence. Report average precision, Brier score, log loss, reliability bins, event detection, false warnings, lead time and per-pair results. Compare raw upstream and innovation ablations with the combined local baseline. Also report uncalibrated ranking AP so a calibration slope cannot hide a reversed physical ranking. Preserve all candidates, not just the best result.

A further geographic study is worth proposing only if an upstream candidate improves AP by at least .02 and detects at least as many events at its frozen 1% threshold, without increasing false warnings by more than 10%. This is a development gate, not independent validation or a deployment criterion. Failure stops this pilot; passing still requires a larger connected development population, precise connectivity checks and genuinely untouched watershed outcomes.

## Preservation and reproduction

Commit this document, metadata snapshots and their protocol hashes before upstream flow collection. Record raw request hashes, paired source coverage, row exclusions, fitted artifacts, package versions, runtime and peak RSS. Reconstruct causal features from retained raw data and rerun focused tests and the regression suite. Keep the CPU worker separate from Torch. Use no paid API or new dependency. Preserve production model hashes, previous experiments, application data and untracked UI work.
