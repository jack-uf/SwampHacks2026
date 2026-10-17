# Flood forecasting: frozen protocol

This study may fit models only after the historical label and feature-availability audit passes. Current flood thresholds alone do not establish historical flood labels. A failed audit ends the study with a documented feasibility result, without metrics or production changes.

## Question and scope

Predict a measured river stage at or above the applicable NWS minor flood stage during the interval from 24 to 48 hours after a decision at 00:00 UTC. Report a separate onset analysis for decisions below flood stage, so persistence during a long flood cannot masquerade as advance warning. The experiment is isolated under `experiments/sprint-20261017-flood/`. Production models, application data and earlier artifacts are outside its write scope.

The research design review prioritizes label validity, connected watersheds, publication time and independent events over model complexity. The user authorized autonomous implementation; no interview or additional test-interface approval is required. Tests cover the public label-contract validator and observation-window builder, if a builder becomes justified. They use fictional fixtures only, outside experiment data.

## Feasibility audit

Start with six named gauges selected for different hydrological settings, before inspecting their historical observations. Candidate LIDs are POPW2, HRSP1, PORM2, RALN7, DALT2 and CRNW1. Resolve each USGS identifier from NOAA metadata, never from proximity. Read current USGS monitoring-location metadata and a short January 2024 continuous-data sample before any multi-year request. Failure to resolve a candidate is a recorded exclusion, not permission to silently substitute a station.

For each gauge, require:

1. An exact USGS identifier join, matching physical location within one kilometre, correct stage units and a surface-water station.
2. Documentary evidence of the NWS threshold's effective interval covering the study period. A current API response, old crest list or lack of a change notice is insufficient proof of an unchanged threshold.
3. A common gauge zero and vertical reference over that interval, or an authoritative dated conversion. Equal vertical datum names alone are insufficient. Do not infer an offset from rounded elevations or assume a gauge height is an absolute elevation.
4. Measured stage timestamps, units, quality qualifiers and sufficiently complete coverage. Never use daily mean stage to label an instantaneous exceedance. Exclude estimated, ice-affected, rejected and missing observations under the selected service's documented qualifier meanings.
5. Historical publication or ingestion evidence for predictor observations. Final approved archives can label outcomes, but cannot establish what stage or flow was available at a historical decision. Record this limitation separately from measurement timestamps.

Inspect official NOAA API schemas, USGS collections, station pages and applicable official change notices. Absence of historical metadata in these sources is a scope-limited finding, not a claim that no archive exists. If no sampled gauge has a defensible historical contract, stop before large downloads or modeling. More event data cannot repair an unknown label definition.

## Population, split and expansion rule

If the audit passes, use 2024 for fitting, 2025 for selection and January 1 through October 14, 2026 for a single final forward-time test. These river labels have not been inspected by the previous wind experiments. Set aside entire independent drainage systems before collecting multi-year outcomes. HUC codes support the grouping but do not prove independence: connected upstream and downstream reaches must remain in the same group, even across HUC boundaries. Freeze an explicit gauge-to-drainage-group registry and roles before acquisition. No station identifier enters the model.

Require at least three independent drainage systems in fitting, two in selection and two in the final geographic test. Require at least 30 distinct flood episodes in fitting and 20 each in temporal and geographic tests, across at least two gauges per test. Separate episodes by seven consecutive below-threshold days. Add audited gauges or extend the fitting period backward, keeping the forward test fixed, if counts fail. Record an amendment before acquisition and never expand because a model's score disappoints. If sufficient audited independent events cannot be obtained within this sprint, stop and report insufficiency.

## Decision-time feature contract

Hydrology uses only stage and flow transmitted before the decision, with a minimum two-hour observation lag plus evidence of actual availability. Candidate features are distance below flood stage, 6-, 24- and 72-hour stage change, preceding flow and its change, and missingness. Upstream gauges require an explicit directed river-network connection and the same time rules; nearest-gauge joins are prohibited. Normalize transferable stage features by an audited station scale fitted on training data or a documented physical quantity, not test distributions.

Weather uses archived GFS runs issued and available before the decision. Basin precipitation aggregates are preferable to a point forecast; a point-only pilot must state that limit. Include forecast rain for the target interval, antecedent rain available at decision time, temperature and snowmelt indicators only if supported by the archived variables. Do not reuse the wind experiment's per-hour previous-run fields without checking that every value derives from a run available at this single decision time.

Satellite candidates are NDWI, NDVI, classified surface-water fraction, antecedent water change, cloud/valid fractions, scene age and missing indicators. Require both acquisition and publication before the decision, preserve georeferencing and mask provenance, and use basin or floodplain geometry rather than arbitrary station buffers for hydrological claims. A one-day latency assumption alone does not prove availability. The existing Sentinel-2 pipeline is reusable machinery, not certified data: inspect pixel alignment, reflectance offsets, cloud masks, pagination and cached scene age before reuse. Soil moisture and terrain may enter only with verified product availability and drainage relevance. No new advisory Jev image calls are justified by the failed visual quality pilot.

## Models and selection

Compare persistence and a fixed recent-stage-trend baseline with logistic regression, XGBoost and LightGBM on hydrology-only, weather-only, weather plus satellite, weather plus hydrology, and the combined features. Use identical eligible rows across each satellite ablation and also report missing-scene coverage. Fit imputation, scaling and calibration on fitting/selection data only. Keep each model family in its own subprocess to avoid the known OpenMP conflict.

Use seed 42 and the existing wind experiment's fixed boosting/logistic configurations. Select on mean selection AP across forward and unseen-drainage slices. Select thresholds on selection data at false-alert budgets of 1 and 5 false alerts per 100 decisions. Thresholds stay frozen on final tests; any test-derived matched-budget curve is descriptive only.

Run the pinned Apache-2.0 TabPFN v2 checkpoint with its recorded hash and benchmarked MPS configuration only if the label/availability gates pass and it fits the existing 10K-row budget. Use a deterministic training-only sample if needed, and report sampling. A temporal neural architecture is justified only after complete hourly sequences and sufficient independent events exist and the simpler models leave a credible gap. No hyperparameter search in this sprint.

## Evaluation and integration gates

Report AP, prevalence, precision, recall, Brier score, log loss, reliability bins and false alerts per 100 decisions. Report pooled and per-gauge/drainage results, onset recall and actual time from issue to the first crossing. A retained flood episode is not a new onset. Report persistence at the same budgets and avoid interpreting station-days as independent floods.

Estimate paired satellite-minus-no-satellite AP differences with 2,000 resamples of drainage groups and seven-day blocks, seed 42, keeping paired rows together. Report event counts and instability where few basins/events make intervals unreliable. Satellite value requires a positive lower 95 percent bound on both independent forward and geographic tests, no recall loss at the frozen budgets, and no Brier regression greater than 0.005. Custom ML must also beat persistence on onset recall at a matched selection budget. A threshold-validity or availability failure prevents a deployment recommendation regardless of scores.

## Resource and completion rules

One model worker at a time, at most 6 GB worker RSS and 500 MB new satellite cache. Do not install a dependency or request large imagery before the label gate passes. Pause scaling if macOS memory pressure or swapping rises. Cache and hash the small API responses, URLs, retrieval times, HTTP status and errors. Avoid retry storms; NOAA requests are paced to at most ten in five minutes. Do not interrupt other evaluations.

Commit this protocol before modeling. Any necessary amendment is dated and committed before the affected experiment. Run focused tests, then the full Python suite. Record measured results or the exact stopping gate, resource use, limitations and next required evidence. Commit verified work. Nothing enters the production hazard index in this sprint.
