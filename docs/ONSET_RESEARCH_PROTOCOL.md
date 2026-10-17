# Emerging high-flow warning protocol

This experiment tests whether onset-specific learning, residual discharge trajectories and delayed-feedback alert control improve warnings of newly emerging high flow. It is separate from flood-stage prediction. All candidates remain outside production. This document and a metadata-only station registry must be committed before collecting new outcomes or fitting models.

## Population and independent evaluation

Reuse the archived Caravan MultiMet v1.1 ECMWF HRES forecasts and exact-approved USGS daily mean discharge parser from `scripts/hydrology_research.py`. Retain the source licenses and quality rules in [the hydrology report](HYDROLOGY_RESEARCH_RESULTS.md). References use only 2000-2015 observations, require 3,000 accepted days, and define high flow as discharge strictly above the station's reference 95th percentile. This is not an official flood threshold.

Development uses the previous station registry outside HUC2 01, 02 and 04. HUC2 03 and 12 remain geographic selection regions; other retained regions supply fitting and calibration. Earlier inspected western outcomes are development knowledge, never new independent evidence.

Fresh test stations come from the cached 671-station USGS metadata inventory. Require HUC2 01, 02 or 04 and a HUC4 absent from every station in the previous 96-station registry. Accept documented 8- or 12-digit HUC codes and use their first eight digits. Select the lowest station ID per HUC8 with coordinates, elevation, drainage area and an interpretable USGS timezone. A `local_time_fg=N` station uses fixed standard time. Freeze identities before requesting discharge. Exclude ambiguous discharge series or inadequate references without replacements.

Fit on development decisions in 2017-2020, calibrate in 2021, and select on 2022 development and geographic-selection decisions, with equal weight for the two selection cohorts. Exclude January 1-7 and December 25-31 from each development year. New final outcomes cover January 8, 2023 through September 25, 2024, at fresh test stations only. Report each test HUC2 and year separately. Do not inspect fresh final labels, prevalence or event counts until artifacts, source hashes, thresholds and model selection have been committed. Require at least 30 unique certified test onsets across eight HUC4s to make an improvement claim. Otherwise report feasibility only.

## Decision and target

Issue once per day at 00:00 UTC. Reuse weather runs dated D-1, leads 2-5, antecedent lead-1 forecasts through D-2, the 270-day complete-memory warmup and discharge lags D-3, D-4, D-7 and D-14. These availability buffers are assumptions, not verified historical publication records. Approved discharge may have been revised. No experiment here establishes an as-issued operational replay.

Only issue emerging-event predictions when all seven accepted daily flows D-9 through D-3 are at or below the reference. This eligibility screen uses information nominally available at issue. For labels, require complete accepted daily flows D-6 through D+3. A certified onset is a high-flow day preceded by seven accepted below-or-equal-reference days. The binary target is any certified onset on local days D+1 through D+3. Retain its date and lead time, and count a unique station/onset date once in event recall. Repeated warning decisions count separately toward alert burden. Missing days never become negative labels.

Also retain log-normalized flow trajectories for days D+1, D+2 and D+3. A daily-mean target cannot locate an instantaneous crossing. Forecast lead is measured to the start of the first certified onset's local day. No post-issue flow or eligibility information enters any predictor.

## Frozen model recipes

Use the previous experiment's static, seasonal, weather, hydrology and snow-memory features without station IDs or HUC identifiers. Add no unverified upstream joins or satellite features. Two-thread LightGBM workers use 200 trees, learning rate .04, seven leaves, depth three and seed 42. Logistic regression uses standardized inputs, C=1 and at most 2,000 iterations. No hyperparameter search.

Compare the following candidates on a common eligible population:

1. Calibrated lagged-flow proximity, calibrated linear flow extrapolation, and calibrated forecast rainfall ranking.
2. A LightGBM high-water classifier trained on any high day in D+1 through D+3, then calibrated against onset labels. This tests whether changing the target adds value beyond recognizing high water.
3. Direct onset logistic regression with combined weather, hydrology and snow features.
4. Direct onset LightGBM with weather only, hydrology only, combined inputs, and combined plus snow memory.
5. Residual trajectory learning: nine LightGBM quantile regressors for three lead days and quantiles .1, .5 and .9, predicting log-normalized flow minus its D-3 value. Sort predicted quantiles per day. An onset logistic head learns from all nine predicted absolute flow quantiles, three interval widths, and the D-3 flow and recent change. Train regressors only on fitting years and the head only on 2021 calibration. Its probabilities are evaluated without a second calibration pass.

Every other candidate uses a one-dimensional logistic probability calibrator fitted to 2021 onset outcomes. All continuous flow targets use all eligible decisions, not just positive events. Compare trajectory pinball loss and central 80% interval coverage with the lagged-flow point baseline; do not interpret marginal intervals as simultaneous trajectory coverage.

Select by highest mean onset AP across the two 2022 cohorts, then lower mean Brier, then name. Freeze the strongest direct onset classifier and high-water comparator as separate references. No refit or choice after final evaluation.

## Alert policies and delayed adaptation

Evaluate each candidate at frozen pooled-selection thresholds targeting 1 and 5 false alerts per 100 decisions, using the existing tie-preserving budget rule. Also freeze a conservative threshold equal to the maximum per-HUC2 threshold for each budget. Report the possible recall cost of this policy. It has no guarantee in an unseen region.

For the selected model, direct-onset reference and high-water comparator, evaluate one additional predeclared online policy at a 1% negative-score tail. At each station, combine the pooled 2022 negative-score empirical distribution with the station's negative scores from the previous 365 calendar days whose entire outcome window is at least eight days old. The source distribution has total pseudo-weight 100; each eligible local negative has weight one. The upper weighted 99th percentile supplies the threshold, with alerts strictly above it to keep ties conservative. No local final label is used early. This is prequential delayed-label adaptation, not a label-free geographic test or a conformal guarantee. Starting from the source distribution is the mathematical prior of this one policy, not a legacy-state fallback.

The frozen policies provide the untouched-label evaluation. The adaptive policy consumes only past test labels after its declared delay and must be reported separately. The eight-day delay remains an availability assumption for revised historical daily data.

## Metrics, uncertainty and decisions

Report onset AP, Brier, log loss, ten-bin reliability, precision, decision recall, unique-event recall, false alerts per 100 decisions, and onset warning lead. Include per-station, HUC2 and year results. Report test prevalence. A false alert is an alert without a certified onset in its three-day target window; it need not mean conditions were safe. Report descriptive attainable recall at a matched 1% test budget, explicitly as an oracle diagnostic rather than an operational threshold.

Use 500 seeded paired bootstraps by HUC4 and separately by decision calendar month for AP differences and unique-event recall differences. Retain whole clusters and paired predictions. Require positive AP lower bounds against both references under both resampling schemes, a unique-event recall gain of at least .10 over the high-water reference at the frozen 1% policy, false alerts no greater than 1.5 per 100 decisions, and adequate event coverage before claiming a useful static warning improvement. A secondary adaptive-policy claim additionally requires event recall no lower than the same model's frozen pooled policy, with fewer false alerts and no test HUC2 above 2 false alerts per 100 decisions. Report failures instead of revising the criteria.

## Research and stop rules

Primary-source reading motivates three distinct hypotheses. The [multi-timescale LSTM study](https://hess.copernicus.org/articles/25/2045/2021/) warns that daily targets miss short peaks and motivates multi-lead outputs. [Conformal risk control](https://arxiv.org/abs/2208.02814) motivates explicit loss budgets; this experiment does not assert its theorem under basin shift. [Covariate-shift conformal prediction](https://arxiv.org/abs/1904.06019) requires density-ratio conditions not established here. [Adaptive conformal inference](https://arxiv.org/abs/2106.00170) motivates delayed online recalibration, but long-run coverage is not instantaneous false-alert control. Only abstracts and the publicly served article content were reviewed; no uninspected result is claimed. Retrieval manifests preserve exact responses.

Do not add graph or foundation models without a defensible river graph, checkpoint contract and measurable remaining gap. Complete all declared target, memory, trajectory and alert-policy ablations before drawing conclusions, including when one fails. Stop after final evaluation or if source validity fails. New iterations must reserve genuinely new outcomes rather than tuning on these final stations. Workers run sequentially, at most two threads and below 6 GB RSS; source cache expansion is bounded to 300 MB. Preserve the previous protocol, code, dataset, evaluation, served models and application database.
