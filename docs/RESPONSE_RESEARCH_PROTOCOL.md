# Catchment-conditioned emerging high-flow protocol

This experiment tests whether historical catchment response signatures and a conditional neural mixture improve early warnings in unfamiliar watersheds. It continues the onset research without reusing its inspected final outcomes as independent evidence. All candidates remain experimental. Commit this protocol, the metadata-only station registry and polygon audit before requesting new flow records or fitting models.

## Rationale and scope

The previous [onset study](ONSET_RESEARCH_RESULTS.md) found useful rain information but weak transfer of false-alert rates. The hypothesis here is that forecast rain has different consequences in a flashy, rain-driven catchment and a buffered or snow-dominated catchment. Pre-evaluation flow signatures can describe some of that response without a station identifier. A conditional mixture can specialize its rain-to-onset relationship while sharing evidence across catchments. An auxiliary flow-trajectory loss tests whether measured river evolution improves that representation.

The design draws on Kratzert et al.'s [entity-aware regional model](https://hess.copernicus.org/articles/23/5089/2019/), Feng et al.'s [differentiable regional hydrology study](https://hess.copernicus.org/articles/27/2357/2023/), and Addor et al.'s [CAMELS attributes](https://hess.copernicus.org/articles/21/5293/2017/). Public abstracts and relevant served article sections were read; exact retrieval manifests are retained. These papers motivate conditioning on catchment behavior, not a claim that this implementation reproduces their models. The neural mixture is neither an EA-LSTM nor a mass-conserving hydrological simulator. Its combination and evaluation are an application research contribution, not a claim of architectural priority.

## Fresh geography and fixed population

Reuse the previous 74 usable development gauges and their exact collected 2017-2022 rows. Retain their original fitting, calibration and selection assignments. Do not add previously inspected final labels. New gauges must have a HUC4 absent from every station in both earlier hydrology and onset registries, valid USGS metadata and a documented timezone. Choose the lowest two station identifiers per eligible HUC4 before requesting flow. Order groups by SHA256 of `response-20261017-` plus HUC4.

Retrieve simplified upstream polygons from the USGS NLDI flowline basin endpoint. Resolve each station through the indexed `nwissite/USGS-{station}` record, require the exact station identifier, one flowline within one kilometer, then retrieve that COMID upstream basin. One prior station, 04127918, is absent from the gauge index. For this station only, use its verified hydrolocation at longitude -84.59783, latitude 46.18585, requiring COMID 12206226 and the same distance and HUC checks. This exception is explicit source metadata, not an automatic fallback for failed requests. Persist request URLs, retrieval times and hashes. Record disagreements between station-metadata HUC4 and indexed reach HUC4; the exact gauge identity and polygon intersection audit determine independence. A verified station 01118300 has this source disagreement before any outcome access. Coordinate-only resolution for all gauges was rejected before outcome access because the service returned coordinate-dependent HTTP 502 errors. Using the entire reach rather than a partial gauge reach makes the overlap screen conservative. Reject a new candidate whose polygon has any positive-area intersection with any prior station's polygon or any earlier accepted new HUC4 group. Retain up to two gauges in the same group, which may overlap each other. The first eight accepted HUC4 groups are geographic selection only; the next twenty are final test only. Require all fetched geometries to be valid. A new candidate failing the location check is excluded before selection, without substituting a third gauge from its group. An unverifiable prior gauge stops preparation because independence could not be established. No replacements after flow inspection. Preserve every exclusion and geometry hash. Simplified present-day basin boundaries support this conservative geographic audit but are not a survey of historical diversions.

New selection uses 2022 only. Final test uses January 8, 2023 through September 25, 2024. Final outcomes, prevalence and event counts remain uninspected until model artifacts, code hashes, alert thresholds and selection are committed. Require at least 50 unique final events and ten HUC4 groups for an improvement claim. Retain every eligible row and report all exclusions without replacing gauges.

## Target and availability

Use the exact approved USGS daily mean flow parser and previous onset definition. High flow means strictly exceeding the station's 2000-2015 95th percentile, with at least 3,000 accepted reference days. An onset requires seven consecutive accepted quiet days before the exceedance. Forecast any certified onset on local D+1 through D+3. Issue at 00:00 UTC and require seven known quiet days ending D-3. Unknown observations cannot be negative labels. Report lead to local onset-day start, not instantaneous crossing.

Reuse the archived ECMWF HRES forecasts, D-1 run leads 2-5, antecedent forecast histories through D-2, 270-day memory warmup, D-3/D-4/D-7/D-14 discharge, seasonal/static attributes and causal degree-day snow memories. D-1 forecast and D-3 discharge availability remain explicit assumptions; archived approved data can contain later revisions. No as-issued validity claim is justified. No observed future rain, present-day flood thresholds, satellite images or unverified upstream joins are added. These are measured high-flow warnings, not verified floods or inundation predictions.

## Historical response and inputs

Compute ten station descriptors from accepted observations strictly before 2016: mean, tenth percentile, median and 99th percentile divided by the reference; coefficient of variation; lag-one correlation over exactly consecutive accepted days; sum of absolute consecutive differences divided by total flow; flow-weighted seasonal sine and cosine; and log1p of the reference converted to millimeters per day using drainage area. Require at least 2,000 consecutive pairs and finite variable flow. Never compute a signature from calibration, selection or final flows.

The feature ablation adds these descriptors and a dimensionless storm-load feature: sum of forecast precipitation for leads 3-5 divided by reference discharge in millimeters per day. It is a relative loading proxy, not an infiltration or rainfall-runoff measurement. Neural dynamic inputs contain the previous weather/hydrology/snow inputs except latitude, longitude, elevation and drainage area. The context gate uses the ten signatures, log drainage area and elevation. Station IDs and HUCs are never model features. Standardization uses fitting rows only, with zero-variance scales fixed to one.

## Fixed candidates and training

Run these eight candidates once, without hyperparameter search:

1. Calibrated forecast rainfall ranking, sum of rain leads 3-5.
2. Standardized logistic regression on the prior combined/snow inputs, C=1, maximum 2,000 iterations.
3. Direct onset LightGBM on those same inputs.
4. Direct onset LightGBM with response descriptors and storm load added.
5. A pooled neural classifier using dynamic inputs, without context or trajectory loss.
6. A four-expert neural mixture with a historical-response context gate, without trajectory loss.
7. The same mixture with auxiliary residual-flow trajectory learning.
8. The same auxiliary mixture with the context gate fixed to uniform weights, testing whether specialization adds value beyond capacity.

LightGBM uses 200 trees, learning rate .04, seven leaves, depth three, two threads and seed 42. Neural experts have two 32-unit SiLU layers, a four-category onset-day output and three residual-flow outputs. The pooled network has one expert. The gate is a linear context-to-four-expert softmax. Mixture category probabilities are probability mixtures, never mixed logits. Aggregate onset probability is one minus the no-onset category. The trajectory output is the weighted residual prediction plus D-3 normalized log flow.

Fit neural models on 2017-2020 fitting rows for 60 epochs with AdamW, learning rate .002, weight decay .001, batches of 2,048, gradient norm limit five and deterministic seed 42. Shuffle using the seeded generator. Use ordinary four-category negative log likelihood, with no future-label-derived predictor or sampling weight. Auxiliary loss is .5 times Huber loss on the three normalized log-flow residuals, delta .2. No early stopping or checkpoint choice. Calibrate every candidate's aggregate log odds with a one-dimensional logistic model on 2021 development only. No final retraining.

Evaluate selection on 2022 temporal development and the new geographic-selection groups. Do not use the prior geographic-selection groups for choosing this iteration. Select by mean AP of the two cohorts, then lower mean Brier, then name. Freeze the strongest baseline among rainfall, logistic and original-input LightGBM, plus a separate neural winner among the four neural candidates. Record all ablations regardless of selection.

## Alert decisions, uncertainty and claim

Freeze pooled 2022 thresholds targeting .5, one and two false alerts per 100 eligible decisions using the existing tie-preserving rule. Also freeze a conservative maximum-HUC2 threshold for the 1% budget. These are empirical selection budgets, not distribution-free guarantees. No delayed test-label adaptation is used in this experiment.

Report AP, Brier, log loss, ten-bin calibration, precision, decision recall, unique-event recall, false alerts per 100 decisions, counts warned at least 48 hours ahead, and per-station/HUC2/year results. Compare paired lead differences only among events detected by both methods. Report 1% test-label oracle thresholds solely as matched-budget diagnostics.

Use 500 seeded paired bootstrap draws by HUC4 and separately by calendar month. Compare the selected overall candidate and the frozen neural winner with the strongest baseline, preserving paired rows. Report AP intervals and event-recall intervals. A useful advance requires the selected candidate to improve AP with positive lower bounds under both cluster schemes, improve unique-event recall by at least .05 at the frozen 1% policy, keep false alerts at or below 1.5 per 100 decisions, and have no test HUC2 above three per 100. It must also be a response-conditioned candidate, rather than an unchanged baseline. A neural architecture claim applies these same criteria to the frozen neural winner. Passing does not authorize production integration because historical availability and real flood validity remain unresolved.

## Resource and stop rules

Run neural and LightGBM workers in separate processes to preserve OpenMP isolation. Use CPU with two threads; the current environment reports MPS unavailable. Record cached-data fit/inference time, RSS and package versions. Stop a worker above 6 GB RSS or 15 minutes, and limit new sources including basin polygons to 300 MB. Complete all ablations unless resource or source validity prevents them. Do not add another model after final outcomes are inspected. Preserve served models, source data, prior experiments and the application database. Finish with source-level label and artifact audits, relevant tests, documentation and a commit.
