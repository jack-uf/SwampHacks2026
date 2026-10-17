# Onset supervision and regional warning budgets

This experiment tests two unresolved questions from the [response study](RESPONSE_RESEARCH_RESULTS.md). Does predicting the onset day improve geographic transfer when architecture and random seeds are held constant? Can a regional warning budget reduce false alerts without losing the early detections that make the model useful? Commit this protocol and metadata before requesting new flow records. Keep every candidate outside production.

## Independent population

The metadata screen reserves 32 previously unused gauges in 18 HUC4 groups. Start with the mapped candidates already cached by the response study, excluding every station and HUC4 in the hydrology, onset and response registries. Verify the response study's location and basin hashes. Order remaining groups by SHA256 of `warning-20261017-` plus HUC4. Accept a station only when its simplified upstream polygon has zero positive-area intersection with every previous gauge and every earlier accepted group. Overlap within a group is permitted. The registry and geography record are authoritative. Do not replace gauges after inspecting flow quality or outcomes.

Use January 8, 2023 through September 25, 2024 as final evaluation. Reuse the response study's exact training, calibration and two 2022 selection cohorts. These development cohorts have been inspected and are not independent evidence. Final outcomes, prevalence and event counts remain uninspected until model selection, thresholds, implementation hashes and artifact hashes are committed. Require at least ten usable HUC4 groups and 50 distinct events for a positive improvement claim. Report all quality exclusions.

## Target and availability

Reuse the approved USGS daily mean discharge parser, pre-2016 reference and onset definition without changes. High flow means daily mean discharge strictly above the station's 2000-2015 95th percentile. Require 3,000 accepted reference days and sufficient consecutive history for the frozen response descriptors. A certified onset follows seven consecutive quiet days. Predict onset on local D+1 through D+3, issuing at 00:00 UTC with seven known quiet days ending D-3. Report lead to local onset-day start. These outcomes are measured high flow, not verified flooding.

Reuse archived ECMWF HRES D-1 forecasts, known D-3 discharge and causal rain/snow memories. Preserve all source hashes and completeness filters. D-1 forecast and D-3 discharge availability are assumptions, not verified as-issued archives. Final approved discharge may contain later revisions. No satellite or future observed rain features are added. Source-quality failures exclude gauges without replacements.

## Matched supervision comparison

Retain the response study's frozen rainfall ranking, logistic regression, direct LightGBM and pooled seed-42 neural artifacts as baselines. Add two three-seed neural ensembles, seeds 42, 314 and 2718. Both use the exact pooled `ResponseMixture` architecture, one expert, two 32-unit SiLU layers and the same seven outputs. Dynamic inputs, fitting rows, training-only standardization, initialization, shuffled batches, optimizer, epochs and parameter count match between each paired seed.

The day-supervised ensemble minimizes four-category negative log likelihood over no onset and onset on D+1, D+2 or D+3. The binary-supervised ensemble minimizes binary cross entropy after summing those same three onset probabilities. Neither uses residual-flow loss or context gating. This controls architecture and head capacity while changing only target supervision. Train each for 60 epochs using AdamW, learning rate .002, weight decay .001, batch size 2,048 and gradient norm limit five. Average aggregate probabilities across seeds, then fit one logistic calibrator on 2021 development log odds. Do not choose a seed or epoch using selection results.

Select the ensemble with higher mean AP over the two 2022 cohorts, breaking ties by lower mean Brier then name. Report both regardless of selection. Freeze pooled selection thresholds targeting one and five false alerts per 100 decisions, plus the maximum-HUC2 threshold targeting one. These empirical thresholds carry no guarantee under geographic shift. Retain the old pooled artifact unchanged but compute its new thresholds on the same selection cohorts.

## Regional warning budget

Each HUC2 starts with zero credits. Each eligible decision earns one credit. An issued warning costs 100 credits. Cap the balance at 300 credits. At each daily batch, first add earned credits and refunds from correct warnings issued at least ten calendar days earlier, then cap the balance. A confirmed correct warning refunds 100 credits once. Incorrect or unavailable outcomes earn no refund. The ten-day delay is an explicit historical replay assumption, leaving seven days after the D+3 target; it is not verified operational data latency.

Propose warnings above the frozen one-percent or five-percent threshold. When credits are scarce, choose the highest probabilities first, breaking ties by station identifier. Unissued proposals expire that day. Start with no warmup credit. Replay the two selection cohorts independently. For each model choose the proposal threshold with higher mean unique-event recall over those cohorts, breaking ties by policy name. Reset budgets before the final period, then carry them across the full test chronologically.

For every regional prefix, false warnings are at most eligible decisions divided by 100. Spending minus refunds cannot exceed earned credits, and refunds never exceed issued true warnings. This is an accounting bound on false warning decisions under correct retrospective labels. It is not a calibrated probability, a false-discovery guarantee, or a guarantee for real floods. It may suppress useful warnings and delay first alerts. Production safety rules never use this policy. Report the warmup and recall costs explicitly.

## Evaluation and claims

Report AP, Brier score, log loss, reliability bins, precision, decision recall, distinct-event recall, false alerts per 100 decisions, counts detected at least 48 hours ahead and per-station/HUC2/year metrics. Compare static one-percent, static five-percent, robust one-percent and both budget policies. Show test-label oracle one-percent results only as non-deployable matched-budget diagnostics. Use 500 seeded paired HUC4 bootstrap draws and separately 500 calendar-month draws for AP and event-recall differences, reusing the response study's weighting convention. Compare day versus binary and each ensemble versus direct LightGBM. Show budget versus static event-recall intervals for the selected ensemble.

A supervision claim requires day-supervised AP to exceed binary-supervised AP with positive lower bounds under both bootstrap schemes, with no loss of static one-percent event recall. A useful budget claim requires the selected model's frozen budget policy to retain or improve its static one-percent event recall, reduce false warning counts and keep every HUC2 at or below one false warning per 100 decisions. The accounting bound alone is not a research success. Report negative results and tradeoffs without changing criteria after final inspection. Neither success authorizes production integration while availability assumptions and real-flood validity remain unresolved.

## Reproduction and resource limits

Reuse the response study's pinned dependencies. Run Torch and LightGBM in separate processes. Use CPU and two threads. Stop a neural fit above 6 GB RSS or 15 minutes; use no more than 300 MB of new source data. Record training and inference time, RSS, package versions, source provenance and artifact hashes. Verify source-level targets, polygon independence, delayed feedback, regional prefix bounds and artifact predictions. Run relevant regression tests, document results and commit. Stop after this fixed comparison; add no model or policy after inspecting final outcomes.
