# Onset supervision and regional warning budgets

Neither frozen hypothesis passed on the fresh watershed test. Predicting the onset day did not improve on matched binary supervision. A regional warning budget enforced its accounting limit, but reduced event detection from 42 to 17 of 118 events. Keep both changes experimental and retain the served models and deterministic safety rules.

This study follows the [catchment-response experiment](RESPONSE_RESEARCH_RESULTS.md). Its contribution is a controlled supervision comparison and a causal, delayed-feedback warning replay on another independent population. It does not establish a new flood forecasting method or justify production integration.

## Protocol and independent data

The [protocol](WARNING_RESEARCH_PROTOCOL.md), station registry and geography screen were committed in `c25a2a2` before new discharge collection. Implementation, fitted artifact hashes, thresholds and policy selection were committed in `e4f53fd` before final outcomes were opened. Both claims and stopping criteria remained unchanged. All reported final results live in [evaluation.json](../experiments/sprint-20261017-warning/evaluation.json).

The metadata screen excluded all 183 gauges and 135 HUC4 groups reserved by the hydrology, onset and response studies. It used their cached NLDI basin polygons and required zero positive-area overlap with previous basins and previously accepted new groups. Overlap within a new HUC4 group was allowed. The reserved sample contained 32 gauges in 18 HUC4 groups. Gauges `09386900` and `09035800` failed the historical-flow quality requirements and were excluded without replacement.

The usable test contains 30 gauges in 17 HUC4 groups and nine HUC2 regions, with 12,395 eligible decisions, 344 positive decision windows and 118 distinct onset events. Dates run from January 8, 2023 through September 25, 2024. This is a forward-time test beyond development through 2022 and a geographic test outside every previous registry. It is a metadata-screened convenience sample, not a representative national sample. Simplified polygon separation does not prove independence from shared storms or climatic dependence.

Labels use accepted, approved USGS daily mean discharge. An event starts when flow exceeds the gauge's 2000 through 2015 95th percentile after seven consecutive quiet days. Issue time is 00:00 UTC; the target is onset on local D+1, D+2 or D+3. Known quiet conditions end at D-3. Lead time runs to the start of the local onset day, not the hour a river first rises or reaches a flood stage. A high-flow percentile is not verified flooding, road inundation or a crew-safety outcome.

Features reuse archived ECMWF HRES forecasts, D-3 discharge, causal rain and snow memories and the response pipeline's completeness filters. Reference-period flow descriptors are available before the test period. No satellite, future observed rainfall or upstream-neighbor feature was added. D-1 forecast availability and D-3 discharge availability remain assumptions rather than verified publication timestamps. Approved discharge can contain revisions unavailable at the historical issue time.

[Provenance](../experiments/sprint-20261017-warning/provenance.json) records source paths, hashes, coverage and exclusions. New source downloads total 9,169,071 bytes. Raw data and fitted artifacts remain preserved locally and ignored by Git; manifests and evaluation records are committed.

## Matched supervision experiment

The new binary and onset-day candidates each average three networks initialized with seeds 42, 314 and 2718. Paired seeds use identical architecture, initialization, training rows, optimizer, shuffled batches and epochs. Each member has 1,959 parameters, two 32-unit SiLU layers and the same output capacity. Binary supervision optimizes whether any onset occurs. Day supervision optimizes four categories: no onset, D+1, D+2 and D+3. Both receive one calibration model fitted on 2021 ensemble probabilities.

The prior frozen rainfall ranking, logistic regression, direct LightGBM and single-seed pooled neural model provide baselines. Development selection reuses the response study's inspected 2022 chronological and geographic cohorts. Mean selection AP favored the day ensemble over binary, 0.2492 versus 0.2438; direct LightGBM remained higher at 0.2532. The selected day ensemble and its `bucket_5` warning policy were frozen before testing. No seed or test threshold was selected retrospectively.

The table uses each model's development threshold targeting one false warning per 100 eligible decisions. That target is not a guarantee on unfamiliar watersheds. AP is average precision; Brier is mean squared probability error, where lower is better. Precision and decision recall count windows; event detection counts distinct station/onset-date pairs.

| Model | AP | Brier | Precision | Decision recall | Events detected / 118 | False warnings | False / 100 decisions | Events warned at least 48h ahead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forecast rainfall ranking | 0.1770 | 0.02497 | 0.297 | 0.119 | 27 | 97 | 0.783 | 24 |
| Logistic regression | 0.2431 | 0.02389 | 0.323 | 0.186 | 37 | 134 | 1.081 | 31 |
| Direct LightGBM | 0.2581 | 0.02376 | 0.338 | 0.201 | 41 | 135 | 1.089 | 38 |
| Prior pooled neural, seed 42 | 0.2637 | 0.02304 | 0.409 | 0.241 | 46 | 120 | 0.968 | 43 |
| Binary neural ensemble | 0.2789 | 0.02289 | 0.381 | 0.233 | 44 | 130 | 1.049 | 37 |
| Day neural ensemble, selected | 0.2763 | 0.02294 | 0.369 | 0.221 | 42 | 130 | 1.049 | 38 |

Day minus binary AP was -0.00257. Its paired 95% bootstrap interval was [-0.01030, 0.00515] across HUC4 draws and [-0.01019, 0.00662] across calendar-month draws. Day supervision also detected two fewer events. The frozen supervision claim fails.

Both ensembles had higher point AP than LightGBM, but neither improvement excludes zero under either uncertainty scheme. Binary minus LightGBM AP was 0.02084, with HUC4 interval [-0.00099, 0.04675] and month interval [-0.00484, 0.04702]. Day minus LightGBM was 0.01827, with intervals [-0.00332, 0.04246] and [-0.00460, 0.03926]. Each scheme uses 500 paired draws. These intervals do not establish superiority under geographic transfer.

Calibration also remains imperfect. For the day model's 117 windows with probabilities between 0.3 and 0.4, mean prediction was 0.345 and observed frequency was 0.256. The nine windows between 0.6 and 0.7 averaged 0.655 predicted versus 0.222 observed. The latter bin is too small for a strong calibration conclusion. Complete reliability bins, log loss, station, region and year results are retained in the evaluation record.

The existing pooled model detected more events at its frozen threshold than either ensemble. This is evidence against assuming that more seeds or richer supervision improves useful warnings. Its performance is descriptive here; choosing it as a new production winner from this test would consume the test for selection. Test-label oracle thresholds are included only as non-deployable matched-budget diagnostics.

## Warning-budget experiment

The experimental policy starts each HUC2 with zero credits. Each eligible decision earns one credit, a warning costs 100, and the balance is capped at 300. Correct warnings receive one refund after ten calendar days. False or unavailable outcomes receive no refund. Each daily batch ranks proposals by probability, with station identifier breaking ties; unissued proposals expire. Final budgets reset before 2023 and carry forward through 2024.

This is adapted from the token-bucket accounting idea described in RFC 2697, whose fetched source and request hash are retained under the experiment's `research/` directory. It is not an implementation of that RFC's two-bucket meter. The separately inspected conformal-risk-control paper, arXiv:2208.02814, did not supply this algorithm. No conformal coverage or exchangeability guarantee is claimed.

For every regional daily-batch prefix, issued false warnings cannot exceed eligible decisions divided by 100: spending minus correct-warning refunds cannot exceed earned credits. This retrospective accounting property depends on correct outcome feedback. It is not probability calibration, a false-discovery guarantee, or a safety guarantee. Production stop-work rules never use it.

| Selected day-model policy | Events detected / 118 | False warnings | False / 100 decisions | Precision | Events warned at least 48h ahead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static one-percent development threshold | 42 | 130 | 1.049 | 0.369 | 38 |
| Static five-percent development threshold | 85 | 577 | 4.655 | 0.232 | 75 |
| Maximum-region one-percent threshold | 18 | 21 | 0.169 | 0.523 | 14 |
| Budget with one-percent proposals | 18 | 37 | 0.299 | 0.373 | 17 |
| Budget with five-percent proposals, selected | 17 | 67 | 0.541 | 0.230 | 14 |

The selected budget cut false warnings by 63, but event recall fell from 35.6% to 14.4%. The paired event-recall difference was -21.2 percentage points: HUC4 interval [-33.9, -7.9] and month interval [-35.5, -6.3]. It found 14 events missed by the static threshold, lost 39 static detections and shared only three. The 24-hour median lead gain among those three shared events cannot offset the lost detections. Both policies had a 76-hour median lead among their own detected events, illustrating why conditional lead time alone is misleading.

Starting without credits imposed substantial warmup. Each region needed 32 through 143 elapsed days from its first eligible test decision to accumulate its first 100 decision credits. This diagnostic follows directly from regional cumulative row counts in `predictions/day.test.csv.gz`; it is not a new selection criterion. Capped credits, early spending and delayed refunds also limit later warnings. Allowing more low-threshold proposals can consume credit before a stronger future signal arrives. The experiment rejects this particular budget, not every possible adaptive warning policy.

## Geographic transfer

Regional results for the selected day ensemble show why pooled false-alert rates are insufficient. The budget stayed below one false warning per 100 decisions in every HUC2, while sacrificing most regional detections.

| HUC2 | Decisions | AP | Static false / 100 | Static event recall | Selected budget false / 100 | Budget event recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 05 | 2,690 | 0.325 | 1.524 | 0.395 | 0.558 | 0.158 |
| 06 | 395 | 0.539 | 4.304 | 0.625 | 0.506 | 0.000 |
| 07 | 830 | 0.094 | 3.012 | 0.286 | 0.843 | 0.000 |
| 10 | 2,118 | 0.284 | 0.755 | 0.148 | 0.519 | 0.185 |
| 11 | 1,522 | 0.173 | 0.329 | 0.333 | 0.723 | 0.083 |
| 12 | 698 | 0.468 | 0.430 | 0.250 | 0.573 | 0.250 |
| 14 | 2,077 | 0.324 | 0.433 | 0.286 | 0.337 | 0.143 |
| 15 | 1,045 | 0.600 | 0.191 | 0.600 | 0.287 | 0.400 |
| 17 | 1,020 | 0.424 | 1.176 | 0.600 | 0.686 | 0.100 |

Small regional event counts make these estimates unstable. The budget can increase false warnings in a region that was already below its limit, because its selected proposal threshold is looser. Region 10 gains detections, but this isolated result does not justify selecting a regional exception after inspection.

## Verification, resources and reproduction

The independent audit reconstructs all 12,395 labels from source discharge, verifies reference-period descriptors and lag features, checks basin hashes and geographic exclusions, and reproduces all six candidate probability arrays and warning policies. It checks every regional budget prefix and verifies that changing labels too recent to have matured leaves warnings unchanged. The day ensemble's seed-42 member exactly reproduces the previous pooled network's weights. Both audit records report `passed`.

All new fitting used CPU with two threads, with Torch and LightGBM in separate processes. The neural selection worker took 19.47 seconds including fits, calibration and development policy replays; its peak RSS was 562 MB. Classic selection took 11.64 seconds and 452 MB. Final evaluation workers took 9.84 and 10.54 seconds including policy replays, with peaks of 248 and 423 MB. Predicting all 12,395 rows took about 0.022 seconds for LightGBM, 0.010 seconds for the binary ensemble and 0.012 seconds for the day ensemble. These are single-run timings, not repeated speed benchmarks. New fitted artifacts total 50,261 bytes. No paid model API, GPU or new dependency was needed.

The original run order was metadata preparation, protocol commit, collection, separate family selection, selection freeze and commit, separate final evaluation, report and audits. Entry points are:

```sh
.venv/bin/python -m scripts.warning_research prepare
.venv/bin/python -m scripts.warning_research collect
.venv/bin/python -m scripts.warning_worker classic
.venv/bin/python -m scripts.warning_worker neural
.venv/bin/python -m scripts.warning_worker freeze
# Commit selection before opening final outcomes.
.venv/bin/python -m scripts.warning_worker evaluate-classic
.venv/bin/python -m scripts.warning_worker evaluate-neural
.venv/bin/python -m scripts.warning_worker report
.venv/bin/python -m scripts.audit_warning classic
.venv/bin/python -m scripts.audit_warning neural
.venv/bin/python -m pytest -q
```

These commands record the run order, not an overwrite recipe for the completed experiment. Preparation, collection and selection fail when their protected outputs already exist. Audits can be rerun against the preserved local files. A clean checkout also needs the ignored raw data, training tables, basin files and prior fitted artifacts documented by the response study and hashed in these manifests. Preserve or regenerate those under their original protocols before attempting reconstruction; do not delete frozen evidence to make a rerun pass. Dependencies are pinned in the response experiment's `requirements.txt`; package versions, artifact hashes and source hashes accompany selection. Exact byte-level reproduction can depend on platform and source revisions.

The complete regression suite passed: 182 tests, with three existing SHAP deprecation warnings. The four new tests exercise delayed feedback, regional accounting, priority and permutation behavior, validation, HUC identifiers and the matched loss functions. No frontend, production model, persistent database or safety-rule change was made, so no browser workflow was rerun for this research-only change.

## Decision and next research step

Stop this frozen comparison without adding another model or policy. Both final populations and the earlier 2026 holdouts are now inspected evidence and must not be reused as fresh tests. Neither new ensemble nor the credit policy should contribute to the served hazard index.

The strongest next hypothesis is to improve the signal before constraining warnings: predict downstream onset from causally lagged upstream hydrograph innovations and rainfall travel times, then test whether those features reduce false positives at a fixed event-recall target. Start with a small connected river network whose gauge timestamps and availability can be verified. Freeze travel-time estimation on training periods and hold out entire connected networks plus a later untouched period. Compare with the current pooled model, LightGBM and authoritative forecasts where as-issued archives exist. This is a proposed study, not a measured benefit. Historical availability and real flood-stage validity remain prerequisites for an operational claim.
