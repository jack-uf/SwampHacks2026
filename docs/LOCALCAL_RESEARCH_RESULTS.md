# Local calibration of emerging high-water warnings

Station-local calibration does not improve the frozen hybrid on the new catchments. The selected partially pooled correction reduces false warning decisions from 160 to 113, but detects 54 of 139 events instead of 66. It loses ranking performance and supplies no statistically demonstrated Brier improvement. The experiment rejects this calibration method for integration. Production models and mandatory safety rules remain unchanged.

The contribution is a new, physically screened evaluation cohort and a controlled test of the previous study's proposed calibration remedy. This is not a new validated forecasting model. The previous analog study's AP .214 and this study's AP .321 describe different populations; their difference is not an improvement estimate.

## Frozen design and evidence

The [protocol](LOCALCAL_RESEARCH_PROTOCOL.md), implementation and metadata-only cohort were committed in `9d032e5` before any new discharge request. After collecting 2021 calibration records, the selected method, station intercepts and alert thresholds were committed in `0e5d124`. Only then did collection request the new gauges' 2023-2024 outcomes. No candidate or threshold was changed after that evaluation. The report audit added afterward verifies results without changing the frozen experiment.

The study follows the separation of calibration from discrimination discussed by Kull, Silva Filho and Flach, AISTATS 2017, PMLR 54, pages 623-631. The publisher abstract and request manifest are retained under `experiments/sprint-20261017-localcal/research/`. That paper does not establish hydrological performance, and this study does not implement its beta calibration algorithm.

For probability p from the frozen analog-prior plus LightGBM correction, the new map is `sigmoid(logit(clip(p, 1e-6, 1-1e-6)) + b_station)`. Each station's intercept minimizes summed 2021 binary log loss plus `lambda*b_station^2/2`, with lambda .1, 2 or 10. The normal prior shrinks toward the global probability map; its variance is fixed rather than learned. No slopes, neural layers, new predictors or production code were fitted. Clipping can collapse extremely small probabilities into ties, but the correction cannot invert a station's ranking.

Selection uses the previous study's already inspected 2022 temporal and geographic development cohorts. It selects lambda 10 under the frozen recall and regional-burden rule. Mean development event recall is 40.4%, versus 38.0% for unchanged global calibration. The worst development region still has 3.889 false decisions per 100, failing the three-per-100 requirement. Fresh results are therefore a test of an explicitly selected candidate, not a choice among test winners.

Machine-readable evidence lives in `experiments/sprint-20261017-localcal/`: `protocol.json`, `cohort.json`, `stations.json`, calibration and test provenance, `selection.json`, both family evaluation files, `evaluation.json` and audit records. These hold source URLs, hashes, exclusions, package versions, complete station results, policy tradeoffs and calibration bins.

## Data and geographic independence

The metadata screen selects 24 previously unused CAMELS gauges in 24 HUC4 groups. It checks exact NLDI identifiers, point distance within one kilometre and zero positive-area intersection of simplified upstream polygons against all 250 previously reserved gauges and each other. The screen uses 75 candidate HTTP attempts plus four requests for missing occupied-basin metadata. It never examines discharge or labels to choose replacements.

One gauge, USGS 08202700, fails the positive pre-2016 reference-flow requirement during calibration and is excluded before the final history request. The final population has 23 gauges in 23 HUC4 groups across 13 HUC2 regions, with 10,394 eligible decisions, 390 positive windows and 139 distinct onset events. There are no outcome-based replacements.

These are new catchments within broadly represented climate regions. Earlier studies have already consumed almost all unused HUC4 geography, so this study does not claim new regional climates. Simplified watershed polygons certify geometric separation of those retained boundaries, not independence of storms, groundwater or unsimplified watersheds.

Sources are approved USGS daily mean discharge and the existing basin-aligned ECMWF HRES archive in Caravan MultiMet, with the unchanged hydrological, snow and historical-transition features. The archive covers January 2016 through September 2024. For the new gauges, calibration requests end on December 31, 2021; final requests end on September 30, 2024. Existing development sources already contain later observations, but the calibration collector explicitly truncates them to 2021 before feature or target construction. No new satellite or transformer experiment was run: this study isolates probability adaptation.

The target is a new daily mean flow exceedance of the station's pre-2016 p95 after seven quiet days, on local D+1 through D+3. This is emerging high flow, not verified flooding, flood-stage exceedance or infrastructure damage. The latest available measured flow is assumed to be D-3; HRES inputs use D-1 initialization. Final revised observations and assumed publication lags do not establish an as-issued forecast replay. Daily mean labels cannot identify the hour of a flood peak or resolve short flash floods.

Across the 23 final gauges, the earlier calibration sample has 6,033 decisions and 193 positive windows; two gauges have no positive windows. Positive windows can describe the same event. This is a sparse one-year estimate of local probability bias, not a stable estimate of local flood climatology.

## Candidate comparisons

AP means average precision. False warnings count decisions, not merged warning episodes. Every displayed alert cutoff is frozen using development data to target one false warning per 100 decisions. Actual final burdens differ; these are not exactly matched final-budget comparisons. No test-label threshold adjustment is used.

| Candidate | AP | Brier | Log loss | Events detected /139 | Precision | False decisions | False /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged global hybrid | .3213 | .02976 | .11176 | 66 | 41.8% | 160 | 1.539 |
| Local intercept, lambda .1 | .2780 | .03091 | .11904 | 39 | 45.7% | 82 | .789 |
| Partially pooled, lambda 2 | .2960 | .03029 | .11321 | 44 | 44.0% | 94 | .904 |
| **Selected partially pooled, lambda 10** | **.3138** | **.02979** | **.11088** | **54** | **43.8%** | **113** | **1.087** |
| Frozen calibrated HRES rainfall | .2385 | .03190 | .12304 | 47 | 34.0% | 136 | 1.308 |
| Frozen logistic regression | .3039 | .03087 | .12011 | 72 | 33.9% | 240 | 2.309 |
| Frozen direct LightGBM | .3235 | .02985 | .11088 | 58 | 41.0% | 141 | 1.357 |
| Frozen pooled neural onset model | .3266 | .02938 | .10707 | 66 | 42.5% | 161 | 1.549 |

The neural baseline has the best point AP and Brier in this population. It was not a new selected candidate, and this study defines no superiority test for promoting it. Direct LightGBM has fewer detections and fewer false decisions than unchanged global hybrid. Neither tradeoff supplies an operational replacement decision by itself.

Strict persistence issues no warnings because eligibility requires the latest available flow to be below threshold. This weak onset baseline cannot establish superiority over an operational river forecast. No historically available authoritative river-flow forecast was verified for these gauge/time pairs. HRES rainfall is a weather-only comparator, not an official hydrological forecast.

## Uncertainty, lead time and reliability

The selected method's event recall falls from 47.48% to 38.85%, a loss of 8.63 percentage points. It adds three detected events and loses 15. Paired 500-draw bootstrap 95% intervals for selected minus global are:

| Difference | HUC4 blocks | Calendar-month blocks |
| --- | --- | --- |
| Event recall | [-.1661, -.0075] | [-.1405, -.0501] |
| AP | [-.01865, .00509] | [-.02794, .01760] |
| Brier | [-.000415, .000424] | [-.000559, .000704] |

Both recall intervals exclude zero in the wrong direction. AP and Brier differences are inconclusive. These are separate cluster analyses, not simultaneous confidence guarantees. Month resampling retains the preceding study's maximum event-weight convention for events spanning month boundaries. Neither analysis captures all dependence from shared weather systems.

The selected method warns 44 events at least 48 hours ahead, versus 55 globally. Median earliest nominal lead among its detected events is 53 hours versus 66 globally; those subsets differ. Among 51 events both detect, median paired lead change is zero, with two earlier warnings, 41 unchanged and eight later. A zero paired median therefore does not mean that warning trajectories are unchanged. The timings refer to midnight on a daily-label date, not measured threshold-crossing timestamps.

The selected method improves aggregate log loss slightly but not Brier. Calibration bins remain imperfect: its .3-.4 bin averages .346 predicted probability against .406 observed frequency in 101 decisions. Its .5-.6 bin averages .550 against .382 in 34 decisions. Local recalibration neither establishes probability reliability nor provides a coverage guarantee.

| HUC2 | Events | Global recall | Selected recall | Global false /100 | Selected false /100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 15 | 53.3% | 33.3% | 2.167 | 1.238 |
| 02 | 18 | 50.0% | 44.4% | 2.255 | 1.550 |
| 03 | 30 | 53.3% | 56.7% | 2.448 | 2.517 |
| 04 | 4 | 50.0% | 0.0% | 1.299 | .325 |
| 05 | 19 | 73.7% | 36.8% | 1.756 | .492 |
| 06 | 6 | 50.0% | 16.7% | 1.674 | 1.255 |
| 07 | 6 | 16.7% | 33.3% | 2.606 | 3.257 |
| 10 | 13 | 7.7% | 7.7% | .498 | .355 |
| 11 | 9 | 33.3% | 33.3% | 1.559 | 1.559 |
| 12 | 7 | 57.1% | 57.1% | 2.062 | .825 |
| 16 | 3 | 0.0% | 0.0% | .215 | .215 |
| 17 | 7 | 71.4% | 71.4% | .712 | .445 |
| 18 | 2 | 0.0% | 50.0% | .000 | .000 |

Small regional counts limit inference. Local calibration shifts warnings away from some productive regions while increasing the burden above the prespecified limit in HUC2 07. In 2024, its pooled false burden is 1.622 per 100, compared with 2.125 globally. Aggregate burden of 1.087 hides that temporal failure. Its conservative regional policy detects only 18 of 139 events at .212 false decisions per 100. Tightening alerts is not a demonstrated detection advance.

The frozen success gate passes population size, pooled burden and nonnegative median paired lead. It fails recall gain, recall uncertainty, Brier improvement and regional burden. No new model qualifies for integration.

## Resources, verification and reproduction

Calibration collection takes 17.83 seconds at 396 MB peak RSS. Selection takes 1.98 seconds at 645 MB. Final collection takes 8.79 seconds at 336 MB. Classic final evaluation takes 1.77 seconds at 287 MB; the separate neural worker takes 1.13 seconds at 407 MB. Global hybrid inference for the materialized frame takes .019 seconds; local correction plus shared hybrid inference takes .073 seconds. These exclude forecast extraction and historical analog construction, so they are not end-to-end live latency measurements. Workers use CPU with two threads and retain framework isolation. No new dependency or checkpoint was downloaded.

The source audit verifies 202 unique weather/discharge files, including 12,824,802 bytes of new observation responses. Metadata screening is separately retained. Both family audits pass source hashes, all 250 prior catchment boundaries, calibration-history consistency, independently reconstructed final labels, station ranking and exact frozen prediction replay. The summary audit also checks row alignment, group metrics, uncertainty, lead comparisons and the final decision. The complete repository suite passes: 218 tests in 26.49 seconds, with three existing SHAP color deprecation warnings. This includes 25 new summary-verifier tests covering alignment failures, metric and decision tampering, clipping ties and event-lead accounting, plus five frozen local-calibration tests.

Run from the repository root, using the existing virtual environment and `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2`. Collection needs network access. The executed sequence was:

```sh
.venv/bin/python -m scripts.localcal_cohort --self-test
.venv/bin/python -m scripts.localcal_cohort
# Commit metadata and protocol before collecting observations.
.venv/bin/python -m scripts.localcal_research calibration
.venv/bin/python -m scripts.localcal_worker select
# Commit selection before opening final histories.
.venv/bin/python -m scripts.localcal_research test
.venv/bin/python -m scripts.localcal_worker classic
.venv/bin/python -m scripts.localcal_worker neural
.venv/bin/python -m scripts.localcal_worker report
.venv/bin/python -m scripts.audit_localcal classic
.venv/bin/python -m scripts.audit_localcal neural
.venv/bin/python -m scripts.audit_localcal_summary
.venv/bin/python -m pytest -q
```

Collectors and modeling commands intentionally refuse to overwrite frozen results. To verify this retained run, execute the three audit commands. A full reproduction requires the exact retained source bytes, prior artifacts and an isolated research workspace with a newly declared output directory. Source responses, compressed datasets and per-decision predictions are ignored local files. Committed manifests alone do not make a clean checkout self-contained; fresh API responses may contain revisions. Do not delete retained evidence or silently substitute fresh responses.

The protocol's absolute within-station ordering language is qualified by the explicitly specified clipping operation: ties can form at clipping boundaries. The actual prediction audit passes even the stricter identical-sort check; the summary verifier allows ties but rejects inversions. This qualification does not alter probabilities, alerts, selection or success criteria.

## Decision and next research direction

Keep the globally calibrated research model and all served models as they are. One-year station bias correction is not a reliable remedy for geographic warning transfer. The result is consistent with a noisy or time-varying bias estimate, but does not causally establish which explanation dominates. It does not reject all local adaptation or all calibration methods.

The highest-value next experiment is a multi-year, forward-blocked probability-transport model that separates local hydrological response from changing event frequency, with rolling-origin selection and a completely new forecast period. Compare it against an as-issued authoritative hydrological forecast and verified subdaily gauge data. The present cohort and period are now inspected development evidence and cannot be reused as a fresh test. More model capacity alone does not address this experiment's demonstrated detection tradeoff.

A credible path to PlanIT's 0-10 index still needs verified as-issued inputs, event-level false-warning episodes, reliable site-to-watershed mapping, and an independently calibrated relation between discharge onset and worksite consequence. Until those are validated, this probability must not be represented as flood danger or used to weaken mandatory safety rules. No application or browser behavior changed in this study.
