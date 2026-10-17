# Historical river transitions and weather-corrected onset risk

A station-local historical transition prior, corrected by shared LightGBM weather and hydrology predictors, improves average precision on fresh watersheds. It also reduces false warning decisions relative to the prespecified direct LightGBM comparator. The event-detection gain is small, probability calibration remains poor, and the frozen operational improvement criteria fail. Keep this model experimental. Production models and safety logic are unchanged.

This report describes the October 17, 2026 experiment implemented in [`analog_research.py`](../scripts/analog_research.py) and [`analog_worker.py`](../scripts/analog_worker.py). The contribution is a reproducible test of local historical adaptation with weather correction, not a new neural architecture or a validated flood-warning service.

## Evidence and frozen choices

The [protocol](ANALOG_RESEARCH_PROTOCOL.md) and metadata-only station cohort were committed in `e4178a4`. Implementation, model selection, calibration, alert thresholds and artifact hashes were committed in `d94359c` before downloading discharge for the fresh cohort. No model or decision rule changed after opening those outcomes. The final population is now inspected and must never be described as an untouched holdout for another experiment.

[`selection.json`](../experiments/sprint-20261017-analog/selection.json) owns candidate selection and artifact hashes. [`evaluation.json`](../experiments/sprint-20261017-analog/evaluation.json) owns all final scores, regional and station results, reliability bins, warning policies and uncertainty estimates. Separate family files record runtime and memory. [`test.provenance.json`](../experiments/sprint-20261017-analog/test.provenance.json) records source URLs, retrieval times, checksums, reference-library counts and exclusions. [`audit.json`](../experiments/sprint-20261017-analog/audit.json) records independent verification.

## Why this approach

The [connected-river feasibility study](UPSTREAM_RESEARCH_RESULTS.md) could not meet its frozen development coverage requirement. Training a graph model on that incomplete cohort would not test propagation reliably. This study instead asks whether each river's historical transitions provide useful adaptation without recent local labels or unverified upstream joins.

Lall and Sharma's *A Nearest Neighbor Bootstrap For Resampling Hydrologic Time Series*, DOI `10.1029/95WR02966`, motivates conditional distributions from neighboring historical states. The publisher-deposited abstract retrieved through Crossref is retained in the experiment's research cache with its request manifest. It describes resampling, including monthly streamflow, and does not establish this daily forecast's performance. The earlier response study's retained review of entity-aware and differentiable hydrology motivates catchment conditioning. These are methodological precedents; this experiment makes no claim of architectural priority.

Each gauge supplies an approved pre-2016 flow library. A query describes delayed flow level, recent changes and season, normalized by that gauge's historical discharge percentile. The nearest historical windows produce an onset prior and a distribution of subsequent normalized flows. One candidate uses those quantities as ordinary predictors. The selected candidate also starts LightGBM at the prior's log odds and learns a correction from forecast weather, flow history and snow-memory inputs. Inference explicitly adds the prior back to the tree score. A focused regression test covers that step because LightGBM does not automatically retain a training `init_score` at prediction time.

This is transfer to new watersheds with local historical records. It is not prediction at ungauged rivers. The prior describes observed flow transitions, not a physical simulation, water balance or upstream travel-time estimate.

## Data, geography and availability

The metadata-only reserve contains 31 gauges, one per HUC4, screened against all 215 prior reserved gauges and one another using retained simplified NLDI basin polygons. The final audit additionally checks separation against all 223 boundary files retained in the response study, including candidate boundaries outside those prior registries. Polygon separation does not establish independence of regional weather or climate.

Final evaluation contains 28 gauges in 28 HUC4 groups and 14 HUC2 regions. The period is January 8, 2023 through September 25, 2024. There are 12,389 eligible decisions, 294 positive windows and 102 distinct station/onset events. Positive-window prevalence is 2.37%. Three gauges have no positive windows, so their AP is undefined; they still contribute false-alert and calibration results.

No gauge was replaced after source inspection:

| Gauge | Exclusion |
| --- | --- |
| `02196000` | Multiple documented mean-discharge series; no series chosen by guesswork |
| `02479560` | No complete eligible final decisions |
| `08324000` | No complete eligible final decisions |

USGS daily mean discharge, parameter `00060` and statistic `00003`, supplies measured outcomes and historical inputs. Only exact approved-quality `A` records pass the shared parser. Reference thresholds and local transition libraries use accepted observations from 2000 through 2015. Missing calendar days remain unavailable. Complete library targets end before 2016. Reference requirements and neighbor settings are fixed in the protocol.

Caravan MultiMet v1.1 supplies archived ECMWF HRES basin precipitation and temperature forecasts, reusing existing source chunks. The decision uses D-1 forecast initialization and measurements through D-3. Existing causal rain and snow memories are unchanged. Source licensing and attribution are documented in [the hydrology source report](HYDROLOGY_RESEARCH_RESULTS.md). There is no new checkpoint, paid API, satellite input or synthetic training observation.

An event is daily mean discharge above the station's pre-2016 95th percentile after seven quiet days. An eligible issue has seven known quiet days ending at D-3; the target is onset on local D+1 through D+3. Nominal lead to onset-day start is 28-79 hours. This is emerging high flow, not verified flooding, flood-stage exceedance, inundation or danger at a construction site. Actual intraday threshold-crossing time is not observed by this target.

Final approved discharge can contain retrospective revisions. D-3 discharge availability and D-1 forecast dissemination are assumptions without as-issued historical archives. The latency convention reduces obvious future-data leakage but does not certify a live service. No target-period local labels enter fitting, calibration, analog retrieval or deployable threshold selection.

Development reuses the exact 107,689 decisions from the response study. Actual fitting uses 49,980 decisions at 57 gauges in 2017-2020. Calibration uses 13,562 decisions at 55 gauges in 2021. Temporal selection uses 13,322 decisions at 55 gauges in 2022, and geographic selection uses 3,950 decisions at 15 gauges in 2022. All selection cohorts were previously inspected development data. The direct and pooled neural artifacts remain frozen, with no retraining.

## All candidate results

AP means average precision. Smaller Brier and log loss are better. Selection AP averages the temporal and geographic selection cohorts. Each warning threshold was fixed using selection data to target one false warning per 100 eligible decisions. Actual test burdens differ, so this is not an exactly matched final-budget comparison. Precision counts decisions; event recall counts distinct station/onset events.

| Candidate | Selection AP | Fresh AP | Brier | Event recall | False alerts /100 | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated HRES rainfall | .1628 | .1307 | .02235 | 31.4% | 1.372 | 23.1% |
| Logistic regression | .1990 | .1726 | .02515 | 48.0% | 3.003 | 19.8% |
| Prespecified direct LightGBM comparator | .2532 | .1768 | .02395 | 41.2% | 2.155 | 23.7% |
| Calibrated flow proximity | .0321 | .0480 | .02300 | 10.8% | 1.526 | 8.3% |
| Local analog prior alone | .0539 | .0630 | .02264 | 2.0% | .387 | 4.0% |
| Hydrology plus analog LightGBM | .0655 | .0605 | .02298 | 14.7% | 2.381 | 8.4% |
| Weather, hydrology, snow and analog LightGBM | .2429 | .1958 | .02272 | 45.1% | 1.816 | 26.2% |
| Selected analog prior plus learned correction | .2439 | .2137 | .02186 | 43.1% | 1.711 | 27.1% |
| Frozen pooled day-of-onset neural model | .2324 | .2084 | .02219 | 45.1% | 2.050 | 25.1% |

The correction model beats the direct comparator by .03692 AP. Paired 500-draw bootstrap 95% intervals are [.01382, .05808] by HUC4 and [.01806, .05402] by calendar month. Both exclude zero. These are separate cluster analyses, not simultaneous confidence guarantees or proof of transfer to every river. The month analysis retains the existing maximum event-weight convention when an event has windows in multiple months.

At the frozen warning policy, the selected model detects 44 events versus 42 for direct LightGBM, adding eight and losing six. It produces 212 false warning decisions versus 267. Decision recall is 26.9% versus 28.2%; log loss is .08761 versus .09671. Event-recall-difference intervals are [-.04652, .08744] by HUC4 and [-.03123, .08491] by month. The detection gain is uncertain and misses the required five-percentage-point improvement.

The selected model warns 41 events at least 48 hours ahead versus 39 for direct LightGBM. Median earliest lead is 77 hours for both detected subsets. Among the 36 commonly detected events, median paired lead gain is zero. The evidence supports better ranking and fewer false decisions, not a demonstrated advance in typical warning time. The neural comparator detects more events than the selected candidate at a higher burden; its AP difference from the selected candidate has no prespecified superiority test.

Strict persistence issues no warnings because eligibility already requires the latest available flow to be below threshold. This is an intentionally weak onset baseline, not evidence that these models beat a strong operational hydrological forecast. No historical authoritative river-flow forecast was available for this comparison.

## Geographic reliability and calibration

The selected model improves station AP at 15 of the 25 gauges with positive windows and regresses at ten. Aggregate AP improvement does not imply universal geographic improvement. Its AP is .2076 versus .1646 in 2023 and .2376 versus .1990 in 2024. Selection geography is distinct from final basins, but broad climate regions overlap.

| HUC2 | Events | Direct AP | Selected AP | Direct event recall | Selected event recall | Direct false /100 | Selected false /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 6 | .2999 | .3092 | 50.0% | 50.0% | 3.474 | 3.474 |
| 02 | 11 | .3038 | .2453 | 45.5% | 54.5% | 2.311 | 2.578 |
| 03 | 22 | .1017 | .1103 | 45.5% | 31.8% | 4.894 | 2.945 |
| 05 | 5 | .3718 | .3739 | 40.0% | 60.0% | 2.957 | 4.301 |
| 07 | 5 | .1872 | .1939 | 20.0% | 40.0% | 1.671 | 1.542 |
| 08 | 9 | .4285 | .4592 | 55.6% | 55.6% | 2.802 | 2.802 |
| 10 | 11 | .3544 | .3399 | 36.4% | 45.5% | 1.411 | 1.881 |
| 11 | 9 | .3245 | .3619 | 44.4% | 44.4% | 1.042 | 1.875 |
| 12 | 4 | .1612 | .1605 | 25.0% | 25.0% | .358 | .537 |
| 15 | 3 | .0446 | .1998 | 0.0% | 0.0% | .000 | .000 |
| 17 | 14 | .2067 | .2714 | 42.9% | 50.0% | 1.524 | .938 |
| 18 | 3 | .2426 | .3916 | 33.3% | 33.3% | .840 | .420 |

HUC2 regions 04 and 09 have no observed target events, so their AP and event recall are undefined. Selected-model false alerts are .913 and .767 per 100 decisions there. Small regional event counts limit conclusions about local performance.

The selected model fails the pooled false-alert ceiling of 1.5 and the regional ceiling of three. Its frozen conservative regional policy reduces the pooled burden to .242 but detects only 16.7% of events. Tightening the alert policy sacrifices most useful detection.

Calibration also fails to transfer. In the selected model's .3-.4 probability bin, 160 decisions average .344 predicted probability and .181 observed frequency. Low Brier loss in this rare-event population is not sufficient evidence of reliable probabilities. All bin counts remain in the evaluation artifact.

Test-label matched-budget thresholds give selected-model event recall 34.3% at .985 false alerts per 100 decisions, versus direct LightGBM's 29.4% at .969. These are non-deployable oracle diagnostics. They use test outcomes and cannot rescue the failed operational claim or justify choosing a new threshold.

## Failed hypotheses and decision

Local transition risk alone is inadequate at this observation delay and forecast horizon. Adding local analog features to hydrology alone also fails. Weather correction is necessary in this comparison, although these ablations do not isolate every individual weather or snow predictor. Treating the analog prior as a correction starting point produces higher final AP than ordinary feature concatenation, but only the selected-versus-direct comparison has frozen uncertainty criteria.

The aggregate AP criterion passes. Adequate population passes. Event-recall gain, pooled false-alert control and regional false-alert control fail. There is no overall operational improvement claim and no integration into PlanIT's 0-10 hazard index. A discharge-onset probability cannot yet be interpreted as site danger, injury probability or verified flooding.

The next research priority is a frozen comparison of station-local historical calibration and shared calibration, with a new independent cohort and as-issued input timing. A blocked pre-evaluation local record may support calibration without target-period labels, but its transfer across changing climate and measurement practices remains a hypothesis. Keep the current ranking model fixed for that test. More model capacity is not the demonstrated bottleneck. Upstream propagation needs verified overlapping subdaily observations before another graph experiment is defensible.

## Runtime and reproducibility

The experiment uses CPU workers with two model threads and separate LightGBM and Torch processes. Classic selection, including new candidate fitting and evaluation, takes 2.38 seconds at 589 MB peak RSS; neural selection takes 1.56 seconds at 661 MB. Classic final evaluation takes 2.16 seconds at 285 MB; neural final evaluation takes 1.14 seconds at 392 MB. Selected-model inference for the complete final frame takes .0175 seconds with analog predictors already materialized. These measurements exclude source collection and analog-library construction; no end-to-end live latency claim follows.

The source audit checks 143 unique files totaling 204,247,324 bytes, mostly reused forecast chunks and discharge caches. New discharge responses occupy 8,912,419 bytes. No new dependency or checkpoint was downloaded. Peak memory figures describe model workers, not collection. Package versions are frozen in both selection records.

The benchmark sequence below was run from the repository root. Collection, selection and evaluation commands intentionally refuse to overwrite existing frozen outputs. Do not delete retained evidence to rerun them. Verify the retained run with the audit command. A full rerun requires the implementation from `d94359c`, the same original source bytes and baseline artifacts, and a separate research workspace with a new output directory. Fresh API responses can contain revisions and are not a bitwise reproduction. Large source caches, datasets, fitted artifacts and per-decision predictions remain ignored local files; committed hashes alone do not make a clean checkout self-contained.

```sh
.venv/bin/python -m scripts.analog_research training
.venv/bin/python -m scripts.analog_worker select --family classic
.venv/bin/python -m scripts.analog_worker select --family neural
.venv/bin/python -m scripts.analog_worker freeze
.venv/bin/python -m scripts.analog_research test
.venv/bin/python -m scripts.analog_worker evaluate --family classic
.venv/bin/python -m scripts.analog_worker evaluate --family neural
.venv/bin/python -m scripts.analog_worker report
.venv/bin/python -m scripts.audit_analog
.venv/bin/python -m pytest -q
```

[`audit_analog.py`](../scripts/audit_analog.py) verifies source and frozen-input hashes, exact reuse of development rows, reference thresholds, calendar-lag predictors, every final label and nominal event lead, sampled analog queries, basin separation, prediction hashes and reported aggregate metrics. Saved predictions were additionally replayed exactly from all nine artifacts in isolated classic and neural processes. The full repository suite passes 188 tests with three existing SHAP color deprecation warnings. Three focused tests cover causal reference fitting, missing calendar days and offset inference. No frontend code or application behavior changed; browser checks were not repeated.
