# Catchment-response and emerging-flow research

A small day-of-onset neural model detects more emerging high-flow events than the selected LightGBM baseline at slightly fewer false warning decisions on fresh watersheds. Its AP advantage remains uncertain, and regional false-alert control fails. The ambitious catchment-conditioned mixture and auxiliary trajectory objective do not outperform it. Keep every candidate experimental and preserve production models and safety rules.

The defensible contribution is an independently screened watershed benchmark, a measured early-detection improvement for a prespecified neural comparator, and evidence against adding catchment gating or trajectory supervision in this configuration. This is not a demonstrated flood prediction system or a claim of architectural novelty.

## Evidence and frozen decisions

[The protocol](RESPONSE_RESEARCH_PROTOCOL.md) defines the hypotheses, eight candidates, data contracts, split, calibration, alert budgets and improvement criteria. Protocol and metadata-only geography were committed in `8f4abc4` before collecting new observations. Selection, source provenance, model hashes and thresholds were committed in `d6e51b4` before any final evaluation. No model, feature, threshold or selection criterion changed after that evaluation.

The implementation uses [`response_research.py`](../scripts/response_research.py) for data, [`response_network.py`](../scripts/response_network.py) for neural models and [`response_worker.py`](../scripts/response_worker.py) for selection and evaluation. Results below come from [`evaluation.json`](../experiments/sprint-20261017-response/evaluation.json); every station, region, year, probability-reliability bin and alert policy remains there. [`selection.json`](../experiments/sprint-20261017-response/selection.json) records selection performance and hashes. [`provenance.json`](../experiments/sprint-20261017-response/provenance.json) owns source URLs, timestamps, checksums, eligibility exclusions and historical signatures.

## Research choice

The hypothesis was that the same forecast rainfall has different consequences in a flashy catchment, a buffered catchment and a snow-dominated catchment. Ten measured flow descriptors computed strictly before 2016 describe response without using station identifiers. A four-expert neural mixture learns context-dependent weights from those descriptors, drainage area and elevation. Three auxiliary outputs predict residual normalized log flow at D+1 through D+3. Each expert predicts four onset-day categories, including no onset; the warning probability is one minus the no-onset probability.

Kratzert et al.'s entity-aware regional model, Feng et al.'s differentiable hydrological regionalization, and Addor et al.'s CAMELS attributes motivated conditioning on catchment behavior. The protocol links the primary papers and retains exact retrieval manifests. This implementation does not reproduce those papers, enforce mass conservation or model a river graph. Static response signatures are one testable approximation to catchment heterogeneity.

The experiment compares that design with a pooled network, a uniform four-expert mixture, direct LightGBM, signature-augmented LightGBM, logistic regression and calibrated forecast rainfall. These ablations test context, model capacity, auxiliary flow learning and conventional response features. They do not isolate every difference between a neural categorical target and binary LightGBM.

## Independent geography and measured target

The station screen starts with metadata, not event counts. It considers two lowest-identifier gauges per eligible HUC4, then orders HUC4 groups with the frozen hash rule. USGS NLDI upstream polygons exclude any positive-area overlap with all 131 prior-registry gauges and with earlier accepted new groups. Gauges within one new group may share drainage, so uncertainty resamples groups rather than pretending every gauge is independent.

The metadata audit covered 93 new candidates and reserved 52 gauges in 28 disjoint groups. Eight groups supply geographic selection; twenty supply final evaluation. Six candidate polygons intersected prior populations or previously selected groups. One candidate, `05291000`, was excluded because indexed and station-metadata coordinates differ by 2.01 km. No third gauge replaced an excluded candidate. [`geography.json`](../experiments/sprint-20261017-response/geography.json) retains exclusions, boundary hashes, indexed-gauge request records and three HUC-code disagreements. Exact indexed gauge identity, coordinates and polygon checks determine the overlap screen. The explicitly verified coordinate mapping for prior gauge `04127918` is documented in the protocol because NLDI has no indexed gauge record for it.

Final data contain 36 gauges in all 20 reserved HUC4 groups, spanning eleven HUC2 regions. Gauge `03592718` has no complete eligible final decisions and was not replaced. The final period is January 8, 2023 through September 25, 2024. There are 15,234 decisions, 459 positive windows and 159 distinct station/onset events, with 3.01% positive-window prevalence. This population is different from the earlier onset test; AP values across those populations are not an improvement comparison.

An onset is approved USGS daily mean discharge strictly above the station's pre-2016 95th percentile after seven consecutive quiet days. Eligible prediction times require seven known quiet days ending D-3. The window is local D+1 through D+3; nominal lead to local onset-day start ranges from 28 to 79 hours. An unknown measurement cannot certify a negative. This target is newly emerging high flow, not persistent high water, a verified flood-stage crossing, inundation or danger at a job site.

Development reuses the exact 74 usable prior-development gauges and adds 15 geographic-selection gauges. The collected development dataset contains 107,689 decisions, including earlier geographic rows retained but not used for fitting. Actual fitting uses 49,980 decisions from 57 gauges in 2017-2020, containing 581 events. Calibration uses 13,562 decisions and 122 events in 2021. Temporal selection uses 13,322 decisions and 107 events in 2022; fresh geographic selection uses 3,950 decisions and 21 events in 2022. The latter is small and limits model selection reliability. All source exclusions remain in provenance.

USGS exact-approved daily mean discharge supplies outcomes and lagged measurements. Caravan MultiMet v1.1 supplies archived ECMWF HRES basin precipitation and temperature forecasts. The existing causal snow/liquid-memory inputs and forecast timing conventions are unchanged. Added response descriptors use no post-2015 measurements; storm load divides forecast rain by historical reference discharge converted to millimeters per day. This is a relative loading proxy, not measured infiltration. Source licensing and original forecast attribution are documented in [the hydrology source report](HYDROLOGY_RESEARCH_RESULTS.md). No paid API, synthetic labels, satellite images, flood-stage thresholds or pretrained checkpoint enter this experiment.

## All eight results

AP is average precision. Smaller Brier and log loss are better. Event recall counts distinct events; precision counts warning decisions. Each candidate's threshold was frozen on selection data to target one false warning per 100 eligible decisions. Actual final false-alert rates differ, so this table is not an exactly matched final-budget comparison.

| Candidate | Selection AP | Fresh AP | Brier | Event recall | False alerts /100 | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated HRES rainfall | .1628 | .1341 | .02777 | 19.5% | .945 | 26.9% |
| Logistic regression | .1990 | .1877 | .02769 | 32.7% | 1.575 | 27.7% |
| Selected direct LightGBM | .2532 | .2059 | .02716 | 27.7% | 1.326 | 28.4% |
| Response-feature LightGBM | .2339 | .2183 | .02654 | 28.9% | 1.306 | 28.9% |
| Prespecified neural winner, pooled | .2324 | .2254 | .02624 | 38.4% | 1.267 | 33.7% |
| Catchment-gated mixture | .2215 | .2038 | .02691 | 34.6% | 1.589 | 29.0% |
| Gated mixture with trajectory loss | .2156 | .1952 | .02715 | 34.6% | 1.648 | 28.3% |
| Uniform mixture with trajectory loss | .2237 | .2192 | .02616 | 34.0% | 1.201 | 34.6% |

LightGBM remains the overall selected candidate because it won before final inspection. The pooled neural model was separately selected among neural candidates before evaluation, so its comparison is prespecified. Its higher final AP does not retroactively make it the overall winner.

The pooled network detects 61 events versus 44 for LightGBM, adding 22 and losing five. It produces 193 false warning decisions versus 202. Its decision-window recall is 21.4% versus 17.4%, and log loss is .10289 versus .10359. It warns 54 events at least 48 hours before onset-day start, compared with 39. Median earliest lead is 76 hours for both detected subsets. Among their 39 shared detected events, median paired lead gain is zero. The gain is broader early detection, not earlier warnings for the typical shared event.

The AP difference is .01945. Paired 500-draw bootstrap 95% intervals are [-.00361, .04289] by HUC4 and [-.00541, .04089] by month. Both include zero. The event-recall difference is .1069, with intervals [.0396, .2003] and [.0402, .1559]. These are separate descriptive cluster analyses, not multiplicity-adjusted guarantees or a joint independence model. Warning windows overlap; month resampling retains the implementation's maximum event-weight convention across included windows.

At approximately one false warning per 100 final decisions, test-label oracle thresholds give pooled neural event recall 35.2%, LightGBM 23.9% and forecast rainfall 20.1%. Their rates are .998, .991 and .945, respectively. This supports further investigation of the neural ranking at a similar burden, but oracle thresholds use outcomes and cannot be deployed. Frozen .5% and 2% selection-budget results are retained in the evaluation, without choosing whichever policy looks best afterward.

## Geographic reliability remains the obstacle

| HUC2 | Events | LightGBM AP | Neural AP | LightGBM event recall | Neural event recall | LightGBM false /100 | Neural false /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 03 | 44 | .2539 | .2514 | 29.5% | 38.6% | 1.536 | 2.135 |
| 06 | 11 | .4489 | .4523 | 54.5% | 63.6% | 3.285 | 2.669 |
| 07 | 22 | .1366 | .1435 | 27.3% | 31.8% | 1.681 | 1.595 |
| 08 | 14 | .1845 | .1930 | 50.0% | 50.0% | 6.101 | 3.826 |
| 09 | 6 | .3401 | .3686 | 33.3% | 50.0% | 1.434 | .522 |
| 10 | 36 | .2159 | .2081 | 19.4% | 36.1% | .440 | .595 |
| 11 | 5 | .0520 | .0632 | 0.0% | 0.0% | .284 | .095 |
| 12 | 7 | .0957 | .0779 | 14.3% | 0.0% | 1.196 | .368 |
| 15 | 7 | .3160 | .3696 | 14.3% | 28.6% | .120 | .600 |
| 16 | 2 | .8069 | .5523 | 0.0% | 50.0% | .000 | .000 |
| 17 | 5 | .4318 | .4726 | 20.0% | 80.0% | .302 | 1.813 |

Some regions have only two to seven events; their rates are unstable. The pooled network misses every event in HUC2 11 and 12, and its HUC2 08 false-alert rate exceeds the frozen three-per-100 regional ceiling. Mean station AP is .2836 versus .2944 for LightGBM despite better pooled AP. Two stations have no positive windows and undefined AP. The neural gain is not uniform geographic improvement.

Pooled neural AP is .2067 in 2023 and .2402 in 2024, compared with .1841 and .2288 for LightGBM. The conservative maximum-region threshold lowers neural false alerts to .230 per 100 but detects only 10.7% of events. It does not solve geographic reliability without sacrificing usefulness. Probability-reliability bins and log loss remain available alongside Brier, rather than treating AP as calibration evidence.

Both prespecified improvement claims fail. The overall winner is the unchanged baseline. The neural winner is not response-conditioned, its AP interval includes zero, and regional false alerts exceed the ceiling. No model earns integration into the 0-10 index or safety decisions.

## Failed hypotheses and useful direction

Historical response features improve final AP modestly over direct LightGBM but lose on selection and add only two detected events. This is suggestive, not independently confirmed incremental value. Learned catchment gating reduces final AP relative to the pooled network, and adding trajectory loss reduces it further. The uniform auxiliary mixture outperforms the gated auxiliary mixture despite having no learned catchment gate. Greater parameter count or more hydrological context did not justify the more elaborate model.

The gated auxiliary model's normalized log-flow MAE at D+1/D+2/D+3 is .05736/.06105/.06583, versus lagged persistence .05632/.06188/.06628. Improvement is small and absent at D+1. The uniform mixture achieves .05542/.05951/.06237, yet still trails pooled neural onset AP. Average trajectory quality is not a reliable proxy for onset-warning usefulness.

The strongest next step is to test whether onset-day supervision itself explains the small network's detection advantage. Reserve new geography or a genuinely uninspected later period first, then compare binary and onset-day heads on the same pooled network with identical inputs and fixed regional alert costs. Add audited hourly rise observations only after establishing as-issued observation and forecast timestamps. Regional calibration should be evaluated with delayed labels and a fixed update rule, not tuned on these now-inspected outcomes. An upstream graph remains a separate hypothesis until connectivity, travel times and feature availability are audited. Do not add more experts to this dataset on the strength of these results.

## Resources, verification and limits

Classic selection takes 2.23 seconds at 469 MB peak RSS; neural selection takes 16.95 seconds at 635 MB. The pooled network has 1,959 parameters; gated models have 7,888. Per-candidate fit/calibration/selection times and package versions are retained in the family-selection files. Final classic and neural evaluation take 1.40 and 2.15 seconds, respectively. A recorded full-test probability call takes about .025 seconds for both pooled neural and LightGBM. These are cached-data process measurements on this arm64 Mac, with two threads per worker, not portable benchmarks. Neural and boosted models run in separate processes. No MPS or checkpoint download was used. Added source bytes, including diagnostic polygon caches, total 18,248,333, below the frozen budget.

[`audit_response.py`](../scripts/audit_response.py) reconstructs all 15,234 final labels, quiet-history eligibility, measured lags and local-day leads from original USGS records. It checks boundary independence and hashes, recomputes historical signatures, and reproduces all eight saved probability vectors from frozen artifacts to absolute tolerance 1e-14. Family audit files record the exact evaluation and audit-code hashes. The full Python suite passes 178 tests with three existing SHAP deprecation warnings. The frontend build passed during implementation; no frontend or backend application source changed. Preserved production and prior-experiment hashes pass.

Approved observations may include later revisions. D-3 discharge and D-1 forecast availability remain assumptions, not verified historical publication times. Forecasts are basin averages, daily means miss instantaneous crests, simplified modern polygons do not reconstruct historical diversions, and long pre-2016 histories are required. This is transfer to unfamiliar gauged catchments, not prediction without gauge history. Positive predictive performance says nothing about crew incidents or site-specific flood depths. The finite sample and repeated research iterations also limit broad claims, even though this final cohort was untouched for the current experiment.

## Reproduction and retained files

The experiment pins dependencies in [`requirements.txt`](../experiments/sprint-20261017-response/requirements.txt). Existing hydrology and onset caches and frozen manifests are prerequisites. Raw observations, polygons, feature tables, model artifacts and per-row predictions remain in ignored local directories; Git retains source hashes, registry, protocol, selection, summaries and audits. Archive those local files with the manifests for exact future auditing. An upstream API revision can prevent byte-identical recovery from a URL alone; checksum disagreement fails explicitly.

The commands below were run in order from the repository root. Preparation is metadata-only and fails if the registry already exists. Collection and selection refuse to overwrite completed frozen state. In a clean reproduction, retrieve the recorded sources and verify their hashes before proceeding. Do not delete a completed experiment to rerun it in place.

```sh
.venv/bin/python -m scripts.response_research prepare
# Commit registry, geography and protocol before new observations.
.venv/bin/python -m scripts.response_research collect
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
.venv/bin/python -m scripts.response_worker classic
.venv/bin/python -m scripts.response_worker neural
.venv/bin/python -m scripts.response_worker freeze
# Commit selection, family selection files and provenance before evaluation.
.venv/bin/python -m scripts.response_worker evaluate-classic
.venv/bin/python -m scripts.response_worker evaluate-neural
.venv/bin/python -m scripts.response_worker report
.venv/bin/python -m scripts.audit_response classic
.venv/bin/python -m scripts.audit_response neural
.venv/bin/python -m pytest -q
```

The frozen `protocol.json` is committed data, not a generated CLI command. Its input hashes bind the protocol text, station registry, geography audit, previous scripts and previous committed experiment records. Reuse that manifest for this exact run. A new population requires a new experiment directory and a new pre-outcome freeze, not edits to these accepted records.
