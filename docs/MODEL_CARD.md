# Flood and storm county-event model card

Research decision date: October 16, 2026. Read this alongside [the frozen protocol](RESEARCH_PLAN.md), [measured results](EVALUATION.md), [primary-source research](RESEARCH.md) and `models/evaluation.json`. There is one serving artifact, `models/hazards.joblib`; the original experiment is a read-only research archive.

## Intended use and target

Experimental decision support for supervisors: estimate whether NOAA reports a qualifying event somewhere in a county during tomorrow's UTC day. Flood includes Flood/Flash Flood; storm includes Thunderstorm Wind/Tornado/Hail. Neither target represents a facility incident, worker injury, lightning occurrence, site inundation, verified hazard absence or permission to work. No report is a reporting-based negative, not a measured safe condition.

NOAA's county/forecast-zone/marine namespaces are different. The pipeline audits every event-type/geography combination in `data/noaa-geography-audit.json`, checks all five target types are county records, then joins selected county FIPS codes. Zone categories such as Coastal Flood, High Wind and Tropical Storm are outside these targets. Standard NOAA timezone offsets apply even during summer; they are converted to UTC without substituting daylight time. Closed event intervals overlap a day when start < next midnight and end >= day midnight. Instantaneous events belong to their UTC day; an event ending exactly at midnight also labels that instant in the next day. Bad intervals and unknown offsets fail explicitly.

## Data and information timing

The expanded table has **21,336 complete county-days, 24 counties, January 24, 2024–June 30, 2026** (889 per county). Twelve original representative points are unchanged. Twelve new points come from official 2024 Census county gazetteer internal points; `data/regions.json` records coordinates, county/state FIPS and experiment role. These are diverse convenience samples, not a probability sample of US construction sites. A point forecast is an imperfect proxy for an event anywhere in its county.

Predictors are Open-Meteo GFS `previous_day2` forecasts. For target day D and nominal decision time D−1 00 UTC, each target hour's nominal 48-hour forecast initialization precedes the decision time. This is a fixed-lead series, **not one model initialization**. Original publication timestamps and later archive revisions are not reconstructed; nominal timing is not a prospective delivery guarantee. GFS Single Runs begins April 2, 2026, insufficient for the full benchmark. Do not use realized target weather or reanalysis to extend it.

The target spans D 00:00–24:00 UTC, starting 24 hours after the decision time. It is not a rolling next-24-hour forecast or a facility-local work shift. Historical and serving paths both call `daily_features`: require 24 exact, unique, ordered target hours with non-null precipitation, temperature, humidity and gust, plus 72 preceding hourly precipitation forecasts. Missing hours discard a day; no imputation turns it dry. Each antecedent value also uses its own archived nominal 48-hour lead, so this feature is an aggregate of earlier forecasts, **not observed accumulated rainfall**. Withheld predictions remain unknown in the index.

Base features: target precipitation sum (mm), maximum temperature (°C), mean humidity (%), maximum gust (km/h), representative latitude/longitude, and sine/cosine of day-of-year using 365.25. Enhanced features add maximum hourly precipitation, number of hours with precipitation >0, preceding 72-hour forecast precipitation sum, rain × humidity /100 and gust × temperature /100. Soil moisture, event narratives, future observations and realized rainfall are excluded. Soil and recent modeled weather still have separately labeled heuristic index roles.

## Development and final evaluation

| Partition | Counties | Dates | Rows | Flood positives | Storm positives |
|---|---:|---|---:|---:|---:|
| Fit | 18 | Jan 24–Dec 31, 2024 | 6,174 | 82 | 152 |
| Sigmoid calibration | 18 | Jan–Jun 2025 | 3,258 | 58 | 108 |
| Temporal selection | 18 | Jul–Dec 2025 | 3,312 | 34 | 57 |
| Geographic selection | 2 (GA, CO) | Jul–Dec 2025 | 368 | 1 | 3 |
| Final temporal | 18 | Jan–Jun 2026 | 3,258 | 42 | 80 |
| Final geographic | 4 (CA, AZ, WA, UT) | Jan–Jun 2026 | 724 | 4 | 3 |

New candidates never fit/calibrate on any GA, CO, CA, AZ, WA or UT row. Geographic selection and final-test states are disjoint. The frozen baseline previously fit Arizona, so **its combined geographic result is not wholly independent**. All 2025 data are development data now; the already inspected original 2025 test cannot become a fresh test.

Seven new candidates per hazard: balanced logistic/XGBoost/LightGBM, each on base/enhanced features, and unweighted enhanced logistic. Logistic standardizes; no missing-value adapter is needed. Trees retain the baseline's fixed 140 estimators, depth three, learning rate .05; XGBoost subsampling .8/column .9, LightGBM seven leaves. Class weighting is training-only; all evaluation retains natural prevalence. Seeds are 42. Each receives sigmoid calibration on logit probabilities in the separate calibration period. A constant training-prevalence comparator is also reported. The eighth candidate is the frozen original estimator and calibrator, without refitting.

Selection maximizes mean temporal/geographic-validation average precision, temporal Brier tie-break. The geographic selection sample is extremely small (one flood, three storm positives); selection stability is therefore weak. `models/selection.json` freezes winners and thresholds before any final metric is calculated. Commit 408ac8e froze the expanded data and experiment implementation before its first final evaluation. Final results were not used to tune models or expand serving scope.

Thresholds are chosen only on temporal validation. Report F1/F2 optima and a budget policy: at most 20% flagged days, at least 30% recall, then highest F1. Budget and recall floor are **unvalidated operational hypotheses**, not elicited crew costs or safety standards. If infeasible, save diagnostics but set `warning_supported=false` and withhold automatic elevated flags. Budget feasibility on validation does not guarantee that budget in later data. No threshold clears work.

## Findings and uncertainty

**Flood retained the original logistic estimator and calibrator**, with a revised validation-selected threshold of 0.0306398. Its ranking/probabilities are unchanged; reduced alerts trade away recall. **Storm selected enhanced LightGBM**, threshold 0.0756798, on validation. Final temporal development-population AP is .104 flood and .130 storm; precision .073/.088, recall .595/.613, false alerts per 100 county-days 9.76/15.56. This eighteen-county summary includes counties outside the frozen flood model's original fitting population; it is a benchmark, not a claim of expanded flood serving coverage.

On the identical ten original development counties in Jan–Jun 2026, flood AP/Brier remain .120/.0168; false alerts fall from 41.49 to 13.37 per 100 county-days and recall falls .875→.625. Storm AP worsens .150→.130, Brier .0250→.0257; false alerts fall 40.55→19.45 and recall falls .812→.667. The candidate selected on validation thus **did not establish improved final-test ranking**. Retain the predeclared selection for reproducibility rather than reselecting on the test.

Storm missed all three final geographic positive county-days at its selected threshold; flood's combined geographic set has only four positives and is not wholly independent for the frozen estimator. Nationwide inference remains unsupported. These results warrant further independent evaluation, not operational claims.

The [saved-artifact storm investigation](STORM_DIAGNOSTICS.md) reproduces the loss without refitting or selecting on known tests. Positive-slope calibration leaves AP unchanged. The selected model predicts 81.524 expected positives versus 48 observed on matched rows. Three geographic-validation positives in one county receive half the selection criterion's weight; one supplies 74.5% of geographic AP. The missed Los Angeles February 16/18 and Emery May 13, 2026 county-days join 17 NOAA records across two episodes. Their western coordinates, cold coastal winter conditions or arid outflow features expose plausible population/aggregation mismatch; calibrated-probability SHAP supports association hypotheses rather than causation. Validation-only feature perturbations are not fitted ablations. Existing final tests are now known diagnostic data, so future changes require a new protocol and untouched evaluation. Original model/data/selection artifacts remain unchanged.

All candidate metrics, [[TN,FP],[FN,TP]] matrices, prevalence, alert rate, false-positive rate, ten reliability bins/counts, Brier and ECE are saved. AP is average precision, not trapezoidal PR integration. False alerts/100 county-days =100×FP/N; FPR=FP/(FP+TN). Coarse equal-width calibration bins are sparse at rare-event probabilities; small ECE does not establish calibration. Quarter/county breakdowns expose heterogeneity and no-positive groups return null AP/ROC where undefined. Calibration improved sample size over the original four positives per hazard, but remains a retrospective diagnostic.

Compatible comparisons use paired calendar-week bootstrap resampling, 300 replicates/seed42, keeping all same-week counties together. These descriptive 95% intervals do not capture archive uncertainty, inter-week dependence or selection uncertainty. Flood AP/Brier delta intervals are exactly zero (same estimator). Storm delta AP interval is [-.0913,.0287], Brier [-.00134,.00266]; neither establishes a reliable gain.

## Geographic transfer: leave one state out

The question was how much the served recipes lose in a state they never saw, measured without touching any inspected holdout. [The protocol](SPATIAL_CV_PROTOCOL.md) was committed (393ef39) before the only run. It used the 18 development counties in 12 states, fitting on 2024 and calibrating on January to June 2025. Both arms scored the same 3,312 July to December 2025 county-days: one with a model fitted on every state, one with a model fitted without the scored row's state. No 2026 row and no holdout state was read. Output: `models/spatial-cv.json`; rerun with `.venv/bin/python -m scripts.spatial_cv`.

| Hazard | Arm | Positives | AP | Brier | Expected positives | Recall at served threshold | False alerts /100 county-days |
|---|---|---:|---:|---:|---:|---:|---:|
| flood | state seen | 34 | 0.059 | 0.0100 | 35.7 | 0.32 | 6.85 |
| flood | state left out | 34 | 0.056 | 0.0100 | 31.2 | 0.29 | 5.59 |
| storm | state seen | 57 | 0.091 | 0.0163 | 65.7 | 0.35 | 5.74 |
| storm | state left out | 57 | 0.066 | 0.0167 | 60.5 | 0.25 | 5.56 |

Paired calendar-week bootstrap (300 resamples) of left-out minus seen: flood AP [-0.021, 0.025], so no loss is detectable; storm AP [-0.057, -0.003] and Brier [+0.00005, +0.00073], a loss in both. Prevalence is 1.0% for flood and 1.7% for storm, so even the seen-state AP is only 4 to 6 times the base rate.

Per-state storm AP moves in both directions (Texas 0.067 to 0.041, Illinois 0.164 to 0.213) on 0 to 12 positives per state. The pooled drop therefore looks more like left-out probabilities that compare badly across states than uniformly worse ranking within each state. That matters because the portfolio compares sites. The result supports the exact-county serving restriction. As the protocol required, it changes no model, threshold or scope. Limits: July to December 2025 was the selection period, so the seen-state threshold metrics are optimistic, and neighbouring states share weather, so a genuinely new region could lose more.

## Serving scope, explanations and lifecycle

Serving matches the EIA county/state **exactly** to each selected bundle's actual fitting inventory, using case-insensitive county names. No distance-radius extrapolation or fuzzy county mapping remains. Flood supports its eleven original fitted counties (including Arizona); storm supports eighteen development counties. A county name mismatch is explicit unknown. Predictions use the same county representative coordinates as training, not each plant's coordinates. Geographic tests do not become supported because a retrospective score looks favorable. Matching a fitting county is a conservative population restriction, not proof of site validity.

Genuine `shap.PermutationExplainer` explains the complete **calibrated probability function**, with identity output, 16 seeded training-background rows, four forward/reverse permutation budgets and independent masking. Baseline plus all signed contributions equals probability; the UI shows the largest three and the baseline. Values are probability-point contributions, not log odds. Inputs are correlated and engineered interactions can become inconsistent under independent masking; approximate SHAP is descriptive, not causal or a physically valid intervention. Explainability does not rescue an invalid model.

In the shipped 60-facility inventory, exact county matching permits flood estimates only at Palo Verde (6008), and storm estimates at Braidwood Generation Station (6022) and Catawba (6036), subject to complete forecast availability. No facility currently has both. Expanded research geography therefore does not imply national serving coverage. The combined ML index component requires both flood and storm probabilities; one missing model keeps its fixed 20% weight unknown, while the individually supported forecast remains visible. Other index weights remain transparent heuristics, without renormalization or an injury-probability interpretation.

The saved bundle contains feature order, estimators/calibrators, threshold policy feasibility, background rows, supported regions, training timestamp and dataset SHA-256. Load only trusted joblib files. Retraining overwrites canonical artifacts and `docs/EVALUATION.md`; inspect the changes and restart the backend, which caches the artifact. There are no old-schema adapters or alternate runtime models. See README for reproduction and archive checksum verification.
