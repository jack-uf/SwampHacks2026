# Flood and storm county-event models

## Intended interpretation

The classifiers estimate whether NOAA reports a qualifying event somewhere in a county during tomorrow's UTC day. They do not estimate worker injury, site inundation, lightning occurrence, or clearance. Even within a sampled region, the model is experimental decision support. The 150 km inference guard is a conservative demo restriction, not a validated transfer boundary.

Training covers twelve convenience-selected counties: Alachua, Hillsborough, Miami-Dade (FL); Harris, Dallas, Travis (TX); Oklahoma, Tulsa (OK); Shelby, Davidson (TN); Maricopa (AZ); and Los Angeles (CA). Each county uses one explicitly stored representative point, not every plant or event coordinate. All Los Angeles rows are excluded from fitting, calibration and selection; its 2025 rows form a geographic test. Its 2024 rows are intentionally unused.

## Prediction-time contract and leakage control

An issue time is 00:00 UTC on the day before the target date. The target spans 00:00–24:00 UTC on the following date: a 24-hour window starting 24 hours after issue, rather than a rolling next-24-hour window. Each target hour uses the archived GFS `_previous_day2` forecast, with a 48-hour lead, available before the issue time. Hourly forecasts can come from different initializations. Archive publication delays and historical revisions are not fully reconstructed; this is a retrospective archive benchmark, not a prospective operational trial.

The eight predictors are target-day forecast precipitation sum, maximum temperature, mean relative humidity, maximum gust, representative latitude, longitude, and sine/cosine of day of year using 365.25 days. Inference constructs the identical feature list and units. Realized target-day observations, reanalysis, event narratives and future outcome variables are not predictors.

Antecedent rainfall, soil moisture, elevation and FEMA exposure are not ML features because a consistent issue-time archive was not assembled for them. Recent modeled rainfall, soil moisture and mapped exposure instead contribute transparently to the deterministic index.

All four hourly archived variables must have 24 non-null hours or the entire day is discarded. No incomplete day becomes a dry day. The table has 8,532 rows: 711 days per county. NOAA county events are matched by state/county FIPS and UTC interval overlap. Zero means no qualifying report in the chosen archive, not verified absence.

## Partitions, fitting and selection

| Partition | Dates | Rows | Flood positives | Storm positives |
|---|---|---:|---:|---:|
| Training, 11 counties | Jan 21–Aug 31, 2024 | 2,464 | 51 | 102 |
| Calibration, 11 counties | Sep–Oct 2024 | 671 | 4 | 4 |
| Validation, 11 counties | Nov–Dec 2024 | 671 | 9 | 7 |
| Temporal test, 11 counties | Jan–Dec 2025 | 4,015 | 101 | 115 |
| Geographic test, Los Angeles | Jan–Dec 2025 | 365 | 5 | 3 |

XGBoost is the primary candidate; LightGBM and standardized, median-imputed logistic regression are benchmarks. Trees use 140 estimators, depth three, learning rate 0.05; XGBoost uses 0.8 subsampling and 0.9 column sampling, LightGBM seven leaves. Training-only class weighting addresses imbalance. Seeds are 42. Fixed hyperparameters are in `scripts/train.py`.

Each candidate receives a sigmoid calibrator fitted on logit probabilities from the disjoint calibration period, preserving natural event prevalence. Four positive calibration examples per hazard are far too few to establish reliable probability calibration; Brier scores are diagnostics, not proof. The classification threshold maximizes F2 on validation, prioritizing recall. The winner maximizes validation average precision, with Brier as a tie-breaker. Neither model nor threshold is chosen using test results. Models are not refitted on validation/test afterward.

Both winners are logistic regression. All six candidate results, confusion matrices, thresholds, partition sizes and geographic metrics are saved in `models/evaluation.json` and displayed in the app. PR-AUC here means **average precision**, not trapezoidal integration. False alerts per 100 county-days is `100 * FP / N`, distinct from false-positive rate `FP / (FP + TN)`.

## Measured performance and interpretation

See the generated EVALUATION.md for regional temporal test metrics. The flood confusion matrix is `[[2468,1446],[22,79]]`; storm is `[[2931,969],[38,77]]`, with rows actual negative/positive and columns predicted negative/positive. Selected thresholds are approximately 0.007802 and 0.013141 respectively. Very low thresholds explain high recall and large false-alert burdens.

Training-prevalence constant baselines have 2025 PR-AUC 0.02516 flood and 0.02864 storm, and Brier 0.02454/0.02798. The selected models improve ranking over those baselines but only modestly improve Brier. These comparisons do not justify operational deployment.

The geographic test is deliberately reported despite poor results. Selected flood precision is 0.0182, recall 1.0, PR-AUC 0.1069, with 73.97 false alerts per 100 days. Selected storm precision and recall are both zero at the selected threshold (three events), PR-AUC 0.0861. Small event counts, regional convenience sampling, imperfect reports, seasonal distribution shifts and spatially correlated days limit generalization. Confidence intervals and prospective external validation have not been established.

The API flags facilities farther than 150 km from any of the eleven training representatives as extrapolation and excludes their model component from the index. Predictions can still be inspected with that warning. Proximity alone does not demonstrate validity, and no national or plant-level validation is claimed.

## Explanations and artifact lifecycle

For each prediction, replace one feature with its training median and recompute calibrated probability. The signed difference shows reference-feature sensitivity. It is neither causal nor additive; correlated features and interactions affect it. Geographic and seasonal sensitivity can reflect sampling patterns. Do not describe these values as SHAP or as proof of physical mechanisms.

`models/hazards.joblib` stores selected estimators, calibrators, thresholds, reference medians and feature order. `models/evaluation.json` records measured results; `data/training-provenance.json` records source URLs, retrievals and hashes. Raw archives remain in the ignored cache. Only load trusted artifacts because joblib deserialization can execute code. Retraining overwrites the canonical artifacts; review changed results and restart the backend. No automatic model substitution, old-format adapter or silent migration exists.
