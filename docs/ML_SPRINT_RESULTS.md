# Geographic ML sprint: measured results

Completed October 17, 2026. **Retain both served models.** Broader geography produced a promising flood result on new states, but too few positive days support promotion. The selected storm candidate raised false alerts in both independent populations. No production model, threshold, geographic restriction, SHAP contract or safety rule changed.

This report describes one frozen retrospective experiment. The historical served-model evaluations remain in [EVALUATION.md](EVALUATION.md). Their results were already inspected and were not reused as independent validation.

## Sources and eligibility

The [source investigation](ML_DATA_RESEARCH.md) records authoritative documentation, licensing, quality, availability, overlap and bounded metadata probes. The experiment uses:

| Source | Actual use | Boundary |
| --- | --- | --- |
| NOAA Storm Events, frozen 2024/2025/2026 snapshots | County-day Flood/Flash Flood and Thunderstorm Wind/Tornado/Hail labels | Same raw revisions as the existing dataset; new geography and time are independent of fitting/selection, not an independent reporting system |
| Open-Meteo GFS seamless Previous Runs | Six archived hourly forecast variables, nominal day-two lead | Forecasts only; no observed or reanalysis weather substitution |
| Census 2024 county internal points | Coordinates for 32 additional counties | Exact county FIPS labels; county points do not establish power-plant validity |

GSOD/SSOD station observations and SWDI radar detections were investigated but not added as predictors. Their measurement, geographic mapping and issue-time availability need separate work. SWDI detections are not surface-event ground truth. Relevant Kaggle mirrors overlap authoritative NOAA sources or have incomplete provenance; synthetic datasets were excluded. No Kaggle, synthetic, station-observation or radar rows entered training.

NOAA/Census products have public US-government provenance. Open-Meteo's free service has noncommercial terms and request limits; it is not a blanket commercial utility license. Dataset-specific terms and the evidence for each source are in the source investigation. Cached request manifests retain retrieval times, byte counts and SHA-256 hashes.

## Geography, completeness and timing

The isolated table contains **44,250 complete county-days across 50 counties**. It adds 32 counties in 16 states to the original 18 development counties. The original six geographic-validation/test counties are excluded. New counties were selected as population centers or counties containing capitals before outcomes were inspected; this is a convenience sample, not national coverage.

| New county role | States | Counties |
| --- | --- | ---: |
| Development | KS, MT, ND, NE, NM, NV, SD, WI | 16 |
| Geographic validation | IA, ID, MI, MO | 8 |
| Geographic test | KY, NY, OR, VA | 8 |

All 50 locations have the same 885 complete days, January 28, 2024 through June 30, 2026, from 912 requested days. The first 27 days are unavailable under the common completeness contract, yielding 1,350 withheld county-days. No imputation was used. The fit population covers 34 counties in 20 states. See exact coordinates/FIPS in [protocol.json](../experiments/sprint-20261017/protocol.json) and request-level evidence in [provenance.json](../experiments/sprint-20261017/provenance.json).

| Partition | Dates | Rows | Counties | States | Flood positives | Storm positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fit | Jan 28-Dec 31, 2024 | 11,526 | 34 | 20 | 145 | 291 |
| Calibration | Jan-Jun 2025 | 6,154 | 34 | 20 | 76 | 172 |
| Temporal validation | Jul-Dec 2025 | 6,256 | 34 | 20 | 65 | 132 |
| Geographic validation | Jul-Dec 2025 | 1,472 | 8 | 4 | 12 | 23 |
| Fresh temporal test | Jan-Jun 2026 | 2,896 | 16 | 8 | 13 | 82 |
| Fresh geographic test | Jan-Jun 2026 | 1,448 | 8 | 4 | 9 | 27 |

The complete table also retains unused dates for the new validation/test counties, so these partition counts do not sum to 44,250. All years of new test states stay outside fitting, calibration and selection. Final temporal evaluation uses only new development counties; it does not recycle the inspected original 2026 holdout.

The decision time is 00:00 UTC on the day before the target UTC day. Each target-hour forecast has nominal 48-hour lead, implying initialization 1 to 24 hours before that decision. Antecedent rainfall uses older archived forecasts too. Actual dissemination timestamps are unavailable, so this establishes a nominal initialization constraint, not verified operational publication availability. The timing correction was committed before selection. Labels join exact FIPS and existing closed UTC event intervals, with the same standard-time conversion as the original pipeline. No nearest-county or forecast-grid join is used.

## Models and features

Fixed, seeded logistic regression, XGBoost and LightGBM recipes were compared on three feature sets. Logistic regression uses scaling and balanced class weights. Tree settings reuse the fixed recipes in `scripts/train.py`; no hyperparameter search was performed. Separate January-June 2025 data fit sigmoid calibration on logit probabilities at natural event prevalence.

| Feature set | Inputs |
| --- | --- |
| Full, 13 | Existing daily rain, temperature, humidity, gust, latitude/longitude, season sine/cosine, hourly rain maximum, wet hours, 72-hour antecedent forecast rain, rain-humidity and gust-temperature interactions |
| Weather, 9 | Full set with coordinates and season removed |
| Process, 13 | Weather set plus seven-day antecedent forecast rain, minimum target-day sea-level pressure, target-minus-prior-day mean pressure, maximum target-day CAPE |
| Base, 8; flood only | Existing served flood feature set: daily rain, temperature, humidity, gust, coordinates and season |

A separate original-geography control refits each served recipe using this experiment's common complete rows and periods on the original 18 development counties. There are 11 flood and 10 storm candidates, including these controls. Their validation scores appear below; discarded candidates were never evaluated on final outcomes.

Threshold selection uses pooled temporal/geographic validation recall at least as high as the untouched served model, with a 0.30 floor and alert rate at most 20%. Select the largest threshold satisfying recall, then select the candidate by equal-weight mean temporal/geographic AP, with mean Brier as tie-breaker. All candidates had supported thresholds. AP is average precision, the experiment's PR-AUC summary; it is not trapezoidal PR integration.

### Validation results

Recall and alert rate below refer to pooled validation; Brier is the equal-weight mean of the two validation populations. These are selection results, not independent improvement evidence.

| Hazard | Recipe | Temporal AP | Geographic AP | Mean Brier | Recall | Alert rate | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| flood | logistic_full | 0.0582 | 0.0783 | 0.00901 | 0.312 | 0.038 |  |
| flood | xgboost_full | 0.0563 | 0.0588 | 0.00895 | 0.312 | 0.048 |  |
| flood | lightgbm_full | 0.0576 | 0.0606 | 0.00896 | 0.312 | 0.048 |  |
| flood | logistic_weather | 0.0437 | 0.0382 | 0.00926 | 0.312 | 0.066 |  |
| flood | xgboost_weather | 0.0444 | 0.0361 | 0.00917 | 0.312 | 0.075 |  |
| flood | lightgbm_weather | 0.0461 | 0.0392 | 0.00914 | 0.312 | 0.080 |  |
| flood | logistic_process | 0.0440 | 0.0556 | 0.00911 | 0.312 | 0.053 |  |
| flood | xgboost_process | 0.0443 | 0.0433 | 0.00910 | 0.312 | 0.056 |  |
| flood | lightgbm_process | 0.0397 | 0.0484 | 0.00910 | 0.312 | 0.071 |  |
| flood | logistic_base | 0.0607 | 0.1152 | 0.00901 | 0.312 | 0.036 | yes |
| flood | original_geography | 0.0456 | 0.0926 | 0.00907 | 0.312 | 0.052 |  |
| storm | logistic_full | 0.1335 | 0.1824 | 0.01707 | 0.445 | 0.083 | yes |
| storm | xgboost_full | 0.1407 | 0.0869 | 0.01725 | 0.445 | 0.066 |  |
| storm | lightgbm_full | 0.1469 | 0.0914 | 0.01721 | 0.445 | 0.071 |  |
| storm | logistic_weather | 0.1044 | 0.1228 | 0.01801 | 0.445 | 0.089 |  |
| storm | xgboost_weather | 0.1137 | 0.0756 | 0.01834 | 0.445 | 0.096 |  |
| storm | lightgbm_weather | 0.1165 | 0.1088 | 0.01818 | 0.445 | 0.089 |  |
| storm | logistic_process | 0.1410 | 0.1543 | 0.01706 | 0.445 | 0.073 |  |
| storm | xgboost_process | 0.1390 | 0.0685 | 0.01758 | 0.445 | 0.079 |  |
| storm | lightgbm_process | 0.1584 | 0.0960 | 0.01735 | 0.445 | 0.074 |  |
| storm | original_geography | 0.1097 | 0.0898 | 0.01742 | 0.445 | 0.085 |  |

The selected candidates are **flood logistic_base**, threshold 0.0375672621, and **storm logistic_full**, threshold 0.0595886917. Neither process-feature recipe nor a boosted model won the frozen selection. The original-geography controls are diagnostics; final outcomes cannot establish an isolated causal effect of adding geography.

## Independent results

Only the selected candidate, untouched served comparator and fit-prevalence climatology were scored on the fresh tests. The served comparator is deliberately evaluated outside its serving population for transfer diagnosis; the application still withholds predictions there. Candidate artifacts remain isolated and are not loaded by inference.

### Ranking, classification and calibration

Temporal test has 2,896 rows; geographic test has 1,448. The positive counts are 13/9 for flood and 82/27 for storm respectively. False alerts are false-positive county-days per 100 total county-days. FPR divides by negative days; alert rate includes true positives. Precision and recall use frozen thresholds.

| Hazard | Test | Model | AP | Precision | Recall | Brier | ECE | False alerts /100 | FPR | Alert rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flood | temporal | served | 0.0312 | 0.0273 | 0.2308 | 0.00455 | 0.00248 | 3.69 | 0.0371 | 0.0380 |
| flood | temporal | candidate | 0.0287 | 0.0305 | 0.3077 | 0.00476 | 0.00387 | 4.39 | 0.0441 | 0.0452 |
| flood | geographic | served | 0.0221 | 0.0095 | 0.2222 | 0.00734 | 0.00947 | 14.36 | 0.1445 | 0.1450 |
| flood | geographic | candidate | 0.1614 | 0.0536 | 0.6667 | 0.00613 | 0.00744 | 7.32 | 0.0737 | 0.0773 |
| storm | temporal | served | 0.2004 | 0.1818 | 0.5122 | 0.02479 | 0.00854 | 6.53 | 0.0672 | 0.0798 |
| storm | temporal | candidate | 0.2052 | 0.1386 | 0.5610 | 0.02493 | 0.00566 | 9.88 | 0.1016 | 0.1146 |
| storm | geographic | served | 0.1429 | 0.1040 | 0.4815 | 0.01728 | 0.00386 | 7.73 | 0.0788 | 0.0863 |
| storm | geographic | candidate | 0.1890 | 0.0893 | 0.5556 | 0.01697 | 0.00828 | 10.57 | 0.1077 | 0.1160 |

The climatology AP/Brier pairs are flood temporal 0.00449/0.00453, flood geographic 0.00622/0.00622, storm temporal 0.02831/0.02752 and storm geographic 0.01865/0.01834. Low Brier alone is insufficient on these rare outcomes. ECE uses ten equal-width probability bins; most predictions occupy the first bin, so its small value does not prove reliable calibration. Bin counts, means, confusion matrices and all regional metrics are retained in the JSON evaluation.

### Paired uncertainty

Intervals below are candidate minus served, from 300 paired calendar-week block resamples, seed 42, across 27 weeks. All counties in a sampled week move together. These are descriptive 95% intervals; dependence across weeks and sparse states limit coverage. AP resamples without a positive are excluded (297 valid for geographic flood).

| Hazard | Test | AP difference 95% | Brier difference 95% | Recall difference 95% | False alerts /100 difference 95% |
| --- | --- | --- | --- | --- | --- |
| flood | temporal | [-0.0671, 0.0231] | [-0.000076, 0.000580] | [-0.286, 0.400] | [-0.84, 2.17] |
| flood | geographic | [0.0125, 0.2173] | [-0.002126, -0.000533] | [0.200, 1.000] | [-8.96, -4.87] |
| storm | temporal | [-0.0358, 0.0449] | [-0.000558, 0.000926] | [-0.045, 0.162] | [2.00, 5.06] |
| storm | geographic | [-0.0698, 0.1336] | [-0.001329, 0.000673] | [-0.102, 0.250] | [1.16, 4.52] |

### New-state generalization

Each state below has 362 county-days in two counties. These are fresh geographic-test states, unseen in fitting, calibration or selection. AP and recall for one or two positives are highly unstable. A zero-positive state cannot validate event detection; the stored numeric zero recall in that case has no event-recall interpretation.

| Hazard | State | Positive days | Served AP | Candidate AP | Served recall | Candidate recall | Served false alerts /100 | Candidate false alerts /100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flood | KY | 6 | 0.412 | 0.456 | 0.333 | 0.833 | 0.83 | 3.59 |
| flood | NY | 1 | 1.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.00 |
| flood | OR | 0 | n/a | n/a | n/a | n/a | 56.63 | 25.14 |
| flood | VA | 2 | 0.060 | 0.071 | 0.000 | 0.000 | 0.00 | 0.55 |
| storm | KY | 11 | 0.164 | 0.257 | 0.636 | 0.636 | 18.51 | 13.54 |
| storm | NY | 7 | 0.184 | 0.145 | 0.429 | 0.571 | 3.31 | 6.08 |
| storm | OR | 0 | n/a | n/a | n/a | n/a | 0.83 | 16.02 |
| storm | VA | 9 | 0.235 | 0.306 | 0.333 | 0.444 | 8.29 | 6.63 |

The storm temporal test also loses recall in New Mexico: 2/9 detected days for served versus 1/9 for candidate (0.222 to 0.111), exceeding the protocol's allowed 0.10 state loss. All eight temporal-test states and individual county results are preserved, including regressions, in `evaluation.json`.

## Decision and limits

The frozen gates require both fresh cohorts to have at least 20 positive county-days, supported thresholds, alert rate at most 20%, an AP difference interval above zero, a nonpositive Brier difference interval, recall at least 0.30 and at least served recall, 10% fewer false alerts, and no state with five or more positives losing over 0.10 recall.

| Hazard / population | Failed promotion gates |
| --- | --- |
| flood / temporal | Fewer than 20 positive county-days; AP improvement interval includes zero or a regression; Brier improvement interval includes a regression; False alerts not reduced by at least 10% |
| flood / geographic | Fewer than 20 positive county-days |
| storm / temporal | AP improvement interval includes zero or a regression; Brier improvement interval includes a regression; False alerts not reduced by at least 10%; Recall loss exceeds 0.10 in NM |
| storm / geographic | AP improvement interval includes zero or a regression; Brier improvement interval includes a regression; False alerts not reduced by at least 10% |

Broader-geography flood ranking and false alerts improve on unseen states: AP 0.0221 to 0.1614, recall 0.222 to 0.667, false alerts 14.36 to 7.32 per 100 county-days. This cohort still has only nine positive days, below the precommitted minimum. Temporal flood AP decreases and false alerts increase. Storm recall rises in aggregate, but false alerts rise from 6.53 to 9.88 temporally and 7.73 to 10.57 geographically; AP/Brier intervals do not establish improvement. **Neither hazard passes.**

The sprint stops after this fixed evaluation. Repeated selection, threshold changes or additional final-test scoring would consume the holdout further without creating independent evidence. A later candidate needs a new protocol and genuinely new evaluation outcomes. Geographic coverage, event counts, label/reporting bias and issue-time verification are more binding than another model family before the October 18 submission deadline.

Additional limits:

- County reported-event labels are not site incidents, injury risk or complete observations of all hazards. Nearby days and counties may share episodes. Source revisions can alter labels.
- The new sample is small and selected by convenience. Oregon has no positive days for either outcome here. National generalization is not established.
- The served comparator is frozen, but its historical fitting and threshold periods differ from this experiment. Candidate/control comparisons and final matched-row comparisons answer different questions.
- Weather archive lead time is nominal. Actual publication latency remains unverified. Neither a hindcast result nor calibration authorizes work.
- No production SHAP explanation was generated for the isolated candidates because neither was promoted. Existing inference, county restrictions and calibrated-probability SHAP remain the canonical application path.

## Reproduction, integrity and handoff

The protocol was committed in `1376bcd`. Benchmark checks and timing/dependency corrections were committed before selection (`c05790d`, `c761351`). The table, fitted candidates and selection manifest were committed in `241420d` before final scoring. The exclusive `evaluation-started.json` records consumption of the independent holdout; preserve it even if a run fails.

| Artifact | Purpose |
| --- | --- |
| [protocol.json](../experiments/sprint-20261017/protocol.json), [protocol document](ML_SPRINT_PROTOCOL.md) | Fixed counties, timing, recipes and gates |
| [preserved.json](../experiments/sprint-20261017/preserved.json) | SHA-256 of 23 original model/data/evaluation artifacts |
| [training.csv.gz](../experiments/sprint-20261017/training.csv.gz), [provenance.json](../experiments/sprint-20261017/provenance.json) | Frozen complete county-day table and source manifests |
| [selection.json](../experiments/sprint-20261017/selection.json), `selected.joblib` | All validation scores, recipes, thresholds, code/package/data hashes and selected fitted models |
| [evaluation-started.json](../experiments/sprint-20261017/evaluation-started.json) | Exclusive one-use final evaluation marker |
| [evaluation.json](../experiments/sprint-20261017/evaluation.json), [predictions.csv.gz](../experiments/sprint-20261017/predictions.csv.gz) | Pooled/state/county metrics, bins, paired intervals, failure reasons and matched predictions |

Run from the repository root with the pinned environment:

```sh
# Safe replay: validates preserved hashes, frozen dependencies, data and selection,
# then reproduces prediction rows and pooled metrics without selecting or refitting.
.venv/bin/python -m scripts.ml_sprint verify
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
```

The measured run uses NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, joblib 1.5.3 and SciPy 1.18.1. Exact versions and implementation hashes are checked by verification. `verify` checks pooled metric/prediction replay; it does not independently recompute every stored bootstrap interval.

For understanding the pipeline, `scripts.ml_sprint` has `ingest`, `select`, `evaluate` and `verify` commands. The first three produce frozen artifacts and are deliberately not rerunnable over existing results. Do not delete guards or overwrite this sprint to try another model. Retain this directory and use a separately defined future experiment with new independent data.

Verification completed: 121 backend tests, the shared map-resize behavioral test, frontend production build, original-artifact hash checks, and frozen prediction/pooled-metric replay. Thirteen new ML tests cover timing, units/completeness, partition exclusions, threshold/gate behavior, paired uncertainty, frozen dependencies and interrupted evaluation guards. [UI verification](UI_VERIFICATION.md) documents the separate visual iteration. Persistent application data was not used as experiment output.
