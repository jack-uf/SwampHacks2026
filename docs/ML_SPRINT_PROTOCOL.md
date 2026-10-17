# Geographic model improvement: frozen protocol

Frozen October 17, 2026, before new county outcomes or validation scores are inspected. The separate protocol commit establishes ordering. This is a bounded retrospective experiment, not evidence of prospective field safety.

## Question and population

Can broader fitting geography, weather-only inputs and pressure, instability and antecedent-rain features improve reported flood and severe-storm prediction while reducing false alerts without losing recall?

The existing evaluations and leave-one-state-out diagnostic are inspected development evidence. Their test results cannot validate this sprint. Original models, data and evaluations stay intact. The experiment lives in `experiments/sprint-20261017/`; its `protocol.json` owns county identities and `preserved.json` records original artifact hashes.

The new sample consists of two convenience-selected counties in each of 16 states absent from the existing 24-county training dataset. Counties were chosen as population centers or counties containing state capitals, without inspecting their event counts. Census 2024 county internal points supply coordinates. Neither population weighting nor national representativeness is claimed.

| Role | States | New counties | Use |
| --- | --- | ---: | --- |
| Development | KS, MT, ND, NE, NM, NV, SD, WI | 16 | Fit, calibrate, temporal selection; fresh 2026 temporal evaluation |
| Geographic validation | IA, ID, MI, MO | 8 | Selection only, July to December 2025 |
| Geographic test | KY, NY, OR, VA | 8 | Independent evaluation only, January to June 2026 |

The 18 original development counties join the fitting population only through December 2025. Original geographic-validation and geographic-test counties never enter this experiment. All years of the four new test states are excluded from fitting, calibration and selection. Predictions for new 2026 temporal-test counties also remain unseen until selection is frozen.

## Data and timing

Reuse the exact cached NOAA Storm Events 2024, 2025 and 2026 files that produced the existing labels, with checksums. The 2026 snapshot ends in June. Use reported Flood/Flash Flood and Thunderstorm Wind/Tornado/Hail intervals overlapping UTC county-days, with the existing standard-time conversion and closed-interval policy. Join on county FIPS, never proximity or forecast-grid coordinates. These are county reports, not site incidents. A negative label means no report in that snapshot; reporting and revisions limit reliability.

Use archived Open-Meteo GFS seamless Previous Runs, nominal 48-hour lead for every hourly forecast, January 2024 through June 2026. Request precipitation, temperature, humidity, gust, mean sea-level pressure and CAPE. A forecast-only probe confirmed non-null pressure and CAPE for January 25 to 31, 2024 at Alachua before protocol freezing. No new county labels were inspected for that probe. Original development counties reuse their cached four-variable forecasts and request only pressure/CAPE.

The decision is at 00:00 UTC on D minus 1; the target is UTC day D. Under the nominal 48-hour-lead assumption, target-hour initialization precedes the decision by 1 to 24 hours. Antecedent windows also use archived forecasts, not subsequently observed weather. This prevents target-weather substitution but does not establish actual dissemination timestamps, which the archive lacks. GSOD observations and SWDI radar products are excluded as predictors unless issue-time availability can be established; this sprint does not attempt that reconstruction.

Pre-selection correction, October 17: static review corrected the original claim of an at-least-24-hour initialization margin to 1 to 24 hours. No data, feature, split, recipe or selection rule changed, and no validation or final outcome was inspected for this correction. The selection manifest also pins imported implementation hashes and package versions. An exclusive evaluation-start marker prevents retries after an interrupted final evaluation; retain that marker and report the interruption rather than restarting on the same holdout.

Require complete ordered, unique UTC hours, provider units, finite values, 24 target hours and 168 antecedent rainfall hours. Pressure trend requires the previous day's 24 forecast hours. Withhold incomplete days identically for every candidate and baseline; do not impute or substitute reanalysis. Stop for a repeated provider rate limit or request exceeding the free research budget. At most 50 full-period point requests, paced at least 10 seconds apart; cached downloads retain URLs, retrieval times and SHA-256 hashes.

## Fixed recipes and features

Fit on 2024, sigmoid-calibrate logit probabilities on January to June 2025, choose on July to December 2025. Calibration retains natural event prevalence. Use the installed library versions and seed 42. No hyperparameter search or additional model family after scores are seen.

Evaluate class-balanced standardized logistic regression, XGBoost and LightGBM using the exact fixed tree settings in `scripts/train.py`, on each of three feature sets:

1. Existing enhanced 13 features, including coordinates and season.
2. The same features without latitude, longitude, season sine or season cosine (9 weather features).
3. Those 9 weather features plus seven-day antecedent forecast rainfall, minimum target-day sea-level pressure, target-day mean pressure minus previous-day mean pressure, and maximum target-day CAPE (13 weather features).

Also fit the served recipe on the original development geography alone as an expansion control: base-eight-feature logistic regression for flood, enhanced-feature LightGBM for storm. Fit that recipe on expanded geography too; for storm it is already among the nine candidates. Controls use this sprint's common complete rows and periods. Compare with the untouched served model at its saved threshold on exactly the same evaluation rows. Do not retune the served comparator.

Choose each candidate's highest validation threshold attaining at least the served model's pooled validation recall, with a minimum recall of 0.30. The pool combines original/new temporal validation and new geographic validation. Mark thresholds with alert rate above 20% unsupported; do not silently relax the recall floor. Among supported candidates, select the highest equal-weight mean of temporal and geographic validation average precision, breaking ties by lower mean Brier score, then recipe name. If none is supported, record no promotable candidate and evaluate the ranking winner diagnostically. Freeze winner, model, feature list, threshold, validation results and data hashes before final scoring.

## Independent evaluation and promotion

Evaluate only the frozen winner, served comparator and training-prevalence climatology on each fresh January to June 2026 population: 16 new development counties (temporal test) and 8 new geographic-test counties. Do not score discarded candidates on these outcomes. Report average precision (AP, the specified PR-AUC summary), precision, recall, confusion matrix, Brier score, ten equal-width reliability bins with counts, calibration error, false-positive rate, alert rate and false alerts per 100 county-days. Include state and county results, and coverage/missingness by role. AP from one positive is descriptive only.

Use 300 paired calendar-week bootstrap resamples, retaining all same-week counties together, seed 42. Report descriptive 95% intervals for winner-minus-served AP, Brier, recall and false alerts per 100 county-days. Weather dependence across weeks and the small state sample limit these intervals.

Promotion requires **every** condition below in **both** fresh populations:

- At least 20 positive county-days, a supported validation threshold and test alert rate at most 20%.
- AP-difference interval lower endpoint above zero and Brier-difference interval upper endpoint at most zero.
- Recall at least 0.30 and no lower than the served model; false alerts at least 10% lower than the served model.
- No test state with at least five positives loses more than 0.10 recall relative to the served comparator.

These are conservative experiment gates, not supervisor-approved operating costs. A passing result permits integration at the trained county points only, with the existing geographic restrictions, shared probability contract, SHAP explanations and safety rules. A failing result retains the served model. No threshold, feature, calibration or county change may be chosen from final outcomes. A later attempt needs new independent data and a new protocol.

## Stop and verification

Run this one fixed experiment, inspect source quality and report all selected-model outcomes, including regressions. Stop after its independent evaluation; do not start a second search on these tests. Preserve time for the complete backend test suite, frontend build and read-only local application smoke check. Verify preserved hashes, document results and limitations, and commit verified changes. No persistent application data, frontend redesign or unrelated integration is in scope.
