# Storm regression investigation: saved October 16, 2026 experiment

This note diagnoses the saved October 16, 2026 experiment. It does not change
the deployed models, selection, thresholds, evaluation, baseline archive or data.
The evidence is generated offline by `scripts/diagnose_storm.py` into
`output/storm-diagnostics.json`. Both are committed with this note.

## Scope and fixed diagnostic procedure

Use the existing fitted model and calibrator objects, the existing partition,
metric and block-bootstrap helpers, and the serving calibrated-probability SHAP
helper. Do not fit models or calibrators, choose replacement models, or tune on
the January–June 2026 final tests, which have already been inspected.

For feature reliance, the fixed procedure is twelve permutations, seed 42, of
the latitude/longitude pair and the season sine/cosine pair, separately, on each
of temporal and geographic validation. These are inference perturbations of
the saved model, not fitted ablations. Retain both columns of each pair together;
use identical row orders across feature groups. Report AP drop and Brier
increase, all replicates and their means. Do not use the result to select a model.
The procedure is declared here and in the script before its first execution.

For validation sensitivity, report all eight saved candidate scores, alternative
weightings as descriptive arithmetic only, and delete each geographic positive
row in turn for the two saved model objects. This changes prevalence and is a
sensitivity check, not an independent performance estimate or selection rerun.
The six nonselected new candidate model objects were not saved; do not refit them
to manufacture a retrospective full candidate bootstrap.

## Reproduction

From the repository root with the existing pinned environment:

```sh
.venv/bin/python -m scripts.diagnose_storm
.venv/bin/python -m pytest -q tests/test_storm_diagnostics.py tests/test_training.py
```

The diagnostic only writes `output/storm-diagnostics.json`. It does not load
`.env`, access keys, call providers, download data, train or tune anything.
It records runtime versions and SHA-256 digests of its inputs and source helpers,
checks the baseline manifest, reproduces saved comparison and candidate metrics,
and verifies protected inputs again before writing. Missing or inconsistent
artifacts are errors: restore the matching saved files rather than regenerate
models or silently substitute a new dataset. The local 2026 NOAA compressed
archive must match `data/training-provenance.json`; there is no network fallback.
The output has no wall-clock timestamp, so identical inputs/environment reproduce
identical JSON. Tests cover tied-score AP accounting, NOAA interval/county joins,
and rejection of mismatched evidence.

## Findings and interpretation

The matched regression reproduces exactly: average precision (AP) falls from
**0.149767921 to 0.129690924**, a change of **−0.020076997** (13.4% relative),
on the same 1,810 county-days and 48 positive labels. The paired calendar-week
bootstrap interval for the change is **[−0.091303130, +0.028711016]**. This is a
real difference in these saved predictions, with insufficient evidence to claim
a systematic population-level decline. Changing the decision threshold cannot
change AP. The saved positive-slope calibrators also leave AP unchanged here.

The selected model reduces false alerts at its saved policy, while detecting
fewer positive days. Its matched probabilities overpredict the event rate and
have slightly worse Brier score. Selection gave half its AP criterion to a
geographic validation set with only three positive days, all in one county;
one of those days contributes 74.5% of its geographic AP. These facts support
concerns about transfer, calibration and selection instability, but do not
identify which modeling change caused the regression.

All three independent-geography test positives are missed. They are **three
county-days, 17 NOAA source records and only two NOAA episodes**: Los Angeles on
February 16 and February 18, 2026, and Emery County, Utah on May 13, 2026. The
first two days belong to the same winter storm episode. They must not be
treated as three independent meteorological replications.

### Matched comparison and where the AP changed

The matched population is January 1–June 30, 2026 in the original ten counties:
Alachua, Hillsborough and Miami-Dade, FL; Dallas, Harris and Travis, TX;
Oklahoma and Tulsa, OK; Davidson and Shelby, TN. Each supplies 181 rows. Labels,
weather features and representative coordinates are identical for both model
evaluations. AP is the existing helper's `pr_auc`: average precision, not
trapezoidal precision–recall area. Confusion matrices use `[[TN, FP], [FN, TP]]`.

| Matched metric | Frozen logistic baseline | Selected `lightgbm_enhanced` |
| --- | ---: | ---: |
| County-days / positives / prevalence | 1,810 / 48 / 2.6519% | 1,810 / 48 / 2.6519% |
| AP | 0.149767921 | 0.129690924 |
| ROC AUC | 0.775941165 | 0.826824395 |
| Brier score, lower is better | 0.025025646 | 0.025724619 |
| Original / selected threshold | 0.013141327 | 0.075679783 |
| Confusion matrix | `[[1028,734],[9,39]]` | `[[1410,352],[16,32]]` |
| Precision / recall | 5.0453% / 81.25% | 8.3333% / 66.6667% |
| Alerts / false alerts per 100 county-days | 42.7072 / 40.5525 | 21.2155 / 19.4475 |

The Brier change is +0.000698973; its descriptive paired interval is
[−0.001335238, +0.002662451]. The bootstrap uses the existing helper: 300 paired
resamples, seed 42, 27 calendar-week blocks, retaining all spatial rows in each
sampled week. It does not capture all cross-week storm dependence, alternative
training realizations or the uncertainty in the earlier model selection.

The script decomposes pooled AP by summing each positive row's precision at its
score, divided by the total 48 positives. Tied rows share the precision at the
whole tied score. It checks that all 48 changes sum to the total AP change.
This measures contributions to the observed ranking difference, not causation.

| Matched period | Rows / positives | Baseline AP | Selected AP | Contribution to pooled AP change |
| --- | ---: | ---: | ---: | ---: |
| 2026 Q1 | 900 / 14 | 0.043865049 | 0.192401900 | +0.009125738 |
| 2026 Q2 | 910 / 34 | 0.193459564 | 0.134232922 | −0.029202735 |

Dallas's nine positives contribute −0.015648942, about 78% of the net decline.
Yet AP within Dallas is nearly unchanged (0.358249 versus 0.358312): ordering
across counties matters. County APs cannot be averaged to explain pooled AP.
Dallas April 28 moves from rank 13 (baseline probability 0.060312) to rank 681
(selected probability 0.026924), contributing −0.006758541. Tulsa June 22 moves
from rank 4 to rank 20, contributing −0.006250000. The JSON retains every
positive's scores, tied ranks and AP contributions, plus all county metrics.

The expanded 18-county temporal test has 3,258 rows and 80 positives; its AP is
0.122077692 for the baseline and 0.130157391 for the selected model. That
different population reverses the direction of the comparison. It does not
cancel the ten-county matched regression or establish geographic transfer.

### Fitting geography, prevalence and season coverage

The baseline is the archived class-balanced logistic model with eight features.
The selected model is the saved class-balanced LightGBM with thirteen features.
Data geography, fitting dates, feature set, estimator and calibration period all
changed together. This investigation does not fit an ablation to separate them.

| Stage | Dates, inclusive | Counties | Rows | Positive days | Prevalence |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline fit | 2024-01-21–2024-08-31 | 11 | 2,464 | 102 | 4.1396% |
| Baseline calibration | 2024-09-01–2024-10-31 | 11 | 671 | 4 | 0.5961% |
| Baseline validation | 2024-11-01–2024-12-31 | 11 | 671 | 7 | 1.0432% |
| Selected fit | 2024-01-24–2024-12-31 | 18 | 6,174 | 152 | 2.4619% |
| Selected calibration | 2025-01-01–2025-06-30 | 18 | 3,258 | 108 | 3.3149% |
| Selected temporal validation | 2025-07-01–2025-12-31 | 18 | 3,312 | 57 | 1.7210% |
| Geographic validation | 2025-07-01–2025-12-31 | 2 | 368 | 3 | 0.8152% |
| Expanded temporal final test | 2026-01-01–2026-06-30 | 18 | 3,258 | 80 | 2.4555% |
| Independent-geography final test | 2026-01-01–2026-06-30 | 4 | 724 | 3 | 0.4144% |

The current CSV contains 21,336 rows, 479 positives and 24 counties. The six
current partitions use 17,094 rows and 403 positives. The remaining 4,242 rows
and 76 positives are observations from held-out geographic roles outside their
allocated validation/test windows; they remain unused by this protocol.
The matched test is a subset of the temporal test, not another disjoint split.

Baseline fitting includes the ten matched counties plus Maricopa, AZ (224 days
per county). Selected fitting removes Maricopa and adds Hennepin, MN; Will, IL;
Indiana, PA; Brunswick, NC; York, SC; St. Charles, LA; Jackson, AL; and Jefferson,
OH (343 days per county). The state coverage changes from five to twelve states.
More states do not mean the western test geography is represented: selected-fit
longitudes range from −97.78 to −78.227765, whereas the baseline extends west
to −112.49. Los Angeles (−118.26) and Emery (−110.7211) both lie west of every
selected-fit coordinate. Selected-fit latitude spans 25.61–45.006121, compared
with baseline 25.61–36.17.

Selected-fit positive counts, each out of 343 rows, are:

| County | Positives | County | Positives |
| --- | ---: | --- | ---: |
| Alachua, FL | 5 | Hennepin, MN | 9 |
| Hillsborough, FL | 5 | Will, IL | 15 |
| Miami-Dade, FL | 9 | Indiana, PA | 3 |
| Dallas, TX | 8 | Brunswick, NC | 5 |
| Harris, TX | 17 | York, SC | 5 |
| Travis, TX | 7 | St. Charles, LA | 0 |
| Oklahoma, OK | 12 | Jackson, AL | 15 |
| Tulsa, OK | 13 | Jefferson, OH | 5 |
| Davidson, TN | 12 | Shelby, TN | 7 |

Selected fitting has 14/1,224 positives in Q1, 83/1,638 in Q2, 44/1,656 in Q3,
and 11/1,656 in Q4. Calibration has 26/1,620 in 2025 Q1 and 82/1,638 in Q2;
temporal validation has 48/1,656 in Q3 and 9/1,656 in Q4. Seasonal prevalence
therefore differs substantially across stages. Geography validation comprises
Adams, CO (3/184) and Burke, GA (0/184); all three positives occur in Q3.
Independent-geography testing comprises Los Angeles (2/181), Emery (1/181),
Maricopa (0/181) and Grant, WA (0/181). Maricopa was part of baseline fitting,
so this four-county set is independent of selected fitting, not wholly
independent of baseline fitting.

### Calibration changes probability levels, not the AP ranking here

Both saved calibrators apply a sigmoid to a linear function of the clipped raw
probability's logit. Baseline slope/intercept are 0.608540332 / −4.105736648;
selected values are 0.847414442 / −2.865839937. Both slopes are positive.
Recomputed raw and calibrated AP agree in every reported population. Thus the
AP regression lies in the saved estimators' ranking, rather than this calibration
mapping. Probability levels and decisions at a numeric threshold do change.

On the matched rows the baseline predicts 25.686 expected positives (mean
probability 1.4191%) against 48 observed, while the selected model predicts
81.524 (4.5041%). The selected mean positive probability is 13.5656%, but its
mean negative probability is also higher, 4.2572% versus baseline 1.3586%.
Raw Brier improves from 0.198282 to 0.161966; after calibration the ordering
reverses, 0.025026 versus 0.025725. The uncalibrated class-balanced classifier's
scores should not be interpreted as reliable event probabilities.

| Selected-model population | Observed rate | Mean probability | Raw Brier | Calibrated Brier |
| --- | ---: | ---: | ---: | ---: |
| Its calibration window | 3.3149% | 3.3110% | 0.118408 | 0.028813 |
| Temporal validation | 1.7210% | 1.9833% | 0.072957 | 0.016324 |
| Geographic validation | 0.8152% | 1.8172% | 0.067923 | 0.008442 |
| Expanded temporal test | 2.4555% | 3.6643% | 0.133431 | 0.023407 |
| Matched test | 2.6519% | 4.5041% | 0.161966 | 0.025725 |
| Independent-geography test | 0.4144% | 0.5303% | 0.012514 | 0.004377 |

The calibration-window agreement is in-sample for the calibrator. The baseline
calibrator was fitted to only four positives at 0.5961% prevalence, whereas the
selected calibrator saw 108 positives at 3.3149%. This makes calibration-period
shift a plausible contributor to probability bias, but slopes/intercepts from
different underlying models do not isolate a causal prior correction.

For context, the selected training-prevalence constant has matched Brier
0.025819672; the matched empirical-prevalence constant has 0.025816062.
The latter uses final-test labels and is descriptive, not an available forecast.
The selected model only narrowly beats those constants. On geographic testing,
its Brier 0.004377223 is worse than always predicting zero (0.004143646), despite
ROC AUC 0.937587 and AP 0.040781. Rare labels dominate these comparisons.

The ten fixed calibration bins, including empty bins, are retained in JSON.
For the selected model's nonempty matched bins:

| Probability bin | Rows | Mean predicted probability | Observed positive fraction |
| --- | ---: | ---: | ---: |
| [0, 0.1) | 1,513 | 0.019505 | 18/1,513 = 0.011897 |
| [0.1, 0.2) | 197 | 0.141429 | 17/197 = 0.086294 |
| [0.2, 0.3) | 94 | 0.235062 | 12/94 = 0.127660 |
| [0.3, 0.4) | 5 | 0.320404 | 0/5 |
| [0.4, 0.5) | 1 | 0.452907 | 1/1 |

Matched equal-width ECE rises from 0.012328 to 0.019126. These bins are coarse
and the upper bins have very few observations. Small ECE on geographic testing
(0.001159) does not establish calibration or successful positive detection.

### Validation criterion instability

Selection averages temporal-validation and geographic-validation AP, using
Brier as a tie breaker. The 368-row, three-positive geographic set receives
the same 50% weight as 3,312 rows and 57 positives in temporal validation.
All eight saved candidates are shown below; none was retrained or reselected.

| Candidate | Temporal AP | Geographic AP | Saved selection score |
| --- | ---: | ---: | ---: |
| **lightgbm_enhanced** | 0.090536431 | 0.074576695 | **0.082556563** |
| xgboost_enhanced | 0.102752226 | 0.032547881 | 0.067650053 |
| lightgbm_base | 0.088604324 | 0.046254174 | 0.067429249 |
| xgboost_base | 0.096591632 | 0.033012399 | 0.064802015 |
| logistic_unweighted | 0.088645613 | 0.031093358 | 0.059869486 |
| logistic_base | 0.083639773 | 0.026562518 | 0.055101146 |
| logistic_enhanced | 0.077204270 | 0.026709710 | 0.051956990 |
| frozen_baseline | 0.057249528 | 0.039131722 | 0.048190625 |

Temporal AP alone would favor `xgboost_enhanced`. Weighting the two saved APs
by row count also favors it (0.095731791 versus selected 0.088940458). A weighted
mean of APs is not pooled AP and does not resolve dependence or uncertainty.
These illustrate criterion sensitivity, not a retrospective replacement choice.

The selected model's geographic positives are Adams County on July 8, July 24
and August 2, 2025. Ranks among 368 rows are 91, 6 and 83; probabilities are
0.019573, 0.144166 and 0.023356. July 24 contributes 0.055555556 of geographic
AP 0.074576695 (74.5%). Removing it reduces AP to 0.017208672; removing either
other positive produces 0.095381526 or 0.094444444. The baseline's corresponding
range is 0.027346637–0.036309524. Positive deletion changes prevalence; these
are sensitivity values, not confidence intervals.

Only the selected and frozen baseline fitted objects are retained. The other
six new candidate objects and their per-row predictions are unavailable. Saved
validation scores establish criterion dependence but cannot support a paired
bootstrap of full selection or causal feature-ablation claims. No missing model
was refitted to fill that evidential gap.

### Threshold tradeoffs under already saved policies

These policies were chosen on validation in the saved experiment. The budget
policy maximizes validation F1 subject to at most 20% alerts and at least 30%
recall. The diagnostic applies the saved thresholds without searching tests.

| Model / policy | Threshold | Matched TN / FP / FN / TP | Precision | Recall | Alert rate |
| --- | ---: | --- | ---: | ---: | ---: |
| Baseline original | 0.013141327 | 1,028 / 734 / 9 / 39 | 5.0453% | 81.25% | 42.7072% |
| Selected F1 | 0.116666886 | 1,542 / 220 / 21 / 27 | 10.9312% | 56.25% | 13.6464% |
| Selected budget | 0.075679783 | 1,410 / 352 / 16 / 32 | 8.3333% | 66.6667% | 21.2155% |
| Selected F2 | 0.039111262 | 1,224 / 538 / 9 / 39 | 6.7591% | 81.25% | 31.8785% |

The selected policy reduces false alerts by 382 and detections by seven. Its
saved F2 policy has the same detection count as baseline with 196 fewer false
alerts; this does not mean it detects the same individual days. It does not
resolve the AP regression or justify a post-test policy change.

Validation budget performance is 211 alerts (6.3708%) and 21/57 detections;
expanded temporal testing gives 556 alerts (17.0657%) and 49/80 detections.
The matched subset's alert rate exceeds 20%: a validation constraint is not a
guarantee on a different population. All three selected policies miss all three
geographic positives; their geographic false-positive counts are 3 (F1),
5 (budget) and 10 (F2). Lowering thresholds to catch inspected positives would
be test tuning.

The saved evaluation also includes a re-budgeted baseline threshold 0.012412747,
distinct from its original 0.013141327. It produces 774 matched false positives
and 39 detections, rather than 734 and 39. The headline comparison consistently
uses the original baseline threshold.

### Geography and season feature reliance

The eight base inputs are rain, temperature, humidity, gust, latitude, longitude,
season sine and season cosine. Enhanced inputs add maximum hourly rain, wet
hours, antecedent forecast rain over 72 hours, `rain * humidity / 100` and
`gust * temp / 100`. Rain is in mm, temperature in °C, humidity in percent,
and gust in km/h. These are forecast predictors, not observed NOAA measurements
at every storm location; construction is defined in `scripts/build_training.py`.

Saved-tree gain shares are 7.4557% for geography and 15.2628% for season. Gain
measures fitted split use, not independent predictive value. The predeclared
validation-only paired-column permutations yield:

| Validation set | Permuted pair | Mean AP drop | Range, 12 permutations | Mean Brier increase |
| --- | --- | ---: | --- | ---: |
| Temporal, 3,312 rows / 57 positives | Geography | +0.029419124 | +0.016475 to +0.035139 | +0.000527375 |
| Temporal | Season | +0.006138124 | −0.014127 to +0.024035 | +0.000109219 |
| Geographic, 368 rows / 3 positives | Geography | +0.026721183 | −0.025298 to +0.054158 | +0.000488479 |
| Geographic | Season | −0.034998845 | −0.277405 to +0.028259 | −0.000127469 |

Positive drop means perturbation hurt ranking. Geographic season permutations
improve AP on average, but the huge spread and three positives prevent a
reliable claim that removing season improves transfer. Pairing preserves valid
coordinates or sine/cosine combinations, while breaking relationships with
weather and date. These can be implausible inputs and are not causal ablations.
No final-test feature was perturbed and no feature-drop model was fitted.

## Three geographic misses and primary NOAA evidence

All 724 geographic labels reproduce from the local archive
`data/raw/StormEvents_details-ftp_v1.0_d2026_c20260918.csv.gz` identified in
`data/training-provenance.json`. Its SHA-256 is
`61c1dfebdfb24d561bad4528a31659791cf9a83d74663dd4e97b1e0088003f44`
(7,490,892 bytes). Retrieval is recorded as
`2026-10-17T01:03:35.568063+00:00`, October 16 in America/New_York.
JSON `raw_noaa.url` retains the original NOAA URL. The filename identifies a
September 18, 2026 revision; completeness of later revisions is not established.

The join uses county namespace `CZ_TYPE == 'C'`, state FIPS × 1,000 plus county
FIPS, and Thunderstorm Wind, Tornado or Hail. Source intervals overlap a UTC
target day when `start_utc < next_midnight` and `end_utc >= midnight`. The
existing normalizer uses the recorded fixed `PST-8` / `MST-7` offsets.
Multiple records become one positive county-day. All 17 source records, IDs,
episode IDs, local/UTC times, source, magnitude/type, coordinates and complete
event/episode narratives are retained in JSON.

| Positive county-day | Raw score | Calibrated probability | Baseline probability | Selected rank / 724 | Records / episode |
| --- | ---: | ---: | ---: | ---: | --- |
| Los Angeles, 2026-02-16 | 0.138636604 | **0.011964458** | 0.012784211 | 32 | 9 / 211215 |
| Los Angeles, 2026-02-18 | 0.099231457 | **0.008705462** | 0.007344440 | 57 | 7 / 211215 |
| Emery, 2026-05-13 | 0.106039749 | **0.009263211** | 0.007908461 | 52 | 1 / 212868 |

All fall below selected threshold 0.075679783 and original baseline threshold
0.013141327. The different re-budgeted baseline policy catches February 16.
Ranks are retrospective descriptions, not proposed alert cutoffs.

Explanations use `backend.predict.explain` on the complete calibrated probability
function: permutation SHAP, 16 saved background rows, seed 42 and four
permutations per explanation. The common base is **0.030183491**. All thirteen
contributions plus that base reconstruct each probability within 1e-10.
Below, **pp means percentage points**: JSON `probability_delta` multiplied by
100. These are approximate background-relative attributions, not log odds or
causal effects. Independent masking of correlated weather, interaction and
season inputs can construct unrealistic combinations. JSON percentiles are
inclusive (`fit <= value`) against all 6,174 fit rows and separately the 152
positive fit rows; every feature and outside-range flag is retained.

### Los Angeles County, California — February 16, 2026

Nine Thunderstorm Wind records match county FIPS **06037**: **1319694, 1319695,
1319697, 1319698, 1319700, 1319701, 1319702, 1319704, 1319705**, all episode
**211215**. Times span 18:43–20:05 UTC (10:43–12:05 PST). NOAA describes a
powerful winter storm with rain, mountain snow and convective outflow. Records
include measured mesonet gusts of 81 and 62 mph and damage reports around
Hollywood, San Fernando, South Los Angeles, San Pedro, Culver City and Reseda.
Narrative mph values differ in units and meaning from forecast gusts in km/h.

At representative point (34.18, −118.26), forecast rain is 28.5 mm (97.81st
fit percentile / 92.11th positive-fit percentile), maximum hourly rain 16.2 mm
(99.34th / 96.71st), gust 54.7 km/h (88.66th / 78.29th), temperature 14.3°C
(13.95th / 0.66th), humidity 68.875%, wet hours 6 and antecedent rain 0.4 mm.
Temperature equals the positive-fit minimum; longitude is outside all fit rows.

Largest negative SHAP values are temperature **−2.6703 pp** and gust–temperature
**−1.0958 pp**. Rain–humidity **+1.0213 pp**, rain **+0.9337 pp**, maximum hourly
rain **+0.5462 pp** and season sine **+0.4322 pp** partly offset them. Geography
totals **−0.4397 pp**; season totals **+0.2605 pp**. Probability is **1.1964%**,
versus base 3.0183% and threshold 7.5680%.

Cold coastal winter convection may be poorly represented relative to warmer
positive fitting examples; western extrapolation and single-point county
aggregation may also matter. The feature/SHAP pattern supports these as plausible
hypotheses, not established causes or a measurement of forecast error.

### Los Angeles County, California — February 18, 2026

Seven Thunderstorm Wind records match county FIPS **06037**: **1319726, 1319727,
1319728, 1319729, 1319730, 1319731, 1319732**, again episode **211215**.
Times span 08:20–10:22 UTC (00:20–02:22 PST). Narratives report measured gusts
of 59–78 mph at Sombrero Road, Magic Mountain, Clear Creek, Barley Flats and
Catalina Island sites. This is another day of the same episode.

At the same forecast point, rain is 12.3 mm, temperature **12.4°C** (10.54th
fit percentile, below every positive fitting temperature), humidity 62.625%,
gust **76 km/h** (99.34th / 98.68th), maximum hourly rain 2.6 mm, wet hours 13,
and antecedent rain 44 mm. Longitude is again outside the fit range.

Temperature contributes **−2.7765 pp**, gust **−0.6771 pp**, antecedent rain
**−0.4391 pp**. Rain **+0.9297 pp**, rain–humidity **+0.7734 pp**, maximum hourly
rain **+0.4546 pp** and season sine **+0.3684 pp** partly offset them. Geography
totals **−0.3569 pp**, season **+0.0779 pp**. Final probability is **0.8705%**.
Negative gust SHAP is relative to the background and interactions; it does not
mean wind protects against storms. The fitted model is not constrained to
increase probability monotonically with gust.

The cold winter regime hypothesis remains plausible, as does mismatch between
the representative point and mountain/island measurements. Saved evidence
cannot isolate their roles, and the shared episode limits independent support.

### Emery County, Utah — May 13, 2026

Thunderstorm Wind event **1337243**, episode **212868**, county FIPS **49015**,
occurs at **21:20 UTC (14:20 MST)**. NOAA reports a mesonet observation of
61 mph outflow from nearby showers/thunderstorms southwest of Huntington.
The episode narrative describes a cold front and high-based convection.
Event coordinates are (39.31, −110.98); the forecast point is
(39.009025, −110.7211).

Forecast rain, maximum hourly rain, wet hours and antecedent rain are all zero.
Humidity **16.0833%** is below the entire fit range (minimum 17.875%) and the
positive fit range (minimum 34.2083%). Temperature is 29.5°C and gust 61.9 km/h
(94.83rd fit / 84.21st positive-fit percentile). Longitude is outside the fit
range. Zero rain occurs in 9.87% of fit positives and zero antecedent rain in
14.47%; dry inputs are not automatically incompatible with a storm label.

Humidity contributes **−2.6407 pp**, rain–humidity **−2.0978 pp**, antecedent
rain **−0.4166 pp**, maximum hourly rain **−0.3191 pp**. Temperature **+1.6629 pp**
and gust–temperature **+0.9823 pp** counteract them. Season contributes
**+0.8325 pp** and geography **+0.0760 pp**. Probability is **0.9263%**.
Geography therefore does not directly depress all three missed probabilities.

An arid convective outflow event potentially displaced from local rain may
transfer poorly from wetter/humid fitting associations. NOAA supports the event
type; feature extremes and SHAP support a model-association hypothesis. They do
not prove precipitation forecast error, a physical humidity effect, or that a
particular new feature or geographic fitting choice caused the miss.

## Evidence map, integrity and limitations

JSON is the numerical record; this document rounds it for reading.

| JSON key | Evidence |
| --- | --- |
| `input_sha256`, `raw_noaa`, `source_code_sha256`, `runtime` | Artifact identity and reproduction environment |
| `dataset`, `fit_feature_distributions` | Complete split/county/quarter counts and fitting distributions |
| `matched` | Metrics, paired intervals, all 48 positive ranks and pooled AP decomposition |
| `probabilities`, `calibrators` | Raw/calibrated metrics, probability means, constants and bin counts |
| `selection`, `threshold_policies` | All candidate scores, sensitivity and fixed policy outcomes |
| `reliance` | Saved split/gain use and every validation-only permutation replicate |
| `geographic_misses`, `geographic_label_audit` | Every feature, calibrated SHAP and all matching NOAA records |

Key frozen SHA-256 identities:

```text
current models/hazards.joblib
39330197a074f817000cc2fc4f187d32341ea81b240cb84ed2f0512dcb96efca
baseline models/hazards.joblib
54aca5de29f8b3c7197f1023b1f42f24a2c8b39cdccbac7cfc5c655913ac148a
current data/training.csv
dc4cebada4a66fbfc6e72cea7a66d71b36d4f022669c05a7924f3d069479d9a9
```

The script checks every baseline file against its manifest, records current
selection/evaluation/data identities, and rechecks protected inputs and the NOAA
archive after inference. Source helper hashes and package versions identify
implementation dependencies. Unrelated application tests are not protected
inputs. The baseline archive is the requested comparison evidence, not an
application compatibility path. No original artifact is overwritten.

Primary sources are the local saved `models/{selection,evaluation}.json`, model
objects, `data/{training.csv,regions.json,training-provenance.json}`, baseline
manifest/data/evaluation and raw NOAA archive. Definitions come from
`scripts/train.py`, `scripts/build_training.py`, `backend/model.py` and
`backend/predict.py`, identified by hash. No external secondary evidence was
needed. No key, paid service, provider call, retraining or test optimization is
part of this investigation.

Already inspected final tests cannot provide an unbiased estimate for changes
suggested here. County labels represent reports anywhere in a county rather
than observations at its forecast point. Reporting completeness, forecast error,
spatial aggregation, cross-day dependence, approximate SHAP with a small
background and unavailable candidate objects limit attribution. The validation
checks establish reliance and fragility, not a causal explanation or a better
replacement model. Future changes need a predeclared protocol and fresh,
untouched evaluation periods/geographies. The saved selection remains unchanged.
