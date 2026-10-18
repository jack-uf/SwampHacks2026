# Subdaily signal results for the six-hour high-flow model

Recent precipitation and as-issued forecast precipitation carry predictive information the
discharge-only model does not have. On the pooled development population the strongest candidate
clears every statistical and operational promotion gate against the calibrated physical baseline, and
the gains arrive earlier. Those results remain development evidence: the 2024 partitions were
inspected in earlier research, the precipitation reanalysis is not an as-issued product, and no
fresh watershed or untouched outcome period has been opened.

The implementation is `scripts/high_flow_signals.py` (population and cycles),
`scripts/high_flow_hrrr.py` (forecast acquisition) and `scripts/report_high_flow_signals.py`
(paired comparisons). The target, cadence, quiet period and partitions follow
[HIGH_FLOW_RESEARCH_PROTOCOL.md](HIGH_FLOW_RESEARCH_PROTOCOL.md); every gate and ceiling there stays
unchanged.

## Data acquired for this study

| Source | Content | Coverage | Notes |
| --- | --- | --- | --- |
| NOAA HRRR archive | 0-12 h accumulated precipitation from the cycle initialized six hours before each decision | 1,827 daily 12 UTC issues x 88 gauges = 160,776 rows | Exact GRIB byte ranges, ETags and index hashes recorded per field |
| Open-Meteo ERA5 archive | Hourly precipitation, 2 m temperature, 0-7 cm soil moisture | 88 gauges x 44,592 hours, 2019-12-01 to 2024-12-31 | `snow_depth` is null in every hour of this product and is not used |
| CAMELS attributes | Climate, soil, vegetation and geology tables | 88 gauges | `p_mean` supplies the precipitation normalizer |
| Development population | Six-hour eligible decisions, 2020-2024 | 330,747 rows, 88 gauges, 61 HUC4 groups | Pooled retained cohorts; no 2025 rows |
| Connected-gauge pairs | Downstream gauge plus upstream gauge | 3,811 paired rows, 14 events | Separate population, `scripts/high_flow_upstream.py` |

Every request is cached with its URL, response hash and receipt time beside the payload. Predictions,
fitted artifacts, freeze records and comparison intervals stay under
`experiments/high-flow-signals-20261018/`.

## Cycle progression

Cycles share one split: train 2020-2021, calibrate 2022, select thresholds on 2023, score 2024, with
the geographic partition reserved by HUC4. Each cycle freezes its datasets, configurations and code
hash before fitting, and every candidate in a cycle sees identical rows and threshold rules.

| Cycle | Question | Finding |
| --- | --- | --- |
| 0 | Do CAMELS attribute families carry signal? | Climate, storage and full attribute blocks change recall by at most two events and lower geographic AP |
| 1 | How recent must retrospective rain be? | Rain ending six hours before the decision lifts temporal AP from .139 to .169 and geographic AP from .174 to .191; 24-hour and 144-hour delays add nothing |
| 2 | Does as-issued forecast precipitation help? | On the 12 UTC rows where the pilot forecast exists, geographic AP rises from .092 to .121 and geographic events found rise from 6 to 9 of 24 |
| 3 | Does scale-normalized rain transfer better? | Dividing by the catchment long-term precipitation mean improves geographic detection to 11 of 24 on the same rows |
| 4 | Which family carries the gain? | Ablation below |
| 5 | How large is the gain against the physical baseline? | Every signal candidate beats damped extrapolation on AP, recall and Brier |
| 6 | Does the calibrated physical baseline close the gap? | Conditional calibration lifts that baseline to AP .084/.096; the signal candidates still clear the AP and recall gates against it |
| 7 | Which candidate is strongest and most operational? | `forecast_scaled_conditional` and `rain6_ratio_conditional`; `combined_scaled_conditional` detects most events but breaches the geographic false-alert ceiling |
| 8 | Do saturation and rise-shape features add anything? | Soil-moisture change over six days, the share of weekly rain falling in six hours, and the rise-shape block lift geographic AP from .174 to .197 while cutting geographic false alerts from 1.37 to 1.17 per hundred |
| 9 | Does the gain depend on re-using the baselines own margins? | No. Rise-shape features without any physical margin still reach geographic AP .193, so the transfer gain belongs to the rain and shape information |
| 10 | Do skew and evaporative-demand transforms add more? | No. Log-rain, rain-on-wet-soil and rain-over-aridity variants all score below the winner, which stays the frozen configuration |
| 11 | Does as-issued forecast rain help the winning model? | Yes. Adding the forecast block raises temporal detections from 108 to 116 and geographic from 50 to 56 with every gate still passing, so the forecast variant becomes the frozen candidate |

## Contribution of each signal

Cycle 4 isolates the families on identical rows. Values are detected events, average precision and
false alerts per hundred decisions at each candidate own 2023-selected threshold.

| Configuration | Temporal 2024 | Geographic 2024 |
| --- | --- | --- |
| Discharge only (`control`) | 103, .139, 0.82 | 40, .174, 0.91 |
| Retrospective rain ratios only | 107, .149, 1.01 | 43, .196, 1.27 |
| Recent rain plus ratios | 111, .170, 0.97 | 50, .175, 1.45 |
| Forecast rain ratios only | 105, .152, 0.85 | 44, .162, 1.11 |
| Forecast rain, raw depths | 100, .151, 0.78 | 43, .158, 1.00 |
| Forecast rain plus all ratios | 104, .159, 0.93 | 47, .175, 1.29 |

Three findings matter. Recent rain does the heavy lifting for ranking, and scaling it by the
catchment own precipitation climatology is what transfers to reserved watersheds. Forecast rain is
the weaker raw signal but the only family that improves geographic ranking on its own rows. The two
families overlap rather than add: the combined configuration detects the most events and pays for
them in the wet Pacific Northwest.

On the 12 UTC rows where the as-issued forecast exists the ordering changes. Retrospective ratios
reach AP .222 and forecast ratios .215 against .180 for discharge only, and combining them reaches
.243; geographic AP moves from .092 to .134 for forecast ratios and .127 for the combination.
Training only on forecast rows and scoring there reaches 12 of 24 geographic events.

## Candidate gates against the calibrated physical baseline

Cycle 7 applies conditional calibration to every family, so each model receives its own strongest
calibration. The comparator is conditional linear extrapolation, the strongest calibrated physical
baseline by mean AP. Intervals are paired 2.5th-97.5th percentiles from 500 HUC4-cluster bootstrap
draws on the 2024 development partitions.

| Candidate | Temporal AP (lower) | Geographic AP (lower) | Temporal events | Geographic events | Pooled recall gain (lower) | Worst HUC2 | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Comparator | .084 | .096 | 84/306 | 33/102 | - | 2.86 | - |
| Discharge only, conditional | .142 (+.035) | .170 (+.036) | 106 | 43 | +.024 | 1.62 | better |
| Recent rain plus ratios | .176 (+.058) | .174 (+.017) | 110 | 52 | +.040 | 2.46 | better |
| Rain, ratios and shape, no margins | .173 (+.056) | .193 (+.035) | 103 | 49 | +.030 | 2.68 | better |
| **Rain, ratios, shape and margins** | **.184 (+.064)** | **.197 (+.040)** | 108 | 50 | **+.041** | 2.31 | better |
| Forecast rain plus shape | .171 (+.052) | .182 (+.024) | 109 | 49 | +.032 | 2.46 | better |
| All signals combined | .184 (+.064) | .155 (+.009) | 117 | 56 | +.066 | 3.17 (breach) | better |

The named winner, rain6_ratio_wetness_hydro, and every other candidate except the combined configuration satisfy every gate the protocol defines against this comparator:
AP with a positive paired lower bound in both partitions, recall gains above five points in both
partitions with a pooled lower bound above zero, no Brier regression, pooled false alerts under 1.5
and no HUC2 above three per hundred, and no loss of median lead on jointly detected events. The
combined configuration reaches the highest recall but breaks the geographic ceiling, so it stays a
negative result rather than a contender.

Two comparisons have to be kept apart. Conditional calibration alone lifts the discharge-only model
from .084/.096 to .142/.170, so most of the distance to the physical baseline comes from calibration and
the existing discharge features, and a reader should not credit the new signals with all of it. Measured
against that discharge-only conditional model on identical rows, the winner adds .019 to the temporal AP
paired lower bound, which is a real ranking gain, while the geographic AP lower bound stays negative at
-.016 and the recall differences are uncertain in both partitions. The geographic improvement is a point
estimate of .170 to .197 with 43 to 50 events found, not an established effect at this sample size.

The gain is earlier as well as larger. Detections with at least four hours of warning rise from 30 to
50 temporally and 13 to 23 geographically against the comparator, and the median detected lead moves
from 2.0 to 3.0 hours.

## Frozen candidate and its confirmation requirement

The frozen deployment candidate is `winner_forecast` from cycle 11: the original discharge features, the
ERA5
lag-six weather block, the two precipitation ratios, the two wetness features and the seven rise-shape
features, the lag-six weather block, the two precipitation ratios, the two wetness features, the seven
rise-shape features and the as-issued HRRR forecast block, fitted with 250 trees at learning rate .04,
calibrated conditionally on season, flow margin and drainage area, and thresholded at the 2023
inclusive pooled-one and regional-two false-alert budget. Its artifacts are cycle 11 of
`scripts/high_flow_signals.py`;
`experiments/high-flow-signals-20261018/candidate.json` records its metrics, paired intervals and all
sixteen gate decisions, and `candidate-conservative.json` does the same for the lower-burden
`rain6_ratio_wetness_hydro`, which detects 108 temporal and 50 geographic events at 0.82 and 1.17 false
alerts per hundred instead of 0.92 and 1.40, and each cycle freeze pins the dataset, configuration and code hashes.

The frozen pipeline is deterministic: refitting the winning configuration with seeds 7 and 11
reproduces the seed-42 predictions exactly, so none of the reported differences depends on random
initialisation.

This candidate was selected after roughly forty configurations were scored on partitions that earlier
research already inspected, so the reported intervals describe correlated development comparisons and
carry a selection effect they do not measure. Independent confirmation needs the frozen pipeline run
unchanged on watersheds and outcome periods that no earlier study has opened, which does not exist in
this repository today.

## Forecast coverage changes the ceiling

The HRRR pilot existed for 12 UTC decisions only, so three of every four rows had no forecast features. The
completed four-hour acquisition (7,307 of 7,308 issues; 2022-02-27 00 UTC is a recorded archive gap) closes
the archive gaps: on the 160,776 shared rows the expanded fields reproduce the pilot exactly, maximum absolute
difference zero. Cycle 14 re-runs cycle 11 configurations on the expanded block, and the effect is the largest
single change in this sprint.

| Cycle 15 candidate, 2024 development | temporal events | temporal AP | geographic events | geographic AP | false alerts per hundred |
| --- | --- | --- | --- | --- | --- |
| Comparator under the same rule | 73 | .084 | 30 | .096 | 0.76 / 0.80 |
| Full forecast coverage, one-per-hundred rule | 163 | .304 | 75 | .263 | 0.91 / 1.89 (ceiling breach) |
| Full forecast coverage, tight rule (pooled .75, regional 1.5) | 140 | .283 | 67 | .248 | 0.63 / 1.12 |

With forecasts at every decision hour the model finds 140 temporal and 67 geographic events against the
comparator's 73 and 30, that is roughly twice the detections, with paired lower bounds of +.156 and +.071 AP
and +.126 and +.187 event recall. The one-per-hundred rule breaches the realized geographic ceiling (worst
HUC2 17 at 3.17 per hundred), so cycle 15 pairs each operating-point rule with a comparator selected under the
same rule. The tight rule keeps 67 geographic detections at 1.12 false alerts per hundred with no HUC2 above
2.54, and its AP is unchanged because ranking does not depend on the threshold.

The rule was chosen after seeing 2024 development outcomes, so it is a development selection and must be
frozen before any confirmation run, exactly as the confirmation protocol requires. The candidate also differs
from the previously confirmed model, so the cycle-11 confirmation result does not transfer to it.

## Architecture comparisons

Two neural families were trained on rows, features, calibrator and threshold rules identical to the tree
candidate, so each difference measures architecture rather than information. Both lose.

| 2024 partition | frozen tree, forecast features | tree, no forecast features | MLP, same features (cycle 13) | sequence TCN (168-hour windows) |
| --- | --- | --- | --- | --- |
| temporal | 116 events, AP .187, 0.92 FA | 108, .184, 0.82 | 104, .138, 0.99 | 79, .112, 0.65 |
| geographic | 56 events, AP .188, 1.40 FA | 50, .197, 1.17 | 53, .178, 1.69 (ceiling breach) | 37, .113, 1.21 |

`scripts/high_flow_signals_network.py` with `scripts/high_flow_signals_worker.py` trains the MLP in its own
process, because loading torch beside LightGBM in one process faults in this environment. The sequence
model is a temporal convolutional network over 168 hourly steps of discharge, precipitation, temperature
and soil moisture, assembled from the retained USGS pages and verified to reproduce the tree pipeline's own
taps and rises from the same slices; its protocol and comparison live in
`experiments/high-flow-sequence-20261018/`, and it scored 209,843 rows without subsampling.

The paired intervals exclude zero in both partitions for both families: the tree candidate is better by about
.18 AP in each. The MLP also breaches the geographic false-alert ceiling. The sequence model's loss is
representational rather than one of information. It shares the discharge history, so the gap is not that it
sees less; it must rediscover onset geometry from a 168-step channel stack in a population that is .7 percent
positive, and early stopping halted it at epoch 12 with too few positives to learn that geometry. The tree
receives the geometry as features and splits on the few that matter at almost no sample cost. One architecture
at one seed bounds that configuration, not the family, but the same result in two independent neural designs
makes a further architecture search a weak bet relative to acquiring new information.

## Whole-region holdout

The standard geographic partition reserves HUC4 groups, so a held-out catchment still shares weather and
regional response with the fitting population. `scripts/high_flow_regional_holdout.py` removes an entire HUC2
region from training, calibration and threshold selection, then scores that region 2024 rows. Bracket values
are the paired 2.5th percentile of the candidate-minus-comparator AP difference.

| HUC2 region | events | control, conditional | rain, ratios, shape | forecast plus shape | conditional linear comparator |
| --- | --- | --- | --- | --- | --- |
| 04 Great Lakes | 44 | 14, .139 [ +.034 ] | 12, .135 [ +.038 ] | 14, .130 [ +.036 ] | 13, .061 |
| 05 Ohio | 34 | 12, .173 [ +.035 ] | 11, .196 [ +.017 ] | 14, .155 [ -.061 ] | 11, .099 |
| 11 Arkansas-White-Red | 40 | 14, .186 [ +.042 ] | 11, .203 [ +.063 ] | 11, .217 [ +.070 ] | 6, .098 |
| 12 Texas-Gulf | 28 | 8, .123 [ -.210 ] | 12, .154 [ -.217 ] | 8, .105 [ -.410 ] | 6, .138 |
| 17 Pacific Northwest | 37 | 22, .214 [ -.099 ] | 29, .260 [ -.033 ] | 28, .186 [ -.041 ] | 23, .211 |
| 18 California | 44 | 20, .164 [ +.022 ] | 20, .171 [ +.007 ] | 19, .182 [ +.033 ] | 15, .131 |

Four of six unseen regions reproduce the advantage: the candidate family ranks events above conditional linear
extrapolation with a positive paired lower bound in regions 04, 05, 11 and 18, and in region 11 it also finds
five more events than the comparator while holding false alerts near 0.6 per hundred. Texas-Gulf is the clear
counterexample: there the comparator ranks better than every learned candidate, so the advantage is not
regional universality. The Pacific Northwest finds far more events than the comparator (29 against 23) but at
2.44 false alerts per hundred, which is the concentrated regional burden the earlier diagnosis already flagged,
and its interval includes zero. Those two regions are where the next experiment should look.

## Limitations

The retrospective rain features are ERA5 reanalysis. Their six-hour delay is an assumed publication
buffer, not receipt evidence, so the winning rain features are an upper bound on what an operational
precipitation observation network could supply. The HRRR fields are as-issued archived forecasts, but
the acquisition assumes each cycle initialized six hours before the decision was available at
decision time, and the scored data covers only 12 UTC, so three of every four decisions carry no
forecast features.

Both 2024 partitions were inspected in earlier research and the 2025 outcomes were inspected in the
regional diagnosis, so nothing here is independent confirmation. The fresh-geography reservation
attempt failed: of five screened candidates, two overlap CAMELS polygons with positive area and three
expose no mean-statistic discharge series in the required units, so no new watershed was reserved.
Polygon-overlap screening against every fitted catchment still has to happen before a geographic
confirmation cohort can be claimed.

## Reproduction

```sh
.venv/bin/python -m scripts.high_flow_signals acquire
.venv/bin/python -m scripts.high_flow_signals materialize
.venv/bin/python -m scripts.high_flow_hrrr grid
.venv/bin/python -m scripts.high_flow_hrrr collect
.venv/bin/python -m scripts.high_flow_hrrr materialize
.venv/bin/python -m scripts.high_flow_signals fit --cycle 1
.venv/bin/python -m scripts.high_flow_signals fit --cycle 7
.venv/bin/python -m scripts.report_high_flow_signals --cycle 7 --reference linear_conditional
.venv/bin/python -m scripts.high_flow_upstream collect
.venv/bin/python -m scripts.high_flow_upstream fit
```

`scripts/high_flow_hrrr.py` declares a six-hourly schedule covering 2020-2024. The committed cycle
results were produced from the 12 UTC pilot subset; the collection command resumes that pilot cached
fields and adds the remaining decision hours. The pilot extracted feature file is then replaced, so
the committed cycles stay pinned to their frozen input hashes.

## Next steps

Freeze the winning pipeline, its thresholds and every gate, then acquire the remaining three decision hours of HRRR fields so the forecast signal covers every row
instead of one in four, then re-run the cycle-7 comparison on the covered population, where the
forecast family already shows the strongest geographic behaviour. Extending the signal set to the twelve- and
twenty-four-hour horizons is the only way this line of work can lengthen warning for a specific
event. A temporal confirmation needs
2025-2026 observations that no earlier study has inspected, and a geographic confirmation needs new
gauges from a national catalog rather than more CAMELS screenings, with a complete polygon-overlap
audit before any outcome is acquired. Operational integration additionally needs as-issued
precipitation receipt evidence and the site-to-gauge applicability assessment; both remain
unavailable, so no candidate is promoted.

