# Six-hour high-flow forecasting research results

Broader training improves the development evidence for plain LightGBM, but event-recall uncertainty still prevents promotion. Four research cycles tested feature removal, physical crossing features, watershed weighting, conditional calibration and threshold season coverage. None supplies independent operational confirmation. Production models and deterministic safety rules remain unchanged.

The development reference detects more events with fewer false-alert decisions than the strongest calibrated physical comparator. Its pooled recall interval still includes zero. Accounting for shared weather also removes a positive geographic AP lower bound. More experiments on the inspected archive cannot resolve those confirmation limits.

The [protocol](HIGH_FLOW_RESEARCH_PROTOCOL.md) owns the hypotheses and timing rules. [`summary.json`](../experiments/high-flow-20261018/summary.json) owns common-row comparisons and paired uncertainty. [`decision.json`](../experiments/high-flow-20261018/decision.json) locks the development reference, comparator, thresholds, evidence hashes and non-promotion decision. The [source review](HIGH_FLOW_EVIDENCE_REVIEW.md) documents watershed independence and input-availability requirements.

## Regional failure diagnosis

The earlier HUC2 04 ceiling failure comes entirely from station 04127997, the Sturgeon River at Wolverine, Michigan. Its registry gives a drainage area of 497.3 square kilometres and gauge elevation of 232.7 metres. Those attributes alone do not establish snow, storage, land-cover or geology mechanisms. The study did not obtain independent measurements of those mechanisms.

[`diagnosis.json`](../experiments/high-flow-20261018/diagnosis.json) reproduces 28 false alerts among 797 eligible 2025 decisions, or 3.51 per hundred. April contributes nine and July eight. Only two precede a certified onset within twelve hours, and six within twenty-four hours. Three have a future six-hour maximum above the reference despite lacking a certified future onset. A high-flow maximum and a new onset are different labels under the quiet-period and observation-buffer rules.

Most false alerts therefore cannot be explained by a slightly early warning alone. [`huc04-false-alerts.csv`](../experiments/high-flow-20261018/huc04-false-alerts.csv) retains their probabilities, rising-flow features, future maxima and later-onset labels. Several predicted rises flatten before crossing the reference. This motivates the persistence hypothesis; it does not establish a causal watershed explanation.

The site's median normalized flow increases from .456 in 2024 to .478 in 2025, within earlier annual medians of .463 to .564. Annual onset prevalence also varies in the retained history. This is a distribution change, but it provides no evidence for a uniform national shift. Missing complete history and the target's quiet period make these eligible-decision distributions conditional samples.

## Experiment progression

Cycle freezes record every source, configuration, production hash, partition population and dependency version before fitting. All validation outcomes are 2024 observations already inspected during earlier research. The previously inspected 2025 outcomes enter the regional diagnosis only. Each threshold uses earlier labels, and every comparator receives identical rows and policies within its cycle.

| Cycle | Training and calibration | Threshold population | Finding |
| --- | --- | --- | --- |
| 1 | Latest cohort; train 2020-2022, calibrate first half of 2023 | Second half of 2023 | Removing coordinates or adding crossing features fails to improve both validation partitions. |
| 2 | Pool four retained cohorts with the same dates | Second half of 2023 | Expansion improves common-row ranking. Weighting and conditional calibration remain inconsistent. |
| 3 | Expanded pool; train 2020-2021, calibrate all of 2022 | Second half of 2023 | Full-year calibration does not remove threshold drift. Comparisons with cycle 2 also change fitting years. |
| 4 | Identical fitting and calibration to cycle 3 | All of 2023 | Thresholds transfer more conservatively. Plain LightGBM with headroom supplies the development reference. |

Cycle 1 trains on 45,144 decisions from fourteen gauges in fourteen HUC4 groups. Cycle 2 expands this to 136,590 decisions from forty-nine gauges in thirty-eight groups. Cycles 3 and 4 use 85,550 training decisions from the same gauge and HUC4 population, followed by 50,936 calibration decisions. Geography expands substantially within development; that expansion is not fresh confirmation.

The geographic partition reserves eleven HUC4 groups across the expanded pool, including sixteen gauges. Neither fitting nor probability calibration sees those groups. Threshold selection does use their 2023 observations, as geographic development evidence. Their 2024 results therefore measure transfer within this development design, not an untouched geographic test.

Cycle 2 initially included unfitted gauges in its temporal partition because earlier geographic cohorts lacked pre-2023 rows. Its frozen scores retain that mixed interpretation. The canonical partition requires temporal stations to have fitting history. Common-row rescoring isolates population effects, and cycles 3 and 4 use the corrected partition directly.

## Controlled ablations

Expansion improves original-feature AP on identical observations: temporal AP rises from .1283 to .1571 and geographic AP from .1428 to .1696 between cycles 1 and 2. These comparisons use the same 109 temporal and 60 geographic events. Calibration and threshold labels also expand, so the gain belongs to the broader pipeline. This experiment cannot attribute it to training volume alone.

Coordinate removal does not help. On those common rows in cycle 2, AP falls to .1282 temporally and .1321 geographically. Adding hydrologic features recovers some temporal ranking but leaves geographic AP at .1321. The features include distance to the correct `log(2)` reference, extrapolated crossing margins, estimated crossing time and rise persistence. Their causal construction is tested against changes to future labels.

Watershed weighting and conditional calibration fail to improve both ranking and alert control consistently. Cycle 2's conditional model detects more events but exceeds the geographic pooled ceiling even under the stricter headroom rule. The worst regional burden reaches 3.37 false alerts per hundred in HUC2 17. A regional correction can move the failure elsewhere; no region-specific threshold was fitted.

Cycles 3 and 4 isolate threshold season coverage because model and calibrator inputs are identical. At the ceiling policy, plain LightGBM's temporal burden falls from 1.514 to 1.452 and geographic burden from 1.599 to 1.515. It loses four temporal detections and keeps the same geographic detections. Full-year selection reduces drift but still misses the geographic ceiling. The stricter headroom policy supplies additional protection at a cost in recall.

All model families receive scalar and conditional calibration where applicable. Both threshold policies apply to every comparator. The ordinary ceiling selects under 1.5 pooled and three regional false alerts per hundred. Headroom selects under one pooled and two regional false alerts. Final acceptance ceilings stay unchanged.

## Development reference and operational comparison

The retained reference is plain LightGBM with the original features, 250 trees, learning rate .04, fifteen leaves, depth four, minimum leaf size eighty and regularization five. It uses a standardized scalar logistic calibrator. Its inclusive probability threshold is .10851271591094244. This is an experimental artifact, not a served model.

The table uses cycle 4's corrected populations. Each policy chooses its own threshold on identical 2023 observations under identical headroom budgets. Achieved 2024 burdens can differ. Conditional-linear extrapolation has the highest mean validation AP among the calibrated physical baselines and is the strongest comparator for this assessment.

| Model | Temporal events / 306 | Temporal AP | Temporal FA/100 | Geographic events / 102 | Geographic AP | Geographic FA/100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Damped, scalar calibration | 64 | .0730 | .615 | 21 | .0881 | .516 |
| Linear, scalar calibration | 76 | .0832 | .717 | 28 | .0964 | .662 |
| Linear, conditional calibration | 84 | .0840 | .982 | 33 | .0964 | 1.060 |
| LightGBM, scalar calibration | 103 | .1395 | .818 | 40 | .1739 | .909 |

Against the strongest comparator, event recall improves by 6.2 temporal and 6.9 geographic percentage points. Brier loss improves from .00570 to .00538 temporally and .00553 to .00512 geographically. Jointly detected events have a two-hour median lead for both methods. The model therefore adds detections without establishing longer warning lead.

The candidate's worst temporal regional burden is 2.24, in HUC2 06; its worst geographic regional burden is 1.49, in HUC2 11. HUC2 04 measures 1.45 temporally and 1.30 geographically. These are 2024 development results on expanded populations. They cannot be read as a demonstrated repair of the original 2025 gauge failure.

False-warning episodes also deserve attention. The candidate has 292 temporal and 115 geographic episodes, compared with 281 and 110 for conditional-linear extrapolation. Fewer false-alert decisions do not imply fewer interruptions. Episode separation remains the inherited greater-than-24-hour rule; this is not a measured crew-notification policy.

## Uncertainty and promotion decision

The required paired HUC4 bootstrap uses 500 draws. Against conditional-linear extrapolation, the pooled event-recall gain is 6.4 percentage points with a 95% interval of [-.3, 13.2] points. Geographic recall spans [-4.2, 24.7] points. The positive point estimates do not establish a reliable detection advantage.

The [shared-weather sensitivity](../experiments/high-flow-20261018/shared-weather-sensitivity.json) samples HUC4 groups and seven-day UTC blocks with paired product weights. The pooled recall interval widens to [-2.7, 15.4] points. Geographic AP difference spans [-.033, .230]. These exploratory percentile intervals assume exchangeable drainage and weekly blocks; they do not estimate post-selection error or every possible weather dependence.

The existing gate function checks fifteen individual criteria, despite the earlier prose saying thirteen. The retained earlier result passes thirteen and fails two. The new development reference passes fourteen exploratory checks and fails pooled event-recall uncertainty. This accounting corrects the reported denominator and leaves every criterion intact. A development gate count grants no promotion eligibility.

Geographic AP remains a hard gate. Against scalar extrapolation, the candidate's recall gains accompany higher realized alert rates. Against conditional extrapolation, the candidate has a better point tradeoff but uncertain recall improvement. Neither comparison justifies weakening evaluation to secure a promotion.

[`independence-audit.json`](../experiments/high-flow-20261018/independence-audit.json) finds 208 unused eligible station IDs in the retained metadata pool, but none in a previously unexposed HUC4. A reused HUC4 could contain disjoint contributing areas. No complete polygon audit against all prior exposure exists for this study, so those stations cannot supply the required independent confirmation claim here.

The stopping decision is non-promotion. Further tuning on this archive would reuse exposed outcomes while leaving the recall uncertainty and operational evidence gaps unresolved. No fresh final cohort or outcome period was frozen or evaluated. The retained pipeline lock is a development reference only.

The next research owner needs a broader USGS metadata cohort or a complete polygon-based independence assessment, followed by an untouched outcome period. Confirmation must freeze the full pipeline and every gate before inspecting those outcomes. Historical receipt evidence or prospective capture must certify the actual values available at issuance. A site-to-gauge applicability assessment remains necessary before integration. High-flow percentile onset also remains distinct from verified flooding and field-safety decisions.

## Reproduction and verification

The implementation lives in `scripts/high_flow_research.py` and the audit in `scripts/report_high_flow_research.py`. Raw inputs remain in their original experiment directories with request manifests. Cycle freezes pin dataset hashes and source-code hashes. The fitted joblib artifacts include the models and calibrators; saved predictions preserve station, HUC, decision time and event identity.

The following commands ran from the repository root. Output commands refuse overwrite. For reconstruction, retain the committed experiment directory separately and run in a disposable checkout with an empty `experiments/high-flow-20261018/`. Historical cycle 2 scores require the earlier source revision recorded in its freeze; current code applies the corrected temporal partition. The common-row audit reproduces their comparable results without rewriting historical evidence.

```sh
.venv/bin/python -m scripts.high_flow_research diagnose
.venv/bin/python -m scripts.high_flow_research run --cycle 1
.venv/bin/python -m scripts.high_flow_research run --cycle 2
.venv/bin/python -m scripts.high_flow_research run --cycle 3
.venv/bin/python -m scripts.high_flow_research run --cycle 4
.venv/bin/python -m scripts.report_high_flow_research
.venv/bin/python -m scripts.report_high_flow_research sensitivity
.venv/bin/python -m scripts.report_high_flow_research independence > experiments/high-flow-20261018/independence-audit.json
.venv/bin/python -m scripts.report_high_flow_research decision
.venv/bin/python -m pytest -q
```

All eighty-six prediction files replay from their fitted artifacts, with maximum absolute probability difference 1.12e-16. The audit matches their identities and labels to retained input rows and records prediction checksums. Boundary tests cover threshold ties, zero-alert selection, causal features, temporal purges, fitted-station membership, common-row label agreement and paired spatial-temporal resampling. Production hashes match each cycle's frozen record.

The full Python suite passes 351 tests with four warnings from existing SHAP deprecations and a feature-name warning. The frontend was not changed or rebuilt. The research does not validate live telemetry receipt, forecast-weather availability, field performance or site applicability.
