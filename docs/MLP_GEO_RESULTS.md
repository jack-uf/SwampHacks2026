# Geographic generalization study: measured result

> Gate-count correction from the retained `evaluation.json`: the six-hour result passes 13 of 15 individual gates. The narrative below counts 11 of 13. The same two failures remain. See [the follow-up research](HIGH_FLOW_RESEARCH_RESULTS.md) for development experiments and the unchanged promotion decision.

The 6-hour MLP does not deliver a credible operational improvement over simpler models across unseen watersheds, and the answer is now decisive rather than marginal. Under matched threshold rules on a third independent cohort, LightGBM detects 25 of 76 temporal events where the MLP detects 19, and the MLP's ceiling-rule variants collapse to 7 of 76 because its false alerts concentrate in one region and the regional ceiling forces it to a far higher threshold. The booster with the ceiling rule clears 11 of 13 promotion gates at 6 hours, the best result any candidate has reached, and it still fails two. Nothing is promoted and production is unchanged.

The [frozen protocol](MLP_GEO_PROTOCOL.md) and cohort were committed in `4aedc27` before any observation was acquired. Artifacts live under [`experiments/sprint-20261017-mlp-geo/`](../experiments/sprint-20261017-mlp-geo/); [`evaluation.json`](../experiments/sprint-20261017-mlp-geo/evaluation.json) owns the final scores and gate decisions.

## The flaw this experiment was built to test

The hybrid experiment reported that the 6-hour perceptron beat the linear extrapolation on temporal event recall. That comparison was not matched: the network used the ceiling rule while every comparator used the stricter row rule, so the network was scored at a more permissive operating point. This experiment gives every candidate family both rules.

A diagnostic on the already-inspected hybrid cohort, reported here as a re-analysis and not as fresh evidence, shows what the mismatch bought. At 6 hours with both models on the ceiling rule, LightGBM detects 25 temporal and 12 geographic events where the perceptron detects 21 and 11. The perceived advantage belonged to the rule, not the network.

## Cohort and populations

Every station identifier registered by any prior experiment was excluded before screening. The cohort froze 29 gauges across 18 HUC2 regions, one development gauge per region plus a second gauge as geographic test where a region had two in distinct HUC4 groups. Five gauges returned unusable series and were excluded without replacement (02112120, 03238500, 03504000, 06278300, 12388400), and a sixth fell out of the final year only (08171300), leaving fourteen development and ten geographic gauges.

Development materialization covered 279,870 rows with 135,216 training rows. The 2025 test held 76 eligible temporal events and 58 geographic events, with coverage of 93 and 89 percent. That is a materially larger and wider cohort than either predecessor: 18 regions against 12, and 135,216 training rows against 102,876.

All eligible CAMELS HUC4 groups have appeared in an earlier reservation, so the cohort is fresh at the station level only. No watershed-polygon screen was applied. This is the main threat to any geographic claim here and it is not resolved by this experiment.

## Model comparison on the frozen 2025 test

Selection chose LightGBM with the ceiling rule at 6 and 12 hours and XGBoost with it at 24 hours. The perceptron did not win selection at any horizon: its best validation average precision is .1389 at 24 hours against XGBoost's .1470, and .1212 at 6 hours against LightGBM's .1367.

| Horizon | Partition | Candidate | Candidate AP | Baseline AP | AP interval | Events (cand / base) | False alerts (cand / base) |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 6h | Temporal | lightgbm | .1416 | .0711 | [.0393, .0975] | 25 / 18 of 76 | .93 / .63 |
| 6h | Geographic | lightgbm | .1325 | .1389 | [-.1228, .0462] | 24 / 18 of 58 | 1.08 / .66 |
| 12h | Temporal | lightgbm | .1427 | .0756 | [.0349, .0938] | 25 / 21 of 76 | .78 / .82 |
| 12h | Geographic | lightgbm | .1785 | .1500 | [-.0411, .0772] | 28 / 25 of 58 | .95 / .78 |
| 24h | Temporal | xgboost | .1359 | .0940 | [.0079, .0672] | 23 / 30 of 77 | .72 / .91 |
| 24h | Geographic | xgboost | .1512 | .1520 | [-.0338, .0192] | 22 / 27 of 58 | .73 / .93 |

The 6-hour candidate gains 9.2 points of temporal event recall and 10.3 points geographically. The pooled event-recall interval is [.033, .173], which clears zero, and the pooled false-alert burden is under the ceiling at every horizon.

## Gate outcomes

| Horizon | Gates failed | Which |
| --- | ---: | --- |
| 6h | 2 of 13 | Temporal regional ceiling (HUC2 04 at 3.51 against three), geographic average-precision interval |
| 12h | 3 of 13 | Temporal regional ceiling, geographic average-precision interval, pooled event-recall interval |
| 24h | 5 of 13 | Temporal event gain and regional ceiling, geographic average-precision interval and event gain, pooled event-recall interval |

The two 6-hour failures are different in kind. The regional ceiling is a genuine defect: the candidate spends 3.51 false alerts per 100 decisions in HUC2 04 where the gate allows three, and every other region is inside the limit. The geographic average-precision failure is not a recall failure at all. The candidate detects 24 of 58 geographic events against the baseline's 18 and still loses on average precision, .1325 against .1389, because average precision rewards where the ranking places the events it does catch, not how many it catches. A gate that fails a model for ranking while its recall improves by ten points is measuring the wrong thing for this decision, and that is a finding about the gate, not about the model.

## Where the perceptron loses

The matched comparison isolates the failure, and the cause is regional.

| 6h model | Rule | Temporal events | Temporal false alerts | Geographic events |
| --- | --- | --- | --- | --- |
| mlp direct | row | 19 of 76 | .39 | 21 of 58 |
| mlp direct | ceiling | 7 of 76 | .14 | 8 of 58 |
| mlp hybrid | ceiling | 5 of 76 | .09 | 5 of 58 |
| baseline linear | row | 18 of 76 | .66 | 18 of 58 |
| lightgbm | ceiling | 25 of 76 | .93 | 24 of 58 |

The ceiling rule lifts LightGBM and it destroys the perceptron. The reason is visible on validation. At its row threshold, the perceptron's false-alert rate in HUC2 17 reaches 6.70 per 100 decisions where the gate allows three, while the linear baseline never exceeds 2.94 in any region. The regional constraint therefore forces the perceptron to a threshold of .392 instead of .251, which removes two thirds of its alerts and most of its detections. A model whose errors concentrate in one region cannot use the operational budget the gate offers, and that is a property the previous experiment could not see because its baseline was scored under the stricter rule.

The perceptron keeps the best temporal average precision of any candidate at 6 hours, .1531 against LightGBM's .1416, and it cannot convert that ranking into detections inside the regional limits. Ranking quality is not the binding constraint.

## Hypotheses

**H1, matched operating points, holds.** Under identical rules the perceptron does not beat the damped or linear baselines or the boosted models on event recall in both partitions. It does not win selection at any horizon, and at 24 hours under the row rule it detects 15 of 77 temporal events against the ceiling-rule baseline's 30.

**H2, geographic ranking, is not supported.** The perceptron's geographic average precision does not clear zero against the baseline, and at 6 hours the linear baseline actually scores higher geographically (.1389 against the perceptron's best geographic value).

Both frozen hypotheses were falsifiable by the gates and both came back negative. The earlier claim does not survive fair threshold selection.

## Failed and inconclusive hypotheses

The damped offset again failed. On this cohort the hybrid loses to the direct perceptron on validation at 6 and 12 hours (.1212 and .1181 against .1136 and .1171, the one place it wins marginally) and never wins selection at any horizon. That is now two independent cohorts where the physical offset did not help, which is enough to retire it.

The balanced loss was inconsistent rather than uniformly bad. It was the best perceptron variant at 12 and 24 hours on the test (18 of 76 and 22 of 77 temporal events against the direct network's 15 and 15) while ranking last on validation at 6 hours. It also happened to be the only perceptron whose ceiling variant did not collapse, because its probabilities are flatter. That is a curiosity, not a finding, and the cohort is now inspected.

The trailing exceedance features failed again, ranking behind the plain network on validation at all three horizons.

## Runtime and reproducibility

The full suite passes 345 tests. All saved prediction files replay exactly from the frozen artifacts, and the `*-replay-audit.json` files are written for each family. Collection, fitting and evaluation need `USGS_API_KEY` in the environment or the ignored project `.env`, sent only in the official `X-Api-Key` header and never stored in provenance.

## Decision

Nothing is promoted. Production models, safety rules and the served hazard index are unchanged. The perceptron line of work is closed: across three independent cohorts it never converted its ranking advantage into matched-rule event recall, and its false alerts are regionally concentrated in a way that costs it the operational budget.

The transferable result is the combination of LightGBM with the ceiling threshold rule, which reaches 11 of 13 gates at 6 hours and beats the linear baseline by 9.2 points of temporal and 10.3 points of geographic event recall. Two things block promotion, and neither needs a new model: the regional false-alert ceiling in one HUC2, and a geographic average-precision gate that the recall-improving model fails. The next honest step is a regional false-alert investigation on new data, and a decision about whether average precision is the right geographic criterion for a warning system whose value is detection.
