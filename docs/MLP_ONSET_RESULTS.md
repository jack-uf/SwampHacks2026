# MLP hourly onset experiment: measured result

A plain multilayer perceptron ranks hourly high-flow onset better than XGBoost, LightGBM and the six-hour linear extrapolation at 24 hours, in the temporal and the reserved geographic partition at once, and it warns earlier. It still fails promotion, because it detects fewer events than the trivial linear baseline at the frozen alert budget. Nothing is integrated and no production model changed.

The [frozen protocol](MLP_ONSET_PROTOCOL.md), cohort and implementation were committed in `827c049` before any cohort observation was downloaded. Candidate selection was locked before the 2025 acquisition. Artifacts live under [`experiments/sprint-20261017-mlp-onset/`](../experiments/sprint-20261017-mlp-onset/); [`evaluation.json`](../experiments/sprint-20261017-mlp-onset/evaluation.json) owns the final scores and gate decisions.

## What was asked and what came back

The response experiment's pooled neural mixture detected 61 measured high-flow events against LightGBM's 44 on a daily target, then failed promotion on regional false alerts and an average-precision interval that included zero. This experiment asked whether that advantage reproduces on the hourly pipeline, which is a different target, a different dataset and a different alert cadence.

It partly does. The MLP reproduces the ranking advantage: at 24 hours `mlp_direct` reaches average precision .1487 temporal and .1555 geographic against the linear baseline's .1088 and .1177, with paired HUC4-cluster intervals of [.0117, .0877] and [.0111, .0666]. Both lower bounds sit above zero, which no earlier hourly model cleared in both partitions, and it beats XGBoost (.1427 and .1012) and LightGBM (.1213 and .0949) on the same rows. It also cuts the median first-warning lead the other way, to 12.0 hours temporal and 7.5 geographic against the baseline's 7.0 and 6.0.

It does not reproduce the detection advantage. At the frozen one-percent budget the linear baseline detects 37 of 69 temporal events and the MLP detects 32; geographically the baseline detects 33 and the MLP 30. The MLP produces slightly fewer false alerts (1.18 and 1.01 per 100 decisions against 1.16 and 1.05), so this is not an over-alerting baseline winning on volume. The baseline simply ranks the events it does catch earlier in its own ordering.

## Cohort, data and independence

The cohort is 28 CAMELS gauges in 14 HUC2 regions, one development and one geographic gauge per region in distinct HUC4 groups. Every station identifier is new to the project. HUC4-level separation was not available: all eligible CAMELS HUC4s appeared in an earlier reservation, so this cohort is fresh at the station level and shares regional climate with earlier studies. That limitation applies to every geographic claim here.

Source is USGS OGC v1 approved instantaneous discharge (parameter 00060, statistic 00011) with the same hourly binning, two-hour observation delay and eligibility rules as the [hourly onset experiment](SHIFT_HYDROLOGY_RESULTS.md). The target is an hourly instantaneous discharge above the gauge's 2000-2015 daily-mean 95th percentile after 24 quiet hours, which is high flow, not verified flooding.

Development materialization covered 267,426 rows and 25 gauges over 2020-2024, with 114,843 training rows before calibration. Four gauges returned unusable series and were excluded without replacement: 02143000, 08176900 and 08086290 for an unexpected statistic or unit, and 06280300 for no accepted instantaneous discharge in the final year.

| Partition | Gauges | Rows | Positives | Distinct events |
| --- | ---: | ---: | ---: | ---: |
| Temporal test | 11 | 39,732 | 475 | 69 |
| Geographic test | 13 | 39,501 | 464 | 69 |

Event coverage is 93 percent temporal and 91 percent geographic at 24 hours. The losses are three to four events behind missing history, two to three inside the quiet-interval rule, and no events lost to the observation buffer.

## Model comparison on the frozen 2025 test

Twenty-seven candidates were fit on 2020-2022 development rows, calibrated on 2023, and ranked on 2024 temporal plus geographic validation by equal-weight average precision. Thresholds target one false alert per 100 decisions on the pooled 2024 validation rows. Tests below use those frozen thresholds.

Average precision, best per row:

| Horizon | Best temporal | AP | Best geographic | AP | MLP direct | Baseline linear |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 6h | lightgbm | .1196 | xgboost | .0946 | .1102 / .0812 | .0641 / .0636 |
| 12h | xgboost | .1276 | baseline linear | .0962 | .1091 / .0930 | .0915 / .0962 |
| 24h | **mlp direct** | **.1487** | **mlp direct** | **.1555** | .1487 / .1555 | .1088 / .1177 |

The horizon-24 row is the result that matters. Every other horizon keeps the boosters and the extrapolation ahead of the MLP on at least one partition, and at 12 hours the linear baseline beats every learned model geographically.

## Architecture ablation

The frozen ablation compared three shapes on identical rows: a two-layer 64-unit perceptron (`mlp_direct`), the same network with two additive residual blocks (`mlp_residual`), and a shared trunk with one head per horizon (`mlp_multitask`). All three train on CPU with dropout 0.2, weight decay, gradient clipping and early stopping on the 2023 development year.

| Horizon | mlp_direct | mlp_residual | mlp_multitask |
| --- | ---: | ---: | ---: |
| 6h, validation AP | .1177 | .1116 | .1105 |
| 12h, validation AP | .1209 | .1087 | .1170 |
| 24h, validation AP | **.1448** | .1279 | .1374 |

Residual blocks and multi-horizon weight sharing both cost ranking accuracy here, and the gap widens with horizon. The plainest network wins, which is the same lesson the response study reached when catchment gating and trajectory loss reduced its final average precision. Extra depth or shared structure is not the bottleneck; the parameters are 18,177 for the residual shape against a few thousand for the direct one, and the larger model does less with them.

## Why the MLP fails promotion

Selection picked XGBoost at all three horizons on validation average precision, so the pre-registered gate applied to XGBoost, and XGBoost fails everywhere. The MLP's gate outcome is descriptive, since choosing it now would consume the test set.

Applied to `mlp_direct` at 24 hours against the same linear baseline, three of the thirteen criteria fail: the event-recall gain in each partition and one regional false-alert ceiling. HUC2 08 sits at 4.69 false alerts per 100 decisions in the temporal partition against a ceiling of three. Everything else passes, including both average-precision intervals, both Brier comparisons, every geographic regional ceiling, and the joint-lead criterion.

That failure is a property of the target, not a transfer accident. On the 2024 selection data the same ordering already holds: the linear baseline's event recall is .537 temporal and .544 geographic, against .522 and .441 for `mlp_direct` and .388 and .368 for XGBoost. A one-percent budget on an autocorrelated flow signal rewards whichever model puts persistent high water at the top of its ordering, and the frozen selection rule chooses on average precision, which does not measure that. The MLP is not losing an advantage the daily response experiment found; the hourly target's base structure gives a trivial extrapolation a recall floor that no ranker in this comparison beats.

Calibration is not the problem. The MLP's temporal reliability bins track closely (.013 predicted against .013 observed in the lowest bin, .142 against .131 in the next), and its Brier score beats the baseline in both partitions at 24 hours (.01846 against .01981 temporal, .01802 against .01953 geographic).

## Decision

No candidate passes the frozen promotion gate at any horizon, so the MLP stays experimental and PlanIT's served models, safety rules and 0-10 hazard index are untouched. The defensible claim is narrow: on this cohort the MLP is the strongest ranker at 24 hours in both partitions and warns earlier, and it is not strong enough at the operating point to replace a three-line extrapolation.

## Runtime and reproducibility

Fitting is cheap and stays far inside the 18 GB budget. Across three horizons the MLP family takes 13.3 seconds at 707 MB peak resident memory, LightGBM 1.8 seconds at 527 MB, XGBoost 1.1 seconds at 523 MB, and the baseline and logistic families under a second each. Final inference over the full 2025 frame takes .109 seconds for the MLP, .194 for LightGBM, .047 for XGBoost and .004 for the physical baselines. Versions are frozen in the family selection records; torch is 2.14.1.

Run from the repository root:

```sh
.venv/bin/python -m scripts.mlp_onset freeze
.venv/bin/python -m scripts.mlp_onset development
.venv/bin/python -m scripts.mlp_onset_worker fit baseline
.venv/bin/python -m scripts.mlp_onset_worker fit logistic
.venv/bin/python -m scripts.mlp_onset_worker fit lightgbm
.venv/bin/python -m scripts.mlp_onset_worker fit xgboost
.venv/bin/python -m scripts.mlp_onset_worker fit mlp
.venv/bin/python -m scripts.mlp_onset_worker select
.venv/bin/python -m scripts.mlp_onset test
.venv/bin/python -m scripts.mlp_onset_worker evaluate mlp
.venv/bin/python -m scripts.report_mlp_onset
.venv/bin/python -m pytest -q
```

Collection, fitting and evaluation need `USGS_API_KEY` in the environment or the ignored project `.env`, sent only in the official `X-Api-Key` header and never stored in provenance. Each command refuses to overwrite a frozen output. All 54 saved prediction files replay exactly from the frozen artifacts and match the canonical labels before scoring; the `*-replay-audit.json` files record the checksums.

## Strongest justified next step

The recall gate is the thing to attack, and it needs a fresh cohort because this one's 2025 outcomes are now inspected. Two changes are worth pre-registering: give the MLP the damped extrapolation as an input or an additive offset so it learns a correction instead of rediscovering flow persistence, which is the mechanism the historical-transition study used to raise its average precision; and choose the alert threshold against event recall under the false-alert budget rather than against row true positives, so selection measures the quantity the gate tests. Neither should be tried on this cohort.
