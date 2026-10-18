# Hourly high-flow onset experiment: measured result

Multiscale river dynamics do not beat a damped flow extrapolation at 2025 onset warning on unseen watersheds. The one statistically supported temporal improvement, the 12-hour residual LightGBM, buys 3.3 average-precision points and roughly half the false alerts by giving up 9 event-recall points and one hour of median lead. No horizon passes every frozen promotion gate, so nothing is integrated and production models are unchanged.

The [frozen protocol](SHIFT_HYDROLOGY_PROTOCOL.md) defines the target and the gates. Implementation and acquisition were committed across `9a911c0`, `de65e72`, `fc9de10` and `ae889f4`, the protocol and metadata cohort at `9435fa5`, and candidate selection and thresholds at `4b9e66a` (2026-10-17 22:39 EDT), before any 2025 observation was downloaded (first retrieval 2026-10-18 02:41 UTC). The experiment directory is [`experiments/sprint-20261017-shift-hydrology/`](../experiments/sprint-20261017-shift-hydrology/); [`evaluation.json`](../experiments/sprint-20261017-shift-hydrology/evaluation.json) owns the final scores and gate decisions.

## The question and the target

The earlier hydrology and analog studies scored persistent high water on a daily grid. This study asks a narrower operational question: does recent measured river behavior predict a river that is about to rise above its high-flow reference within 6, 12 or 24 hours, including at gauges the fit never saw?

An onset is an hourly instantaneous discharge above the gauge's 2000-2015 daily-mean 95th percentile, after 24 consecutive accepted hourly readings at or below it. This is high flow, not certified flooding. No historical flood-stage threshold or vertical-datum history was established for this cohort, so a stage-based flood label would repeat the [flood audit](FLOOD_SPRINT_RESULTS.md)'s unresolved validity problem. Discharge needs no datum conversion, so the percentile is a defensible substitute. Each candidate is compared on the same eligible decisions, so the reference level cancels out of the comparison.

## Data, geography and availability

Source is the USGS OGC v1 `continuous` collection, discharge parameter 00060, instantaneous statistic 00011, approved values with null qualifiers. Quarterly requests, two download workers and two model threads fit the 18 GB machine. Exact URLs, retrieval times and checksums are retained per request; compressed bytes stay local.

The cohort reuses the metadata registry from the earlier hydrology work for 21 development gauges, then reserves up to three gauges per HUC4 in the western regions 17 and 18 from the full CAMELS registry, screened against every station ID in prior experiments. Most western CAMELS HUC4s already appeared in an earlier reservation, so HUC2 17/18 is a new gauge population but not a new regional climate.

| Partition | Gauges with data | Accepted 2025 hours | 2025 decisions at 24h | Unique 24h events |
| --- | ---: | ---: | ---: | ---: |
| Development (temporal test) | 15 | - | 12,911 | 47 |
| Reserved western (geographic test) | 5 | - | 5,671 | 17 |

Development materialization reached 27 cohort gauges and 516,972 accepted hours with 465 raw certificates; two gauges returned no accepted instantaneous discharge and were excluded before any model fit. The 2025 collection covered 24 gauges and 163,792 accepted hours with 101 raw certificates. Three 2025 gauges returned no accepted readings and are recorded as exclusions, not data-quality passes: 03338780 and 06311000 in the development group, 11237500 in the western group.

Two limits belong with these counts. The reservations cover 27 slots, so four western and one development gauge carry an empty 2025 stream; a horizon-24 geographic test therefore has 17 eligible events against a frozen minimum of 20, and the promotion gate cannot pass there on population alone. Separately, the two-hour observation delay before each decision is an assumption. Approved USGS records are retrospective and the service's `last_modified` field is a refresh time, not a first-publication time, so no as-issued replay was available. The protocol records this rather than substituting a retrospective forecast.

## Model comparison on the frozen 2025 test

Ten candidate configurations were fit on 2021-2022, calibrated on 2023, and ranked on 2024 development plus geographic-validation gauges by equal-weight average precision. Selection then chose the strongest non-baseline and strongest physical baseline per horizon. Scores below are the frozen 2025 outcomes; `AP` is average precision, `FA/100` is false alerts per 100 eligible decisions.

| Horizon | Selected candidate | Temporal AP | Baseline AP | Temporal FA/100 | Baseline FA/100 | Geographic AP (cand / base) | Geographic events (cand / base) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6h | xgboost-dynamics-6 | .088 | .101 | .94 | .70 | .043 / .042 | 2 / 3 of 17 |
| 12h | lightgbm-residual-12 | .139 | .107 | .48 | .94 | .045 / .053 | 5 / 4 of 17 |
| 24h | xgboost-dynamics-24 | .074 | .107 | .73 | 1.03 | .078 / .085 | 2 / 7 of 17 |

The 12-hour residual model is the only candidate that clears a statistical gate. Its paired HUC4-cluster AP difference on the temporal test is +.033 with a 95% interval of [.003, .092], above zero. Every other AP comparison, including all three at 24 hours and both western comparisons, straddles zero.

What that gain costs is the reason to reject it. On the temporal test the residual model detects 16 of 46 eligible events against the baseline's 20, an 8.7 point recall drop whose paired interval tops out at 0, and its median lead on the sixteen events both models catch falls from 4.5 to 3.5 hours. The pooled event-recall interval across both partitions is [-.123, .049], which fails the frozen requirement that the lower bound exceed zero. Reducing false alerts from .94 to .48 per 100 decisions is real, but a warning service that misses four more events and warns an hour later is not the same product.

## Geographic generalization

The reserved western gauges resist the learned models. At 6 hours the XGBoost candidate and its linear baseline both land near .042 AP, with two and three detections out of 17. At 24 hours the baseline detects 7 of 17 while XGBoost detects 2, a reversal of the temporal ordering. With five gauges contributing to the geographic partition, the paired bootstrap sets `weak_clusters` on every horizon, so these intervals describe the same five catchments sampled with replacement rather than an estimate of western performance in general. The honest reading is that the west neither confirms nor rescues the learned models; it simply lacks the population to decide.

## Calibration and false alerts

Calibration does not transfer. The 12-hour residual model's .1-.2 temporal bin holds 50 decisions at a mean predicted probability of .129 and an observed onset rate of .32, while its .3-.4 bin holds 3 decisions at .371 predicted and none observed. The 24-hour XGBoost western .1-.2 bin happens to agree closely (.131 predicted, .132 observed) on 121 decisions, but the surrounding bins are thin. Low Brier loss in this rare-event population is not evidence of usable probabilities.

Regional false-alert ceilings also fail where the population is honest about it. In the development partition the 6-hour candidate exceeds three false alerts per 100 decisions in HUC2 08 (3.2). No western region exceeds the ceiling, but that reflects five gauges rather than reliable regional control.

## Failed hypotheses and decision

Three expectations did not survive contact with the frozen test. Recent multiscale dynamics did not beat a six-hour damped trend projection at any horizon where the population permitted a decision. The lagged-flow LightGBM variants that look strongest in the development stage (`lightgbm-delayed`, AP .19-.25 at 24 hours) collapsed on the final year to .019-.041, the clearest sign that a development-stage winner was fitting the 2021-2024 regimes. And the western cohort never reached 20 eligible events per horizon, so the population gate closed before any model could claim transfer.

Every horizon fails at least one gate. The 12-hour candidate fails event-recall gain, pooled event-recall uncertainty, joint lead, and the geographic population and average-precision gates. The 6- and 24-hour candidates fail average precision, event-recall gain, pooled uncertainty, and the geographic gates as well. There is no operational improvement claim, no integration into PlanIT's 0-10 hazard index, and no change to production models or safety rules. The selected artifacts remain experimental.

## Runtime and reproducibility

Fitting is cheap. Across all configurations the baseline family takes .78 s, logistic .31 s, LightGBM 5.0 s and XGBoost .96 s, each at under 500 MB peak RSS. Final inference over the complete 2025 frame is .017 s for baseline, .008 s for logistic, .53 s for LightGBM and .034 s for XGBoost. These figures exclude source collection, which dominates wall time through per-page rate-limit waits.

Run from the repository root:

```sh
.venv/bin/python -m scripts.shift_hydrology freeze
.venv/bin/python -m scripts.shift_hydrology development
.venv/bin/python -m scripts.shift_hydrology_worker fit baseline
.venv/bin/python -m scripts.shift_hydrology_worker fit logistic
.venv/bin/python -m scripts.shift_hydrology_worker fit lightgbm
.venv/bin/python -m scripts.shift_hydrology_worker fit xgboost
.venv/bin/python -m scripts.shift_hydrology_worker select
.venv/bin/python -m scripts.shift_hydrology test
.venv/bin/python -m scripts.shift_hydrology_worker evaluate baseline
.venv/bin/python -m scripts.shift_hydrology_worker evaluate logistic
.venv/bin/python -m scripts.shift_hydrology_worker evaluate lightgbm
.venv/bin/python -m scripts.shift_hydrology_worker evaluate xgboost
.venv/bin/python -m scripts.report_shift_hydrology
.venv/bin/python -m pytest -q
```

Each command refuses to overwrite a frozen output, so a rerun needs a fresh output directory rather than deleting retained evidence. The fit and evaluation commands need `USGS_API_KEY` in the environment or the ignored project `.env`, sent only in the official `X-Api-Key` header and never stored in provenance.

All 56 saved prediction files were replayed exactly from the frozen artifacts in isolated family processes, and their probabilities match to within 1e-15; the `*-replay-audit.json` files record the checksums. Saved predictions are additionally checked against the canonical test labels before scoring, so a valid checksum cannot hide a relabeled row. The full suite passes 327 tests with four warnings.

## Strongest justified next step

The population, not the architecture, is the binding constraint in the west. A defensible geographic-transfer claim needs more reserved gauges with overlapping 2025 records, screened against all prior registries, before another model family is added. Until then the useful result is this negative one: at a two-hour observation delay, six-hour-trend extrapolation is a strong and cheap baseline for hourly high-flow onset, and the extra capacity in the tree and residual models does not pay for the events it loses.
