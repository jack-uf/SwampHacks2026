# Hybrid MLP onset experiment: measured result

The gain came from the operating point, not the architecture. Selecting the alert threshold inside the promotion gate's own false-alert ceilings recovers events for every network at every horizon and lets the MLP beat the linear extrapolation on temporal event recall at 6 and 24 hours. Adding the damped projection as a predictive offset, adding trailing exceedance-history features, and balancing the training loss each failed to improve ranking. The 6-hour candidate now clears ten of thirteen gates, and the three it misses are the geographic ranking interval, a one-point shortfall in geographic event recall, and the pooled recall interval. No horizon passes everything, so nothing is integrated and production models are unchanged.

The [frozen protocol](MLP_HYBRID_PROTOCOL.md) and cohort were committed in `58d5411` before any observation was acquired; two pre-fit amendments ([1](../experiments/sprint-20261017-mlp-hybrid/amendment-1.md), [2](../experiments/sprint-20261017-mlp-hybrid/amendment-2.md)) and the implementation were committed before the first fit. Artifacts live under [`experiments/sprint-20261017-mlp-hybrid/`](../experiments/sprint-20261017-mlp-hybrid/); [`evaluation.json`](../experiments/sprint-20261017-mlp-hybrid/evaluation.json) owns the final scores and gate decisions.

## The rule that mattered

The earlier experiment picked the alert threshold by maximizing row true positives under a self-imposed budget of one false alert per 100 decisions, then measured the candidate against a promotion gate whose ceilings are 1.5 false alerts per 100 decisions pooled and three per region. That self-imposed target was stricter than the gate. The ceiling rule takes the lowest validation threshold whose pooled rate stays under 1.5 and whose every region stays under three, which by monotonicity is also the highest-recall operating point the gate permits.

Applied to the same fitted networks, it recovers events in every architecture, horizon and partition combination and never loses any:

| Horizon | Model | Temporal, row to ceiling | Geographic, row to ceiling |
| --- | --- | --- | --- |
| 6h | mlp hybrid prior | 17 to 24 of 54 | 10 to 11 of 50 |
| 6h | mlp hybrid | 19 to 23 of 54 | 9 to 13 of 50 |
| 12h | mlp direct | 18 to 23 of 54 | 9 to 11 of 50 |
| 24h | mlp direct | 26 to 28 of 55 | 11 to 13 of 50 |
| 24h | mlp hybrid prior | 25 to 27 of 55 | 8 to 12 of 50 |

The cost is small and never reaches the ceiling. Pooled false alerts land at .86, .68 and .73 per 100 decisions at 6, 12 and 24 hours against a ceiling of 1.5, and the worst single region is 2.18 against a ceiling of three. The rule spends budget the gate always allowed and the earlier experiment left unused.

## Cohort and data

The earlier cohort's 2025 outcomes are inspected, so this experiment drew a second disjoint cohort from the pool left after excluding every station any prior experiment registered. Twenty-four gauges in twelve HUC2 regions were frozen; four returned unusable series and were excluded without replacement (02092500, 07346045, 08029500, 08164300, all for an unexpected statistic or unit), leaving eleven development and nine geographic gauges. The cohort shares HUC2 regions with earlier studies, so no regional-climate independence is claimed.

Development materialization covered 232,248 rows with 102,876 training rows across 2020-2022. The 2025 test held 31,020 temporal rows with 55 eligible events and 23,544 geographic rows with 50. Event coverage is 77 to 79 percent temporal and 83 percent geographic; the losses are almost entirely missing history before the warmup window.

## Model comparison on the frozen 2025 test

Forty-two candidates were fit on identical rows, calibrated on 2023 and selected on 2024 development plus geographic validation by equal-weight average precision. Selection chose `mlp-direct-ceiling-6`, `mlp-hybrid-ceiling-12` and `mlp-direct-ceiling-24`, with `baseline-linear-row` at every horizon as the physical comparator.

| Horizon | Partition | Candidate AP | Baseline AP | AP interval | Events (cand / base) | False alerts (cand / base) | Median lead (cand / base) |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 6h | Temporal | .1627 | .1042 | [.0127, .0950] | 21 / 18 of 54 | .90 / .66 | 3.0 / 3.0 h |
| 6h | Geographic | .0994 | .0565 | [-.0070, .1115] | 11 / 9 of 50 | .80 / .99 | 2.0 / 2.0 h |
| 12h | Temporal | .1788 | .1336 | [-.0050, .1208] | 21 / 18 of 54 | .74 / .74 | 4.0 / 4.0 h |
| 12h | Geographic | .0812 | .0695 | [-.0276, .0466] | 6 / 12 of 50 | .60 / 1.04 | 3.5 / 3.5 h |
| 24h | Temporal | .1821 | .1522 | [-.0013, .0762] | 28 / 26 of 55 | .77 / .87 | 5.5 / 5.5 h |
| 24h | Geographic | .1072 | .0956 | [-.0182, .0346] | 13 / 16 of 50 | .66 / 1.10 | 3.0 / 6.0 h |

The MLP now leads the extrapolation on average precision in five of six partition-horizon cells and detects more temporal events at 6 and 24 hours. Two weaknesses remain visible. The geographic partition is inconsistent, with the 12-hour candidate detecting half the baseline's events, and the geographic average-precision interval narrowly includes zero at 6 hours where its lower bound is -.007.

## Gate outcomes

| Horizon | Gates failed | Which |
| --- | ---: | --- |
| 6h | 3 of 13 | Geographic AP interval, geographic event gain (.22 against .18, needing five points), pooled event-recall interval |
| 12h | 4 of 13 | Temporal and geographic AP intervals, geographic event gain, pooled event-recall interval |
| 24h | 5 of 13 | Temporal and geographic AP intervals, temporal and geographic event gain, pooled event-recall interval |

The 6-hour candidate is close. Its temporal AP interval clears zero, it gains 5.6 points of temporal event recall over the baseline, its Brier score is better in both partitions, no region exceeds three false alerts, and its pooled false-alert rate is .86 against a ceiling of 1.5. It fails because the geographic ranking interval does not clear zero and geographic recall gains only four points where five are required, and because the pooled event-recall interval spans zero.

## Failed hypotheses

**The damped offset did not help.** Validation average precision for the hybrid sits at .111, .129 and .136 against the direct network's .119, .122 and .138 at 6, 12 and 24 hours, and on the test the two are within .004 average precision at every horizon. The mechanism that raised average precision in the daily historical-transition study does not transfer to this hourly target, where the network already learns the flow persistence the offset encodes.

**The trailing exceedance features did not help either.** `mlp_hybrid_prior` reaches the best temporal average precision at 24 hours (.1833 against .1821 for the direct network) and the worst geographic average precision of the three architectures (.0975 against .1075 and .1072), while ranking last of the three on validation at 6 and 24 hours. That pattern is noise, not signal.

**The balanced loss hurt.** `mlp_hybrid_balanced` is the weakest candidate everywhere, at .084, .118 and .113 validation average precision, and on the test it detects 11 of 54 temporal events at 6 hours where the unweighted direct network detects 21. Reweighting the positives distorted the ranking without improving the operating point the threshold rule already controls.

The clearest joint result is that none of the three architectural or objective changes beat the plainest network, which is what the earlier experiment found about residual blocks and multi-horizon weight sharing. Model capacity is not the binding constraint on this target.

## Geographic generalization

The geographic partition is where the system still fails. The candidate leads on average precision in all three geographic cells, by .043, .012 and .012, but only the 6-hour interval approaches significance and it stops one thousandth short of zero. Event recall swings from a four-point gain at 6 hours to a twelve-point loss at 12 hours, and the 24-hour candidate warns four hours later than the baseline on the events both detect.

Part of this is a population effect that the gates already anticipated: the geographic partition carries 50 events across eight gauges, and the paired bootstrap over twelve HUC4 clusters is thin. The honest reading is that the ranking advantage is real but unproven geographically, and the recall advantage is not consistent enough to claim.

## Runtime and reproducibility

Fitting stays well inside the budget. Across three horizons the 24 MLP candidates take 47.2 seconds at 796 MB peak resident memory, LightGBM 1.8 seconds at 612 MB, logistic 1.7 seconds at 522 MB, and XGBoost and the physical baselines under a second each. The largest network holds 5,953 parameters. Final inference over the full 2025 frame takes .157 seconds for the MLP family, .125 for LightGBM, .031 for XGBoost and .003 for the physical baselines. torch is 2.14.1; every family selection record pins its versions.

Run from the repository root:

```sh
.venv/bin/python -m scripts.mlp_hybrid freeze
.venv/bin/python -m scripts.mlp_hybrid development
.venv/bin/python -m scripts.mlp_hybrid_worker fit baseline
.venv/bin/python -m scripts.mlp_hybrid_worker fit logistic
.venv/bin/python -m scripts.mlp_hybrid_worker fit lightgbm
.venv/bin/python -m scripts.mlp_hybrid_worker fit xgboost
.venv/bin/python -m scripts.mlp_hybrid_worker fit mlp
.venv/bin/python -m scripts.mlp_hybrid_worker select
.venv/bin/python -m scripts.mlp_hybrid test
.venv/bin/python -m scripts.mlp_hybrid_worker evaluate mlp
.venv/bin/python -m scripts.report_mlp_hybrid
.venv/bin/python -m pytest -q
```

Collection, fitting and evaluation need `USGS_API_KEY` in the environment or the ignored project `.env`, sent only in the official `X-Api-Key` header and never stored in provenance. Each command refuses to overwrite a frozen output. All 84 saved prediction files replay exactly from the frozen artifacts and match the canonical labels before scoring; the `*-replay-audit.json` files record the checksums. The full suite passes 341 tests.

## Decision and next step

No horizon passes every gate, so nothing is promoted. The defensible claim is that the MLP now ranks better than a three-line extrapolation and, with the ceiling rule, detects more temporal events at 6 and 24 hours without exceeding any operational false-alert ceiling. The remaining blockers are geographic: an unproven ranking interval and inconsistent geographic recall.

The next step is not another architecture. It is a geographic one. The 6-hour candidate misses promotion by one point of geographic recall gain and one thousandth of an interval, and the underlying limit is eight gauges in twelve paired clusters. A larger reserved cohort, screened for HUC4 disjointness rather than station freshness, would test whether the geographic ranking advantage is real. Until that exists, the ceiling rule is the transferable finding and it can be applied to any ranking model in this pipeline without new training.
