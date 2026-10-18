# Internal high-flow benchmark findings

This is an internal development benchmark, not an external benchmark and not a state-of-the-art claim. It re-scores saved per-row probabilities on one canonical 2024 cohort. It fits nothing, downloads nothing and changes no production artifact.

The complete machine-readable record is `benchmark.json` in this directory. It contains 149 candidate records grouped into 110 distinct probability-vector and threshold combinations, with temporal, geographic and pooled metrics at two operating points, paired HUC4-cluster intervals against the calibrated physical comparator, the comparability catalog and the source-record reproduction check.

## Reproduce

```sh
.venv/bin/python -B scripts/report_high_flow_benchmark.py
```

The command runs from the repository root and writes `experiments/high-flow-benchmark-20261018/benchmark.json`. It reads only local artifacts and does not call the network.

## Cohort and operating points

The canonical cohort is the cycle-11 2024 validation population: 51,851 temporal decisions with 306 unique onsets and 17,826 geographic decisions with 102 unique onsets, pooled to 69,677 decisions and 408 events. All admitted prediction files were aligned on station, decision time and horizon; target, event time and HUC labels had to match exactly. The canonical row digests are in `benchmark.json`.

Two operating points are reported for every group. The `own_threshold` point uses the threshold frozen in the candidate's own experiment record. The `fixed_budget` point selects one threshold per candidate on the pooled 2024 rows to spend at most 1.5 false alerts per hundred. The fixed point is an outcome-matched diagnostic because it uses benchmark labels; it is not a deployable threshold policy. The paired intervals do not include threshold-selection uncertainty.

The comparator is `signal/c7/linear_conditional`, the conditional linear extrapolation selected as the strongest calibrated physical comparator in the retained cycle-7 report. Its frozen threshold is 0.044244290725; its matched-budget threshold is 0.035780880450.

## What can be compared on identical rows

| Source family | Admitted | Basis |
| --- | ---: | --- |
| `experiments/high-flow-signals-20261018/cycle-0..13` | 83 prediction files | Every saved prediction file resolves to the same canonical temporal and geographic row keys and labels. |
| `experiments/high-flow-20261018/cycle-2,3,4` | 66 candidate records | Cycle 2 predictions contain the canonical rows as a subset; cycles 3 and 4 cover them directly. Cycle 2 is restricted to the canonical rows before scoring. |

The following declared families cannot enter this full-row benchmark.

| Source family | Reason |
| --- | --- |
| `experiments/sprint-20261017-hydrology` | Daily 1-day and 2-day targets on forward and geographic daily cohorts; no shared six-hour 2024 cohort. |
| `experiments/sprint-20261017-shift-hydrology` | 69,411 test rows across 6-, 12- and 24-hour horizons in 2025; different gauges, labels and period. |
| `experiments/sprint-20261017-mlp-onset` | 79,233 test rows across 6-, 12- and 24-hour horizons in 2025; different population. |
| `experiments/sprint-20261017-mlp-hybrid` | 2025 temporal and geographic test populations across 6-, 12- and 24-hour horizons; different gauges and inspected outcomes. |
| `experiments/sprint-20261017-mlp-geo` | 70,992 test rows across 6-, 12- and 24-hour horizons in 2025; different population. |
| `experiments/sprint-20261017-transformer` | 36,800 station-date predictions for an archived maximum-gust target in 2026; different target and observation contract. |
| `experiments/sprint-20261017-analog` | 12,389 daily fresh-gauge rows in 2023 with 294 onsets and a one-day horizon. |
| `experiments/high-flow-20261018/cycle-1` | Predictions cover only 14,157 temporal rows and 10,219 geographic rows, not the full canonical cohort. |
| `experiments/high-flow-signals-20261018/upstream` | Separate downstream and upstream paired-gauge population, reported as 3,811 paired rows and 14 events. |

## Comparison at each candidate's own threshold

The table reports unique event counts, AP, pooled Brier loss, pooled false alerts per hundred decisions, the worst HUC2 burden, median detected lead, and the paired lower 95 percent bounds against the calibrated physical comparator. Event recall is the first number over the second; for example the frozen candidate's temporal recall is 116/306 = .379. The bounds are temporal AP, geographic AP and pooled event recall, in that order.

| Candidate | Temporal events | Temporal AP | Geographic events | Geographic AP | Pooled Brier | Pooled FA/100 | Worst HUC2 | Median lead | Temporal AP lower | Geographic AP lower | Pooled recall lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `signal/c7/linear_conditional` comparator | 84/306 | .0840 | 33/102 | .0964 | .005660 | 1.002 | 15: 1.70 | 2.0 | +.000 | +.000 | +.000 |
| `signal/c11/winner_forecast` frozen | 116/306 | .1869 | 56/102 | .1883 | .005195 | 1.039 | 17: 2.27 | 3.0 | +.066 | +.040 | +.069 |
| `signal/c10/winner_extended` | 111/306 | .1854 | 47/102 | .1793 | .005208 | 0.936 | 06: 2.09 | 3.0 | +.066 | +.021 | +.041 |
| `signal/c10/winner_runoff` | 111/306 | .1833 | 48/102 | .1958 | .005206 | 0.929 | 06: 2.16 | 3.0 | +.060 | +.034 | +.039 |
| `signal/c10/rain6_ratio_wetness_hydro` | 108/306 | .1842 | 50/102 | .1968 | .005200 | 0.910 | 06: 2.16 | 3.0 | +.064 | +.040 | +.041 |
| `signal/c7/combined_scaled_conditional` | 117/306 | .1844 | 56/102 | .1549 | .005244 | 1.114 | 17: 2.76 | 3.0 | +.064 | +.009 | +.066 |
| `signal/c11/forecast_wetness_hydro` | 121/306 | .1808 | 51/102 | .1867 | .005192 | 0.979 | 17: 1.90 | 3.0 | +.059 | +.034 | +.072 |
| `signal/c9/rain6_ratio_shape` | 103/306 | .1728 | 49/102 | .1932 | .005238 | 0.924 | 06: 2.32 | 3.0 | +.055 | +.033 | +.020 |
| `signal/c13/winner_forecast_mlp` | 104/306 | .1383 | 53/102 | .1779 | .005316 | 1.170 | 17: 3.01 | 3.0 | +.026 | +.051 | +.043 |
| `signal/c13/control_mlp` | 93/306 | .1333 | 36/102 | .1497 | .005363 | 0.848 | 06: 1.31 | 2.0 | +.026 | +.017 | -.016 |
| `research/c2/expanded_conditional/ceiling` | 148/306 | .1454 | 61/102 | .1628 | .005319 | 1.795 | 17: 4.09 | 3.0 | +.038 | +.017 | +.162 |

## Comparison at the common pooled 1.5 false-alert budget

Every row uses a threshold selected to spend at most 1.5 pooled false alerts per hundred. The tie-preserving rule lands at a realized pooled rate of 1.4998 for every group because all candidates score the same pooled rows. Realized per-partition rates can exceed 1.5 because the budget is pooled. The worst HUC2 can also exceed the original regional ceiling of 3.0, which the fixed point does not enforce.

| Candidate | Temporal events | Temporal AP | Geographic events | Geographic AP | Pooled Brier | Pooled FA/100 | Worst HUC2 | Median lead | Temporal AP lower | Geographic AP lower | Pooled recall lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `signal/c7/linear_conditional` comparator | 107/306 | .0840 | 46/102 | .0964 | .005660 | 1.500 | 13: 3.02 | 3.0 | +.000 | +.000 | +.000 |
| `signal/c11/winner_forecast` frozen | 144/306 | .1869 | 65/102 | .1883 | .005195 | 1.500 | 17: 2.99 | 3.0 | +.066 | +.040 | +.064 |
| `signal/c10/winner_extended` | 140/306 | .1854 | 59/102 | .1793 | .005208 | 1.500 | 06: 3.17 | 3.0 | +.066 | +.021 | +.043 |
| `signal/c10/winner_runoff` | 138/306 | .1833 | 60/102 | .1958 | .005206 | 1.500 | 06: 3.09 | 3.0 | +.060 | +.034 | +.037 |
| `signal/c10/rain6_ratio_wetness_hydro` | 141/306 | .1842 | 60/102 | .1968 | .005200 | 1.500 | 06: 3.17 | 3.0 | +.064 | +.040 | +.048 |
| `signal/c7/combined_scaled_conditional` | 138/306 | .1844 | 63/102 | .1549 | .005244 | 1.500 | 17: 3.54 | 3.0 | +.064 | +.009 | +.047 |
| `signal/c11/forecast_wetness_hydro` | 150/306 | .1808 | 57/102 | .1867 | .005192 | 1.500 | 06: 2.94 | 3.0 | +.059 | +.034 | +.051 |
| `signal/c9/rain6_ratio_shape` | 133/306 | .1728 | 62/102 | .1932 | .005238 | 1.500 | 06: 3.25 | 3.0 | +.055 | +.033 | +.028 |
| `signal/c13/winner_forecast_mlp` | 127/306 | .1383 | 60/102 | .1779 | .005316 | 1.500 | 17: 3.83 | 3.0 | +.026 | +.051 | +.034 |
| `signal/c13/control_mlp` | 130/306 | .1333 | 55/102 | .1497 | .005363 | 1.500 | 01: 2.43 | 3.0 | +.026 | +.017 | +.017 |
| `research/c2/expanded_conditional/ceiling` | 130/306 | .1454 | 58/102 | .1628 | .005319 | 1.500 | 17: 3.54 | 3.0 | +.038 | +.017 | +.014 |

## Does any existing family beat the cycle-11 frozen candidate

No. At either operating point, no other admitted group is at least as good on all four headline partition metrics, temporal events, temporal AP, geographic events and geographic AP, and strictly better on at least one of them. The benchmark records this as zero dominators.

The frozen candidate is not the top of every single column. At its own threshold, `signal/c7/combined_scaled_conditional` detects one more temporal event, 117 versus 116, but its geographic AP falls to .1549 and its geographic false-alert rate rises. `signal/c10/rain6_ratio_wetness_hydro` has a higher geographic AP, .1968 versus .1883, but finds fewer geographic events, 50 versus 56, and fewer temporal events, 108 versus 116. At the common budget, `signal/c11/forecast_wetness_hydro` finds 150 temporal events versus 144, but its temporal AP is lower, .1808 versus .1869, and it finds 57 geographic events versus 65. `signal/c10/winner_runoff` and `signal/c10/rain6_ratio_wetness_hydro` edge the frozen candidate on pooled AP, .1842 versus .1824, but neither matches its geographic event count or its temporal AP.

The paired comparison against the calibrated physical comparator supports the frozen candidate on ranking and recall. At its own threshold the temporal AP difference is +.1029 with a 95 percent interval of [.0665, .1401], and the geographic AP difference is +.0918 with [.0398, .1937]. The pooled event-recall difference is +.1348 with [.0686, .1997]. At the common budget the pooled recall difference is +.1373 with [.0640, .2108]. The false-alert difference intervals include zero, so the comparison does not establish a burden difference at this sample size.

## What external comparisons would require

The internal benchmark cannot be extended to external systems from local files. The repository contains only cached documentation and small research samples for those systems, not the archives or labels needed to score them.

NWM. The cached `experiments/sprint-20261017-hydrology/research/nwm-readme.md` describes the retrospective archive as historical simulations driven by observed rainfall and weather reanalysis, with output at hourly streamflow points. It is not an as-issued forecast archive. A fair NWM comparison needs the retrospective or operational files themselves, a documented reach-to-gauge mapping, matching decision times and lead times, and the same onset labels and quiet-period rules. An operational comparison additionally needs an archive that preserves issue time and vintage. None of that exists offline here.

GloFAS. No GloFAS artifact exists in this repository. A comparison needs the reanalysis or forecast discharge files, a river-grid-to-gauge mapping, matching lead times and an explicit statement of whether the product is a reforecast, a reanalysis or an operational issue. That requires network access and external licensing review.

Published neural benchmarks. The local caches include `Caravan-code.zip`, `caravan-readme.md`, `neuralhydrology-readme.md`, `mclstm-readme.md` and a small Caravan MultiMet sample under `experiments/sprint-20261017-hydrology/research/`. They do not include the benchmark discharge records, the published train and test splits, the fitted checkpoints or per-row predictions. Reproducing those numbers needs the original data and code, the exact label and metric definitions, and enough compute to run the published training procedure. None of that is obtainable offline from this repository.

The honest statement is therefore narrow. This benchmark orders the repository's own saved candidates on one inspected 2024 development cohort at two controlled operating points. It says nothing about how any of them rank against NWM, GloFAS or published neural models.

## Limitations

The cohort was inspected in earlier research. This is development evidence and cannot support a confirmation or promotion claim.

The fixed-budget thresholds are selected on the same 2024 labels used for evaluation. They remove the threshold-choice difference between candidates but introduce a selection effect that the paired intervals do not measure. The frozen thresholds remain the only deployable points in this benchmark.

The bootstrap resamples HUC4 groups. It preserves shared weather within a group but does not remove dependence between groups, and the geographic partition has only 11 clusters.

The precipitation features are ERA5 reanalysis with an assumed six-hour availability delay, and the as-issued HRRR forecast block covers only the 12 UTC decision hours. The target is a measured high-flow onset, not verified flooding or field safety.

The benchmark-side metric code reproduces 1,270 frozen values with three known mismatched values across two candidate policies: `research/c3/proximity-platt/ceiling` differs by one detected geographic event, and two `proximity-platt` false-alert counts differ by about 0.0019 per hundred. The benchmark uses the per-row artifacts as the source of truth and records the mismatches in `benchmark.json` under `checks.source_metric_reproduction`.

At the common pooled budget, per-partition false-alert rates and worst-HUC2 burdens can exceed their original ceilings. The table reports them rather than hiding them.
