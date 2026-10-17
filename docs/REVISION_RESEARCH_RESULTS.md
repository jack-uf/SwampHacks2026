# Bridge rainfall and forecast revisions do not clear the reliability gate

The two proposed inputs move every point estimate in the intended direction. The selected candidate passes both subset quality criteria and the pooled alert budget, and fails the four criteria that decide: event-recall gain, regional alert burden, and both paired bootstrap tests. On both forward-only folds the combined candidate ranks better, separates the two classes slightly better and warns a little more often, but its AP and event-recall gains are not distinguishable from zero under paired resampling, its recall gain is below the precommitted five points, and the fold 2021 regional alert burden stays above the ceiling. The fresh geographic cohort was not opened, no candidate was promoted, and production models and safety rules are unchanged.

The measured outcome is a negative result with a clear diagnosis: the binding constraint on this population is per-region alert calibration, not missing forecast inputs. Both the control and the combined candidate exceed three false alerts per 100 decisions in HUC2 06 during the 2021 fold, so no input addition alone can pass the regional criterion that this study precommitted to.

## Frozen design and evidence

The [protocol](REVISION_RESEARCH_PROTOCOL.md), implementation, tests and 16 pinned input hashes were committed in `63c2a80` before any fit. The development collection was committed in `fa21548`, the rolling-fold selection in `18ebb1c`, and the gate evaluation in the commit that adds this report. One amendment, [amendment-1.md](../experiments/sprint-20261017-revision/amendment-1.md), records a parsing fix in the reporting loader: saved predictions are read with round-trip float precision, because one boundary decision sat exactly on its own threshold. Re-running collection and selection after that fix reproduced the identical dataset digest and all eight artifact and prediction checksums, so no fitted probability, threshold, alert or gate outcome changed.

Machine-readable evidence lives in `experiments/sprint-20261017-revision/`: `protocol.json`, `training.provenance.json`, `selection.json`, `evaluation.json`, per-candidate artifacts and per-fold predictions. Every file carries its own checksums and the selection pins the artifacts it produced.

## Data, alignment and folds

The population is the analog study's 89 gauged catchments, 107,689 decisions from 2017 through 2022, all previously inspected development evidence. The new inputs are rebuilt from locally cached immutable HRES chunks, so this experiment downloaded no weather data. Alignment was verified against the retained archive convention that a run dated I at lead k covers UTC day I + k - 1: bridge rain for day D-1 comes from lead 2 of the run dated D-2, bridge rain for day D-2 comes from lead 2 of the run dated D-3, and each revision feature for valid day D through D+3 compares the runs dated D-1, D-2 and D-3. Every feature is a shift of at least one day, so no run issued on or after the decision date can enter, and the newest admissible run remains D-1 as in the frozen feature set. Unavailable values stay missing; in this archive no decision row was dropped.

| Fold | Fit | Calibrate | Temporal validation | Geographic validation |
| --- | --- | --- | --- | --- |
| 2021 | development 2017 to 2019 | development 2020 | 13,562 development rows, 122 events | 4,565 geographic rows, 67 events |
| 2022 | development 2017 to 2020 | development 2021 | 13,322 development rows, 107 events | 8,662 geographic rows, 77 events |

The geographic cohorts differ in size and breadth: fold 2021 holds 17 stations in 2 HUC2 regions, fold 2022 holds 32 stations in 8. Regional results in the smaller cohort rest on few stations.

## Candidate comparison

AP and Brier are reported separately for the temporal and geographic validation subsets. Event recall, precision and false alerts are the pooled 1 percent policy over both subsets of a fold. Lead is the median over warned events.

| Fold | Candidate | Temporal AP | Geographic AP | Temporal Brier | Geographic Brier | Event recall | Precision | False per 100 | Median lead (h) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | control | .2866 | .2950 | .02159 | .03567 | .397 | .397 | .987 | 53.0 |
| 2021 | bridge | .2819 | .2927 | .02173 | .03578 | .413 | .405 | .999 | 53.0 |
| 2021 | revision | .2952 | .2925 | .02137 | .03554 | .397 | .405 | .999 | 53.0 |
| 2021 | combined | .2980 | .2983 | .02133 | .03550 | .402 | .414 | .982 | 53.5 |
| 2022 | control | .3558 | .2526 | .01786 | .02224 | .429 | .401 | .992 | 55.0 |
| 2022 | bridge | .3564 | .2530 | .01786 | .02215 | .435 | .401 | .987 | 55.0 |
| 2022 | revision | .3679 | .2569 | .01772 | .02225 | .446 | .403 | .983 | 54.0 |
| 2022 | combined | .3644 | .2653 | .01776 | .02210 | .451 | .406 | .992 | 54.0 |

Selection takes the highest mean AP across the four subsets, ties broken by Brier and name. The combined candidate leads at .3065, ahead of revision at .3031, control at .2975 and bridge at .2960. The control is the same recipe retrained inside each fold window, so these differences are forward comparisons.

## Gate outcome

The gate is evaluated once for the selected candidate against the control and requires every criterion in both folds.

| Criterion | Result | Evidence |
| --- | --- | --- |
| AP not below control in every subset | pass | four of four subsets higher, the narrowest being 2021 geographic at .2983 against .2950 |
| Brier not above control in every subset | pass | four of four lower |
| Event recall at least five points above control | fail | +0.5 points in 2021, +2.2 points in 2022 |
| Pooled false alerts at most 1.5 per 100 | pass | .982 and .992 |
| Every HUC2 at most three per 100 | fail | fold 2021 HUC2 06 at 6.05, against 5.69 for the control |
| Paired AP difference above zero | fail | lower bounds -0.0049 to -0.0034 and -0.0042 to -0.0017 |
| Paired event-recall difference above zero | fail | lower bounds -0.0408 to -0.0105 |

Paired 500-draw bootstrap intervals for combined minus control, seed 42:

| Fold | Resampling | AP difference 95 percent | Event-recall difference 95 percent |
| --- | --- | --- | --- |
| 2021 | HUC4 blocks | [-.00488, .02374] | [-.0408, .0484] |
| 2021 | Calendar months | [-.0034, .02165] | [-.0225, .0338] |
| 2022 | HUC4 blocks | [-.00418, .02782] | [-.0179, .0593] |
| 2022 | Calendar months | [-.00168, .02004] | [-.0105, .0603] |

Every interval crosses zero. The point estimates fall inside the range this population of about 190 pooled events per fold can resolve, which is the honest reading rather than a claim that the inputs carry no signal at all.

## Geographic detail

The additions behave differently on the two geographic subsets. In fold 2022 the combined candidate raises geographic AP from .2526 to .2653 and geographic event recall from .416 to .442, while in fold 2021 geographic recall is unchanged at .493 and geographic AP moves from .2950 to .2983. Temporal recall rises by half a point in 2021 and 2.2 points in 2022. Precision is marginally higher everywhere. The revision block carries most of the fold 2022 temporal gain on its own, which is consistent with run disagreement mattering more in a year with more active weather, but a single fold cannot separate that from noise.

The regional failure is pre-existing rather than introduced: fold 2021 HUC2 06 reaches 5.69 false alerts per 100 decisions for the control and 6.05 for combined. Fold 2022 has no region above 2.7. The geographic cohort of fold 2021 is small enough that one weak region dominates it, and neither candidate addresses that.

## Decision and next research direction

No candidate is promoted. The fresh geographic cohort stays unopened, the served models and the deterministic safety rules are untouched, and no probability from this study feeds PlanIT's hazard index.

The result points at the next experiment rather than the next model. Alert rates transfer poorly across regions while ranking transfers acceptably, and the 2021 fold shows the burden concentrated in one HUC2 region for the control as well. A calibration study that predicts per-region alert burden from development data and holds itself to the same regional ceiling on a fresh cohort addresses the demonstrated failure. Two further conditions travel with that: an as-issued forecast availability check, since these inputs are reforecasts in places and the dissemination buffer is assumed, and an evaluation on genuinely new geography, since both folds here are development history.

## Limits

The target is high flow above a pre-2016 p95 after seven quiet days, not verified flooding, flood stage or worksite consequence. Archived forecasts are reforecasts in places and the D-2 and D-3 retention is a research assumption. Discharge records include retrospective revisions. Coverage is daily means, so sub-daily and flash floods are invisible. Both folds are development data already inspected by earlier studies, and the geographic subsets are small, so the intervals capture resampling variation only and not the dependence created by shared weather systems. The regional criterion was precommitted and is reported as failed even though the control fails it too.

## Verification and reproduction

The complete repository suite passes: 225 tests in 28.97 seconds, including seven new tests for same-valid-day alignment, decision-day leakage, missing runs, duplicate and out-of-window decisions, and the gate's criteria and boundaries. The report stage re-reads the frozen predictions, rechecks that saved decisions and labels match the frozen dataset, and recomputes every alert from its frozen threshold before computing intervals.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.revision_research
.venv/bin/python -m scripts.revision_worker select
.venv/bin/python -m scripts.revision_worker report
```

The collection needs no network access; it reads the retained HRES chunks with manifest and SHA256 verification.
