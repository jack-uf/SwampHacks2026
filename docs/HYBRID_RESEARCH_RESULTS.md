# Hybrid experts: real complementary detections, no validated improvement

The experts are genuinely complementary at the event level and the combination rules do not turn that into a better warning system. At their own 1 percent budgets the five experts together catch 57 percent of emerging high-water events while the best single expert catches about 40 percent, and roughly 42 percent of events are missed by every expert. On the forward evaluation fold the combiner selected on the fit fold lost pooled average precision to the control, .2955 against .3065, and failed six of the eight precommitted gate criteria. The equal-weight and free-coefficient combiners edged above the control on pooled AP, .3093 and .3110, with intervals that cross zero, so neither establishes an improvement.

Two useful findings survive the failure. Every combiner ranked the geographic subset better than the control did (.2667, .2700 and .2914 against .2526) while every combiner ranked the temporal subset worse (.3308, .3539 and .3296 against .3558), so combination buys unfamiliar-watershed ranking at the cost of familiar-watershed ranking. And the alert operating point, not the ranking, is what fails to transfer across folds: thresholds frozen on the fit fold warned at .13 to .26 per 100 decisions on the evaluation fold against a 1 percent target, cutting the control's own event recall from .429 to .207.

Nothing was integrated. Production models, safety rules and application behaviour are unchanged, and no fresh cohort was opened.

## Frozen design and evidence

The [protocol](HYBRID_RESEARCH_PROTOCOL.md), implementation, tests and ten pinned input hashes were committed before any fit. Two amendments are recorded in the experiment directory: [amendment-1.md](../experiments/sprint-20261017-hybrid/amendment-1.md) replaces the pooled neural expert, whose training stalled three times at low memory while a standalone torch import completed in 1.5 seconds, with randomized trees; [amendment-2.md](../experiments/sprint-20261017-hybrid/amendment-2.md) fixes subset reporting that indexed pooled positions with reset subset indices. Both predate the fold 2022 evaluation, and both are reported with their evidence.

Machine-readable evidence lives in `experiments/sprint-20261017-hybrid/`: `protocol.json`, `experts.json`, `selection.json`, `evaluation.json`, per-fold expert predictions and combiner artifacts.

## Experts and complementarity

Five experts predict the same target and horizon: a new daily mean discharge exceedance of a pre-2016 p95 after seven quiet days on local D+1 through D+3. Each is refit inside its fold and calibrated on that fold's calibration year, and each gets its own 1 percent threshold on the fold it is measured on.

| Fold | Expert | AP | Brier | Event recall | False per 100 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2021 | gbm | .2831 | .02514 | .397 | .987 |
| 2021 | extra | .2734 | .02547 | .365 | .982 |
| 2021 | logistic | .2690 | .02623 | .402 | .960 |
| 2021 | rainfall | .2322 | .02604 | .312 | .987 |
| 2021 | analog | .0783 | .02894 | .079 | .905 |
| 2022 | gbm | .3065 | .01959 | .429 | .992 |
| 2022 | extra | .2924 | .01980 | .408 | .992 |
| 2022 | logistic | .2561 | .02033 | .402 | .987 |
| 2022 | rainfall | .1980 | .02091 | .348 | .951 |
| 2022 | analog | .0606 | .02285 | .043 | .864 |

Do the experts make complementary errors? Partly, and thinly. Pairwise agreement between alert masks runs from .974 to .989 in both folds, so on any given decision the experts almost always agree. At the event level the picture differs: 189 events in fold 2021 of which only 3 are detected by all five experts and 26 by exactly one; 184 events in fold 2022 with 4 shared and 21 single-expert. Union recall is .571 and .576 against roughly .40 for the best single expert. The remaining 81 and 78 events are missed by every expert, which caps any combination well below what the union of alert masks suggests.

## Selection on the fit fold

Combiners are fit on fold 2021 temporal rows only and selected on the fold 2021 geographic rows, which they never see during fitting.

| Combiner | Selection AP | Selection Brier |
| --- | ---: | ---: |
| gated | .3166 | .03404 |
| stack | .3015 | .03553 |
| control (gbm) | .2950 | reference |
| mean | .2895 | .03529 |

The gated combiner wins selection, so the gate is evaluated for it. This is the study's clearest failure mode: a 4,565-row, 67-event selection population rewards the most flexible rule, and that rule then transfers worst.

## Evaluation fold

Thresholds and combiner coefficients are frozen from fold 2021 and applied unchanged to fold 2022. The control is the gbm expert with thresholds derived by the same budget rule on the same fold 2021 geographic rows.

| Candidate | Pooled AP | Temporal AP | Geographic AP | Brier | Event recall | False per 100 | Median lead (h) | Worst HUC2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | .3065 | .3558 | .2526 | .01959 | .207 | .255 | 54.5 | 1.083 |
| mean | .3093 | .3296 | .2914 | .01949 | .158 | .127 | 53.0 | .452 |
| stack | .3110 | .3539 | .2700 | .01943 | .185 | .159 | 53.5 | .722 |
| gated (selected) | .2955 | .3308 | .2667 | .01961 | .234 | .227 | 54.0 | .603 |

Gate outcome for the selected combiner, every criterion required:

| Criterion | Result | Evidence |
| --- | --- | --- |
| AP strictly above control | fail | .2955 against .3065 |
| Brier not above control | fail | .01961 against .01959 |
| Event recall at least five points above control | fail | .234 against .207, a gain of 2.7 points |
| Pooled false alerts at most 1.5 per 100 | pass | .227 |
| Every HUC2 at most three per 100 | pass | .603 worst |
| Paired AP lower bound above zero | fail | [-.0549, .0268] and [-.0500, .0350] |
| Paired event-recall lower bound above zero | fail | [-.0590, .1209] and [-.0767, .1457] |
| Median event lead not below control | fail | 54.0 against 54.5 hours |

Paired 500-draw bootstrap intervals for each combiner minus control, seed 42:

| Combiner | Resampling | AP difference 95 percent | Event-recall difference 95 percent |
| --- | --- | --- | --- |
| mean | HUC4 blocks | [-.02609, .02670] | [-.0948, -.0058] |
| mean | Calendar months | [-.02135, .03802] | [-.0969, -.0054] |
| stack | HUC4 blocks | [-.01358, .02314] | [-.0687, .0231] |
| stack | Calendar months | [-.00614, .02092] | [-.0590, .0233] |
| gated | HUC4 blocks | [-.05491, .02676] | [-.0590, .1209] |
| gated | Calendar months | [-.04995, .03500] | [-.0767, .1457] |

The only interval that excludes zero does so against a combiner: the equal-weight rule loses event recall, significantly in both resampling schemes. Every other interval spans zero.

## What the failure teaches

Complementary detections are real but concentrated in a minority of events, and most events are invisible to every expert, so the ceiling on any mixture is far below the union of the five alert masks. The alert masks agree almost always, which is why weighted and unweighted combinations of the same experts produce nearly the same ranking.

The geographic-versus-temporal split is consistent in direction across all three combiners: mixing experts improves ranking on stations the experts were not trained on and costs ranking on the stations the strongest expert knows best. For a safety console that serves unfamiliar sites, that trade is the interesting part of an otherwise negative result, and it needs a second geographic cohort before anyone acts on it.

Threshold transfer dominates the operational numbers. Every candidate warned at a small fraction of the intended budget on the evaluation fold, which is why pooled false alerts and the regional ceiling pass trivially while recall collapses for everyone. A relative or quantile-based threshold rule, recalibrated per region on rolling data, is the change most likely to matter, and it is a different change from adding another model.

Selecting a combiner on 67 events rewarded the most flexible rule, and it lost to a fixed equal-weight average on the evaluation fold. If this line continues, selection should use larger cohorts or leave-one-region-out resampling rather than a single small geographic year.

## Limits

The evaluation is a single fold of already inspected development history, so the gate is one shot and passing would not have justified deployment. The pooled analog prior is weak in this population and drags shared weight down, which is a property of the fixed member list rather than of combination itself. The neural member never trained in this environment, so architectural diversity here is bagged randomized trees against boosted trees and three conventional baselines, not a different model family. Daily means hide sub-daily peaks, archived forecasts are reforecasts in places, and crew-report intelligence from Jev stayed out of scope because it addresses a different target.

## Verification and reproduction

The complete repository suite passes: 230 tests in 28.19 seconds, including five new tests for equal-weight combination, event-level complementarity accounting, the ability of stacking and gating to beat a specialist expert on synthetic specialists, fold-boundary refusal, and the gate's criteria and boundaries. The evaluation stage re-reads saved expert predictions, checks that decisions and labels match the frozen dataset, and recomputes every alert from its frozen threshold.

Reproduction commands, from the repository root: `.venv/bin/python -m pytest -q`, then `.venv/bin/python -m scripts.hybrid_research run`, then `.venv/bin/python -m scripts.hybrid_research evaluate`.

One reproduction caveat is recorded rather than hidden: reruns reproduce the frozen thresholds and metrics exactly for the mean and gated combiners and for the control, but the stack combiner's logistic coefficients move in the last floating-point digits, so serialized artifact checksums are not reproducible even though the comparable metrics are.
