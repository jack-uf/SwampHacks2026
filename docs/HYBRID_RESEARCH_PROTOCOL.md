# Complementary experts and learned combination for emerging high water

This study asks two questions in order. First, do the project's existing hazard models make complementary detection errors at all? Second, does a combination rule learned on one forward fold beat the strongest single model on the next one? Both answers are measured on the same target and horizon the earlier studies use: a new daily mean discharge exceedance of a pre-2016 p95 after seven quiet days, on local D+1 through D+3.

## Population and folds

The input is the forecast-revision study's frozen development dataset: 89 gauged catchments and 107,689 decisions from 2017 through 2022, verified against that study's dataset digest. Two folds, each with decision windows from January 8 through December 24:

| Role | Fit fold 2021 | Evaluation fold 2022 |
| --- | --- | --- |
| Expert training | development 2017 to 2019 | development 2017 to 2020 |
| Expert calibration | development 2020 | development 2021 |
| Temporal validation | development 2021 | development 2022 |
| Geographic validation | geographic 2021 | geographic 2022 |

Every expert is refit inside a fold, so expert predictions on that fold's validation rows are out of fold for the expert. The combiner is fit on fold 2021 temporal rows only, and evaluated on fold 2022. Fold 2021 geographic rows are used to select and threshold a combiner but never to fit it, so the fold 2022 geographic subset is out of fold for the combiner as well. Both folds are previously inspected development history, which this study states rather than hides.

## Experts

All five address the same target and horizon. Each is calibrated with a one-dimensional logistic model on its fold's calibration year.

| Expert | Model |
| --- | --- |
| gbm | LightGBM, 200 trees, learning rate .04, depth 3, 7 leaves, seed 42, fitted with the historical analog logit as initial score |
| neural | pooled response mixture, 60 epochs, batch 2048, AdamW at .002 with .001 weight decay, seed 42 |
| analog | pre-2016 nearest-neighbour transition logit, no fitted weather term |
| rainfall | HRES precipitation summed over forecast leads 3 to 5 |
| logistic | standardized logistic regression on the combined weather, hydrology, snow and analog feature set |

## Complementarity

For each fold, each expert's alert cutoff is its own 1 percent budget threshold on that fold. Reported diagnostics: pairwise alert agreement between experts, events detected by every expert, events detected by exactly one expert, events missed by all, and the union recall. These describe the population and are not promotion criteria.

## Combiners

All combiners take the five experts' calibrated log odds and are fit on fold 2021 temporal rows.

| Combiner | Rule |
| --- | --- |
| mean | equal-weight mean of the five log odds, then a one-parameter logistic recalibration |
| stack | logistic regression with one free coefficient per expert |
| gated | LightGBM on the five log odds plus the gating features, so expert trust can depend on conditions |

Gating features are the ten pre-2016 catchment signature descriptors, log area, elevation, seasonal sine and cosine, the snow store, history completeness, the current lagged log flow, the first two forecast spread columns and the current forecast-day rain mean. They describe watershed character, season, and forecast disagreement, and include no label-derived quantity.

## Thresholds, selection and gate

For each combiner, thresholds on its fold 2021 geographic score target .5, 1 and 2 false alerts per 100 decisions under the existing budget rule, plus a conservative 1 percent maximum over HUC2 groups. The control is the gbm expert with thresholds from the same rule on the same rows. Selection among the three combiners takes the highest average precision on the fold 2021 geographic population, breaking ties by Brier score and then by name. The control does not compete in selection; it is the reference.

The gate is evaluated once, on the fold 2022 pooled temporal and geographic rows, and every criterion must hold:

1. Average precision strictly above the control.
2. Brier score not above the control.
3. Event recall at the 1 percent policy at least five percentage points above the control.
4. Pooled false alerts at most 1.5 per 100 decisions.
5. Every HUC2 group at most three false alerts per 100 decisions.
6. Paired 500-draw bootstrap lower bounds above zero for the AP difference and for the event-recall difference, under both HUC4 and calendar-month resampling, seed 42.
7. Median event lead on warned events not below the control.

Boundaries are inclusive with a fixed 1e-9 floating-point tolerance. Gate results for the non-selected combiners are reported as diagnostics and cannot authorize integration. No criterion is relaxed after results are seen, and a failure is the expected outcome to report if the experts turn out not to be complementary.

## Scope limits

Crew-report intelligence from Jev stays out of this study: it addresses crew observations rather than measured discharge, and no validated bridge between the two exists. No satellite or image feature enters, because the multimodal pilot did not establish basin-aligned availability. The evaluation fold is single, so the gate is a one-shot test on already inspected development data and passing would justify a prospective evaluation rather than deployment. Archived forecasts are reforecasts in places and the dissemination buffer is assumed. Daily means hide sub-daily peaks. The neural expert is a small CPU model trained for reproducibility, not a large pretrained model.

## Resources and stopping

CPU only, two threads, no new dependency, peak memory below 6 GB, and one worker. Stop after the predeclared comparisons; do not add a combiner or tune a gate against the fold 2022 result. Run the repository tests, document artifacts and failures, and keep production models, safety rules and application behaviour unchanged until evidence justifies a safe integration.

Commands, from the repository root:

```sh
.venv/bin/python -m pytest -q tests/test_hybrid_research.py
.venv/bin/python -m scripts.hybrid_research run
.venv/bin/python -m scripts.hybrid_research evaluate
```
