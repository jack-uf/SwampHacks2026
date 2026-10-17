# Bridge rainfall and forecast revisions for emerging high water

This experiment tests two inputs the frozen onset hybrid does not currently receive: the forecast day it has no rain signal for, and the disagreement between consecutive archived ECMWF runs for the same valid day. The comparison is forward in time, on previously inspected development catchments. No candidate enters production.

## Hypothesis

At a decision date D the frozen hybrid reads measured flow with a three day lag, causal rain memories whose newest day is D-2, forecast rain and temperature for UTC days D through D+3 from the run dated D-1, and pre-2016 historical transitions. UTC day D-1 therefore has no forecast input at all, and no comparison between issuances exists anywhere in the feature set.

Two additions are proposed. The bridge adds the archived forecast for UTC days D-1 and D-2, taken from the runs dated D-2 and D-3, so the model sees the day it currently cannot observe and a second reading of the day before. The revision block adds, for each valid day D through D+3, the mean, spread and newest-minus-oldest difference across the runs dated D-1, D-2 and D-3. Large spread means successive issuances disagreed about the same day.

The MultiMet paper (Kratzert and others, *Caravan MultiMet* extension, arXiv 2411.09459v1) documents the archive: the run date is the instant a forecast was supposedly issued, dissemination delay is ignored, some values are reforecasts, and lead one covers the run date itself. The paper and its retrieval manifest are retained under `experiments/sprint-20261017-hydrology/research/`. The spread features are input disagreement, not a calibrated uncertainty interval, and the retention of the D-2 and D-3 runs is a research assumption about availability, not proof that a forecaster received those runs as issued.

## Availability and alignment

Alignment is fixed as follows, for UTC daily means decoded from the retained HRES archive (leads 1 through 10, verified at load):

- Run dated I at lead k covers UTC day I + k - 1.
- Bridge day D-1 is lead 2 of the run dated D-2. Bridge day D-2 is lead 2 of the run dated D-3.
- For a valid day D+j, the three issuances are the run dated D-1 at lead j+2, the run dated D-2 at lead j+3 and the run dated D-3 at lead j+4. The newest admissible run is D-1, matching the existing one-day dissemination buffer.
- Missing archive values stay missing. A decision row whose new inputs are not all finite is dropped. No interpolation, no substitution of reanalysis, no later-run backfill.

Every feature is a shift of at least one day, so no forecast issued on or after the decision date can enter. The development fold features are rebuilt from locally cached immutable archive chunks with per-file manifest and SHA256 verification; this experiment downloads no weather data.

## Population and folds

The population is the analog study's development frame: 89 gauged catchments (57 development stations and 30 geographic stations) with decision rows from 2017 through 2022. All of it is development evidence: the 2022 development and 2022 geographic cohorts were used for selection in earlier studies, and the geographic rows of 2017 through 2021 were retained but not selected on. Passing this study's gate therefore authorizes opening a new geographic cohort, never integration or a performance claim.

Two forward-only folds, each with decision windows of January 8 through December 24:

| Fold | Fit | Calibrate | Temporal validation | Geographic validation |
| --- | --- | --- | --- | --- |
| 2021 | development 2017 to 2019 | development 2020 | development 2021 | geographic 2021 |
| 2022 | development 2017 to 2020 | development 2021 | development 2022 | geographic 2022 |

Observed fold populations are 36,807 and 49,980 fit rows, 13,173 and 13,562 calibration rows, 13,562 and 13,322 temporal rows, and 4,565 and 8,662 geographic rows. A fold part with fewer than five positive decisions, a duplicate decision, or an overlap with any other part fails the run instead of proceeding on a changed population.

## Candidates

All four candidates share the frozen recipe and differ only in their inputs: LightGBM with 200 trees, learning rate .04, depth 3, 7 leaves, two threads, seed 42, deterministic column-wise training, fitted with the historical analog logit as the initial score, then one-dimensional logistic calibration on the fold's calibration year. No new hyperparameter, station identifier, neural layer or threshold rule is introduced.

| Candidate | Inputs |
| --- | --- |
| control | Combined weather and hydrology, snow-aware memories, historical transitions |
| bridge | control plus four bridge columns |
| revision | control plus 24 mean, spread and revision columns |
| combined | control plus all 28 new columns |

The control is the same recipe retrained inside each fold window. The earlier studies' frozen artifacts were fitted on 2017 through 2020 and cannot serve as a forward comparison for a 2019 fit boundary.

## Thresholds, selection and gate

For every candidate and fold, pooled thresholds over the two validation subsets target .5, 1 and 2 false alerts per 100 decisions under the existing budget rule, and a conservative threshold targets a 1 percent maximum over HUC2 groups. Selection takes the highest mean AP over the four validation subsets, breaking ties by mean Brier and then by name. The control competes, so the study can conclude that neither addition helps.

The gate is evaluated once, for the selected candidate against the control, and every criterion must hold in both folds:

1. AP is not below the control in any of the four validation subsets.
2. Brier score is not above the control in any subset.
3. Pooled distinct-event recall at the 1 percent policy is at least five percentage points above the control.
4. Pooled false alerts at the 1 percent policy are at most 1.5 per 100 decisions.
5. Every HUC2 group is at most three false alerts per 100 decisions.
6. The paired 500-draw bootstrap lower bound of the AP difference is above zero under both HUC4 and calendar-month resampling, seed 42.
7. The same holds for the event-recall difference.

Criterion boundaries are inclusive with a fixed 1e-9 floating-point tolerance. Gate results for the non-selected candidates are reported as diagnostics and cannot open the fresh cohort. No criterion is relaxed after results are seen.

## Fresh cohort, only if the gate passes

If and only if the selected candidate passes the gate, a new geographic cohort is built and opened once. Screening is metadata only: unused CAMELS gauges, one per HUC4, ranked by SHA256 of `revision-20261017-<station>`, each requiring an exact USGS identifier within one kilometre of inventory coordinates and a simplified upstream polygon with zero positive-area intersection against all previously reserved polygons. Previously represented HUC4 groups are permitted because earlier studies consumed most unused ones. The registry and protocol hashes are committed before any discharge is requested. At most 80 candidate mapping attempts are made and no gauge is replaced after outcomes are seen.

The evaluation then fits the selected candidate on development rows from 2017 through 2021, calibrates on development 2022, refreezes its thresholds on the 2022 validation population under the same budget rule, and predicts the new catchments once for decision dates from January 8, 2023 through September 25, 2024, reusing the approved daily mean discharge parser, pre-2016 reference and onset definition. Geography is new; the period is not. The same seven criteria are applied once on the new cohort, and passing supports a follow-up study rather than deployment.

## Limits

The target is high flow above a pre-2016 p95 after seven quiet days, not verified flooding, flood stage or worksite damage. Archived forecasts are reforecasts in places and the dissemination buffer is assumed. Final discharge records contain retrospective revisions. Neither fold is independent of the earlier studies' development history. Coverage is daily means, so short flash floods and sub-daily peaks are invisible.

## Resources and stopping

CPU only, two threads, one worker, no new dependency, no new weather download, and the fresh cohort limited to at most 80 metadata attempts. Stop after the predeclared comparisons and the single fresh evaluation; do not tune against inspected results. Run the repository tests, document source and artifact hashes, and keep production models and mandatory safety rules unchanged.

Commands, from the repository root with the existing virtual environment:

```sh
.venv/bin/python -m pytest -q tests/test_revision_research.py tests/test_revision_worker.py
.venv/bin/python -m scripts.revision_research
.venv/bin/python -m scripts.revision_worker select
.venv/bin/python -m scripts.revision_worker report
```
