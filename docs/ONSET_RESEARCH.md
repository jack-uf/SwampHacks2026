# Emerging high-flow research

This isolated experiment asks whether PlanIT can warn about newly emerging high flow rather than mostly recognizing persistent high water. It implements the committed [protocol](ONSET_RESEARCH_PROTOCOL.md). Results belong in `ONSET_RESEARCH_RESULTS.md` after the final evaluation. Nothing here changes served models, the screening index, mission ratings or production safety rules.

## Data and target

`scripts/onset_research.py` reuses the approved-discharge parser and forecast-memory feature builder from the previous hydrology experiment. Source cache records retain retrieval time, exact request URL, byte count and SHA-256. `experiments/sprint-20261017-onset/provenance.json` identifies reused files separately from new downloads and hashes the two datasets. The fresh test dataset is a separate file, never opened by model fitting.

The station registry is frozen before outcome collection. New gauges belong to HUC4 drainage regions absent from every role in the previous experiment, not just its training set. Their reference flows use accepted observations before 2016. Each reference is a station-specific 95th percentile, not an official flood-stage threshold. Approved daily means can miss short peaks and may contain revisions unavailable historically. The study therefore measures reconstructed high-flow warnings, not an operational replay or verified flooding.

An eligible prediction has seven known quiet daily flows through D-3. Its target is a certified new high-flow episode starting on D+1 through D+3, with seven quiet days before the episode. A separate label requires all daily observations needed to certify the outcome. Missing observations are excluded, never labeled safe. The event date identifies the episode, so repeated warning windows do not inflate event recall. False alerts still count each warning decision.

## Models and alert policies

`scripts/onset_worker.py` runs only LightGBM and scikit-learn, in one isolated process with two computational threads. It fits the ten recipes in the protocol without a hyperparameter search. Raw proximity uses D-3 normalized log flow. Linear extrapolation adds six days of the four-day D-7 to D-3 log-flow trend: `q_3 + 1.5 * q_change`. Rainfall ranking sums forecast leads 3, 4 and 5. Probability calibration uses 2021 onset outcomes; 2022 outcomes select models and thresholds.

The residual trajectory candidate fits nine regressors: three lead days times three quantiles. Each predicts change from D-3 log-normalized flow. Sorting the three predictions per lead enforces quantile order. The logistic onset head receives nine absolute quantiles, three interval widths and the latest flow and trend. Only the regressors use fitting years; the head uses 2021. It gets no second calibration fit. This is a testable hybrid, not a claim of architectural novelty over the literature.

Thresholds are fixed from pooled 2022 selection data. A regional policy takes the largest threshold among selection HUC2s. Neither policy has an unseen-region false-alert guarantee. The separate adaptive policy combines a fixed empirical source prior with station-local negative scores whose outcomes are at least eight days old. It can adapt using delayed historical labels; it is not the label-free transfer result. Its strict comparison suppresses alerts tied at the upper weighted percentile.

Bootstrap AP differences use the existing tie-preserving weighted AP implementation. Both candidates receive the same cluster draws. Unique-event recall counts an episode once even when it has several prediction windows. In a month bootstrap, an episode's weight is the maximum multiplicity of its retained decision windows, and it is detected if any retained window warns. This makes boundary episodes explicit rather than silently counting repeated windows as separate events. Intervals are descriptive uncertainty estimates over this finite sample, not guarantees for future climates or watersheds.

## Reproduction and audit

Run from the repository root with the existing research Python environment, including `numcodecs` for the original compressed forecast cache:

```sh
.venv/bin/python -m pytest tests/test_onset_research.py -q
.venv/bin/python -m scripts.onset_research collect
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python -m scripts.onset_worker select
```

Commit `selection.json`, source manifests, implementation and tests before running:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 .venv/bin/python -m scripts.onset_worker evaluate
```

Collection refuses an already frozen provenance record. Selection refuses an existing selection record. Final evaluation refuses an existing evaluation and checks implementation, model, dataset and provenance hashes. Do not delete these guards to rerun a study after seeing test results. A new experiment needs a new protocol and untouched outcomes.

Raw source files, training/test tables, fitted artifacts and prediction tables are ignored by Git. Committed manifests and reports identify them by hashes. Reproduction requires network access for public sources and the dependencies recorded in selection metadata. Preserve the previous experiment cache because this study intentionally shares its immutable source bytes.

The focused tests cover certification and missing days, local-day warning lead, exclusion of fresh geography from fitting, development-year buffers, delayed feedback, conservative ties, quantile ordering and unique-event accounting. Existing production tests remain required before delivery. Dataset hashes and the preserved-artifact checks in the frozen protocol establish that this experiment does not replace the application's models or source data.
