# Transformer experiment: results

This sprint asked whether a tabular transformer predicts measured wind-gust exceedances better than gradient boosting and the archived GFS forecast, especially at geographically unseen stations. TabPFN v2 and a frozen FT-Transformer were compared against the wind sprint's logistic regression, XGBoost and LightGBM under the same frozen protocol. TabPFN ranked best everywhere and cut false alerts sharply, but the pre-registered recall gate failed on the fresh window, so the decision is **retain the forecast threshold**: nothing is integrated, and no application code changed.

On the freshly acquired July to mid-October 2026 window at the 15 development stations (1,587 station-days, 156 positives, 9.8 percent), TabPFN's average precision was 0.384 against the forecast rule's 0.209, with the paired interval at +[0.123, 0.224] and false alerts of 2.58 against 7.50 per 100 station-days. Its recall at the frozen threshold was 0.231, below both the 0.30 floor and the rule's 0.237, and five stations lost more than 0.10 recall. At the pre-registered matched-budget operating point its recall was 0.340 against the rule's 0.237, so the ranking gain is real and the threshold transfer is the failure. The FT-Transformer finished mid-pack on ranking (fresh AP 0.291) with the highest fresh recall (0.340) at twice the rule's false-alert rate. Nothing here justifies replacing a deterministic threshold.

## Hardware benchmark: MPS against CPU

Benchmarked before continuing, one configuration per process, the same 500-row prediction slice, the worker environment, TabPFN v2 at its default settings except for the device.

| Configuration | Fit | Predict 500 rows | Repeat | Deterministic | Peak resident | Device memory |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| CPU, 1,000 fitting rows | 0.1 s | 10.5 s | 10.5 s | yes | 1.17 GB | n/a |
| CPU, 5,000 rows (override) | 0.1 s | 84.1 s | 82.7 s | yes | 2.03 GB | n/a |
| CPU, 5,115 rows (override) | 0.2 s | 92.9 s | 89.0 s | yes | 2.05 GB | n/a |
| MPS, 1,000 rows | 0.1 s | 2.9 s | 1.7 s | yes | 0.58 GB | 28 MB |
| MPS, 5,000 rows | 0.2 s | 12.0 s | 11.7 s | yes | 0.62 GB | 28 MB |
| MPS, 5,115 rows | 0.2 s | 12.5 s | 12.1 s | yes | 0.62 GB | 28 MB |

- **PyTorch MPS support:** confirmed on this M3 Pro under PyTorch 2.14.1. TabPFN v2 accepts the mps device and produced identical repeated predictions. CPU mode refuses more than 1,000 fitting rows by default; the GPU path needs no override.
- **Speed:** MPS is about 7.4 times faster than CPU at the frozen 5,115-row fitting population (12.5 s against 92.9 s per 500 predictions) and about three times faster at 1,000 rows.
- **Memory:** 0.6 GB resident and 28 MB of device memory on MPS against 2.0 GB resident on CPU. The 18 GB budget was never close to binding.
- **Numerical consistency:** CPU and MPS agreed to a 0.002 maximum absolute difference over the sampled predictions with identical ranking; the calibration step removes residual scale differences within the experiment.
- **10,000 rows:** not reachable. The frozen fitting population is 5,115 rows, so no larger fitting set exists in this experiment; memory is flat across the tested range (28 MB device at both 1,000 and 5,115 rows), so a 10,000-row population would fit comfortably.
- **Stop-early decision:** CPU at 5,000+ rows was marked impractical after 84 to 93 seconds per 500 predictions, and a full-size CPU evaluation attempt was aborted after 10.5 minutes with an estimated twenty minutes of prediction work still ahead for that single pass. Full-size MPS is the frozen configuration; the FT-Transformer keeps training on CPU because it is a small model that trains in about twenty seconds.

## Execution environment

The instrument changed in two ways, both recorded in the protocol before selection, neither affecting data, features, recipes, thresholds, gates or the selection rule. First, each candidate runs in its own worker process: xgboost or lightgbm initializing OpenMP before torch in one process segfaults or hangs on this machine, and the reverse order is safe, so the coordinator never imports a model library and every candidate gets a clean process. Second, TabPFN runs on the MPS device at the full fitting population as benchmarked above. Production requirements stayed pinned; the tabpfn install briefly upgraded python-dotenv and it was restored to 1.2.1.

## Data and candidates

The frozen wind dataset supplies fitting, calibration and selection: fit on 2024 development rows, sigmoid calibration on 2025 H1 development rows, selection on 2025 H2 development plus geographic-validation rows. A new extension acquired the same archives for June 27 to October 15, 2026, giving 3,266 fresh station-days of which 1,587 are the fresh temporal window at development stations. The previously inspected 2026 H1 populations are carried as exploratory cohorts only.

| Candidate | Threshold | Temporal AP | Geographic AP | Mean Brier | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| logistic_extended | 0.3132 | 0.4898 | 0.5281 | 0.0622 | |
| xgboost_extended | 0.3782 | 0.5264 | 0.5478 | 0.0604 | |
| lightgbm_extended | 0.3430 | 0.5201 | 0.5444 | 0.0612 | |
| **tabpfn_extended** | **0.3854** | **0.5365** | **0.5817** | **0.0588** | yes |
| ft_transformer_extended | 0.3028 | 0.4783 | 0.5042 | 0.0645 | |
| Forecast rule | 60 km/h | 0.4225 pooled | | | reference |

Worker wall time at selection: logistic 0.0 s, xgboost 0.1 s, lightgbm 0.1 s, TabPFN 87.8 s (19.0 s of it fitting and calibrating), FT-Transformer 21.8 s. Calibration used the shared sigmoid step on 2025 H1 rows at natural prevalence for every candidate.

## Fresh temporal evaluation

The primary population has 1,587 rows and 156 positives. False alerts are per 100 station-days; the matched-budget column uses each candidate's validation-matched threshold.

| Model | AP | Precision | Recall | Alert rate | False alerts /100 | Matched recall | Matched false alerts /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forecast rule | 0.2085 | 0.237 | 0.237 | 0.098 | 7.50 | 0.237 | 7.50 |
| logistic_extended | 0.2814 | 0.317 | 0.212 | 0.066 | 4.47 | 0.288 | 6.81 |
| xgboost_extended | 0.3558 | 0.383 | 0.263 | 0.067 | 4.16 | 0.372 | 7.81 |
| lightgbm_extended | 0.3439 | 0.375 | 0.308 | 0.081 | 5.04 | 0.372 | 8.44 |
| tabpfn_extended | 0.3836 | 0.468 | 0.231 | 0.049 | 2.58 | 0.340 | 5.61 |
| ft_transformer_extended | 0.2905 | 0.282 | 0.340 | 0.118 | 8.51 | 0.340 | 8.51 |

Paired calendar-week intervals for TabPFN against the rule: AP +[0.123, 0.224], recall [-0.084, +0.075], false alerts per 100 [-6.71, -2.76]. All alerts carried at least 24 hours of lead; 36 positives were alerted with a 35.5-hour median. Reliability was reasonable in range (ECE 0.0138) with mild over-prediction in the 0.3 to 0.4 bins.

### Gate outcome

The winner failed on **fresh_temporal**: recall 0.231 is below the 0.30 floor and below the forecast rule's 0.237, and recall fell more than 0.10 at FSD (0.82 to 0.27), ICT (0.44 to 0.06), FAR (0.44 to 0.33), MKE (0.20 to 0.00) and TOP (0.14 to 0.00). The AP interval, alert-rate and matched-budget conditions all passed; a conservative interpretation is that the model's ranking transferred while its probability scale and a handful of station behaviours did not. ABQ moved the other way, from 0.07 to 0.64.

## Exploratory populations

| Cohort | Rows | Positives | Rule AP | TabPFN AP | TabPFN recall | Rule recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh geographic validation | 834 | 45 | 0.1041 | 0.2320 | 0.022 | 0.089 |
| Fresh geographic test | 845 | 25 | 0.1551 | 0.1524 | 0.000 | 0.080 |
| Inspected temporal (2026 H1) | 2,682 | 459 | 0.5609 | 0.6747 | 0.597 | 0.523 |
| Inspected geographic test | 1,412 | 139 | 0.4983 | 0.5105 | 0.288 | 0.403 |

The fresh geographic-validation AP interval excludes zero, but the thresholds calibrated on development stations barely fire on the other station groups, so recall collapses. These cohorts are exploratory: the wind sprint already inspected the 2026 H1 populations, and the fresh geographic rows come from stations that earlier selections had seen. The honest reading is that the transfer problem documented in the wind sprint persists for every model family, transformer included.

## Limitations

The fresh window is about 15 weeks at 15 stations and is not a prospective test. Station-day labels are airport METAR gusts, not site conditions. Thresholds were chosen once on 2025 validation and never re-tuned; the recall failure is partly a threshold-transfer failure rather than a pure ranking failure, and the matched-budget result shows what a differently operated model would have done without having been the promoted configuration. The FT-Transformer is one frozen recipe; a tuned version could rank better, but the protocol banned tuning and the fresh numbers do not suggest a ceiling break. TabPFN's MPS path segfaulted once during verification and passed on retry; that flakiness is a real operational risk and part of why nothing is integrated. Calibration bins at high probabilities are sparse. The TabPFN v2 checkpoint is the permissively licensed one; newer TabPFN weights are non-commercial and were not used.

## Integration recommendation

Keep the forecast threshold. If a probability signal is wanted later, the evidence supports a *new* experiment with a pre-registered operating point chosen for the target population (the matched-budget result suggests that is where the value is), plus a station-level guard, rather than promoting a transformer on selection metrics. TabPFN is the strongest ranking model tested so far and MPS makes it practical to run, so a future sprint with more fitting geography should start from it.

## Reproduce

```sh
.venv/bin/python -m scripts.transformer_sprint extend
.venv/bin/python -m scripts.transformer_sprint select
.venv/bin/python -m scripts.transformer_sprint evaluate
.venv/bin/python -m scripts.transformer_sprint verify
.venv/bin/python -m pytest tests/test_transformer_sprint.py -q
```

Artifacts: dataset-extension, provenance-extension, frames, per-candidate probabilities and models, selection.json, evaluation.json with predictions.csv.gz, and the exclusive evaluation-start marker. The experiment writes only inside its experiment directory and the shared download cache. One post-evaluation housekeeping change moved cohort prediction files into their own directory; the coordinator hash in selection.json was re-pointed explicitly with a note, and a full re-verification reproduced every prediction row and pooled metric.

