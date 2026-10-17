# Transformer experiment: frozen protocol

Frozen October 17, 2026, before any transformer is fitted and before the July to October 2026 rows are ingested. The document and its code are committed before the fresh window is downloaded so the ordering is visible in history. This is a bounded retrospective experiment on a research prototype, not field evidence.

## Question and freshness contract

Can a tabular transformer improve measured wind-gust prediction over gradient boosting and over the archived GFS forecast, particularly on geographically unseen stations?

The wind sprint already inspected its 2026 H1 temporal and geographic tests, and its selection saw everything through June 30, 2026. Those rows are **exploratory** here and cannot support a fresh claim. The genuinely fresh evidence is the window that no earlier selection has seen:

- **Fresh temporal test (primary):** station-days at the 15 development stations with target dates from July 1, 2026 through the last complete day the archives provide (requested through October 15, 2026). New weeks, familiar stations.
- **Fresh rows at validation and test stations (exploratory):** the same window at the 8 geographic-validation and 8 geographic-test stations. The stations themselves were inspected earlier, so station-level generalization is no longer a fresh claim.
- **Previously inspected 2026 H1 populations (exploratory):** carried for continuity only.

Nothing in the application reads any model from this sprint.

## Candidates

| Name | Definition |
| --- | --- |
| Forecast rule | Archived forecast maximum gust at 60 km/h; ranking score is the forecast gust |
| Climatology | Fitting prevalence as a constant |
| logistic_extended, xgboost_extended, lightgbm_extended | The frozen recipes and extended feature set from the wind sprint, refitted here |
| tabpfn_extended | TabPFN v2 classifier, raw extended features, device cpu, random_state 42 |
| ft_transformer_extended | The frozen FT-Transformer below |

TabPFN v2 is a pretrained tabular foundation model; fitting puts the training rows into its context, and no task-specific weights are learned. Raw features are passed because TabPFN's own preprocessing handles scale. Every probabilistic candidate then goes through the identical shared step: logit transform of the raw probability, sigmoid calibration fitted on the 2025 H1 development rows at natural prevalence.

Pre-selection clarifications, October 17, before any candidate was fitted for selection. TabPFN's CPU guard rejects more than 1000 training rows by default; the classifier is constructed with ignore_pretraining_limits=True so the frozen 5,115-row fitting population is used unchanged, and the measured runtime is reported with the results. Each candidate also runs in its own worker process because xgboost/lightgbm and torch cannot both initialize OpenMP in one process on this machine (segfault or hang). Process isolation changes execution only, not data, features, recipes, thresholds, gates or the selection rule.

### Frozen FT-Transformer

Feature tokenizer over the 16 standardized extended features (training means and standard deviations only): one learned weight and bias vector per feature, d_token = 32, plus a learned CLS token. Two pre-norm TransformerEncoderLayer blocks (d_model 32, 4 heads, feed-forward 64, ReLU, dropout 0.1). Final LayerNorm and a linear head to one logit. Loss BCEWithLogits with pos_weight equal to training negatives over positives. AdamW, lr 1e-3, weight decay 1e-5, batch 256, at most 60 epochs, shuffle seeded 42, torch seed 42. Early stopping checks the calibration-set BCE each epoch with patience 10 and restores the best state; that check uses the calibration rows only for the stopping decision, and the sigmoid calibration is fitted afterward on the same rows under the shared procedure. No hyperparameter search of any kind. The wrapper exposes fit and predict_proba so the shared calibration applies unchanged.

## Data and timing

Fit, calibration and validation reuse the frozen wind dataset unchanged: fit on 2024 development rows, sigmoid calibration on 2025 H1 development rows, selection on 2025 H2 development rows pooled with 2025 H2 geographic-validation rows. The extension requests the same archives with the same frozen parsing and completeness contract (helpers in scripts/wind_sprint.py): Iowa Mesonet ASOS gusts and archived GFS Previous Runs previous_day2 features, requested from June 27, 2026 through October 15, 2026 so that fresh target days begin July 1, 2026. Extension rows must not overlap the frozen dataset; the loader enforces a July 1, 2026 start and a disjoint check against the frozen file. Decision time remains 00:00 UTC on D minus 1.

## Selection rule

Identical to the wind sprint. For each candidate choose the highest pooled-validation threshold attaining at least the forecast rule's pooled-validation recall, with a recall floor of 0.30 and an alert-rate ceiling of 0.20; mark unsupported otherwise. Among supported candidates select the highest equal-weight mean of temporal and geographic validation average precision, breaking ties by lower mean Brier, then name. Freeze winner, thresholds and hashes before any fresh row is scored.

## Matched-budget operating point

Defined once, before evaluation: let F be the forecast rule's false-alert fraction on pooled validation negatives. For each candidate, its matched-budget threshold is the ceil(F times the negative count)-th largest validation negative score, so the validation false-alert count equals the forecast rule's count up to tied scores. On every evaluation cohort report each model's recall and false-alert rate at its matched-budget threshold next to its selection threshold. Actual validation counts at that threshold are recorded, because tied scores can move the count.

## Evaluation and gates

Cohorts: fresh_temporal (primary), fresh_geographic_validation, fresh_geographic_test, inspected_temporal, inspected_geographic_test (exploratory). Every candidate, the forecast rule and climatology are scored on every cohort. Metrics: AP, precision, recall, Brier, ten-bin reliability and ECE, alert rate, false alerts per 100 station-days, confusion matrix, per-station detail, and paired calendar-week block intervals against the forecast rule (300 resamples, seed 42, AP, recall and false-alert differences). Warning lead time is reported as in the wind sprint.

The gate applies to the winner on **fresh_temporal only**:

- At least 20 positive station-days and alert rate at most 20 percent.
- AP-difference interval lower endpoint above zero against the forecast rule.
- Recall at least 0.30 and no lower than the forecast rule at the selection threshold.
- No station with at least five positives loses more than 0.10 recall.
- At the matched-budget operating point, recall at least the forecast rule's recall on the same cohort.

A pass permits proposing an integration with a separate change and the usual safety labelling; a failure retains the existing forecast screening. A transformer is promoted only through the same gates as everything else; sophistication earns nothing.

## Temporal Fusion Transformer

The archived forecast payloads do contain hourly sequences (24 target hours and 72 antecedent hours per station-day), so sequences exist. The TFT is declined as a third path for this sprint: the frozen protocol forbids tuning, an untuned TFT on 5,115 training rows is unlikely to beat the fitted boosters, the documented failure mode is geographic transfer rather than temporal pattern modeling, and CPU-only training of an encoder-decoder attention model with quantile heads would cost hours per fit. If a later sprint has more fitting geography and a GPU budget, the cached hourly arrays are sufficient to build it.

## Licensing, hardware and reproducibility

- Library code: tabpfn 9.2.0 is Apache-2.0.
- Weights: TabPFN v2 classifier checkpoint tabpfn-v2-classifier-finetuned-zk73skhh.ckpt, 29,009,539 bytes, SHA-256 cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49, licensed under the Prior Labs License (Apache 2.0 plus an attribution requirement). The newer TabPFN 2.5, 2.6, 3 and 3.5 weights are non-commercial and were deliberately not used. Weights are cached outside the repository and are not redistributed.
- Hardware: Apple Silicon CPU, 12 cores, no GPU; torch 2.14.1 pinned to 8 threads for the experiment; mlx 0.32.4 installed as a tabpfn dependency but the classifier is pinned to device cpu for determinism.
- Measured before freezing: on 5,115 training rows and 2,747 prediction rows, TabPFN v2 fit takes about 0.3 seconds and predict_proba about 17.6 seconds; device-cpu predictions are bit-identical across repeated calls, and a fitted classifier round-trips through joblib identically.
- Versions recorded in the selection manifest: tabpfn, torch, mlx, scikit-learn, xgboost, lightgbm, numpy, pandas, joblib, scipy. Production requirements.txt is untouched; the tabpfn install briefly upgraded python-dotenv and it was restored to the pinned 1.2.1.
- Reproducibility limits: torch CPU training is not guaranteed bit-exact across machines, and TabPFN inference may differ across library versions; the checkpoint hash and package versions are pinned here.

## Stop conditions

Run the frozen candidates once, report every outcome including regressions, and stop. Do not start a second search on the fresh window. Do not wire a transformer into the serving path in this sprint.

