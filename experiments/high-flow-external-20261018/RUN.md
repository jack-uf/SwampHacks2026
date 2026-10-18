# External benchmark (priority 2): Google architecture under our protocol

Status: complete on 2026-10-18. No external method beats the frozen cycle-11 candidate.

The Google Flood Hub encoder-decoder LSTM family, adapted to this repository's six-hour decision
population, was trained from scratch under the frozen protocol (train 2020-2021, calibrate 2022,
threshold 2023, score 2024) and compared with the frozen cycle-11 `winner_forecast` on identical
rows and matched operating points, with the same conditional calibration, the same inclusive
pooled-1/regional-2 threshold rule and the same sixteen promotion gates. Everything is labelled
"Google architecture under our protocol": it is not Google's published performance, and the
released pretrained weights were not applied to our gauges; `audit.json` records why.

Machine-readable record in this directory: `audit.json` and `audit/` (evidence copies with hashes),
`freeze.json` (protocol before fitting), `series.json`, `results.json`, `comparison.json`,
`predictions/` with per-row probabilities for every scored partition, and
`superseded-fresh-discharge/` (a development note, section 4.3). New files only; no existing script
or experiment was modified.

## 1. Overlap audit of the released Google weights

Sources: the reference clone `/tmp/planit-ext/flood-forecasting`, the released pretrained-weights
README, the shipped run config and the released full-period metrics table. All are copied under
`audit/` with hashes in `audit.json`.

| Item | Released value |
| --- | --- |
| Model | `mean_embedding_forecast_lstm`, CMAL head, hidden size 512, 125 configured epochs |
| Basins | 15,954 lines in the shipped basin list; the README states 15,955 listed and 10,137 evaluable with observations; the released metrics table has 10,137 rows |
| Training window | 1982-01-01 to 2023-09-30, with training, validation and test all on the full period and no temporal holdout (shipped config; the README states the same) |
| Hindcast inputs | 9 daily channels: hres net solar radiation, hres net thermal radiation, hres surface pressure, hres temperature 2m, hres total precipitation, graphcast temperature 2m, graphcast total precipitation, imerg precipitation, cpc precipitation |
| Forecast inputs | 7 daily channels: the hres and graphcast sets above |
| Statics | 84 Multimet attributes; the shipped feature scalers require the exact released set |
| Sequence contract | seq_length 365 daily steps, lead_time 7 daily steps, predict_last_n 8, forecast_overlap 365, daily streamflow targets |

Gauge overlap: all 88 of our gauges appear in the training basin list as `camels_<id>`, and all 88
appear in the released metrics table. Those released numbers are a full-period in-sample
evaluation; over our 88 gauges the median NSE is .7393 (10th percentile .4573, 90th percentile
.8601) and the median KGE is .7051. They support no claim about our task or our 2024 rows.

Year overlap with our split:

| Partition | Window | Overlaps the weights' training window |
| --- | --- | --- |
| Train | 2020-01-01 to 2021-12-31 | yes, fully |
| Calibration | 2022-01-01 to 2022-12-31 | yes, fully |
| Threshold | 2023-01-01 to 2023-12-31 | partly, 2023-01-01 to 2023-09-30 |
| 2024 development partitions | 2024-01-01 to 2024-12-31 | no |
| 2025-2026 confirmation | 2025-01-01 to 2026-12-31 | no |

Conclusion. The released weights cannot be applied to our 88 gauges under this protocol without
training overlap contamination: every gauge is in the training list, and the weights have seen the
2020 through 2023-09 records that our train, calibration and threshold stages consume. The 2024
score partitions and the 2025-2026 confirmation years are outside the training window, but the
daily input contract cannot be built from the caches either. None of the required
hres/graphcast/imerg/cpc channel names exists in the weather cache (only ERA5 precipitation,
temperature 2m and 0-7 cm soil moisture are cached, two of the nine hindcast quantities under
different names), no daily as-issued forecast product is cached, six of the 84 statics match our
cached CAMELS attribute names, and the first train decision on 2020-01-08 has 38 days of cached
history against the required 365. There is no defensible zero-shot or fine-tuned application here,
so this sprint trains the architecture from scratch under our protocol.

## 2. RiverMamba released reforecasts

Sources: the RiverMamba clone README and the public Hugging Face dataset listing, fetched
2026-10-18 as metadata only (the AIFAS archives are 177 and 181 MB compressed and the full-map
archive is 27.2 GB, so nothing was downloaded).

The reforecasts are daily. Each archive holds one NetCDF file per issue day at 00:00 UTC with 7
time steps for the 7 days of lead time; the variable is `dis24`, discharge averaged over 24 hours;
the AIFAS files carry 3,366 river points indexed to GRDC stations, and the full map carries
6,221,926 points at 0.05 degrees for 2024. A daily cadence with a one-day minimum lead cannot
express a six-hour decision or a six-hour onset horizon, and the AIFAS index carries GRDC station
identifiers rather than our 88 USGS gauge identifiers. No comparison was run, per the sprint
instruction to stop at the daily finding.

## 3. What was run

The released Mean-Embedding-Forecast-LSTM itself cannot run on these caches, so the closest honest
external comparison is the Google encoder-decoder conventions trained from scratch on our rows
under our protocol. Two candidates share one trunk:

| Candidate | Inputs |
| --- | --- |
| `google_lstm_engineered` | the raw hourly window plus the frozen candidate's engineered feature set (51 features) and the static attributes, each through its own embedding |
| `google_lstm_raw` | the raw hourly window only |

The window is the same four-channel series the sequence sprint built, 168 hours ending at the
decision time: log1p(discharge/reference), precipitation, temperature 2m and soil moisture 0-7 cm.
Two parity rules make the window carry the same hours the frozen candidate's inputs carry: the
discharge channel is delayed two hours so its newest value in every window equals the frozen
receipt-delay `q_2`, and the weather channels keep the six-hour receipt delay. The two-hour receipt
delay is the same assumption the frozen candidate rests on; it is not certified live availability.

Trunk: a per-timestep Linear(4 to 16) with tanh, one-layer LSTM with hidden size 64, and, when
present, tanh embeddings for the engineered features (32) and the statics (16), concatenated into a
dropout-0.2 head (Linear to 32, then a single logit). Training: AdamW at 3e-3 with weight decay
1e-4 and gradient clipping at 5, batch 512, at most 30 epochs, patience 5, early stopping on a
contiguous tail of the last 10 percent of training rows by decision time (76,995 inner-training
and 8,555 inner-validation rows; 8 epochs run with 26,497 parameters for the engineered candidate,
11 epochs with 23,185 parameters for the raw candidate, 198.5 and 251.6 seconds). Calibration:
StandardScaler plus LogisticRegression(C=1, max_iter=2000, random_state=42) on the score and the
four conditional inputs (season_sin, season_cos, margin, log_area), fitted on the 2022 partition
with a positive slope required. Threshold: the inclusive pooled-1 per hundred, regional-2 per
hundred rule on the shared 2023 partition, selected separately for each model. Scoring: the 2024
temporal and geographic partitions on identical rows, paired bootstrap intervals over 48 HUC4
clusters at 500 draws, and the sixteen promotion gates of `scripts/report_high_flow_signals.py`
computed against the frozen candidate.

Commands, in order:

    .venv/bin/python -B -m scripts.high_flow_external audit
    .venv/bin/python -B -m scripts.high_flow_external freeze
    .venv/bin/python -B -m scripts.high_flow_external assemble
    .venv/bin/python -B -m scripts.high_flow_external fit
    .venv/bin/python -B -m scripts.high_flow_external evaluate

Every stored series byte was verified identical to the frozen sequence sprint, and every partition
row identical to the frozen sequence index and to the stored cycle-11 prediction rows.

### 3.1 Control integrity

The frozen candidate was rebuilt from its stored bundle to select the shared 2023 thresholds. Its
threshold reproduces exactly (0.11252824434770284), and its recomputed 2024 probabilities match
the stored files to 1.0e-16 (9.999e-17 temporal, 9.996e-17 geographic). The matched operating
points below compare against a bit-faithful reconstruction of the frozen candidate.

## 4. Results

### 4.1 Matched rows, each model at its own rule-selected threshold

| Partition | Model | Events | AP | Brier | FA/100 | Worst HUC2 | Median lead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Temporal 2024 | frozen winner_forecast | 116/306 | .1869 | .005219 | .916 | 06: 2.24 | 3.0 |
| Temporal 2024 | google_lstm_engineered | 108/306 | .1503 | .005335 | .783 | 03: 2.43 | 3.0 |
| Temporal 2024 | google_lstm_raw | 48/306 | .1205 | .005468 | .361 | 17: 2.15 | 2.0 |
| Geographic 2024 | frozen winner_forecast | 56/102 | .1883 | .005124 | 1.397 | 18: 2.68 | 3.0 |
| Geographic 2024 | google_lstm_engineered | 47/102 | .1682 | .005103 | 1.128 | 17: 2.00 | 4.0 |
| Geographic 2024 | google_lstm_raw | 34/102 | .1204 | .005405 | 1.133 | 18: 3.93 | 3.0 |
| Pooled 2024 | frozen winner_forecast | 172/408 | .1824 | .005195 | 1.039 | 17: 2.27 | 3.0 |
| Pooled 2024 | google_lstm_engineered | 155/408 | .1527 | .005275 | .871 | - | - |
| Pooled 2024 | google_lstm_raw | 82/408 | .1129 | .005452 | .558 | - | - |

Paired differences, candidate minus control, HUC4-cluster 95 percent intervals at 500 draws:

| Comparison | dAP [interval] | dEvent recall [interval] | dFA/100 [interval] | Lead delta |
| --- | --- | --- | --- | --- |
| engineered minus frozen, temporal | -.0366 [-.0651, -.0145] | -.0261 [-.0652, +.0105] | -.133 [-.279, +.008] | 0.0 on 90 joint events |
| engineered minus frozen, geographic | -.0201 [-.0790, +.0476] | -.0882 [-.1881, +.0123] | -.269 [-.584, +.050] | 0.0 on 42 joint events |
| engineered minus frozen, pooled | -.0297 [-.0541, -.0097] | -.0417 [-.0832, -.0042] | -.168 [-.309, -.039] | - |
| raw minus frozen, temporal | -.0663 [-.1006, -.0359] | -.2222 [-.2726, -.1759] | -.555 [-.718, -.375] | 0.0 on 44 joint events |
| raw minus frozen, geographic | -.0679 [-.1739, -.0220] | -.2157 [-.3872, -.0696] | -.264 [-.701, +.246] | 0.0 on 31 joint events |
| raw minus frozen, pooled | -.0696 [-.1025, -.0423] | -.2206 [-.2731, -.1677] | -.481 [-.646, -.286] | - |

Nothing is in a candidate's favor at these thresholds. The engineered candidate's temporal AP
interval and pooled AP and recall intervals exclude zero in the control's favor; its geographic
intervals and its false-alert intervals include zero; the raw candidate's pooled intervals exclude
zero in the control's favor on all three metrics.

### 4.2 Matched operating points on the 2023-selected budget ladder

Each model's threshold is selected on the shared 2023 partition at a common pooled false-alert
ceiling with the regional ceiling fixed at 2 per hundred. The regional ceiling binds in this
population, so the ladder collapses to two distinct operating points: pooled 0.5, and pooled 1.0,
which equals each model's primary threshold and stays the same at 1.5 and 2.0.

| Budget | Model | Threshold | Temporal | Geographic | Pooled | Pooled dEvent recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | frozen | .167012 | 87/306, AP .1869 | 40/102, AP .1883 | 127/408, AP .1824 | - |
| 0.5 | engineered | .136340 | 86/306, AP .1503 | 36/102, AP .1682 | 122/408, AP .1527 | -.0123 [-.0670, +.0371] |
| 0.5 | raw | .141484 | 48/306, AP .1205 | 34/102, AP .1204 | 82/408, AP .1129 | -.1103 [-.1604, -.0612] |
| 1.0 | frozen | .112528 | 116/306 | 56/102 | 172/408 | - |
| 1.0 | engineered | .113696 | 108/306 | 47/102 | 155/408 | -.0417 [-.0832, -.0042] |
| 1.0 | raw | .141484 | 48/306 | 34/102 | 82/408 | -.2206 [-.2731, -.1677] |

At the tighter budget the engineered arm's pooled recall gap (-.0123) is not distinguishable from
zero; at the primary budget it is. Matching operating points does not change the ordering.

### 4.3 Development note: the first window implementation

The first implementation kept the sequence sprint window unchanged, ending at the decision hour
inclusive, which handed the candidates two hours of fresher discharge than the frozen candidate is
allowed to read. Its artifacts are preserved under `superseded-fresh-discharge/` and are not the
delivered comparison. Under that window the engineered candidate scored 121/306 temporal events at
AP .1788 and 55/102 geographic at AP .1840, and the raw candidate scored 177/306 at AP .3343 and
71/102 at AP .3506. The parity revision described in `freeze.json` (discharge delayed two hours)
moved the raw candidate to 48/306 at AP .1205 and 34/102 at AP .1204, and the engineered candidate
to 108/306 at AP .1503 and 47/102 at AP .1682. The apparent raw-candidate win came from the two
freshest discharge hours, not from the architecture, which is consistent with the sequence
sprint's finding that information, not architecture, was the binding constraint.

### 4.4 The sixteen promotion gates against the frozen candidate

| Gate | engineered | raw |
| --- | --- | --- |
| at_least_20_events_per_partition | pass | pass |
| at_least_50_events_total | pass | pass |
| temporal_test_events | pass | pass |
| temporal_test_ap | fail | fail |
| temporal_test_event_gain | fail | fail |
| temporal_test_brier | fail | fail |
| temporal_test_false_alerts | pass | pass |
| temporal_test_lead | pass | pass |
| geographic_test_events | pass | pass |
| geographic_test_ap | fail | fail |
| geographic_test_event_gain | fail | fail |
| geographic_test_brier | pass | fail |
| geographic_test_false_alerts | pass | fail |
| geographic_test_lead | pass | pass |
| total_events | pass | pass |
| pooled_event_uncertainty | fail | fail |

The engineered candidate passes 10 of 16 gates; the raw candidate passes 8. Both fail every AP and
event-gain gate and the pooled uncertainty gate. A pass set is not a promotion: the comparison
reference here is the frozen candidate itself.

## 5. Verdict

No external method beats the frozen candidate. The engineered arm is the closest: fewer temporal
events (108 against 116), lower AP on both partitions, and pooled recall and AP intervals that
exclude zero in the control's favor, though it carries a slightly lighter false-alert load and a
marginally better geographic Brier (.005103 against .005124). The raw arm is clearly behind (48
temporal and 34 geographic events). Matching operating points does not change the ordering, and no
candidate dominates the frozen candidate anywhere. The honest external result for this sprint is
therefore twofold: the two released external systems are incompatible with the six-hour onset task
on our caches (section 1 and section 2), and a from-scratch adaptation of the Google architecture
under our protocol loses to the frozen candidate.

## 6. Limitations

- These are inspected 2024 development partitions. The frozen candidate was selected using them
  across cycles 0-13, so this comparison is not independent confirmation; the retained 2025-2026
  confirmation set is untouched by this sprint.
- One training seed (42) per candidate; no seed-stability study.
- The adapted trunk borrows the release's encoder conventions; it is not the released
  Mean-Embedding-Forecast-LSTM, and no claim about Google's published skill follows from it.
- Windows are retrospective information experiments (approved observations and ERA5 reanalysis
  under the frozen receipt-delay assumptions), not as-issued feeds.
- The paired intervals resample 48 HUC4 clusters at 500 draws and do not include
  threshold-selection uncertainty.
- The engineered branch standardizes with inner-training statistics and zero-fills missing
  entries, including the mostly empty hrrr forecast columns; the raw branch's only fills are the
  counted series-start head introduced by the two-hour discharge delay (98 train, 70 threshold and
  2 geographic entries).

## 7. Files

| Path | Contents |
| --- | --- |
| `scripts/high_flow_external.py` | audit, freeze, assemble, fit and evaluate driver (torch-free) |
| `scripts/high_flow_external_worker.py` | torch worker, one candidate per subprocess |
| `audit.json`, `audit/` | overlap audit with hashes and evidence copies |
| `freeze.json` | protocol manifest, revision note, code hashes, control hashes |
| `series.json` | per-gauge series hashes, partition indices, window checks |
| `results.json` | both candidates: thresholds, calibration, training records |
| `comparison.json` | paired intervals, matched budgets, sixteen gates, control integrity |
| `predictions/` | per-row probabilities, calibration, threshold, temporal and geographic partitions |
| `superseded-fresh-discharge/` | the first window implementation's artifacts, development note only |

