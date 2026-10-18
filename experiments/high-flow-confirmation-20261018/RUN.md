# High-flow confirmation run, batch 1

Independent confirmation attempt for the frozen six-hour high-flow candidate, executed on the
first HUC2-stratified batch of the fresh-geography cohort. The frozen spec is
`docs/HIGH_FLOW_CONFIRMATION_PROTOCOL.md`; the new implementation is
`scripts/confirm_high_flow.py`. No existing script was modified, and no git commands were run.

**Result: 13 of 16 promotion gates pass; `passed` is `false`.** On 14,254 Stage A decisions and
86 eligible events from 15 stations, the candidate beats the frozen comparator on average
precision (paired 95% interval [.0251, .1371] for the +.0681 difference), Brier score and false
alerts, with no loss of median lead, but its event-recall gain is 4.65 points against the frozen
5-point requirement and the pooled paired lower bound includes zero. Under the frozen gates the
confirmation does not promote the candidate on this batch.

**These outcomes are now inspected.** The 2025-01-01 to 2026-09-30 outcome period acquired here
can never again serve as untouched confirmation evidence. The four stations whose reference
acquisition failed before any discharge page was read have no acquired outcomes in that period,
but the cohort and protocol freeze cannot be reused for them without a new protocol run.

## Exact commands

From the repository root, in order:

```sh
.venv/bin/python -m scripts.confirm_high_flow freeze    # writes freeze.json, before acquisition
.venv/bin/python -m scripts.confirm_high_flow collect   # network access; run escalated
.venv/bin/python -m scripts.confirm_high_flow score     # deterministic, offline
```

`collect` requests `api.waterdata.usgs.gov` and `archive-api.open-meteo.com`. Every response is
cached beside its payload with the exact URL, receipt time and SHA-256
(`sources/<station>/*.request.json`, `weather/*.request.json`), so a rerun replays from cache.
`score` performs no network access. `freeze` refuses to overwrite an existing `freeze.json`.

## Frozen design (all recorded in `freeze.json`, written before acquisition)

- Candidate: `experiments/high-flow-signals-20261018/cycle-11/winner_forecast.joblib`, SHA-256
  `700201d02801c4300ab8b3ff4042c813f506c76272886703fe0127a15b58375f`, threshold
  `0.11252824434770284`. Applied unchanged: same model, features and calibrator.
- Comparator: `.../cycle-11/linear_conditional.joblib`, SHA-256
  `9533b8b7c344b7e51bdacd33308efea00ff4733cba6d7bfb7c7c14fb11cffbd1`, threshold
  `0.044244290724789984`. Applied unchanged: same physical score and calibrator.
- Cohort: first 20 stations of the HUC2 round-robin order over `disjoint_selection` in
  `fresh-geography/camels-screen.json` (file SHA-256
  `63f244dbda177f9eab1a7792071f1fc9c45d127a09dc73eac20d43d2fb15c955`). The rule, the full
  179-station stratified order and the batch list are recorded in `freeze.json`.
- Features mirror `high_flow_signals.configurations(11)` `winner_forecast` and `population(11)`:
  `FEATURES`, the lag-6 weather block, `rain24_ratio` and `rain168_ratio` (denominators
  `2*p_mean` and `7*p_mean`, with `p_mean` from `sources/camels_clim.txt`),
  `soil_change_6d` and `rain_share_6h`, the seven `HYDRO` features, and the HRRR block, which is
  absent for Stage A rows (`hrrr_available = 0`; raw forecast fields and ratios are NaN).
- Stage A only: horizon 6, decision hours 00, 06 and 18 UTC, 2025-01-01 through 2026-09-30.
  Stage B (12 UTC with HRRR fields) was not run and is never pooled with Stage A.
- Metrics: `shift_hydrology_metrics.metrics` at each model's frozen threshold,
  `paired_uncertainty` with 500 paired HUC4-cluster draws (seed 42), and
  `jointly_detected_lead_delta`. The sixteen gates are evaluated exactly as
  `scripts/report_high_flow_signals.py` `promotion()` does; every rule is spelled out in
  `freeze.json` under `gates`.
- Partitions: geographic = every included station; temporal = stations with pre-2023 records
  (instantaneous `00011` series beginning before 2022-01-01 and ending on or after 2021-12-31,
  read from USGS series metadata). All 15 included stations qualify, so the two partitions
  coincide and the pre-registered degeneracy rule applies: pooled statistics and total-event
  counts use the deduplicated union. `comparison.json` records `coincident: true` and the
  `pooled_note`; the temporal gates are duplicates of the geographic gates on this batch.

## What was acquired

| Source | Content | Requests | Raw response bytes |
| --- | --- | ---: | ---: |
| USGS OGC daily series | mean discharge 2000-2015 for the reference p95 | 20 | 77.2 MB |
| USGS OGC continuous series | approved instantaneous discharge, quarters of 2024, 2025 and 2026 | 192 | 978.3 MB |
| USGS series metadata | instantaneous `00011` span probe, one page per station | 20 | 0.1 MB |
| Open-Meteo ERA5 archive | hourly precipitation, temperature_2m, soil_moisture_0_to_7cm, snow_depth; 2019-12-01 to 2026-09-30 | 20 | 47.7 MB |

252 cached requests, 1.10 GB of raw response bytes, stored gzip-compressed: this directory is
61 MB on disk. `snow_depth` is requested (same shape as the development acquisition) and is null
in every hour, matching the frozen `meteorology()` contract. Acquisition summary and per-request
hashes: `acquisition.json`; per-station detail: `coverage.json`.

## Coverage and exclusions

20 planned stations, 15 included, 5 excluded. Included stations contributed between 572 and
1,502 eligible decisions each (14,254 rows total, 86 distinct events; rows
`eligible_decision_exposure_rows_divided_by_4` = 3,563.5 station-days).

| Station | HUC2 | Decisions | Events | Approved hours in window | Reference cfs |
| --- | --- | ---: | ---: | ---: | ---: |
| 14092750 Shitike Crk nr Warm Springs, OR | 17 | 866 | 5 | 8,248 / 16,080 | 180.0 |
| 01669000 Piscataway Creek nr Tappahannock, VA | 02 | 820 | 10 | 8,631 | 88.4 |
| 02300700 Bullfrog Creek nr Wimauma, FL | 03 | 1,430 | 4 | 13,095 | 138.8 |
| 10259200 Deep C nr Palm Desert, CA | 18 | 1,378 | 2 | 11,976 | 2.5 |
| 03165000 Chestnut Creek at Galax, VA | 05 | 737 | 11 | 7,805 | 154.3 |
| 07263295 Maumelle River at Williams Junction, AR | 11 | 1,195 | 6 | 11,784 | 284.0 |
| 06632400 Rock Creek ab King Canyon Canal, NE Arlington, WY | 10 | 572 | 1 | 8,849 | 511.2 |
| 05593575 Little Crooked Creek nr New Minden, IL | 07 | 970 | 13 | 10,598 | 551.7 |
| 01137500 Ammonoosuc River at Bethlehem Junction, NH | 01 | 919 | 14 | 10,113 | 829.6 |
| 09505200 Wet Beaver Creek nr Rimrock, AZ | 15 | 1,502 | 3 | 13,208 | 92.8 |
| 04197100 Honey Creek at Melmore, OH | 04 | 724 | 4 | 7,236 | 922.7 |
| 09081600 Crystal River above Avalanche Creek, CO | 14 | 955 | 1 | 8,466 | 1,200.0 |
| 14141500 Little Sandy River nr Bull Run, OR | 17 | 814 | 7 | 8,272 | 436.8 |
| 01586610 Morgan Run nr Louisville, MD | 02 | 643 | 4 | 7,437 | 87.1 |
| 02198100 Beaverdam Creek nr Sardis, GA | 03 | 729 | 1 | 6,915 | 75.0 |

Exclusions:

| Station | HUC2 | Reason |
| --- | --- | --- |
| 03455500 W F Pigeon R ab Lake Logan, NC | 06 | `missing_reference: Unexpected statistic or unit` |
| 03463300 South Toe River nr Celo, NC | 06 | `missing_reference: Unexpected statistic or unit` |
| 08164600 Garcitas Ck nr Inez, TX | 12 | `missing_reference: Unexpected statistic or unit` |
| 05087500 Middle River at Argyle, MN | 09 | `missing_reference: Insufficient pre-2016 reference` |
| 08380500 Gallinas Creek nr Montezuma, NM | 13 | `no_eligible_decisions` (reference 66.4 cfs; only 702 of 16,080 in-window hours approved, so no decision row qualified) |

The four `missing_reference` stations failed on the daily 2000-2015 mean-discharge page before any
continuous discharge was requested; their 2025-2026 outcomes were not acquired. Station 08380500
did acquire continuous pages but produced no eligible decisions. Twelve of the sixteen HUC2
regions in the batch keep at least one included station (regions 06, 12, 13 and 09 drop out).

## Results

Both partitions are the same 15 stations and 14,254 rows (86 events), so the candidate and
comparator numbers below apply to both; `results.json` repeats them per partition.

| Metric | winner_forecast (candidate) | linear_conditional (comparator) |
| --- | ---: | ---: |
| Detected events | 29 / 86 | 25 / 86 |
| Event recall | .3372 | .2907 |
| Average precision | .1557 | .0876 |
| Brier | .005448 | .005823 |
| Precision (row) | .2071 | .1799 |
| False alerts per 100 decisions | .779 | .800 |
| Worst HUC2 false alerts per 100 | 2.171 (region 05) | n/a |
| Median lead (hours) | 2.0 | 2.0 |

Paired candidate-minus-comparator deltas over 500 HUC4-cluster draws (15 clusters):

| Quantity | Difference | 95% interval |
| --- | ---: | --- |
| Average precision | +.0681 | [.0251, .1371] |
| Event recall | +.0465 | [-.0379, .1369] |
| False alerts per 100 | -.0210 | [-.3170, .2572] |
| Median lead, jointly detected (21 events) | 0.0 | n/a |
| Brier | -.000375 | n/a |

The sixteen gates (`comparison.json`): events per partition (86 >= 20), total events (86 >= 50),
both `_test_ap` (lower bound > 0), both `_test_brier`, both `_test_false_alerts` (pooled .779 <=
1.5, worst region 2.171 <= 3.0) and both `_test_lead` pass. The three failures are
`temporal_test_event_gain` and `geographic_test_event_gain` (4.65 points against the 5-point
requirement) and `pooled_event_uncertainty` (pooled recall lower bound -0.0379 is not above
zero). `passed` is therefore `false`.

## Interpretation and limitations

- The event-recall gain misses its threshold by 0.35 points and its interval is wide; the run is
  a near miss on one gate family, not a blanket failure. It is still a failure under the frozen
  rules, and no threshold, model or metric may be adjusted after seeing these outcomes.
- The temporal partition duplicates the geographic partition because every included station has
  pre-2023 records. The sixteen gates therefore rest on one 15-station evidence set, not two.
- This is batch 1 of 179 selected stations; the remaining 159 stations were not acquired, and
  any future batch requires its own freeze before its outcomes are touched.
- Polygon independence removes overlap and nesting; it does not remove spatial weather
  correlation between stations. The HUC4-cluster bootstrap only partly addresses that.
- The rain features rest on archived ERA5 reanalysis with an assumed six-hour availability; this
  is not as-issued receipt evidence, and HRRR fields do not exist for these rows (Stage A).
- The protocol prose describes the cohort as "Oregon and Washington 78, Upper Missouri 21, Ohio
  21, Texas-Gulf 13, California 11". The screen artifact actually records HUC2 17: 38, 03: 40,
  02: 21, 10: 15, 05: 14, 12: 13, 18: 11 and smaller groups. The artifact is the operative
  cohort; the prose counts need a correction in the protocol document before any reuse.
- One recorded interpretation: the task instruction requires both frozen artifacts applied
  unchanged, so the comparator (including its calibrator and threshold) was not refit on cohort
  rows. An alternative reading of the protocol sentence about a refitted comparator is recorded
  in `freeze.json` and was not executed.

## Integrity record

- Before freezing, the scoring path was validated offline: re-applying both frozen artifacts to
  the cycle-11 development frame reproduces the committed `winner_forecast` and
  `linear_conditional` prediction files with a maximum absolute probability difference of
  ~1e-16.
- `freeze.json` was written before acquisition and regenerated twice with identical cohort,
  artifact hashes, thresholds, features, inclusion rules, partitions and gates: first after
  adding per-station progress output to `collect` (before acquisition began), and then after the
  first score attempt showed that the planned duplicate-concatenation diagnostic cannot be
  computed (the metrics validator rejects duplicate decisions). `predictions.csv.gz` and
  `results.json` were written by that first attempt; `comparison.json` was produced only after
  the final regeneration, and the fixed diagnostic is now the recorded `pooled_note`. The
  pre-fix code SHA-256 began `951c865628efc2d63af6`; the
  frozen code SHA-256 is
  `87a052a00f6def27fc565c7e28a994452f7b4a5a5d8f3d106bd36a2a6c7855f8`.
- No model, feature set, calibrator or threshold was refit, adjusted or re-selected at any point.
  `rows.csv.gz` (SHA-256 `48672aa3bc0c4e52ee4c12e29b0ebf5c1a797526f291a8146a16adc7dace3197`)
  is the exact scored frame; `predictions.csv.gz` carries per-row probabilities for both models.

## Outputs in this directory

| File | Content |
| --- | --- |
| `freeze.json` | cohort, artifact hashes, thresholds, inclusion and partition rules, metric definitions, sixteen gates |
| `acquisition.json` | request counts and hashes, exclusions, row and event totals |
| `coverage.json` | per-station coverage, series spans, references, decisions, events, exclusion reasons |
| `rows.csv.gz` | the scored Stage A feature frame (identity, labels, 51 candidate features) |
| `predictions.csv.gz` | per-row candidate and comparator probabilities, partition flags |
| `results.json` | candidate and comparator metrics per partition at their frozen thresholds |
| `comparison.json` | paired uncertainty, lead, Brier deltas, the sixteen gates and `passed` |
| `sources/`, `weather/` | cached response payloads with request provenance |

## Reproduction

```sh
.venv/bin/python -m scripts.confirm_high_flow freeze
.venv/bin/python -m scripts.confirm_high_flow collect
.venv/bin/python -m scripts.confirm_high_flow score
```

Reruns are deterministic from the cached responses. Reusing these stations or this outcome
period as "untouched" evidence is not possible after this run; a new confirmation needs new
stations from the remaining stratified order and a new freeze written before their outcomes are
acquired.
