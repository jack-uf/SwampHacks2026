# Measured wind-hazard experiment: results

This sprint trained on measured environmental hazards instead of county storm reports. The target is the observed maximum gust at an airport weather station reaching 60 km/h in a UTC day, predicted from archived GFS forecasts one day ahead. The frozen protocol is [WIND_SPRINT_PROTOCOL.md](WIND_SPRINT_PROTOCOL.md); every number below comes from `experiments/sprint-20261017-wind/`.

## Answer

A fitted model improved *ranking* over the raw forecast rule on both fresh populations, and it caught more hazardous days at familiar stations with the same number of false alarms. It failed the frozen integration gates on unseen stations, where recall fell below the forecast rule and below the 30 percent floor, and it did not reduce false alerts on the temporal population. The campaign decision is **retain the forecast threshold** and treat the probability as an experiment, not a served signal. Nothing in the application reads this model.

Operationally relevant byproducts: the archived forecast at 60 km/h was right about half the time (precision 0.556 temporal, 0.496 geographic), which quantifies today's over-alerting; the fitted probability was well calibrated in range (temporal ECE 0.013) and its alerts carried a median 37 hours of lead time.

## Question and population

The served county-report flood and storm models were not touched, and no application code was changed. The experiment asks whether a fitted probability of a measured gust exceedance beats the archived GFS gust forecast itself on later dates and unseen stations.

Counties frozen in the previous sprint selected the stations: for each of the 32 counties, the nearest eligible station within 100 km in that state's Iowa Mesonet ASOS network. Thirty-one stations were claimed (15 development, 8 geographic validation, 8 geographic test). Washoe County, NV was dropped because its nearest eligible station is 118.3 km from the county internal point; Reno-Tahoe airport is over 130 km from that point. Claim order and drops are recorded in `protocol.json`.

## Data

Measured target: Iowa Mesonet ASOS archive (`cgi-bin/request/asos.py`, gust, report types 3 and 4, UTC). A missing gust value is a report below the threshold, not a missing observation. A station-day requires at least 18 reports and at least one report in each 6-hour UTC quarter; a day is positive when the maximum reported gust reaches 60 km/h (1 knot = 1.852 km/h exactly).

Predictors: archived GFS seamless Previous Runs `_previous_day2` hourly values at the station coordinates for precipitation, temperature, humidity, gust, sea-level pressure and CAPE. Decision time is 00:00 UTC on D minus 1 for target day D, so every target hour carries a nominal 48-hour lead whose initialization precedes the decision. Actual dissemination timestamps are not reconstructed.

| Source coverage | Value |
| --- | ---: |
| Stations | 31 |
| Complete observation days | 28,020 |
| Merged station-days (both sides complete) | 27,318 |
| Positive station-days | 3,221 (11.8 percent) |
| Date range | 2024-01-24 to 2026-06-29 |

The first requested day is 2024-01-01, but the archived previous-runs forecast response begins finite values at 2024-01-20 12:00 UTC (inspection of the cached ABQ response). With a 72-hour antecedent window the first complete target day is 2024-01-24, matching the withheld start reported for the same provider in the county sprint. Station and day coverage per station are in `provenance.json`.

| Partition | Rows | Positives | Stations | States |
| --- | ---: | ---: | ---: | --- |
| Fit (2024) | 5,115 | 671 | 15 | KS, MT, ND, NE, NM, NV, SD, WI |
| Calibration (2025 H1) | 2,702 | 441 | 15 | same |
| Temporal validation (2025 H2) | 2,747 | 288 | 15 | same |
| Geographic validation (2025 H2) | 1,470 | 119 | 8 | IA, ID, MI, MO |
| Temporal test (2026 H1) | 2,682 | 459 | 15 | same as fit |
| Geographic test (2026 H1) | 1,412 | 139 | 8 | KY, NY, OR, VA |

## Validation and selection

Pooled validation combines the temporal and geographic populations. Every candidate met the recall floor because the threshold rule required at least the forecast rule's recall. Selection was the frozen rule: highest equal-weight mean validation AP among supported candidates, then mean Brier, then name.

| Candidate | Temporal AP | Geographic AP | Mean Brier | Alert rate | Threshold | Supported |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_forecast | 0.4017 | 0.4353 | 0.0685 | 0.100 | 0.2600 | yes |
| xgboost_forecast | 0.4071 | 0.4739 | 0.0676 | 0.094 | 0.2610 | yes |
| lightgbm_forecast | 0.4023 | 0.4734 | 0.0673 | 0.093 | 0.2593 | yes |
| logistic_extended | 0.4898 | 0.5281 | 0.0622 | 0.072 | 0.3132 | yes |
| lightgbm_extended | 0.5201 | 0.5444 | 0.0612 | 0.074 | 0.3430 | yes |
| **xgboost_extended (selected)** | **0.5264** | **0.5478** | **0.0604** | **0.068** | **0.3782** | yes |
| Forecast rule | 0.4225 pooled AP | | | 0.093 | 60 km/h | reference |

The extended feature set beat the forecast-only set for every family, so station identity, pressure, CAPE and antecedent rain carried information the raw gust did not. The selected threshold matched the forecast rule's pooled recall, recovering 166 of 407 pooled validation positives while alerting on 6.8 percent of station-days.

## Independent evaluation

Only the frozen winner, the forecast rule and training-prevalence climatology were scored, once, on the fresh 2026 populations. AP is average precision. False alerts are false-positive station-days per 100 station-days.

| Test | Model | AP | Precision | Recall | Alert rate | False alerts /100 | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal | Forecast rule | 0.561 | 0.556 | 0.523 | 0.161 | 7.16 | n/a | n/a |
| Temporal | Selected model | 0.650 | 0.588 | 0.597 | 0.174 | 7.16 | 0.0924 | 0.013 |
| Temporal | Climatology | 0.171 | 0.000 | 0.000 | 0.000 | 0.00 | 0.1434 | n/a |
| Geographic | Forecast rule | 0.498 | 0.496 | 0.403 | 0.080 | 4.04 | n/a | n/a |
| Geographic | Selected model | 0.529 | 0.760 | 0.273 | 0.035 | 0.85 | 0.0662 | 0.035 |
| Geographic | Climatology | 0.098 | 0.000 | 0.000 | 0.000 | 0.00 | 0.0898 | n/a |

Paired calendar-week block intervals, 300 resamples, seed 42, model minus forecast rule:

| Test | AP difference 95% | Recall difference 95% | False alerts /100 difference 95% |
| --- | --- | --- | --- |
| Temporal | +[0.055, 0.131] | +[0.020, 0.132] | [-1.08, +0.97] |
| Geographic | [-0.007, +0.075] | [-0.244, -0.044] | [-4.23, -1.96] |

Warning lead time, measured from the 00:00 UTC issue on D minus 1 to the first exceeding hour of the target day: every alert carried at least 24 hours. The selected model alerted 274 temporal positives with a 37-hour median and 38 geographic positives with a 35.5-hour median; the forecast rule alerted 240 and 56.

### Gate outcomes

Temporal test failures: false alerts were not at least 10 percent lower than the forecast rule (identical count), and station recall fell more than 0.10 at MKE (0.54 to 0.29), MLE (0.71 to 0.29), MSN (0.67 to 0.42) and TOP (0.74 to 0.52). Gains were large at ABQ (0.23 to 0.66), FAR (0.51 to 0.68), BIS (0.38 to 0.59), ICT (0.80 to 0.88), LNK (0.55 to 0.65) and RAP (0.47 to 0.61).

Geographic test failures: the AP interval includes zero, recall fell below the 0.30 floor and below the forecast rule, and recall dropped more than 0.10 at BUF (0.57 to 0.33), LEX (0.67 to 0.44), LOU (0.73 to 0.55) and RIC (0.29 to 0.14). Precision at those stations stayed high (BUF 0.94, LOU 1.00), so the model became selective rather than wrong. The mid-probability bins were also under-predicted on unseen stations (bin 0.3: predicted 0.35, observed 0.73), which is the transfer problem the recall floor exposed.

## Datasets investigated but not used as the label

- **USGS river flood-stage exceedance.** Feasible and next. USGS NWIS daily values return gage height (probe: site 07022000, HTTP 200). The NWS NWPS per-gauge JSON publishes flood categories in feet keyed by `usgsId` (probe: POPW2, action 38.0, minor 40.0 ft), which supplies the threshold join. Two constraints: the bulk gauge list endpoint timed out at 40 seconds, so gauges must be selected by bbox or state first, and the JSON API allows 10 requests per 5 minutes, so threshold collection needs pacing. Remaining work: gauge selection near the frozen counties, a gage-height datum check against NWPS categories, and a rain-based predictor contract.
- **USGS Real-Time Flood Impacts.** Not usable for training. The API is current-status only: endpoints list reference points and current flooding locations with no time parameter for historical reconstruction. It is a prospective validation source.
- **Iowa Mesonet one-minute observations.** Rejected: the service has no gust variable (HTTP 422, "Unknown variable gust"). Hourly METAR reports, including specials, are the only measured gust archive with multi-year coverage.
- Google Summary of the Day and SWDI radar products remain excluded under the published data audit (`docs/ML_DATA_RESEARCH.md`) for the same availability and provenance reasons established before this sprint.

## Limitations

Station gusts are airport point measurements, not construction-site conditions. The forecast grid is far coarser than a station, and station gusts can exceed grid values. METAR gust groups cap at report resolution and miss between-report peaks; reporting practice and sensor changes are not reconstructed. Every alert carries 24 to 48 hours of nominal lead, which exceeds the shift-scale question the JHA asks; the model says nothing about the next six hours. Thresholds, splits and the 60 km/h label were frozen once and not revisited. The station sample inherits a convenience county sample, and the geographic test covers four states. The archived forecast contract is nominal, not a verified dissemination log. Calibration holds at natural prevalence for the temporal population and under-predicts mid probabilities on unseen stations. The alert-budget and recall-floor gates are experiment rules, not supervisor-approved operating costs.

The first evaluation attempt crashed in the per-station baseline metric helper (km/h scores passed to a probability metric) before any metric value was computed or written; the traceback exposed only a single forecast value in the error text. After the fix, the frozen `code_sha256` in `selection.json` was updated explicitly with a note, and the evaluation ran once. No model, threshold, feature or split was chosen from test data.

## Integration recommendation

Keep the served storm screening on the archived forecast band. Do not wire this probability into the JHA. If the measured-hazard line of work continues, the two supported next steps are: (1) run the river flood-stage experiment, which has an authoritative measured label (gauge height against a published flood stage) and a real operational gap (the app's flood lines are deterministic screening heuristics); (2) if gust work resumes, extend the fitting geography before touching any gate, because the failure mode here is transfer, not fit. A future integration, if a new independent experiment passes the same style of gates, would be a clearly labeled experimental second opinion that never changes a rating, matching how Jev suggestions are treated.

## Reproduce

```sh
.venv/bin/python -m scripts.wind_sprint stations   # writes protocol.json (already frozen)
.venv/bin/python -m scripts.wind_sprint ingest     # cached archives under data/raw/
.venv/bin/python -m scripts.wind_sprint select
.venv/bin/python -m scripts.wind_sprint evaluate
.venv/bin/python -m scripts.wind_sprint verify
.venv/bin/python -m pytest tests/test_wind_sprint.py -q
```

Artifacts: `training.csv.gz`, `provenance.json`, `protocol.json`, `selection.json`, `selected.joblib`, `evaluation.json`, `predictions.csv.gz` and the exclusive evaluation-start marker. The experiment writes nothing outside its directory and the shared download cache.

