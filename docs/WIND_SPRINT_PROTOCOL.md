# Measured wind-hazard experiment: frozen protocol

Frozen October 17, 2026, before any station-day label is scored and before any station is downloaded. The protocol document and station list are committed before ingestion so the ordering is visible in history. This is a bounded retrospective experiment, not field evidence and not a replacement for any served model.

## Question

Can a fitted probability of a **measured** wind-gust exceedance at an airport station beat the archived GFS gust forecast itself, on later dates and on stations never used for fitting?

The served county-report flood and storm models are untouched. This experiment lives in `experiments/sprint-20261017-wind/` and writes only to that directory and to new cache files under `data/raw/`.

## Population and station selection

The 32 counties frozen in `experiments/sprint-20261017/protocol.json` (16 development, 8 geographic validation, 8 geographic test) act as selectors. For each county, the station used is the nearest station within 100 km in that state's Iowa Environmental Mesonet ASOS network (`https://mesonet.agron.iastate.edu/geojson/network/{ST}_ASOS.geojson`). A station is eligible only when its archive begins by December 1, 2023 and it has no recorded archive end.

Counties are processed by role in the order geographic_test, geographic_validation, development, and within a role by (state, county FIPS). A station can carry only one role, so an early claim wins; a later county takes its next-nearest eligible station within the bound. A county with no station left is dropped with a recorded reason. The resulting station list is frozen in `protocol.json` with the document hash and the county-source hash before any gust value is read.

One-minute archive: the Iowa Mesonet one-minute service was probed and rejected. It does not publish a gust variable (HTTP 422, "Unknown variable gust"), so hourly METAR reports are the only measured gust source with a multi-year archive.

## Data and timing

The decision time is 00:00 UTC on the day before the target UTC day (issue date D-1, target day D), matching the existing pipeline. Predictors are the archived GFS seamless Previous Runs `_previous_day2` hourly values at the station coordinates, so every target hour has a nominal 48-hour lead whose initialization precedes the issue time. Actual dissemination timestamps are not reconstructed; the contract is nominal, as in the prior sprint.

The measured target comes from the Iowa Mesonet ASOS archive (`cgi-bin/request/asos.py`, `data=gust`, report types 3 and 4, `tz=UTC`). Gust values are METAR peak gusts in knots, generally the maximum over roughly the ten minutes before the report; peaks between reports can be missed. Duration keywords are not interpretable in this archive, so only the threshold is used.

A station-day requires at least 18 reported hours and at least one report in each of the four 6-hour UTC quarters. Days that fail this are withheld for every candidate and baseline, never imputed. The same rule applies to forecast completeness: 24 complete target hours and 72 complete antecedent hours, otherwise the day is withheld.

### Target definition

A station-day is positive when the maximum reported gust reaches **60 km/h** (1 knot = 1.852 km/h exactly). This matches a common gust stop-work range for elevated work and sits inside the app's existing storm band. The unit conversion is applied to the knots in the archive; no rounding to whole knots beyond what METAR already publishes.

Pre-ingestion correction, October 17: the first ingestion attempt crashed on a duplicate keyword before writing a dataset, and its day builder treated a report without a gust value as a missing observation. METAR omits the gust group when no gust occurs, so a missing gust value is a report below the threshold, not a missing observation. From the corrected ingestion onward, day completeness counts reports, and a report without a gust counts as below threshold. No station, threshold, split, feature, recipe or selection rule changed, and no model was fitted before this correction.

## Feature sets and recipes

Fit on development stations, 2024-01-04 through 2024-12-31. Sigmoid-calibrate logit probabilities on development stations, January to June 2025, at natural prevalence. Choose on development stations July to December 2025 pooled with geographic-validation stations July to December 2025.

| Set | Features |
| --- | --- |
| forecast (4) | target-day maximum forecast gust, target-day mean forecast gust, season sine, season cosine |
| extended (16) | forecast set plus target-day rainfall, maximum hourly rainfall, wet hours, 72-hour antecedent forecast rainfall, maximum temperature, mean humidity, minimum sea-level pressure, 24-hour pressure change, maximum CAPE, latitude, longitude, station elevation |

Recipes are the fixed, seeded recipes already used in `scripts/train.py`: class-balanced standardized logistic regression, XGBoost and LightGBM with the same hyperparameters. No hyperparameter search. Six candidates in total.

## Baselines

- **Forecast rule:** alert when the archived forecast maximum gust reaches 60 km/h; ranking score is the forecast gust itself. Alerting at the same physical threshold is what the application effectively does today, so this is the operational comparator.
- **Climatology:** the development fitting prevalence as a constant.

## Selection rule

For each candidate, choose the highest pooled-validation threshold that attains at least the forecast rule's pooled-validation recall, with a recall floor of 0.30 and an alert-rate ceiling of 0.20. Candidates whose ceiling cannot be satisfied are recorded unsupported. Among supported candidates, select the highest equal-weight mean of temporal and geographic validation average precision (AP), breaking ties by lower mean Brier score, then recipe name. Freeze the winner, threshold, features and hashes before any January to June 2026 row is scored.

## Independent evaluation and gates

Only the frozen winner, the forecast rule and climatology are scored on the fresh January to June 2026 populations: development stations (temporal test) and geographic-test stations. Report AP (the PR-AUC summary), precision, recall, Brier score, ten-bin reliability with counts, alert rate, false alerts per 100 station-days, station and state detail, and the warning lead-time distribution (first exceeding hour of the target day measured from the 00:00 UTC issue on D-1, so every alert carries at least 24 hours).

Paired uncertainty uses 300 calendar-week block resamples, seed 42, keeping all stations of a week together.

The candidate is worth integrating only if **all** hold in **both** fresh populations:

- At least 20 positive station-days and an alert rate of at most 20%.
- AP-difference interval lower endpoint above zero against the forecast rule.
- Recall at least 0.30 and no lower than the forecast rule.
- False alerts at least 10% lower than the forecast rule.
- No station with at least five positives loses more than 0.10 recall against the forecast rule.

A pass permits proposing an experimental integration with a separate change; a fail retains the current forecast-threshold screening. Gates are experiment rules, not supervisor-approved operating limits.

## Stop conditions

Run this one fixed experiment, report every selected-model outcome including regressions, and stop. Do not start a second search on these tests. Do not commit the winner into the serving path in this sprint.

## Other datasets investigated

- **USGS river flood-stage exceedance.** USGS NWIS daily values return gage height (probe: site 07022000, HTTP 200). The NWS NWPS per-gauge JSON publishes flood categories in feet keyed by `usgsId` (probe: POPW2, action 38.0 / minor 40.0 ft), which supplies the threshold join. The bulk gauge list endpoint timed out at 40 seconds, and the JSON API is limited to 10 requests per 5 minutes, so gauge selection and threshold collection need a paced batch step. Feasible next target, deferred in this sprint for time.
- **USGS Real-Time Flood Impacts.** The API is current-status only: its endpoints list reference points and the locations flooding right now, with no time parameter for historical reconstruction. It cannot label past floods and is recorded as a prospective validation source, not training data.
- **One-minute observations.** Rejected above: no gust variable.

## Limits

Station gusts are point measurements at airports, not construction-site conditions; the 100 km bound can leave a county with a distant station; the forecast grid is far coarser than a station; METAR gusts under-report between-report peaks; reporting practice and sensor changes are not reconstructed; the 60 km/h threshold is one choice; and the experiment reports convenience-sampled geography.

