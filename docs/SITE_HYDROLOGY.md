# River intelligence and experimental high-flow inference

PlanIT can retrieve current river discharge at a facility or mission work point, explain its network relationship, and cite the captured response in a JHA. The six-hour LightGBM remains an unpromoted research component. Operational model predictions are withheld, and river context changes neither hazard ratings nor work-window rankings.

The implementation lives in `backend/hydrology.py`, with a shared panel in `frontend/src/HydrologyPanel.tsx`. The [model research results](HIGH_FLOW_RESEARCH_RESULTS.md) own predictive-performance claims. This guide owns the live integration contract and its measured coverage. The [confirmation opportunities](HYDROLOGY_CONFIRMATION_OPPORTUNITIES.md) identify acquisition work required before promotion.

## Application behavior

Open a facility or mission and select **Check live river data**. The panel reports USGS discharge, observation time, provisional or approved status, observation age, six-hour discharge change and a source link. A mission uses its actual work coordinates, which may differ from the facility point. Each point has a separate cache key.

USGS NLDI identifies the catchment containing that point, then searches upstream tributaries and the downstream mainstem within fifty network kilometres. The service screens every returned candidate for fresh instantaneous discharge. It fetches eight days of history at up to three active gauges, ordered by straight-line distance. The panel distinguishes upstream, downstream and same-indexed-reach relationships.

These are catchment-level network candidates. The search does not establish flow arrival time, a precise work-area drainage boundary, intervening regulation, or inundation. An upstream tributary may represent only part of the site's contributing water. A downstream gauge does not confer upstream warning lead. Displayed distance is straight-line distance, not network distance. [USGS navigation limitations](https://api.water.usgs.gov/docs/nldi/navigation/), [basin and hydrolocation semantics](https://api.water.usgs.gov/docs/nldi/basin/).

A river check adds fresh readings to the flood JHA entry as `observed`, with `counts: false` and no rating. Provisional status stays in the text. The evidence references an exact saved response. Missing or expired context is `unavailable`. The existing NWS gauge categories, weather screening, lightning restrictions and supervisor decisions remain authoritative.

The work-window API includes the latest point-specific river context alongside its existing comparison. Its digest and ranking exclude that context because the experimental model has no validated authority over either. The panel appears beside work-window planning, prompting drainage and access review. No river discharge value or experimental score independently restricts, clears or reschedules work.

## Data availability and withholding

The continuous-data contract accepts only station-matched discharge parameter `00060`, instantaneous statistic `00011`, and `ft^3/s`. It rejects multiple series, duplicate timestamps, future observations, incomplete pagination and unexpected units. Non-numeric and negative values provide no measurement. Qualified observations are excluded. Unqualified provisional measurements remain visible as observations and never satisfy approved-only ML inputs. [USGS continuous schema](https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/schema?f=html).

An observation becomes stale after two hours. Point context expires after fifteen minutes; network metadata caches for one day and the current-discharge inventory for five minutes. Failed discharge captures produce a new failure receipt. The service does not substitute an older successful response for that failure. A transport failure therefore stays unavailable.

The browser updates displayed age and expiration while the panel remains open. It clears old data when a new request fails. Changing sites or missions resets the panel. Users can inspect the captured source response through its receipt link. Receipt age measures how old a reading was when retrieved; it cannot identify first publication or telemetry transmission delay.

The experimental path uses the latest scheduled issue at 00, 06, 12 or 18 UTC. It selects a response actually received before that issue, within two hours of it. A first capture after issuance cannot create a forecast for that earlier issue. The model uses observations through issue time minus two hours, without interpolation or forward filling.

| Condition | Experimental numeric score | Operational prediction |
| --- | --- | --- |
| No frozen gauge reference or attributes | Withheld | Withheld |
| No response received by the issue, or response too old | Withheld | Withheld |
| Wrong cadence, future issue, failed response or checksum mismatch | Withheld | Withheld |
| Incomplete approved 169-hour history | Withheld | Withheld |
| An exceedance during the preceding 24-hour quiet period | Withheld | Withheld |
| Complete eligible inputs and verified artifacts | Gauge research score and calibrated physical comparator, inside experimental details | Withheld pending promotion and site applicability |

`data/hydrology-model.json` pins the candidate, physical comparator, historical references and gauge attributes. Models load only after their artifact hashes match. The manifest covers eighty-eight retained research gauges; that count describes reference availability, not trained gauges or site coverage. The candidate's training population remains the one documented in the research results.

The feature builder reproduces the frozen lagged-flow, dynamics, seasonal and static features. Its normalized flow is `log1p(discharge/reference)`, so the reference crossing is `log(2)`. The comparator is the same conditional-calibrated linear extrapolation used in the development assessment. Neither later approval nor a revised value can backdate receipt availability.

## Measured live coverage and reliability

The [network inventory](../experiments/site-hydrology-20261018/coverage.json) covers all sixty EIA facilities on October 18, 2026. Every catchment lookup and navigation request succeeded. Fifty-three facilities returned network-linked candidates; the union contained 1,423 monitoring locations. Those locations include historical, reservoir and other records that need not supply current discharge.

The [current-discharge screen](../experiments/site-hydrology-20261018/current-coverage.json) found fresh unqualified discharge linked to thirty-one facilities. Its successful inventory queries identified fifty unique active gauges. This measures observational coverage within the bounded search. It establishes no site-flood or ML forecast coverage.

The first pilot simply took nearby network candidates. All three pilot facilities returned no current discharge in that screen. The [retained failed screen](../experiments/site-hydrology-20261018/live.json) motivated the active-discharge inventory step. The [refined pilot](../experiments/site-hydrology-20261018/live-refined.json) then found observations at Grand Coulee and Peach Bottom. Gavin still had no fresh discharge in the searched network; its separate NWS stage-status feed remains independent.

Only Bath County's network included a gauge from the frozen research registry, Back Creek near Sunrise, Virginia, station `02011460`. The [Bath County capture](../experiments/site-hydrology-20261018/bath-county.json) also found two other active gauges. The registry match establishes a known reference and attributes. It does not establish applicability near a pumped-storage project or validate regulation effects.

Across the refined pilot and Bath County, five gauge histories supplied 5,357 observations. All were provisional; none was approved. The latest observations were 28.3 to 58.3 minutes old at receipt. No operational score was released. The snapshots also predated the next scheduled issue, so they cannot supply an earlier as-issued prediction. This is measured withholding and data access, not predictive validation.

Exact discharge response bytes and request metadata are retained under the experiment's `receipts/` directory. Compressed files preserve the response body; each manifest records the uncompressed body hash, actual receipt time, URL and HTTP status. The application keeps the corresponding bytes in SQLite. No credentials appear in these artifacts.

## Validation and remaining limits

The [historical parity audit](../experiments/site-hydrology-20261018/historical-parity.json) reconstructs fifty existing decisions across five gauges from retained raw 2023-2024 USGS records. Candidate and comparator probabilities match the frozen development predictions with maximum absolute errors below 1e-16. The first reconstruction omitted the previous year's warmup and correctly failed incomplete-history checks. Including that source history resolves the reconstruction without weakening eligibility.

Those outcomes were inspected before this phase. This audit verifies feature and inference parity only. It supplies no new recall, lead, false-alert or calibration estimate. Raw capture and synthetic boundary tests likewise cannot establish operational model accuracy.

No model qualifies for deployment. The pooled recall uncertainty failure from the prior study remains open, and the live pilot establishes an additional practical limit: current observations at its known gauges are provisional. A change to provisional inputs requires a new frozen quality contract and revision study. It cannot silently substitute for the current approved-only model.

Independent confirmation needs geography reserved from metadata and polygon screening against prior exposure, plus an untouched outcome period. The [acquisition review](HYDROLOGY_CONFIRMATION_OPPORTUNITIES.md) identifies broader USGS/GAGES-II discovery, prospective receipts, operational National Water Model archives and HRRR as concrete sources to investigate. No new confirmation cohort or predictive outcome result was established here. Promotion gates remain unchanged.

## API and operation

| Endpoint | Behavior | Persistence |
| --- | --- | --- |
| `GET /api/sites/{site_id}/hydrology` | Check river context at the facility point and recheck open JHAs | Point cache and immutable capture rows |
| `GET /api/missions/{mission_id}/hydrology` | Check the actual work point and recheck an open mission's JHA | Point cache and immutable capture rows |
| `GET /api/hydrology/receipts/{receipt_id}` | Return exact captured bytes with receipt-time, upstream-status and SHA256 headers | Read only; corrupt bytes return 503 |
| Existing mission detail and work-window endpoints | Include latest cached point context | Existing JHA and decision rules apply |

The app initializes the `hydrology_receipts` table in its canonical schema. Captures occur on demand, at most once per gauge per fifteen minutes. They are not an unattended monitoring service. The stored receipt bodies accumulate with use; preserve those needed for confirmation before manually removing research evidence. No automatic deletion or historical-state migration runs.

`USGS_API_KEY` is optional. When configured, the app sends it only in the official USGS request header. Missing access or rate limiting remains visible through unavailable provider results. Tokens never enter request URLs or saved response metadata.

The following commands ran against isolated validation databases. Audit output files refuse overwrite; choose new output names for subsequent captures. The manifest is already included and refuses overwrite too. Restart the backend after backend changes or a frontend build.

```sh
.venv/bin/python -m scripts.site_hydrology manifest
PLANIT_DB=/private/tmp/planit-hydrology-validation.sqlite .venv/bin/python -m scripts.site_hydrology coverage --output experiments/site-hydrology-20261018/coverage.json
PLANIT_DB=/private/tmp/planit-hydrology-validation.sqlite .venv/bin/python -m scripts.site_hydrology current-coverage --output experiments/site-hydrology-20261018/current-coverage.json
PLANIT_DB=/private/tmp/planit-hydrology-refined.sqlite .venv/bin/python -m scripts.site_hydrology live --sites 8102,3166,6163 --output experiments/site-hydrology-20261018/live-refined.json
PLANIT_DB=/private/tmp/planit-hydrology-refined.sqlite .venv/bin/python -m scripts.site_hydrology live --sites 6167 --output experiments/site-hydrology-20261018/bath-county.json
.venv/bin/python -m scripts.verify_site_hydrology
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
```

The browser verification uses an isolated database and a clearly fictional Grand Coulee mission. It checks the actual river panel, withheld status and mission/work-window routing. A frontend API-prefix defect in the work-window fetch and decision calls was fixed so those requests use the shared client's canonical path convention. The frontend layout otherwise retains its existing components and styles.

The live walkthrough also exposed a planner alignment defect: candidate starts inherited request seconds, so ordinary shifts could disappear from their own comparison. Candidate starts now use stable UTC hours and include the exact planned shift when it satisfies the existing constraints. Regression tests verify comparison across routine refreshes and continued enforcement of restrictions. Scoring bands, material-improvement requirements and human decision rules are unchanged.

Verification passes 361 Python tests and the frontend test and production build. The [live API record](../experiments/site-hydrology-20261018/end-to-end.json) confirms an exact planned-window comparison, stable successive snapshots, non-counting JHA evidence and receipt integrity. The [mobile capture](../experiments/site-hydrology-20261018/mission-river-mobile.png) has a 390-pixel viewport and matching document width. One browser connection error occurred while intentionally restarting the isolated server; it was not a river-provider result. Field use, inundation accuracy and unattended monitoring remain unvalidated.
