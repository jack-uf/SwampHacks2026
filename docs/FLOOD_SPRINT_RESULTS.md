# Measured river flood forecasting: feasibility result

This study stopped before modeling because historical flood-label validity could not be established from the audited sources. Real stage measurements are available. The unresolved question is which flood threshold and gauge reference applied to those measurements at the time. No predictive improvement, satellite benefit or geographic transfer result is claimed, and nothing is integrated into PlanIT.

The [frozen protocol](FLOOD_SPRINT_PROTOCOL.md) defines a minor-flood-stage exceedance between 24 and 48 hours after a midnight UTC decision, with a separate analysis of new crossings from below flood stage. It was committed in `f683012` before modeling or multi-year outcome collection. The experiment directory is [`experiments/sprint-20261017-flood/`](../experiments/sprint-20261017-flood/); [`audit.json`](../experiments/sprint-20261017-flood/audit.json) owns the machine-readable findings.

## Measurements and geographic coverage

Four of six preselected NOAA identifiers resolved to USGS stream gauges. Their coordinates agreed within 95 metres. HRSP1 and RALN7 returned HTTP 404 and were recorded as exclusions, without substituting other stations. These failures apply to the requested identifiers, not to river-data availability in Pennsylvania or North Carolina.

The sample covers January 1, 2024 at 00:00 UTC through January 3 at 00:00 UTC, including both endpoints. Each nonempty series contains 193 observations at 15-minute intervals. There are 772 stage readings and 579 streamflow readings. All returned readings are marked Approved, with null qualifiers; numeric and timestamp spot checks found finite values and no duplicate timestamps. This short sample does not establish multi-year completeness, flood-event coverage or the accuracy of every measurement.

| NOAA LID | River and USGS station | HUC reported by USGS | Current minor stage, feet | Stage readings | Flow readings | Coordinate difference, metres |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| POPW2 | Ohio at Point Pleasant, 03201500 | 05090101 | 40 | 193 | 0 | 14.6 |
| PORM2 | Potomac at Point of Rocks, 01638500 | 020700080401 | 16 | 193 | 193 | 5.7 |
| DALT2 | Trinity at Dallas, 08057000 | 120301050102 | 30 | 193 | 193 | 29.9 |
| CRNW1 | Snoqualmie near Carnation, 12149000 | 171100100601 | 54 | 193 | 193 | 94.2 |

Stage is parameter 00065 in feet. Streamflow is 00060 in cubic feet per second. Point Pleasant's flow query returned an empty collection for this interval; it does not prove flow is absent for other dates or services. The station joins use NOAA's explicit `usgsId`, never a nearest-location match. HUCs are preserved as returned, including their different lengths. They do not by themselves establish independent watersheds.

The evidence for each row is in `sources/<LID>-noaa.json`, `<LID>-location.json`, `<LID>-00065.json` and `<LID>-00060.json`. Every observation request was bounded to this interval and 1,000 records. None of these sample responses was paginated.

## Why the historical label gate stays closed

None of the four resolved gauges has a certified historical threshold contract in this audit. This is an unverified-history result, not evidence that their flood thresholds changed or that a suitable archive cannot exist.

The cached NOAA OpenAPI schema defines flood categories as stage and flow numbers. It exposes no threshold effective interval or historical gauge-metadata query. Historical crests have occurrence times and an `olddatum` flag, but those fields do not certify which minor threshold applied between crests. The sampled gauge responses contain current categories and datum descriptions, without the historical threshold intervals required by the protocol. See `sources/noaa-openapi.json`, especially `FloodCategoriesDatum`, `FloodCrestsDatum`, and the gauge endpoint's parameters.

Current elevation metadata also needs careful interpretation:

| LID | NOAA NAVD88 value, feet | USGS altitude, feet | Difference, NOAA minus USGS | USGS altitude accuracy, feet |
| --- | ---: | ---: | ---: | ---: |
| POPW2 | 513.08 | 513.04 | +0.04 | 0.07 |
| PORM2 | 199.92 | 199.91 | +0.01 | 0.02 |
| DALT2 | Not supplied | 367.84 | Unresolved | 0.20 |
| CRNW1 | 2.12 | 2.12 | 0.00 | 0.05 |

The small differences at Point Pleasant and Point of Rocks are within the stated USGS altitude accuracy. They are not demonstrated datum errors and must not become automatic stage corrections. USGS monitoring-location altitude describes the gage or land-surface elevation; the field alone is insufficient to prove a shared, historically stable gauge zero. Carnation's equal current values are encouraging, but still do not establish historical validity. Dallas lacks a NOAA vertical datum value in the sampled response.

USGS time-series metadata, method intervals and revision records were also inspected. The schema's `thresholds` field describes numerical data-quality limits, such as sensor limits or historic maxima. It does not supply NWS flood-category history. Method records identify measurement methods over intervals, without a historical gauge-zero conversion. All four sampled stage-series revision queries returned empty collections. Empty revision results are not proof that the gauge reference never changed.

The Carnation method response was capped at 100 records and has a next-page link. It is a partial inspection and supports no claim of complete method history. The remaining method responses returned one record at Point Pleasant, two at Point of Rocks and none at Dallas. Their request files preserve the exact series IDs and URLs.

The documentary search also inspected the current USGS Point Pleasant station page, two legacy water-year-summary URLs, and NOAA's service-change index. The legacy URLs returned 404. The current station page did not provide the required historical threshold contract, and no applicable contract was established from the inspected notice index. These responses and their hashes are in `documentary/`. This was a bounded search, not an exhaustive search of RFC records, local WFO notices, station manuscripts or offline archives.

## Predictor availability is a separate unresolved requirement

Approved historical measurements can be outcome labels after the threshold contract is established. Their approval does not prove what a supervisor could have seen at an earlier prediction time. USGS documents `last_modified` as a database refresh time that can change during ordinary processing. It is not the first-publication timestamp. A nominal two-hour lag would therefore remain an assumption without transmission or ingestion evidence. The exact definition is cached in `sources/continuous-schema.json` and `sources/time-series-metadata-schema.json`.

The existing GFS and Sentinel-2 machinery was reviewed, but no flood features were acquired or fitted. The earlier wind forecasts have nominal previous-run leads, not verified dissemination records for a single flood decision time. Upstream predictors need a directed river-network join and travel-time reasoning; geographic proximity alone is insufficient. No such joins were constructed in this audit.

The existing [`scripts/s2_pipeline.py`](../scripts/s2_pipeline.py) uses a one-day acquisition-latency allowance rather than actual publication evidence. Its cached scene metadata includes a decision date and scene age that are returned unchanged when the same scene is reused. It selects point-centred crops rather than basin features and reads only the first STAC result page. These are reasons to audit that pipeline before a flood experiment; this study did not repair it or recalculate earlier results. Pixel-grid alignment, reflectance offsets and valid water-area denominators also require verification before reuse. There is no satellite incremental-value result here.

## Model comparisons and independent evaluation

Persistence, recent-stage trend, logistic regression, XGBoost, LightGBM and TabPFN were not run. Hydrology-only, weather-only, weather plus satellite, and combined ablations were likewise not run. The protocol makes verified historical labels a prerequisite for every candidate. An approximate label would answer a different question and would not satisfy that requirement.

Validated flood-event counts, AP, calibration, matched-budget recall, false alerts, warning lead time and satellite confidence intervals are unavailable, not zero. No watershed-disjoint or forward-time performance was measured. No multi-year flood outcomes were collected, and no model-selection or test-set tuning occurred. The sample is geographically varied but cannot support a geographic-generalization claim.

Earlier wind and multimodal reports described measured river flooding as the next promising target based on accessible measurements and current thresholds. This audit narrows that preliminary feasibility statement. The measurement join works for the resolved gauges, but historical label validity and predictor availability still need evidence before training.

## Reproduction, verification and resources

Run from the repository root:

```sh
.venv/bin/python -m scripts.flood_sprint summarize
.venv/bin/python -m pytest tests/test_flood_sprint.py -q
.venv/bin/python -m pytest -q
```

The summary command is offline. It verifies the frozen protocol document hash and hashes of the cached API responses it consumes, then recreates `audit.json`. Two offline executions produced identical summaries. The auditor deliberately cannot certify historical eligibility from current metadata alone. Historical documentation must be reviewed before adding a label builder; setting an eligibility flag would not replace that review.

To collect missing API records, run `.venv/bin/python -m scripts.flood_sprint collect` with network access. It reuses existing records, including recorded failures, and refuses changed request URLs or altered cached bytes. It does not refresh old responses. A new study needs a new directory and a frozen protocol rather than overwriting this evidence. Documentary HTML responses are supporting research records with retrieval URLs and hashes; `collect` does not fetch those pages.

There were 35 API requests, including the two NOAA 404 responses, transferring 1,518,739 response bytes. Their summed request durations were 18.807 seconds, excluding pacing waits. The four additional documentary responses transferred 1,306,054 bytes. The offline resource sample in [`resources.json`](../experiments/sprint-20261017-flood/resources.json) took 0.13 seconds with 36,257,792 bytes maximum resident memory and zero process swaps. That measurement covers one summary run, not ingestion or a model benchmark. No new imagery, checkpoints or dependencies were downloaded.

Eight focused tests pass. They cover station identity and location, missing and invalid threshold metadata, unresolved vertical references, elevation differences, cached parse failures and modified source bytes. The full Python suite passes 160 tests, with three existing SHAP deprecation warnings. Production models, safety rules, frontend and persistent application data have no changes in this study.

## Strongest justified next step

Start with Carnation, USGS 12149000 and NOAA CRNW1, because its current reference values match and the sample has both stage and flow. Obtain an authoritative threshold effective-history and station gauge-zero history covering the intended study years from the responsible NWS office or River Forecast Center and USGS Water Science Center. Preserve those documents and any dated conversions alongside the label contract. Obtain as-issued stage or telemetry archives before using historical hydrology as a predictor; alternatively, evaluate a weather-only model once its archived forecast availability is verified and the labels pass.

After one gauge passes, repeat the documentary checks across enough independent drainage systems for the frozen population and event-count requirements. Only then acquire multi-year outcomes, freeze basin roles, count separate flood episodes and run persistence before fitted models. Satellite work should follow that benchmark, with publication-valid scenes and basin geometry. A prospective immutable capture of thresholds, gauge metadata and arrivals is another defensible path when retrospective evidence cannot be recovered, but it cannot produce a historical validation result today.

Retain the existing production behavior. The useful result of this study is a reproducible audit and a precise evidence requirement; it is not a new flood predictor or a validated contribution to the 0-10 hazard index.
