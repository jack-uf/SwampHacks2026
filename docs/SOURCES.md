# Sources, coverage and provenance

These integrations were exercised against public endpoints during implementation on October 16, 2026 (some retrieval manifests are October 17 in UTC). Source freshness is displayed independently of event/forecast validity. Retrieval time is not observation time. Links below identify the primary provider documentation and actual data contracts.

## Facility inventory

- Source: [EIA-860M December 2024 operating-generator workbook](https://www.eia.gov/electricity/data/eia860m/archive/xls/december_generator2024.xlsx).
- Extraction: group operating generator rows by EIA plant ID; aggregate nameplate capacity; retain plant coordinates and largest-capacity technology. Select up to two large plants per state, then the 60 largest qualifying plants. The resulting inventory covers 33 states.
- Validation: unique plant IDs, required metadata, continental coordinate ranges, at least 50 sites, and IANA timezones derived from coordinates. Coordinates are source-validated, not field-surveyed. The selected inventory is a fixed historical snapshot; it does not establish current operating status.
- [EIA reuse policy](https://www.eia.gov/about/copyrights_reuse.php): federal data are generally public domain except credited third-party material; retain attribution. EIA-860M is monthly, but this app does not automatically update the fixed inventory.

`data/sites.json` carries source metadata. Raw workbook bytes and checksum manifests are cached locally when ingestion runs. An attempted newer workbook URL was unavailable; the reproducible, valid December 2024 workbook is the canonical source.

## Weather and archived forecasts

[Open-Meteo forecast API](https://open-meteo.com/en/docs) supplies current modeled conditions and hourly forecasts: temperature, apparent temperature, humidity, precipitation, gusts, and 0–1 cm soil moisture. The app requests seven past days and three forecast days in UTC. Past hours and “current” values are numerical model estimates, not facility instruments or verified rainfall gauges. Soil moisture is a shallow modeled volumetric fraction, not a geotechnical stability assessment.

Live weather is cached for 30 minutes. Native model updates vary by model and region; this application does not claim each request represents a new model run. Failed requests mark the feed unavailable and preserve the last successful timestamp; old values are not silently reused in the index.

[Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) provides `gfs_seamless` predictors with the `_previous_day2` suffix. For each target hour, the selected archived forecast has a 48-hour lead. This contract differs from a single fixed model initialization. Historical training requests January 2024–June 2026. Exact 24-hour completeness for all four variables and 72 prior precipitation forecast hours yields January 24, 2024–June 30, 2026: 21,336 rows across 24 counties. Inference uses the same extraction at the evaluated representative point, cached six hours per county and UTC target date. Previous Runs is a nominal-lead archive; publication delays are not reconstructed. [GFS Single Runs](https://open-meteo.com/en/docs/single-runs-api) begins April 2, 2026 and cannot replace the full historical contract.

[Terms](https://open-meteo.com/en/terms): the open-access service is noncommercial, currently limited to 10,000 daily, 5,000 hourly, and 600 minute calls, with no availability guarantee. Data require attribution under CC BY 4.0. Long historical requests may count as multiple calls. Check terms and service entitlements before commercial use. This implementation uses no paid account and surfaces rate-limit failures.

## Active warnings

[NWS API documentation](https://www.weather.gov/documentation/services-web-api) describes the public, cache-friendly API and identifying `User-Agent` requirement. Configure a real contact in `PLANIT_USER_AGENT`. The app calls `https://api.weather.gov/alerts/active?point=LAT,LON`, retains warning descriptions/instructions and expiry times, and omits expired alerts when assessing a site. Alert absence from a successful request differs from feed failure.

The provider cache TTL is five minutes **when requested**. Assessments run portfolio-wide after each previous refresh plus 30 minutes, so the normal dashboard alert snapshot can be older than five minutes. Detail polling does not force a new assessment until its 35-minute stale limit. Manual authenticated refresh requests a fresh assessment subject to provider TTLs. This is not a push alert service or an emergency warning channel; consult authoritative alerts directly during rapidly changing conditions.

## Historical outcomes

[NOAA Storm Events details](https://www.ncei.noaa.gov/stormevents/details.jsp) and [CSV archive](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/) supply county-event reports. The measured dataset uses 2024 revision `c20260728` and 2025 revision `c20260819` and 2026 revision `c20260918`. NOAA archives can be revised and publication is delayed; these records are training outcomes, not live alerts.

Only county (`CZ_TYPE=C`) reports in the selected twenty-four counties are joined. Flood labels include Flood and Flash Flood. Storm labels include Thunderstorm Wind, Tornado and Hail. Standard-time offsets in NOAA timestamps are explicitly converted to UTC; events spanning midnight label each overlapping target day. One county report does not establish an incident at the representative point or any plant. Missing reports do not establish hazard absence. Federal source attribution is retained in `data/training-provenance.json`.

The [2024 Census county gazetteer](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip) supplies official FIPS/internal-point coordinates for twelve additions. Original representatives are unchanged. Federal source data retain attribution; `data/regions.json` freezes each chosen point and experiment role. `data/noaa-geography-audit.json` records county/zone/marine category counts. Never treat zone FIPS as county FIPS. Closed UTC event overlap and standard-time conversion are documented in MODEL_CARD.md.

## Static mapped flood exposure

[FEMA NFHL layer metadata](https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28) and its `/query` endpoint provide intersecting polygons at each plant point. Requests use WGS84 longitude/latitude and return `FLD_ZONE` and `SFHA_TF`. A real query returned zone X with special-flood-hazard flag F during verification.

The app caches this static layer for seven days. Effective map updates depend on local map revisions, not the dashboard refresh cadence. No polygon or unrecognized flag means unknown, not no flood risk. FEMA mapping is not a current inundation forecast, hydraulic model, site survey, or guarantee. Retain FEMA attribution and consult layer metadata and local effective mapping before operational reliance.

## Mapping, privacy and unsupported sources

Map data: © OpenStreetMap contributors; [standard tile policy](https://operations.osmfoundation.org/policies/tiles/). The browser uses the canonical `https://tile.openstreetmap.org/{z}/{x}/{y}.png` endpoint with visible attribution, normal browser caching and no bulk prefetch. Community tiles provide no SLA. Map requests expose IP address, referrer and viewed tile areas to the tile service. Google Fonts requests also leave the browser; browser fonts render if fonts are unavailable.

Crew reports are user observations with manual classification and explicit review status. Photos are re-encoded to remove embedded metadata, but the submitted report coordinates and visible image content remain sensitive. Optional Jev sends observation text only after explicit per-report consent and separately enabled server configuration. Photos and coordinates are never sent; descriptions can still disclose sensitive content. [The contract audit](JEV_RESEARCH.md) links the retrieved official documentation and distinguishes documented capabilities from unverified live inference. No external inference was performed; hypothetical response fixtures are not provider accuracy measurements.

No live commercial strike service, active wildfire perimeter service, or wildlife observation network is configured. Lightning ingestion is an authenticated vendor-neutral contract described in OPERATIONS.md; replay is always labeled and stored separately. NWS fire-related warnings and crew reports provide the supported wildfire coverage. No source failure is replaced by synthetic live data.
