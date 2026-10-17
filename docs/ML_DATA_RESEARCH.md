# ML data source eligibility audit

Keep GSOD observations out of this sprint's predictors. Neither NOAA's daily archive nor its Kaggle catalog establishes when each historical value first became available. SWDI can support retrospective diagnostics, provided each product keeps its own meaning and coverage rules. Storm Events files hosted under NOAA's `swdi` archive path duplicate the existing outcome source. Kaggle mirrors add no independent evidence; synthetic flood benchmarks are ineligible.

Sources were retrieved on October 17, 2026. This audit covers provenance, coverage, licensing and availability at decision time. It does not select counties, inspect the new splits' target outcome counts, design experiments or change training. Existing source contracts remain in [SOURCES.md](SOURCES.md), interpretation in [MODEL_CARD.md](MODEL_CARD.md), and research context in [RESEARCH.md](RESEARCH.md).

## Decision-time contract

For target UTC day `D`, the decision cutoff is `D-1 00:00 UTC`. A target of February 1, 2024 therefore has a cutoff of January 31, 2024 at 00:00 UTC. A predictor must have been available by that cutoff. An observation's timestamp, a forecast's nominal initialization, and the time an archive was downloaded answer different questions.

For any proposed observation feature, require its reporting interval to end by the cutoff and require evidence that the exact historical version had been published by then. A current archive's modification time proves neither first publication nor historical content. A conservative lag without publication evidence does not establish eligibility. In particular, a daily summary for `D-2` finishes at the cutoff before processing and dissemination have occurred. Even `D-3` is not automatically eligible under an unbounded delay description.

Reject target-day realized temperature, pressure, precipitation, gusts, radar detections, storm reports, and GSOD weather flags as predictors. A whole `D-1` daily observation is also ineligible because most of that day occurs after the cutoff. These records may describe outcomes retrospectively. Labels can mature after an event; predictor availability cannot.

The existing `gfs_seamless` Previous Runs contract selects a nominal lead for each valid hour, rather than one fixed issue run. It already discloses that historical dissemination is not reconstructed. GSOD cannot be added under a weaker, undisclosed timing assumption. The main experiment owns any pressure/CAPE forecast change and its validation. [SOURCES.md](SOURCES.md)

## Source decisions

| Source | Authentic origin and coverage | Recommended role in this sprint | Blocking issue |
| --- | --- | --- | --- |
| NOAA GSOD | Station daily summaries derived from ISD/ISH. Global archive directories begin in 1929 and the inspected tree ends in 2025. Coverage varies by station, field and day. [N1], [N2], [N3] | Retrospective weather and missingness diagnostics. No observation predictor approved. | No historical per-record publication/version history; conflicting official daily time-basis descriptions; global reuse terms need qualification. |
| SWDI radar products | NEXRAD algorithm detections at radar/cell coordinates, with product-specific archive spans. [N4], [N5] | Separate radar-detection diagnostics or separately defined research outcomes. | A detection is not confirmed surface severe weather; no archive publication-time guarantee or complete coverage mask. |
| NOAA Storm Events archive (`swdi/stormevents`) | NCEI Storm Events, already used for county labels. [N7] | Same label source under the existing contract. | Archive hosting does not establish a current SWDI query layer. This is not independent ground truth or a new predictor source. |
| SWDI preliminary reports and lightning tiles | Preliminary Local Storm Reports and aggregated proprietary NLDN detections. [N13], [N14] | Product-specific retrospective diagnostics only, after coverage and terms checks. | Reports are not final incident truth. Public lightning tiles are not individual strike records. |
| Kaggle NOAA GSOD / NOAA Significant US Storms | Redistributed NOAA observations or selected Storm Events records. [K1], [K3] | Provenance comparison only. Prefer the upstream source. | Duplicate evidence, uncertain mirror completeness/revisions, no historical release reconstruction. |
| Kaggle US Weather Events | Author-created events from airport weather observations, January 2016 through December 2022. [K2] | Optional separate, noncommercial historical research. | Outside the existing 2024-2026 training period; different event definitions; no point-in-time delivery evidence. |
| Kaggle synthetic or unverified flood-factor data | Playground flood competition generated data; original factor dataset's observational provenance is unverified here. [K4] | Reject for training, evaluation, augmentation and calibration. | No verified dated county incidents or compatible forecast contract. Unresolved terms do not cure synthetic provenance. |

## NOAA GSOD

### Coverage and schema

GSOD is retired. NOAA's end-of-life notice says updates ended on August 29 and identifies SSOD as its replacement. The notice omits the year; the GSOD root index dates its access directory August 29, 2025, and the inspected year tree ends in 2025. Together these support the 2025 transition, not a continuing 2026 GSOD feed. They do not establish the last observation date at every station. Legacy metadata still says "to present" and must not override the retirement notice. [N11]

GSOD summarizes ISD/ISH surface observations, including observations obtained through the USAF Climatology Center. NOAA describes more than 9,000 stations typically available and more complete coverage from 1973 onward. These are dataset descriptions, not assurances that any selected county or field has daily coverage. The inspected access tree has years 1929-2025 and no 2026 directory. [N1], [N2], [N3]

NOAA's current successor documentation identifies SSODv2 as daily summaries computed from GHCNh, using a common `00-23 UTC` synoptic day worldwide. It also warns of underreported precipitation and extrema that may miss the true daily extremes. SSODv2 is a distinct source contract, not a continuation that supplies historical GSOD publication evidence. Its page describes a short delay but provides no verified per-record historical availability guarantee here. Do not substitute SSODv2 into this sprint without a separate schema, coverage, rights and publication-time audit. [N12]

The authentic sample is the first three records of the 2024 Los Angeles International Airport file. It contains `STATION`, `DATE`, coordinates, elevation, name, daily weather values and attribute fields. `DATE` is ISO format in these CSV bytes despite the README's older format description. `STATION=72295023174` is an identifier, not an integer feature or county code. Preserve station identifiers as strings. [N8]

| Fields | Interpretation that affects reuse |
| --- | --- |
| `TEMP`, `DEWP`, `MAX`, `MIN` | Fahrenheit. Daily means and extrema have different derivations. `MAX_ATTRIBUTES` / `MIN_ATTRIBUTES` distinguish explicit extreme reports from extrema derived from hourly observations. |
| `SLP`, `STP` | Sea-level and station pressure in millibars. They are observations, not the existing forecast pressure variable. |
| `WDSP`, `MXSPD`, `GUST`; `VISIB` | Wind in knots; visibility in miles. Preserve missing-value sentinels and observation counts before any unit conversion. |
| `PRCP`, `PRCP_ATTRIBUTES`; `SNDP` | Inches. Precipitation flags describe differing accumulation periods. `H` indicates incomplete information; `I` does not establish a dry day. Snow-depth missingness is not zero snow. |
| `FRSHTT` | Six occurrence indicators, including hail, thunder, and tornado/funnel cloud. Zero means no occurrence **or not reported**. These are realized weather indicators, not predictors or complete severe-event labels. |

The format notes define field-specific all-nine missing values, such as `99.99` for precipitation and `9999.9` for temperature. Some means require only four observations. An airport day with four readings is not equivalent to continuous county surveillance. Automated quality control leaves residual errors. Retain attribute fields and missingness; do not replace missing precipitation with zero. [N2]

### UTC, release timing and revisions

The detailed README says summaries use GMT `0000Z-2359Z`, while warning that precipitation and extremes may include part of the preceding day. The official catalog instead says the UTC daily-period statement has an exception for U.S. stations. The sample has dates without a time-zone or interval field. This discrepancy is unresolved for the U.S. subset. Do not silently interpret every GSOD column as an exact UTC calendar-day aggregate. [N2], [N3], [N8]

NOAA's README describes normal availability as "a few days" after observation. Its catalog says usually 24-48 hours; the Kaggle card says 1-2 days. These descriptions differ and none is a worst-case publication guarantee. The current C00516 landing page explicitly lists publication as unknown and updates as needed. The sample includes no publication or revision timestamp. Its annual file's HTTP `Last-Modified` was January 21, 2025, which cannot reconstruct publication of the January 2024 rows. No original release log, immutable daily vintage or maximum late-arrival bound was verified. [N2], [N3], [K1], [N8]

GSOD can overlap 2024-2025 training dates as retrospective observations. It cannot support the complete 2024-2026 feature period from the inspected archive. For future reconsideration, the data owner must obtain a documented release schedule plus historical vintages or contemporaneous captures that establish availability at each cutoff. Merely choosing a larger lag or substituting a newer daily product does not supply that evidence.

### Geography and reuse terms

Map a station to a county using dated station coordinates and a pinned county polygon vintage. NOAA publishes an ISD station-history resource through its [ISD access documentation](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database). Check identifier validity dates, coordinate changes and station relocations before joining. Preserve the station-to-county relation separately from any representative-point distance. A nearest airport can lie outside the county; even an airport inside it does not observe the entire county. No station observation establishes a plant incident.

NOAA's GSOD README retains WMO Resolution 40 language restricting commercial use/re-export of some non-U.S. data. The current catalog's constraint section supplies citation and liability wording rather than resolving that restriction. Kaggle's CC0 badge is a publisher declaration, not evidence that every upstream international observation has unrestricted commercial rights. A U.S. NOAA subset with attribution is the practical research scope here; resolve product-specific global/commercial terms with NOAA before broader reuse. [N2], [N3], [K1]

## NOAA SWDI

### Product coverage is separate from observed occurrence

SWDI is an inventory of U.S. records from several sources, not one homogeneous event dataset. NOAA expressly says it is incomplete and that absence of a record does not mean severe weather did not occur. Automated radar signatures indicate probable conditions and need not correspond to actual surface conditions. [N4]

The current product page lists radar signatures and lightning; the C00773 catalog also identifies Preliminary Local Storm Reports. Neither establishes a current SWDI warning layer or Storm Events query layer. The separate Storm Events archive resides under the `swdi` URL path. Preserve that source distinction rather than inferring service capabilities from directory names. [N4], [N13], [N7]

The directly retrieved `database-csv/v2` index listed the following file spans. These are filenames, not verified complete observation periods. No event archive was downloaded to obtain this table. [N5]

| Archive product | Files listed on retrieval | Consequence for 2024-2026 research |
| --- | --- | --- |
| `hail`, `mda`, `tvs`, `structure` | Annual files beginning in 1995 through 2025, plus monthly `202601` through `202610` files. | Potential temporal overlap. A current month is necessarily partial on October 17; file presence does not establish daily, radar or geographic completeness. |
| Legacy `meso` | Annual 1995-2010 files. | Do not mistake it for current `mda` coverage. The inspected product has no 2024-2026 files. |
| `plsr` | Annual 2005-2015 files. | The inspected bulk directory does not establish 2024-2026 preliminary-report availability. Check the service separately before proposing this product. |
| `nldn-tiles` | Annual files beginning in 1986 through 2025, plus monthly `202601` through `202610` files. | Potential retrospective overlap, subject to aggregation, coverage and proprietary-source terms. |

NEXRAD coverage follows the radar network and successful product ingestion, not county boundaries. A detection's coordinates can be assigned to a county with a pinned polygon layer, but radar ID and cell ID are not county IDs. Preserve radar/product identity, repeated scans and storm dependence. A day with no detection needs a separate coverage/ingestion assessment before anyone calls it a negative.

### Authentic sample, quality and release timing

The bounded service request in [N9] returned the header `ZTIME,WSR_ID,CELL_ID,PROB,SEVPROB,MAXSIZE,LAT,LON`. Four retained records begin on May 5, 2006 with explicit `Z` timestamps. This verifies a UTC radar-event schema and accessible historical records, not 2024-2026 completeness or availability at an earlier decision time. No new validation/final-state outcome counts were queried. The service landing page returned HTTP 400, while the specific CSV request succeeded; landing-page failure alone does not establish service failure.

`ZTIME` is a radar-event timestamp, not proof of publication. The inspected index and sample do not expose immutable original release versions. No product-specific historical ingestion latency bound was verified. Accordingly, even antecedent SWDI observations are blocked as predictors under this sprint's strict contract. Whole-day radar summaries, daily lightning tiles and reports for `D` or most of `D-1` are directly after the cutoff.

NOAA's known-issues file, dated March 25, 2014, documents rounding of hail `MAXSIZE` and several structure, mesocyclone and TVS fields. It says both service results and CSV dumps were affected and were being reprocessed. That old notice does not prove which current records remain affected or that every product is repaired. Pin the actual bytes and parser, preserve reported precision, and confirm units/threshold semantics before any severity comparison. [N6]

### Overlap, labels and terms

The NOAA Storm Events archive and Kaggle Storm Events mirrors share the existing labels' origin. Duplicate retrieval paths do not create independent confirmation. Preliminary Local Storm Reports may feed later official records; a report and a final event can describe the same storm. A radar scan series may also describe that storm many times. Resolve duplicates and storm dependence before interpreting apparent agreement. [N13], [N7], [K3]

If warning polygons are considered later, define them as official warnings, not observed impacts; no current SWDI warning feed was verified here. Radar hail and vortex signatures are suitable for a separately named detection outcome or a discrepancy review, not interchangeable flood/storm truth. The existing target combines Flood/Flash Flood and Thunderstorm Wind/Tornado/Hail county reports. Radar hail alone neither covers flooding nor reproduces that combined storm definition. [SOURCES.md](SOURCES.md)

NOAA describes public NLDN access as daily 0.10-degree tiles counting cloud-to-ground lightning flashes. Its lightning documentation restricts raw flash data and SWDI strike data to NOAA employees and contractors under the Vaisala contract. These public tiles cannot supply individual-strike timestamps or validate an operational strike-radius timer. Do not infer unrestricted redistribution of underlying records from public download access. The SWDI catalog requires citation and supplies quality/liability constraints; it does not clear third-party rights. Federal NWS material is generally reusable with attribution, but its policy expressly excludes third-party copyrighted material. Keep the product-specific restriction. [N14], [N13], [N10]

## Kaggle provenance and licenses

Public dataset metadata was retrieved without authentication or dataset downloads. A card's modification time identifies the catalog revision, not each observation's original publication. Publisher claims below are attributed to the card; unverified coverage is not promoted to a guarantee.

| Candidate | Origin, date range and schema evidence | Declared license and verified catalog revision | Recommendation |
| --- | --- | --- | --- |
| [`noaa/gsod`](https://www.kaggle.com/datasets/noaa/gsod) | NOAA ISH-derived GSOD, described as global 1929 to present. This is the same observation family as upstream GSOD. Card links a BigQuery workflow but contains an unrelated `github_repos` table example; current table contents and mirror coverage were not verified. | CC0: Public Domain; version 4; metadata modification August 30, 2019. This does not establish when a live BigQuery table was last populated. Upstream non-U.S. terms remain unresolved. | Prefer NOAA bytes for retrospective diagnostics. No predictor eligibility and no new independent labels. |
| [`sobhanmoosavi/us-weather-events`](https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events) | Author-processed airport weather observations. Card reports 2,071 stations, 49 states, January 2016 through December 2022. Its "Storm" definition starts at 60 km/h and includes ordinary rain, snow and fog categories. These differ from NOAA severe-event county labels. The cited collection paper identifies Weather Underground as an observation supplier, but does not establish the entire 2022 release's upstream rights. [P1] | CC BY-NC-SA 4.0; version 4; May 23, 2023. Card additionally limits use to noncommercial research/academic applications. Attribution and applicable ShareAlike conditions apply; original provider rights are not independently cleared here. [L1] | No direct temporal overlap with current training. Optional separate historical research only. Current CSV bytes, full schema, county matching and per-event publication times were not verified. Airport observations still cannot become target-day predictors. |
| [`lucassteuber/noaa-significant-us-storms`](https://www.kaggle.com/datasets/lucassteuber/noaa-significant-us-storms) | Card identifies NOAA NCEI Storm Events, 1950-2025, filtered by realized severity, damage, injury and fatality. Simplified fields include coordinates, date, event type, magnitude and impacts. A date-only mirror does not preserve a verified UTC interval or all original geography identifiers. | CC0: Public Domain; version 2; February 15, 2026. Original annual revision filenames and complete transformation lineage were not verified. | Reject as an expanded or independent label set. Outcome-based filtering changes prevalence and excludes routine reported events. Never use damage, casualties, narratives or selection membership as pre-event predictors. |
| [`playground-series-s4e5`](https://www.kaggle.com/competitions/playground-series-s4e5/data) and its linked Flood Prediction Factors source | The competition's indexed data description identifies generation by a deep learning model trained on Flood Prediction Factors. It is a synthetic benchmark, not verified county/day flood incidence. The original factors dataset's real observational origin, geography, dates and publication history were not established. | Competition-specific terms were not readable in this audit. The original factors metadata request returned HTTP 403. No license is asserted for either. | Explicitly reject synthetic Kaggle records and the unverified factors source for this sprint, including pretraining, augmentation and calibration. A permissive license, if later confirmed, would not establish authentic observations. |

The synthetic competition's current card could not be independently re-read. Its indexed description is limited evidence, and the rejection also follows from the absence of verified observational lineage. Automatic approval review denied a request to open Kaggle in the browser because it would expand browser-origin access. No browser permission was granted or bypassed; the three dataset cards above rely on the already retrieved public metadata endpoints.

## Leakage and matching requirements

1. Keep event time, interval end, first publication, revision time and retrieval time distinct. Reject joins that use `DATE <= D-1` as a substitute for publication evidence. Final event duration, damage or later quality flags can encode information learned after the cutoff.
2. Preserve original identifiers and source lineage. The Storm Events archive and Kaggle mirrors can duplicate the current NOAA labels. Reports, warnings and radar signatures also share storm dependence. Do not call agreement between these sources an independent test without establishing the reporting relationship.
3. Keep county, forecast-zone and marine identifiers separate. Match station/detection coordinates to versioned county polygons and retain spatial uncertainty. A name, postal code, airport code or nearest station is not a county FIPS key. Storm Events standard-time conversion remains the existing contract; do not copy that conversion onto explicit SWDI `Z` timestamps. [SOURCES.md](SOURCES.md)
4. Surface observations and numerical forecasts describe overlapping weather systems. GFS may also use surface observations in its analysis, but the particular GSOD stations' assimilation was not traced. Observation/forecast correlation is not new independent evidence. An observation after cutoff remains leakage even when it is highly correlated with an eligible forecast.
5. Treat empty responses, missing values and unobserved areas as unknown. The GSOD request for one Gainesville station/date range returned only a header. That verifies neither station inactivity nor absence of weather. The empty bounded Florida SWDI query likewise establishes no county-level negative.
6. Distinguish mature retrospective labels from a historical model deployment claim. Later label revisions may support a fixed retrospective evaluation. A claim that a model could have been fitted at an earlier date would additionally require proof that its training labels existed then. Freeze outcome revisions and leave split selection to the preregistered experiment.

## Actions for the source owner

Proceed with forecast-only predictor work under the existing disclosed Previous Runs limitation. No GSOD or SWDI observation predictor meets the required availability evidence in this audit. To reconsider one, obtain historical publication evidence, resolve interval semantics, define a reproducible availability rule and pin the bytes used. If that evidence cannot be obtained, keep the source retrospective.

Use SWDI radar data only if a clearly named detection diagnostic answers a separate question. First establish product/radar/day coverage, current precision and version provenance. Prefer official NOAA records over Kaggle mirrors. Do not change the current labels, select counties by outcomes or score a new source on the reserved validation/final geography as part of this audit.

## Retrieval evidence and primary URLs

All retrieval dates below are October 17, 2026. Small samples and public metadata remain only in `/tmp/planit-ml-source-audit-20261017/`. No credentials, Kaggle archives or bulk NOAA event files were used. Temporary files are verification aids, not a durable training-data cache.

| Retained artifact | What was verified | SHA-256 |
| --- | --- | --- |
| `gsod-lax-prefix.csv` | HTTP 206 range request `bytes=0-1999`; retained header and three January 1-3, 2024 records, 1,065 bytes. [N8] | `9c3528c5777cd07273d2b389cf3baf4698c5ea7283f4b37a076d5bd3384de4ee` |
| `swdi-query-small.csv` | Bounded HTTP read; retained header and four May 5, 2006 radar records, 273 bytes. The URL's `limit` parameter was not relied on as a response-size guarantee. [N9] | `dfd1f07a1eb1b646fbe0f8a0b12ca669305fcad288535fa71305e7be37e31fc4` |
| `swdi-v2-index.html` | Complete 70,443-byte directory listing used for the product filename spans, without event archive downloads. [N5] | `2b1e41de20b9aa0cfd3cc73f413900baa82da3a5aa3bff67fc22f4fe3335ad05` |
| `kaggle-gsod.json` | Public catalog metadata, not observation rows. [K1] | `e2f5bc2197f6f28ab5e605e9bb26d551d0a1f41ce68898f2e2dd8efefa35e7e2` |
| `kaggle-weather.json` | Public catalog metadata, not event rows. [K2] | `59aa3e1257d45be456c6a892aa94d1bc5b4b731cb5e90d805c711e622ee0cebe` |
| `kaggle-storm.json` | Public catalog metadata; event-count summaries were not used for county selection or split assessment. [K3] | `55531f1fe1156740b6c88ef2773f9ee9a80c760fc5ce50cad30eaecfa9c913c8` |

- N1: [NOAA ISD product and access documentation](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database).
- N2: [NOAA GSOD format README, dated October 28, 2020](https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt).
- N3: [NOAA C00516 landing page](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00516), [GSOD catalog](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc%3AC00516/html) and [GSOD year directories](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/).
- N4: [NOAA SWDI product, limitations and lightning access](https://www.ncei.noaa.gov/products/severe-weather-data-inventory).
- N5: [NOAA SWDI v2 bulk-file index](https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/). The directly retrieved HTML was used for the product filename spans, not a search engine's cached listing.
- N6: [NOAA SWDI known issues](https://www.ncei.noaa.gov/pub/data/swdi/KNOWN_ISSUES.txt).
- N7: [NOAA Storm Events FAQ](https://www.ncei.noaa.gov/stormevents/faq.jsp) and [annual revision archive](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/). NOAA describes a typical three-month publication delay; it is not an immutable release guarantee.
- N8: [NOAA 2024 Los Angeles GSOD station file](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/72295023174.csv). Only the bounded prefix was retrieved.
- N9: [NOAA historical hail schema request](https://www.ncei.noaa.gov/swdiws/csv/nx3hail/20060505:20060506?limit=3). Empty-response checks used [a small Florida bounding box](https://www.ncei.noaa.gov/swdiws/csv/nx3hail/20060505:20060506?bbox=-83,29,-82,30&limit=3) and [a short GSOD station/date request](https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-summary-of-the-day&stations=72214612816&startDate=2025-08-25&endDate=2025-08-30&format=csv).
- N10: [NWS disclaimer, reuse and third-party exclusions](https://www.weather.gov/disclaimer).
- N11: [NOAA operating-system upgrade and product retirement notice](https://www.ncei.noaa.gov/operating-system-upgrade-outage), corroborated by the [GSOD root directory's 2025 access-directory timestamp](https://www.ncei.noaa.gov/data/global-summary-of-the-day/). The exact final observation date is station-specific and was not sampled.
- N12: [NOAA SSODv2 successor documentation](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly/synoptic-summary-of-the-day).
- N13: [NOAA SWDI C00773 catalog, product layers and constraints](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00773). Its dataset publication date of January 1, 2006 is not the release time of each historical record; its update frequency is as needed.
- N14: [NOAA lightning public products and restricted internal products](https://www.ncei.noaa.gov/products/lightning-products).
- K1: [Kaggle NOAA GSOD card](https://www.kaggle.com/datasets/noaa/gsod), verified through its [public metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/noaa/gsod).
- K2: [Kaggle US Weather Events card](https://www.kaggle.com/datasets/sobhanmoosavi/us-weather-events), verified through its [public metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/sobhanmoosavi/us-weather-events).
- K3: [Kaggle NOAA Significant US Storms card](https://www.kaggle.com/datasets/lucassteuber/noaa-significant-us-storms), verified through its [public metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/lucassteuber/noaa-significant-us-storms).
- K4: [Kaggle Playground flood competition data description](https://www.kaggle.com/competitions/playground-series-s4e5/data) and [linked Flood Prediction Factors dataset](https://www.kaggle.com/datasets/brijlaldhankour/flood-prediction-factors). Current full terms and the original factors' provenance remain unresolved.
- P1: [Moosavi et al., US-Accidents collection paper](https://arxiv.org/pdf/1906.05409), cited by the Weather Events card; [Moosavi et al., event-pattern research](https://arxiv.org/abs/1902.06792). These papers establish a research collection lineage, not current provider permissions or historical publication timestamps.
- L1: [Creative Commons BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[N1]: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
[N2]: https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt
[N3]: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00516
[N4]: https://www.ncei.noaa.gov/products/severe-weather-data-inventory
[N5]: https://www.ncei.noaa.gov/pub/data/swdi/database-csv/v2/
[N6]: https://www.ncei.noaa.gov/pub/data/swdi/KNOWN_ISSUES.txt
[N7]: https://www.ncei.noaa.gov/stormevents/faq.jsp
[N8]: https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2024/72295023174.csv
[N9]: https://www.ncei.noaa.gov/swdiws/csv/nx3hail/20060505:20060506?limit=3
[N10]: https://www.weather.gov/disclaimer
[N11]: https://www.ncei.noaa.gov/operating-system-upgrade-outage
[N12]: https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly/synoptic-summary-of-the-day
[N13]: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00773
[N14]: https://www.ncei.noaa.gov/products/lightning-products
[K1]: https://www.kaggle.com/api/v1/datasets/view/noaa/gsod
[K2]: https://www.kaggle.com/api/v1/datasets/view/sobhanmoosavi/us-weather-events
[K3]: https://www.kaggle.com/api/v1/datasets/view/lucassteuber/noaa-significant-us-storms
[K4]: https://www.kaggle.com/competitions/playground-series-s4e5/data
[P1]: https://arxiv.org/pdf/1906.05409
[L1]: https://creativecommons.org/licenses/by-nc-sa/4.0/
