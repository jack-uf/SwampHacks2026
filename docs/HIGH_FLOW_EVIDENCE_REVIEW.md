# Evidence requirements for six-hour high-flow confirmation

The existing hourly studies support retrospective comparisons at new station IDs. They do not establish independent watershed replication or operational input availability. Keep every frozen promotion criterion, including average precision, unchanged. Higher event recall at a higher false-alert rate does not establish superior detection at equal burden.

This review separates inspected evidence from proposed confirmation requirements. It reads source documents, cached metadata and saved evaluations without fitting models or selecting thresholds. The target remains measured high-flow onset, not verified flooding or construction-site danger.

## Evidence on geographic independence

The [MLP onset protocol](MLP_ONSET_PROTOCOL.md) and [geographic protocol](MLP_GEO_PROTOCOL.md) explicitly say that all eligible CAMELS HUC4 groups appeared in earlier reservations. Their cohorts are fresh at station level. The latter also explicitly records that no watershed-polygon screen was applied. This review treats exhaustion as the protocols' recorded finding, rather than claiming a new exhaustive metadata count.

Distinct station IDs and HUC4 labels do not prove disjoint contributing areas. USGS describes HUCs as basin addresses that need not represent natural watersheds. Its NLDI basin endpoint instead accumulates upstream catchments for a mapped river reach. Reusing a HUC4 does not itself prove drainage overlap either. A same-HUC4 basin could be disjoint, but it cannot support an unseen-HUC4 claim. These are separate tests. [USGS hydrologic units](https://pubs.usgs.gov/gip/hydrologic_units/), [NLDI basin documentation](https://api.water.usgs.gov/docs/nldi/basin/).

The repository already contains a stronger geographic audit. [Response geography](../experiments/sprint-20261017-response/geography.json) records positive-area overlap exclusions despite its HUC4 screening, plus disagreements between station and reach HUC4 metadata. [The implementation](../scripts/response_research.py) checks indexed gauge identity, location and polygon validity before intersection testing. Its [protocol](RESPONSE_RESEARCH_PROTOCOL.md) permits overlap within a group, so those gauges cannot be counted as independent drainage replicates. These artifacts establish the need for geometry checks, not independence of later hourly cohorts.

Proposed requirements for a fresh watershed claim:

1. Freeze an exposure ledger covering every prior fitted, calibrated, selected, inspected and reserved cohort. Distinguish metadata-only candidate pools from actual reservations. Preserve the union of prior HUC4 exclusions required by the protocol.
2. Resolve each candidate and prior gauge to its exact indexed USGS identity and reach. Check coordinates, drainage area and HUC disagreements. Unresolved prior geometry blocks the independence claim; exclude unresolved candidates before viewing outcomes.
3. Retrieve and hash valid upstream polygons for the entire ledger and candidate set. Require zero positive-area intersection with all prior basins and all other independent confirmation groups. Boundary contact alone is permissible. Never erase overlap with a convenience tolerance or silent geometry repair.
4. Confirm with full-resolution polygons and an explicit coordinate reference system. NLDI defaults to simplified display geometry and whole local catchments. Whole-catchment accumulation can conservatively reject a candidate; a precise gauge-boundary claim requires the documented split-catchment operation. Preserve geometry provenance and manual adjudications. [NLDI parameters](https://api.water.usgs.gov/docs/nldi/basin/).

If no eligible unseen HUC4 remains in CAMELS, expand the metadata search to other USGS gauges before acquiring outcomes. Do not weaken HUC4 or polygon exclusions to fill a cohort. Non-overlapping basins can still share storms and climate. Spatial exclusion supports drainage separation, not statistical independence of their weather.

## Evidence on operational inputs

The hourly model does not currently require forecast weather. Its [feature definition and eligibility logic](../scripts/shift_hydrology.py) use lagged discharge, discharge dynamics, season and static attributes. Its two-hour receipt delay is an assumption. The parser accepts final approved observations, which are unsuitable as evidence of the values visible at historical issue time.

A concrete cached example is `experiments/sprint-20261017-mlp-geo/sources/07066000/2025-0-0.json.gz` and its `.request.json` manifest. The first reading has observation time `2025-01-01T00:00:00+00:00`, status `Approved`, and `last_modified` in August 2025. Retrieval occurred in October 2026. These dates establish neither first publication nor the original value. USGS defines `last_modified` as database refresh time, which can change without a measurement revision. The [continuous schema](https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/schema?f=html) exposes no historical value-vintage reconstruction in that field.

USGS reports typical quarter-hour recording and hourly transmission, subject to telemetry conditions. That typical cadence provides no station-specific two-hour receipt guarantee. Provisional values can change during review. A deployable input policy must explicitly address provisional measurements and unavailable history; this review does not change the approved-only research contract. [USGS update frequency](https://www.usgs.gov/faqs/how-often-are-real-time-streamflow-data-updated), [provisional-data statement](https://water.usgs.gov/data/provisional.html).

| Weather or discharge source | Inspected evidence | Remaining availability limit |
| --- | --- | --- |
| Caravan MultiMet HRES | `experiments/sprint-20261017-hydrology/research/hres-metadata.json` has daily dates and daily leads. Its request manifest pins the provider URL and hash. | The authors describe daily basin aggregation, occasional reforecasts and ignored dissemination delays. Daily aggregates cannot reconstruct within-day rainfall timing for six-hour features. [MultiMet sections 2 and 2.1](https://arxiv.org/html/2411.09459v1). |
| Open-Meteo historical products | Provider documentation distinguishes reanalysis, stitched historical forecasts, fixed lead offsets and individual runs. | A stitched series does not preserve an individual issue's forecast horizon. Use a documented run archive for that purpose. [Historical forecast documentation](https://open-meteo.com/en/docs/historical-forecast-api). |
| Open-Meteo Single Runs | Runs are addressable by initialization time. The documentation labels the early ECMWF archive as Cycle 49R1 hindcasts. | Initialization precedes public availability. Neither an addressable run nor its archive date proves operational issuance or timely historical receipt. Verify original operational versus hindcast provenance for each period. [Single Runs details](https://open-meteo.com/en/docs/single-runs-api). |
| Open-Meteo availability metadata | The provider distinguishes initialization, conversion completion and actual API availability, with server replication delays. | Current metadata is useful for prospective capture. It does not establish historical receipt for the repository's decisions. [Availability fields](https://open-meteo.com/en/docs/model-updates). |
| NOAA NWM retrospective | The cached `experiments/sprint-20261017-hydrology/research/nwm-readme.md` describes simulations driven by observed rain and reanalysis. | This is simulated historical discharge, not an as-issued discharge comparator. A separate operational forecast archive would need a cycle, publication and gauge-reach audit. [NOAA source README](https://raw.githubusercontent.com/NOAA-Big-Data-Program/bdp-data-docs/main/nwm/README.md). |

The viable operational confirmation is prospective capture, unless an independently auditable historical vintage archive is obtained. At each issue, save the exact response bytes, receipt timestamp, observation times, approval flags and available forecast cycles before scoring. Record missing inputs and withheld predictions. Preserve later approved discharge separately for outcome adjudication. An assumed larger lag cannot replace this evidence.

## Average precision and alert burden

Average precision, or AP, summarizes row-level precision across score thresholds. Event recall measures distinct onsets detected at one policy threshold. Neither metric implies the other. [Scikit-learn's AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [repository metric definitions](../scripts/shift_hydrology_metrics.py).

The saved six-hour comparison uses the ceiling rule for both LightGBM and linear extrapolation. Its achieved burdens still differ. Values below were read directly from [the evaluation artifact](../experiments/sprint-20261017-mlp-geo/evaluation.json), without recomputing predictions.

| Partition | Model | Events detected | False alerts per 100 eligible decisions | AP |
| --- | --- | ---: | ---: | ---: |
| Temporal | Linear | 18 / 76 | 0.629 | 0.0711 |
| Temporal | LightGBM | 25 / 76 | 0.932 | 0.1416 |
| Geographic | Linear | 18 / 58 | 0.662 | 0.1389 |
| Geographic | LightGBM | 24 / 58 | 1.079 | 0.1325 |

The candidate detects more events while issuing more false alerts. The geographic AP-difference interval includes zero, and the temporal HUC2 04 burden exceeds the frozen regional ceiling. The artifact marks both gates failed. Its shared-event median lead is three hours for both methods. A six-hour prediction horizon therefore does not mean six hours of warning.

Assessment: retain AP as a promotion gate under the existing [protocol](MLP_GEO_PROTOCOL.md). The [results document's](MLP_GEO_RESULTS.md) suggestion that AP measures the wrong thing is an interpretation, not an established consequence of these comparisons. The recall gain does not isolate model benefit from spending more alert budget. AP failure also does not prove that useful fixed-budget detection is impossible.

Proposed additional analysis should compare predeclared recall-versus-burden policies using development-selected thresholds for every comparator. Keep all existing gates. Test-label oracle thresholds may describe a frontier, but cannot support deployment or fresh confirmation. Report false-warning episodes and calendar coverage alongside false alerts per eligible decision; that denominator is neither false-positive rate nor the fraction of alerts that are false.

## Proposed fresh confirmation design

Freeze one six-hour comparison before any new final outcomes are inspected. Treat the current LightGBM result as hypothesis-generating. Use the polygon and exposure audit above to reserve genuinely new geography, with development, threshold selection and final confirmation groups separated before outcome acquisition. If sufficient geography cannot be found, report the design infeasible under current criteria.

Retain the existing onset definition, reference period, missingness rules, calibration separation and every promotion gate. These include event minimums, positive AP uncertainty bounds, event-recall gains, Brier comparisons, pooled and regional false-alert ceilings, and shared-event lead. Pin comparator artifacts and threshold rules, source contracts, hashes, exclusions and the analysis code before opening confirmation outcomes. Choose the final period and sample-size rationale in advance; stop inconclusively if event requirements fail.

Use drainage groups as spatial uncertainty blocks, preserving paired model comparisons. Add prespecified temporal or storm blocks to examine shared-weather dependence, without replacing the required HUC4 analysis. Evaluate each frozen policy at its achieved burden and report all regional failures. Repeated experiments on newly selected stations from exposed regions do not restore a regional holdout.

Run the input-availability confirmation alongside the statistical test. Freeze one explicit observation-quality policy and require actual receipt by issue time for every feature. If weather is added, require original operational runs, subdaily valid intervals and audited basin aggregation. Keep revised outcome records out of feature reconstruction. Passing statistical gates still leaves the protocol's site-to-gauge applicability requirement unresolved and supports no claim of verified flood or field-safety performance.
