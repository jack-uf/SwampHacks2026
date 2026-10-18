# Additional data for six-hour high-flow onset

Acquire archived HRRR precipitation first. It supplies forecast rain during the warning horizon, which lagged discharge and ERA5 historical precipitation cannot supply. A small upstream-discharge experiment at Tug Fork is also feasible from metadata. For new geography, expand to the national USGS catalog or GAGES-II rather than searching CAMELS again. These are acquisition recommendations, not measured predictive gains.

This review used existing raw catalogs and public metadata/index probes on October 18, 2026. It acquired no flow observations and inspected no new final outcomes. Open-Meteo ERA5 precipitation and CAMELS static attributes remain outside this task. Commands below reproduce successful probes unless explicitly marked as unrun acquisition examples.

## Archived operational precipitation

### HRRR: first choice for forecast rainfall

[NOAA documents](https://rapidrefresh.noaa.gov/hrrr/) hourly, 3-km HRRR forecasts and links the [public archive](https://registry.opendata.aws/noaa-hrrr-pds/), which extends back to 2014. Index requests for January 1, 2020 and January 1, 2024 both returned HTTP 200. Each six-hour file contains separate APCP messages for cumulative hours 0-6 and interval 5-6. Select by accumulation interval, not the first precipitation match.

```sh
curl -fsS 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20200101/conus/hrrr.t00z.wrfsfcf06.grib2.idx'
curl -fsS 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240101/conus/hrrr.t00z.wrfsfcf06.grib2.idx'
```

For the 2024 example, the 0-6 APCP message starts at byte 60721951 and the next message at 61195477. A 16-byte range probe returned HTTP 206 and the GRIB signature. The full precipitation message is therefore retrievable without downloading the 150,990,209-byte surface file. This acquisition example was not run:

```sh
curl -f --range 60721951-61195476 \
  'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240101/conus/hrrr.t00z.wrfsfcf06.grib2' \
  -o /tmp/hrrr-20240101-00-f06-apcp.grib2
```

Offsets are specific to that object. Derive them anew from each index. Fetch interval APCP for each needed valid hour, then aggregate over upstream basin cells. This tests storm placement and rain timing; a gauge-point extraction alone does not represent the contributing watershed.

The cycle timestamp is initialization, not receipt. At issue time T, a cycle initialized at T is not automatically usable. An older cycle needs leads that cover T through T+6. Preserve initialization, valid interval and availability evidence separately. HRRR v4 began December 2, 2020, so the existing training period crosses a model change. Two successful dates establish access, not complete historical coverage. Archive object modification times alone do not certify original dissemination.

### NWM operational forcing: usable historical alternative, larger downloads

The [AWS operational bucket](https://registry.opendata.aws/noaa-nwm-pds/) retains only four weeks. However, direct Google Cloud listings found historical `forcing_short_range` objects. This materially differs from the NWM retrospective simulation archive. [NOAA describes](https://water.noaa.gov/about/nwm) operational short-range forcing as HRRR/RAP, with hourly forecast cycles out to 18 hours. The [NCEP forcing inventory](https://www.nco.ncep.noaa.gov/pmb/products/nwm/nwm.t00z.short_range.forcing.f001.conus.nc.shtml) exposes `RAINRATE` in mm/s, `reference_time`, valid `time` and `time_bounds`.

These bounded listing commands succeeded:

```sh
curl -fsS 'https://storage.googleapis.com/storage/v1/b/national-water-model/o?prefix=nwm.20200101%2Fforcing_short_range%2F&maxResults=2'
curl -fsS 'https://storage.googleapis.com/storage/v1/b/national-water-model/o?prefix=nwm.20220101%2Fforcing_short_range%2Fnwm.t00z&maxResults=10'
curl -fsS 'https://storage.googleapis.com/storage/v1/b/national-water-model/o?prefix=nwm.20240101%2Fforcing_short_range%2Fnwm.t00z&maxResults=10'
```

An exact usable precipitation-forcing object is [2024-01-01 00Z, lead 1](https://storage.googleapis.com/national-water-model/nwm.20240101/forcing_short_range/nwm.t00z.short_range.forcing.f001.conus.nc). Listing metadata reports 140,489,000 bytes. The sampled files are roughly 140-152 MB per lead, so HRRR field-only ranges are the cheaper first acquisition. The NetCDF payloads were not downloaded or decoded here. Verify the historical file's variable units, packing and time bounds before integrating rain rate over an interval.

Timing is the main limitation. The 2020 lead-1 object was created at 01:45:39Z and the 2024 lead-1 object at 01:38:14Z. Those objects were unavailable in this bucket at their nominal 00Z initialization. The sampled January 2022 files instead have May 2023 creation times, consistent with later archive ingestion. Their metadata cannot reconstruct 2022 receipt. A January 2018 listing of the same forcing prefix returned no objects. Do not infer continuous coverage or a universal publication delay from these samples.

### NWM retrospective precipitation: lower priority

The [retrospective registry](https://registry.opendata.aws/nwm-archive/) describes AORC historical forcing, not as-issued precipitation forecasts. A metadata listing confirmed `CONUS/zarr/forcing/precip.zarr/`:

```sh
curl -fsS 'https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/?list-type=2&prefix=CONUS%2Fzarr%2Fforcing%2F&delimiter=%2F&max-keys=10'
```

It is an alternative historical rainfall estimate if ERA5 proves insufficient. Prioritize it after the current ERA5 experiment because it does not fill the operational forecast gap. Retrospective NWM streamflow is simulated and is not a substitute for a historical issued forecast or observed outcome. No flow payload was requested.

## Subdaily connected-gauge training data

Start with upstream **03212980, Dry Fork at Beartown**, predicting downstream **03213700, Tug Fork at Williamson**. The downstream station is already a development station in `experiments/sprint-20261017-shift-hydrology/stations.json`. Its upstream area ratio is 0.223 in `experiments/sprint-20261017-upstream/expanded-pairs.json`. This is the smallest addition to an existing six-hour cohort.

Three earlier development pairs have promising subdaily catalog spans. A fresh [USGS series-catalog request](https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=03213700,03212980,07056000,07055875,07376000,07375800&seriesCatalogOutput=true&outputDataTypeCd=iv&parameterCd=00060) returned the following `uv`, parameter `00060` entries. All extend through October 2026; endpoints do not establish completeness.

| Upstream to downstream | Upstream subdaily start | Downstream subdaily start | Acquisition use |
| --- | --- | --- | --- |
| 03212980 to 03213700 | 1997-05-01 | 1997-07-03 | First development addition |
| 07055875 to 07056000 | 1995-06-01 | 1991-10-01 | Additional development pair |
| 07375800 to 07376000 | 2001-10-01 | 1995-10-01 | Additional development pair |

The retained catalog for upstream 12357500 starts subdaily discharge on September 1, 2026. Exclude that upstream source from the earlier training years. The previous daily pilot already stopped on unavailable reference observations for this station, as recorded in [UPSTREAM_RESEARCH_RESULTS.md](../../docs/UPSTREAM_RESEARCH_RESULTS.md).

The retained `research/*catalog-*.rdb` files contain 94 distinct upstream sites with subdaily discharge entries; 48 have endpoints spanning 2020-2024. Seven development downstream stations have at least one such navigation candidate. This metadata-only recount deduplicated complete RDB rows and filtered `data_type_cd=uv`, `parm_cd=00060`, `begin_date < 2020-01-01`, `end_date >= 2024-12-31`. It did not verify flow coverage, area ratios or hydraulic connectivity for the wider pool. The earlier 33-site count applied daily-record filters and is not the subdaily candidate count.

Use the [current USGS metadata API](https://api.waterdata.usgs.gov/docs/ogcapi/) to identify the instantaneous series before downloading:

```sh
curl -fsS 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/time-series-metadata/items?f=json&monitoring_location_id=USGS-03212980&parameter_code=00060&limit=10'
```

That probe returned the instantaneous series with `statistic_id=00011` and `computation_period_identifier=Points`, alongside daily and annual-event series. Preserve that distinction. An unrun, development-only acquisition example is:

```sh
curl -fsS 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items?f=json&monitoring_location_id=USGS-03212980&parameter_code=00060&datetime=2020-01-01T00%3A00%3A00Z%2F2020-01-08T00%3A00%3A00Z&limit=50000&skipGeometry=true' \
  -o /tmp/dry-fork-development-2020-week.json
```

Follow pagination and retain qualifiers when actually acquiring training data. The [existing evidence review](../../docs/HIGH_FLOW_EVIDENCE_REVIEW.md) explains why final approved observations cannot certify historical receipt. Keep the six-hour model's assumed two-hour delay visible. Upstream information helps only if its propagation time leaves useful downstream lead after that delay. Neither catalog spans nor navigation measure travel time. Do not open the thirteen reserved pairs' outcomes for this development check.

## Watersheds beyond CAMELS

The [existing independence audit](../high-flow-20261018/independence-audit.json) found no unused eligible CAMELS station in an unexposed HUC4. A broader USGS metadata search does find additional HUC4s. Single-state requests work; the attempted comma-separated state request returned HTTP 400.

```sh
curl -fsS 'https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=fl&siteType=ST&hasDataTypeCd=iv&parameterCd=00060&siteStatus=all&siteOutput=expanded'
curl -fsS 'https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=tx&siteType=ST&hasDataTypeCd=iv&parameterCd=00060&siteStatus=all&siteOutput=expanded'
```

Comparing these responses against all HUC4s in the retained 671-station `experiments/sprint-20261017-hydrology/research/sites.json` found 30 Florida and 21 Texas station records outside those HUC4s with drainage area strictly between 20 and 1,200 square miles. Oregon supplied none under that HUC4 screen. These are metadata candidates, not independent watershed counts or a complete eligibility screen.

Concrete candidates with fresh cataloged subdaily discharge spans are:

| Gauge | Name | HUC4 | Subdaily start |
| --- | --- | --- | --- |
| 02255600 | Fisheating Creek near Lake Placid, FL | 0309 | 2007-10-01 |
| 02256500 | Fisheating Creek at Palmdale, FL | 0309 | 2007-10-01 |
| 07235000 | Wolf Creek at Lipscomb, TX | 1110 | 2007-10-01 |
| 08033900 | East Fork Angelina River near Cushing, TX | 1202 | 2015-09-30 |
| 08034500 | Mud Creek near Jacksonville, TX | 1202 | 2007-10-01 |

[The five-site metadata query](https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=02255600,02256500,07235000,08033900,08034500&seriesCatalogOutput=true&outputDataTypeCd=iv&parameterCd=00060) returned endpoints through October 2026. The two Fisheating sites are candidates for one drainage group, not two independent replicates. Regulation, complete subdaily history and the historical daily reference still need checking.

For a national metadata seed with disturbance classification and basin boundaries, use [USGS GAGES-II](https://pubs.usgs.gov/publication/70046617). It describes 9,322 gauges, with eligibility assessed as of water year 2009. Current subdaily availability must come from USGS metadata. The [ScienceBase file manifest](https://www.sciencebase.gov/catalog/item/631405bbd34e36012efa304a?format=json) returned these acquisition targets:

- [Gauge points and summary attributes](https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a?name=gagesII_9322_point_shapefile.zip), 743,919 bytes.
- [Basin characteristics and report](https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a?name=basinchar_and_report_sept_2011.zip), 55,147,470 bytes.
- [Basin boundaries](https://www.sciencebase.gov/catalog/file/get/631405bbd34e36012efa304a?name=boundaries_shapefiles_by_aggeco.zip), 204,809,860 bytes.

Only the manifest was fetched. These files expand watershed discovery beyond CAMELS; they do not establish new independent outcomes. Full-resolution [NLDI basin geometry](https://api.water.usgs.gov/docs/nldi/basin/) can avoid downloading all boundaries initially. This documented example was not run:

```sh
curl -fsS 'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-07235000/basin?f=json&simplified=false&splitCatchment=true'
```

Check candidate polygons against every prior fitted, inspected and reserved basin, not just the old CAMELS list. Group connected gauges together. Preserve the existing HUC exclusions; disjoint drainage does not establish independent storms. This review establishes actionable metadata sources and candidate IDs, not a certified final cohort.
