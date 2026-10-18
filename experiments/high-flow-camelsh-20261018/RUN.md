# CAMELSH bounded acquisition

CAMELSH is usable for the high-flow onset population. A 570-gauge subset covering every HUC02 region was fetched by HTTP byte range from the open Hourly2.zip streamflow archive (Zenodo record 16729675, CC-BY-4.0), and the existing `scripts.shift_hydrology` decision-row build runs on it unmodified. This directory holds the evidence; extracted NetCDF files and small metadata artifacts live under `/tmp/camelsh-20261018`. No existing script or experiment was modified.

Kind: runbook with reference tables. Audience: whoever continues the high-flow work. Purpose: reproduce the acquisition, check the compatibility claim, and see exactly what was and was not downloaded. Non-goals: independence certification, full-population labelling, model training.

## Files in this directory

| File | Contents |
| --- | --- |
| `archive-index.json` | All 5,767 archive members with local header offsets, compressed and uncompressed sizes, CRC32, plus central-directory offsets and hashes of the fetched head and tail. |
| `metadata-artifacts.json` | Source URLs, byte counts and verified MD5s for `attributes.7z` and `info.csv`. |
| `selection.json` | The 570 chosen stations with GAGES-II metadata, tiers, the per-region quota rule, and sha256 hashes of the fitted and reserved exclusion sources. |
| `fetch-log.jsonl` | One row per fetched member: byte range, request bytes, member CRC, extracted-file sha256. |
| `fetch-summary.json` | Fetch totals and failures (zero failures). |
| `coverage.csv`, `coverage-summary.json` | Per-gauge and aggregate coverage of the fetched subset. |
| `onset-check.json`, `onset-sample-rows.csv.gz` | Decision-row results for the 20-gauge onset sample. |
| `usgs-crosscheck.json` | Live USGS comparison for two gauges, which pins down units and aggregation. |
| `acquisition-manifest.json` | The single assembly of dataset facts, source URLs, byte counts, overlap counts and the not-acquired list. |

## Dataset facts

Streamflow archive. `Hourly2.zip`, Zenodo record 16729675, 4,176,462,827 bytes, MD5 `8bab1c99329db3fa6b1b6cb464a2a573` as published. The full file was never downloaded, so that MD5 is unverified here; member CRC32 values from the central directory are the integrity check this acquisition used. The archive holds 5,767 members named `Hourly2/<STAID>_hourly.nc`, 4,175,308,985 compressed bytes, 19,759,884,874 bytes uncompressed. Evidence: `archive-index.json`.

Per-gauge file format. NetCDF4/HDF5. Variables: `time` (float64, units `hours since 1980-01-01 00:00:00`, 394,488 hourly steps from 1980-01-01 00:00:00 to 2024-12-31 23:00:00 UTC), `streamflow` (float32, `m3 s-1`), `water_level` (float32, `m`). Missing hours are IEEE NaN; no interpolation exists in the released files. The same NetCDF structure is enforced by `read_streamflow` in `scripts/high_flow_camelsh.py`.

Statistic semantics. Streamflow is the hourly mean of USGS Instantaneous Values Service readings (parameter 00060), labelled by hour start, converted from cfs to cubic metres per second and standardised to UTC. Evidence: `functions/usgsdownload_Hourly.m` (hour-start binning with a per-hour mean), `functions/datatransform.m` (the 0.0283168 cfs-to-cms constant), the repository README and the paper (Tran et al., Scientific Data 2025, doi:10.1038/s41597-025-05612-6, PMC12304116). `usgs-crosscheck.json` confirms this against live USGS data.

Quality flags. All USGS flag classes are retained; the paper states the original records were used "in its entirety rather than removing any records". The August 2025 update description says significant negative values were removed and data quality checks were performed. Evidence: paper text and the record 16729675 description.

Attributes and boundaries. The 9,008-station attribute set ships as `attributes.7z` (19,067,073 bytes, verified MD5 `3c43c3df3809e69f5f1798f58d5dc62a`) with 439 GAGES-II attributes, 195 HydroATLAS attributes and nine NLDAS-2 climate indices. Basin polygons are provided in `shapefiles.7z` (530,604,106 bytes) as `CAMELSH_shapefile.shp` (GAGES-II derived) and `CAMELSH_shapefile_hydroATLAS.shp`, both carrying a `GAGE_ID` field. Evidence: repository README, Zenodo file lists, extracted attribute tables.

Station inventory. `info.csv` (1,162,939 bytes, verified MD5 `cb5872491b17c077a27fa24dc9aaac98`) lists 9,008 stations with per-year observed-hour counts for 1980 through 2024; 3,166 stations report nonzero observed hours. Evidence: `metadata-artifacts.json` and the file itself.

Licence and access. Records 15066778, 16729675, 15264814 and 21302482 are CC-BY-4.0 and open. The older streamflow updates are access-restricted: 15413207 (5,188 gauges, `Hourly.7z`), 16716635 and 14889026; the open August 2025 `Hourly2.zip` supersedes them. Evidence: Zenodo record metadata.

Other release sizes. `timeseries.7z` 21,247,136,353 bytes (NLDAS-2 forcing plus streamflow for 3,166 gauges), `timeseries_nonobs.7z` 35,655,472,043 bytes, ERA5-Land yearly archives about 2.45 GB per year of 46. Evidence: Zenodo file lists for records 15066778, 15070091 and 15264814.

## Compatibility with the frozen onset definition

The frozen protocol in `scripts/shift_hydrology.py` builds hourly series from USGS instantaneous values (statistic 00011, cfs), takes the 95th percentile of 2000-2015 daily-mean discharge as the reference, and labels an onset where hourly flow exceeds the reference after 24 hours below it. CAMELSH differs in four documented ways, none of which block the build.

| Property | Frozen protocol | CAMELSH | Effect |
| --- | --- | --- | --- |
| Aggregation | last instantaneous reading per right-closed hour | mean of instantaneous readings per hour | sub-hourly peaks are slightly smoothed; thresholds cross marginally less sharply |
| Hour label | hour end | hour start | the two series overlay with a one-hour offset |
| Units | cfs | m3 s-1 | exact conversion constant available; the model features are scale-free |
| Quality policy | Approved status, no qualifier, non-negative | every flag class retained; negative values kept in the original release, significant negatives removed in Hourly2 | hours our pipeline would drop are present in CAMELSH |

Missing values behave the same in both: NaN or excluded hours never carry interpolated water, and the eligibility masks in `decision_rows` treat them as ineligible.

Evidence that the build works. `onset-check.json` records the decision-row build for a deterministic 20-gauge sample: 17 gauges produced labels, three were excluded because fewer than 3,000 daily values exist over 2000-2015 to form a reference. The sample produced 3,846 events and 22,103 positive rows across the three horizons; event counts range from 57 to 911 per gauge over the full 45-year record, and positives scale with horizon as expected. `onset-sample-rows.csv.gz` carries the label rows for three of those gauges. The reference uses daily means reconstructed from the hourly series, the same statistic the frozen `freeze()` step takes from USGS daily values.

The first three events of station 01031500 reproduce the exact rule by hand:

```
.venv/bin/python -c "from pathlib import Path; from scripts.high_flow_camelsh import read_streamflow, reference_from_hourly, CFS_PER_CMS; q = read_streamflow(Path('/tmp/camelsh-20261018/extracted/01031500_hourly.nc')) * CFS_PER_CMS; ref, days, d0, d1 = reference_from_hourly(q); below = q.le(ref).where(q.notna()); onset = q.gt(ref) & below.shift(1).rolling(24).sum().eq(24); print(ref, len(q.index[onset]), q.index[onset][:3].tolist())"
```

The command prints reference 3206.73, 160 events, and first events on 1990-10-14, 1990-10-24 and 1990-10-29, each with a full 24 hours below the reference beforehand.

The cross-check against live USGS data is decisive. For stations 10183500 and 01105880 over 2024-05-01 to 2024-06-30, CAMELSH and the hourly mean of USGS readings agree with correlation 1.0000 and median ratio 1.0000 on 1,464 overlapping hours each, with a maximum absolute difference of 0.0 and no hours present on only one side. All readings in that window are Approved with no qualifier, so the flag-policy difference is not observable there and stays a documented caveat.

Verdict: keep CAMELSH rows as a separate provenance from the USGS-derived rows, convert units with the 0.0283168 constant, and the frozen label definition applies unchanged. Re-applying the Approved-only filter inside CAMELSH is not possible because the flags are not shipped.

## What was acquired

Index step: 4,096 header bytes plus the last 8,000,000 bytes of the archive; the central directory (553,846 bytes at offset 4,175,908,959) yields every member address. Metadata step: `attributes.7z` and `info.csv`, both MD5-verified against Zenodo.

Member step: the 570 selected gauges were fetched as local header plus compressed payload byte ranges, then inflated and checked against the central directory's uncompressed size and CRC32. Total request bytes 419,982,291, which is 10.1 percent of the archive's compressed bytes, and every member passed. One rate-limit episode (HTTP 429) interrupted the first attempt after 47 members; the fetch honours Retry-After and resumes from `fetch-log.jsonl` without refetching verified members. Evidence: `fetch-summary.json`, `fetch-log.jsonl`.

Selection rule: per-HUC02 quota of 30 gauges across 19 regions; within a region, tier A (info.csv shows at least 8,760 observed hours over 2019-2024) first, then tier U (archive member but info.csv reports zero hours), then tier B; ties broken by `sha256("camelsh42:" + station)`. The result is 538 tier A and 32 tier U gauges. Evidence: `selection.json`.

Overlap with our cohorts. All 88 fitted gauges and all 179 reserved confirmation gauges appear in the 9,008-station registry and in the archive; the selection excludes both sets by construction, and no fitted or reserved gauge was extracted or read. Of the 3,166 stations with observed hours in `info.csv`, 3,056 are in the archive and 110 are absent; `acquisition-manifest.json` lists the first ten absent ids. The 179 reserved gauges remain untouched as a confirmation cohort; only their member offsets and sizes appear in `archive-index.json`.

Coverage of the fetched subset. 570 gauges read; 555 have data somewhere in 2021-2024, 553 have at least half a year, 550 have at least a full year, and the subset carries 114,781,707 valid hourly values overall. Per-gauge numbers are in `coverage.csv`.

Disk. The volume had 68 GiB free when the acquisition started and 64 GiB free afterwards. This directory occupies 4.6 MB of the repository; `/tmp/camelsh-20261018` occupies 1.9 GB (extracted NetCDF files plus the small raw artifacts). Rebuilding the extracted files needs about 420 MB of network traffic and the commands below.

## Exact commands

Run from the repository root. `probe`, `metadata`, `fetch` and `crosscheck` need outbound network access; `7z` must be on PATH for `metadata`; `h5py` must be installed once into the venv.

```
.venv/bin/python -m pip install h5py
.venv/bin/python -m scripts.high_flow_camelsh probe
.venv/bin/python -m scripts.high_flow_camelsh metadata
.venv/bin/python -m scripts.high_flow_camelsh select
.venv/bin/python -m scripts.high_flow_camelsh fetch
.venv/bin/python -m scripts.high_flow_camelsh verify
.venv/bin/python -m scripts.high_flow_camelsh crosscheck
.venv/bin/python -m scripts.high_flow_camelsh manifest
```

`fetch --limit N` fetches only the first N selected members, which is how the path was smoke-tested. `all` runs the full sequence.

## What was not acquired

| File | Record | Size | Reason |
| --- | --- | --- | --- |
| `timeseries.7z` | 15066778 | 21,247,136,353 | Original NLDAS-2 forcing plus streamflow bundle; superseded for streamflow by Hourly2.zip and not needed for the compatibility check. |
| `timeseries_nonobs.7z` | 15070091 | 35,655,472,043 | Gauges without observed streamflow; outside the training population. |
| `shapefiles.7z` | 15066778 | 530,604,106 | Catchment polygons for all 9,008 basins, including the reserved cohort; deferred until a candidate cohort is chosen. |
| `Hourly.7z` | 15413207 | not published | Access-restricted 5,188-gauge archive, superseded by the open Hourly2.zip. |
| ERA5-Land yearly archives | 15264814 | about 2.45 GB per year | Hourly ERA5-Land forcing for 9,008 basins; the project already provides its own weather pipeline, and this acquisition needs streamflow semantics. |
| `NWM.7z`, `NWM_Q_nc.zip`, `CAMELSH_FCS.zip`, `StreamTemperature.zip` | 16763144, 21302482, 21324088 | 3.4 GB, 2.5 GB, 32.5 GB, 0.97 GB | Benchmark simulations, actual-forecast archives and stream temperature; not streamflow observations. |

## Next step

The subset is ready to become a labelled cohort, in this order. First, independence screening: the 570 gauges share HUC regions with prior work, so check boundary overlap against the fitted and previously exposed catchments before training use (the polygons come from `shapefiles.7z` once a cohort is fixed, and `selection.json` carries HUC02, HUC8 and coordinates). Second, run the decision-row build across all fetched gauges with recent coverage (553 have at least half a year in 2021-2024), which extends the current 1,948 positive rows in `experiments/high-flow-signals-20261018/development.csv.gz`. Third, decide the forcing source for the neural models: CAMELSH ships NLDAS-2 inside `timeseries.7z` and ERA5-Land in record 15264814, while the project already has an Open-Meteo pipeline; mixing the two forcing provenances within one training matrix would break the dataset contract. The 45-year CAMELSH record also opens a fourth option, labels on pre-2021 periods, which needs a protocol change rather than a silent extension.
