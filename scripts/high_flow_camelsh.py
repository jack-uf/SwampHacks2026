"""Bounded CAMELSH acquisition and onset-compatibility check (high-flow PRIORITY 3).

CAMELSH (Catchment Attributes and Hourly HydroMeteorology for Large-Sample
Studies, Tran et al., Scientific Data 2025, doi:10.1038/s41597-025-05612-6) is
a large-sample hourly hydrometeorological dataset for CONUS. The latest open
streamflow archive, Hourly2.zip (Zenodo record 16729675, CC-BY-4.0), holds one
NetCDF file per gauge: hourly streamflow and gage height for 5,767 USGS gauges,
1980-01-01 00:00:00 to 2024-12-31 23:00:00 UTC, streamflow in cubic metres per
second, missing hours as IEEE NaN.

This module runs one canonical path:

1. `probe` fetches the ZIP tail and header over HTTP byte ranges and parses the
   central directory, so individual members can be addressed without
   downloading the 4.18 GB archive.
2. `metadata` downloads and verifies the small CAMELSH metadata artifacts
   (attributes.7z and info.csv from record 15066778).
3. `select` builds a region-stratified gauge sample (HUC02 quota) that excludes
   the 88 fitted gauges and the 179 reserved confirmation gauges.
4. `fetch` pulls only the selected members by byte range and verifies every
   member CRC. Bulk bytes live under /tmp/camelsh-20261018, never in the repo.
5. `verify` reads every fetched series with h5py, writes coverage statistics,
   and runs the existing scripts.shift_hydrology decision-row build on a
   deterministic sample to confirm onset labels are produced sensibly.
6. `crosscheck` compares fetched series against live USGS instantaneous values
   for one window, which pins down units, aggregation and hour labelling.
7. `manifest` assembles the acquisition manifest with source URLs, byte
   counts, checksums and the list of artifacts that were not acquired.

Everything writes to experiments/high-flow-camelsh-20261018 (small JSON/CSV
evidence) and /tmp/camelsh-20261018 (archive bytes and extracted NetCDF files).
Existing scripts and experiments are read-only inputs.
"""
import argparse
import binascii
import hashlib
import json
import struct
import subprocess
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from scripts.flood_sprint import write_json
from scripts.shift_hydrology import decision_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/high-flow-camelsh-20261018'
BULK = Path('/tmp/camelsh-20261018')
EXTRACTED = BULK / 'extracted'
ATTRIBUTES_DIR = BULK / 'attributes' / 'attributes'

ARCHIVE = {
    'record': 16729675,
    'record_url': 'https://zenodo.org/records/16729675',
    'file': 'Hourly2.zip',
    'url': 'https://zenodo.org/api/records/16729675/files/Hourly2.zip/content',
    'bytes': 4_176_462_827,
    'md5': '8bab1c99329db3fa6b1b6cb464a2a573',
}
META_ARTIFACTS = {
    'attributes.7z': {
        'url': 'https://zenodo.org/api/records/15066778/files/attributes.7z/content',
        'bytes': 19_067_073, 'md5': '3c43c3df3809e69f5f1798f58d5dc62a'},
    'info.csv': {
        'url': 'https://zenodo.org/api/records/15066778/files/info.csv/content',
        'bytes': 1_162_939, 'md5': 'cb5872491b17c077a27fa24dc9aaac98'},
}
TAIL_BYTES = 8_000_000
MEMBERS_EXPECTED = 5_767
CD_SIZE_EXPECTED = 553_846

FITTED_SOURCE = ROOT / 'experiments/high-flow-signals-20261018/development.csv.gz'
RESERVED_SOURCE = ROOT / 'experiments/high-flow-signals-20261018/fresh-geography/camels-screen.json'

QUOTA_PER_REGION = 30
TIER_A_HOURS = 8_760
RECENT_YEARS = [str(y) for y in range(2019, 2025)]
DECISION_LIMIT = 20
DECISION_MIN_HOURS = 4_380
REFERENCE_START = pd.Timestamp('2000-01-01', tz='UTC')
REFERENCE_END = pd.Timestamp('2015-12-31', tz='UTC')
REFERENCE_MIN_DAYS = 3_000

# CAMELSH datatransform.m converts streamflow with this exact constant (cfs to
# m3 s-1). The inverse restores the USGS cubic-feet value for the helpers.
CMS_PER_CFS = 0.0283168
CFS_PER_CMS = 1.0 / CMS_PER_CFS

HOURS = 394_488  # 1980-01-01 00:00 UTC through 2024-12-31 23:00 UTC inclusive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def http_range(client, url, start, end, tries=6):
    """Fetch an inclusive byte range; every response must be a complete 206.

    Zenodo answers bursts with HTTP 429. The retry honours Retry-After when it
    is a small integer and otherwise backs off, so a rate-limit episode slows
    the fetch instead of failing it.
    """
    expected = end - start + 1
    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    last = None
    for attempt in range(tries):
        time.sleep(0.2 + 0.1 * attempt)
        try:
            response = client.get(url, headers=headers)
        except httpx.HTTPError as error:
            last = error
            time.sleep(min(3 * (attempt + 1), 20))
            continue
        if response.status_code == 206:
            if len(response.content) != expected:
                last = RuntimeError('Short range: %d of %d bytes' % (len(response.content), expected))
                time.sleep(min(3 * (attempt + 1), 20))
                continue
            return response.content
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', '')
            wait = (int(retry_after) if retry_after.isdigit() and 0 < int(retry_after) <= 300
                    else min(10 * (attempt + 1), 120))
            last = RuntimeError('HTTP 429')
            print('rate limited; waiting %d seconds' % wait, flush=True)
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            last = RuntimeError('HTTP %d' % response.status_code)
            time.sleep(min(5 * (attempt + 1), 60))
            continue
        raise RuntimeError('Unexpected HTTP %d for range request' % response.status_code)
    raise RuntimeError('Range fetch failed for %s [%d-%d] after %d tries: %s'
                       % (url, start, end, tries, last))


def parse_central_directory(tail, total_bytes):
    """Parse the end-of-central-directory record and every entry after it."""
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0:
        raise ValueError('No end-of-central-directory record in fetched tail')
    eocd = struct.unpack_from('<IHHHHIIH', tail, pos)
    comment_len = eocd[7]
    if pos + 22 + comment_len != len(tail):
        raise ValueError('EOCD is not at the end of the fetched tail')
    n_total, cd_size, cd_off = eocd[4], eocd[5], eocd[6]
    if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        raise ValueError('Zip64 archives are out of scope for this bounded reader')
    tail_start = total_bytes - len(tail)
    if cd_off < tail_start or cd_off + cd_size > total_bytes:
        raise ValueError('Central directory is not fully inside the fetched tail')
    base = cd_off - tail_start
    cd = tail[base:base + cd_size]
    members, p = [], 0
    while p + 46 <= len(cd):
        if cd[p:p + 4] != b'PK\x01\x02':
            raise ValueError('Bad central-directory signature at offset %d' % p)
        (made, need, flags, method, mtime, mdate, crc, csize, usize, fnlen,
         extralen, clen, dstart, iattr, eattr, lho) = struct.unpack_from('<HHHHHHIIIHHHHHII', cd, p + 4)
        name = cd[p + 46:p + 46 + fnlen].decode('utf-8')
        extra = cd[p + 46 + fnlen:p + 46 + fnlen + extralen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            raise ValueError('Zip64 entry fields are out of scope: %s' % name)
        # Every member sets general-purpose bit 3 (a data descriptor follows the
        # compressed payload). The central directory still carries authoritative
        # sizes and CRCs, and each member is addressed as local header + csize.
        if flags & ~0x8:
            raise ValueError('Unexpected general-purpose flags %d on %s' % (flags, name))
        members.append(dict(name=name, method=method, crc32=crc, csize=csize,
                            usize=usize, local_off=lho))
        p += 46 + fnlen + extralen + clen
    if p != len(cd):
        raise ValueError('Central directory has %d trailing bytes' % (len(cd) - p))
    return dict(entries=members, cd_off=cd_off, cd_size=cd_size)


def probe():
    """Index Hourly2.zip without downloading it: header, tail and CRC fields."""
    BULK.mkdir(parents=True, exist_ok=True)
    (BULK / 'raw').mkdir(exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        head = http_range(client, ARCHIVE['url'], 0, 4095)
        tail = http_range(client, ARCHIVE['url'], ARCHIVE['bytes'] - TAIL_BYTES, ARCHIVE['bytes'] - 1)
    (BULK / 'raw' / 'zip-head.bin').write_bytes(head)
    (BULK / 'raw' / 'zip-tail.bin').write_bytes(tail)
    parsed = parse_central_directory(tail, ARCHIVE['bytes'])
    entries = parsed['entries']
    if parsed['cd_size'] != CD_SIZE_EXPECTED:
        raise ValueError('Central directory size changed: %d' % parsed['cd_size'])
    nc_members = [e for e in entries if e['name'].endswith('_hourly.nc')]
    if len(nc_members) != MEMBERS_EXPECTED:
        raise ValueError('Expected %d NetCDF members, found %d' % (MEMBERS_EXPECTED, len(nc_members)))
    for e in nc_members:
        stem = e['name'].split('/')[-1][:-len('_hourly.nc')]
        if not stem.isdigit() or not 8 <= len(stem) <= 16:
            raise ValueError('Unexpected member name: %s' % e['name'])
        if e['method'] != 8:
            raise ValueError('Unexpected compression method %d on %s' % (e['method'], e['name']))
    index = dict(
        archive=ARCHIVE,
        probed_utc=utcnow(),
        head_sha256=hashlib.sha256(head).hexdigest(),
        tail_sha256=hashlib.sha256(tail).hexdigest(),
        tail_bytes=TAIL_BYTES,
        cd_off=parsed['cd_off'],
        cd_size=parsed['cd_size'],
        entries_total=len(entries),
        members=[dict(station=e['name'].split('/')[-1][:-len('_hourly.nc')],
                      local_off=e['local_off'], csize=e['csize'], usize=e['usize'],
                      crc32=e['crc32']) for e in nc_members],
    )
    write_json(OUT / 'archive-index.json', index)
    compressible = sum(e['csize'] for e in nc_members)
    expanded = sum(e['usize'] for e in nc_members)
    print('indexed %d members; compressed %d bytes; uncompressed %d bytes'
          % (len(nc_members), compressible, expanded), flush=True)
    return index


def metadata():
    """Download and verify attributes.7z and info.csv; extract the attributes."""
    BULK.mkdir(parents=True, exist_ok=True)
    retrieved = {}
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for name, spec in META_ARTIFACTS.items():
            path = BULK / name
            if not path.exists() or path.stat().st_size != spec['bytes'] or file_md5(path) != spec['md5']:
                response = client.get(spec['url'])
                response.raise_for_status()
                if len(response.content) != spec['bytes']:
                    raise ValueError('%s downloaded %d bytes, expected %d'
                                     % (name, len(response.content), spec['bytes']))
                path.write_bytes(response.content)
            if file_md5(path) != spec['md5']:
                raise ValueError('%s md5 mismatch: %s' % (name, file_md5(path)))
            retrieved[name] = dict(url=spec['url'], bytes=spec['bytes'], md5=spec['md5'],
                                   verified_utc=utcnow())
    if not ATTRIBUTES_DIR.exists():
        subprocess.run(['7z', 'x', '-y', '-o' + str(BULK / 'attributes'), str(BULK / 'attributes.7z')],
                       check=True, stdout=subprocess.DEVNULL)
    for required in ['attributes_gageii_BasinID.csv', 'attributes_gageii_Regions.csv',
                     'attributes_gageii_Topo.csv']:
        if not (ATTRIBUTES_DIR / required).exists():
            raise FileNotFoundError('Missing extracted attribute file: ' + required)
    write_json(OUT / 'metadata-artifacts.json', dict(retrieved=retrieved, checked_utc=utcnow()))
    print('metadata artifacts verified:', ', '.join(sorted(retrieved)), flush=True)


def attributes_table():
    basin = pd.read_csv(ATTRIBUTES_DIR / 'attributes_gageii_BasinID.csv', dtype={'STAID': str})
    regions = pd.read_csv(ATTRIBUTES_DIR / 'attributes_gageii_Regions.csv', dtype={'STAID': str})
    topo = pd.read_csv(ATTRIBUTES_DIR / 'attributes_gageii_Topo.csv', dtype={'STAID': str})
    frame = basin[['STAID', 'STANAME', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE', 'STATE']].copy()
    frame = frame.merge(regions[['STAID', 'HUC8_SITE']], on='STAID', how='left')
    frame = frame.merge(topo[['STAID', 'ELEV_SITE_M']], on='STAID', how='left')
    frame['station_id'] = frame.STAID.str.strip().astype(int)
    if frame.station_id.duplicated().any():
        raise ValueError('Duplicate station ids in CAMELSH attributes')
    return frame.set_index('station_id')


def exclusion_sets():
    fitted = pd.read_csv(FITTED_SOURCE, usecols=['station'])
    fitted_ids = {int(s) for s in fitted.station.astype(str)}
    reserved_doc = json.loads(RESERVED_SOURCE.read_text())
    reserved_ids = {int(row['station']) for row in reserved_doc['disjoint_selection']}
    provenance = dict(
        fitted=dict(source=str(FITTED_SOURCE.relative_to(ROOT)), count=len(fitted_ids),
                    sha256=digest(FITTED_SOURCE)),
        reserved=dict(source=str(RESERVED_SOURCE.relative_to(ROOT)), count=len(reserved_ids),
                      sha256=digest(RESERVED_SOURCE)),
    )
    return fitted_ids, reserved_ids, provenance


def build_selection():
    index = json.loads((OUT / 'archive-index.json').read_text())
    attributes = attributes_table()
    info = pd.read_csv(BULK / 'info.csv', dtype={'STAID': str})
    info['station_id'] = info.STAID.str.strip().astype(int)
    info = info.set_index('station_id')
    info['recent_hours'] = info[RECENT_YEARS].sum(axis=1)
    fitted_ids, reserved_ids, provenance = exclusion_sets()

    rows, edge = [], dict(
        fitted_in_archive=0, reserved_in_archive=0, fitted_total=len(fitted_ids),
        reserved_total=len(reserved_ids), info_observed_absent=[])
    for member in index['members']:
        station = int(member['station'])
        if station in fitted_ids:
            edge['fitted_in_archive'] += 1
            continue
        if station in reserved_ids:
            edge['reserved_in_archive'] += 1
            continue
        meta = attributes.loc[station]
        info_row = info.loc[station] if station in info.index else None
        total = float(info_row['data_availability [hrs]']) if info_row is not None else 0.0
        recent = float(info_row['recent_hours']) if info_row is not None else 0.0
        tier = 'A' if recent >= TIER_A_HOURS else ('B' if total > 0 else 'U')
        rows.append(dict(
            station=member['station'], name=str(meta.STANAME), state=str(meta.STATE),
            huc02=str(meta.HUC02), huc8=(str(meta.HUC8_SITE) if pd.notna(meta.HUC8_SITE) else None),
            latitude=float(meta.LAT_GAGE), longitude=float(meta.LNG_GAGE),
            area_km2=float(meta.DRAIN_SQKM),
            elevation_m=(float(meta.ELEV_SITE_M) if pd.notna(meta.ELEV_SITE_M) else None),
            tier=tier, info_total_hours=(int(total) if info_row is not None else None),
            info_recent_hours=(int(recent) if info_row is not None else None),
            csize=member['csize'], usize=member['usize'], local_off=member['local_off']))

    frame = pd.DataFrame(rows)
    rank = {'A': 0, 'U': 1, 'B': 2}
    frame['rank'] = frame.tier.map(rank)
    frame['sort_key'] = [hashlib.sha256(('camelsh42:' + s).encode()).hexdigest()
                         for s in frame.station]
    picked = []
    for region in sorted(frame.huc02.unique()):
        group = frame[frame.huc02 == region].sort_values(['rank', 'sort_key'])
        taken = group.head(QUOTA_PER_REGION)
        picked.extend(taken.to_dict('records'))
    selection = dict(
        generated_utc=utcnow(),
        method='Per-HUC02 quota capped at %d gauges; within a region tiers A (info.csv '
               '2019-2024 hours >= %d), U (archive member but info.csv reports zero hours), '
               'B (info.csv hours but fewer recent hours); ties broken by '
               'sha256("camelsh42:" + station).' % (QUOTA_PER_REGION, TIER_A_HOURS),
        exclusions=provenance,
        excluded_from_extraction=dict(fitted_in_archive=edge['fitted_in_archive'],
                                      reserved_in_archive=edge['reserved_in_archive']),
        station_count=len(picked),
        bytes_compressed_planned=sum(r['csize'] for r in picked),
        stations=sorted(picked, key=lambda r: r['station']),
    )
    write_json(OUT / 'selection.json', selection)
    counts = {}
    for row in picked:
        counts[row['huc02']] = counts.get(row['huc02'], 0) + 1
    print('selected %d stations across %d regions; planned compressed bytes %d'
          % (len(picked), len(counts), selection['bytes_compressed_planned']), flush=True)
    return selection


def fetch():
    """Fetch selected members by byte range and verify each member CRC."""
    selection = json.loads((OUT / 'selection.json').read_text())
    index = json.loads((OUT / 'archive-index.json').read_text())
    members = {m['station']: m for m in index['members']}
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    log_path = OUT / 'fetch-log.jsonl'
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row['station']] = row
    lock = threading.Lock()

    def one(entry):
        station = entry['station']
        try:
            member = members[station]
            target = EXTRACTED / (station + '_hourly.nc')
            prior = done.get(station)
            if prior and target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == prior['sha256']:
                return prior
            with httpx.Client(timeout=180, follow_redirects=True) as client:
                head = http_range(client, ARCHIVE['url'], member['local_off'], member['local_off'] + 29)
                fnlen, extralen = struct.unpack_from('<HH', head, 26)
                hsize = 30 + fnlen + extralen
                if hsize > 30:
                    rest = http_range(client, ARCHIVE['url'], member['local_off'] + 30, member['local_off'] + hsize - 1)
                    head = head + rest
                name = head[30:30 + fnlen].decode('utf-8')
                if name != member['station'] + '_hourly.nc' and name != 'Hourly2/' + station + '_hourly.nc':
                    raise ValueError('Local header name mismatch for %s: %s' % (station, name))
                payload = http_range(client, ARCHIVE['url'], member['local_off'] + hsize,
                                     member['local_off'] + hsize + member['csize'] - 1)
            raw = zlib.decompress(payload, -15)
            if len(raw) != member['usize']:
                raise ValueError('%s uncompressed %d bytes, expected %d' % (station, len(raw), member['usize']))
            if binascii.crc32(raw) & 0xFFFFFFFF != member['crc32']:
                raise ValueError('%s CRC mismatch' % station)
            target.write_bytes(raw)
            record = dict(station=station, member=member, fetched_utc=utcnow(),
                          request_bytes=hsize + member['csize'], sha256=hashlib.sha256(raw).hexdigest())
            with lock:
                with log_path.open('a') as handle:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
            print('fetched %s (%d bytes compressed -> %d bytes)' % (station, member['csize'], member['usize']),
                  flush=True)
            return record
        except Exception as error:  # noqa: BLE001 - collected and retried by rerunning fetch
            print('fetch failed for %s: %s' % (station, error), flush=True)
            return dict(station=station, error=str(error))

    entries = selection['stations']
    if args.limit:
        entries = entries[:args.limit]
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entry, outcome in zip(entries, pool.map(one, entries)):
            if isinstance(outcome, dict) and 'error' in outcome and 'member' not in outcome:
                failures.append(outcome)
    total_request = sum(json.loads(l)['request_bytes'] for l in log_path.read_text().splitlines())
    summary = dict(fetched_utc=utcnow(), requested=len(entries), failures=failures,
                   request_bytes_total=total_request,
                   extracted_dir=str(EXTRACTED),
                   extracted_files=sorted(p.name for p in EXTRACTED.glob('*_hourly.nc')))
    write_json(OUT / 'fetch-summary.json', summary)
    if failures:
        raise SystemExit('Fetch failures: %d member(s); rerun fetch to resume' % len(failures))
    print('fetch complete: %d members, %d requested bytes' % (len(entries), total_request), flush=True)


def read_streamflow(path):
    """Read one CAMELSH NetCDF file with strict attribute and axis checks."""
    try:
        import h5py
    except ImportError as error:
        raise SystemExit('h5py is required: .venv/bin/python -m pip install h5py') from error
    with h5py.File(path, 'r') as handle:
        for name in ['time', 'streamflow']:
            if name not in handle:
                raise ValueError('%s lacks %s' % (path.name, name))
        time_units = handle['time'].attrs.get('units')
        flow_units = handle['streamflow'].attrs.get('units')
        if time_units not in (b'hours since 1980-01-01 00:00:00', 'hours since 1980-01-01 00:00:00'):
            raise ValueError('%s unexpected time units %r' % (path.name, time_units))
        if flow_units not in (b'm3 s-1', 'm3 s-1'):
            raise ValueError('%s unexpected streamflow units %r' % (path.name, flow_units))
        hours = np.asarray(handle['time'][:], dtype=np.int64)
        values = np.asarray(handle['streamflow'][:], dtype=np.float64)
    if len(hours) != HOURS or hours[0] != 0 or not (np.diff(hours) == 1).all():
        raise ValueError('%s time axis is not the expected hourly 1980-2024 calendar' % path.name)
    index = pd.date_range('1980-01-01', periods=HOURS, freq='h', tz='UTC')
    # CAMELSH labels each hourly value by the hour start, not the hour end: the
    # time coordinate is an offset from 1980-01-01 00:00 and was built with
    # dateshift(..., 'start', 'hour') in usgsdownload_Hourly.m.
    return pd.Series(values, index=index, name='streamflow_m3s')


def coverage_stats(series):
    valid = series.notna()
    frame = pd.DataFrame({'valid': valid})
    frame['year'] = frame.index.year
    yearly = frame.groupby('year').valid.sum()
    stats = {
        'valid_hours': int(valid.sum()),
        'valid_1980_1999': int(valid[(series.index.year <= 1999)].sum()),
        'valid_2000_2014': int(valid[(series.index.year >= 2000) & (series.index.year <= 2014)].sum()),
        'valid_2015_2020': int(valid[(series.index.year >= 2015) & (series.index.year <= 2020)].sum()),
        'valid_2021_2024': int(valid[(series.index.year >= 2021)].sum()),
        'first_valid': (series.first_valid_index().isoformat() if valid.any() else None),
        'last_valid': (series.last_valid_index().isoformat() if valid.any() else None),
        'yearly_valid_hours': {int(y): int(v) for y, v in yearly.items()},
    }
    window = valid['2021-01-01':'2024-12-31']
    runs, current = 0, 0
    for value in window:
        current = current + 1 if value else 0
        runs = max(runs, current)
    stats['longest_run_2021_2024'] = int(runs)
    return stats


def reference_from_hourly(q_cfs):
    """95th percentile of daily-mean discharge over 2000-2015, mirroring freeze().

    The frozen shift protocol used USGS daily-mean values (statistic 00003).
    CAMELSH ships hourly values only, so the daily mean is reconstructed from
    the hourly series; USGS computes daily means from the same sub-hourly
    readings, so the two agree except where the reading cadence varies.
    """
    daily = q_cfs['2000-01-01':'2015-12-31'].resample('D').mean()
    daily = daily[daily.notna()]
    if len(daily) < REFERENCE_MIN_DAYS or daily.quantile(.95) <= 0:
        raise ValueError('insufficient 2000-2015 daily coverage for a reference')
    return float(daily.quantile(.95)), int(len(daily)), daily.index[0], daily.index[-1]


def verify():
    """Coverage for every fetched gauge; decision-row build on a sample."""
    selection = json.loads((OUT / 'selection.json').read_text())
    by_station = {row['station']: row for row in selection['stations']}
    coverage, errors = [], []
    for path in sorted(EXTRACTED.glob('*_hourly.nc')):
        station = path.name[:-len('_hourly.nc')]
        row = by_station.get(station)
        if row is None:
            errors.append(dict(station=station, error='fetched but not in selection'))
            continue
        series = read_streamflow(path)
        stats = coverage_stats(series)
        coverage.append(dict(station=station, huc02=row['huc02'], tier=row['tier'],
                             csize=row['csize'], **stats))
    coverage.sort(key=lambda r: r['station'])
    pd.DataFrame(coverage).to_csv(OUT / 'coverage.csv', index=False)
    summary = dict(
        completed_utc=utcnow(), gauges=len(coverage), read_errors=errors,
        with_2021_2024=sum(1 for r in coverage if r['valid_2021_2024'] > 0),
        at_least_half_year_2021_2024=sum(1 for r in coverage if r['valid_2021_2024'] >= 4_380),
        at_least_full_year_2021_2024=sum(1 for r in coverage if r['valid_2021_2024'] >= 8_766),
        valid_hours_total=sum(r['valid_hours'] for r in coverage),
        compressed_bytes=sum(r['csize'] for r in coverage),
        note='valid hours are hours with a non-NaN streamflow value; NaN hours are '
             'missing observations, not gaps that were filled.',
    )
    write_json(OUT / 'coverage-summary.json', summary)
    print('coverage written for %d gauges; %d with >= half year in 2021-2024'
          % (summary['gauges'], summary['at_least_half_year_2021_2024']), flush=True)

    pool = [r for r in coverage if r['valid_2021_2024'] >= DECISION_MIN_HOURS]
    pool.sort(key=lambda r: hashlib.sha256(('decision42:' + r['station']).encode()).hexdigest())
    chosen = pool[:DECISION_LIMIT]
    results, sample_frames = [], []
    for row in chosen:
        station = row['station']
        path = EXTRACTED / (station + '_hourly.nc')
        series = read_streamflow(path)
        q_cfs = series * CFS_PER_CMS
        entry = dict(station=station, huc02=row['huc02'], tier=row['tier'])
        try:
            reference, days, daily_first, daily_last = reference_from_hourly(q_cfs)
        except ValueError as error:
            entry['excluded'] = str(error)
            results.append(entry)
            continue
        station_record = dict(station=station, huc=row['huc02'], role='camelsh_research',
                              latitude=by_station[station]['latitude'],
                              longitude=by_station[station]['longitude'],
                              elevation_m=by_station[station]['elevation_m'],
                              area_km2=by_station[station]['area_km2'])
        frame, features, diagnostics = decision_rows(q_cfs, reference, station_record)
        horizon_counts = frame.groupby('horizon').target.sum().astype(int).to_dict()
        row_counts = frame.groupby('horizon').size().to_dict()
        reasons = {}
        for event in diagnostics['event_eligibility']:
            for horizon, reason in event['reasons'].items():
                key = horizon + ':' + reason
                reasons[key] = reasons.get(key, 0) + 1
        events_by_year = {}
        for event in diagnostics['events']:
            year = int(event[:4])
            events_by_year[year] = events_by_year.get(year, 0) + 1
        entry.update(dict(
            reference_cfs=round(reference, 6), reference_days=days,
            reference_window=[str(daily_first.date()), str(daily_last.date())],
            events=len(diagnostics['events']),
            events_first5=diagnostics['events'][:5],
            events_by_year=events_by_year,
            decision_rows={int(k): int(v) for k, v in row_counts.items()},
            positives={int(k): int(v) for k, v in horizon_counts.items()},
            event_eligibility_counts=reasons,
        ))
        results.append(entry)
        if len(sample_frames) < 3:
            trimmed = frame[['station', 'decision_time', 'horizon', 'target', 'event_time',
                             'reference_cfs', 'future_max_log_flow']].copy()
            sample_frames.append(trimmed)
    if sample_frames:
        pd.concat(sample_frames).to_csv(OUT / 'onset-sample-rows.csv.gz', index=False,
                                        compression={'method': 'gzip', 'mtime': 0})
    report = dict(
        completed_utc=utcnow(), sample_size=len(chosen), decision_min_hours=DECISION_MIN_HOURS,
        stations=results,
        totals=dict(events=sum(r.get('events', 0) for r in results),
                    positives=sum(sum(r.get('positives', {}).values()) for r in results),
                    with_events=sum(1 for r in results if r.get('events')),
                    excluded=sum(1 for r in results if r.get('excluded'))),
        note='Built with scripts.shift_hydrology.decision_rows; reference is the 95th percentile '
             'of 2000-2015 daily-mean discharge, matching the frozen shift protocol.',
    )
    write_json(OUT / 'onset-check.json', report)
    print('decision rows built for %d gauges; %d events total'
          % (report['sample_size'] - report['totals']['excluded'], report['totals']['events']), flush=True)


def crosscheck():
    """Compare CAMELSH hourly values with live USGS instantaneous readings."""
    from dotenv import dotenv_values
    coverage = pd.read_csv(OUT / 'coverage.csv', dtype={'station': str})
    coverage = coverage[coverage.valid_2021_2024 >= 8_000]
    if coverage.empty:
        raise SystemExit('No fetched gauge has a full year in 2021-2024 for a crosscheck')
    coverage = coverage.sort_values('valid_2021_2024', ascending=False).head(2)
    api = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items'
    key = (dotenv_values(ROOT / '.env') or {}).get('USGS_API_KEY')
    headers = {'X-Api-Key': key} if key else {}
    results = []
    for station in coverage.station:
        series = read_streamflow(EXTRACTED / (station + '_hourly.nc'))
        window = ('2024-05-01T00:00:00Z', '2024-06-30T23:59:59Z')
        params = dict(f='json', monitoring_location_id='USGS-' + station, parameter_code='00060',
                      datetime=window[0] + '/' + window[1], limit=50_000, skipGeometry='true')
        readings, approvals, qualifiers = [], {}, {}
        with httpx.Client(timeout=120, follow_redirects=True, headers=headers) as client:
            response = client.get(api, params=params)
            for page in range(10):
                response.raise_for_status()
                payload = response.json()
                for feature in payload['features']:
                    props = feature['properties']
                    if props['monitoring_location_id'] != 'USGS-' + station or props['parameter_code'] != '00060':
                        raise ValueError('Unexpected USGS station or parameter')
                    stamp = pd.Timestamp(props['time'])
                    if stamp.tzinfo is None:
                        raise ValueError('USGS reading without UTC offset')
                    approvals[props.get('approval_status')] = approvals.get(props.get('approval_status'), 0) + 1
                    qualifier = props.get('qualifier') or 'none'
                    qualifiers[qualifier] = qualifiers.get(qualifier, 0) + 1
                    try:
                        readings.append((stamp, float(props['value'])))
                    except (TypeError, ValueError, KeyError):
                        continue
                links = [x['href'] for x in payload.get('links', []) if x['rel'] == 'next']
                if not links:
                    break
                response = client.get(links[0])
            else:
                raise ValueError('USGS pagination exceeded the bounded budget')
        usgs = pd.Series(dict(readings)).sort_index() * CMS_PER_CFS
        hourly_floor = usgs.resample('h', closed='left', label='left').mean()
        hourly_last = usgs.resample('h', closed='right', label='right').last()
        camelsh = series['2024-05-01':'2024-06-30']
        align = pd.DataFrame({'camelsh': camelsh, 'usgs_floor_mean': hourly_floor,
                              'usgs_right_last': hourly_last}).dropna()
        only_usgs = hourly_floor.notna() & camelsh.reindex(hourly_floor.index).isna()
        only_camelsh = hourly_floor.isna() & camelsh.reindex(hourly_floor.index).notna()
        ratio = align.camelsh / align.usgs_floor_mean
        results.append(dict(
            station=station, window=window, readings=len(readings),
            approvals=approvals, qualifiers=qualifiers,
            overlap_hours=len(align),
            correlation=float(align.camelsh.corr(align.usgs_floor_mean)),
            median_ratio=float(ratio.median()), ratio_p10=float(ratio.quantile(.10)),
            ratio_p90=float(ratio.quantile(.90)),
            hours_only_usgs=int(only_usgs.sum()), hours_only_camelsh=int(only_camelsh.sum()),
            max_abs_diff=float((align.camelsh - align.usgs_floor_mean).abs().max()),
            labelling='CAMELSH hour-start means correspond to USGS readings within [t, t+1h) '\
                      'labelled t, per usgsdownload_Hourly.m (dateshift start, accumarray mean).',
        ))
        print('crosschecked %s: n=%d corr=%.4f median ratio=%.4f'
              % (station, len(align), results[-1]['correlation'], results[-1]['median_ratio']),
              flush=True)
    report = dict(completed_utc=utcnow(), source='USGS OGC continuous items (parameter 00060)',
                  unit_conversion='USGS cfs x 0.0283168 = CAMELSH m3 s-1', stations=results)
    write_json(OUT / 'usgs-crosscheck.json', report)


def manifest():
    """Assemble the acquisition manifest from the step outputs."""
    def padded(station):
        text = str(station)
        return text.zfill(8) if len(text) <= 8 else text

    needed = ['archive-index.json', 'metadata-artifacts.json', 'selection.json',
              'fetch-summary.json', 'coverage-summary.json']
    for name in needed:
        if not (OUT / name).exists():
            raise FileNotFoundError('Run the earlier steps; missing ' + name)
    index = json.loads((OUT / 'archive-index.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    fetch_summary = json.loads((OUT / 'fetch-summary.json').read_text())
    coverage = json.loads((OUT / 'coverage-summary.json').read_text())
    fitted_ids, reserved_ids, provenance = exclusion_sets()
    members = {int(m['station']): m for m in index['members']}
    info = pd.read_csv(BULK / 'info.csv', dtype={'STAID': str})
    info['station_id'] = info.STAID.str.strip().astype(int)
    observed = set(info[info['data_availability [hrs]'] > 0].station_id)
    overlap = dict(
        fitted_in_archive=len(fitted_ids & set(members)),
        fitted_total=len(fitted_ids),
        reserved_in_archive=len(reserved_ids & set(members)),
        reserved_total=len(reserved_ids),
        info_observed_total=len(observed),
        info_observed_in_archive=len(observed & set(members)),
        info_observed_not_in_archive=[padded(s) for s in sorted(observed - set(members))[:10]],
        info_observed_not_in_archive_count=len(observed - set(members)),
    )
    not_acquired = [
        dict(file='timeseries.7z', record='15066778', bytes=21_247_136_353,
             reason='Original-release NLDAS-2 forcing plus streamflow bundle; superseded for '
                    'streamflow by Hourly2.zip and not needed for onset compatibility.'),
        dict(file='timeseries_nonobs.7z', record='15070091', bytes=35_655_472_043,
             reason='Gauges without observed streamflow; outside the training population.'),
        dict(file='shapefiles.7z', record='15066778', bytes=530_604_106,
             reason='Catchment polygons for all 9,008 basins, including the reserved cohort; '
                    'deferred until a candidate cohort is chosen.'),
        dict(file='Hourly.7z', record='15413207', bytes=None,
             reason='5,188-gauge streamflow and water level archive; record is access '
                    'restricted and was superseded by the open Hourly2.zip.'),
        dict(file='ERA5-Land annual 7z archives', record='15264814', bytes=None,
             reason='Hourly ERA5-Land forcing for 9,008 basins, about 2.4 GB per year; this '
                    'task needs streamflow semantics, and the project already provides its '
                    'own weather pipeline.'),
        dict(file='NWM.7z / NWM_Q_nc.zip / CAMELSH_FCS.zip / StreamTemperature.zip',
             record='16763144, 21302482, 21324088', bytes=None,
             reason='Benchmark simulations, actual-forecast archives and stream temperature; '
                    'not streamflow observations.'),
    ]
    manifest_doc = dict(
        generated_utc=utcnow(),
        dataset=dict(
            name='CAMELSH: A Large-Sample Hourly Hydrometeorological Dataset and Attributes at '
                 'Watershed-Scale for CONUS',
            citation='Tran et al., Scientific Data (2025), doi:10.1038/s41597-025-05612-6',
            licence='CC-BY-4.0 (Zenodo records 15066778, 16729675, 15264814, 21302482)',
            gauges_hourly_streamflow=5_767,
            gauges_with_attributes=9_008,
            coverage='1980-01-01 00:00:00 to 2024-12-31 23:00:00 UTC, hourly',
            streamflow_units='m3 s-1 (converted from USGS cfs with the 0.0283168 constant in '
                             'datatransform.m)',
            streamflow_statistic='hourly mean of USGS Instantaneous Values Service readings '
                                 '(parameter 00060, typically 15 to 60 minute cadence), '
                                 'labelled by hour start',
            missing_values='IEEE NaN, no interpolation; all USGS flag classes are retained',
            basin_polygons='Provided as CAMELSH_shapefile.shp (GAGES-II derived) and '
                           'CAMELSH_shapefile_hydroATLAS.shp, field GAGE_ID, in shapefiles.7z',
            archive_format='Hourly2.zip: one deflate-compressed NetCDF4/HDF5 file per gauge, '
                           'named <STAID>_hourly.nc with variables time, streamflow, water_level',
            source_urls=dict(
                streamflow_archive=ARCHIVE['url'], attributes=META_ARTIFACTS['attributes.7z']['url'],
                info_csv=META_ARTIFACTS['info.csv']['url'],
                repository='https://github.com/vinhngoctran/CAMELSH',
                alternate_drive='https://drive.google.com/drive/folders/15dk6qlU38LqsUkf9hiIZHXGtzzrDfl-q'),
        ),
        acquired=dict(
            archive_bytes=index['archive']['bytes'], archive_md5=index['archive']['md5'],
            head_bytes=4096, tail_bytes=index['tail_bytes'],
            central_directory=dict(offset=index['cd_off'], bytes=index['cd_size']),
            members_indexed=len(index['members']),
            selected=selection['station_count'],
            fetched=fetch_summary['requested'], fetch_failures=fetch_summary['failures'],
            request_bytes=fetch_summary['request_bytes_total'],
            extracted_files=len(fetch_summary['extracted_files']),
            extracted_dir=fetch_summary['extracted_dir'],
            member_manifest='selection.json plus fetch-log.jsonl (per-station csize, usize, '
                            'crc32, sha256, request bytes)',
        ),
        coverage=dict(summary=coverage, csv='coverage.csv'),
        overlap=overlap,
        exclusions=provenance,
        not_acquired=not_acquired,
        verification=dict(
            onset_check='onset-check.json (decision_rows sample)',
            usgs_crosscheck='usgs-crosscheck.json' if (OUT / 'usgs-crosscheck.json').exists() else None,
        ),
    )
    write_json(OUT / 'acquisition-manifest.json', manifest_doc)
    print('manifest written to', OUT / 'acquisition-manifest.json', flush=True)


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['probe', 'metadata', 'select', 'fetch', 'verify',
                                            'crosscheck', 'manifest', 'all'])
    parser.add_argument('--limit', type=int, default=None,
                        help='debug aid: fetch only the first N selected members')
    args = parser.parse_args()
    if args.command == 'probe':
        probe()
    elif args.command == 'metadata':
        metadata()
    elif args.command == 'select':
        build_selection()
    elif args.command == 'fetch':
        fetch()
    elif args.command == 'verify':
        verify()
    elif args.command == 'crosscheck':
        crosscheck()
    elif args.command == 'manifest':
        manifest()
    else:
        probe()
        metadata()
        build_selection()
        fetch()
        verify()
        crosscheck()
        manifest()


if __name__ == '__main__':
    main()
