"""Bounded, isolated subdaily USGS research. Production never imports this module."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import gzip
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, daily_discharge, digest, IneligibleGauge
from scripts.flood_sprint import write_json

OUT = ROOT / 'experiments/sprint-20261017-shift-hydrology'
OLD = ROOT / 'experiments/sprint-20261017-hydrology'
API = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/'
STATIC = ['season_sin', 'season_cos', 'latitude', 'longitude', 'log_area', 'elevation_m']
LAGS = [2, 8, 14, 26, 50, 74, 170]
FEATURES = [f'q_{lag}' for lag in LAGS] + [f'rise_{lag}' for lag in [6, 12, 24, 48, 72, 168]] + ['acceleration', 'range_24', 'range_72'] + STATIC


def parse_observations(features, station, statistic='00011'):
    """Reject ambiguity; quality exclusions leave gaps, never interpolated water."""
    rows, series, times = {}, set(), set()
    for feature in features:
        p = feature['properties']
        if p['monitoring_location_id'] != 'USGS-' + station or p['parameter_code'] != '00060':
            raise ValueError('Unexpected station or parameter')
        if p['statistic_id'] != statistic or p['unit_of_measure'] != 'ft^3/s':
            raise ValueError('Unexpected statistic or unit')
        series.add(p['time_series_id'])
        if len(series) > 1:
            raise ValueError('Ambiguous discharge series')
        t = pd.Timestamp(p['time'])
        if statistic == '00011' and t.tzinfo is None:
            raise ValueError('Observation timestamp lacks UTC offset')
        if t in times:
            raise ValueError('Duplicate observation timestamp')
        times.add(t)
        try:
            value = float(p['value'])
        except (TypeError, ValueError):
            continue
        if p['approval_status'] == 'Approved' and not p.get('qualifier') and np.isfinite(value) and value >= 0:
            rows[t] = value
    if not rows:
        return pd.Series(index=pd.DatetimeIndex([], tz='UTC' if statistic=='00011' else None), dtype=float)
    return pd.Series(rows, dtype=float).sort_index()


def hourly_flow(series):
    if series.empty:
        return series
    # Right-closed bins use only readings at or before their labelled cutoff.
    return series.resample('h', closed='right', label='right').last()


def decision_rows(q, reference, station):
    if reference <= 0 or q.empty or not q.index.equals(pd.date_range(q.index[0], q.index[-1], freq='h')):
        raise ValueError('Require positive reference and a complete hourly calendar')
    below = q.le(reference).where(q.notna()).astype(float)
    onset = q.gt(reference) & below.shift(1).rolling(24).sum().eq(24)
    events = [t.isoformat() for t in q.index[onset]]
    x = np.log1p(q / reference)
    frame = pd.DataFrame({f'q_{lag}': x.shift(lag) for lag in LAGS})
    for lag in [6, 12, 24, 48, 72, 168]:
        frame[f'rise_{lag}'] = x.shift(2) - x.shift(lag + 2)
    frame['acceleration'] = frame.rise_6 - (x.shift(8) - x.shift(14))
    for hours in [24, 72]:
        frame[f'range_{hours}'] = (x.rolling(hours).max() - x.rolling(hours).min()).shift(2)
    frame['season_sin'] = np.sin(2*np.pi*q.index.dayofyear/365.25)
    frame['season_cos'] = np.cos(2*np.pi*q.index.dayofyear/365.25)
    for name in ['station', 'huc', 'role', 'latitude', 'longitude', 'elevation_m']:
        frame[name] = station[name]
    frame['huc'] = station['huc'][:8]
    frame['log_area'] = np.log1p(station['area_km2'])
    # Eligibility is observable through t-2h, not through the prediction issue time.
    quiet = below.rolling(24).sum().shift(2).eq(24)
    history = q.notna().rolling(169).sum().shift(2).eq(169)
    complete_future = q.notna().shift(-1)[::-1].rolling(24).sum()[::-1].eq(24)
    certified_buffer = q.notna().rolling(2).sum().eq(2)
    scheduled = pd.Series(q.index.hour % 6 == 0, index=q.index)
    # Buffer readings certify outcomes only; they never determine observable quiet eligibility.
    eligible = scheduled & quiet & history & complete_future & certified_buffer
    result = []
    for horizon in [6, 12, 24]:
        f = frame.loc[eligible].copy()
        future = pd.Series('', index=q.index, dtype=object)
        for lead in range(horizon, 0, -1):
            occurs = onset.shift(-lead, fill_value=False)
            future.loc[occurs] = (q.index[occurs] + pd.Timedelta(hours=lead)).map(lambda t: t.isoformat())
        f['target'] = future.loc[f.index].ne('').astype(int)
        f['event_time'] = future.loc[f.index]
        f['decision_time'] = f.index.map(lambda t: t.isoformat())
        f['horizon'] = horizon
        f['reference_cfs'] = reference
        f['future_max_log_flow'] = x.shift(-1)[::-1].rolling(horizon).max()[::-1].loc[f.index]
        result.append(f)
    safe_year = pd.Series((q.index+pd.Timedelta(hours=24)).year == q.index.year,index=q.index)
    losses = []
    for event in q.index[onset]:
        reasons = {}
        for horizon in [6,12,24]:
            possible = scheduled & (q.index < event) & (q.index >= event-pd.Timedelta(hours=horizon))
            reason = 'no_prior_decision'
            for name,mask in [('missing_history',history),('quiet_interval',quiet),
                              ('missing_buffer',certified_buffer),('missing_future',complete_future),
                              ('year_boundary',safe_year)]:
                if not possible.any():
                    break
                reason = name
                possible &= mask
            if possible.any():
                reason = 'eligible'
            reasons[str(horizon)] = reason
        losses.append(dict(event_time=event.isoformat(),reasons=reasons))
    return pd.concat(result, ignore_index=True), FEATURES, dict(events=events,event_eligibility=losses)


def cached_page(path, url):
    """Persist compressed response bytes and exact request provenance."""
    import httpx
    record = path.with_suffix(path.suffix + '.request.json')
    if path.exists():
        meta = json.loads(record.read_text())
        if meta['url'] != url or digest(path) != meta['sha256']:
            raise ValueError('Cache checksum or URL changed')
        return json.loads(gzip.decompress(path.read_bytes()))
    waits = []
    response = httpx.get(url, timeout=90, follow_redirects=True)
    for attempt in range(3):
        if response.status_code != 429:
            break
        delay = response.headers.get('Retry-After','')
        if not delay.isdigit() or not 0 < int(delay) <= 3600:
            raise RuntimeError('USGS rate limit requires an explicit later rerun: '+delay)
        waits.append(int(delay))
        print('USGS rate limit; honoring Retry-After', delay, 'seconds',flush=True)
        remaining = int(delay)
        while remaining:
            pause = min(remaining,60)
            time.sleep(pause)
            remaining -= pause
        response = httpx.get(url, timeout=90, follow_redirects=True)
    response.raise_for_status()
    if not response.content or len(response.content) > 40_000_000:
        raise ValueError('Empty or oversized USGS response')
    payload = response.json()
    if payload.get('type') != 'FeatureCollection':
        raise ValueError('Expected USGS FeatureCollection')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(response.content, mtime=0))
    write_json(record, dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
        status=response.status_code, rate_limit_waits_seconds=waits,
        uncompressed_bytes=len(response.content), sha256=digest(path)))
    return payload


def observations(station, year=None):
    collection = 'continuous' if year else 'daily'
    intervals = [(f'{year}-{month:02d}-01T00:00:00Z', f'{year+int(month==10)}-{(month+3-1)%12+1:02d}-01T00:00:00Z')
                 for month in [1, 4, 7, 10]] if year else [('2000-01-01', '2015-12-31')]
    pieces = []
    for quarter, (start, end) in enumerate(intervals):
        stop = end
        if year and quarter == 3:
            end = (pd.Timestamp(end)-pd.Timedelta(seconds=1)).isoformat()
        url = API + collection + '/items?' + urlencode(dict(f='json', monitoring_location_id='USGS-'+station,
            parameter_code='00060', datetime=start+'/'+end, limit=50000, skipGeometry='true'))
        features, page = [], 0
        while url:
            if page >= 10:
                raise ValueError('USGS pagination exceeds bounded request budget')
            suffix = '-bounded' if year and quarter == 3 else ''
            path = OUT / 'sources' / station / f'{year or "reference"}-{quarter}-{page}{suffix}.json.gz'
            payload = cached_page(path, url)
            features.extend(payload['features'])
            links = [x['href'] for x in payload['links'] if x['rel'] == 'next']
            if len(links) > 1 or (links and not links[0].startswith(API)):
                raise ValueError('Unexpected pagination link')
            url = links[0] if links else None
            page += 1
        # API intervals include both endpoints. Remove the end here, not by silently deduplicating.
        series = parse_observations(features, station, '00011' if year else '00003')
        if year:
            series = series.loc[series.index < pd.Timestamp(stop)]
        pieces.append(series)
    result = pd.concat(pieces).sort_index()
    if result.index.has_duplicates:
        raise ValueError('Overlapping USGS chunks')
    return result


def freeze():
    if (OUT / 'protocol.json').exists() and any((OUT/name).exists() for name in ['sources','development.csv.gz','selection.json']):
        raise FileExistsError('Protocol already frozen')
    old = json.loads((OLD / 'stations.json').read_text())
    references, eligible = {}, []
    for s in old:
        try:
            q = daily_discharge((OLD / 'sources' / (s['station']+'.rdb')).read_text(), s['station'])
        except (IneligibleGauge, FileNotFoundError):
            continue
        q = q.loc[(q.index >= '2000-01-01') & (q.index < '2016-01-01')]
        if len(q) >= 3000 and q.quantile(.95) > 0:
            references[s['station']] = dict(reference_cfs=float(q.quantile(.95)), days=len(q),
                source=str((OLD / 'sources' / (s['station']+'.rdb')).relative_to(ROOT)),
                sha256=digest(OLD / 'sources' / (s['station']+'.rdb')))
            eligible.append(s)
    chosen = []
    for region in sorted({s['huc'][:2] for s in eligible} - {'17', '18'}):
        group = sorted([s for s in eligible if s['huc'].startswith(region)],
                       key=lambda s: hashlib.sha256(('shift42'+s['station']).encode()).hexdigest())
        count = 2 if region in ['01', '02', '03', '05', '07', '12'] else 1
        chosen.extend(group[:count])
    # New stations in HUC2s disjoint from this fit; these regions were studied before.
    exposed, exposed_stations = set(), set()
    for folder in ROOT.glob('experiments/sprint-20261017-*'):
        p = folder / 'stations.json'
        if p.exists() and folder != OUT:
            records = json.loads(p.read_text())
            if isinstance(records, list):
                exposed.update(s['huc'][:4] for s in records if isinstance(s, dict) and 'huc' in s)
                exposed_stations.update(s['station'] for s in records if isinstance(s, dict) and 'station' in s)
    sites = json.loads((OLD / 'research/sites.json').read_text())
    reserved = []
    for region in ['17', '18']:
        options = [s for s in sites if s['huc_cd'].startswith(region) and s['site_no'] not in exposed_stations
                   and s.get('drain_area_va') and 20 < float(s['drain_area_va']) < 1200 and s.get('alt_va')]
        options.sort(key=lambda s: hashlib.sha256(('shift42'+s['site_no']).encode()).hexdigest())
        hucs = set()
        for s in options:
            if s['huc_cd'][:4] in hucs:
                continue
            hucs.add(s['huc_cd'][:4])
            reserved.append(dict(station=s['site_no'], name=s['station_nm'], huc=s['huc_cd'],
                role='geographic_test', latitude=float(s['dec_lat_va']), longitude=float(s['dec_long_va']),
                area_km2=float(s['drain_area_va'])*2.589988110336, elevation_m=float(s['alt_va'])*.3048))
            if len(hucs) == 3:
                break
    chosen.extend(reserved)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'stations.json', chosen)
    write_json(OUT / 'references.json', references)
    doc = ROOT / 'docs/SHIFT_HYDROLOGY_PROTOCOL.md'
    write_json(OUT / 'protocol.json', dict(document=str(doc.relative_to(ROOT)), document_sha256=digest(doc),
        stations_sha256=digest(OUT/'stations.json'), references_sha256=digest(OUT/'references.json'),
        preserved={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'models').glob('*') if p.is_file()},
        reserved_huc4s=sorted({s['huc'][:4] for s in reserved}), historical_exposed_huc4s=sorted(exposed),
        historical_exposed_stations=sorted(exposed_stations)))
    print('Frozen', len(chosen), 'gauges;', len(reserved), 'new geographic gauges', flush=True)


def protocol():
    p = json.loads((OUT/'protocol.json').read_text())
    for path, checksum in [(ROOT/p['document'], p['document_sha256']), (OUT/'stations.json',p['stations_sha256']),
                           (OUT/'references.json',p['references_sha256'])]:
        if digest(path) != checksum:
            raise ValueError('Frozen protocol changed: '+str(path))
    for path, checksum in p['preserved'].items():
        if digest(ROOT/path) != checksum:
            raise ValueError('Production model changed')
    return p


def collect(stage):
    protocol()
    if stage == 'test':
        verify_selection()
    stations = json.loads((OUT/'stations.json').read_text())
    references = json.loads((OUT/'references.json').read_text())
    destination = OUT / (stage+'.csv.gz')
    if destination.exists():
        raise FileExistsError('Dataset already materialized')
    def one(s):
        sid = s['station']
        try:
            if sid in references:
                reference = references[sid]['reference_cfs']
            else:
                historical = observations(sid)
                if len(historical) < 3000 or historical.quantile(.95) <= 0:
                    raise IneligibleGauge('Insufficient pre-2016 reference')
                reference = float(historical.quantile(.95))
            years = [2025] if stage == 'test' else ([2023, 2024] if s['role']=='geographic_validation' else [2021, 2022, 2023, 2024])
            if stage == 'development' and s['role']=='geographic_test':
                # Certify reference availability now without opening final observations.
                return None, dict(station=sid, role=s['role'], reference_cfs=reference, reserved=True)
            pieces = [observations(sid, year) for year in years]
            q = hourly_flow(pd.concat(pieces).sort_index())
            if q.empty:
                raise IneligibleGauge('No accepted instantaneous discharge')
            q = q.reindex(pd.date_range(f'{years[0]}-01-01', f'{years[-1]+1}-01-01', freq='h', tz='UTC', inclusive='left'))
            rows, _, diagnostics = decision_rows(q, reference, s)
            coverage = dict(station=sid, huc=s['huc'], role=s['role'], reference_cfs=reference,
                years=years, accepted_hours=int(q.notna().sum()), total_hours=len(q),
                decisions=int(len(rows)//3), **diagnostics, positives=rows.groupby('horizon').target.sum().to_dict())
            print(sid, s['role'], 'hours', coverage['accepted_hours'], 'events',len(diagnostics['events']), flush=True)
            return rows, coverage
        except (IneligibleGauge, ValueError) as e:
            print(sid, 'EXCLUDED', str(e), flush=True)
            return None, dict(station=sid, role=s['role'], excluded=str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, stations))
    frames = [f for f,c in records if f is not None and len(f)]
    if not frames:
        raise ValueError('No usable research data')
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(destination, index=False, compression={'method':'gzip','mtime':0})
    write_json(OUT/(stage+'-coverage.json'), [c for f,c in records])
    write_json(OUT/(stage+'-provenance.json'), dict(dataset_sha256=digest(destination), rows=len(frame),
        source_requests={str(p.relative_to(OUT)):json.loads(p.read_text())['sha256'] for p in (OUT/'sources').rglob('*.request.json')},
        observation_availability='Retrospective approved USGS discharge; assumed two-hour receipt delay, not an as-issued replay'))


def implementation_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/shift_hydrology.py',
        ROOT/'scripts/shift_hydrology_worker.py', ROOT/'scripts/shift_hydrology_metrics.py']}


def verify_selection():
    selection = json.loads((OUT/'selection.json').read_text())
    if selection['protocol_sha256'] != digest(OUT/'protocol.json') or selection['dataset_sha256'] != digest(OUT/'development.csv.gz'):
        raise ValueError('Frozen development inputs changed')
    if selection['code_sha256'] != implementation_hashes():
        raise ValueError('Candidate implementation changed after selection')
    for r in selection['candidates']:
        if digest(OUT/'artifacts'/(r['name']+'.joblib')) != r['artifact_sha256']:
            raise ValueError('Frozen candidate artifact changed')
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','development','test'])
    args = parser.parse_args()
    freeze() if args.command == 'freeze' else collect(args.command)
