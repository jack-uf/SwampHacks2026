"""Isolated hybrid MLP onset research on a second fresh cohort. Production never imports this module."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import time
from urllib.parse import urlencode

import pandas as pd

from scripts.hydrology_research import IneligibleGauge, ROOT, digest
from scripts.mlp_onset import exposed_stations
from scripts.shift_hydrology import API, decision_rows, hourly_flow, parse_observations
from scripts.flood_sprint import write_json

OUT = ROOT / 'experiments/sprint-20261017-mlp-hybrid'
REGISTRY = ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json'
DEVELOPMENT_YEARS = [2020, 2021, 2022, 2023, 2024]
GEOGRAPHIC_YEARS = [2023, 2024]
TEST_YEAR = 2025


def freeze():
    if (OUT / 'protocol.json').exists() and any((OUT / name).exists() for name in ['sources', 'development.csv.gz']):
        raise FileExistsError('Protocol already frozen')
    registry = json.loads(REGISTRY.read_text())
    used = exposed_stations()
    eligible = []
    for site in registry:
        station, huc = site['site_no'], (site.get('huc_cd') or '').strip()
        if station in used or len(huc) < 8:
            continue
        try:
            area, elevation = float(site['drain_area_va']), float(site['alt_va'])
            latitude, longitude = float(site['dec_lat_va']), float(site['dec_long_va'])
        except (KeyError, ValueError, TypeError):
            continue
        if not 20 < area < 1200:
            continue
        eligible.append(dict(station=station, name=site['station_nm'], huc=huc,
                             latitude=latitude, longitude=longitude,
                             area_km2=area * 2.589988110336, elevation_m=elevation * .3048))
    by_region = {}
    for site in eligible:
        by_region.setdefault(site['huc'][:2], []).append(site)
    cohort = []
    for region in sorted(by_region):
        groups, seen = [], set()
        for site in sorted(by_region[region], key=lambda s: hashlib.sha256(('hyb29' + s['station']).encode()).hexdigest()):
            if site['huc'][:4] in seen:
                continue
            seen.add(site['huc'][:4])
            groups.append(site)
            if len(groups) == 2:
                break
        if len(groups) < 2:
            continue
        cohort.append(dict(groups[0], role='development'))
        cohort.append(dict(groups[1], role='geographic_test'))
    if len({s['huc'][:4] for s in cohort}) != len(cohort):
        raise ValueError('Cohort HUC4 groups must be disjoint')
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'stations.json', cohort)
    document = ROOT / 'docs/MLP_HYBRID_PROTOCOL.md'
    write_json(OUT / 'protocol.json', dict(document=str(document.relative_to(ROOT)),
        document_sha256=digest(document), stations_sha256=digest(OUT / 'stations.json'),
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'models').glob('*') if p.is_file()},
        excluded_prior_stations=len(used), development_years=DEVELOPMENT_YEARS,
        geographic_years=GEOGRAPHIC_YEARS, test_year=TEST_YEAR))
    print('Frozen', len(cohort), 'gauges;', len({s['huc'][:2] for s in cohort}), 'regions', flush=True)


def protocol():
    frozen = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in [(ROOT / frozen['document'], frozen['document_sha256']),
                           (OUT / 'stations.json', frozen['stations_sha256'])]:
        if digest(path) != checksum:
            raise ValueError('Frozen protocol changed: ' + str(path))
    for path, checksum in frozen['preserved'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Production model changed')
    return frozen


def cached_page(path, url, client=None):
    """Persist compressed response bytes and exact request provenance."""
    import httpx
    from dotenv import dotenv_values
    record = path.with_suffix(path.suffix + '.request.json')
    if path.exists():
        meta = json.loads(record.read_text())
        if meta['url'] != url or digest(path) != meta['sha256']:
            raise ValueError('Cache checksum or URL changed')
        return json.loads(gzip.decompress(path.read_bytes()))
    waits = []
    key = os.environ.get('USGS_API_KEY') or dotenv_values(ROOT / '.env').get('USGS_API_KEY')
    headers = {'X-Api-Key': key} if key and url.startswith(API) else {}
    response = (client or httpx).get(url, headers=headers, timeout=90, follow_redirects=True)
    for _ in range(3):
        if response.status_code != 429:
            break
        delay = response.headers.get('Retry-After', '')
        if not delay.isdigit() or not 0 < int(delay) <= 3600:
            raise RuntimeError('USGS rate limit requires an explicit later rerun: ' + delay)
        waits.append(int(delay))
        print('USGS rate limit; honoring Retry-After', delay, 'seconds', flush=True)
        remaining = int(delay)
        while remaining:
            pause = min(remaining, 60)
            time.sleep(pause)
            remaining -= pause
        response = (client or httpx).get(url, headers=headers, timeout=90, follow_redirects=True)
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
    intervals = [(f'{year}-{month:02d}-01T00:00:00Z', f'{year + int(month == 10)}-{(month + 3 - 1) % 12 + 1:02d}-01T00:00:00Z')
                 for month in [1, 4, 7, 10]] if year else [('2000-01-01', '2015-12-31')]
    pieces = []
    for quarter, (start, end) in enumerate(intervals):
        stop = end
        if year and quarter == 3:
            end = (pd.Timestamp(end) - pd.Timedelta(seconds=1)).isoformat()
        url = API + collection + '/items?' + urlencode(dict(f='json', monitoring_location_id='USGS-' + station,
            parameter_code='00060', datetime=start + '/' + end, limit=50000, skipGeometry='true'))
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
        series = parse_observations(features, station, '00011' if year else '00003')
        if year:
            series = series.loc[series.index < pd.Timestamp(stop)]
        pieces.append(series)
    result = pd.concat(pieces).sort_index()
    if result.index.has_duplicates:
        raise ValueError('Overlapping USGS chunks')
    return result


def reference(station):
    historical = observations(station)
    if len(historical) < 3000 or historical.quantile(.95) <= 0:
        raise IneligibleGauge('Insufficient pre-2016 reference')
    return float(historical.quantile(.95))


def collect(stage):
    protocol()
    if stage == 'test':
        verify_selection()
    stations = json.loads((OUT / 'stations.json').read_text())
    destination = OUT / (stage + '.csv.gz')
    if destination.exists():
        raise FileExistsError('Dataset already materialized')

    def one(station):
        sid = station['station']
        try:
            cutoff = reference(sid)
            if stage == 'test':
                years = [TEST_YEAR]
            elif station['role'] == 'geographic_test':
                years = GEOGRAPHIC_YEARS
            else:
                years = DEVELOPMENT_YEARS
            q = hourly_flow(pd.concat([observations(sid, year) for year in years]).sort_index())
            if q.empty:
                raise IneligibleGauge('No accepted instantaneous discharge')
            q = q.reindex(pd.date_range(f'{years[0]}-01-01', f'{years[-1] + 1}-01-01', freq='h', tz='UTC', inclusive='left'))
            rows, _, diagnostics = decision_rows(q, cutoff, station)
            coverage = dict(station=sid, huc=station['huc'], role=station['role'], reference_cfs=cutoff,
                years=years, accepted_hours=int(q.notna().sum()), total_hours=len(q),
                decisions=int(len(rows) // 3), **diagnostics,
                positives=rows.groupby('horizon').target.sum().to_dict())
            print(sid, station['role'], 'hours', coverage['accepted_hours'],
                  'events', len(diagnostics['events']), flush=True)
            return rows, coverage
        except (IneligibleGauge, ValueError) as error:
            print(sid, 'EXCLUDED', str(error), flush=True)
            return None, dict(station=sid, role=station['role'], excluded=str(error))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one, stations))
    frames = [frame for frame, _ in records if frame is not None and len(frame)]
    if not frames:
        raise ValueError('No usable research data')
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(destination, index=False, compression={'method': 'gzip', 'mtime': 0})
    write_json(OUT / (stage + '-coverage.json'), [c for _, c in records])
    write_json(OUT / (stage + '-provenance.json'), dict(dataset_sha256=digest(destination), rows=len(frame),
        source_requests={str(p.relative_to(OUT)): json.loads(p.read_text())['sha256']
                         for p in (OUT / 'sources').rglob('*.request.json')},
        observation_availability='Retrospective approved USGS discharge; assumed two-hour receipt delay, not an as-issued replay'))


def implementation_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / 'scripts/mlp_hybrid.py',
        ROOT / 'scripts/mlp_hybrid_network.py', ROOT / 'scripts/mlp_hybrid_worker.py',
        ROOT / 'scripts/shift_hydrology_metrics.py']}


def verify_selection():
    selection = json.loads((OUT / 'selection.json').read_text())
    if selection['protocol_sha256'] != digest(OUT / 'protocol.json') or \
            selection['dataset_sha256'] != digest(OUT / 'development.csv.gz'):
        raise ValueError('Frozen development inputs changed')
    if selection['code_sha256'] != implementation_hashes():
        raise ValueError('Candidate implementation changed after selection')
    for record in selection['candidates']:
        if digest(OUT / 'artifacts' / (record['name'] + '.joblib')) != record['artifact_sha256']:
            raise ValueError('Frozen candidate artifact changed')
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'development', 'test'])
    arguments = parser.parse_args()
    freeze() if arguments.command == 'freeze' else collect(arguments.command)
