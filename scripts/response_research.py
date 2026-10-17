"""Catchment-conditioned onset experiment, separate from served models."""
import json

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, digest, write_json

OUT = ROOT / 'experiments/sprint-20261017-response'
SIGNATURE = ['sig_mean', 'sig_low', 'sig_median', 'sig_extreme', 'sig_cv',
             'sig_lag1', 'sig_flash', 'sig_sin', 'sig_cos', 'sig_runoff']


def flow_signature(discharge, area_km2):
    """Flow response descriptors from accepted history ending in 2015 only."""
    if discharge.index.has_duplicates or not discharge.index.is_monotonic_increasing:
        raise ValueError('Unordered or duplicate flow history')
    q = discharge[discharge.index < '2016-01-01'].dropna()
    if len(q) < 3000 or area_km2 <= 0 or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Insufficient or invalid historical flow')
    reference = float(q.quantile(.95))
    if reference <= 0:
        raise ValueError('Nonpositive flow reference')
    full = q.reindex(pd.date_range(q.index.min(), q.index.max()))
    pairs = pd.concat([full.shift(1), full], axis=1).dropna()
    if len(pairs) < 2000 or pairs.iloc[:, 0].std() == 0 or q.mean() == 0:
        raise ValueError('Insufficient variable consecutive flows')
    phase = 2 * np.pi * q.index.dayofyear / 365.25
    values = [q.mean() / reference, q.quantile(.1) / reference,
              q.quantile(.5) / reference, q.quantile(.99) / reference,
              q.std() / q.mean(), pairs.iloc[:, 0].corr(pairs.iloc[:, 1]),
              np.abs(pairs.iloc[:, 1] - pairs.iloc[:, 0]).sum() / q.sum(),
              np.sum(q * np.sin(phase)) / q.sum(), np.sum(q * np.cos(phase)) / q.sum(),
              np.log1p(reference * .028316846592 * 86.4 / area_km2)]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite flow signature')
    return dict(zip(SIGNATURE, map(float, values)))


def disjoint_basin(candidate, occupied):
    if candidate.is_empty or not candidate.is_valid or candidate.area <= 0:
        raise ValueError('Invalid basin geometry')
    return all(candidate.intersection(other).area == 0 for other in occupied)


def prepare():
    """Reserve new watersheds using metadata and upstream boundaries, never flow."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from shapely.geometry import shape
    from scripts.hydrology_research import fetch, OUT as old
    from scripts.onset_research import OUT as onset
    if (OUT / 'stations.json').exists():
        raise FileExistsError('Station registry already frozen')
    previous = {s['station']: s for path in [old, onset]
                for s in json.loads((path / 'stations.json').read_text())}
    used_huc = {s['huc'][:4] for s in previous.values()}
    sites = json.loads((old / 'research/sites.json').read_text())
    zones = dict(EST='America/New_York', CST='America/Chicago', MST='America/Denver', PST='America/Los_Angeles')
    fixed = dict(EST='Etc/GMT+5', CST='Etc/GMT+6', MST='Etc/GMT+7', PST='Etc/GMT+8')
    candidates = []
    for s in sites:
        huc = s['huc_cd']
        if len(huc) not in [8, 12] or not huc.isdigit() or huc[:4] in used_huc:
            continue
        if s['tz_cd'] not in zones or s['local_time_fg'] not in ['Y', 'N']:
            continue
        try:
            station = dict(station=s['site_no'], name=s['station_nm'], huc=huc[:8],
                latitude=float(s['dec_lat_va']), longitude=float(s['dec_long_va']),
                area_km2=float(s['drain_area_va']) * 2.589988110336,
                elevation_m=float(s['alt_va']) * .3048,
                timezone=(zones if s['local_time_fg'] == 'Y' else fixed)[s['tz_cd']])
            if not np.isfinite([station[k] for k in ['latitude', 'longitude', 'area_km2', 'elevation_m']]).all() or station['area_km2'] <= 0:
                continue
        except ValueError:
            continue
        candidates.append(station)
    # Two deterministic metadata candidates per HUC4, before any outcome request.
    candidates = [s for h in sorted({s['huc'][:4] for s in candidates})
                  for s in sorted([s for s in candidates if s['huc'][:4] == h], key=lambda s: s['station'])[:2]]
    mapping_exclusions = []
    candidate_count = len(candidates)
    def basin(s):
        sid = s['station']
        location_path = OUT / 'basins' / f'{sid}.gauge-location.json'
        # NLDI has no indexed record for 04127918; this explicit coordinate match was verified before outcome access.
        url = ('https://api.water.usgs.gov/nldi/linked-data/hydrolocation?coords=POINT(-84.59783%2046.18585)'
               if sid == '04127918' else f'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}')
        location = json.loads(fetch(location_path, url))
        matched = [f for f in location['features'] if f['properties']['source'] ==
                   ('indexed' if sid == '04127918' else 'nwissite')]
        if len(matched) != 1 or (sid != '04127918' and matched[0]['properties']['identifier'] != 'USGS-' + sid):
            raise ValueError('Ambiguous flowline mapping: ' + sid)
        feature = matched[0]; x, y = feature['geometry']['coordinates']
        distance_km = 111.2 * np.hypot((x - s['longitude']) * np.cos(np.deg2rad(y)), y - s['latitude'])
        if distance_km > 1 or (sid == '04127918' and not feature['properties']['reachcode'].startswith(s['huc'][:4])):
            if sid in previous:
                raise ValueError('Prior gauge mapping cannot be verified: ' + sid)
            mapping_exclusions.append(dict(station=sid, reason='indexed location disagrees with station metadata',
                distance_km=float(distance_km), location_sha256=digest(location_path)))
            return sid, None
        comid = feature['properties']['comid']
        if sid == '04127918' and int(comid) != 12206226:
            raise ValueError('Unexpected verified coordinate mapping')
        path = OUT / 'basins' / f'{sid}.gauge-basin.json'
        raw = fetch(path, f'https://api.water.usgs.gov/nldi/linked-data/comid/{comid}/basin?simplified=true')
        data = json.loads(raw)
        if len(data['features']) != 1:
            raise ValueError('Expected one upstream basin: ' + sid)
        geometry = shape(data['features'][0]['geometry'])
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise ValueError('Invalid upstream basin: ' + sid)
        return sid, geometry
    with ThreadPoolExecutor(max_workers=6) as pool:
        geometries = dict(pool.map(basin, list(previous.values()) + candidates))
    candidates = [s for s in candidates if geometries[s['station']] is not None]
    geometries = {sid: g for sid, g in geometries.items() if g is not None}
    occupied = [geometries[sid] for sid in previous]
    rank = lambda h: hashlib.sha256(('response-20261017-' + h).encode()).hexdigest()
    groups = sorted({s['huc'][:4] for s in candidates}, key=rank)
    selected, excluded, accepted_groups = [], sorted(mapping_exclusions, key=lambda s: s['station']), []
    for h in groups:
        group = []
        for s in [s for s in candidates if s['huc'][:4] == h]:
            if disjoint_basin(geometries[s['station']], occupied):
                group.append(s)
            else:
                excluded.append(dict(station=s['station'], reason='upstream polygon intersects prior population'))
        if not group:
            continue
        role = 'geographic_validation' if len(accepted_groups) < 8 else 'fresh_test'
        for s in group:
            s['role'] = role
        selected.extend(group)
        occupied.extend(geometries[s['station']] for s in group)
        accepted_groups.append(h)
        if len(accepted_groups) == 28:
            break
    if len(accepted_groups) < 28:
        raise ValueError('Insufficient disjoint groups')
    development = [s for s in json.loads((onset / 'stations.json').read_text()) if s['role'] != 'fresh_test']
    write_json(OUT / 'stations.json', development + selected)
    write_json(OUT / 'geography.json', dict(previous_stations=len(previous), candidate_stations=candidate_count,
        selected_groups=accepted_groups, exclusions=excluded,
        huc_disagreements=[dict(station=s['station'], metadata_huc4=s['huc'][:4],
            reach_huc4=json.loads((OUT / 'basins' / f"{s['station']}.gauge-location.json").read_text())['features'][0]['properties']['reachcode'][:4])
            for s in list(previous.values()) + candidates
            if not json.loads((OUT / 'basins' / f"{s['station']}.gauge-location.json").read_text())['features'][0]['properties']['reachcode'].startswith(s['huc'][:4])],
        new_stations=selected, location_requests={sid: json.loads((OUT / 'basins' / f'{sid}.gauge-location.json.request.json').read_text()) for sid in geometries},
        location_hashes={sid: digest(OUT / 'basins' / f'{sid}.gauge-location.json') for sid in geometries}, boundary_hashes={sid: digest(OUT / 'basins' / f'{sid}.gauge-basin.json') for sid in geometries},
        note='No positive-area intersection with previous populations or earlier selected HUC4 groups; simplified NLDI boundaries.'))
    print('Reserved', len(selected), 'new stations in', len(accepted_groups), 'disjoint groups', flush=True)


def response_features():
    from scripts.hydrology_research import COMBINED, SNOW
    return COMBINED + SNOW + SIGNATURE + ['storm_load']


def onset_categories(frame):
    days = (pd.to_datetime(frame.event_date.replace('', pd.NA)) - pd.to_datetime(frame.decision_date)).dt.days.fillna(0).astype(int)
    if not days.between(0, 3).all() or not days.gt(0).equals(frame.exceedance.eq(1)):
        raise ValueError('Invalid onset target day')
    return days.to_numpy()


def verify_protocol():
    from scripts.onset_research import verify_protocol as verify_previous
    verify_previous()
    frozen = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in frozen['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen response input changed: ' + path)
    return frozen


def collect():
    from concurrent.futures import ThreadPoolExecutor
    from numcodecs import Blosc
    from scripts.hydrology_research import (OUT as previous, ZARR, fetch, daily_discharge,
                                           IneligibleGauge, basin_features)
    from scripts.onset_research import OUT as onset, onset_rows
    verify_protocol()
    if (OUT / 'provenance.json').exists():
        raise FileExistsError('Response dataset already frozen')
    used = {}
    def cached(relative, url):
        path = previous / relative
        if not path.exists():
            path = OUT / relative
        raw = fetch(path, url)
        used[str(path.relative_to(ROOT))] = json.loads(path.with_name(path.name + '.request.json').read_text())
        return raw
    metadata = json.loads(cached('research/hres-metadata.json', ZARR + '.zmetadata'))['metadata']
    coords = {}
    for field in ['basin', 'date', 'lead_time']:
        spec = metadata[field + '/.zarray']; pieces = []
        for i in range((spec['shape'][0] + spec['chunks'][0] - 1) // spec['chunks'][0]):
            pieces.append(np.frombuffer(Blosc().decode(cached(f'research/{field}-{i}.bin', ZARR + f'{field}/{i}')), dtype=spec['dtype']))
        coords[field] = np.concatenate(pieces)[:spec['shape'][0]]
    if not np.array_equal(coords['lead_time'], np.arange(1, 11)):
        raise ValueError('Unexpected forecast leads')
    dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coords['date'].astype('timedelta64[D]'))
    stations = json.loads((OUT / 'stations.json').read_text())
    old_ids = {s['station'] for s in json.loads((onset / 'stations.json').read_text())}
    new = [s for s in stations if s['station'] not in old_ids]
    weather = {s['station']: {} for s in new}
    indices = {s['station']: int(np.flatnonzero(coords['basin'] == 'camels_' + s['station'])[0]) for s in new}
    for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
        spec = metadata[variable + '/.zarray']
        if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
            raise ValueError('Unexpected forecast layout')
        for chunk in sorted({i // 128 for i in indices.values()}):
            values = np.frombuffer(Blosc().decode(cached(f'sources/{variable}-{chunk}.bin', ZARR + f'{variable}/{chunk}.0.0')), dtype='<f4').reshape(spec['chunks'])
            for sid, i in indices.items():
                if i // 128 == chunk:
                    weather[sid][variable] = values[i % 128].copy()
    def observation(s):
        sid = s['station']
        url = ('https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=' + sid +
               '&startDT=2000-01-01&endDT=2024-09-30&parameterCd=00060&statCd=00003')
        return sid, cached(f'sources/{sid}.rdb', url).decode()
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict(pool.map(observation, stations))
    source = onset / 'training.csv.gz'
    before = json.loads((onset / 'provenance.json').read_text())
    if digest(source) != before['datasets']['training']['sha256']:
        raise ValueError('Prior training dataset changed')
    old = pd.read_csv(source, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
    rows = {'training': [], 'test': []}; coverage = []; signatures = {}
    for s in stations:
        sid = s['station']; record = dict(station=sid, huc=s['huc'], role=s['role'])
        try:
            q = daily_discharge(observations[sid], sid); q.index = pd.to_datetime(q.index)
            signature = flow_signature(q, s['area_km2'])
        except (IneligibleGauge, ValueError) as error:
            # Only declared source-quality/reference failures may exclude a station.
            if not isinstance(error, IneligibleGauge) and str(error) not in ['Insufficient or invalid historical flow', 'Nonpositive flow reference', 'Insufficient variable consecutive flows']:
                raise
            record['excluded'] = str(error); coverage.append(record); continue
        threshold = float(q[q.index < '2016-01-01'].quantile(.95))
        if sid in old_ids:
            f = old[old.station.eq(sid)].copy()
        else:
            rain, temp = [weather[sid][key] for key in ['hres_total_precipitation', 'hres_temperature_2m']]
            if np.nanmedian(temp[:366]) > 100:
                raise ValueError('Unexpected forecast temperature units')
            f = onset_rows(basin_features(dates, rain, temp, q, threshold, s, 1), q, threshold, s['timezone'])
            f = f[f.decision_date.between('2023-01-08', '2024-09-25')] if s['role'] == 'fresh_test' else f[f.decision_date.between('2022-01-08', '2022-12-24')]
        for key, value in signature.items():
            f[key] = value
        f['storm_load'] = f[['rain_3', 'rain_4', 'rain_5']].sum(axis=1) / np.expm1(signature['sig_runoff'])
        if not np.isfinite(f[response_features()]).all(axis=None):
            raise ValueError('Nonfinite response features')
        if not len(f):
            record['excluded'] = 'no eligible complete decisions'
        else:
            rows['test' if s['role'] == 'fresh_test' else 'training'].append(f)
            signatures[sid] = signature
            record.update(rows=len(f), reference_cfs=threshold)
        coverage.append(record)
    datasets = {}
    for split, pieces in rows.items():
        f = pd.concat(pieces, ignore_index=True).sort_values(['decision_date', 'station'])
        if f.duplicated(['station', 'decision_date']).any():
            raise ValueError('Duplicate response decision')
        path = OUT / f'{split}.csv.gz'
        f.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
        datasets[split] = dict(rows=len(f), gauges=int(f.station.nunique()), sha256=digest(path))
    new_bytes = sum(p.stat().st_size for folder in ['sources', 'basins'] for p in (OUT / folder).glob('*'))
    if new_bytes > 300_000_000:
        raise ValueError('Source expansion exceeds budget')
    write_json(OUT / 'provenance.json', dict(datasets=datasets, coverage=coverage, signatures=signatures,
        sources=used, new_source_bytes=new_bytes, previous_training_sha256=digest(source),
        protocol_sha256=digest(OUT / 'protocol.json')))
    print('Frozen response datasets; no final target summary inspected', flush=True)


def load_data(split):
    verify_protocol()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    path = OUT / f'{split}.csv.gz'
    if digest(path) != provenance['datasets'][split]['sha256']:
        raise ValueError('Response dataset changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})


def partitions(frame):
    from scripts.onset_research import partitions as previous_parts
    from scripts.onset_research import OUT as onset
    old_ids = {s['station'] for s in json.loads((onset / 'stations.json').read_text())}
    if frame.role.eq('fresh_test').any():
        raise ValueError('Final watersheds cannot enter selection')
    parts = previous_parts(frame[frame.station.isin(old_ids)])
    parts['geographic_validation'] = frame[~frame.station.isin(old_ids)]
    if not parts['geographic_validation'].role.eq('geographic_validation').all() or not parts['geographic_validation'].decision_date.str.startswith('2022').all():
        raise ValueError('Invalid new geographic selection population')
    return parts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect'])
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else collect)()
