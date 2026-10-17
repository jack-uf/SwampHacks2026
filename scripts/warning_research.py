"""Experimental regional warning budgets, independent of production safety alerts."""
import json
import hashlib

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, digest, write_json

OUT = ROOT / 'experiments/sprint-20261017-warning'


def prepare():
    """Screen previously mapped but unused basins without requesting outcomes."""
    from shapely.geometry import shape
    from scripts.response_research import OUT as response, disjoint_basin
    if (OUT / 'stations.json').exists():
        raise FileExistsError('Registry already frozen')
    previous = {s['station']: s for name in ['hydrology', 'onset', 'response']
                for s in json.loads((ROOT / f'experiments/sprint-20261017-{name}/stations.json').read_text())}
    geography = json.loads((response / 'geography.json').read_text())
    geometries = {}
    for sid, checksum in geography['boundary_hashes'].items():
        path = response / 'basins' / f'{sid}.gauge-basin.json'
        if digest(path) != checksum:
            raise ValueError('Basin changed: ' + sid)
        location = response / 'basins' / f'{sid}.gauge-location.json'
        if digest(location) != geography['location_hashes'][sid]:
            raise ValueError('Gauge mapping changed: ' + sid)
        geometries[sid] = shape(json.loads(path.read_text())['features'][0]['geometry'])
    used_huc = {s['huc'][:4] for s in previous.values()}
    sites = json.loads((ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    zones = dict(EST='America/New_York', CST='America/Chicago', MST='America/Denver', PST='America/Los_Angeles')
    fixed = dict(EST='Etc/GMT+5', CST='Etc/GMT+6', MST='Etc/GMT+7', PST='Etc/GMT+8')
    candidates = []
    for s in sites:
        sid, huc = s['site_no'], s['huc_cd']
        if sid not in geometries or sid in previous or huc[:4] in used_huc:
            continue
        candidates.append(dict(station=sid, name=s['station_nm'], huc=huc[:8],
            latitude=float(s['dec_lat_va']), longitude=float(s['dec_long_va']),
            area_km2=float(s['drain_area_va']) * 2.589988110336,
            elevation_m=float(s['alt_va']) * .3048,
            timezone=(zones if s['local_time_fg'] == 'Y' else fixed)[s['tz_cd']], role='fresh_test'))
    occupied = [geometries[sid] for sid in previous]
    selected, exclusions, groups = [], [], []
    rank = lambda h: hashlib.sha256(('warning-20261017-' + h).encode()).hexdigest()
    for h in sorted({s['huc'][:4] for s in candidates}, key=rank):
        accepted = []
        for s in sorted([s for s in candidates if s['huc'][:4] == h], key=lambda s: s['station']):
            if disjoint_basin(geometries[s['station']], occupied):
                accepted.append(s)
            else:
                exclusions.append(dict(station=s['station'], reason='upstream polygon overlaps previous population'))
        if accepted:
            groups.append(h)
            selected.extend(accepted)
            occupied.extend(geometries[s['station']] for s in accepted)
    write_json(OUT / 'stations.json', selected)
    write_json(OUT / 'geography.json', dict(previous_stations=len(previous), candidates=len(candidates),
        groups=groups, exclusions=exclusions, stations=len(selected),
        parent_geography_sha256=digest(response / 'geography.json'),
        note='Metadata-only selection. Every accepted upstream polygon has zero positive-area intersection with prior gauges and earlier accepted HUC4 groups.'))
    print('Reserved', len(selected), 'stations in', len(groups), 'new HUC4 groups', flush=True)


def budget_alerts(frame, probabilities, threshold):
    """Spend one credit per decision; refund correct warnings after ten days."""
    p = np.asarray(probabilities, dtype=float)
    f = frame.reset_index(drop=True).copy()
    if p.shape != (len(f),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid threshold')
    if not f.exceedance.isin([0, 1]).all() or f.duplicated(['station', 'decision_date']).any():
        raise ValueError('Invalid labels or duplicate decisions')
    if not f.huc.astype(str).str.fullmatch(r'\d{8}(\d{4})?').all():
        raise ValueError('Invalid HUC8 or HUC12')
    f['day'] = pd.to_datetime(f.decision_date, format='%Y-%m-%d', errors='raise')
    f['probability'] = p
    alerts = np.zeros(len(f), dtype=bool)
    for _, region in f.groupby(f.huc.str[:2]):
        credits, pending = 0, []
        for day, decisions in region.groupby('day', sort=True):
            mature = [i for i in pending if f.at[i, 'day'] <= day - pd.Timedelta(days=10)]
            credits = min(300, credits + len(decisions) + 100 * int(f.loc[mature, 'exceedance'].sum()))
            pending = [i for i in pending if i not in mature]
            proposals = decisions[decisions.probability >= threshold].sort_values(
                ['probability', 'station'], ascending=[False, True])
            chosen = proposals.index[:credits // 100].tolist()
            alerts[chosen] = True
            credits -= 100 * len(chosen)
            pending.extend(chosen)
    return alerts


def verify_protocol():
    from scripts.response_worker import verify_selection
    verify_selection()
    frozen = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in frozen['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen warning input changed: ' + path)
    return frozen


def collect():
    """Reuse archived forecast chunks and collect only the reserved USGS gauges."""
    from concurrent.futures import ThreadPoolExecutor
    from numcodecs import Blosc
    from scripts.hydrology_research import (OUT as previous, ZARR, fetch, daily_discharge,
                                           IneligibleGauge, basin_features)
    from scripts.response_research import flow_signature, response_features
    from scripts.onset_research import onset_rows
    verify_protocol()
    if (OUT / 'provenance.json').exists():
        raise FileExistsError('Dataset already frozen')
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
        spec = metadata[field + '/.zarray']
        pieces = [np.frombuffer(Blosc().decode(cached(f'research/{field}-{i}.bin', ZARR + f'{field}/{i}')), dtype=spec['dtype'])
                  for i in range((spec['shape'][0] + spec['chunks'][0] - 1) // spec['chunks'][0])]
        coords[field] = np.concatenate(pieces)[:spec['shape'][0]]
    if not np.array_equal(coords['lead_time'], np.arange(1, 11)):
        raise ValueError('Unexpected forecast leads')
    dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coords['date'].astype('timedelta64[D]'))
    stations = json.loads((OUT / 'stations.json').read_text())
    weather = {s['station']: {} for s in stations}
    indices = {s['station']: int(np.flatnonzero(coords['basin'] == 'camels_' + s['station'])[0]) for s in stations}
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
    # Inspect one source's schema before expanding the fixed registry.
    first = observation(stations[0])
    daily_discharge(first[1], first[0])
    print('First gauge schema accepted; collecting reserved population', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict([first] + list(pool.map(observation, stations[1:])))
    rows, coverage = [], []
    for s in stations:
        sid = s['station']; record = dict(station=sid, huc=s['huc'])
        try:
            q = daily_discharge(observations[sid], sid); q.index = pd.to_datetime(q.index)
            signature = flow_signature(q, s['area_km2'])
        except (IneligibleGauge, ValueError) as error:
            if not isinstance(error, IneligibleGauge) and str(error) not in ['Insufficient or invalid historical flow', 'Nonpositive flow reference', 'Insufficient variable consecutive flows']:
                raise
            record['excluded'] = str(error); coverage.append(record); continue
        reference = float(q[q.index < '2016-01-01'].quantile(.95))
        rain, temp = [weather[sid][key] for key in ['hres_total_precipitation', 'hres_temperature_2m']]
        if np.nanmedian(temp[:366]) > 100:
            raise ValueError('Unexpected temperature units')
        f = onset_rows(basin_features(dates, rain, temp, q, reference, s, 1), q, reference, s['timezone'])
        f = f[f.decision_date.between('2023-01-08', '2024-09-25')].copy()
        for key, value in signature.items():
            f[key] = value
        f['storm_load'] = f[['rain_3', 'rain_4', 'rain_5']].sum(axis=1) / np.expm1(signature['sig_runoff'])
        if not np.isfinite(f[response_features()]).all(axis=None):
            raise ValueError('Nonfinite features')
        if len(f):
            rows.append(f); record.update(rows=len(f), reference_cfs=reference)
        else:
            record['excluded'] = 'no eligible complete decisions'
        coverage.append(record)
    frame = pd.concat(rows, ignore_index=True).sort_values(['decision_date', 'station'])
    if frame.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate decisions')
    path = OUT / 'test.csv.gz'
    frame.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
    source_bytes = sum(p.stat().st_size for p in (OUT / 'sources').glob('*'))
    if source_bytes > 300_000_000:
        raise RuntimeError('Source budget exceeded')
    write_json(OUT / 'provenance.json', dict(datasets=dict(test=dict(rows=len(frame), gauges=int(frame.station.nunique()), sha256=digest(path))),
        coverage=coverage, sources=used, new_source_bytes=source_bytes, protocol_sha256=digest(OUT / 'protocol.json')))
    print('Dataset frozen without inspecting final outcomes', flush=True)


def load_test():
    verify_protocol()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    path = OUT / 'test.csv.gz'
    if digest(path) != provenance['datasets']['test']['sha256']:
        raise ValueError('Dataset changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'collect'])
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else collect)()
