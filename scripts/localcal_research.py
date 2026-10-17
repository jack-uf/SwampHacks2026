"""Local probability adaptation; see docs/LOCALCAL_RESEARCH_PROTOCOL.md."""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from scripts.hydrology_research import ROOT, digest, write_json

OUT = ROOT / 'experiments/sprint-20261017-localcal'


def adjusted(probability, intercept):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isfinite(intercept):
        raise ValueError('Invalid calibration probabilities or intercept')
    return expit(logit(np.clip(p, 1e-6, 1-1e-6)) + intercept)


def fit_intercept(probability, outcomes, penalty):
    p = np.asarray(probability, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    adjusted(p, 0)
    if p.ndim != 1 or y.shape != p.shape or not len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Invalid calibration sample or penalty')
    z = logit(np.clip(p, 1e-6, 1-1e-6))
    result = minimize_scalar(lambda b: np.sum(np.logaddexp(0, z+b) - y*(z+b)) + penalty*b*b/2,
                             bounds=(-30, 30), method='bounded', options={'xatol': 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('Calibration optimization failed')
    return float(result.x)


def calibration_rows(frame):
    if not frame.decision_date.between('2021-01-08', '2021-12-24').all():
        raise ValueError('Local fitting accepts 2021 calibration decisions only')
    if frame.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate calibration decisions')
    return frame


def verify_protocol():
    record = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in record['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen local-calibration input changed: ' + path)
    return record


def source(relative, url, used):
    from scripts.hydrology_research import fetch
    choices = [ROOT / f'experiments/sprint-20261017-{name}' / relative
               for name in ['hydrology', 'response', 'analog', 'localcal']]
    path = next((p for p in choices if p.exists() and
        json.loads(p.with_name(p.name + '.request.json').read_text())['url'] == url), OUT / relative)
    raw = fetch(path, url)
    used[str(path.relative_to(ROOT))] = json.loads(path.with_name(path.name + '.request.json').read_text())
    return raw


def collect(split):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from numcodecs import Blosc
    from scripts.hydrology_research import ZARR, daily_discharge, basin_features, IneligibleGauge
    from scripts.analog_research import analog_features, ANALOG
    from scripts.response_research import flow_signature
    from scripts.onset_research import onset_rows
    from scripts.analog_worker import bounded
    started = time.monotonic()
    verify_protocol()
    if (OUT / f'{split}.provenance.json').exists():
        raise FileExistsError('Collection already frozen')
    if split == 'test':
        from scripts.localcal_worker import verify_selection
        verify_selection()
    registry = json.loads((OUT / 'stations.json').read_text())
    if split == 'test':
        eligible = {s['station'] for s in json.loads((OUT/'calibration.provenance.json').read_text())['coverage'] if 'excluded' not in s}
        registry = [s for s in registry if s['station'] in eligible]
    old = json.loads((ROOT / 'experiments/sprint-20261017-response/stations.json').read_text())
    if split == 'calibration':
        registry = old + registry
    old_ids = {s['station'] for s in old}
    used = {}
    metadata = json.loads(source('research/hres-metadata.json', ZARR + '.zmetadata', used))['metadata']
    coords = {}
    for field in ['basin', 'date', 'lead_time']:
        spec = metadata[field + '/.zarray']
        coords[field] = np.concatenate([np.frombuffer(Blosc().decode(source(f'research/{field}-{i}.bin', ZARR + f'{field}/{i}', used)), dtype=spec['dtype'])
            for i in range((spec['shape'][0]+spec['chunks'][0]-1)//spec['chunks'][0])])[:spec['shape'][0]]
    if not np.array_equal(coords['lead_time'], np.arange(1, 11)):
        raise ValueError('Unexpected forecast leads')
    dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coords['date'].astype('timedelta64[D]'))
    weather = {s['station']: {} for s in registry}; positions = {}
    for sid in weather:
        matches = np.flatnonzero(coords['basin'] == 'camels_' + sid)
        if len(matches) != 1:
            raise ValueError('Ambiguous forecast basin: ' + sid)
        positions[sid] = int(matches[0])
    for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
        spec = metadata[variable + '/.zarray']
        if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
            raise ValueError('Unexpected forecast layout')
        for chunk in sorted({i//128 for i in positions.values()}):
            values = np.frombuffer(Blosc().decode(source(f'sources/{variable}-{chunk}.bin', ZARR + f'{variable}/{chunk}.0.0', used)), dtype='<f4').reshape(spec['chunks'])
            for sid, i in positions.items():
                if i//128 == chunk:
                    weather[sid][variable] = values[i % 128].copy()
    def observation(s):
        sid = s['station']
        end = '2024-09-30' if split == 'test' or sid in old_ids else '2021-12-31'
        relative = f'sources/{sid}.rdb' if end == '2024-09-30' else f'calibration-sources/{sid}.rdb'
        url = f'https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sid}&startDT=2000-01-01&endDT={end}&parameterCd=00060&statCd=00003'
        return sid, source(relative, url, used).decode()
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict(pool.map(observation, registry))
    rows, coverage = [], []
    for s in registry:
        sid = s['station']; record = dict(station=sid, huc=s['huc'])
        raw = observations[sid]
        if raw.startswith('#  No sites found matching all criteria\n'):
            record['excluded'] = 'official empty historical response'; coverage.append(record); continue
        try:
            q = daily_discharge(raw, sid); q.index = pd.to_datetime(q.index)
            if split == 'calibration':
                q = q.loc[:'2021-12-31']
            signature = flow_signature(q, s['area_km2'])
            threshold = float(q[q.index < '2016-01-01'].quantile(.95))
            rain, temp = [weather[sid][v] for v in ['hres_total_precipitation', 'hres_temperature_2m']]
            if np.nanmedian(temp[:366]) > 100:
                raise ValueError('Unexpected forecast temperature units')
            f = onset_rows(basin_features(dates, rain, temp, q, threshold, s, 1), q, threshold, s['timezone'])
            start, end = ('2021-01-08', '2021-12-24') if split == 'calibration' else ('2023-01-08', '2024-09-25')
            f = f[f.decision_date.between(start, end)].copy()
            if split == 'calibration' and len(f) < 150:
                raise ValueError('Insufficient 2021 calibration decisions')
            for key, value in signature.items():
                f[key] = value
            f['storm_load'] = f[['rain_3', 'rain_4', 'rain_5']].sum(axis=1) / np.expm1(signature['sig_runoff'])
            analog, record['reference'] = analog_features(q, pd.DatetimeIndex(f.decision_date))
        except (IneligibleGauge, ValueError) as error:
            allowed = ['Insufficient or invalid historical flow', 'Nonpositive flow reference', 'Insufficient variable consecutive flows',
                       'Insufficient reference flow', 'Insufficient analog windows', 'Insufficient 2021 calibration decisions']
            if not isinstance(error, IneligibleGauge) and str(error) not in allowed:
                raise
            record['excluded'] = str(error); coverage.append(record); continue
        f[ANALOG] = analog.to_numpy(); f = f[np.isfinite(f[ANALOG]).all(axis=1)]
        if split == 'calibration':
            calibration_rows(f)
            if len(f) < 150:
                record['excluded'] = 'Insufficient complete 2021 calibration decisions'; coverage.append(record); continue
        rows.append(f); record['rows'] = len(f); coverage.append(record)
        bounded(started)
    if not rows:
        raise ValueError('No complete data')
    f = pd.concat(rows, ignore_index=True).sort_values(['decision_date', 'station'])
    if f.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate decisions')
    path = OUT / f'{split}.csv.gz'
    f.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
    if sum(p.stat().st_size for folder in ['sources', 'calibration-sources'] for p in (OUT/folder).glob('*')) > 100_000_000:
        raise ValueError('Source budget exceeded')
    write_json(OUT / f'{split}.provenance.json', dict(dataset_sha256=digest(path), coverage=coverage, sources=used,
        protocol_sha256=digest(OUT/'protocol.json'), rows=len(f), gauges=int(f.station.nunique()),
        seconds=time.monotonic()-started, peak_rss_bytes=bounded(started)))
    print(split, len(f), 'decisions', f.station.nunique(), 'gauges', flush=True)


def load_data(split):
    verify_protocol()
    path = OUT / f'{split}.csv.gz'
    record = json.loads((OUT / f'{split}.provenance.json').read_text())
    if digest(path) != record['dataset_sha256']:
        raise ValueError('Dataset changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['calibration', 'test'])
    collect(parser.parse_args().split)
