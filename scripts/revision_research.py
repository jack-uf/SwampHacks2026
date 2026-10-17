"""Forecast availability-gap and revision experiment, isolated from serving."""
import json
import resource
import time

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, digest, write_json

OUT = ROOT / 'experiments/sprint-20261017-revision'
BRIDGE = [f'{kind}_bridge_{lag}' for kind in ['rain', 'temp'] for lag in [1, 2]]
REVISION = [f'{kind}_{stat}_{day}' for kind in ['rain', 'temp']
            for stat in ['mean', 'spread', 'revision'] for day in range(4)]


def forecast_revisions(dates, rain, temperature):
    """Align three buffered initializations to the same UTC forecast days."""
    dates = pd.DatetimeIndex(dates).as_unit('ns')
    if not len(dates) or not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError('Require ordered daily forecast dates')
    rain, temperature = np.asarray(rain), np.asarray(temperature)
    if rain.shape != (len(dates), 10) or temperature.shape != rain.shape:
        raise ValueError('Invalid forecast shape')
    if np.isinf(rain).any() or np.isinf(temperature).any() or np.any(rain < -.1):
        raise ValueError('Invalid precipitation or temperature')
    result = pd.DataFrame(index=dates)
    for kind, values in [('rain', np.maximum(rain, 0)), ('temp', temperature)]:
        for lag in [1, 2]:
            result[f'{kind}_bridge_{lag}'] = pd.Series(values[:, 1], index=dates).shift(lag + 1)
        for day in range(4):
            aligned = np.column_stack([pd.Series(values[:, day + age], index=dates).shift(age)
                                       for age in [1, 2, 3]])
            result[f'{kind}_mean_{day}'] = aligned.mean(axis=1)
            result[f'{kind}_spread_{day}'] = aligned.std(axis=1)
            result[f'{kind}_revision_{day}'] = aligned[:, 0] - aligned[:, 2]
    return result[BRIDGE + REVISION]


def verify_protocol():
    record = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in record['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen revision input changed: ' + path)
    return record


def collect():
    """Enrich existing development decisions using immutable cached forecasts only."""
    from numcodecs import Blosc
    from scripts.analog_research import load_data as analog_data
    from scripts.hydrology_research import ZARR
    verify_protocol()
    if (OUT / 'training.provenance.json').exists() or (OUT / 'training.csv.gz').exists():
        raise FileExistsError('Development collection already retained')
    started = time.monotonic(); used = {}

    def source(relative, url):
        for name in ['hydrology', 'response', 'analog', 'localcal']:
            path = ROOT / f'experiments/sprint-20261017-{name}' / relative
            manifest = path.with_name(path.name + '.request.json')
            if not path.exists():
                continue
            record = json.loads(manifest.read_text())
            if record['url'] != url or record['sha256'] != digest(path):
                raise ValueError('Invalid retained forecast: ' + str(path))
            used[str(path.relative_to(ROOT))] = record
            return path.read_bytes()
        raise FileNotFoundError('Required cached forecast unavailable: ' + relative)

    metadata = json.loads(source('research/hres-metadata.json', ZARR + '.zmetadata'))['metadata']
    coordinates = {}
    for field in ['basin', 'date', 'lead_time']:
        spec = metadata[field + '/.zarray']
        coordinates[field] = np.concatenate([np.frombuffer(Blosc().decode(source(
            f'research/{field}-{i}.bin', ZARR + f'{field}/{i}')), dtype=spec['dtype'])
            for i in range((spec['shape'][0] + spec['chunks'][0] - 1) // spec['chunks'][0])])[:spec['shape'][0]]
    if not np.array_equal(coordinates['lead_time'], np.arange(1, 11)):
        raise ValueError('Unexpected forecast lead convention')
    dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coordinates['date'].astype('timedelta64[D]'))
    f = analog_data('training').reset_index(drop=True)
    if f.role.eq('fresh_test').any() or not f.decision_date.between('2017-01-08', '2022-12-24').all():
        raise ValueError('Only previously inspected development decisions are allowed')
    positions = {}
    for sid in f.station.unique():
        match = np.flatnonzero(coordinates['basin'] == 'camels_' + sid)
        if len(match) != 1:
            raise ValueError('Ambiguous forecast basin: ' + sid)
        positions[sid] = int(match[0])
    weather = {sid: {} for sid in positions}
    for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
        spec = metadata[variable + '/.zarray']
        if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
            raise ValueError('Unexpected forecast layout')
        for chunk in sorted({i // 128 for i in positions.values()}):
            values = np.frombuffer(Blosc().decode(source(f'sources/{variable}-{chunk}.bin',
                ZARR + f'{variable}/{chunk}.0.0')), dtype='<f4').reshape(spec['chunks'])
            for sid, i in positions.items():
                if i // 128 == chunk:
                    weather[sid][variable] = values[i % 128].copy()
    rows = []; coverage = []
    for sid, g in f.groupby('station', sort=True):
        rain, temp = [weather[sid][v] for v in ['hres_total_precipitation', 'hres_temperature_2m']]
        if np.nanmedian(temp[:366]) > 100:
            raise ValueError('Unexpected forecast temperature units')
        extra = forecast_revisions(dates, rain, temp).reindex(pd.DatetimeIndex(g.decision_date))
        g = g.copy(); g[BRIDGE + REVISION] = extra.to_numpy()
        good = np.isfinite(g[BRIDGE + REVISION]).all(axis=1)
        coverage.append(dict(station=sid, original_rows=len(g), retained_rows=int(good.sum())))
        rows.append(g[good])
    result = pd.concat(rows).sort_values(['decision_date', 'station']).reset_index(drop=True)
    if result.duplicated(['station', 'decision_date']).any() or result.empty:
        raise ValueError('Invalid common decision population')
    path = OUT / 'training.csv.gz'
    result.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
    write_json(OUT / 'training.provenance.json', dict(dataset_sha256=digest(path), sources=used,
        protocol_sha256=digest(OUT / 'protocol.json'), coverage=coverage,
        rows=len(result), gauges=int(result.station.nunique()), seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('Cached-only development collection:', len(result), 'decisions', flush=True)


def load_data():
    verify_protocol()
    path = OUT / 'training.csv.gz'
    record = json.loads((OUT / 'training.provenance.json').read_text())
    if record['dataset_sha256'] != digest(path) or record['protocol_sha256'] != digest(OUT / 'protocol.json'):
        raise ValueError('Frozen development data changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})


if __name__ == '__main__':
    collect()
