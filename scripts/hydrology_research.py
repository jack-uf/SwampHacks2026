"""Isolated forecast-memory experiment; see docs/HYDROLOGY_RESEARCH_PROTOCOL.md."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.flood_sprint import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sprint-20261017-hydrology'
ZARR = 'https://storage.googleapis.com/caravan-multimet/v1.1/HRES/timeseries.zarr/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daily_discharge(text, station):
    """Parse one USGS mean-discharge series, excluding estimated/revised-quality flags."""
    lines = [line for line in text.splitlines() if line and not line.startswith('#')]
    if len(lines) < 2 or 'Discharge, cubic feet per second (Mean)' not in text:
        raise ValueError('Expected documented mean discharge in cubic feet per second')
    reader = csv.DictReader([lines[0]] + lines[2:], delimiter='\t')
    fields = [c for c in reader.fieldnames if c.endswith('_00060_00003')]
    if len(fields) != 1 or fields[0] + '_cd' not in reader.fieldnames:
        raise ValueError('Expected exactly one mean-discharge series')
    rows = list(reader)
    if any(r['site_no'] != station or r['agency_cd'] != 'USGS' for r in rows):
        raise ValueError('Unexpected agency or station')
    if len({r['datetime'] for r in rows}) != len(rows):
        raise ValueError('Duplicate observation date')
    field = fields[0]
    result = {}
    for row in rows:
        if row[field + '_cd'] != 'A':
            continue
        try:
            value = float(row[field])
        except ValueError:
            continue
        if np.isfinite(value) and value >= 0:
            pd.Timestamp(row['datetime'])
            result[row['datetime']] = value
    return pd.Series(result, dtype=float).sort_index()


STATIC = ['season_sin', 'season_cos', 'latitude', 'longitude', 'log_area', 'elevation_m']
WEATHER = [f'{kind}_{lead}' for kind in ['rain', 'temp'] for lead in range(2, 6)] + STATIC
HYDRO = [f'q_{lag}' for lag in [3, 4, 7, 14]] + ['q_change'] + STATIC
MEMORY = [f'memory_{tau}' for tau in [2, 7, 30, 90]]
SNOW = [f'liquid_{tau}' for tau in [2, 7, 30, 90]] + ['snow_store']
COMBINED = list(dict.fromkeys(WEATHER + HYDRO))
ALL_FEATURES = COMBINED + MEMORY + SNOW


def basin_features(dates, rain, temperature, discharge, reference, station, horizon):
    """One decision-time feature population shared by every ablation."""
    dates = pd.DatetimeIndex(dates).as_unit("ns")
    if horizon not in (1, 2) or not dates.equals(pd.date_range(dates[0], dates[-1])):
        raise ValueError('Require daily ordered forecasts and a declared horizon')
    if rain.shape != (len(dates), 10) or temperature.shape != rain.shape or reference <= 0:
        raise ValueError('Invalid forecast shape or discharge reference')
    if np.any(rain[np.isfinite(rain)] < -.1):
        raise ValueError('Negative precipitation')
    rain = np.maximum(rain, 0)
    result = pd.DataFrame(index=dates)
    for kind, values in [('rain', rain), ('temp', temperature)]:
        for lead in range(2, 6):
            result[f'{kind}_{lead}'] = pd.Series(values[:, lead-1], index=dates).shift(1)
    history = pd.Series(rain[:, 0], index=dates)
    snow, liquid, store = [], [], 0.
    for p, t in zip(rain[:, 0], temperature[:, 0]):
        if not np.isfinite([p, t]).all():
            snow.append(np.nan); liquid.append(np.nan); store = 0.
            continue
        if t <= 0:
            store += p; water = 0.
        else:
            melt = min(store, 3 * t); store -= melt; water = p + melt
        snow.append(store); liquid.append(water)
    for tau in [2, 7, 30, 90]:
        alpha = 1 - np.exp(-1 / tau)
        result[f'memory_{tau}'] = history.ewm(alpha=alpha, adjust=False).mean().shift(2)
        result[f'liquid_{tau}'] = pd.Series(liquid, index=dates).ewm(alpha=alpha, adjust=False).mean().shift(2)
    result['snow_store'] = pd.Series(snow, index=dates).shift(2)
    complete = pd.Series(np.isfinite(rain[:, 0]) & np.isfinite(temperature[:, 0]), index=dates)
    result['history_complete'] = complete.rolling(270).sum().shift(2).eq(270)
    q = discharge.reindex(dates)
    for lag in [3, 4, 7, 14]:
        result[f'q_{lag}'] = np.log1p(q.shift(lag) / reference)
    result['q_change'] = result.q_3 - result.q_7
    target = q.shift(-horizon)
    result['exceedance'] = (target > reference).astype(int)
    result['target_discharge_cfs'] = target
    result['reference_cfs'] = reference
    result['season_sin'] = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    result['season_cos'] = np.cos(2 * np.pi * dates.dayofyear / 365.25)
    for column in ['station', 'role', 'huc', 'latitude', 'longitude', 'elevation_m']:
        result[column] = station[column]
    result['log_area'] = np.log1p(station['area_km2'])
    result['horizon'] = horizon
    result['decision_date'] = dates.strftime('%Y-%m-%d')
    local_dates = dates + pd.Timedelta(days=horizon)
    result['target_date'] = local_dates.strftime('%Y-%m-%d')
    local_start = local_dates.tz_localize(station['timezone']).tz_convert('UTC').tz_localize(None)
    local_end = (local_dates + pd.Timedelta(days=1)).tz_localize(station['timezone']).tz_convert('UTC').tz_localize(None)
    result['lead_start_hours'] = (local_start - dates).total_seconds() / 3600
    result['lead_end_hours'] = (local_end - dates).total_seconds() / 3600
    # A certified onset needs seven observed below-threshold days, not seven retained rows.
    prior = pd.Series(index=dates, data=np.where(q.notna(), (q <= reference).astype(float), np.nan))
    onset = (prior.shift(1).rolling(7).sum() == 7) & (q > reference)
    result['onset'] = onset.shift(-horizon, fill_value=False)
    return result.loc[result.history_complete & target.notna() & np.isfinite(result[ALL_FEATURES]).all(axis=1)]


def fetch(path, url):
    """Cache immutable source bytes with checksums, without silent retries."""
    import httpx
    from datetime import datetime, timezone
    record = path.with_name(path.name + '.request.json')
    if path.exists() and record.exists():
        metadata = json.loads(record.read_text())
        if metadata['url'] != url or digest(path) != metadata['sha256']:
            raise ValueError('Source cache mismatch: ' + str(path))
        return path.read_bytes()
    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    if len(response.content) > 25_000_000:
        raise ValueError('Source exceeds bounded chunk budget')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    write_json(record, dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                           status=response.status_code, bytes=len(response.content), sha256=digest(path)))
    return response.content


def protocol():
    frozen = json.loads((OUT / 'protocol.json').read_text())
    if digest(ROOT / frozen['document']) != frozen['document_sha256'] or digest(OUT / 'stations.json') != frozen['stations_sha256']:
        raise ValueError('Frozen protocol or registry changed')
    for path, checksum in frozen['preserved'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Production artifact changed: ' + path)
    return frozen


def collect():
    import concurrent.futures
    from numcodecs import Blosc
    protocol()
    if (OUT / 'training.csv.gz').exists():
        raise FileExistsError('Dataset frozen; use select or evaluate')
    research, sources = OUT / 'research', OUT / 'sources'
    metadata = json.loads(fetch(research / 'hres-metadata.json', ZARR + '.zmetadata'))['metadata']
    coords = {}
    for field in ['basin', 'date', 'lead_time']:
        spec = metadata[field + '/.zarray']; pieces = []
        for i in range((spec['shape'][0] + spec['chunks'][0] - 1) // spec['chunks'][0]):
            raw = fetch(research / f'{field}-{i}.bin', ZARR + f'{field}/{i}')
            pieces.append(np.frombuffer(Blosc().decode(raw), dtype=spec['dtype']))
        coords[field] = np.concatenate(pieces)[:spec['shape'][0]]
    if not np.array_equal(coords['lead_time'], np.arange(1, 11)):
        raise ValueError('Forecast lead semantics changed')
    dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coords['date'].astype('timedelta64[D]'))
    stations = json.loads((OUT / 'stations.json').read_text())
    indices = {s['station']: int(np.flatnonzero(coords['basin'] == 'camels_' + s['station'])[0]) for s in stations}
    weather = {s['station']: {} for s in stations}
    for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
        spec = metadata[variable + '/.zarray']
        if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
            raise ValueError('Unexpected Zarr layout; review before downloading')
        for chunk in sorted({idx // 128 for idx in indices.values()}):
            raw = fetch(sources / f'{variable}-{chunk}.bin', ZARR + f'{variable}/{chunk}.0.0')
            array = np.frombuffer(Blosc().decode(raw), dtype='<f4').reshape(spec['chunks'])
            for sid, idx in indices.items():
                if idx // 128 == chunk:
                    weather[sid][variable] = array[idx % 128].copy()
            print(variable, chunk, 'cached', flush=True)
    def observation(station):
        sid = station['station']
        url = ('https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=' + sid +
               '&startDT=2000-01-01&endDT=2024-09-30&parameterCd=00060&statCd=00003')
        return sid, fetch(sources / f'{sid}.rdb', url).decode()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict(pool.map(observation, stations))
    rows, coverage = [], []
    for station in stations:
        sid = station['station']; q = daily_discharge(observations[sid], sid)
        q.index = pd.to_datetime(q.index)
        reference = q[q.index < '2016-01-01']
        record = dict(station=sid, role=station['role'], accepted_days=len(q), reference_days=len(reference))
        if len(reference) < 3000 or reference.quantile(.95) <= 0:
            record['excluded'] = 'insufficient_positive_reference'; coverage.append(record); continue
        threshold = float(reference.quantile(.95))
        rain, temp = [weather[sid][key] for key in ['hres_total_precipitation', 'hres_temperature_2m']]
        if np.nanmedian(temp[:366]) > 100:
            raise ValueError('Temperature is not in the documented Celsius units')
        for horizon in (1, 2):
            frame = basin_features(dates, rain, temp, q, threshold, station, horizon)
            frame = frame[(frame.target_date >= '2017-01-01') & (frame.target_date <= '2024-09-28')]
            rows.append(frame)
            record[f'rows_h{horizon}'] = len(frame)
        record['reference_cfs'] = threshold; record['negative_rain_values_truncated'] = int(np.sum(rain < 0)); coverage.append(record)
    frame = pd.concat(rows, ignore_index=True).sort_values(['horizon', 'target_date', 'station'])
    if frame.duplicated(['station', 'horizon', 'target_date']).any():
        raise ValueError('Duplicate experiment decisions')
    frame.to_csv(OUT / 'training.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    records = [dict(path=str(p.relative_to(OUT)), **json.loads(p.read_text())) for p in sorted(sources.glob('*.request.json'))]
    source_bytes = sum(p.stat().st_size for p in sources.glob('*') if not p.name.endswith('.json'))
    if source_bytes > 500_000_000:
        raise ValueError('Source cache budget exceeded')
    write_json(OUT / 'provenance.json', dict(dataset_sha256=digest(OUT / 'training.csv.gz'),
        protocol_sha256=digest(OUT / 'protocol.json'), sources=records, source_bytes=source_bytes,
        coverage=coverage, forecast_dates=[str(dates[0].date()), str(dates[-1].date())],
        availability='Reconstructed approved inputs; per-record historical publication is unverified'))
    print('Frozen dataset', len(frame), 'rows;', frame.station.nunique(), 'gauges', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect'])
    args = parser.parse_args()
    collect()


def partitions(frame):
    if not frame.role.isin(['development', 'geographic_validation', 'geographic_test']).all():
        raise ValueError('Unknown geographic role')
    if frame.groupby(frame.huc.str[:2]).role.nunique().gt(1).any():
        raise ValueError('A geographic HUC2 crosses roles')
    for role, regions in [('geographic_validation', ['03', '12']), ('geographic_test', ['17', '18'])]:
        if not frame.role.eq(role).equals(frame.huc.str[:2].isin(regions)):
            raise ValueError('Unexpected geographic assignment')
    d = frame.target_date; development = frame.role.eq('development')
    middle = d.str[5:].between('01-08', '12-24')
    return dict(train=frame[development & d.between('2017-01-08', '2020-12-24') & middle],
                calibration=frame[development & d.between('2021-01-08', '2021-12-24')],
                validation=frame[development & d.between('2022-01-08', '2022-12-24')],
                geographic_validation=frame[frame.role.eq('geographic_validation') & d.between('2022-01-08', '2022-12-24')],
                forward_test=frame[development & d.between('2023-01-08', '2024-09-28')],
                geographic_test=frame[frame.role.eq('geographic_test') & d.between('2023-01-08', '2024-09-28')])


def budget_threshold(y, p, budget):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or len(y) == 0 or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not 0 <= budget <= 1:
        raise ValueError('Invalid outcomes, probabilities or budget')
    order = np.argsort(-p, kind='stable'); sorted_p = p[order]
    ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    tp, fp = np.cumsum(y[order])[ends], np.cumsum(1 - y[order])[ends]
    feasible = np.flatnonzero(fp <= budget * len(y) + 1e-9)
    if not len(feasible):
        return float(np.nextafter(p.max(), np.inf))
    best = sorted(feasible, key=lambda i: (-tp[i], fp[i], -sorted_p[ends[i]]))[0]
    return float(sorted_p[ends[best]])
