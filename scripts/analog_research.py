"""Reference-transition features; governed by docs/ANALOG_RESEARCH_PROTOCOL.md."""
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import ROOT, digest, write_json
from scripts.onset_research import onset_rows

OUT = ROOT / 'experiments/sprint-20261017-analog'
ANALOG = ['analog_logit'] + [f'analog_{kind}_{lead}' for kind in ['mean', 'sd'] for lead in [1, 2, 3]] + ['analog_distance']


def analog_features(discharge, issues):
    """Fit a pre-2016 local transition library and query delayed calendar histories."""
    if discharge.index.has_duplicates or not discharge.index.is_monotonic_increasing or (discharge.dropna() < 0).any() or not np.isfinite(discharge.dropna()).all():
        raise ValueError('Invalid flow history')
    reference = discharge.loc[discharge.index < '2016-01-01'].dropna()
    if len(reference) < 3000 or reference.quantile(.95) <= 0:
        raise ValueError('Insufficient reference flow')
    threshold = float(reference.quantile(.95))
    dates = pd.date_range(discharge.index.min(), discharge.index.max())
    q = np.log1p(discharge.reindex(dates) / threshold)
    inputs = pd.DataFrame({'q3': q.shift(3), 'change1': q.shift(3) - q.shift(4),
        'change4': q.shift(3) - q.shift(7), 'change7': q.shift(7) - q.shift(14),
        'sin': np.sin(2 * np.pi * dates.dayofyear / 365.25),
        'cos': np.cos(2 * np.pi * dates.dayofyear / 365.25)}, index=dates)
    library = onset_rows(inputs.loc[dates < '2015-12-29'].dropna(), discharge, threshold, 'UTC')
    if len(library) < 1000:
        raise ValueError('Insufficient analog windows')
    scaler = StandardScaler().fit(library[inputs.columns])
    neighbors = NearestNeighbors(n_neighbors=128, n_jobs=2).fit(scaler.transform(library[inputs.columns]))
    query = inputs.reindex(pd.DatetimeIndex(issues))
    valid = np.isfinite(query).all(axis=1)
    result = pd.DataFrame(np.nan, index=query.index, columns=ANALOG)
    if valid.any():
        distance, positions = neighbors.kneighbors(scaler.transform(query.loc[valid]))
        probability = (library.exceedance.to_numpy()[positions].sum(axis=1) + 1) / 178
        result.loc[valid, 'analog_logit'] = np.log(probability / (1 - probability))
        trajectories = library[['flow_1', 'flow_2', 'flow_3']].to_numpy()[positions]
        for kind, values in [('mean', trajectories.mean(axis=1)), ('sd', trajectories.std(axis=1))]:
            result.loc[valid, [f'analog_{kind}_{lead}' for lead in [1, 2, 3]]] = values
        result.loc[valid, 'analog_distance'] = distance.mean(axis=1)
    return result, dict(reference_days=len(reference), reference_cfs=threshold, library_windows=len(library),
        positive_windows=int(library.exceedance.sum()), last_library_target=str((library.index.max() + pd.Timedelta(days=3)).date()))


def verify_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in protocol['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen analog input changed: ' + path)
    return protocol


def cached(relative, url):
    from scripts.hydrology_research import fetch
    # Reuse immutable source caches; all responses still require matching manifests.
    paths = [ROOT / f'experiments/sprint-20261017-{name}' / relative for name in ['hydrology', 'response']]
    path = next((p for p in paths if p.exists()), OUT / relative)
    return path, fetch(path, url)


def collect(split):
    from concurrent.futures import ThreadPoolExecutor
    from numcodecs import Blosc
    from scripts.hydrology_research import daily_discharge, basin_features, ZARR, IneligibleGauge
    from scripts.response_research import flow_signature, response_features
    verify_protocol()
    if (OUT / f'{split}.provenance.json').exists():
        raise FileExistsError('Dataset already frozen')
    if split == 'test':
        from scripts.analog_worker import verify_selection
        verify_selection()
    response = ROOT / 'experiments/sprint-20261017-response'
    registry = json.loads(((OUT if split == 'test' else response) / 'stations.json').read_text())
    original = None
    if split == 'training':
        original = pd.read_csv(response / 'training.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
        registry = [s for s in registry if s['station'] in set(original.station)]
    used = {}
    def source(relative, url):
        path, raw = cached(relative, url)
        used[str(path.relative_to(ROOT))] = json.loads(path.with_name(path.name + '.request.json').read_text())
        return raw
    weather = {}
    if split == 'test':
        metadata = json.loads(source('research/hres-metadata.json', ZARR + '.zmetadata'))['metadata']
        coords = {}
        for field in ['basin', 'date', 'lead_time']:
            spec = metadata[field + '/.zarray']
            pieces = [np.frombuffer(Blosc().decode(source(f'research/{field}-{i}.bin', ZARR + f'{field}/{i}')), dtype=spec['dtype'])
                      for i in range((spec['shape'][0] + spec['chunks'][0] - 1) // spec['chunks'][0])]
            coords[field] = np.concatenate(pieces)[:spec['shape'][0]]
        if not np.array_equal(coords['lead_time'], np.arange(1, 11)):
            raise ValueError('Unexpected forecast leads')
        dates = pd.DatetimeIndex(np.datetime64('2012-01-01') + coords['date'].astype('timedelta64[D]'))
        indices = {}
        for s in registry:
            positions = np.flatnonzero(coords['basin'] == 'camels_' + s['station'])
            if len(positions) != 1:
                raise ValueError('Ambiguous forecast basin')
            indices[s['station']] = int(positions[0])
            weather[s['station']] = {}
        for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
            spec = metadata[variable + '/.zarray']
            if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
                raise ValueError('Unexpected forecast layout')
            for chunk in sorted({i // 128 for i in indices.values()}):
                values = np.frombuffer(Blosc().decode(source(f'sources/{variable}-{chunk}.bin', ZARR + f'{variable}/{chunk}.0.0')), dtype='<f4').reshape(spec['chunks'])
                for sid, i in indices.items():
                    if i // 128 == chunk:
                        weather[sid][variable] = values[i % 128].copy()
    def observation(s):
        sid = s['station']
        url = f'https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sid}&startDT=2000-01-01&endDT=2024-09-30&parameterCd=00060&statCd=00003'
        return sid, source(f'sources/{sid}.rdb', url).decode()
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict(pool.map(observation, registry))
    rows, coverage = [], []
    for s in registry:
        sid = s['station']; record = dict(station=sid, huc=s['huc'])
        text = observations[sid]
        if text.startswith('#  No sites found matching all criteria\n'):
            record['excluded'] = 'official empty historical response'; coverage.append(record); continue
        try:
            q = daily_discharge(text, sid); q.index = pd.to_datetime(q.index)
            signature = flow_signature(q, s['area_km2'])
            threshold = float(q[q.index < '2016-01-01'].quantile(.95))
            if split == 'training':
                f = original[original.station.eq(sid)].copy()
            else:
                rain, temp = [weather[sid][v] for v in ['hres_total_precipitation', 'hres_temperature_2m']]
                if np.nanmedian(temp[:366]) > 100:
                    raise ValueError('Unexpected forecast temperature units')
                f = onset_rows(basin_features(dates, rain, temp, q, threshold, s, 1), q, threshold, s['timezone'])
                f = f[f.decision_date.between('2023-01-08', '2024-09-25')].copy()
                for key, value in signature.items():
                    f[key] = value
                f['storm_load'] = f[['rain_3', 'rain_4', 'rain_5']].sum(axis=1) / np.expm1(signature['sig_runoff'])
            features, record['reference'] = analog_features(q, pd.DatetimeIndex(f.decision_date))
        except (IneligibleGauge, ValueError) as error:
            allowed = ['Insufficient or invalid historical flow', 'Nonpositive flow reference', 'Insufficient variable consecutive flows', 'Insufficient reference flow', 'Insufficient analog windows']
            if not isinstance(error, IneligibleGauge) and str(error) not in allowed:
                raise
            record['excluded'] = str(error); coverage.append(record); continue
        f[ANALOG] = features.to_numpy()
        before = len(f); f = f[np.isfinite(f[ANALOG]).all(axis=1)]
        record.update(rows=len(f), incomplete_issues=before-len(f))
        if not len(f):
            record['excluded'] = 'no eligible complete decisions'
        else:
            if not np.isfinite(f[response_features()]).all(axis=None):
                raise ValueError('Invalid shared forecast features')
            rows.append(f)
        coverage.append(record)
    if not rows:
        raise ValueError('No eligible station data')
    f = pd.concat(rows, ignore_index=True).sort_values(['decision_date', 'station'])
    if f.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate decisions')
    path = OUT / f'{split}.csv.gz'
    f.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
    if sum(p.stat().st_size for p in (OUT / 'sources').glob('*')) > 100_000_000:
        raise ValueError('Source budget exceeded')
    write_json(OUT / f'{split}.provenance.json', dict(dataset_sha256=digest(path), coverage=coverage, sources=used,
        protocol_sha256=digest(OUT / 'protocol.json'), rows=len(f), gauges=int(f.station.nunique())))
    print(split, 'collected', len(f), 'decisions', f.station.nunique(), 'gauges', flush=True)


def load_data(split):
    verify_protocol()
    record = json.loads((OUT / f'{split}.provenance.json').read_text())
    path = OUT / f'{split}.csv.gz'
    if digest(path) != record['dataset_sha256']:
        raise ValueError('Dataset changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['training', 'test'])
    collect(parser.parse_args().split)
