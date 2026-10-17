"""Emerging high-flow experiment, governed by docs/ONSET_RESEARCH_PROTOCOL.md."""
import json

import numpy as np
import pandas as pd

from scripts.hydrology_research import (ROOT, OUT as PREVIOUS, ZARR, basin_features,
    daily_discharge, IneligibleGauge, fetch, digest, write_json)

OUT = ROOT / 'experiments/sprint-20261017-onset'


def onset_rows(frame, discharge, reference, timezone):
    """Certify new episodes without treating unknown daily flows as negatives."""
    if not np.isfinite(reference) or reference <= 0 or discharge.index.has_duplicates or not discharge.index.is_monotonic_increasing:
        raise ValueError('Invalid daily discharge or reference')
    dates = pd.date_range(discharge.index.min(), discharge.index.max())
    q = discharge.reindex(dates)
    below = q.le(reference) & q.notna()
    quiet = below.rolling(7).sum().shift(3).eq(7)
    complete = q.notna().rolling(10).sum().shift(-3).eq(10)
    start = q.gt(reference) & below.shift(1).rolling(7).sum().eq(7)
    result = frame.copy()
    target = pd.DataFrame(index=dates)
    event = pd.Series('', index=dates)
    for lead in (3, 2, 1):
        mask = start.shift(-lead, fill_value=False)
        event.loc[mask] = (dates[mask] + pd.Timedelta(days=lead)).strftime('%Y-%m-%d')
        target[f'flow_{lead}'] = np.log1p(q.shift(-lead) / reference)
    result['event_date'] = event.reindex(result.index)
    result['exceedance'] = result.event_date.ne('').astype(int)
    result['onset'] = result.exceedance.astype(bool)
    for column in target:
        result[column] = target[column].reindex(result.index)
    result['high_water'] = result[list(target)].gt(np.log(2)).any(axis=1).astype(int)
    result['event_lead_hours'] = np.nan
    positive = result.exceedance.eq(1)
    local = pd.DatetimeIndex(result.loc[positive, 'event_date']).tz_localize(timezone).tz_convert('UTC').tz_localize(None)
    result.loc[positive, 'event_lead_hours'] = (local - result.index[positive]).total_seconds() / 3600
    return result.loc[quiet.reindex(result.index, fill_value=False) & complete.reindex(result.index, fill_value=False)]


def verify_protocol():
    frozen = json.loads((OUT / 'protocol.json').read_text())
    paths = {frozen['document']: frozen['document_sha256'],
             str((OUT / 'stations.json').relative_to(ROOT)): frozen['stations_sha256'],
             **frozen['preserved'], **frozen['reused_code']}
    for path, checksum in paths.items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen input changed: ' + path)
    for name in ['protocol', 'selection']:
        if digest(PREVIOUS / f'{name}.json') != frozen[f'previous_{name}_sha256']:
            raise ValueError('Previous experiment changed: ' + name)
    return frozen


def partitions(frame):
    if frame.role.eq('fresh_test').any() or frame.huc.str[:2].isin(['01', '02', '04']).any():
        raise ValueError('Fresh geography cannot enter selection')
    if not frame.role.isin(['development', 'geographic_validation']).all():
        raise ValueError('Unknown selection role')
    if not frame.role.eq('geographic_validation').equals(frame.huc.str[:2].isin(['03', '12'])):
        raise ValueError('Invalid geographic selection assignment')
    d = frame.decision_date
    middle = d.str[5:].between('01-08', '12-24')
    dev = frame.role.eq('development') & middle
    return dict(train=frame[dev & d.between('2017-01-08', '2020-12-24')],
        calibration=frame[dev & d.between('2021-01-08', '2021-12-24')],
        validation=frame[dev & d.between('2022-01-08', '2022-12-24')],
        geographic_validation=frame[frame.role.eq('geographic_validation') & middle & d.str.startswith('2022')])


def adaptive_alerts(frame, probabilities, source_negatives):
    """Weighted negative-score tail with station-local feedback delayed eight days."""
    p = np.asarray(probabilities)
    source = np.sort(np.asarray(source_negatives))
    if not len(source) or not np.isfinite(source).all() or len(p) != len(frame):
        raise ValueError('Invalid adaptive probability population')
    if frame.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate adaptive decisions')
    thresholds = np.empty(len(frame))
    work = frame.reset_index(drop=True).assign(_date=pd.to_datetime(frame.decision_date).to_numpy())
    for _, group in work.groupby('station'):
        group = group.sort_values('_date')
        positions = group.index.to_numpy()
        days = group._date.to_numpy()
        negatives = group.exceedance.to_numpy() == 0
        for pos, day in zip(positions, days):
            known = negatives & (days <= day - np.timedelta64(8, 'D')) & (days >= day - np.timedelta64(365, 'D'))
            local = p[positions[known]]
            scores = np.concatenate([source, local])
            weights = np.concatenate([np.full(len(source), 100 / len(source)), np.ones(len(local))])
            order = np.argsort(scores, kind='stable')
            rank = np.searchsorted(np.cumsum(weights[order]), .99 * weights.sum(), side='left')
            thresholds[pos] = scores[order[min(rank, len(order) - 1)]]
    return p > thresholds, thresholds


def alert_metrics(frame, alerts):
    y = frame.exceedance.to_numpy(bool)
    alerts = np.asarray(alerts, dtype=bool)
    if not len(frame) or len(alerts) != len(frame):
        raise ValueError('Invalid alert population')
    events = frame.loc[y, ['station', 'event_date', 'event_lead_hours']].copy()
    events['alert'] = alerts[y]
    hits = events[events.alert].groupby(['station', 'event_date']).event_lead_hours.max()
    count = len(events.drop_duplicates(['station', 'event_date']))
    tp = int(np.sum(alerts & y)); fp = int(np.sum(alerts & ~y))
    return dict(alerts=int(alerts.sum()), true_alerts=tp, false_alerts=fp,
        precision=tp / max(1, int(alerts.sum())), decision_recall=tp / max(1, int(y.sum())),
        events=count, event_recall=len(hits) / count if count else None,
        false_alerts_per_100=100 * fp / len(frame),
        median_event_lead_hours=float(hits.median()) if len(hits) else None,
        min_event_lead_hours=float(hits.min()) if len(hits) else None)


def collect():
    from concurrent.futures import ThreadPoolExecutor
    from numcodecs import Blosc
    verify_protocol()
    if (OUT / 'provenance.json').exists():
        raise FileExistsError('Dataset already frozen')
    used = {}
    def cached(relative, url):
        previous = PREVIOUS / relative
        path = previous if previous.exists() else OUT / relative
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
    indices = {s['station']: int(np.flatnonzero(coords['basin'] == 'camels_' + s['station'])[0]) for s in stations}
    weather = {sid: {} for sid in indices}
    for variable in ['hres_total_precipitation', 'hres_temperature_2m']:
        spec = metadata[variable + '/.zarray']
        if spec['chunks'] != [128, len(dates), 10] or spec['dtype'] != '<f4':
            raise ValueError('Unexpected weather layout')
        for chunk in sorted({i // 128 for i in indices.values()}):
            raw = cached(f'sources/{variable}-{chunk}.bin', ZARR + f'{variable}/{chunk}.0.0')
            values = np.frombuffer(Blosc().decode(raw), dtype='<f4').reshape(spec['chunks'])
            for sid, i in indices.items():
                if i // 128 == chunk:
                    weather[sid][variable] = values[i % 128].copy()
        print(variable, 'cache verified', flush=True)
    def observation(station):
        sid = station['station']
        url = ('https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=' + sid +
               '&startDT=2000-01-01&endDT=2024-09-30&parameterCd=00060&statCd=00003')
        return sid, cached(f'sources/{sid}.rdb', url).decode()
    with ThreadPoolExecutor(max_workers=2) as pool:
        observations = dict(pool.map(observation, stations))
    rows = {'training': [], 'test': []}; coverage = []
    for station in stations:
        sid = station['station']
        record = dict(station=sid, role=station['role'], huc=station['huc'])
        try:
            q = daily_discharge(observations[sid], sid)
        except IneligibleGauge as error:
            record['excluded'] = str(error); coverage.append(record); continue
        q.index = pd.to_datetime(q.index)
        reference = q[q.index < '2016-01-01']
        if len(reference) < 3000 or reference.quantile(.95) <= 0:
            record['excluded'] = 'insufficient_positive_reference'; coverage.append(record); continue
        threshold = float(reference.quantile(.95))
        rain, temp = [weather[sid][key] for key in ['hres_total_precipitation', 'hres_temperature_2m']]
        if np.nanmedian(temp[:366]) > 100:
            raise ValueError('Unexpected temperature units')
        frame = basin_features(dates, rain, temp, q, threshold, station, 1)
        frame = onset_rows(frame, q, threshold, station['timezone'])
        if station['role'] == 'fresh_test':
            frame = frame[frame.decision_date.between('2023-01-08', '2024-09-25')]; split = 'test'
        else:
            frame = frame[frame.decision_date.between('2017-01-08', '2022-12-24') & frame.decision_date.str[5:].between('01-08', '12-24')]; split = 'training'
        rows[split].append(frame)
        record.update(reference_cfs=threshold, reference_days=len(reference), rows=len(frame)); coverage.append(record)
    datasets = {}
    for split, pieces in rows.items():
        frame = pd.concat(pieces, ignore_index=True).sort_values(['decision_date', 'station'])
        if frame.duplicated(['station', 'decision_date']).any():
            raise ValueError('Duplicate decision')
        path = OUT / f'{split}.csv.gz'
        frame.to_csv(path, index=False, compression=dict(method='gzip', mtime=0))
        datasets[split] = dict(sha256=digest(path), rows=len(frame), gauges=int(frame.station.nunique()))
    source_bytes = sum(p.stat().st_size for p in (OUT / 'sources').glob('*') if not p.name.endswith('.json'))
    if source_bytes > 300_000_000:
        raise ValueError('Source expansion exceeds budget')
    write_json(OUT / 'provenance.json', dict(datasets=datasets, coverage=coverage, sources=used,
        new_source_bytes=source_bytes, protocol_sha256=digest(OUT / 'protocol.json')))
    print('Frozen data populations', datasets, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect'])
    args = parser.parse_args()
    collect()
