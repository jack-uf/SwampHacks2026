"""Frozen development-only upstream transport pilot; no geographic test outcomes."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from scripts.hydrology_research import (ROOT, HYDRO, COMBINED, SNOW, daily_discharge,
                                        digest, fetch, write_json, budget_threshold)
from scripts.onset_worker import metrics, raw_score, predict
from scripts.upstream_research import OUT

RAW = ['up_3', 'up_4', 'up_7', 'up_change', 'area_ratio']
TRANSPORT = ['transport_1', 'transport_2', 'transport_3']


def transport_features(down, up, issues):
    """Reference-only local/augmented trajectories evaluated with delayed daily flows."""
    for q in [down, up]:
        if q.index.has_duplicates or not q.index.is_monotonic_increasing or (q.dropna() < 0).any() or not np.isfinite(q.dropna()).all():
            raise ValueError('Invalid flow history')
    dates = pd.date_range(min(down.index.min(), up.index.min()), max(down.index.max(), up.index.max()))
    refs = []
    for q in [down, up]:
        reference = q.loc[q.index < '2016-01-01'].dropna()
        if len(reference) < 3000 or reference.quantile(.95) <= 0:
            raise ValueError('Insufficient reference flow')
        refs.append(float(reference.quantile(.95)))
    d, u = [np.log1p(q.reindex(dates) / ref) for q, ref in zip([down, up], refs)]
    phase = 2 * np.pi * dates.dayofyear / 365.25
    inputs = pd.DataFrame({'d0': d, 'd1': d.shift(1), 'd4': d.shift(4),
                           'sin': np.sin(phase), 'cos': np.cos(phase),
                           'u0': u, 'u1': u.shift(1), 'u4': u.shift(4)}, index=dates)
    local = ['d0', 'd1', 'd4', 'sin', 'cos']
    observed = pd.DatetimeIndex(issues) - pd.Timedelta(days=3)
    usable = inputs.reindex(observed).notna().all(axis=1).to_numpy()
    available = inputs.reindex(observed).loc[usable]
    result = pd.DataFrame(index=issues, columns=['up_3', 'up_4', 'up_7', 'up_change'] + TRANSPORT, dtype=float)
    result.loc[usable, ['up_3', 'up_4', 'up_7']] = available[['u0', 'u1', 'u4']].to_numpy()
    result.loc[usable, 'up_change'] = (available.u0 - available.u1).to_numpy()
    fitted = dict(references=refs, leads={})
    for lead in [1, 2, 3]:
        horizon = lead + 3
        target = d.shift(-horizon)
        mask = inputs.notna().all(axis=1) & target.notna() & (dates + pd.Timedelta(days=horizon) < pd.Timestamp('2016-01-01'))
        if mask.sum() < 2000:
            raise ValueError('Insufficient complete reference trajectories')
        models = []
        for columns in [local, list(inputs)]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(inputs.loc[mask, columns], target.loc[mask])
            models.append(model)
        if len(available):
            result.loc[usable, f'transport_{lead}'] = models[1].predict(available) - models[0].predict(available[local])
        fitted['leads'][lead] = dict(rows=int(mask.sum()), local=models[0], augmented=models[1])
    return result, fitted


def verify_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in protocol['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen pilot input changed: ' + path)
    return protocol


def collect():
    verify_protocol()
    if (OUT / 'provenance.json').exists():
        raise FileExistsError('Pilot data already collected')
    onset = ROOT / 'experiments/sprint-20261017-onset'
    previous = json.loads((onset / 'provenance.json').read_text())
    frame = pd.read_csv(onset / 'training.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
    pairs = [p for p in json.loads((OUT / 'expanded-pairs.json').read_text())['pairs'] if p['role'] == 'development']
    if len(pairs) != 4:
        raise ValueError('Frozen pilot requires exactly four development pairs')
    pieces, coverage, sources, transport = [], [], {}, {}
    for pair in pairs:
        sid, uid = pair['station'], pair['upstream']
        path = next(ROOT / p for p in previous['sources'] if p.endswith('/' + sid + '.rdb'))
        record = previous['sources'][str(path.relative_to(ROOT))]
        if digest(path) != record['sha256']:
            raise ValueError('Prior discharge changed')
        sources[str(path.relative_to(ROOT))] = record
        target = OUT / 'sources' / f'{uid}.rdb'
        url = f'https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={uid}&startDT=2000-01-01&endDT=2022-12-31&parameterCd=00060&statCd=00003'
        fetch(target, url)
        sources[str(target.relative_to(ROOT))] = json.loads(target.with_suffix('.rdb.request.json').read_text())
        q, u = [daily_discharge(p.read_text(), identifier) for p, identifier in [(path, sid), (target, uid)]]
        q.index = pd.to_datetime(q.index); u.index = pd.to_datetime(u.index)
        f = frame[frame.station.eq(sid)].copy()
        features, transport[sid] = transport_features(q, u, pd.DatetimeIndex(f.decision_date))
        f[features.columns] = features.to_numpy()
        f['area_ratio'] = pair['area_ratio']
        complete = f[COMBINED + SNOW + RAW + TRANSPORT].notna().all(axis=1)
        f = f.loc[complete].copy()
        if not np.isfinite(f[COMBINED + SNOW + RAW + TRANSPORT]).all(axis=None):
            raise ValueError('Nonfinite pilot features')
        coverage.append(dict(station=sid, upstream=uid, original_rows=len(features), eligible_rows=len(f),
                             upstream_approved_days=len(u), references=transport[sid]['references']))
        pieces.append(f)
    combined = pd.concat(pieces).sort_values(['decision_date', 'station']).reset_index(drop=True)
    if combined.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate pilot decision')
    combined.to_csv(OUT / 'development.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    (OUT / 'artifacts').mkdir(exist_ok=True)
    joblib.dump(transport, OUT / 'artifacts/transport.joblib', compress=3)
    write_json(OUT / 'provenance.json', dict(coverage=coverage, sources=sources,
        data_sha256=digest(OUT / 'development.csv.gz'), transport_sha256=digest(OUT / 'artifacts/transport.joblib'),
        protocol_sha256=digest(OUT / 'protocol.json'), code_sha256=digest(ROOT / 'scripts/upstream_pilot.py')))
    print('Collected development pairs only:', len(pairs), 'eligible decisions:', len(combined), flush=True)


def evaluate():
    from lightgbm import LGBMClassifier
    verify_protocol()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Development evaluation already exists')
    started = time.monotonic()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if digest(OUT / 'development.csv.gz') != provenance['data_sha256']:
        raise ValueError('Pilot table changed')
    frame = pd.read_csv(OUT / 'development.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
    middle = frame.decision_date.str[5:].between('01-08', '12-24')
    parts = {name: frame[middle & frame.decision_date.between(start, end)].copy().reset_index(drop=True)
             for name, start, end in [('train', '2017-01-01', '2020-12-31'), ('calibration', '2021-01-01', '2021-12-31'),
                                      ('development', '2022-01-01', '2022-12-31')]}
    train, cal, test = [parts[k] for k in ['train', 'calibration', 'development']]
    for part in parts.values():
        if part.station.nunique() != 4 or part.exceedance.nunique() != 2:
            raise ValueError('Insufficient complete development population')
    configs = dict(rainfall=[], hydrology=HYDRO, logistic=COMBINED + SNOW, combined=COMBINED + SNOW,
                   upstream=COMBINED + SNOW + RAW, innovation=COMBINED + SNOW + TRANSPORT)
    records = []
    (OUT / 'predictions').mkdir(exist_ok=True)
    for name, features in configs.items():
        begin = time.monotonic(); bundle = dict(name=name, features=features)
        if features:
            model = (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
                     if name == 'logistic' else LGBMClassifier(n_estimators=200, learning_rate=.04,
                     max_depth=3, num_leaves=7, n_jobs=2, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True))
            bundle['model'] = model.fit(train[features], train.exceedance)
        bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(raw_score(bundle, cal), cal.exceedance)
        cp = predict(bundle, cal)
        thresholds = {f'pooled_{budget}': budget_threshold(cal.exceedance, cp, budget / 100) for budget in [.5, 1, 2]}
        bundle['thresholds'] = thresholds
        p = predict(bundle, test); alerts = {k: p >= v for k, v in thresholds.items()}
        artifact = OUT / 'artifacts' / f'{name}.joblib'; joblib.dump(bundle, artifact, compress=3)
        test[['station', 'huc', 'decision_date', 'event_date', 'exceedance', 'event_lead_hours']].assign(probability=p,
            **alerts).to_csv(OUT / 'predictions' / f'{name}.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        record = dict(name=name, overall=metrics(test, p, alerts), thresholds=thresholds,
            uncalibrated_ap=float(average_precision_score(test.exceedance, raw_score(bundle, test).ravel())),
            by_station={sid: metrics(g, p[g.index], {k: a[g.index] for k, a in alerts.items()}) for sid, g in test.groupby('station')},
            seconds=time.monotonic() - begin, artifact_sha256=digest(artifact),
            prediction_sha256=digest(OUT / 'predictions' / f'{name}.csv.gz'))
        records.append(record)
        print(name, 'development AP', round(record['overall']['ap'], 4), flush=True)
    write_json(OUT / 'evaluation.json', dict(scope='exploratory development only; no geographic test', candidates=records,
        partitions={k: dict(rows=len(f), positives=int(f.exceedance.sum()), events=len(f[f.exceedance.eq(1)].drop_duplicates(['station', 'event_date'])))
                    for k, f in parts.items()}, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'joblib']},
        provenance_sha256=digest(OUT / 'provenance.json'), code_sha256=digest(ROOT / 'scripts/upstream_pilot.py')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'evaluate'])
    args = parser.parse_args()
    {'collect': collect, 'evaluate': evaluate}[args.command]()
