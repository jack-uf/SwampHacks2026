"""Controlled subdaily information experiments. No application or production-model imports."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import time
from datetime import datetime, timezone

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.flood_sprint import write_json
from scripts.high_flow_research import load, split, threshold, LATEST
from scripts.hydrology_research import ROOT, digest
from scripts.shift_hydrology import FEATURES
from scripts.shift_hydrology_metrics import metrics

OUT = ROOT/'experiments/high-flow-signals-20261018'
VARIABLES = ['precipitation', 'temperature_2m', 'soil_moisture_0_to_7cm', 'snow_depth']
ATTRIBUTES = ['p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq', 'high_prec_dur',
    'low_prec_freq', 'low_prec_dur', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
    'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
    'frac_forest', 'lai_max', 'gvf_max', 'geol_porostiy', 'geol_permeability']


def fetch(path, url, params=None):
    certificate = path.with_suffix(path.suffix+'.request.json')
    expected = str(httpx.URL(url, params=params))
    if path.exists():
        meta = json.loads(certificate.read_text())
        if digest(path) != meta['sha256'] or meta['url'] != expected:
            raise ValueError('Cached source changed: '+str(path))
        return gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    response = httpx.get(expected, timeout=90, follow_redirects=True)
    if response.status_code == 429:
        with (OUT/'acquisition-events.jsonl').open('a') as log:
            log.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), url=expected,
                status=429, response=response.text[:1000]))+'\n')
    response.raise_for_status()
    body = response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(body, mtime=0) if path.suffix == '.gz' else body)
    write_json(certificate, dict(url=expected, received_at=datetime.now(timezone.utc).isoformat(),
        sha256=digest(path), response_sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)))
    return body


def acquire():
    frame = load(True)
    stations = frame[['station', 'latitude', 'longitude']].drop_duplicates().to_dict('records')
    for category in ['clim', 'soil', 'vege', 'geol']:
        name = f'camels_{category}.txt'
        fetch(OUT/'sources'/name, 'https://zenodo.org/api/records/15529996/files/'+name+'/content')
    def one(station):
        path = OUT/'weather'/(station['station']+'.json.gz')
        params = dict(latitude=station['latitude'], longitude=station['longitude'],
            start_date='2019-12-01', end_date='2024-12-31', hourly=','.join(VARIABLES), models='era5', timezone='GMT')
        x = json.loads(fetch(path, 'https://archive-api.open-meteo.com/v1/archive', params))
        if x.get('error') or not all(v in x.get('hourly', {}) for v in VARIABLES):
            raise ValueError('Missing weather variables')
        print('Acquired', station['station'], len(x['hourly']['time']), 'hourly weather rows', flush=True)
    for station in stations:
        path = OUT/'weather'/(station['station']+'.json.gz')
        if not path.exists():
            time.sleep(16)
        one(station)


def meteorology(hourly, delay):
    """Weather at decision minus delay; a causal time shift does not prove reanalysis publication."""
    f = pd.DataFrame(hourly)
    f.index = pd.to_datetime(f.pop('time'), utc=True)
    if f.index.has_duplicates or not f.index.equals(pd.date_range(f.index[0], f.index[-1], freq='h')):
        raise ValueError('Weather must use a complete unique UTC hourly calendar')
    if delay < 1 or not np.isfinite(f[VARIABLES].to_numpy(float)).all() or (f.precipitation < 0).any():
        raise ValueError('Invalid weather values or delay')
    result = pd.DataFrame(index=f.index)
    for hours in [3, 6, 12, 24, 48, 72, 168]:
        result[f'rain_{hours}_lag{delay}'] = f.precipitation.rolling(hours).sum().shift(delay)
    result[f'temperature_lag{delay}'] = f.temperature_2m.shift(delay)
    result[f'soil_lag{delay}'] = f.soil_moisture_0_to_7cm.shift(delay)
    result[f'snow_lag{delay}'] = f.snow_depth.shift(delay)
    result[f'positive_degree_24_lag{delay}'] = f.temperature_2m.clip(lower=0).rolling(24).sum().shift(delay)
    result[f'rain_soil_lag{delay}'] = result[f'rain_24_lag{delay}'] * result[f'soil_lag{delay}']
    return result


def catchment_attributes(frame):
    attributes = []
    for category in ['clim', 'soil', 'vege', 'geol']:
        p = OUT/'sources'/f'camels_{category}.txt'
        if digest(p) != json.loads(p.with_suffix(p.suffix+'.request.json').read_text())['sha256']:
            raise ValueError('Attribute checksum mismatch')
        a = pd.read_csv(p, sep=';', dtype={'gauge_id': str})
        attributes.append(a[['gauge_id']+[c for c in ATTRIBUTES if c in a]])
    a = attributes[0]
    for part in attributes[1:]:
        a = a.merge(part, on='gauge_id', validate='one_to_one')
    if set(ATTRIBUTES)-set(a):
        raise ValueError('Missing declared attributes')
    frame = frame.merge(a.rename(columns={'gauge_id': 'station'}), on='station', validate='many_to_one', how='left')
    if frame[ATTRIBUTES].isna().all(axis=1).any():
        raise ValueError('Missing catchment attribute match')
    frame[ATTRIBUTES] = frame[ATTRIBUTES].apply(pd.to_numeric, errors='coerce').replace([-999, -9999], np.nan)
    return frame


def materialize():
    destination = OUT/'development.csv.gz'
    if destination.exists():
        raise FileExistsError(destination)
    frame = catchment_attributes(load(True))
    pieces, sources = [], {}
    for station, group in frame.groupby('station', sort=True):
        path = OUT/'weather'/(station+'.json.gz')
        cert = json.loads(path.with_suffix(path.suffix+'.request.json').read_text())
        if digest(path) != cert['sha256']:
            raise ValueError('Weather checksum mismatch')
        sources[str(path.relative_to(ROOT))] = digest(path)
        hourly = json.loads(gzip.decompress(path.read_bytes()))['hourly']
        weather = pd.concat([meteorology(hourly, delay) for delay in [6, 24, 144]], axis=1)
        t = pd.to_datetime(group.decision_time, utc=True)
        aligned = weather.reindex(t)
        aligned.index = group.index
        pieces.append(pd.concat([group, aligned], axis=1))
    result = pd.concat(pieces, ignore_index=True)
    if result.filter(regex='lag').isna().any().any():
        raise ValueError('Weather history does not cover the shared population')
    result.to_csv(destination, index=False, compression={'method': 'gzip', 'mtime': 0})
    write_json(OUT/'dataset.json', dict(sha256=digest(destination), rows=len(result),
        gauges=result.station.nunique(), weather_sources=sources,
        attribute_sources={str(p.relative_to(ROOT)): digest(p) for p in (OUT/'sources').glob('*.txt')},
        evidence='ERA5 reanalysis. Delay variants are retrospective information experiments, not as-issued inputs.'))


def data():
    path = OUT/'development.csv.gz'
    if digest(path) != json.loads((OUT/'dataset.json').read_text())['sha256']:
        raise ValueError('Signal dataset changed')
    f = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
    # Physical missing values in static tables remain missing, never fit on held-out geography.
    f[ATTRIBUTES] = f[ATTRIBUTES].apply(pd.to_numeric, errors='coerce').replace([-999, -9999], np.nan)
    return f


def fit(cycle):
    from lightgbm import LGBMClassifier
    frame = catchment_attributes(load(True)) if cycle == 0 else data()
    reserved = {s['huc'][:4] for s in json.loads((LATEST/'stations.json').read_text()) if s['role']=='geographic_test'}
    parts = split(frame, reserved, cycle=4)
    weather = {d: [c for c in frame if c.endswith(f'_lag{d}')] for d in [6, 24, 144]}
    if cycle == 0:
        configs = dict(control=FEATURES, attributes=FEATURES+ATTRIBUTES,
            climate=FEATURES+ATTRIBUTES[:8], storage=FEATURES+ATTRIBUTES[8:])
    elif cycle == 1:
        configs = dict(control=FEATURES, attributes=FEATURES+ATTRIBUTES,
            weather6=FEATURES+weather[6], weather24=FEATURES+weather[24],
            weather144=FEATURES+weather[144], combined6=FEATURES+ATTRIBUTES+weather[6],
            combined24=FEATURES+ATTRIBUTES+weather[24])
    else:
        raise ValueError('Unknown experiment cycle')
    out = OUT/f'cycle-{cycle}'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    write_json(out/'freeze.json', dict(dataset_sha256=digest(OUT/'development.csv.gz') if cycle else None, configurations=configs,
        code_sha256=digest(ROOT/'scripts/high_flow_signals.py'), reserved_huc4=sorted(reserved),
        split='Train 2020-2021; calibrate 2022; threshold 2023; score 2024. Inspected development only.',
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'models').glob('*') if p.is_file()}))
    results = []
    for name, features in configs.items():
        started = time.monotonic()
        model = LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=15, max_depth=4,
            min_child_samples=80, reg_lambda=5, n_jobs=2, random_state=42, verbosity=-1, deterministic=True)
        model.fit(parts['train'][features], parts['train'].target)
        def raw(f):
            p = np.clip(model.predict_proba(f[features])[:, 1], 1e-6, 1-1e-6)
            return pd.DataFrame({'score': np.log(p/(1-p))})
        calibration = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
        calibration.fit(raw(parts['calibration']), parts['calibration'].target)
        if calibration[-1].coef_[0, 0] <= 0:
            raise ValueError('Nonpositive calibration slope')
        cut = threshold(parts['threshold'], calibration.predict_proba(raw(parts['threshold']))[:, 1], 1., 2.)
        bundle = dict(model=model, calibrator=calibration, features=features, threshold=cut)
        joblib.dump(bundle, out/(name+'.joblib'), compress=3)
        scores = {}
        for key in ['temporal_validation', 'geographic_validation']:
            f = parts[key]
            p = calibration.predict_proba(raw(f))[:, 1]
            pred = f[['station','huc','decision_time','horizon','target','event_time']].copy()
            pred['probability'] = p
            pred.to_csv(out/(name+'-'+key+'.csv.gz'), index=False, compression={'method':'gzip','mtime':0})
            scores[key] = metrics(pred, p, cut)
        record = dict(name=name, threshold=cut, artifact_sha256=digest(out/(name+'.joblib')), seconds=time.monotonic()-started, metrics=scores)
        results.append(record)
        print(cycle, name, {k:(v['detected_events'],round(v['ap'],4),round(v['false_alerts_per_100'],3)) for k,v in scores.items()}, flush=True)
    write_json(out/'results.json', results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['acquire','materialize','fit'])
    p.add_argument('--cycle', type=int, default=1)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'acquire': acquire()
    elif args.command == 'materialize': materialize()
    else: fit(args.cycle)
