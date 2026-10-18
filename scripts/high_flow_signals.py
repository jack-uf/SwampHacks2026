"""Controlled subdaily information experiments. No application or production-model imports."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.flood_sprint import write_json
from scripts.high_flow_research import HYDRO, load, split, threshold, LATEST
from scripts.hydrology_research import ROOT, digest
from scripts.mlp_hybrid_worker import physical_score
from scripts.shift_hydrology import FEATURES
from scripts.shift_hydrology_metrics import metrics

OUT = ROOT/'experiments/high-flow-signals-20261018'
# The archive request keeps snow_depth because the cached evidence was acquired with it.
VARIABLES = ['precipitation', 'temperature_2m', 'soil_moisture_0_to_7cm', 'snow_depth']
# The ERA5 archive responses carry snow_depth as null in every hour; only these three are usable.
USABLE = ['precipitation', 'temperature_2m', 'soil_moisture_0_to_7cm']
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
    if f.snow_depth.notna().any():
        raise ValueError('Snow depth became available; the canonical variable set must change explicitly')
    if delay < 1 or not np.isfinite(f[USABLE].to_numpy(float)).all() or (f.precipitation < 0).any():
        raise ValueError('Invalid weather values or delay')
    result = pd.DataFrame(index=f.index)
    for hours in [3, 6, 12, 24, 48, 72, 168]:
        result[f'rain_{hours}_lag{delay}'] = f.precipitation.rolling(hours).sum().shift(delay)
    result[f'temperature_lag{delay}'] = f.temperature_2m.shift(delay)
    result[f'soil_lag{delay}'] = f.soil_moisture_0_to_7cm.shift(delay)
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
        weather_variables_used=USABLE,
        weather_variable_notes={'snow_depth': 'Requested in every cached archive response and null in every hour; not used.'},
        evidence='ERA5 reanalysis. Delay variants are retrospective information experiments, not as-issued inputs.'))


def data():
    path = OUT/'development.csv.gz'
    if digest(path) != json.loads((OUT/'dataset.json').read_text())['sha256']:
        raise ValueError('Signal dataset changed')
    f = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
    # Physical missing values in static tables remain missing, never fit on held-out geography.
    f[ATTRIBUTES] = f[ATTRIBUTES].apply(pd.to_numeric, errors='coerce').replace([-999, -9999], np.nan)
    return f


def network_scores(name, configuration, features, fitted, parts, out):
    """Score one neural family in its own process; torch beside LightGBM faults in this environment."""
    inner = fitted['train'].sort_values('decision_time').reset_index(drop=True)
    cut = int(len(inner)*.9)
    frames = dict(inner_train=inner.iloc[:cut], inner_validation=inner.iloc[cut:],
        calibration=fitted['calibration'], threshold=fitted['threshold'],
        temporal_validation=parts['temporal_validation'], geographic_validation=parts['geographic_validation'])
    columns = ['station', 'decision_time', 'target']+features
    table = pd.concat([frame[columns].assign(partition=key) for key, frame in frames.items()], ignore_index=True)
    job = dict(name=name, features=features, seed=configuration.get('seed', 42),
        epochs=configuration.get('epochs', 80), patience=configuration.get('patience', 8),
        input=str(Path(tempfile.gettempdir())/(name+'-network-input.csv.gz')),
        output=str(out/(name+'-logits.csv.gz')),
        bundle=str(out/(name+'-network.pt')))
    table.to_csv(job['input'], index=False, compression={'method': 'gzip', 'mtime': 0})
    job_path = Path(job['input']).with_suffix('.job.json')
    job_path.write_text(json.dumps(job))
    environment = dict(os.environ, OMP_NUM_THREADS='4', KMP_DUPLICATE_LIB_OK='TRUE', MPLCONFIGDIR='/tmp')
    result = subprocess.run([sys.executable, '-m', 'scripts.high_flow_signals_worker', str(job_path)],
        cwd=ROOT, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise ValueError('Network worker failed: '+result.stderr[-600:])
    print(result.stdout.strip(), flush=True)
    return pd.read_csv(job['output'], dtype={'station': str}, keep_default_na=False)


def fit(cycle):
    from lightgbm import LGBMClassifier
    frame = population(cycle)
    reserved = {s['huc'][:4] for s in json.loads((LATEST/'stations.json').read_text()) if s['role']=='geographic_test'}
    parts = split(frame, reserved, cycle=4)
    weather = {d: [c for c in frame if c.endswith(f'_lag{d}')] for d in [6, 24, 144]}
    configs = configurations(cycle, weather)
    out = OUT/f'cycle-{cycle}'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    write_json(out/'freeze.json', dict(dataset_sha256=digest(OUT/'development.csv.gz') if cycle in [1,3,4,5,6,7,8,9,10,11,12,13,14,15] else None, configurations=configs,
        forecast_dataset_sha256=digest(OUT/'hrrr/features.csv.gz') if cycle in [2,3,4,5,6,7,8,9,10,11,12,13,14,15] else None,
        code_sha256=digest(ROOT/'scripts/high_flow_signals.py'), reserved_huc4=sorted(reserved),
        split='Train 2020-2021; calibrate 2022; threshold 2023; score 2024. Inspected development only.',
        matched_training='Configurations with hours restrict model, calibrator and threshold selection to those UTC decision hours.',
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'models').glob('*') if p.is_file()}))
    results = []
    for name, configuration in configs.items():
        results.append(fit_candidate(name, configuration, parts, out, cycle=cycle))
    write_json(out/'results.json', results)


def population(cycle):
    """Feature frame for one cycle, built before partitioning so derived columns exist everywhere."""
    frame = catchment_attributes(load(True)) if cycle in [0,2] else data()
    if cycle in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        path=OUT/'hrrr/features.csv.gz'
        if digest(path)!=json.loads((OUT/'hrrr/features.json').read_text())['sha256']:
            raise ValueError('Forecast features changed')
        rain=pd.read_csv(path,dtype={'station':str})
        frame=frame.merge(rain,on=['station','decision_time'],how='left',validate='one_to_one')
        frame['hrrr_available']=frame.hrrr_point12.notna().astype(int)
    if cycle in [3,4,5,6,7,8,9,10,11,12,13,14,15]:
        # The CAMELS long-term precipitation mean is prior knowledge, available for every gauge, and
        # turns raw depths into "unusual for this watershed" ratios that transfer across climates.
        expected = frame.p_mean
        if expected.isna().any() or (expected <= 0).any():
            raise ValueError('Invalid catchment precipitation climatology')
        frame['hrrr_point_ratio'] = frame.hrrr_point12/expected
        frame['hrrr_basin_ratio'] = frame.hrrr_basin12/expected
        frame['hrrr_max_ratio'] = frame.hrrr_max12/expected
        frame['rain24_ratio'] = frame.rain_24_lag6/(2*expected)
        frame['rain168_ratio'] = frame.rain_168_lag6/(7*expected)
    if cycle in [8,9,10,11,12,13,14,15]:
        # Saturating catchments convert the same rain into more runoff; both additions are causal.
        frame['soil_change_6d'] = frame.soil_lag6 - frame.soil_lag144
        frame['rain_share_6h'] = frame.rain_6_lag6/(frame.rain_168_lag6 + 1e-3)
    if cycle == 10:
        # Rain totals are strongly skewed and runoff depends on how much rain the catchment already
        # absorbed, so the last candidates test log and evaporative-demand scaling.
        frame['log_rain24'] = np.log1p(frame.rain_24_lag6)
        frame['log_rain168'] = np.log1p(frame.rain_168_lag6)
        frame['rain_on_wet_soil'] = frame.rain_6_lag6*frame.soil_lag6
        frame['rain_demand_ratio'] = frame.rain24_ratio/frame.aridity.replace(0, np.nan)
    return frame


def configurations(cycle, weather):
    if cycle == 0:
        configs = dict(control=FEATURES, attributes=FEATURES+ATTRIBUTES,
            climate=FEATURES+ATTRIBUTES[:8], storage=FEATURES+ATTRIBUTES[8:])
    elif cycle == 1:
        configs = dict(control=FEATURES, attributes=FEATURES+ATTRIBUTES,
            weather6=FEATURES+weather[6], weather24=FEATURES+weather[24],
            weather144=FEATURES+weather[144], combined6=FEATURES+ATTRIBUTES+weather[6],
            combined24=FEATURES+ATTRIBUTES+weather[24])
    elif cycle == 2:
        configs=dict(control=FEATURES, availability=FEATURES+['hrrr_available'],
            forecast_point=FEATURES+['hrrr_available','hrrr_point12'],
            forecast_basin=FEATURES+['hrrr_available','hrrr_basin12','hrrr_max12'],
            forecast_both=FEATURES+['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12'],
            logistic_forecast=FEATURES+['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12'])
    elif cycle == 3:
        ratios = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio','rain24_ratio','rain168_ratio']
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        configs=dict(control=dict(features=FEATURES), rain6=dict(features=FEATURES+weather[6]),
            scaled=dict(features=FEATURES+ratios),
            forecast=dict(features=FEATURES+forecast),
            forecast_scaled=dict(features=FEATURES+forecast+ratios),
            combined=dict(features=FEATURES+ATTRIBUTES+weather[6]+forecast+ratios),
            control_matched=dict(features=FEATURES, hours=[12]),
            forecast_matched=dict(features=FEATURES+forecast+ratios, hours=[12]))
    elif cycle == 4:
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        configs=dict(control=dict(features=FEATURES),
            rain_ratio=dict(features=FEATURES+rain_ratio),
            rain6_ratio=dict(features=FEATURES+weather[6]+rain_ratio),
            hrrr_ratio=dict(features=FEATURES+['hrrr_available']+hrrr_ratio),
            hrrr_raw=dict(features=FEATURES+forecast),
            forecast_scaled=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio),
            slim_combined=dict(features=FEATURES+rain_ratio+forecast+hrrr_ratio),
            control_matched=dict(features=FEATURES, hours=[12]),
            forecast_matched=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio, hours=[12]))
    elif cycle == 5:
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        configs=dict(linear=dict(physical='linear'), damped=dict(physical='damped'),
            control=dict(features=FEATURES),
            rain6_ratio=dict(features=FEATURES+weather[6]+rain_ratio),
            hrrr_scaled=dict(features=FEATURES+forecast+hrrr_ratio),
            forecast_scaled=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio),
            combined_scaled=dict(features=FEATURES+ATTRIBUTES+weather[6]+forecast+rain_ratio+hrrr_ratio))
    elif cycle == 6:
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        configs=dict(damped_conditional=dict(physical='damped', conditional=True),
            linear_conditional=dict(physical='linear', conditional=True),
            control=dict(features=FEATURES), control_conditional=dict(features=FEATURES, conditional=True),
            rain6_ratio=dict(features=FEATURES+weather[6]+rain_ratio),
            rain6_ratio_conditional=dict(features=FEATURES+weather[6]+rain_ratio, conditional=True),
            forecast_scaled=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio),
            combined_scaled=dict(features=FEATURES+ATTRIBUTES+weather[6]+forecast+rain_ratio+hrrr_ratio))
    elif cycle == 7:
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            control_conditional=dict(features=FEATURES, conditional=True),
            rain6_ratio_conditional=dict(features=FEATURES+weather[6]+rain_ratio, conditional=True),
            forecast_scaled_conditional=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio, conditional=True),
            combined_scaled_conditional=dict(features=FEATURES+ATTRIBUTES+weather[6]+forecast+rain_ratio+hrrr_ratio,
                conditional=True))
    elif cycle == 8:
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        winner = FEATURES+weather[6]+rain_ratio
        wetness = ['soil_change_6d','rain_share_6h']
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            control_conditional=dict(features=FEATURES, conditional=True),
            rain6_ratio_conditional=dict(features=winner, conditional=True),
            rain6_ratio_wetness=dict(features=winner+wetness, conditional=True),
            rain6_ratio_hydro=dict(features=winner+HYDRO, conditional=True),
            rain6_ratio_wetness_hydro=dict(features=winner+wetness+HYDRO, conditional=True),
            forecast_scaled_wetness=dict(features=FEATURES+forecast+rain_ratio+hrrr_ratio+wetness, conditional=True))
    elif cycle == 9:
        # The full hydrological block contains the physical baselines' own margins. This ablation keeps
        # only decision-time rise-shape features so the rain contribution can be attributed.
        shape = ['rise_persistence', 'recent_fraction', 'relative_range', 'soil_change_6d', 'rain_share_6h']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            rain6_ratio_conditional=dict(features=FEATURES+weather[6]+rain_ratio, conditional=True),
            rain6_ratio_shape=dict(features=FEATURES+weather[6]+rain_ratio+shape, conditional=True),
            rain6_ratio_wetness_hydro=dict(features=FEATURES+weather[6]+rain_ratio+['soil_change_6d','rain_share_6h']+HYDRO,
                conditional=True))
    elif cycle == 10:
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            rain6_ratio_wetness_hydro=dict(features=winner, conditional=True),
            winner_lograin=dict(features=winner+['log_rain24','log_rain168'], conditional=True),
            winner_runoff=dict(features=winner+['rain_on_wet_soil','rain_demand_ratio'], conditional=True),
            winner_extended=dict(features=winner+['log_rain24','log_rain168','rain_on_wet_soil','rain_demand_ratio'],
                conditional=True))
    elif cycle == 11:
        # Operational question: does the as-issued forecast rain help the winning model, or does the
        # four-hourly-missing forecast column set only dilute it?
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            rain6_ratio_wetness_hydro=dict(features=winner, conditional=True),
            winner_forecast=dict(features=winner+forecast+hrrr_ratio, conditional=True),
            forecast_wetness_hydro=dict(features=FEATURES+forecast+hrrr_ratio+wetness+HYDRO, conditional=True))
    elif cycle == 12:
        # Seed stability of the frozen candidate; the frozen pipeline itself keeps seed 42.
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            winner_seed42=dict(features=winner+forecast+hrrr_ratio, conditional=True),
            winner_seed7=dict(features=winner+forecast+hrrr_ratio, conditional=True, seed=7),
            winner_seed11=dict(features=winner+forecast+hrrr_ratio, conditional=True, seed=11),
            control_seed7=dict(features=FEATURES, conditional=True, seed=7))
    elif cycle == 13:
        # Architecture comparison on identical rows, features, calibrator and threshold rule.
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            winner_forecast=dict(features=winner+forecast+hrrr_ratio, conditional=True),
            winner_forecast_mlp=dict(features=winner+forecast+hrrr_ratio, conditional=True, network=True),
            control_mlp=dict(features=FEATURES, conditional=True, network=True))
    elif cycle == 14:
        # Same configurations as cycle 11, but the forecast block now exists at every decision hour
        # instead of 12 UTC only, so the forecast family is tested on the whole population.
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict(linear_conditional=dict(physical='linear', conditional=True),
            control_conditional=dict(features=FEATURES, conditional=True),
            rain6_ratio_wetness_hydro=dict(features=winner, conditional=True),
            winner_forecast=dict(features=winner+forecast+hrrr_ratio, conditional=True),
            forecast_wetness_hydro=dict(features=FEATURES+forecast+hrrr_ratio+wetness+HYDRO, conditional=True))
    elif cycle == 15:
        # Full forecast coverage breaches the realized geographic ceiling at the one-per-hundred rule.
        # This cycle pairs each operating-point rule with a comparator selected under the same rule.
        forecast = ['hrrr_available','hrrr_point12','hrrr_basin12','hrrr_max12']
        hrrr_ratio = ['hrrr_point_ratio','hrrr_basin_ratio','hrrr_max_ratio']
        rain_ratio = ['rain24_ratio','rain168_ratio']
        wetness = ['soil_change_6d','rain_share_6h']
        winner = FEATURES+weather[6]+rain_ratio+wetness+HYDRO
        configs=dict()
        for label, budget in [('headroom', (1., 2.)), ('tight', (.75, 1.5)), ('tighter', (.5, 1.))]:
            configs['linear_'+label]=dict(physical='linear', conditional=True, budget=budget)
            configs['winner_'+label]=dict(features=winner+forecast+hrrr_ratio, conditional=True, budget=budget)
            configs['forecast_only_'+label]=dict(features=FEATURES+forecast+hrrr_ratio+wetness+HYDRO,
                conditional=True, budget=budget)
    else:
        raise ValueError('Unknown experiment cycle')
    return configs


def fit_candidate(name, configuration, parts, out, evaluation=None, cycle=None):
    """Fit one configuration and score the standard partitions, or a caller-supplied frame.

    The evaluation frame, when given, is scored instead of the two standard partitions and is the only
    difference between a cycle candidate and a holdout candidate.
    """
    from lightgbm import LGBMClassifier
    configuration = configuration if isinstance(configuration, dict) else dict(features=configuration)
    features = configuration.get('features', [])
    physical = configuration.get('physical')
    network = configuration.get('network')
    started = time.monotonic()
    if configuration.get('hours'):
        def restrict(part, hours=configuration['hours']):
            keep = pd.to_datetime(part.decision_time, utc=True).dt.hour.isin(hours)
            return part.loc[keep].reset_index(drop=True)
        fitted = {key: restrict(value) for key, value in parts.items()}
    else:
        fitted = parts
    model, network_bundle, network_table = None, None, None
    if physical is None and network:
        # Early stopping holds out a contiguous tail of the training period inside the worker, never
        # a random row set and never any calibration, threshold or validation observation.
        network_table = network_scores(name, configuration, features, fitted, parts, out)
    elif physical is None:
        if name.startswith('logistic'):
            from sklearn.impute import SimpleImputer
            model=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),
                LogisticRegression(C=1,max_iter=2000,random_state=42))
        else:
            model = LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=15, max_depth=4,
                min_child_samples=80, reg_lambda=5, n_jobs=2, random_state=configuration.get('seed', 42),
                verbosity=-1, deterministic=True)
        model.fit(fitted['train'][features], fitted['train'].target)
    def raw(f):
        if physical is not None:
            score = physical_score(physical, f)
        elif network_table is not None:
            aligned = f[['station', 'decision_time']].merge(network_table, on=['station', 'decision_time'],
                how='left', validate='one_to_one')
            if aligned.logit.isna().any():
                raise ValueError('Missing network logits for a scored row')
            score = aligned.logit.to_numpy(float)
        else:
            probability = np.clip(model.predict_proba(f[features])[:, 1], 1e-6, 1-1e-6)
            score = np.log(probability/(1-probability))
        inputs = pd.DataFrame({'score': score})
        if configuration.get('conditional'):
            for column in ['season_sin', 'season_cos', 'margin', 'log_area']:
                inputs[column] = f[column].to_numpy()
        return inputs
    calibration = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
    calibration.fit(raw(fitted['calibration']), fitted['calibration'].target)
    if calibration[-1].coef_[0, 0] <= 0:
        raise ValueError('Nonpositive calibration slope')
    pooled_budget, region_budget = configuration.get('budget', (1., 2.))
    cut = threshold(fitted['threshold'], calibration.predict_proba(raw(fitted['threshold']))[:, 1],
        pooled_budget, region_budget)
    bundle = dict(model=model, network=network_bundle, calibrator=calibration, features=features, threshold=cut)
    joblib.dump(bundle, out/(name+'.joblib'), compress=3)
    scores = {}
    if evaluation is not None:
        f = evaluation
        p = calibration.predict_proba(raw(f))[:, 1]
        pred = f[['station','huc','decision_time','horizon','target','event_time']].copy()
        pred['probability'] = p
        if out is not None:
            pred.to_csv(out/(name+'-holdout.csv.gz'), index=False, compression={'method':'gzip','mtime':0})
        scores['holdout'] = metrics(pred, p, cut)
    else:
        for key in ['temporal_validation', 'geographic_validation']:
            f = parts[key]
            p = calibration.predict_proba(raw(f))[:, 1]
            pred = f[['station','huc','decision_time','horizon','target','event_time']].copy()
            pred['probability'] = p
            pred.to_csv(out/(name+'-'+key+'.csv.gz'), index=False, compression={'method':'gzip','mtime':0})
            scores[key] = metrics(pred, p, cut)
    record = dict(name=name, threshold=cut, artifact_sha256=digest(out/(name+'.joblib')), seconds=time.monotonic()-started, metrics=scores)
    if cycle==2:
        record['forecast_covered']={}
        for key in ['temporal_validation','geographic_validation']:
            f=parts[key].loc[parts[key].hrrr_available.eq(1)]
            p=calibration.predict_proba(raw(f))[:,1]
            record['forecast_covered'][key]=metrics(f,p,cut)
    print(name, {k:(v['detected_events'],round(v['ap'],4),round(v['false_alerts_per_100'],3)) for k,v in scores.items()}, flush=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['acquire','materialize','fit'])
    p.add_argument('--cycle', type=int, default=1)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'acquire': acquire()
    elif args.command == 'materialize': materialize()
    else: fit(args.cycle)
