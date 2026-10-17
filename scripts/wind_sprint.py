"""Measured wind-hazard experiment. See docs/WIND_SPRINT_PROTOCOL.md before running."""
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.model import calibrated
from scripts.common import ROOT, download
from scripts.ml_sprint import recall_threshold
from scripts.train import metrics

OUT = ROOT / 'experiments/sprint-20261017-wind'
COUNTY_SOURCE = ROOT / 'experiments/sprint-20261017/protocol.json'
DOCUMENT = ROOT / 'docs/WIND_SPRINT_PROTOCOL.md'
KNOT_KMH = 1.852
THRESHOLD_KMH = 60.
MAX_STATION_KM = 100.
MIN_STATIONS = 15
START, END = '2024-01-01', '2026-06-30'
UNITS = dict(precipitation='mm', temperature_2m='°C', relative_humidity_2m='%',
             wind_gusts_10m='km/h', pressure_msl='hPa', cape='J/kg')
FORECAST = ['forecast_gust_max_kmh', 'forecast_gust_mean_kmh', 'season_sin', 'season_cos']
EXTENDED = FORECAST + ['forecast_rain_mm', 'forecast_rain_max_hour_mm', 'forecast_wet_hours',
                       'antecedent_forecast_rain_72h_mm', 'forecast_temp_max_c', 'forecast_humidity_pct',
                       'forecast_pressure_min_hpa', 'forecast_pressure_change_24h_hpa', 'forecast_cape_max_jkg',
                       'latitude', 'longitude', 'elevation_m']
DEPENDENCIES = ['scripts/train.py', 'scripts/ml_sprint.py', 'scripts/common.py']
PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib', 'scipy']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def haversine_km(lat1, lon1, lat2, lon2):
    radians = np.radians
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(radians(lat1)) * np.cos(radians(lat2)) * np.sin(dlon / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(a)))


def load_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if digest(COUNTY_SOURCE) != protocol['county_source_sha256'] or digest(DOCUMENT) != protocol['document_sha256']:
        raise ValueError('Frozen county source or protocol document changed')
    return protocol


def network(state):
    path = download(f'https://mesonet.agron.iastate.edu/geojson/network/{state}_ASOS.geojson', f'iem-{state}-asos.geojson')
    stations = []
    for feature in json.loads(path.read_text())['features']:
        properties = feature['properties']
        if properties.get('archive_end') is not None or not properties.get('archive_begin'):
            continue
        if properties['archive_begin'] > '2023-12-01':
            continue
        lon, lat = feature['geometry']['coordinates'][:2]
        stations.append(dict(sid=properties['sid'], name=properties['sname'], state=state,
                             latitude=float(lat), longitude=float(lon), elevation_m=float(properties['elevation'])))
    if not stations:
        raise ValueError('No eligible station in ' + state)
    return stations


def stations():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Station protocol already frozen; use ingest or verify')
    counties = json.loads(COUNTY_SOURCE.read_text())['new_counties']
    claims, claimed = {}, {}
    for role in ['geographic_test', 'geographic_validation', 'development']:
        pool = [county for county in sorted(counties, key=lambda c: (c['state'], c['county_fips'])) if county['role'] == role]
        for county in pool:
            candidates = sorted(network(county['state']), key=lambda s: (haversine_km(county['latitude'], county['longitude'], s['latitude'], s['longitude']), s['sid']))
            choice = next((s for s in candidates if claims.get(s['sid'], role) == role), None)
            distance = haversine_km(county['latitude'], county['longitude'], choice['latitude'], choice['longitude']) if choice else None
            if not choice or distance > MAX_STATION_KM:
                county.update(station=None, distance_km=round(distance, 1) if distance else None, dropped='No unclaimed station within 100 km')
                continue
            claims[choice['sid']] = role
            county.update(station=choice['sid'], distance_km=round(distance, 1), dropped=None)
            record = claimed.setdefault(choice['sid'], dict(choice, role=role, counties=[]))
            record['counties'].append(county['county_fips'])
            print('claim', role, county['state'], county['county_fips'], '->', choice['sid'], round(distance, 1), 'km', flush=True)
    if len(claimed) < MIN_STATIONS:
        raise ValueError('Too few stations resolved for a meaningful experiment')
    write_json(OUT / 'protocol.json', dict(id='sprint-20261017-wind', document='docs/WIND_SPRINT_PROTOCOL.md',
        document_sha256=digest(DOCUMENT), county_source='experiments/sprint-20261017/protocol.json',
        county_source_sha256=digest(COUNTY_SOURCE), seed=42, start=START, end=END, threshold_kmh=THRESHOLD_KMH,
        max_station_km=MAX_STATION_KM, counties=counties, stations=sorted(claimed.values(), key=lambda s: s['sid'])))
    print('Frozen', len(claimed), 'stations across', len({s['role'] for s in claimed.values()}), 'roles', flush=True)


def observation_days(text):
    lines = text.strip().split('\n')
    if lines[0].split(',') != ['station', 'valid', 'gust']:
        raise ValueError('Unexpected ASOS archive columns: ' + lines[0])
    frame = pd.DataFrame([line.split(',') for line in lines[1:]], columns=['station', 'valid', 'gust'])
    frame['valid'] = pd.to_datetime(frame.valid, format='%Y-%m-%d %H:%M')
    frame['gust_kmh'] = pd.to_numeric(frame.gust.replace({'M': None, '': None}), errors='coerce') * KNOT_KMH
    # A missing gust value is a report without a gust, not a missing observation.
    frame = frame.groupby('valid', as_index=False).gust_kmh.max()
    rows = []
    for day, group in frame.groupby(frame.valid.dt.floor('D')):
        hours = group.valid.dt.hour
        if len(group) < 18 or (hours // 6).nunique() < 4:
            continue
        exceed = group[group.gust_kmh >= THRESHOLD_KMH]
        rows.append(dict(target_date=str(day.date()),
                         max_gust_kmh=float(group.gust_kmh.max()) if group.gust_kmh.notna().any() else None,
                         exceedance=int(len(exceed) > 0),
                         first_exceed_hour_utc=int(exceed.valid.dt.hour.min()) if len(exceed) else None))
    return rows


def forecast_features(payload, station):
    if payload.get('utc_offset_seconds') != 0 or payload.get('timezone') not in ('GMT', 'UTC'):
        raise ValueError('Forecast must use UTC')
    expected = {k + '_previous_day2': v for k, v in UNITS.items()}
    if any(payload.get('hourly_units', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected forecast units')
    hourly = pd.DataFrame(payload['hourly']).set_index('time')
    hourly.index = pd.to_datetime(hourly.index)
    if hourly.index.tz is not None or not hourly.index.is_unique or not hourly.index.is_monotonic_increasing:
        raise ValueError('Forecast hours must be unique, ordered UTC timestamps')
    values = hourly[list(expected)].astype(float)
    for variable in ['precipitation', 'wind_gusts_10m', 'cape', 'relative_humidity_2m']:
        if (values[variable + '_previous_day2'].dropna() < 0).any():
            raise ValueError('Negative physical forecast input: ' + variable)
    if (values['relative_humidity_2m_previous_day2'].dropna() > 100).any():
        raise ValueError('Humidity exceeds 100 percent')
    rows = []
    for day, group in hourly.groupby(hourly.index.floor('D')):
        target = values.reindex(pd.date_range(day, periods=24, freq='h'))
        antecedent = values.reindex(pd.date_range(day - pd.Timedelta(hours=72), periods=72, freq='h'))
        pressure = values.reindex(pd.date_range(day - pd.Timedelta(days=1), periods=24, freq='h'))
        if not all(np.isfinite(v.to_numpy()).all() for v in [target, antecedent, pressure]):
            continue
        gust = target['wind_gusts_10m_previous_day2']
        rain = target['precipitation_previous_day2']
        doy = day.dayofyear
        rows.append(dict(target_date=str(day.date()), issue_date=str((day - pd.Timedelta(days=1)).date()),
                         forecast_gust_max_kmh=float(gust.max()), forecast_gust_mean_kmh=float(gust.mean()),
                         forecast_rain_mm=float(rain.sum()), forecast_rain_max_hour_mm=float(rain.max()),
                         forecast_wet_hours=int((rain > 0).sum()),
                         antecedent_forecast_rain_72h_mm=float(antecedent['precipitation_previous_day2'].sum()),
                         forecast_temp_max_c=float(target['temperature_2m_previous_day2'].max()),
                         forecast_humidity_pct=float(target['relative_humidity_2m_previous_day2'].mean()),
                         forecast_pressure_min_hpa=float(target['pressure_msl_previous_day2'].min()),
                         forecast_pressure_change_24h_hpa=float(target['pressure_msl_previous_day2'].mean() - pressure['pressure_msl_previous_day2'].mean()),
                         forecast_cape_max_jkg=float(target['cape_previous_day2'].max()),
                         latitude=station['latitude'], longitude=station['longitude'], elevation_m=station['elevation_m'],
                         season_sin=float(np.sin(2 * np.pi * doy / 365.25)), season_cos=float(np.cos(2 * np.pi * doy / 365.25))))
    return rows


def ingest():
    protocol = load_protocol()
    if (OUT / 'training.csv.gz').exists():
        raise FileExistsError('Dataset already frozen; use select or verify')
    OUT.mkdir(parents=True, exist_ok=True)
    rows, coverage, sources = [], [], []
    for station in protocol['stations']:
        params = [('station', station['sid']), ('data', 'gust'), ('report_type', 3), ('report_type', 4),
                  ('year1', 2024), ('month1', 1), ('day1', 1), ('year2', 2026), ('month2', 6), ('day2', 30),
                  ('tz', 'UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('direct', 'no')]
        observed_path = download('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                                 f"iem-asos-{station['sid']}-202406.csv", params)
        reported = sorted({line.split(',')[0] for line in observed_path.read_text().strip().split('\n')[1:]})
        if len(reported) != 1:
            raise ValueError('ASOS response does not identify exactly one station: ' + station['sid'])
        observed = {row['target_date']: row for row in observation_days(observed_path.read_text())}
        forecast_params = dict(latitude=station['latitude'], longitude=station['longitude'], start_date=START,
                               end_date=END, hourly=','.join(v + '_previous_day2' for v in UNITS),
                               models='gfs_seamless', timezone='GMT')
        forecast_path = download('https://previous-runs-api.open-meteo.com/v1/forecast',
                                 f"wind-forecast-{station['sid']}-202406.json", forecast_params)
        daily = forecast_features(json.loads(forecast_path.read_text()), station)
        for values in daily:
            if values['target_date'] not in observed:
                continue
            day = observed[values['target_date']]
            rows.append(dict(**values, max_gust_kmh=day['max_gust_kmh'], exceedance=day['exceedance'],
                             first_exceed_hour_utc=day['first_exceed_hour_utc'], station=station['sid'],
                             name=station['name'], state=station['state'], role=station['role']))
        coverage.append(dict(station=station['sid'], name=station['name'], state=station['state'], role=station['role'],
                             observed_days=len(observed), complete_forecast_days=len(daily), merged_days=len([v for v in daily if v['target_date'] in observed]),
                             first=min(observed) if observed else None, last=max(observed) if observed else None))
        for path in [observed_path, forecast_path]:
            sources.append(json.loads(path.with_suffix(path.suffix + '.source.json').read_text()))
        print(station['sid'], len(observed), 'observed days;', coverage[-1]['merged_days'], 'merged', flush=True)
        time.sleep(10)
    frame = pd.DataFrame(rows).sort_values(['target_date', 'station'])
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Duplicate or non-finite experiment rows')
    frame.to_csv(OUT / 'training.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    write_json(OUT / 'provenance.json', dict(created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
        coverage=coverage, dataset_sha256=digest(OUT / 'training.csv.gz'), protocol_sha256=digest(OUT / 'protocol.json'),
        forecast_contract='Nominal hourly day2 GFS forecasts, not observations or verified dissemination timestamps',
        target_contract='Measured ASOS METAR gusts in knots at the station, converted at 1 knot = 1.852 km/h; 60 km/h threshold',
        completeness='Station-days require 18 hourly reports and one report per UTC 6-hour quarter; forecast days require 24 target and 72 antecedent hours'))
    print('Isolated dataset frozen:', len(frame), 'rows;', frame.station.nunique(), 'stations;',
          int(frame.exceedance.sum()), 'positive station-days', flush=True)


def partition_rows(frame):
    if not frame.role.isin(['development', 'geographic_validation', 'geographic_test']).all():
        raise ValueError('Unknown experiment role')
    if frame.groupby('station').role.nunique().gt(1).any():
        raise ValueError('A station crosses experiment roles')
    development = frame.role.eq('development')
    date = frame.target_date
    return dict(train=frame[development & (date < '2025-01-01')],
                calibration=frame[development & (date >= '2025-01-01') & (date < '2025-07-01')],
                validation=frame[development & (date >= '2025-07-01') & (date < '2026-01-01')],
                geographic_validation=frame[frame.role.eq('geographic_validation') & (date >= '2025-07-01') & (date < '2026-01-01')],
                test=frame[development & (date >= '2026-01-01')],
                geographic_test=frame[frame.role.eq('geographic_test') & (date >= '2026-01-01')])


def recipes(train):
    ratio = (len(train) - train.exceedance.sum()) / train.exceedance.sum()
    for label, features in [('forecast', FORECAST), ('extended', EXTENDED)]:
        yield 'logistic_' + label, make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42)), features
        yield 'xgboost_' + label, XGBClassifier(n_estimators=140, max_depth=3, learning_rate=.05, subsample=.8,
            colsample_bytree=.9, scale_pos_weight=ratio, random_state=42, n_jobs=2), features
        yield 'lightgbm_' + label, LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7, learning_rate=.05,
            class_weight='balanced', verbosity=-1, random_state=42, n_jobs=2), features


def fitted(model, features, train, calibration, name):
    if any(frame.exceedance.nunique() != 2 for frame in [train, calibration]):
        raise ValueError('Fitting and calibration require both classes')
    model.fit(train[features], train.exceedance)
    raw = np.clip(model.predict_proba(calibration[features])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration.exceedance)
    return dict(name=name, model=model, calibrator=calibrator, features=list(features))


def rule_metrics(y, score, threshold):
    # The forecast rule scores in km/h, so probability-only metrics do not apply.
    y = np.asarray(y)
    predicted = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(int(y.sum()), 1)),
                pr_auc=float(average_precision_score(y, score)) if y.sum() else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                false_alerts_per_100_station_days=float(fp * 100 / len(y)),
                false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
                alert_rate=float(predicted.mean()), threshold=float(threshold))


def station_metrics(y, probability, threshold):
    result = metrics(y, probability, threshold)
    result['false_alerts_per_100_station_days'] = result.pop('false_alerts_per_100_county_days')
    return result


def load_dataset():
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if digest(OUT / 'training.csv.gz') != provenance['dataset_sha256'] or digest(OUT / 'protocol.json') != provenance['protocol_sha256']:
        raise ValueError('Frozen dataset or protocol checksum mismatch')
    frame = pd.read_csv(OUT / 'training.csv.gz')
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Invalid experiment rows')
    dates = pd.to_datetime(frame.target_date)
    if dates.min() < pd.Timestamp(START) or dates.max() > pd.Timestamp(END) or not (
            pd.to_datetime(frame.issue_date) == dates - pd.Timedelta(days=1)).all():
        raise ValueError('Experiment dates violate issue-time contract')
    return frame


def select():
    if (OUT / 'selection.json').exists() or (OUT / 'selected.joblib').exists():
        raise FileExistsError('Selection is already frozen; use evaluate or verify')
    frame = load_dataset()
    parts = partition_rows(frame[frame.target_date < '2026-01-01'])
    validation = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    reference = (validation.forecast_gust_max_kmh >= THRESHOLD_KMH).astype(float)
    candidates = {name: fitted(model, features, parts['train'], parts['calibration'], name)
                  for name, model, features in recipes(parts['train'])}
    results, winners = {}, {}
    baseline_validation = rule_metrics(validation.exceedance, validation.forecast_gust_max_kmh, THRESHOLD_KMH)
    for name, bundle in candidates.items():
        threshold, supported = recall_threshold(validation.exceedance, calibrated(bundle, validation), reference, .5)
        bundle.update(threshold=threshold, warning_supported=supported)
        scores = {part: station_metrics(parts[part].exceedance, calibrated(bundle, parts[part]), threshold)
                  for part in ['validation', 'geographic_validation']}
        scores.update(features=bundle['features'], threshold=threshold, warning_supported=supported,
                      pooled_validation=station_metrics(validation.exceedance, calibrated(bundle, validation), threshold))
        scores['selection_ap'] = float(np.mean([scores[k]['pr_auc'] or 0. for k in ['validation', 'geographic_validation']]))
        scores['selection_brier'] = float(np.mean([scores[k]['brier'] for k in ['validation', 'geographic_validation']]))
        results[name] = scores
        print(name, 'threshold', round(threshold, 4), 'validation AP', round(scores['selection_ap'], 4),
              'supported', supported, flush=True)
    eligible = [name for name in results if results[name]['warning_supported']]
    winner = sorted(eligible or list(results), key=lambda n: (-results[n]['selection_ap'], results[n]['selection_brier'], n))[0]
    winners['exceedance'] = candidates[winner]
    joblib.dump(dict(hazards=winners), OUT / 'selected.joblib')
    write_json(OUT / 'selection.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selected=winner,
        eligible_candidates=eligible, candidates=results, baseline_validation=baseline_validation,
        dataset_sha256=digest(OUT / 'training.csv.gz'), model_sha256=digest(OUT / 'selected.joblib'),
        protocol_sha256=digest(OUT / 'protocol.json'), protocol_document_sha256=digest(DOCUMENT),
        code_sha256=digest(ROOT / 'scripts/wind_sprint.py'),
        dependencies={name: digest(ROOT / name) for name in DEPENDENCIES},
        versions={name: importlib.metadata.version(name) for name in PACKAGES},
        selection_rule='Supported candidates first; equal temporal/geographic validation AP, mean Brier tie-break, then name',
        test_outcomes_inspected=False))
    print('Selected', winner, flush=True)


def paired_intervals(frame, y, baseline_score, probability, baseline_threshold, threshold):
    blocks = pd.to_datetime(frame.target_date).dt.to_period('W').astype(str).to_numpy()
    unique = np.unique(blocks)
    y, baseline_score, probability = np.asarray(y), np.asarray(baseline_score), np.asarray(probability)
    rng, values = np.random.default_rng(42), []
    for _ in range(300):
        ids = np.concatenate([np.flatnonzero(blocks == week) for week in rng.choice(unique, len(unique), replace=True)])
        if y[ids].sum() == 0:
            continue
        old_alert, new_alert = baseline_score[ids] >= baseline_threshold, probability[ids] >= threshold
        values.append([float(average_precision_score(y[ids], probability[ids]) - average_precision_score(y[ids], baseline_score[ids])),
                       float(new_alert[y[ids] == 1].mean() - old_alert[y[ids] == 1].mean()),
                       float(100 * (np.sum(new_alert & (y[ids] == 0)) - np.sum(old_alert & (y[ids] == 0))) / len(ids))])
    if len(unique) < 10 or len(values) < 100:
        return dict(method='paired calendar-week bootstrap', available=False, reason='Insufficient resampled event weeks')
    bounds = np.quantile(values, [.025, .975], axis=0)
    return dict(method='300 paired calendar-week block resamples; stations within week retained; descriptive 95% interval',
                weeks=len(unique), replicates=len(values), available=True,
                delta_ap_95=bounds[:, 0].tolist(), delta_recall_95=bounds[:, 1].tolist(),
                delta_false_alerts_per_100_95=bounds[:, 2].tolist())


def integration_failures(updated, baseline, intervals, stations, supported):
    failures = []
    if not supported:
        failures.append('Validation threshold unsupported by the 20% alert budget')
    if updated['positives'] < 20:
        failures.append('Fewer than 20 positive station-days')
    if updated['alert_rate'] > .20:
        failures.append('Test alert rate exceeds 20%')
    if not intervals.get('available'):
        failures.append('Insufficient bootstrap evidence')
    elif intervals['delta_ap_95'][0] <= 0:
        failures.append('AP improvement interval includes zero or a regression')
    if updated['recall'] < max(.30, baseline['recall']) - 1e-12:
        failures.append('Recall below 30% or below the forecast rule')
    if baseline['false_alerts_per_100_station_days'] == 0 or updated['false_alerts_per_100_station_days'] > .9 * baseline['false_alerts_per_100_station_days']:
        failures.append('False alerts not reduced by at least 10%')
    for station, values in stations.items():
        if values['updated']['positives'] >= 5 and values['baseline']['recall'] - values['updated']['recall'] > .10 + 1e-12:
            failures.append('Recall loss exceeds 0.10 at ' + station)
    return failures


def lead_times(frame, alert):
    hits = frame[frame.exceedance.eq(1) & alert]
    hours = (hits.first_exceed_hour_utc + 24).dropna()
    if not len(hours):
        return dict(count=0)
    return dict(count=int(len(hours)), minimum=float(hours.min()), median=float(hours.median()),
                maximum=float(hours.max()), all_at_least_24_hours=bool(hours.min() >= 24))


def frozen_selection():
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, field in [(OUT / 'selected.joblib', 'model_sha256'), (OUT / 'training.csv.gz', 'dataset_sha256'),
                        (OUT / 'protocol.json', 'protocol_sha256'), (DOCUMENT, 'protocol_document_sha256'),
                        (ROOT / 'scripts/wind_sprint.py', 'code_sha256')]:
        if digest(path) != selection[field]:
            raise ValueError('Frozen selection input changed: ' + str(path))
    for name, checksum in selection['dependencies'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Frozen dependency changed: ' + name)
    for name, version in selection['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Frozen package version changed: ' + name)
    return selection


def evaluate():
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists; do not tune on it. Use verify')
    selection = frozen_selection()
    with (OUT / 'evaluation-started.json').open('x') as marker:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(OUT / 'selection.json')), marker)
    frame = load_dataset()
    parts = partition_rows(frame)
    winner = joblib.load(OUT / 'selected.joblib')['hazards']['exceedance']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(OUT / 'selection.json'),
                  dataset_sha256=digest(OUT / 'training.csv.gz'), protocol='docs/WIND_SPRINT_PROTOCOL.md',
                  baseline_contract='Archived forecast maximum gust at 60 km/h, the same physical threshold as the label',
                  metrics_contract='AP is average precision. False alerts use all station-days as denominator. Baseline Brier and calibration are not defined for a non-probabilistic score.',
                  cohorts={})
    predictions = []
    for part in ['test', 'geographic_test']:
        data = parts[part].copy()
        probability = calibrated(winner, data)
        data['probability'] = probability
        stations = {station: dict(baseline=rule_metrics(group.exceedance, group.forecast_gust_max_kmh, THRESHOLD_KMH),
                                  updated=station_metrics(group.exceedance, group.probability, winner['threshold']))
                    for station, group in data.groupby('station')}
        info = dict(rows=len(data), stations=int(data.station.nunique()), states=sorted(set(data.state)),
                    baseline=rule_metrics(data.exceedance, data.forecast_gust_max_kmh, THRESHOLD_KMH),
                    updated=station_metrics(data.exceedance, probability, winner['threshold']),
                    climatology=station_metrics(data.exceedance, np.full(len(data), parts['train'].exceedance.mean()), .5),
                    paired_uncertainty=paired_intervals(data, data.exceedance, data.forecast_gust_max_kmh, probability, THRESHOLD_KMH, winner['threshold']),
                    lead_time_updated=lead_times(data, probability >= winner['threshold']),
                    lead_time_baseline=lead_times(data, data.forecast_gust_max_kmh >= THRESHOLD_KMH),
                    stations_detail=stations)
        info['integration_failures'] = integration_failures(info['updated'], info['baseline'], info['paired_uncertainty'], stations, winner['warning_supported'])
        report['cohorts'][part] = info
        predictions.append(pd.DataFrame(dict(cohort=part, station=data.station, state=data.state, target_date=data.target_date,
            label=data.exceedance, baseline_score=data.forecast_gust_max_kmh, probability=probability,
            baseline_alert=data.forecast_gust_max_kmh >= THRESHOLD_KMH, alert=probability >= winner['threshold'])))
        print(part, 'baseline/updated AP', info['baseline']['pr_auc'], info['updated']['pr_auc'],
              'recall', info['baseline']['recall'], info['updated']['recall'], 'failures', info['integration_failures'], flush=True)
    report['selected'] = winner['name']
    report['decision'] = 'propose_integration_review' if all(not c['integration_failures'] for c in report['cohorts'].values()) else 'retain_forecast_threshold'
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'predictions.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    report['predictions_sha256'] = digest(OUT / 'predictions.csv.gz')
    write_json(OUT / 'evaluation.json', report)
    print('Decision:', report['decision'], flush=True)


def verify():
    frame = load_dataset()
    selection = frozen_selection()
    report = json.loads((OUT / 'evaluation.json').read_text())
    if digest(OUT / 'selection.json') != report['selection_sha256'] or digest(OUT / 'predictions.csv.gz') != report['predictions_sha256']:
        raise ValueError('Evaluation artifact checksum mismatch')
    predictions = pd.read_csv(OUT / 'predictions.csv.gz')
    parts = partition_rows(frame)
    winner = joblib.load(OUT / 'selected.joblib')['hazards']['exceedance']
    assert report['selected'] == selection['selected']
    for part, cohort in report['cohorts'].items():
        saved, data = predictions[predictions.cohort.eq(part)], parts[part]
        assert len(saved) == len(data) and not saved.duplicated(['station', 'target_date']).any()
        assert list(saved.station) == list(data.station) and list(saved.target_date) == list(data.target_date)
        assert np.array_equal(saved.label, data.exceedance)
        probability = calibrated(winner, data)
        assert np.allclose(probability, saved.probability, atol=1e-14, rtol=0)
        assert station_metrics(data.exceedance, probability, winner['threshold']) == cohort['updated']
        assert rule_metrics(data.exceedance, data.forecast_gust_max_kmh, THRESHOLD_KMH) == cohort['baseline']
    print('Verified dataset, frozen selection, prediction rows and all pooled metrics')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['stations', 'ingest', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    dict(stations=stations, ingest=ingest, select=select, evaluate=evaluate, verify=verify)[args.command]()

