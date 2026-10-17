"""Isolated geographic experiment. See docs/ML_SPRINT_PROTOCOL.md before running."""
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.model import calibrated
from scripts.build_training import BASE_FEATURES, FEATURES, daily_features, normalize_events
from scripts.common import ROOT, download
from scripts.train import block_comparison, metrics


UNITS = dict(precipitation='mm', temperature_2m='°C', relative_humidity_2m='%',
             wind_gusts_10m='km/h', pressure_msl='hPa', cape='J/kg')
EXTRA_FEATURES = ['antecedent_forecast_rain_7d_mm', 'forecast_pressure_min_hpa',
                  'forecast_pressure_change_24h_hpa', 'forecast_cape_max_jkg']
OUT = ROOT / 'experiments/sprint-20261017'
WEATHER = [f for f in FEATURES if f not in ['latitude', 'longitude', 'season_sin', 'season_cos']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_preserved():
    for name, checksum in json.loads((OUT / 'preserved.json').read_text()).items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Original artifact changed: ' + name)


def forecast_features(payload, latitude, longitude):
    if payload.get('utc_offset_seconds') != 0 or payload.get('timezone') not in ('GMT', 'UTC'):
        raise ValueError('Forecast must use UTC')
    expected_units = {k + '_previous_day2': v for k, v in UNITS.items()}
    if any(payload.get('hourly_units', {}).get(k) != v for k, v in expected_units.items()):
        raise ValueError('Unexpected forecast units')
    hourly = pd.DataFrame(payload['hourly']).set_index('time')
    hourly.index = pd.to_datetime(hourly.index)
    if hourly.index.tz is not None or not hourly.index.is_unique or not hourly.index.is_monotonic_increasing:
        raise ValueError('Forecast hours must be unique, ordered UTC timestamps')
    columns = list(expected_units)
    values = hourly[columns].astype(float)
    for variable in ['precipitation', 'wind_gusts_10m', 'cape', 'relative_humidity_2m']:
        if (values[variable + '_previous_day2'].dropna() < 0).any():
            raise ValueError('Negative physical forecast input: ' + variable)
    if (values['relative_humidity_2m_previous_day2'].dropna() > 100).any():
        raise ValueError('Humidity exceeds 100 percent')
    rows = []
    for row in daily_features(payload, latitude, longitude):
        day = pd.Timestamp(row['target_date'])
        target = values.reindex(pd.date_range(day, periods=24, freq='h'))
        rain = values.reindex(pd.date_range(day - pd.Timedelta(days=7), periods=168, freq='h'))['precipitation_previous_day2']
        pressure = values.reindex(pd.date_range(day - pd.Timedelta(days=1), periods=24, freq='h'))['pressure_msl_previous_day2']
        if not all(np.isfinite(v.to_numpy()).all() for v in [target, rain, pressure]):
            continue
        row.update(antecedent_forecast_rain_7d_mm=float(rain.sum()),
                   forecast_pressure_min_hpa=float(target['pressure_msl_previous_day2'].min()),
                   forecast_pressure_change_24h_hpa=float(target['pressure_msl_previous_day2'].mean() - pressure.mean()),
                   forecast_cape_max_jkg=float(target['cape_previous_day2'].max()))
        rows.append(row)
    return rows


def partition_rows(frame):
    if not frame.origin.isin(['original', 'new']).all() or not frame.role.isin(
            ['development', 'geographic_validation', 'geographic_test']).all():
        raise ValueError('Unknown experiment origin or role')
    if frame.groupby('state').role.nunique().gt(1).any():
        raise ValueError('A state crosses experiment roles')
    development = frame.role.eq('development')
    date = frame.target_date
    return dict(
        train=frame[development & (date < '2025-01-01')],
        calibration=frame[development & (date >= '2025-01-01') & (date < '2025-07-01')],
        validation=frame[development & (date >= '2025-07-01') & (date < '2026-01-01')],
        geographic_validation=frame[frame.role.eq('geographic_validation') & (date >= '2025-07-01') & (date < '2026-01-01')],
        test=frame[development & frame.origin.eq('new') & (date >= '2026-01-01')],
        geographic_test=frame[frame.role.eq('geographic_test') & (date >= '2026-01-01')])


def recall_threshold(y, p, reference, reference_threshold):
    y = np.asarray(y); p = np.asarray(p); reference = np.asarray(reference)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Threshold validation needs both classes')
    if len(p) != len(y) or len(reference) != len(y) or not np.isfinite(p).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid probability vectors')
    required = max(int(np.sum((reference >= reference_threshold) & (y == 1))), int(np.ceil(.30 * y.sum())))
    threshold = float(np.sort(p[y == 1])[-required])
    return threshold, bool(np.mean(p >= threshold) <= .20)


def recipes(train, hazard):
    ratio = (len(train) - train[hazard].sum()) / train[hazard].sum()
    for feature_name, features in [('full', FEATURES), ('weather', WEATHER), ('process', WEATHER + EXTRA_FEATURES)]:
        yield 'logistic_' + feature_name, make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42)), features
        yield 'xgboost_' + feature_name, XGBClassifier(n_estimators=140, max_depth=3, learning_rate=.05,
            subsample=.8, colsample_bytree=.9, scale_pos_weight=ratio, random_state=42, n_jobs=2), features
        yield 'lightgbm_' + feature_name, LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7,
            learning_rate=.05, class_weight='balanced', verbosity=-1, random_state=42, n_jobs=2), features
    if hazard == 'flood':
        yield 'logistic_base', make_pipeline(StandardScaler(), LogisticRegression(
            class_weight='balanced', max_iter=2000, random_state=42)), BASE_FEATURES


def fitted(model, features, train, calibration, hazard, name):
    if any(frame[hazard].nunique() != 2 for frame in [train, calibration]):
        raise ValueError('Fitting and calibration require both classes: ' + hazard)
    model.fit(train[features], train[hazard])
    raw = np.clip(model.predict_proba(calibration[features])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration[hazard])
    return dict(name=name, model=model, calibrator=calibrator, features=list(features),
                background=train[features].sample(16, random_state=42).reset_index(drop=True))


def load_dataset():
    verify_preserved()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if digest(OUT / 'training.csv.gz') != provenance['dataset_sha256'] or digest(OUT / 'protocol.json') != provenance['protocol_sha256']:
        raise ValueError('Frozen dataset or protocol checksum mismatch')
    frame = pd.read_csv(OUT / 'training.csv.gz')
    if frame.duplicated(['county_fips', 'target_date']).any() or not np.isfinite(frame[FEATURES + EXTRA_FEATURES]).all().all():
        raise ValueError('Invalid experiment rows')
    dates = pd.to_datetime(frame.target_date)
    if dates.min() < pd.Timestamp('2024-01-01') or dates.max() > pd.Timestamp('2026-06-30') or not (
            pd.to_datetime(frame.issue_date) == dates - pd.Timedelta(days=1)).all():
        raise ValueError('Experiment dates violate issue-time contract')
    return frame


def select():
    if (OUT / 'selection.json').exists() or (OUT / 'selected.joblib').exists():
        raise FileExistsError('Selection is already frozen; use evaluate or verify')
    frame = load_dataset()
    # Drop final outcomes before any fitting, threshold selection or development report.
    parts = partition_rows(frame[frame.target_date < '2026-01-01'])
    served = joblib.load(ROOT / 'models/hazards.joblib')['hazards']
    validation = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    results = {}; winners = {}
    for hazard in ['flood', 'storm']:
        comparator = served[hazard]; reference = calibrated(comparator, validation)
        candidates = {name: fitted(model, features, parts['train'], parts['calibration'], hazard, name)
                      for name, model, features in recipes(parts['train'], hazard)}
        original = parts['train'][parts['train'].origin.eq('original')]
        original_cal = parts['calibration'][parts['calibration'].origin.eq('original')]
        control = 'logistic_base' if hazard == 'flood' else 'lightgbm_full'
        _, model, features = next(r for r in recipes(original, hazard) if r[0] == control)
        candidates['original_geography'] = fitted(model, features, original, original_cal, hazard, 'original_geography')
        scores = {}
        for name, bundle in candidates.items():
            threshold, supported = recall_threshold(validation[hazard], calibrated(bundle, validation), reference, comparator['threshold'])
            bundle.update(threshold=threshold, warning_supported=supported)
            score = {part: metrics(parts[part][hazard], calibrated(bundle, parts[part]), threshold)
                     for part in ['validation', 'geographic_validation']}
            score.update(features=bundle['features'], threshold=threshold, warning_supported=supported,
                         pooled_validation=metrics(validation[hazard], calibrated(bundle, validation), threshold))
            score['selection_ap'] = float(np.mean([score[k]['pr_auc'] or 0. for k in ['validation', 'geographic_validation']]))
            score['selection_brier'] = float(np.mean([score[k]['brier'] for k in ['validation', 'geographic_validation']]))
            scores[name] = score
        eligible = [n for n in scores if scores[n]['warning_supported']]
        winner = sorted(eligible or list(scores), key=lambda n: (-scores[n]['selection_ap'], scores[n]['selection_brier'], n))[0]
        winners[hazard] = candidates[winner]
        results[hazard] = dict(selected=winner, eligible_candidates=eligible, candidates=scores,
            served_validation={k: metrics(parts[k][hazard], calibrated(comparator, parts[k]), comparator['threshold'])
                               for k in ['validation', 'geographic_validation']})
        print(hazard, winner, 'validation selection AP', scores[winner]['selection_ap'], 'supported', bool(eligible), flush=True)
    joblib.dump(dict(hazards=winners), OUT / 'selected.joblib')
    write_json(OUT / 'selection.json', dict(created_at=datetime.now(timezone.utc).isoformat(), hazards=results,
        dataset_sha256=digest(OUT / 'training.csv.gz'), model_sha256=digest(OUT / 'selected.joblib'),
        served_model_sha256=digest(ROOT / 'models/hazards.joblib'),
        protocol_sha256=digest(OUT / 'protocol.json'), protocol_document_sha256=digest(ROOT / 'docs/ML_SPRINT_PROTOCOL.md'),
        code_sha256=digest(ROOT / 'scripts/ml_sprint.py'),
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib']},
        selection_rule='Supported candidates first; equal temporal/geographic validation AP, mean Brier tie-break, then name',
        test_outcomes_inspected=False))


def paired_intervals(frame, y, old, new, old_threshold, new_threshold):
    result = block_comparison(frame, y, old, new)
    if not result.get('available', True):
        return result
    result['available'] = True
    blocks = pd.to_datetime(frame.target_date).dt.to_period('W').astype(str).to_numpy()
    unique = np.unique(blocks); rng = np.random.default_rng(42); values = []
    y = np.asarray(y); old_alert = old >= old_threshold; new_alert = new >= new_threshold
    for _ in range(300):
        ids = np.concatenate([np.flatnonzero(blocks == week) for week in rng.choice(unique, len(unique), replace=True)])
        positive = y[ids] == 1
        if not positive.any():
            continue
        values.append([float(new_alert[ids][positive].mean() - old_alert[ids][positive].mean()),
                       float(100 * (np.sum(new_alert[ids] & ~positive) - np.sum(old_alert[ids] & ~positive)) / len(ids))])
    bounds = np.quantile(values, [.025, .975], axis=0)
    result.update(delta_recall_95=bounds[:, 0].tolist(), delta_false_alerts_per_100_95=bounds[:, 1].tolist())
    return result


def promotion_failures(updated, baseline, intervals, states, supported):
    failures = []
    if not supported:
        failures.append('Validation threshold unsupported by the 20% alert budget')
    if updated['positives'] < 20:
        failures.append('Fewer than 20 positive county-days')
    if updated['alert_rate'] > .20:
        failures.append('Test alert rate exceeds 20%')
    if not intervals.get('available'):
        failures.append('Insufficient bootstrap evidence')
    else:
        if intervals['delta_ap_95'][0] <= 0:
            failures.append('AP improvement interval includes zero or a regression')
        if intervals['delta_brier_95'][1] > 0:
            failures.append('Brier improvement interval includes a regression')
    if updated['recall'] < max(.30, baseline['recall']):
        failures.append('Recall below 30% or below served comparator')
    if baseline['false_alerts_per_100_county_days'] == 0 or updated['false_alerts_per_100_county_days'] > .9 * baseline['false_alerts_per_100_county_days']:
        failures.append('False alerts not reduced by at least 10%')
    for state, values in states.items():
        if values['updated']['positives'] >= 5 and values['served']['recall'] - values['updated']['recall'] > .10 + 1e-12:
            failures.append('Recall loss exceeds 0.10 in ' + state)
    return failures


def frozen_selection():
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, field in [(OUT / 'selected.joblib', 'model_sha256'), (OUT / 'training.csv.gz', 'dataset_sha256'),
                        (OUT / 'protocol.json', 'protocol_sha256'), (ROOT / 'models/hazards.joblib', 'served_model_sha256'),
                        (ROOT / 'docs/ML_SPRINT_PROTOCOL.md', 'protocol_document_sha256'),
                        (ROOT / 'scripts/ml_sprint.py', 'code_sha256')]:
        if digest(path) != selection[field]:
            raise ValueError('Frozen selection input changed: ' + str(path))
    return selection


def evaluate():
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists; do not tune on it. Use verify')
    selection = frozen_selection(); frame = load_dataset(); parts = partition_rows(frame)
    chosen = joblib.load(OUT / 'selected.joblib')['hazards']
    served = joblib.load(ROOT / 'models/hazards.joblib')['hazards']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(OUT / 'selection.json'),
                  dataset_sha256=digest(OUT / 'training.csv.gz'), hazards={},
                  protocol='docs/ML_SPRINT_PROTOCOL.md',
                  baseline_contract='Untouched served calibrated probabilities and saved thresholds, scored offline even outside serving scope',
                  metrics_contract='AP is average precision. Confusion matrices are [[TN, FP], [FN, TP]]. False alerts use all county-days as denominator.')
    prediction_frames = []
    for hazard in ['flood', 'storm']:
        winner = chosen[hazard]; comparator = served[hazard]; cohorts = {}
        for part in ['test', 'geographic_test']:
            data = parts[part].copy(); y = data[hazard]
            old = calibrated(comparator, data); new = calibrated(winner, data)
            data['served_probability'] = old; data['updated_probability'] = new
            states = {state: dict(served=metrics(group[hazard], group.served_probability, comparator['threshold']),
                                  updated=metrics(group[hazard], group.updated_probability, winner['threshold']))
                      for state, group in data.groupby('state')}
            info = dict(rows=len(data), counties=data.county_fips.nunique(), states=states,
                        served=metrics(y, old, comparator['threshold']), updated=metrics(y, new, winner['threshold']),
                        climatology=metrics(y, np.full(len(y), parts['train'][hazard].mean()), .5),
                        paired_uncertainty=paired_intervals(data, y, old, new, comparator['threshold'], winner['threshold']),
                        counties_detail={region: dict(served=metrics(group[hazard], group.served_probability, comparator['threshold']),
                                                       updated=metrics(group[hazard], group.updated_probability, winner['threshold']))
                                         for region, group in data.groupby('region')})
            info['promotion_failures'] = promotion_failures(info['updated'], info['served'], info['paired_uncertainty'], states, winner['warning_supported'])
            cohorts[part] = info
            prediction_frames.append(pd.DataFrame(dict(hazard=hazard, cohort=part, county_fips=data.county_fips,
                state=data.state, target_date=data.target_date, label=y, served_probability=old, updated_probability=new,
                served_alert=old >= comparator['threshold'], updated_alert=new >= winner['threshold'])))
            print(hazard, part, 'served/updated AP', info['served']['pr_auc'], info['updated']['pr_auc'],
                  'recall', info['served']['recall'], info['updated']['recall'], 'failures', info['promotion_failures'], flush=True)
        decision = 'promote' if all(not c['promotion_failures'] for c in cohorts.values()) else 'retain_served'
        report['hazards'][hazard] = dict(selected=winner['name'], cohorts=cohorts, decision=decision)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / 'predictions.csv.gz', index=False,
                                                        compression=dict(method='gzip', mtime=0))
    report['predictions_sha256'] = digest(OUT / 'predictions.csv.gz')
    write_json(OUT / 'evaluation.json', report)
    verify_preserved()


def verify():
    frame = load_dataset(); selection = frozen_selection()
    report = json.loads((OUT / 'evaluation.json').read_text())
    if digest(OUT / 'selection.json') != report['selection_sha256'] or digest(OUT / 'predictions.csv.gz') != report['predictions_sha256']:
        raise ValueError('Evaluation artifact checksum mismatch')
    predictions = pd.read_csv(OUT / 'predictions.csv.gz')
    parts = partition_rows(frame)
    models = joblib.load(OUT / 'selected.joblib')['hazards']
    served = joblib.load(ROOT / 'models/hazards.joblib')['hazards']
    for hazard, info in report['hazards'].items():
        assert models[hazard]['name'] == selection['hazards'][hazard]['selected'] == info['selected']
        for part, cohort in info['cohorts'].items():
            saved = predictions[predictions.hazard.eq(hazard) & predictions.cohort.eq(part)]
            data = parts[part]
            assert len(saved) == len(data) and not saved.duplicated(['county_fips', 'target_date']).any()
            assert list(saved.county_fips) == list(data.county_fips) and list(saved.target_date) == list(data.target_date)
            assert np.array_equal(saved.label, data[hazard])
            for name, model in [('served', served[hazard]), ('updated', models[hazard])]:
                probability = calibrated(model, data)
                assert np.allclose(probability, saved[name + '_probability'], atol=1e-14, rtol=0)
                fresh = metrics(data[hazard], probability, model['threshold'])
                assert fresh == cohort[name]
    print('Verified original hashes, dataset, frozen selection, prediction rows and all pooled metrics')


def ingest():
    """Only new cache files and isolated experiment outputs may be written."""
    verify_preserved()
    if (OUT / 'training.csv.gz').exists():
        raise FileExistsError('Dataset already frozen; use select or verify, not ingestion again')
    protocol = json.loads((OUT / 'protocol.json').read_text())
    original = [dict(r, county_fips=r['state_fips'] * 1000 + r['county_fips'], origin='original')
                for r in json.loads((ROOT / 'data/regions.json').read_text()) if r['role'] == 'development']
    selected = original + [dict(r, origin='new') for r in protocol['new_counties']]
    sources = []; event_frames = []
    for source in json.loads((ROOT / 'data/training-provenance.json').read_text())['sources']:
        name = urlparse(source['url']).path.rsplit('/', 1)[-1]
        if not name.startswith('StormEvents_details'):
            continue
        path = download(source['url'], name)
        if digest(path) != source['sha256']:
            raise ValueError('NOAA source differs from the frozen original label snapshot')
        sources.append(source)
        frame = pd.read_csv(path, low_memory=False)
        keys = frame.STATE_FIPS * 1000 + frame.CZ_FIPS
        event_frames.append(normalize_events(frame[keys.isin([r['county_fips'] for r in selected])]))
    events = pd.concat(event_frames, ignore_index=True)
    if events.EVENT_ID.duplicated().any():
        raise ValueError('Duplicate NOAA event identity')
    rows = []; coverage = []
    for r in selected:
        fips = r['county_fips']
        variables = ['pressure_msl', 'cape'] if r['origin'] == 'original' else list(UNITS)
        params = dict(latitude=r['latitude'], longitude=r['longitude'], start_date=protocol['start'],
                      end_date=protocol['end'], hourly=','.join(v + '_previous_day2' for v in variables),
                      models='gfs_seamless', timezone='GMT')
        name = f"sprint-forecast-{fips}-{'extra' if r['origin'] == 'original' else 'all'}-202406.json"
        uncached = not (ROOT / 'data/raw' / name).exists()
        path = download('https://previous-runs-api.open-meteo.com/v1/forecast', name, params)
        sources.append(json.loads(path.with_suffix('.json.source.json').read_text()))
        payload = json.loads(path.read_text())
        if r['origin'] == 'original':
            base_path = ROOT / f"data/raw/expanded-forecast-{fips // 1000}-{fips % 1000}-202406.json"
            base_source = json.loads(base_path.with_suffix('.json.source.json').read_text())
            if digest(base_path) != base_source['sha256']:
                raise ValueError('Original forecast checksum mismatch')
            base = json.loads(base_path.read_text()); sources.append(base_source)
            if payload['hourly']['time'] != base['hourly']['time'] or any(
                    abs(payload[k] - base[k]) > 1e-5 for k in ['latitude', 'longitude']):
                raise ValueError('Forecast variables do not share times and grid point')
            payload['hourly'].update(base['hourly']); payload['hourly_units'].update(base['hourly_units'])
        daily = forecast_features(payload, r['latitude'], r['longitude'])
        if not daily:
            raise ValueError('No complete forecast days for ' + r['region'])
        regional = events[(events.STATE_FIPS * 1000 + events.CZ_FIPS).eq(fips)]
        for values in daily:
            day = pd.Timestamp(values['target_date'])
            overlap = regional[(regional.start_utc < day + pd.Timedelta(days=1)) & (regional.end_utc >= day)]
            rows.append(dict(**values, region=r['region'], state=r['state'], county_fips=fips,
                             role=r['role'], origin=r['origin'],
                             flood=int(overlap.EVENT_TYPE.isin(['Flood', 'Flash Flood']).any()),
                             storm=int(overlap.EVENT_TYPE.isin(['Thunderstorm Wind', 'Tornado', 'Hail']).any())))
        coverage.append(dict(region=r['region'], role=r['role'], origin=r['origin'], complete_days=len(daily),
                             requested_days=912, first=daily[0]['target_date'], last=daily[-1]['target_date']))
        print(r['region'], len(daily), 'complete forecast days', flush=True)
        if uncached:
            time.sleep(10)
    frame = pd.DataFrame(rows).sort_values(['target_date', 'county_fips'])
    if frame.duplicated(['county_fips', 'target_date']).any():
        raise ValueError('Duplicate county-day')
    frame.to_csv(OUT / 'training.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    write_json(OUT / 'provenance.json', dict(created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
               coverage=coverage, dataset_sha256=digest(OUT / 'training.csv.gz'),
               protocol_sha256=digest(OUT / 'protocol.json'),
               forecast_contract='Nominal hourly day2 GFS forecasts, not observations or verified dissemination timestamps',
               label_contract='Same frozen NOAA county intervals as original data; absence of report is not absence of hazard'))
    verify_preserved()
    print('Isolated dataset frozen:', len(frame), 'rows;', frame.county_fips.nunique(), 'counties', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ingest', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    dict(ingest=ingest, select=select, evaluate=evaluate, verify=verify)[args.command]()
