"""One model family per process; fixed configurations and sealed final labels."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import budget_threshold
from scripts.flood_sprint import write_json
from scripts.mlp_onset import OUT, ROOT, FEATURES, digest, protocol, implementation_hashes, verify_selection

FAMILIES = dict(baseline=['proximity', 'linear', 'damped'], logistic=['dynamics'],
                lightgbm=['dynamics'], xgboost=['dynamics'],
                mlp=['mlp_direct', 'mlp_residual', 'mlp_multitask'])
TRAIN_YEARS = [2020, 2021, 2022]
HORIZONS = [6, 12, 24]


def partitions(frame):
    time = pd.to_datetime(frame.decision_time, utc=True)
    # Every candidate uses 24h-complete labels, even for its shorter horizon.
    safe = (time + pd.Timedelta(hours=24)).dt.year.eq(time.dt.year)
    development = frame.role.eq('development') & safe
    return dict(train=frame.loc[development & time.dt.year.isin(TRAIN_YEARS)],
        calibration=frame.loc[development & time.dt.year.eq(2023)],
        temporal_validation=frame.loc[development & time.dt.year.eq(2024)],
        geographic_validation=frame.loc[frame.role.eq('geographic_test') & safe & time.dt.year.eq(2024)],
        temporal_test=frame.loc[development & time.dt.year.eq(2025)],
        geographic_test=frame.loc[frame.role.eq('geographic_test') & safe & time.dt.year.eq(2025)])


def physical_score(kind, frame):
    x = frame.q_2.to_numpy()
    slope = (x - frame.q_8.to_numpy()) / 6
    horizon = frame.horizon.to_numpy() + 2
    if kind == 'proximity':
        return x
    if kind == 'linear':
        return x + slope * horizon
    if kind == 'damped':
        return x + slope * 6 * (1 - np.exp(-horizon / 6))
    raise ValueError('Unknown physical baseline')


def raw_score(bundle, frame):
    kind = bundle['kind']
    if kind in ['proximity', 'linear', 'damped']:
        return physical_score(kind, frame)
    if kind in ['mlp_direct', 'mlp_residual', 'mlp_multitask']:
        from scripts.mlp_onset_network import network_probability
        p = network_probability(bundle['network'], frame, bundle['horizon'])
    else:
        p = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def probabilities(bundle, frame):
    return bundle['calibrator'].predict_proba(raw_score(bundle, frame).reshape(-1, 1))[:, 1]


def load_data(stage):
    protocol()
    path = OUT / (stage + '.csv.gz')
    if digest(path) != json.loads((OUT / (stage + '-provenance.json')).read_text())['dataset_sha256']:
        raise ValueError('Research dataset checksum changed')
    frame = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
    if not np.isfinite(frame[FEATURES + ['future_max_log_flow']].to_numpy(float)).all():
        raise ValueError('Non-finite features or continuous labels')
    return frame


def fit(family):
    from scripts.shift_hydrology_metrics import metrics
    from scripts.mlp_onset_network import fit_network, horizon_heads
    destination = OUT / (family + '-selection.json')
    if destination.exists() or (OUT / 'selection.json').exists():
        raise FileExistsError('Candidate selection already frozen')
    frame = load_data('development')
    if pd.to_datetime(frame.decision_time, utc=True).dt.year.ge(2025).any():
        raise ValueError('Final outcome leaked into development dataset')
    records = []
    (OUT / 'artifacts').mkdir(exist_ok=True)
    for horizon in HORIZONS:
        parts = partitions(frame.loc[frame.horizon.eq(horizon)])
        train, calibration = parts['train'], parts['calibration']
        validation = pd.concat([parts['temporal_validation'], parts['geographic_validation']])
        for kind in FAMILIES[family]:
            started = time.monotonic()
            features = FEATURES
            bundle = dict(kind=kind, features=features, horizon=horizon, family=family)
            if family == 'baseline':
                pass
            elif kind.startswith('mlp'):
                multi = kind == 'mlp_multitask'
                trained = fit_network(kind, FEATURES,
                    horizon_heads(train) if multi else horizon_heads(train, [horizon]),
                    horizon_heads(calibration) if multi else horizon_heads(calibration, [horizon]))
                bundle['network'] = trained
            else:
                if family == 'logistic':
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
                elif family == 'xgboost':
                    from xgboost import XGBClassifier
                    model = XGBClassifier(n_estimators=250, learning_rate=.04, max_depth=3,
                        min_child_weight=20, subsample=1, colsample_bytree=1, reg_lambda=5,
                        n_jobs=2, random_state=42)
                else:
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=15, max_depth=4,
                        min_child_samples=80, reg_lambda=5, n_jobs=2, random_state=42,
                        verbosity=-1, deterministic=True)
                model.fit(train[features], train.target)
                bundle['model'] = model
            calibrator = LogisticRegression(C=1, max_iter=1000, random_state=42)
            calibrator.fit(raw_score(bundle, calibration).reshape(-1, 1), calibration.target)
            if calibrator.coef_[0, 0] <= 0:
                records.append(dict(name=f'{family}-{kind}-{horizon}', horizon=horizon,
                                    excluded='Nonpositive calibration slope'))
                continue
            bundle['calibrator'] = calibrator
            p = probabilities(bundle, validation)
            threshold = budget_threshold(validation.target.to_numpy(), p, .01)
            name = f'{family}-{kind}-{horizon}'
            bundle.update(name=name, threshold=threshold)
            path = OUT / 'artifacts' / (name + '.joblib')
            joblib.dump(bundle, path, compress=3)
            scores = {key: metrics(parts[key], probabilities(bundle, parts[key]), threshold)
                      for key in ['temporal_validation', 'geographic_validation']}
            record = dict(name=name, horizon=horizon, family=family, kind=kind, features=features,
                threshold=threshold, artifact_sha256=digest(path), validation=scores,
                selection_ap=float(np.mean([s['ap'] for s in scores.values()])),
                selection_brier=float(np.mean([s['brier'] for s in scores.values()])),
                train_rows=len(train), calibration_rows=len(calibration),
                parameters=bundle.get('network', {}).get('parameters', 0),
                fit_seconds=time.monotonic() - started,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
            records.append(record)
            print(name, 'validation AP', round(record['selection_ap'], 4),
                  'seconds', round(record['fit_seconds'], 1), flush=True)
    write_json(destination, dict(candidates=records, dataset_sha256=digest(OUT / 'development.csv.gz'),
        protocol_sha256=digest(OUT / 'protocol.json'), code_sha256=implementation_hashes(),
        versions={x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scikit-learn', 'joblib',
            'torch'] + ([family] if family in ['xgboost', 'lightgbm'] else [])}))


def select():
    protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Final selection already frozen')
    candidates = []
    for family in FAMILIES:
        result = json.loads((OUT / (family + '-selection.json')).read_text())
        if (result['dataset_sha256'] != digest(OUT / 'development.csv.gz') or
                result['protocol_sha256'] != digest(OUT / 'protocol.json') or
                result['code_sha256'] != implementation_hashes()):
            raise ValueError('Candidate family used different frozen inputs')
        for record in result['candidates']:
            if 'excluded' not in record:
                if digest(OUT / 'artifacts' / (record['name'] + '.joblib')) != record['artifact_sha256']:
                    raise ValueError('Candidate artifact changed before selection')
                candidates.append(record)
    rank = lambda r: (-r['selection_ap'], r['selection_brier'], r['name'])
    selected = {}
    for horizon in HORIZONS:
        options = [r for r in candidates if r['horizon'] == horizon]
        selected[str(horizon)] = dict(
            candidate=min([r for r in options if r['family'] != 'baseline'], key=rank)['name'],
            baseline=min([r for r in options if r['family'] == 'baseline'], key=rank)['name'])
    write_json(OUT / 'selection.json', dict(selected=selected, candidates=candidates,
        dataset_sha256=digest(OUT / 'development.csv.gz'), protocol_sha256=digest(OUT / 'protocol.json'),
        code_sha256=implementation_hashes()))


def evaluate(family):
    from scripts.shift_hydrology_metrics import metrics
    selection = verify_selection()
    destination = OUT / (family + '-evaluation.json')
    if destination.exists():
        raise FileExistsError('Final evaluation already recorded')
    frame = load_data('test')
    results = []
    (OUT / 'predictions').mkdir(exist_ok=True)
    for record in selection['candidates']:
        if record['family'] != family:
            continue
        path = OUT / 'artifacts' / (record['name'] + '.joblib')
        if digest(path) != record['artifact_sha256']:
            raise ValueError('Candidate artifact changed')
        bundle = joblib.load(path)
        parts = partitions(frame.loc[frame.horizon.eq(record['horizon'])])
        for partition in ['temporal_test', 'geographic_test']:
            f = parts[partition]
            if f.empty:
                raise ValueError('Final partition has no usable observations')
            started = time.monotonic()
            p = probabilities(bundle, f)
            seconds = time.monotonic() - started
            pred = f[['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']].copy()
            pred['probability'] = p
            output = OUT / 'predictions' / (record['name'] + '-' + partition + '.csv.gz')
            pred.to_csv(output, index=False, compression={'method': 'gzip', 'mtime': 0})
            results.append(dict(name=record['name'], partition=partition,
                metrics=metrics(pred, p, record['threshold']), prediction_sha256=digest(output),
                inference_seconds=seconds, rows_per_second=len(f) / max(seconds, 1e-9)))
    write_json(destination, dict(results=results, dataset_sha256=digest(OUT / 'test.csv.gz'),
        selection_sha256=digest(OUT / 'selection.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'select', 'evaluate'])
    parser.add_argument('family', nargs='?', choices=list(FAMILIES))
    arguments = parser.parse_args()
    if arguments.command == 'select':
        select()
    elif arguments.command == 'fit':
        fit(arguments.family)
    else:
        evaluate(arguments.family)
