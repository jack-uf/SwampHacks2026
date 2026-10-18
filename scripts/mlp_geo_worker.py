"""One family per process; every candidate receives both threshold rules for a matched comparison."""
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

from scripts.flood_sprint import write_json
from scripts.mlp_geo import OUT, digest, implementation_hashes, protocol, verify_selection
from scripts.mlp_hybrid_worker import (PRIOR_FEATURES, partitions, physical_score, prior_features,
                                       probabilities, raw_score, select_threshold)
from scripts.shift_hydrology import FEATURES

FAMILIES = dict(baseline=['proximity', 'linear', 'damped'], logistic=['dynamics'],
                lightgbm=['dynamics'], xgboost=['dynamics'],
                mlp=['direct', 'hybrid', 'hybrid_prior', 'hybrid_balanced'])
RULES = ['row', 'ceiling']
HORIZONS = [6, 12, 24]


def load_data(stage):
    protocol()
    path = OUT / (stage + '.csv.gz')
    if digest(path) != json.loads((OUT / (stage + '-provenance.json')).read_text())['dataset_sha256']:
        raise ValueError('Research dataset checksum changed')
    frame = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
    if not np.isfinite(frame[FEATURES + ['future_max_log_flow']].to_numpy(float)).all():
        raise ValueError('Non-finite features or continuous labels')
    frame['damped_offset'] = physical_score('damped', frame)
    return prior_features(frame)


def fit(family):
    from scripts.shift_hydrology_metrics import metrics
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
            features = FEATURES + (PRIOR_FEATURES if kind == 'hybrid_prior' else [])
            bundle = dict(kind=kind, features=features, horizon=horizon, family=family)
            if family == 'baseline':
                pass
            elif family == 'mlp':
                # Imported here, never at module scope: torch and LightGBM cannot share a process.
                from scripts.mlp_hybrid_network import fit_network
                bundle['network'] = fit_network(kind, features, train, calibration,
                    offset=kind != 'direct', balanced=kind == 'hybrid_balanced')
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
                records.append(dict(name=f'{family}-{kind}-row-{horizon}', horizon=horizon,
                                    excluded='Nonpositive calibration slope'))
                continue
            bundle['calibrator'] = calibrator
            for rule in RULES:
                threshold = select_threshold(rule, validation, probabilities(bundle, validation))
                name = f'{family}-{kind}-{rule}-{horizon}'
                stored = dict(bundle, name=name, threshold=threshold, threshold_rule=rule)
                path = OUT / 'artifacts' / (name + '.joblib')
                joblib.dump(stored, path, compress=3)
                scores = {key: metrics(parts[key], probabilities(stored, parts[key]), threshold)
                          for key in ['temporal_validation', 'geographic_validation']}
                records.append(dict(name=name, horizon=horizon, family=family, kind=kind, rule=rule,
                    threshold=threshold, artifact_sha256=digest(path), validation=scores,
                    selection_ap=float(np.mean([s['ap'] for s in scores.values()])),
                    selection_brier=float(np.mean([s['brier'] for s in scores.values()])),
                    train_rows=len(train), calibration_rows=len(calibration),
                    parameters=stored.get('network', {}).get('parameters', 0),
                    fit_seconds=time.monotonic() - started,
                    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
                print(name, 'validation AP', round(records[-1]['selection_ap'], 4),
                      'events', scores['temporal_validation']['detected_events'], flush=True)
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
