"""One model family per process; never imports both boosting runtimes."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import (OUT, ROOT, WEATHER, HYDRO, COMBINED, MEMORY,
    SNOW, budget_threshold, digest, partitions, protocol, write_json)


def probability_input(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def score(bundle, frame):
    kind = bundle['kind']
    if kind == 'persistence':
        return frame.q_3.to_numpy().reshape(-1, 1)
    if kind == 'rainfall':
        leads = [3, 4] if bundle['horizon'] == 1 else [4, 5]
        return np.log1p(frame[[f'rain_{i}' for i in leads]].sum(axis=1)).to_numpy().reshape(-1, 1)
    if kind == 'seasonal':
        return probability_input(frame.target_date.str[5:7].map(bundle['months']).to_numpy())
    return probability_input(bundle['model'].predict_proba(frame[bundle['features']])[:, 1])


def probabilities(bundle, frame):
    return bundle['calibrator'].predict_proba(score(bundle, frame))[:, 1]


def metrics(frame, p, thresholds):
    y = frame.exceedance.to_numpy()
    if len(frame) == 0:
        raise ValueError('Empty evaluation cohort')
    result = dict(rows=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                  ap=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
                  brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])))
    result['alerts'] = {}
    for name, threshold in thresholds.items():
        alert = p >= threshold
        tp = int(np.sum(alert & (y == 1))); fp = int(np.sum(alert & (y == 0)))
        onset = frame.onset.to_numpy(bool)
        result['alerts'][name] = dict(threshold=threshold, precision=tp / max(1, tp + fp),
            recall=tp / max(1, int(y.sum())), false_alerts_per_100=100 * fp / len(y),
            alerts=int(alert.sum()), onsets=int(onset.sum()),
            onset_recall=float(np.sum(onset & alert) / onset.sum()) if onset.any() else None)
    result['reliability'] = [dict(rows=int(mask.sum()), predicted=float(p[mask].mean()),
        observed=float(y[mask].mean())) for i in range(10)
        if (mask := (np.minimum((p * 10).astype(int), 9) == i)).any()]
    return result


def load_data():
    protocol()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if digest(OUT / 'training.csv.gz') != provenance['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    return pd.read_csv(OUT / 'training.csv.gz', dtype={'station': str, 'huc': str})


def fit_family(family):
    started = time.monotonic()
    destination = OUT / f'{family}-selection.json'
    if destination.exists() or (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen; no refitting')
    # Remove final labels before even constructing any fitting cohort.
    frame = load_data()
    frame = frame[frame.target_date.lt('2023-01-01') & frame.role.ne('geographic_test')].copy()
    configurations = dict(weather=WEATHER, hydrology=HYDRO, combined=COMBINED)
    if family == 'lightgbm':
        configurations.update(memory=COMBINED + MEMORY, snow=COMBINED + SNOW)
    if family == 'baseline':
        configurations = {name: [] for name in ['persistence', 'rainfall', 'seasonal']}
    records = []
    (OUT / 'artifacts').mkdir(exist_ok=True)
    for horizon in (1, 2):
        parts = partitions(frame[frame.horizon.eq(horizon)])
        train, calibration = parts['train'], parts['calibration']
        for kind, features in configurations.items():
            begin = time.monotonic()
            bundle = dict(kind=kind, horizon=horizon, features=features)
            if family == 'baseline':
                if kind == 'seasonal':
                    bundle['months'] = train.groupby(train.target_date.str[5:7]).exceedance.mean().to_dict()
            else:
                if family == 'logistic':
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
                elif family == 'xgboost':
                    from xgboost import XGBClassifier
                    model = XGBClassifier(n_estimators=200, learning_rate=.04, max_depth=3,
                        subsample=.8, colsample_bytree=.9, n_jobs=2, random_state=42, eval_metric='logloss')
                elif family == 'lightgbm':
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(n_estimators=200, learning_rate=.04, max_depth=3,
                        num_leaves=7, n_jobs=2, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
                else:
                    raise ValueError('Unknown family')
                model.fit(train[features], train.exceedance)
                bundle['model'] = model
            bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
                score(bundle, calibration), calibration.exceedance)
            pooled = pd.concat([parts['validation'], parts['geographic_validation']])
            p = probabilities(bundle, pooled)
            thresholds = {str(b): budget_threshold(pooled.exceedance, p, b / 100) for b in [1, 5]}
            bundle['thresholds'] = thresholds
            name = f'{family}_{kind}_h{horizon}'
            artifact = OUT / 'artifacts' / f'{name}.joblib'
            joblib.dump(bundle, artifact, compress=3)
            evaluated = {split: metrics(parts[split], probabilities(bundle, parts[split]), thresholds)
                         for split in ['validation', 'geographic_validation']}
            record = dict(name=name, family=family, kind=kind, horizon=horizon,
                features=features, thresholds=thresholds, selection=evaluated,
                artifact_sha256=digest(artifact), seconds=time.monotonic() - begin)
            record['selection_ap'] = float(np.mean([m['ap'] for m in evaluated.values()]))
            record['selection_brier'] = float(np.mean([m['brier'] for m in evaluated.values()]))
            records.append(record)
            print(name, 'selection AP', round(record['selection_ap'], 4), flush=True)
    write_json(destination, dict(candidates=records, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        packages={p: importlib.metadata.version(p) for p in
                  ['numpy', 'pandas', 'scikit-learn', 'joblib'] + ([family] if family in ['xgboost', 'lightgbm'] else [])}))


def evaluate_family(family):
    started = time.monotonic()
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, checksum in selection['code'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Code changed after selection: ' + path)
    destination = OUT / f'{family}-evaluation.json'
    if destination.exists():
        raise FileExistsError('Final evaluation already exists')
    frame = load_data()
    (OUT / 'predictions').mkdir(exist_ok=True)
    records = []
    for candidate in selection['candidates']:
        if candidate['family'] != family:
            continue
        name = candidate['name']; path = OUT / 'artifacts' / f'{name}.joblib'
        if digest(path) != candidate['artifact_sha256']:
            raise ValueError('Frozen model changed')
        bundle = joblib.load(path)
        parts = partitions(frame[frame.horizon.eq(candidate['horizon'])])
        record = dict(name=name, tests={})
        for split in ['forward_test', 'geographic_test']:
            test = parts[split].copy(); p = probabilities(bundle, test)
            report = metrics(test, p, candidate['thresholds'])
            for column in ['station', 'huc']:
                groups = test[column] if column == 'station' else test.huc.str[:2]
                report[column] = {key: metrics(group, p[test.index.get_indexer(group.index)], candidate['thresholds'])
                                  for key, group in test.groupby(groups)}
            aps = [m['ap'] for m in report['station'].values() if m['ap'] is not None]
            report['macro_station_ap'] = float(np.mean(aps)) if aps else None
            report['single_class_stations'] = sum(m['ap'] is None for m in report['station'].values())
            report['lead_start_hours'] = [float(test.lead_start_hours.min()), float(test.lead_start_hours.max())]
            report['lead_end_hours'] = [float(test.lead_end_hours.min()), float(test.lead_end_hours.max())]
            record['tests'][split] = report
            test[['station', 'huc', 'target_date', 'exceedance', 'onset']].assign(probability=p).to_csv(
                OUT / 'predictions' / f'{name}.{split}.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        records.append(record)
        print(name, 'final evaluation saved', flush=True)
    write_json(destination, dict(candidates=records, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'evaluate'])
    parser.add_argument('family', choices=['baseline', 'logistic', 'xgboost', 'lightgbm'])
    args = parser.parse_args()
    (fit_family if args.command == 'select' else evaluate_family)(args.family)
