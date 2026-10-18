"""One model family per process; frozen configurations and sealed final labels."""
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
from scripts.hydrology_research import budget_threshold
from scripts.mlp_hybrid import OUT, ROOT, digest, implementation_hashes, protocol, verify_selection
from scripts.shift_hydrology import FEATURES

FAMILIES = dict(baseline=['proximity', 'linear', 'damped'], logistic=['dynamics'],
                lightgbm=['dynamics'], xgboost=['dynamics'],
                mlp=['direct', 'hybrid', 'hybrid_prior', 'hybrid_balanced'])
PRIOR_FEATURES = ['prior_above', 'trailing_30d_above', 'trailing_90d_above', 'days_since_above']
TRAIN_YEARS = [2020, 2021, 2022]
HORIZONS = [6, 12, 24]


def partitions(frame):
    time = pd.to_datetime(frame.decision_time, utc=True)
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


def prior_features(frame):
    """Trailing exceedance history; every window excludes the current decision."""
    parts = []
    for _, group in frame.groupby(['station', 'horizon'], sort=False):
        group = group.sort_values('decision_time').copy()
        above = group.q_2.gt(0)
        group['prior_above'] = above.astype(float)
        group['trailing_30d_above'] = above.astype(float).shift(1).rolling(120, min_periods=1).mean().fillna(0.)
        group['trailing_90d_above'] = above.astype(float).shift(1).rolling(360, min_periods=1).mean().fillna(0.)
        times = pd.to_datetime(group.decision_time, utc=True)
        prior_above = above.shift(1, fill_value=False)
        last = times.shift(1).where(prior_above).ffill()
        group['days_since_above'] = ((times - last).dt.total_seconds() / 86400).fillna(365).clip(upper=365)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def ceiling_threshold(frame, p, pooled=1.5, region=3.0):
    """Lowest threshold whose pooled and per-region validation false-alert rates respect the gate ceilings.

    A lower threshold weakly raises both row true positives and unique events detected, so the
    lowest feasible threshold is also the highest-recall one. This deliberately spends more of
    the operational false-alert budget than the one-percent row rule, which self-imposed a target
    the promotion gate does not require.
    """
    y, p = np.asarray(frame.target, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Invalid outcomes or probabilities')
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    thresholds = sorted_p[np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])]
    pooled_rate = 100 * np.cumsum(1 - sorted_y)[np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])] / len(y)
    feasible = pooled_rate <= pooled + 1e-9
    labels = frame.huc.str[:2].to_numpy()
    for name in np.unique(labels):
        rows = labels == name
        descending = -np.sort(-p[rows])
        false_cumulative = np.cumsum(1 - y[rows][np.argsort(-p[rows], kind='stable')])
        # Count region rows at or above each candidate threshold, then count the false ones among them.
        above = np.searchsorted(-descending, -thresholds, side='right')
        region_rate = np.where(above > 0, 100 * false_cumulative[np.maximum(above - 1, 0)] / max(rows.sum(), 1), 0.0)
        feasible &= region_rate <= region + 1e-9
    within = np.flatnonzero(feasible)
    if not len(within):
        return float(np.nextafter(p.min(), np.inf))
    return float(thresholds[within[-1]])


def select_threshold(rule, frame, probabilities, budget=.01):
    if rule == 'row':
        return budget_threshold(frame.target.to_numpy(), probabilities, budget)
    if rule == 'ceiling':
        return ceiling_threshold(frame, probabilities)
    raise ValueError('Unknown threshold rule')


def raw_score(bundle, frame):
    kind = bundle['kind']
    if kind in ['proximity', 'linear', 'damped']:
        return physical_score(kind, frame)
    if bundle['family'] == 'mlp':
        from scripts.mlp_hybrid_network import network_raw
        return network_raw(bundle['network'], frame)
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
            rules = ['row', 'ceiling'] if family == 'mlp' else ['row']
            for rule in rules:
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
