"""Frozen onset ablations in an isolated, two-thread LightGBM worker."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import WEATHER, HYDRO, COMBINED, SNOW, budget_threshold, weighted_ap
from scripts.onset_research import (ROOT, OUT, digest, write_json, verify_protocol,
                                    partitions, adaptive_alerts, alert_metrics)


def load_data(split):
    verify_protocol()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    path = OUT / f'{split}.csv.gz'
    if digest(path) != provenance['datasets'][split]['sha256']:
        raise ValueError('Frozen dataset changed')
    return pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=True).fillna({'event_date': ''})


def logit_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))


def trajectory_inputs(bundle, frame):
    predictions = np.stack([model.predict(frame[bundle['features']]) + frame.q_3.to_numpy()
                            for model in bundle['regressors']], axis=1).reshape(-1, 3, 3)
    predictions.sort(axis=2)
    head = np.column_stack([predictions.reshape(-1, 9), predictions[:, :, 2] - predictions[:, :, 0],
                            frame[['q_3', 'q_change']].to_numpy()])
    return predictions, head


def raw_score(bundle, frame):
    name = bundle['name']
    if name == 'proximity':
        return frame[['q_3']].to_numpy()
    if name == 'extrapolation':
        return (frame.q_3 + 1.5 * frame.q_change).to_numpy().reshape(-1, 1)
    if name == 'rainfall':
        return np.log1p(frame[['rain_3', 'rain_4', 'rain_5']].sum(axis=1)).to_numpy().reshape(-1, 1)
    p = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def predict(bundle, frame):
    if bundle['name'] == 'trajectory':
        return bundle['head'].predict_proba(trajectory_inputs(bundle, frame)[1])[:, 1]
    return bundle['calibrator'].predict_proba(raw_score(bundle, frame))[:, 1]


def metrics(frame, p, alerts):
    y = frame.exceedance.to_numpy()
    result = dict(rows=len(frame), positive_decisions=int(y.sum()), prevalence=float(y.mean()),
        ap=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
        policies={name: alert_metrics(frame, mask) for name, mask in alerts.items()})
    result['reliability'] = [dict(rows=int(mask.sum()), predicted=float(p[mask].mean()),
        observed=float(y[mask].mean())) for i in range(10)
        if (mask := np.minimum((p * 10).astype(int), 9) == i).any()]
    return result


def fit():
    from lightgbm import LGBMClassifier, LGBMRegressor
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection is frozen')
    started = time.monotonic()
    frame = load_data('training')
    parts = partitions(frame)
    train, cal = parts['train'], parts['calibration']
    pooled = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    configs = dict(proximity=[], extrapolation=[], rainfall=[], high_water=COMBINED + SNOW,
                   logistic=COMBINED + SNOW, weather=WEATHER, hydrology=HYDRO,
                   combined=COMBINED, snow=COMBINED + SNOW, trajectory=COMBINED + SNOW)
    params = dict(n_estimators=200, learning_rate=.04, max_depth=3, num_leaves=7, n_jobs=2,
                  random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
    (OUT / 'artifacts').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    records = []
    for name, features in configs.items():
        begin = time.monotonic()
        bundle = dict(name=name, features=features)
        trajectory_report = {}
        if name == 'trajectory':
            bundle['regressors'] = []
            for lead in [1, 2, 3]:
                residual = train[f'flow_{lead}'] - train.q_3
                for alpha in [.1, .5, .9]:
                    regressor = LGBMRegressor(**params, objective='quantile', alpha=alpha)
                    regressor.fit(train[features], residual)
                    bundle['regressors'].append(regressor)
            bundle['head'] = logit_model().fit(trajectory_inputs(bundle, cal)[1], cal.exceedance)
            quantiles, _ = trajectory_inputs(bundle, pooled)
            trajectory_report = trajectory_metrics(pooled, quantiles)
        else:
            if features:
                model = logit_model() if name == 'logistic' else LGBMClassifier(**params)
                model.fit(train[features], train.high_water if name == 'high_water' else train.exceedance)
                bundle['model'] = model
            bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
                raw_score(bundle, cal), cal.exceedance)
        p = predict(bundle, pooled)
        thresholds = {}
        for budget in [1, 5]:
            thresholds[f'pooled_{budget}'] = budget_threshold(pooled.exceedance, p, budget / 100)
            thresholds[f'robust_{budget}'] = max(budget_threshold(group.exceedance,
                p[group.index], budget / 100) for _, group in pooled.groupby(pooled.huc.str[:2]))
        bundle['thresholds'] = thresholds
        bundle['source_negatives'] = p[pooled.exceedance.eq(0).to_numpy()]
        artifact = OUT / 'artifacts' / f'{name}.joblib'
        joblib.dump(bundle, artifact, compress=3)
        reports = {}
        for split in ['validation', 'geographic_validation']:
            subset = parts[split]; pp = predict(bundle, subset)
            reports[split] = metrics(subset, pp, {k: pp >= t for k, t in thresholds.items()})
        pooled[['station', 'huc', 'decision_date', 'exceedance', 'event_date', 'event_lead_hours']].assign(probability=p).to_csv(
            OUT / 'predictions' / f'{name}.selection.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        record = dict(name=name, features=features, thresholds=thresholds, selection=reports,
            trajectory=trajectory_report, artifact_sha256=digest(artifact), seconds=time.monotonic() - begin,
            selection_ap=float(np.mean([r['ap'] for r in reports.values()])),
            selection_brier=float(np.mean([r['brier'] for r in reports.values()])))
        records.append(record)
        print(name, 'selection AP', round(record['selection_ap'], 4), 'seconds', round(record['seconds'], 1), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 6_000_000_000:
            raise MemoryError('Worker exceeded frozen 6 GB RSS limit')
    order = lambda r: (-r['selection_ap'], r['selection_brier'], r['name'])
    selected = min(records, key=order)['name']
    direct = min([r for r in records if r['name'] in ['logistic', 'weather', 'hydrology', 'combined', 'snow']], key=order)['name']
    write_json(OUT / 'selection.json', dict(selected=selected, direct_reference=direct,
        high_water_reference='high_water', candidates=records, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        development={key: dict(rows=len(value), positives=int(value.exceedance.sum()),
            events=len(value[value.exceedance.eq(1)].drop_duplicates(['station', 'event_date']))) for key, value in parts.items()},
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm', 'scikit-learn', 'joblib']},
        code={p: digest(ROOT / p) for p in ['scripts/onset_worker.py', 'scripts/onset_research.py', 'tests/test_onset_research.py']},
        provenance_sha256=digest(OUT / 'provenance.json')))


def trajectory_metrics(frame, quantiles):
    result = {}
    for i, lead in enumerate([1, 2, 3]):
        y = frame[f'flow_{lead}'].to_numpy(); point = frame.q_3.to_numpy()
        result[str(lead)] = dict(pinball={str(alpha): float(mean_pinball_loss(y, quantiles[:, i, j], alpha=alpha))
            for j, alpha in enumerate([.1, .5, .9])},
            persistence_pinball={str(alpha): float(mean_pinball_loss(y, point, alpha=alpha)) for alpha in [.1, .5, .9]},
            coverage_80=float(np.mean((y >= quantiles[:, i, 0]) & (y <= quantiles[:, i, 2]))),
            mean_width=float(np.mean(quantiles[:, i, 2] - quantiles[:, i, 0])))
    return result


def bootstrap(frame, predictions, selection):
    """Paired cluster weights; repeated event windows count one episode per draw."""
    selected = selection['selected']
    result = {}
    for ref in dict.fromkeys([selection['direct_reference'], 'high_water']):
        pair = {}
        for grouping, keys in [('huc4', frame.huc.str[:4]), ('month', frame.decision_date.str[:7])]:
            codes, unique = pd.factorize(keys)
            rng = np.random.default_rng(42)
            differences = []
            y = frame.exceedance.to_numpy()
            positive = y == 1
            event_codes, event_names = pd.factorize(frame.loc[positive, 'station'] + ':' + frame.loc[positive, 'event_date'])
            alerts = {name: predictions[name] >= next(r for r in selection['candidates'] if r['name'] == name)['thresholds']['pooled_1']
                      for name in [selected, ref]}
            for _ in range(500):
                weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[codes]
                if not (weights * y).sum() or not (weights * (1-y)).sum():
                    continue
                ap = weighted_ap(y, predictions[selected], weights) - weighted_ap(y, predictions[ref], weights)
                event_weights = np.zeros(len(event_names)); np.maximum.at(event_weights, event_codes, weights[positive])
                recalls = {}
                for name in alerts:
                    hits = np.zeros(len(event_names), dtype=bool)
                    np.logical_or.at(hits, event_codes, alerts[name][positive] & (weights[positive] > 0))
                    recalls[name] = float(np.sum(event_weights * hits) / max(1, event_weights.sum()))
                differences.append([ap, recalls[selected] - recalls[ref]])
            values = np.asarray(differences)
            pair[grouping] = dict(draws=len(values), ap_difference_95=np.quantile(values[:, 0], [.025, .975]).tolist(),
                                  event_recall_difference_95=np.quantile(values[:, 1], [.025, .975]).tolist())
        result[ref] = pair
    return result


def evaluate():
    started = time.monotonic()
    selection = json.loads((OUT / 'selection.json').read_text())
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists')
    for path, checksum in selection['code'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen implementation changed: ' + path)
    if digest(OUT / 'provenance.json') != selection['provenance_sha256']:
        raise ValueError('Frozen provenance changed')
    frame = load_data('test').reset_index(drop=True)
    if not frame.role.eq('fresh_test').all() or not frame.huc.str[:2].isin(['01', '02', '04']).all():
        raise ValueError('Invalid final geography')
    records = []; predictions = {}
    for candidate in selection['candidates']:
        name = candidate['name']; artifact = OUT / 'artifacts' / f'{name}.joblib'
        if digest(artifact) != candidate['artifact_sha256']:
            raise ValueError('Frozen artifact changed')
        bundle = joblib.load(artifact)
        p = predict(bundle, frame); predictions[name] = p
        alerts = {k: p >= t for k, t in candidate['thresholds'].items()}
        alerts['oracle_1_diagnostic'] = p >= budget_threshold(frame.exceedance, p, .01)
        adaptive_thresholds = None
        if name in {selection['selected'], selection['direct_reference'], 'high_water'}:
            alerts['adaptive_1'], adaptive_thresholds = adaptive_alerts(frame, p, bundle['source_negatives'])
        record = dict(name=name, overall=metrics(frame, p, alerts), groups={})
        for column, keys in [('huc2', frame.huc.str[:2]), ('station', frame.station), ('year', frame.decision_date.str[:4])]:
            record['groups'][column] = {key: metrics(group, p[group.index], {k: mask[group.index] for k, mask in alerts.items()})
                                       for key, group in frame.groupby(keys)}
        if name == 'trajectory':
            record['trajectory'] = trajectory_metrics(frame, trajectory_inputs(bundle, frame)[0])
        saved = frame[['station', 'huc', 'decision_date', 'event_date', 'event_lead_hours', 'exceedance']].assign(probability=p)
        for key, mask in alerts.items():
            saved[key] = mask
        if adaptive_thresholds is not None:
            saved['adaptive_threshold'] = adaptive_thresholds
        saved.to_csv(OUT / 'predictions' / f'{name}.test.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        records.append(record)
        print(name, 'fresh AP', round(record['overall']['ap'], 4), flush=True)
    intervals = bootstrap(frame, predictions, selection)
    by_name = {r['name']: r for r in records}
    chosen = by_name[selection['selected']]; overall = chosen['overall']
    static = overall['policies']['pooled_1']; water = by_name['high_water']['overall']['policies']['pooled_1']
    adaptive = overall['policies']['adaptive_1']
    criteria = dict(adequate_events=static['events'] >= 30 and frame.huc.str[:4].nunique() >= 8,
        positive_ap_intervals=all(ci['ap_difference_95'][0] > 0 for pair in intervals.values() for ci in pair.values()),
        event_recall_gain=static['event_recall'] - water['event_recall'] >= .10,
        false_alert_control=static['false_alerts_per_100'] <= 1.5)
    adaptive_criteria = dict(recall_preserved=adaptive['event_recall'] >= static['event_recall'],
        fewer_false_alerts=adaptive['false_alerts'] < static['false_alerts'],
        regional_control=all(r['policies']['adaptive_1']['false_alerts_per_100'] <= 2 for r in chosen['groups']['huc2'].values()))
    write_json(OUT / 'evaluation.json', dict(selected=selection['selected'], direct_reference=selection['direct_reference'],
        candidates=records, paired_bootstrap=intervals, static_criteria=criteria, static_claim=all(criteria.values()),
        adaptive_criteria=adaptive_criteria, adaptive_claim=all(adaptive_criteria.values()),
        gauges=int(frame.station.nunique()), huc4=int(frame.huc.str[:4].nunique()),
        seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        selection_sha256=digest(OUT / 'selection.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'evaluate'])
    args = parser.parse_args()
    (fit if args.command == 'select' else evaluate)()
