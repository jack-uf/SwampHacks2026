"""Separate classic/neural workers for the frozen catchment-response experiment."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import COMBINED, SNOW, budget_threshold, weighted_ap
from scripts.onset_worker import metrics
from scripts.response_research import (ROOT, OUT, digest, write_json, verify_protocol,
                                       load_data, partitions, response_features)

CLASSIC = ['rainfall', 'logistic', 'direct', 'response']
NEURAL = ['pooled', 'conditioned', 'multitask', 'uniform']
CODE = ['scripts/response_research.py', 'scripts/response_network.py',
        'scripts/response_worker.py', 'tests/test_response_research.py']


def logit_input(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def score(bundle, frame):
    if bundle['name'] == 'rainfall':
        return np.log1p(frame[['rain_3', 'rain_4', 'rain_5']].sum(axis=1)).to_numpy().reshape(-1, 1)
    if bundle['name'] in NEURAL:
        from scripts.response_network import network_output
        p = 1 - network_output(bundle, frame)[0][:, 0]
    else:
        p = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    return logit_input(p)


def predict(bundle, frame):
    return bundle['calibrator'].predict_proba(score(bundle, frame))[:, 1]


def select(family):
    verify_protocol()
    if (OUT / 'selection.json').exists() or (OUT / f'{family}.selection.json').exists():
        raise FileExistsError('Selection already exists')
    started = time.monotonic(); parts = partitions(load_data('training'))
    train, cal = parts['train'], parts['calibration']
    pooled = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    records = []
    (OUT / 'artifacts').mkdir(exist_ok=True); (OUT / 'predictions').mkdir(exist_ok=True)
    for name in CLASSIC if family == 'classic' else NEURAL:
        begin = time.monotonic(); bundle = dict(name=name)
        if name in NEURAL:
            from scripts.response_network import fit_network
            bundle.update(fit_network(name, train))
        elif name != 'rainfall':
            features = response_features() if name == 'response' else COMBINED + SNOW
            bundle['features'] = features
            if name == 'logistic':
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
            else:
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(n_estimators=200, learning_rate=.04, max_depth=3,
                    num_leaves=7, n_jobs=2, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
            bundle['model'] = model.fit(train[features], train.exceedance)
        bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(score(bundle, cal), cal.exceedance)
        p = predict(bundle, pooled)
        thresholds = {f'pooled_{budget}': budget_threshold(pooled.exceedance, p, budget / 100) for budget in [.5, 1, 2]}
        thresholds['robust_1'] = max(budget_threshold(g.exceedance, p[g.index], .01) for _, g in pooled.groupby(pooled.huc.str[:2]))
        bundle['thresholds'] = thresholds
        artifact = OUT / 'artifacts' / f'{name}.joblib'; joblib.dump(bundle, artifact, compress=3)
        reports = {}
        for key in ['validation', 'geographic_validation']:
            f = parts[key]; pp = predict(bundle, f)
            reports[key] = metrics(f, pp, {k: pp >= t for k, t in thresholds.items()})
        pooled[['station', 'huc', 'decision_date', 'exceedance', 'event_date', 'event_lead_hours']].assign(probability=p).to_csv(
            OUT / 'predictions' / f'{name}.selection.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        records.append(dict(name=name, thresholds=thresholds, artifact_sha256=digest(artifact), selection=reports,
            selection_ap=float(np.mean([r['ap'] for r in reports.values()])),
            selection_brier=float(np.mean([r['brier'] for r in reports.values()])),
            parameters=bundle.get('parameters'), seconds=time.monotonic() - begin))
        print(name, 'selection AP', round(records[-1]['selection_ap'], 4), flush=True)
    write_json(OUT / f'{family}.selection.json', dict(candidates=records, code={p: digest(ROOT / p) for p in CODE},
        seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        partitions={k: dict(rows=len(f), gauges=int(f.station.nunique()), positives=int(f.exceedance.sum()),
            events=len(f[f.exceedance.eq(1)].drop_duplicates(['station', 'event_date']))) for k, f in parts.items()},
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'torch', 'joblib', 'shapely']},
        provenance_sha256=digest(OUT / 'provenance.json')))


def freeze():
    verify_protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection is frozen')
    families = [json.loads((OUT / f'{name}.selection.json').read_text()) for name in ['classic', 'neural']]
    records = [r for family in families for r in family['candidates']]
    for family in families:
        for path, checksum in family['code'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Implementation changed between selection runs')
        if family['provenance_sha256'] != digest(OUT / 'provenance.json'):
            raise ValueError('Training sources changed')
    order = lambda r: (-r['selection_ap'], r['selection_brier'], r['name'])
    write_json(OUT / 'selection.json', dict(selected=min(records, key=order)['name'],
        baseline=min([r for r in records if r['name'] in ['rainfall', 'logistic', 'direct']], key=order)['name'],
        neural=min([r for r in records if r['name'] in NEURAL], key=order)['name'], candidates=records,
        code=families[0]['code'], provenance_sha256=digest(OUT / 'provenance.json'),
        family_hashes={name: digest(OUT / f'{name}.selection.json') for name in ['classic', 'neural']}))
    print('Selection frozen', flush=True)


def verify_selection():
    verify_protocol()
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, checksum in selection['code'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen implementation changed: ' + path)
    if selection['provenance_sha256'] != digest(OUT / 'provenance.json'):
        raise ValueError('Frozen provenance changed')
    for family, checksum in selection['family_hashes'].items():
        if digest(OUT / f'{family}.selection.json') != checksum:
            raise ValueError('Frozen candidate selection changed: ' + family)
    return selection


def evaluate(family):
    selection = verify_selection()
    if (OUT / f'{family}.evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists')
    started = time.monotonic(); f = load_data('test').reset_index(drop=True)
    if not f.role.eq('fresh_test').all():
        raise ValueError('Invalid final role')
    records = []
    for r in selection['candidates']:
        name = r['name']
        if name not in (CLASSIC if family == 'classic' else NEURAL):
            continue
        artifact = OUT / 'artifacts' / f'{name}.joblib'
        if digest(artifact) != r['artifact_sha256']:
            raise ValueError('Artifact changed')
        bundle = joblib.load(artifact); begin = time.monotonic(); p = predict(bundle, f)
        inference = time.monotonic() - begin
        alerts = {key: p >= t for key, t in r['thresholds'].items()}
        alerts['oracle_1_diagnostic'] = p >= budget_threshold(f.exceedance, p, .01)
        record = dict(name=name, overall=metrics(f, p, alerts), groups={}, inference_seconds=inference)
        for key, codes in [('huc2', f.huc.str[:2]), ('station', f.station), ('year', f.decision_date.str[:4])]:
            record['groups'][key] = {k: metrics(g, p[g.index], {a: mask[g.index] for a, mask in alerts.items()}) for k, g in f.groupby(codes)}
        hits = f.loc[alerts['pooled_1'] & f.exceedance.eq(1)].groupby(['station', 'event_date']).event_lead_hours.max()
        record['events_warned_48h'] = int(hits.ge(48).sum())
        if name in ['multitask', 'uniform']:
            from scripts.response_network import network_output
            trajectory = network_output(bundle, f)[1]
            truth = f[['flow_1', 'flow_2', 'flow_3']].to_numpy()
            record['trajectory_mae'] = np.mean(np.abs(trajectory - truth), axis=0).tolist()
            record['persistence_mae'] = np.mean(np.abs(f.q_3.to_numpy()[:, None] - truth), axis=0).tolist()
        saved = f[['station', 'huc', 'decision_date', 'event_date', 'event_lead_hours', 'exceedance']].assign(probability=p)
        for key, mask in alerts.items():
            saved[key] = mask
        saved.to_csv(OUT / 'predictions' / f'{name}.test.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
        records.append(record)
        print(name, 'fresh AP', round(record['overall']['ap'], 4), flush=True)
    write_json(OUT / f'{family}.evaluation.json', dict(candidates=records, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, selection_sha256=digest(OUT / 'selection.json')))


def paired_intervals(frame, left, right):
    result = {}; y = frame.exceedance.to_numpy(); positive = y == 1
    event_codes, events = pd.factorize(frame.loc[positive, 'station'] + ':' + frame.loc[positive, 'event_date'])
    for grouping, keys in [('huc4', frame.huc.str[:4]), ('month', frame.decision_date.str[:7])]:
        codes, groups = pd.factorize(keys); rng = np.random.default_rng(42); values = []
        for _ in range(500):
            weights = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[codes]
            if not (weights * y).sum() or not (weights * (1-y)).sum():
                continue
            recalls = []; event_weights = np.zeros(len(events))
            np.maximum.at(event_weights, event_codes, weights[positive])
            for prediction in [left, right]:
                hits = np.zeros(len(events), dtype=bool)
                np.logical_or.at(hits, event_codes, prediction.pooled_1.to_numpy()[positive] & (weights[positive] > 0))
                recalls.append(np.sum(event_weights * hits) / max(1, event_weights.sum()))
            values.append([weighted_ap(y, left.probability, weights) - weighted_ap(y, right.probability, weights), recalls[0] - recalls[1]])
        values = np.asarray(values)
        result[grouping] = dict(draws=len(values), ap_difference_95=np.quantile(values[:, 0], [.025, .975]).tolist(),
            event_recall_difference_95=np.quantile(values[:, 1], [.025, .975]).tolist())
    return result


def report():
    selection = verify_selection()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Evaluation report already exists')
    f = load_data('test').reset_index(drop=True)
    records = [r for family in ['classic', 'neural'] for r in json.loads((OUT / f'{family}.evaluation.json').read_text())['candidates']]
    by_name = {r['name']: r for r in records}; base = by_name[selection['baseline']]
    predictions = {name: pd.read_csv(OUT / 'predictions' / f'{name}.test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''}) for name in by_name}
    pairs = {}; claims = {}; lead_comparisons = {}
    for name in dict.fromkeys([selection['selected'], selection['neural']]):
        pairs[name] = paired_intervals(f, predictions[name], predictions[selection['baseline']])
        selected = by_name[name]; policy = selected['overall']['policies']['pooled_1']; reference = base['overall']['policies']['pooled_1']
        criteria = dict(response_conditioned=name in ['response', 'conditioned', 'multitask'],
            adequate_population=policy['events'] >= 50 and f.huc.str[:4].nunique() >= 10,
            ap_improvement=all(v['ap_difference_95'][0] > 0 for v in pairs[name].values()),
            event_recall_gain=policy['event_recall'] - reference['event_recall'] >= .05,
            pooled_false_alerts=policy['false_alerts_per_100'] <= 1.5,
            regional_false_alerts=all(g['policies']['pooled_1']['false_alerts_per_100'] <= 3 for g in selected['groups']['huc2'].values()))
        claims[name] = dict(criteria=criteria, passes=all(criteria.values()))
        def warned(pred):
            return pred[pred.pooled_1 & pred.exceedance.eq(1)].groupby(['station', 'event_date']).event_lead_hours.max()
        left, right = warned(predictions[name]), warned(predictions[selection['baseline']])
        common = left.index.intersection(right.index)
        lead_comparisons[name] = dict(added_events=len(left.index.difference(right.index)), lost_events=len(right.index.difference(left.index)),
            common_events=len(common), median_paired_lead_gain_hours=float((left.loc[common] - right.loc[common]).median()) if len(common) else None)
    write_json(OUT / 'evaluation.json', dict(selected=selection['selected'], neural=selection['neural'], baseline=selection['baseline'],
        candidates=records, paired_bootstrap=pairs, claims=claims, paired_lead=lead_comparisons,
        gauges=int(f.station.nunique()), huc4=int(f.huc.str[:4].nunique()), selection_sha256=digest(OUT / 'selection.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['classic', 'neural', 'freeze', 'evaluate-classic', 'evaluate-neural', 'report'])
    args = parser.parse_args()
    if args.command in ['classic', 'neural']:
        select(args.command)
    elif args.command.startswith('evaluate-'):
        evaluate(args.command.removeprefix('evaluate-'))
    else:
        (freeze if args.command == 'freeze' else report)()
