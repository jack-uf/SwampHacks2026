"""Frozen supervision ablation and regional warning-policy replay."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from scripts.hydrology_research import budget_threshold
from scripts.onset_worker import metrics
from scripts.response_research import OUT as PREVIOUS, load_data, partitions
from scripts.response_worker import predict as old_predict, logit_input, paired_intervals
from scripts.warning_research import ROOT, OUT, digest, write_json, verify_protocol, load_test, budget_alerts

FAMILIES = dict(classic=['rainfall', 'logistic', 'direct'], neural=['pooled', 'binary', 'day'])
CODE = ['scripts/warning_research.py', 'scripts/warning_network.py',
        'scripts/warning_worker.py', 'tests/test_warning_research.py']


def ensemble_raw(bundle, frame):
    from scripts.response_network import network_output
    return np.mean([1 - network_output(member, frame)[0][:, 0] for member in bundle['members']], axis=0)


def predict(bundle, frame):
    if bundle['name'] in ['binary', 'day']:
        return bundle['calibrator'].predict_proba(logit_input(ensemble_raw(bundle, frame)))[:, 1]
    return old_predict(bundle, frame)


def policies(frame, p, thresholds):
    return dict(static_1=p >= thresholds['static_1'], static_5=p >= thresholds['static_5'],
        robust_1=p >= thresholds['robust_1'],
        bucket_1=budget_alerts(frame, p, thresholds['static_1']),
        bucket_5=budget_alerts(frame, p, thresholds['static_5']))


def select(family):
    verify_protocol()
    if (OUT / f'{family}.selection.json').exists() or (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already exists')
    parts = partitions(load_data('training'))
    train, cal = parts['train'], parts['calibration']
    pooled = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    prior = json.loads((PREVIOUS / 'selection.json').read_text())
    prior = {r['name']: r for r in prior['candidates']}
    (OUT / 'artifacts').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    records = []; started = time.monotonic()
    for name in FAMILIES[family]:
        begin = time.monotonic(); seed_results = []
        if name in ['binary', 'day']:
            from scripts.warning_network import fit_matched
            bundle = dict(name=name, members=[])
            for seed in [42, 314, 2718]:
                member = fit_matched(name, train, seed)
                from scripts.response_network import network_output
                seed_results.append(dict(seed=seed, ap={key: float(average_precision_score(parts[key].exceedance,
                    1 - network_output(member, parts[key])[0][:, 0])) for key in ['validation', 'geographic_validation']}))
                bundle['members'].append(member)
            bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
                logit_input(ensemble_raw(bundle, cal)), cal.exceedance)
            artifact = OUT / 'artifacts' / f'{name}.joblib'
            joblib.dump(bundle, artifact, compress=3)
        else:
            artifact = PREVIOUS / 'artifacts' / f'{name}.joblib'
            if digest(artifact) != prior[name]['artifact_sha256']:
                raise ValueError('Prior artifact changed')
            bundle = joblib.load(artifact)
        p = predict(bundle, pooled)
        thresholds = {f'static_{n}': budget_threshold(pooled.exceedance, p, n / 100) for n in [1, 5]}
        thresholds['robust_1'] = max(budget_threshold(g.exceedance, p[g.index], .01) for _, g in pooled.groupby(pooled.huc.str[:2]))
        reports = {}
        for key in ['validation', 'geographic_validation']:
            f = parts[key].reset_index(drop=True); pp = predict(bundle, f)
            reports[key] = metrics(f, pp, policies(f, pp, thresholds))
        chosen = min(['bucket_1', 'bucket_5'], key=lambda policy: (
            -np.mean([report['policies'][policy]['event_recall'] for report in reports.values()]), policy))
        records.append(dict(name=name, artifact=str(artifact.relative_to(ROOT)), artifact_sha256=digest(artifact),
            thresholds=thresholds, policy=chosen, selection=reports, seed_results=seed_results,
            selection_ap=float(np.mean([r['ap'] for r in reports.values()])),
            selection_brier=float(np.mean([r['brier'] for r in reports.values()])), seconds=time.monotonic() - begin))
        print(name, 'selection AP', round(records[-1]['selection_ap'], 4), 'policy', chosen, flush=True)
    write_json(OUT / f'{family}.selection.json', dict(candidates=records, code={p: digest(ROOT / p) for p in CODE},
        seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        provenance_sha256=digest(OUT / 'provenance.json'),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'torch', 'joblib', 'shapely']}))


def freeze():
    verify_protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    families = [json.loads((OUT / f'{family}.selection.json').read_text()) for family in FAMILIES]
    for family in families:
        for path, checksum in family['code'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Implementation changed between selection runs')
        if family['provenance_sha256'] != digest(OUT / 'provenance.json'):
            raise ValueError('Provenance changed')
    candidates = [r for family in families for r in family['candidates']]
    ensembles = [r for r in candidates if r['name'] in ['binary', 'day']]
    selected = min(ensembles, key=lambda r: (-r['selection_ap'], r['selection_brier'], r['name']))['name']
    write_json(OUT / 'selection.json', dict(selected=selected, candidates=candidates, code=families[0]['code'],
        provenance_sha256=digest(OUT / 'provenance.json'),
        family_hashes={family: digest(OUT / f'{family}.selection.json') for family in FAMILIES}))
    print('Frozen ensemble choice:', selected, flush=True)


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
            raise ValueError('Frozen selection changed')
    return selection


def evaluate(family):
    selection = verify_selection()
    if (OUT / f'{family}.evaluation.json').exists():
        raise FileExistsError('Evaluation already exists')
    f = load_test().reset_index(drop=True); records = []; started = time.monotonic()
    if not f.role.eq('fresh_test').all():
        raise ValueError('Invalid final role')
    for r in selection['candidates']:
        if r['name'] not in FAMILIES[family]:
            continue
        artifact = ROOT / r['artifact']
        if digest(artifact) != r['artifact_sha256']:
            raise ValueError('Artifact changed')
        bundle = joblib.load(artifact); begin = time.monotonic(); p = predict(bundle, f)
        elapsed = time.monotonic() - begin
        alerts = policies(f, p, r['thresholds'])
        alerts['oracle_1_diagnostic'] = p >= budget_threshold(f.exceedance, p, .01)
        groups = {}
        for key, codes in [('huc2', f.huc.str[:2]), ('station', f.station), ('year', f.decision_date.str[:4])]:
            groups[key] = {k: metrics(g, p[g.index], {a: mask[g.index] for a, mask in alerts.items()}) for k, g in f.groupby(codes)}
        warned = {name: int(f.loc[mask & f.exceedance.eq(1)].groupby(['station', 'event_date']).event_lead_hours.max().ge(48).sum()) for name, mask in alerts.items()}
        records.append(dict(name=r['name'], overall=metrics(f, p, alerts), groups=groups,
            inference_seconds=elapsed, events_warned_48h=warned))
        saved = f[['station', 'huc', 'decision_date', 'event_date', 'event_lead_hours', 'exceedance']].assign(probability=p)
        for key, mask in alerts.items():
            saved[key] = mask
        saved.to_csv(OUT / 'predictions' / f"{r['name']}.test.csv.gz", index=False, compression=dict(method='gzip', mtime=0))
        print(r['name'], 'fresh AP', round(records[-1]['overall']['ap'], 4), flush=True)
    write_json(OUT / f'{family}.evaluation.json', dict(candidates=records, seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, selection_sha256=digest(OUT / 'selection.json')))


def report():
    selection = verify_selection()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Report already exists')
    f = load_test().reset_index(drop=True)
    records = [r for family in FAMILIES for r in json.loads((OUT / f'{family}.evaluation.json').read_text())['candidates']]
    by_name = {r['name']: r for r in records}
    predictions = {name: pd.read_csv(OUT / 'predictions' / f'{name}.test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''}) for name in by_name}
    def comparison(left, right, left_policy='static_1', right_policy='static_1'):
        a, b = predictions[left].copy(), predictions[right].copy()
        a['pooled_1'] = a[left_policy]; b['pooled_1'] = b[right_policy]
        return paired_intervals(f, a, b)
    pairs = {f'{a}_vs_{b}': comparison(a, b) for a, b in [('day', 'binary'), ('day', 'direct'), ('binary', 'direct')]}
    selected = selection['selected']; r = next(r for r in selection['candidates'] if r['name'] == selected)
    policy = r['policy']
    pairs['budget_vs_static'] = comparison(selected, selected, policy)
    adequate = f.huc.str[:4].nunique() >= 10 and by_name[selected]['overall']['policies']['static_1']['events'] >= 50
    day, binary = [by_name[name]['overall']['policies']['static_1'] for name in ['day', 'binary']]
    supervision = dict(adequate_population=bool(adequate), ap_intervals_positive=all(v['ap_difference_95'][0] > 0 for v in pairs['day_vs_binary'].values()),
        event_recall_preserved=day['event_recall'] >= binary['event_recall'])
    chosen, static = [by_name[selected]['overall']['policies'][name] for name in [policy, 'static_1']]
    budget = dict(adequate_population=bool(adequate), event_recall_preserved=chosen['event_recall'] >= static['event_recall'],
        fewer_false_alerts=chosen['false_alerts'] < static['false_alerts'],
        regional_budget=all(g['policies'][policy]['false_alerts_per_100'] <= 1 for g in by_name[selected]['groups']['huc2'].values()))
    def warned(mask):
        pred = predictions[selected]
        return pred[mask & pred.exceedance.eq(1)].groupby(['station', 'event_date']).event_lead_hours.max()
    pred = predictions[selected]; a, b = warned(pred[policy]), warned(pred.static_1)
    common = a.index.intersection(b.index)
    lead = dict(added_events=len(a.index.difference(b.index)), lost_events=len(b.index.difference(a.index)),
        common_events=len(common), median_paired_lead_gain_hours=float((a.loc[common] - b.loc[common]).median()) if len(common) else None)
    write_json(OUT / 'evaluation.json', dict(selected=selected, selected_policy=policy, candidates=records,
        paired_bootstrap=pairs, supervision_claim=dict(criteria=supervision, passes=all(supervision.values())),
        budget_claim=dict(criteria=budget, passes=all(budget.values())), budget_lead=lead,
        gauges=int(f.station.nunique()), huc4=int(f.huc.str[:4].nunique()), huc2=int(f.huc.str[:2].nunique()),
        selection_sha256=digest(OUT / 'selection.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['classic', 'neural', 'freeze', 'evaluate-classic', 'evaluate-neural', 'report'])
    command = parser.parse_args().command
    if command in FAMILIES:
        select(command)
    elif command.startswith('evaluate-'):
        evaluate(command.removeprefix('evaluate-'))
    else:
        (freeze if command == 'freeze' else report)()
