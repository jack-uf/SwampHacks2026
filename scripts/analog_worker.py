"""Isolated workers for the frozen historical-transition experiment."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.analog_research import ROOT, OUT, ANALOG, digest, write_json, verify_protocol, load_data
from scripts.hydrology_research import COMBINED, SNOW, HYDRO, budget_threshold
from scripts.onset_worker import metrics
from scripts.response_research import partitions
from scripts.response_worker import score as previous_score, logit_input, paired_intervals

BASELINES = ['rainfall', 'logistic', 'direct', 'pooled']
CANDIDATES = ['prior', 'hydro_analog', 'combined_analog', 'offset']
CODE = ['scripts/analog_research.py', 'scripts/analog_worker.py', 'tests/test_analog_research.py']


def candidate_score(bundle, frame):
    name = bundle['name']
    if name == 'proximity':
        return frame[['q_3']].to_numpy()
    if name == 'prior':
        return frame[['analog_logit']].to_numpy()
    if name == 'offset':
        raw = bundle['model'].predict(frame[bundle['features']], raw_score=True)
        return (raw + frame.analog_logit.to_numpy()).reshape(-1, 1)
    if name in BASELINES:
        return previous_score(bundle, frame)
    return logit_input(bundle['model'].predict_proba(frame[bundle['features']])[:, 1])


def predict(bundle, frame):
    return bundle['calibrator'].predict_proba(candidate_score(bundle, frame))[:, 1]


def bounded(start):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 6_000_000_000 or time.monotonic() - start > 900:
        raise RuntimeError('Frozen resource budget exceeded')
    return rss


def select(family):
    verify_protocol()
    path = OUT / f'{family}.selection.json'
    if path.exists() or (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already exists')
    start = time.monotonic(); parts = partitions(load_data('training'))
    train, cal = parts['train'], parts['calibration']
    pooled = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    records = []
    (OUT / 'artifacts').mkdir(exist_ok=True)
    for name in ['pooled'] if family == 'neural' else ['rainfall', 'logistic', 'direct', 'proximity'] + CANDIDATES:
        begin = time.monotonic()
        if name in BASELINES:
            artifact = ROOT / f'experiments/sprint-20261017-response/artifacts/{name}.joblib'
            bundle = joblib.load(artifact)
        else:
            bundle = dict(name=name)
            if name in ['hydro_analog', 'combined_analog', 'offset']:
                from lightgbm import LGBMClassifier
                features = (HYDRO if name == 'hydro_analog' else COMBINED + SNOW) + ANALOG
                bundle['features'] = features
                model = LGBMClassifier(n_estimators=200, learning_rate=.04, max_depth=3, num_leaves=7,
                    n_jobs=2, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
                kwargs = dict(init_score=train.analog_logit) if name == 'offset' else {}
                bundle['model'] = model.fit(train[features], train.exceedance, **kwargs)
            bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(candidate_score(bundle, cal), cal.exceedance)
            artifact = OUT / 'artifacts' / f'{name}.joblib'
            joblib.dump(bundle, artifact, compress=3)
        p = predict(bundle, pooled)
        thresholds = {f'pooled_{budget}': budget_threshold(pooled.exceedance, p, budget/100) for budget in [.5, 1, 2]}
        thresholds['robust_1'] = max(budget_threshold(g.exceedance, p[g.index], .01) for _, g in pooled.groupby(pooled.huc.str[:2]))
        reports = {}
        for key in ['validation', 'geographic_validation']:
            f = parts[key]; pp = predict(bundle, f)
            reports[key] = metrics(f, pp, {k: pp >= t for k, t in thresholds.items()})
        records.append(dict(name=name, artifact=str(artifact.relative_to(ROOT)), artifact_sha256=digest(artifact),
            thresholds=thresholds, selection=reports, selection_ap=float(np.mean([r['ap'] for r in reports.values()])),
            selection_brier=float(np.mean([r['brier'] for r in reports.values()])), seconds=time.monotonic()-begin))
        bounded(start)
        print(name, 'selection AP', round(records[-1]['selection_ap'], 4), flush=True)
    write_json(path, dict(candidates=records, seconds=time.monotonic()-start, peak_rss_bytes=bounded(start),
        code={p: digest(ROOT/p) for p in CODE}, provenance_sha256=digest(OUT/'training.provenance.json'),
        partitions={k: dict(rows=len(f), gauges=int(f.station.nunique())) for k, f in parts.items()},
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'torch', 'joblib']}))


def freeze():
    verify_protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    families = [json.loads((OUT/f'{f}.selection.json').read_text()) for f in ['classic', 'neural']]
    records = [r for f in families for r in f['candidates']]
    for f in families:
        if f['provenance_sha256'] != digest(OUT/'training.provenance.json') or any(digest(ROOT/p) != h for p, h in f['code'].items()):
            raise ValueError('Selection inputs changed')
    order = lambda r: (-r['selection_ap'], r['selection_brier'], r['name'])
    write_json(OUT/'selection.json', dict(selected=min([r for r in records if r['name'] in CANDIDATES], key=order)['name'],
        comparator=min([r for r in records if r['name'] in ['direct', 'pooled']], key=order)['name'],
        candidates=records, code=families[0]['code'], provenance_sha256=digest(OUT/'training.provenance.json'),
        family_hashes={f:digest(OUT/f'{f}.selection.json') for f in ['classic', 'neural']}))


def verify_selection():
    verify_protocol()
    record = json.loads((OUT/'selection.json').read_text())
    paths = {**record['code'], str((OUT/'training.provenance.json').relative_to(ROOT)):record['provenance_sha256']}
    paths.update({str((OUT/f'{f}.selection.json').relative_to(ROOT)):h for f,h in record['family_hashes'].items()})
    paths.update({r['artifact']:r['artifact_sha256'] for r in record['candidates']})
    if any(digest(ROOT/p) != h for p,h in paths.items()):
        raise ValueError('Frozen selection changed')
    return record


def evaluate(family):
    selection = verify_selection()
    path = OUT/f'{family}.evaluation.json'
    if path.exists():
        raise FileExistsError('Final evaluation already exists')
    start = time.monotonic(); f = load_data('test').reset_index(drop=True)
    if not f.role.eq('fresh_test').all():
        raise ValueError('Invalid test roles')
    records = []
    (OUT/'predictions').mkdir(exist_ok=True)
    for r in selection['candidates']:
        if (r['name'] == 'pooled') != (family == 'neural'):
            continue
        bundle = joblib.load(ROOT/r['artifact']); begin = time.monotonic(); p = predict(bundle, f)
        inference = time.monotonic()-begin
        alerts = {k:p >= t for k,t in r['thresholds'].items()}
        alerts['oracle_1_diagnostic'] = p >= budget_threshold(f.exceedance, p, .01)
        alerts['strict_persistence'] = f.q_3.to_numpy() > np.log(2)
        record = dict(name=r['name'], overall=metrics(f,p,alerts), groups={}, inference_seconds=inference)
        for key,codes in [('station',f.station),('huc2',f.huc.str[:2]),('year',f.decision_date.str[:4])]:
            record['groups'][key] = {k:metrics(g,p[g.index],{a:m[g.index] for a,m in alerts.items()}) for k,g in f.groupby(codes)}
        hits = f.loc[alerts['pooled_1'] & f.exceedance.eq(1)].groupby(['station','event_date']).event_lead_hours.max()
        record['events_warned_48h'] = int(hits.ge(48).sum())
        saved = f[['station','huc','decision_date','event_date','event_lead_hours','exceedance']].assign(probability=p)
        for k,m in alerts.items():
            saved[k] = m
        saved.to_csv(OUT/'predictions'/f'{r["name"]}.test.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        records.append(record); bounded(start)
        print(r['name'], 'test AP', round(record['overall']['ap'],4), flush=True)
    write_json(path,dict(candidates=records,seconds=time.monotonic()-start,peak_rss_bytes=bounded(start),
        selection_sha256=digest(OUT/'selection.json'),test_provenance_sha256=digest(OUT/'test.provenance.json')))


def report():
    selection = verify_selection()
    if (OUT/'evaluation.json').exists():
        raise FileExistsError('Final report already exists')
    f = load_data('test').reset_index(drop=True)
    records = [r for family in ['classic','neural'] for r in json.loads((OUT/f'{family}.evaluation.json').read_text())['candidates']]
    by_name = {r['name']:r for r in records}
    predictions = {name:pd.read_csv(OUT/'predictions'/f'{name}.test.csv.gz',dtype={'station':str,'huc':str}).fillna({'event_date':''}) for name in by_name}
    name,base = selection['selected'],selection['comparator']
    intervals = paired_intervals(f,predictions[name],predictions[base])
    policy = by_name[name]['overall']['policies']['pooled_1']; baseline = by_name[base]['overall']['policies']['pooled_1']
    criteria = dict(adequate_population=policy['events'] >= 50 and f.huc.str[:4].nunique() >= 10,
        ap_improvement=all(v['ap_difference_95'][0] > 0 for v in intervals.values()),
        event_recall_gain=policy['event_recall']-baseline['event_recall'] >= .05,
        pooled_false_alerts=policy['false_alerts_per_100'] <= 1.5,
        regional_false_alerts=all(g['policies']['pooled_1']['false_alerts_per_100'] <= 3 for g in by_name[name]['groups']['huc2'].values()))
    leads = {}
    for n in [name,base]:
        mask = predictions[n].pooled_1 & f.exceedance.eq(1)
        leads[n] = f.loc[mask].groupby(['station','event_date']).event_lead_hours.max()
    common = leads[name].index.intersection(leads[base].index)
    difference = leads[name].reindex(common)-leads[base].reindex(common)
    write_json(OUT/'evaluation.json',dict(selected=name,comparator=base,criteria=criteria,improvement_claim=all(criteria.values()),
        paired_intervals=intervals,common_events=len(common),added_events=len(leads[name])-len(common),
        lost_events=len(leads[base])-len(common),median_common_lead_gain_hours=float(difference.median()) if len(common) else None,
        candidates=records,selection_sha256=digest(OUT/'selection.json'),
        evaluation_hashes={family:digest(OUT/f'{family}.evaluation.json') for family in ['classic','neural']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['select','freeze','evaluate','report'])
    parser.add_argument('--family',choices=['classic','neural'])
    args = parser.parse_args()
    if args.command in ['select','evaluate']:
        if not args.family:
            parser.error('--family is required')
        (select if args.command == 'select' else evaluate)(args.family)
    else:
        (freeze if args.command == 'freeze' else report)()
