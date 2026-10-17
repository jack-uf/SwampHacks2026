"""Frozen calibration selection and isolated baseline evaluation workers."""
import argparse
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd

from scripts.localcal_research import (ROOT, OUT, digest, write_json, fit_intercept, adjusted,
    calibration_rows, verify_protocol, load_data)
from scripts.analog_research import OUT as ANALOG, load_data as analog_data
from scripts.analog_worker import predict, verify_selection as verify_analog, bounded
from scripts.hydrology_research import budget_threshold
from scripts.response_research import partitions
from scripts.response_worker import paired_intervals
from scripts.onset_worker import metrics

METHODS = {'local': .1, 'pooled_2': 2, 'pooled_10': 10}
CODE = ['scripts/localcal_research.py', 'scripts/localcal_worker.py', 'scripts/localcal_cohort.py',
        'scripts/audit_localcal.py', 'tests/test_localcal_research.py']


def local_probabilities(frame, p, intercepts):
    if np.asarray(p).shape != (len(frame),) or not len(frame):
        raise ValueError("Invalid prediction length")
    missing = set(frame.station) - set(intercepts)
    if missing:
        raise ValueError('Missing historical calibration for ' + ','.join(sorted(missing)))
    return np.concatenate([adjusted([value], intercepts[sid])[0:1] for sid, value in zip(frame.station, p)])


def thresholds(frame, p):
    result = {f'pooled_{b}': budget_threshold(frame.exceedance, p, b/100) for b in [.5, 1, 2]}
    result['robust_1'] = max(budget_threshold(g.exceedance, p[g.index], .01) for _, g in frame.groupby(frame.huc.str[:2]))
    return result


def select():
    verify_protocol(); verify_analog()
    if (OUT/'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    started = time.monotonic()
    bundle = joblib.load(ANALOG/'artifacts/offset.joblib')
    cal = calibration_rows(load_data('calibration')); pcal = predict(bundle, cal)
    groups = {sid: dict(rows=len(g), positives=int(g.exceedance.sum()),
        first=g.decision_date.min(), last=g.decision_date.max()) for sid,g in cal.groupby('station')}
    offsets = {name: {sid:fit_intercept(pcal[g.index], g.exceedance, penalty)
        for sid,g in cal.groupby('station')} for name,penalty in METHODS.items()}
    parts = partitions(analog_data('training'))
    f = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    f = f[f.station.isin(groups)].reset_index(drop=True)
    pglobal = predict(bundle, f); records = []
    (OUT/'predictions').mkdir(exist_ok=True)
    for name in ['global'] + list(METHODS):
        p = pglobal if name == 'global' else local_probabilities(f, pglobal, offsets[name])
        cuts = thresholds(f, p)
        alerts = {k:p >= v for k,v in cuts.items()}
        overall = metrics(f, p, alerts)
        regions = {h:metrics(g,p[g.index],{k:a[g.index] for k,a in alerts.items()})
                   for h,g in f.groupby(f.huc.str[:2])}
        subsets = {key:metrics(g,p[g.index],{k:a[g.index] for k,a in alerts.items()})
                   for key,g in f.groupby(f.role.eq('geographic_validation').map({True:'geographic',False:'temporal'}))}
        worst = max(g['policies']['pooled_1']['false_alerts_per_100'] for g in regions.values())
        records.append(dict(name=name, thresholds=cuts, overall=overall, regions=regions, subsets=subsets,
            mean_recall=float(np.mean([g['policies']['pooled_1']['event_recall'] for g in subsets.values()])),
            mean_brier=float(np.mean([g['brier'] for g in subsets.values()])), worst_region=worst,
            acceptable=overall['policies']['pooled_1']['false_alerts_per_100'] <= 1.5 and worst <= 3))
        print(name, 'AP', round(overall['ap'],4), 'recall', round(records[-1]['mean_recall'],4), 'worst region', round(worst,3), flush=True)
    local = [r for r in records if r['name'] != 'global']; feasible = [r for r in local if r['acceptable']]
    order = (lambda r:(-r['mean_recall'],r['mean_brier'],r['name'])) if feasible else (lambda r:(r['worst_region'],-r['mean_recall'],r['mean_brier'],r['name']))
    selected = min(feasible or local, key=order)['name']
    write_json(OUT/'selection.json', dict(selected=selected, candidates=records, intercepts=offsets, calibration=groups,
        code={p:digest(ROOT/p) for p in CODE}, calibration_provenance_sha256=digest(OUT/'calibration.provenance.json'),
        calibration_dataset_sha256=digest(OUT/'calibration.csv.gz'), analog_selection_sha256=digest(ANALOG/'selection.json'),
        seconds=time.monotonic()-started, peak_rss_bytes=bounded(started),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']}))
    print('Selected',selected,flush=True)


def verify_selection():
    verify_protocol(); verify_analog()
    record = json.loads((OUT/'selection.json').read_text())
    paths = {**record['code'], str((OUT/'calibration.provenance.json').relative_to(ROOT)):record['calibration_provenance_sha256'],
             str((OUT/'calibration.csv.gz').relative_to(ROOT)):record['calibration_dataset_sha256'],
             str((ANALOG/'selection.json').relative_to(ROOT)):record['analog_selection_sha256']}
    for path, checksum in paths.items():
        if digest(ROOT/path) != checksum:
            raise ValueError('Frozen selection changed: ' + path)
    return record


def evaluate(family):
    selection = verify_selection()
    path = OUT/f'{family}.evaluation.json'
    if path.exists():
        raise FileExistsError('Evaluation already exists')
    started = time.monotonic(); f = load_data('test').reset_index(drop=True)
    if not f.role.eq('fresh_test').all() or not f.decision_date.between('2023-01-08','2024-09-25').all():
        raise ValueError('Invalid final population')
    names = ['pooled'] if family == 'neural' else ['global'] + list(METHODS) + ['rainfall','logistic','direct']
    records = []
    reference = joblib.load(ANALOG/'artifacts/offset.joblib') if family == 'classic' else None
    ranking_start = time.monotonic()
    pglobal = predict(reference, f) if reference else None
    ranking_seconds = time.monotonic()-ranking_start
    old = verify_analog()
    for name in names:
        begin = time.monotonic()
        if name == 'global' or name in METHODS:
            p = pglobal if name == 'global' else local_probabilities(f,pglobal,selection['intercepts'][name])
            cuts = next(r['thresholds'] for r in selection['candidates'] if r['name'] == name)
        else:
            r = next(r for r in old['candidates'] if r['name'] == name)
            p = predict(joblib.load(ROOT/r['artifact']),f); cuts = r['thresholds']
        inference = time.monotonic()-begin + (ranking_seconds if name == "global" or name in METHODS else 0)
        alerts = {k:p>=v for k,v in cuts.items()}
        alerts['strict_persistence'] = f.q_3.to_numpy() > np.log(2)
        r = dict(name=name, overall=metrics(f,p,alerts), groups={}, inference_seconds=inference)
        for key,codes in [('station',f.station),('huc2',f.huc.str[:2]),('year',f.decision_date.str[:4])]:
            r['groups'][key] = {h:metrics(g,p[g.index],{k:a[g.index] for k,a in alerts.items()}) for h,g in f.groupby(codes)}
        f[['station','huc','decision_date','event_date','event_lead_hours','exceedance']].assign(probability=p,
            **alerts).to_csv(OUT/'predictions'/f'{name}.test.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
        records.append(r); bounded(started)
        print(name,'AP',round(r['overall']['ap'],4),'event recall',round(r['overall']['policies']['pooled_1']['event_recall'],4),flush=True)
    write_json(path,dict(candidates=records, seconds=time.monotonic()-started, peak_rss_bytes=bounded(started),
        selection_sha256=digest(OUT/'selection.json'), test_provenance_sha256=digest(OUT/'test.provenance.json')))


def report():
    selection = verify_selection()
    if (OUT/'evaluation.json').exists():
        raise FileExistsError('Report already exists')
    f = load_data('test').reset_index(drop=True)
    records = [r for family in ['classic','neural'] for r in json.loads((OUT/f'{family}.evaluation.json').read_text())['candidates']]
    by_name = {r['name']:r for r in records}
    predictions = {name:pd.read_csv(OUT/'predictions'/f'{name}.test.csv.gz',dtype={'station':str,'huc':str},float_precision='round_trip').fillna({'event_date':''}) for name in by_name}
    name = selection['selected']; left,right = predictions[name],predictions['global']
    intervals = paired_intervals(f,left,right)
    y = f.exceedance.to_numpy(); delta = (left.probability.to_numpy()-y)**2-(right.probability.to_numpy()-y)**2
    for key,groups in [('huc4',f.huc.str[:4]),('month',f.decision_date.str[:7])]:
        codes,values = pd.factorize(groups); rng = np.random.default_rng(42); draws=[]
        for _ in range(500):
            w = np.bincount(rng.integers(0,len(values),len(values)),minlength=len(values))[codes]
            draws.append(float(np.average(delta,weights=w)))
        intervals[key]['brier_difference_95'] = np.quantile(draws,[.025,.975]).tolist()
    def warned(pred):
        return pred[pred.pooled_1 & pred.exceedance.eq(1)].groupby(['station','event_date']).event_lead_hours.max()
    l,r = warned(left),warned(right); common = l.index.intersection(r.index)
    lead = dict(added_events=len(l.index.difference(r.index)), lost_events=len(r.index.difference(l.index)), common_events=len(common),
        median_paired_lead_gain_hours=float((l.loc[common]-r.loc[common]).median()) if len(common) else None)
    policy = by_name[name]['overall']['policies']['pooled_1']; base = by_name['global']['overall']['policies']['pooled_1']
    criteria = dict(adequate_population=policy['events']>=50 and f.huc.str[:4].nunique()>=10,
        recall_gain=policy['event_recall']-base['event_recall']>=.05,
        recall_intervals=all(v['event_recall_difference_95'][0]>0 for v in intervals.values()),
        brier_improvement=all(v['brier_difference_95'][1]<0 for v in intervals.values()),
        pooled_false_alerts=policy['false_alerts_per_100']<=1.5,
        regional_false_alerts=all(g['policies']['pooled_1']['false_alerts_per_100']<=3 for g in by_name[name]['groups']['huc2'].values()),
        paired_lead=lead['median_paired_lead_gain_hours'] is not None and lead['median_paired_lead_gain_hours']>=0)
    write_json(OUT/'evaluation.json',dict(selected=name,candidates=records,paired_intervals=intervals,paired_lead=lead,
        criteria=criteria,passes=all(criteria.values()),gauges=int(f.station.nunique()),huc4=int(f.huc.str[:4].nunique()),
        selection_sha256=digest(OUT/'selection.json'),test_sha256=digest(OUT/'test.csv.gz'),
        evaluation_hashes={family:digest(OUT/f'{family}.evaluation.json') for family in ['classic','neural']}))
    print(name,'criteria',criteria,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['select','classic','neural','report'])
    command = parser.parse_args().command
    if command in ['classic','neural']:
        evaluate(command)
    else:
        (select if command == 'select' else report)()
