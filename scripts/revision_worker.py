"""Rolling-origin evaluation of the bridge and revision inputs; research only."""
import argparse
import importlib.metadata
import json
import resource
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from scripts.analog_research import ANALOG
from scripts.hydrology_research import COMBINED, SNOW, budget_threshold
from scripts.onset_worker import metrics
from scripts.response_worker import paired_intervals
from scripts.revision_research import (ROOT, OUT, BRIDGE, REVISION, digest, verify_protocol, load_data)

BASE = COMBINED + SNOW + ANALOG
CANDIDATES = dict(control=[], bridge=BRIDGE, revision=REVISION, combined=BRIDGE + REVISION)
FOLDS = [dict(name='2021', train=('2017-01-08', '2019-12-24'), calibration=('2020-01-08', '2020-12-24'),
              validation=('2021-01-08', '2021-12-24')),
         dict(name='2022', train=('2017-01-08', '2020-12-24'), calibration=('2021-01-08', '2021-12-24'),
              validation=('2022-01-08', '2022-12-24'))]
PARAMS = dict(n_estimators=200, learning_rate=.04, max_depth=3, num_leaves=7, n_jobs=2,
              random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
SUBSETS = ['temporal', 'geographic']
TOLERANCE = 1e-9  # criterion boundaries stay inclusive under binary floating point rounding


def folds(frame):
    """Two forward-only fold partitions; inspected development rows only."""
    if frame.role.eq('fresh_test').any() or not frame.role.isin(['development', 'geographic_validation']).all():
        raise ValueError('Only inspected development populations may enter fold selection')
    if frame.duplicated(['station', 'decision_date']).any():
        raise ValueError('Duplicate decisions')
    middle = frame.decision_date.str[5:].between('01-08', '12-24')
    result = []
    for spec in FOLDS:
        development, geographic = frame.role.eq('development'), frame.role.eq('geographic_validation')
        parts = dict(
            train=frame[middle & development & frame.decision_date.between(*spec['train'])],
            calibration=frame[middle & development & frame.decision_date.between(*spec['calibration'])],
            temporal=frame[middle & development & frame.decision_date.between(*spec['validation'])],
            geographic=frame[middle & geographic & frame.decision_date.between(*spec['validation'])])
        seen = set()
        for key, part in parts.items():
            if not len(part):
                raise ValueError(f'Empty {key} population for fold {spec["name"]}')
            decisions = set(zip(part.station, part.decision_date))
            if len(decisions) != len(part):
                raise ValueError('Duplicate decisions inside a fold population')
            if decisions & seen:
                raise ValueError(f'Overlapping fold populations in {spec["name"]}: {key}')
            seen |= decisions
            if part.exceedance.sum() < 5 or not part.exceedance.between(0, 1).all():
                raise ValueError(f'Insufficient positives in {key} for fold {spec["name"]}')
        result.append(dict(name=spec['name'], **{key: part.reset_index(drop=True) for key, part in parts.items()}))
    return result


def raw_score(bundle, frame):
    return bundle['model'].predict(frame[bundle['features']], raw_score=True) + frame.analog_logit.to_numpy()


def probability(bundle, frame):
    return bundle['calibrator'].predict_proba(raw_score(bundle, frame).reshape(-1, 1))[:, 1]


def thresholds(frame, p):
    result = {f'pooled_{budget}': budget_threshold(frame.exceedance, p, budget / 100) for budget in [.5, 1, 2]}
    result['robust_1'] = max(budget_threshold(g.exceedance, p[g.index], .01)
                             for _, g in frame.groupby(frame.huc.str[:2]))
    return result


def alerts(p, cuts):
    return {name: p >= value for name, value in cuts.items()}


def select():
    verify_protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    started = time.monotonic()
    parts = folds(load_data())
    (OUT / 'artifacts').mkdir(parents=True, exist_ok=True)
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    records = []
    for fold in parts:
        for name, extra in CANDIDATES.items():
            begin = time.monotonic()
            features = BASE + extra
            bundle = dict(name=name, features=features)
            bundle['model'] = LGBMClassifier(**PARAMS).fit(fold['train'][features], fold['train'].exceedance,
                                                           init_score=fold['train'].analog_logit)
            bundle['calibrator'] = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
                raw_score(bundle, fold['calibration']).reshape(-1, 1), fold['calibration'].exceedance)
            predictions = {key: probability(bundle, fold[key]) for key in SUBSETS}
            pooled = pd.concat([fold[key] for key in SUBSETS], ignore_index=True)
            pooled_p = np.concatenate([predictions[key] for key in SUBSETS])
            cuts = thresholds(pooled, pooled_p)
            artifact = OUT / 'artifacts' / f'{fold["name"]}.{name}.joblib'
            joblib.dump(bundle, artifact, compress=3)
            path = OUT / 'predictions' / f'{fold["name"]}.{name}.csv.gz'
            pooled.assign(probability=pooled_p, **alerts(pooled_p, cuts)).to_csv(
                path, index=False, compression=dict(method='gzip', mtime=0))
            records.append(dict(fold=fold['name'], name=name, thresholds=cuts,
                artifact_sha256=digest(artifact), predictions_sha256=digest(path),
                subsets={key: metrics(fold[key], p, alerts(p, cuts)) for key, p in predictions.items()},
                pooled=metrics(pooled, pooled_p, alerts(pooled_p, cuts)),
                regions={huc2: dict(policies=metrics(g, pooled_p[g.index],
                    alerts(pooled_p[g.index], cuts))['policies'])
                    for huc2, g in pooled.groupby(pooled.huc.str[:2])},
                seconds=time.monotonic() - begin))
            print(fold['name'], name, 'mean AP',
                  round(float(np.mean([v['ap'] for v in records[-1]['subsets'].values()])), 4), flush=True)
    summary = {name: dict(mean_ap=float(np.mean([r['subsets'][key]['ap'] for r in records if r['name'] == name
                                                 for key in SUBSETS])),
                          mean_brier=float(np.mean([r['subsets'][key]['brier'] for r in records if r['name'] == name
                                                    for key in SUBSETS])))
               for name in CANDIDATES}
    selected = min(summary, key=lambda name: (-summary[name]['mean_ap'], summary[name]['mean_brier'], name))
    write_selection = dict(candidates=records, summary=summary, selected=selected, control='control',
        folds=FOLDS, protocol_sha256=digest(OUT / 'protocol.json'), seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'joblib']})
    (OUT / 'selection.json').write_text(json.dumps(write_selection, indent=2, allow_nan=False) + '\n')
    print('Selected', selected, flush=True)


def verify_selection():
    verify_protocol()
    record = json.loads((OUT / 'selection.json').read_text())
    if record['protocol_sha256'] != digest(OUT / 'protocol.json'):
        raise ValueError('Frozen protocol changed after selection')
    for candidate in record['candidates']:
        for key, kind in [('artifact_sha256', 'artifacts'), ('predictions_sha256', 'predictions')]:
            name = ('joblib' if kind == 'artifacts' else 'csv.gz')
            if digest(OUT / kind / f'{candidate["fold"]}.{candidate["name"]}.{name}') != candidate[key]:
                raise ValueError('Frozen selection artifact changed: ' + candidate['name'])
    return record


def gate(records, intervals, selected, control):
    """Precommitted reliability gate; every criterion must hold in every fold."""
    if selected == control:
        return dict(criteria={'new_candidate_selected': False}, passes=False)
    policy = lambda fold, name: records[fold][name]['pooled']['policies']['pooled_1']
    criteria = dict(
        subset_ap=all(records[fold][selected][key]['ap'] >= records[fold][control][key]['ap']
                      for fold in records for key in SUBSETS),
        subset_brier=all(records[fold][selected][key]['brier'] <= records[fold][control][key]['brier']
                         for fold in records for key in SUBSETS),
        recall_gain=all(policy(fold, selected)['event_recall'] - policy(fold, control)['event_recall'] >= .05 - TOLERANCE
                        for fold in records),
        pooled_false_alerts=all(policy(fold, selected)['false_alerts_per_100'] <= 1.5 + TOLERANCE for fold in records),
        regional_false_alerts=all(region['policies']['pooled_1']['false_alerts_per_100'] <= 3 + TOLERANCE
                                  for fold in records for region in records[fold][selected]['regions'].values()),
        paired_ap=all(value['ap_difference_95'][0] > 0 for fold in records for value in intervals[fold].values()),
        paired_recall=all(value['event_recall_difference_95'][0] > 0
                          for fold in records for value in intervals[fold].values()))
    return dict(criteria=criteria, passes=all(criteria.values()))


def predictions(name):
    # The default float parser is not round-trip exact; boundary ties must replay identically.
    frame = pd.read_csv(OUT / 'predictions' / f'{name}.csv.gz', dtype={'station': str, 'huc': str},
                        float_precision='round_trip')
    return frame.fillna({'event_date': ''})


def report():
    selection = verify_selection()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Report already exists')
    started = time.monotonic()
    parts = folds(load_data())
    records, frames, intervals = {}, {}, {}
    for fold in parts:
        pooled = pd.concat([fold[key] for key in SUBSETS], ignore_index=True)
        for name in CANDIDATES:
            saved = predictions(f'{fold["name"]}.{name}')
            stored = next(r for r in selection['candidates'] if r['fold'] == fold['name'] and r['name'] == name)
            if [tuple(x) for x in zip(saved.station, saved.decision_date)] != [
                    tuple(x) for x in zip(pooled.station, pooled.decision_date)]:
                raise ValueError('Saved predictions disagree with frozen decisions')
            if np.any(saved.exceedance.to_numpy() != pooled.exceedance.to_numpy()):
                raise ValueError('Saved labels disagree with the frozen dataset')
            for key, value in stored['thresholds'].items():
                if not np.array_equal(saved[key].to_numpy(bool), saved.probability.to_numpy() >= value):
                    raise ValueError('Saved alerts disagree with frozen thresholds')
            records.setdefault(fold['name'], {})[name] = dict(
                temporal=stored['subsets']['temporal'], geographic=stored['subsets']['geographic'],
                pooled=stored['pooled'], regions=stored['regions'])
            frames.setdefault(fold['name'], {})[name] = saved
        intervals[fold['name']] = {name: paired_intervals(pooled, frames[fold['name']][name],
                                                          frames[fold['name']]['control'])
                                   for name in CANDIDATES if name != 'control'}
    selected = selection['selected']
    gates = {name: gate(records, {fold: intervals[fold][name] for fold in intervals}, name, 'control')
             for name in CANDIDATES if name != 'control'}
    write = dict(selected=selected, control='control',
        summary=selection['summary'], gate=gates[selected], gates=gates, intervals=intervals,
        decision='gate passed' if gates[selected]['passes'] else 'gate failed; no fresh cohort opened',
        seconds=time.monotonic() - started, selection_sha256=digest(OUT / 'selection.json'))
    (OUT / 'evaluation.json').write_text(json.dumps(write, indent=2, allow_nan=False) + '\n')
    print('Selected', selected, 'gate', gates[selected]['passes'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'report'])
    (select if parser.parse_args().command == 'select' else report)()
