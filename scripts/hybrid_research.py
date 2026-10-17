"""Expert complementarity and hybrid combination, evaluated forward in time."""
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import ROOT, budget_threshold, digest, write_json
from scripts.onset_worker import metrics
from scripts.response_research import SIGNATURE
from scripts.response_worker import paired_intervals
from scripts.revision_worker import BASE, PARAMS, folds

OUT = ROOT / 'experiments/sprint-20261017-hybrid'
REVISION = ROOT / 'experiments/sprint-20261017-revision'
EXPERTS = ['gbm', 'neural', 'analog', 'rainfall', 'logistic']
COMBINERS = ['mean', 'stack', 'gated']
GATING = SIGNATURE + ['log_area', 'elevation_m', 'season_sin', 'season_cos', 'snow_store',
                      'history_complete', 'q_3', 'rain_spread_0', 'rain_spread_1', 'rain_mean_0']
GATE_PARAMS = dict(n_estimators=200, learning_rate=.04, max_depth=3, num_leaves=7, n_jobs=2,
                   random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
FIT, EVALUATION = '2021', '2022'
BUDGETS = [.5, 1, 2]
TOLERANCE = 1e-9


def logits(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mean_score(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('Expert score matrix must be two dimensional')
    return matrix.mean(axis=1)


def alert_cuts(frame, score):
    result = {f'pooled_{budget}': budget_threshold(frame.exceedance, score, budget / 100) for budget in BUDGETS}
    result['robust_1'] = max(budget_threshold(group.exceedance, score[group.index], .01)
                             for _, group in frame.groupby(frame.huc.str[:2]))
    return result


def complementarity(frame, probabilities, cuts):
    """Are the experts' detections complements or copies? Diagnostic only."""
    positive = frame.exceedance.eq(1).to_numpy()
    positions = np.flatnonzero(positive)
    events = list(dict.fromkeys((frame.station[i], frame.event_date[i]) for i in positions))
    masks = {name: np.asarray(p) >= cuts['pooled_1'][name] for name, p in probabilities.items()}
    detected = {name: {key for i, key in ((i, (frame.station[i], frame.event_date[i])) for i in positions)
                       if masks[name][i]} for name in probabilities}
    union = set().union(*detected.values()) if detected else set()
    shared = set.intersection(*detected.values()) if detected else set()
    single = {name: len(detected[name] - set().union(*[v for key, v in detected.items() if key != name]))
              for name in detected}
    names = list(probabilities)
    agreement = {f'{left},{right}': float(np.mean(masks[left] == masks[right]))
                 for index, left in enumerate(names) for right in names[index + 1:]}
    return dict(events=len(events), union_recall=len(union) / len(events) if events else None,
                shared_events=len(shared), single_expert_events=single,
                missed_by_all=len(events) - len(union), pairwise_agreement=agreement)


def fit_combiner(name, frame, matrix, gating):
    """Fit one combination rule on the fit fold only. Predictions must be out-of-fold."""
    matrix = np.asarray(matrix, dtype=float)
    if name not in COMBINERS:
        raise ValueError('Unknown combiner name: ' + str(name))
    if matrix.ndim != 2 or matrix.shape[1] != len(EXPERTS):
        raise ValueError('Expert score matrix must carry one column per expert')
    if set(frame.decision_date.str[:4]) != {FIT}:
        raise ValueError(f'Combiners fit on {FIT} development rows only')
    if len(frame) != len(matrix) or len(frame) != len(gating):
        raise ValueError('Mismatched combiner training rows')
    columns = gating[GATING].to_numpy(dtype=float) if name == 'gated' else None
    if columns is not None and not np.isfinite(columns).all():
        raise ValueError('Gating features are missing for some training rows')
    if name == 'mean':
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(mean_score(matrix).reshape(-1, 1), frame.exceedance)
    elif name == 'stack':
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(matrix, frame.exceedance)
    else:
        model = LGBMClassifier(**GATE_PARAMS).fit(np.column_stack([matrix, columns]), frame.exceedance)
    return dict(name=name, model=model, fit_rows=len(frame))


def combiner_score(bundle, matrix, gating):
    matrix = np.asarray(matrix, dtype=float)
    if bundle['name'] == 'mean':
        return bundle['model'].predict_proba(mean_score(matrix).reshape(-1, 1))[:, 1]
    if bundle['name'] == 'stack':
        return bundle['model'].predict_proba(matrix)[:, 1]
    return bundle['model'].predict_proba(np.column_stack([matrix, gating[GATING].to_numpy(dtype=float)]))[:, 1]


def verify_protocol():
    record = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in record['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen hybrid input changed: ' + path)
    return record


def load_folds():
    provenance = json.loads((REVISION / 'training.provenance.json').read_text())
    path = REVISION / 'training.csv.gz'
    if digest(path) != provenance['dataset_sha256']:
        raise ValueError('Frozen revision dataset changed')
    return folds(pd.read_csv(path, dtype={'station': str, 'huc': str}).fillna({'event_date': ''}))


def fit_expert(name, fold):
    train, calibration = fold['train'], fold['calibration']
    if name == 'gbm':
        model = LGBMClassifier(**PARAMS).fit(train[BASE], train.exceedance, init_score=train.analog_logit)
        score = lambda frame: model.predict(frame[BASE], raw_score=True) + frame.analog_logit.to_numpy()
        bundle = dict(model=model)
    elif name == 'neural':
        from scripts.response_network import fit_network, network_output
        bundle = fit_network('pooled', train)
        score = lambda frame: logits(1 - network_output(bundle, frame)[0][:, 0])
    elif name == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
        model.fit(train[BASE], train.exceedance)
        score = lambda frame: logits(model.predict_proba(frame[BASE])[:, 1])
        bundle = dict(model=model)
    elif name == 'analog':
        score = lambda frame: frame.analog_logit.to_numpy()
        bundle = dict()
    elif name == 'rainfall':
        score = lambda frame: np.log1p(frame[['rain_3', 'rain_4', 'rain_5']].sum(axis=1)).to_numpy()
        bundle = dict()
    else:
        raise ValueError('Unknown expert: ' + name)
    calibrator = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(
        np.asarray(score(calibration)).reshape(-1, 1), calibration.exceedance)
    bundle.update(name=name, calibrator=calibrator, parameters=bundle.get('parameters'))
    return bundle, lambda frame: calibrator.predict_proba(np.asarray(score(frame)).reshape(-1, 1))[:, 1]


def expert_probabilities(fold):
    frame = pd.concat([fold['temporal'], fold['geographic']], ignore_index=True)
    values, bundles = {}, {}
    for expert in EXPERTS:
        bundle, probability = fit_expert(expert, fold)
        artifact = OUT / 'artifacts' / f'{fold["name"]}.{expert}.joblib'
        joblib.dump(bundle, artifact, compress=3)
        values[expert] = probability(frame)
        bundles[expert] = bundle
    cuts = {expert: budget_threshold(frame.exceedance, p, .01) for expert, p in values.items()}
    frame.assign(**{f'p_{expert}': p for expert, p in values.items()}).to_csv(
        OUT / 'predictions' / f'{fold["name"]}.experts.csv.gz', index=False,
        compression=dict(method='gzip', mtime=0))
    return dict(frame=frame, values=values, bundles=bundles, cuts=cuts,
                metrics={expert: metrics(frame, p, {'pooled_1': p >= cuts[expert]}) for expert, p in values.items()},
                complementarity=complementarity(frame, values, {'pooled_1': cuts}),
                boundary=len(fold['temporal']))


def run():
    verify_protocol()
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection already frozen')
    started = time.monotonic()
    (OUT / 'artifacts').mkdir(parents=True, exist_ok=True)
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    folds_by_name = {fold['name']: fold for fold in load_folds()}
    experts = {}
    for name in [FIT, EVALUATION]:
        experts[name] = expert_probabilities(folds_by_name[name])
        print(name, 'experts', {expert: round(value['ap'], 4) if value['ap'] else None
                                for expert, value in experts[name]['metrics'].items()}, flush=True)
    fit, evaluation = experts[FIT], experts[EVALUATION]
    boundary = fit['boundary']
    matrix = np.column_stack([logits(fit['values'][expert]) for expert in EXPERTS])
    temporal_frame = fit['frame'].iloc[:boundary].reset_index(drop=True)
    geographic_frame = fit['frame'].iloc[boundary:].reset_index(drop=True)
    gating = fit['frame'][GATING]
    combiners = {}
    for name in COMBINERS:
        begin = time.monotonic()
        bundle = fit_combiner(name, temporal_frame, matrix[:boundary], gating.iloc[:boundary].reset_index(drop=True))
        artifact = OUT / 'artifacts' / f'combiner.{name}.joblib'
        joblib.dump(bundle, artifact, compress=3)
        score = combiner_score(bundle, matrix[boundary:], gating.iloc[boundary:].reset_index(drop=True))
        cuts = alert_cuts(geographic_frame, score)
        report = metrics(geographic_frame, score, {key: score >= value for key, value in cuts.items()})
        combiners[name] = dict(thresholds=cuts, selection=report, selection_ap=report['ap'],
                               selection_brier=report['brier'], artifact_sha256=digest(artifact),
                               seconds=time.monotonic() - begin)
        print('combiner', name, 'geographic AP', round(report['ap'], 4), flush=True)
    control_cuts = alert_cuts(geographic_frame, fit['values']['gbm'][boundary:])
    control = metrics(geographic_frame, fit['values']['gbm'][boundary:],
                      {key: fit['values']['gbm'][boundary:] >= value for key, value in control_cuts.items()})
    selected = min(combiners, key=lambda name: (-combiners[name]['selection_ap'],
                                                combiners[name]['selection_brier'], name))
    write_json(OUT / 'experts.json', dict(metrics={name: experts[name]['metrics'] for name in experts},
        complementarity={name: experts[name]['complementarity'] for name in experts},
        cuts={name: {expert: float(value) for expert, value in experts[name]['cuts'].items()} for name in experts},
        protocol_sha256=digest(OUT / 'protocol.json')))
    write_json(OUT / 'selection.json', dict(fit=FIT, evaluation=EVALUATION, experts=EXPERTS, combiners=combiners,
        selected=selected, control='gbm', control_thresholds=control_cuts, control_selection=control,
        predictions_sha256={name: digest(OUT / 'predictions' / f'{name}.experts.csv.gz') for name in experts},
        protocol_sha256=digest(OUT / 'protocol.json'), seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'torch', 'joblib']}))
    print('Selected', selected, flush=True)


def verify_selection():
    verify_protocol()
    record = json.loads((OUT / 'selection.json').read_text())
    if record['protocol_sha256'] != digest(OUT / 'protocol.json'):
        raise ValueError('Frozen protocol changed after selection')
    for name, checksum in record['predictions_sha256'].items():
        if digest(OUT / 'predictions' / f'{name}.experts.csv.gz') != checksum:
            raise ValueError('Frozen expert predictions changed: ' + name)
    for name, value in record['combiners'].items():
        if digest(OUT / 'artifacts' / f'combiner.{name}.joblib') != value['artifact_sha256']:
            raise ValueError('Frozen combiner artifact changed: ' + name)
    return record


def gate(record, control, intervals):
    """Precommitted promotion gate; every criterion must hold on the evaluation fold."""
    policy = lambda entry: entry['pooled']['policies']['pooled_1']
    criteria = dict(
        ap_gain=record['pooled']['ap'] > control['pooled']['ap'],
        brier_not_worse=record['pooled']['brier'] <= control['pooled']['brier'] + TOLERANCE,
        recall_gain=policy(record)['event_recall'] - policy(control)['event_recall'] >= .05 - TOLERANCE,
        pooled_false_alerts=policy(record)['false_alerts_per_100'] <= 1.5 + TOLERANCE,
        regional_false_alerts=all(region['policies']['pooled_1']['false_alerts_per_100'] <= 3 + TOLERANCE
                                  for region in record['regions'].values()),
        paired_ap=all(value['ap_difference_95'][0] > 0 for value in intervals.values()),
        paired_recall=all(value['event_recall_difference_95'][0] > 0 for value in intervals.values()),
        lead_preserved=record['median_event_lead_hours'] >= control['median_event_lead_hours'] - TOLERANCE)
    return dict(criteria=criteria, passes=all(criteria.values()))


def entry(frame, score, cuts):
    alerts = {key: score >= value for key, value in cuts.items()}
    boundary = len(frame) - len(frame[frame.role.eq('geographic_validation')])
    subsets = {'temporal': frame.iloc[:boundary].reset_index(drop=True), 'geographic': frame.iloc[boundary:].reset_index(drop=True)}
    return dict(pooled=metrics(frame, score, alerts),
                subsets={key: metrics(part, score[part.index], {k: alerts[k][part.index] for k in alerts})
                         for key, part in subsets.items()},
                regions={huc2: dict(policies=metrics(group, score[group.index],
                    {k: alerts[k][group.index] for k in alerts})['policies'])
                    for huc2, group in frame.groupby(frame.huc.str[:2])},
                median_event_lead_hours=metrics(frame, score, alerts)['policies']['pooled_1']['median_event_lead_hours'])


def evaluate():
    selection = verify_selection()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Evaluation already exists')
    started = time.monotonic()
    fold = next(item for item in load_folds() if item['name'] == EVALUATION)
    frame = pd.concat([fold['temporal'], fold['geographic']], ignore_index=True)
    saved = pd.read_csv(OUT / 'predictions' / f'{EVALUATION}.experts.csv.gz', dtype={'station': str, 'huc': str},
                        float_precision='round_trip').fillna({'event_date': ''})
    if [tuple(x) for x in zip(saved.station, saved.decision_date)] != [tuple(x) for x in zip(frame.station, frame.decision_date)]:
        raise ValueError('Saved expert predictions disagree with the frozen decisions')
    if not np.array_equal(saved.exceedance.to_numpy(), frame.exceedance.to_numpy()):
        raise ValueError('Saved labels disagree with the frozen dataset')
    matrix = np.column_stack([logits(saved[f'p_{expert}']) for expert in EXPERTS])
    gating = saved[GATING].copy()
    control = saved['p_gbm'].to_numpy()
    control_entry = entry(frame, control, selection['control_thresholds'])
    results, frames = {}, {}
    for name in COMBINERS:
        bundle = joblib.load(OUT / 'artifacts' / f'combiner.{name}.joblib')
        score = combiner_score(bundle, matrix, gating)
        cuts = selection['combiners'][name]['thresholds']
        results[name] = entry(frame, score, cuts)
        frames[name] = saved.assign(probability=score, **{key: score >= value for key, value in cuts.items()})
        frames[name].to_csv(OUT / 'predictions' / f'{EVALUATION}.{name}.csv.gz', index=False,
                            compression=dict(method='gzip', mtime=0))
        print('evaluation fold', name, 'AP', round(results[name]['pooled']['ap'], 4), flush=True)
    control_frame = saved.assign(probability=control,
                                 **{key: control >= value for key, value in selection['control_thresholds'].items()})
    intervals = {name: paired_intervals(frame, frames[name], control_frame) for name in COMBINERS}
    selected = selection['selected']
    gates = {name: gate(results[name], control_entry, intervals[name]) for name in COMBINERS}
    write_json(OUT / 'evaluation.json', dict(selected=selected, control='gbm', results=results, control_result=control_entry,
        intervals=intervals, gate=gates[selected], gates=gates,
        decision='gate passed' if gates[selected]['passes'] else 'gate failed; no integration',
        seconds=time.monotonic() - started, selection_sha256=digest(OUT / 'selection.json'),
        protocol_sha256=digest(OUT / 'protocol.json')))
    print('Selected', selected, 'gate', gates[selected]['passes'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'evaluate'])
    (run if parser.parse_args().command == 'run' else evaluate)()
