"""Read-only verification of the frozen local-calibration evaluation summary.

Run with ``python -B -m scripts.audit_localcal_summary``. Existing family audits
anchor model replay and label reconstruction; this additional audit checks their
hash bindings, row alignment, every reported metric, and the combined decision.
Nothing is fitted, selected, downloaded, or written. Diagnostics are explanatory
and do not enter the frozen decision criteria.
"""
import json

import numpy as np
import pandas as pd

from scripts.analog_worker import verify_selection as verify_analog
from scripts.localcal_research import OUT, ROOT, calibration_rows, digest, load_data
from scripts.localcal_worker import METHODS, local_probabilities, verify_selection
from scripts.onset_worker import metrics
from scripts.response_worker import paired_intervals


IDENTITY = ['station', 'huc', 'decision_date', 'event_date',
            'event_lead_hours', 'exceedance']
FAMILIES = {'classic': ['global', *METHODS, 'rainfall', 'logistic', 'direct'],
            'neural': ['pooled']}


def require_equal(actual, expected, path):
    """Compare complete JSON structures exactly, with a useful failure path."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'{path}: keys differ')
        for key in expected:
            require_equal(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{path}: list lengths differ')
        for index, (left, right) in enumerate(zip(actual, expected)):
            require_equal(left, right, f'{path}[{index}]')
    elif (isinstance(expected, bool) and not isinstance(actual, (bool, np.bool_))
          or actual != expected):
        raise ValueError(f'{path}: {actual!r} != {expected!r}')


def verify_alignment(frame, prediction):
    """Reject missing, duplicated, reordered or relabelled decisions before pairing."""
    for name, table in [('test', frame), ('prediction', prediction)]:
        if not set(IDENTITY) <= set(table.columns):
            raise ValueError(f'{name}: missing alignment columns')
        if table[['station', 'decision_date']].isna().any().any():
            raise ValueError(f'{name}: missing decision keys')
        if table.duplicated(['station', 'decision_date']).any():
            raise ValueError(f'{name}: duplicate decision keys')
    try:
        pd.testing.assert_frame_equal(prediction[IDENTITY], frame[IDENTITY],
                                      check_exact=True)
    except AssertionError as error:
        raise ValueError(f'Prediction alignment failed: {error}') from error


def verify_no_inversions(original, adapted):
    """Allow ties from clipping; reject reversals between distinct original scores.

    Equal original scores form one group, so arbitrary input order within a tie
    cannot manufacture an inversion. Sorting plus group extrema costs O(n log n).
    The separate frozen-intercept replay also verifies the adaptation itself.
    """
    original, adapted = np.asarray(original), np.asarray(adapted)
    if (original.ndim != 1 or original.shape != adapted.shape or not len(original)
            or not np.isfinite(original).all() or not np.isfinite(adapted).all()):
        raise ValueError('Invalid ranking vectors')
    order = np.argsort(original, kind='stable')
    ordered, values = original[order], adapted[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    maxima = np.maximum.reduceat(values, starts)
    minima = np.minimum.reduceat(values, starts)
    if np.any(np.maximum.accumulate(maxima)[:-1] > minima[1:]):
        raise ValueError('Within-station ranking inversion')


def candidate_metrics(frame, prediction, cuts):
    """Replay every policy, reliability bin and station/HUC2/year metric."""
    verify_alignment(frame, prediction)
    expected_columns = set(IDENTITY) | {'probability', 'strict_persistence'} | set(cuts)
    require_equal(sorted(prediction.columns), sorted(expected_columns), 'prediction.columns')
    p = prediction.probability.to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid prediction probabilities')
    alerts = {key: p >= cut for key, cut in cuts.items()}
    alerts['strict_persistence'] = frame.q_3.to_numpy() > np.log(2)
    for key, mask in alerts.items():
        if prediction[key].dtype != bool or not np.array_equal(prediction[key], mask):
            raise ValueError(f'Prediction policy mismatch: {key}')
    groups = {}
    for key, codes in [('station', frame.station), ('huc2', frame.huc.str[:2]),
                       ('year', frame.decision_date.str[:4])]:
        groups[key] = {name: metrics(group, p[group.index],
                          {policy: mask[group.index] for policy, mask in alerts.items()})
                       for name, group in frame.groupby(codes)}
    return dict(overall=metrics(frame, p, alerts), groups=groups)


def recompute_intervals(frame, left, right):
    """Use the frozen paired bootstrap, including its seed and cluster sampling."""
    verify_alignment(frame, left)
    verify_alignment(frame, right)
    intervals = paired_intervals(frame, left, right)
    y = frame.exceedance.to_numpy()
    delta = (left.probability.to_numpy() - y)**2 - (right.probability.to_numpy() - y)**2
    for key, groups in [('huc4', frame.huc.str[:4]), ('month', frame.decision_date.str[:7])]:
        codes, values = pd.factorize(groups)
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(500):
            weights = np.bincount(rng.integers(0, len(values), len(values)),
                                  minlength=len(values))[codes]
            draws.append(float(np.average(delta, weights=weights)))
        intervals[key]['brier_difference_95'] = np.quantile(draws, [.025, .975]).tolist()
    return intervals


def lead_comparison(left, right):
    """Compare first warnings on common events and count distinct >=48h warnings."""
    def warned(prediction):
        return prediction[prediction.pooled_1 & prediction.exceedance.eq(1)].groupby(
            ['station', 'event_date']).event_lead_hours.max()

    selected, global_ = warned(left), warned(right)
    common = selected.index.intersection(global_.index)
    gains = selected.loc[common] - global_.loc[common]
    lead = dict(added_events=len(selected.index.difference(global_.index)),
                lost_events=len(global_.index.difference(selected.index)),
                common_events=len(common),
                median_paired_lead_gain_hours=float(gains.median()) if len(common) else None)
    diagnostics = dict(common_event_earlier=int((gains > 0).sum()),
                       common_event_same=int((gains == 0).sum()),
                       common_event_later=int((gains < 0).sum()),
                       events_warned_at_least_48h={'global': int((global_ >= 48).sum()),
                                                  'selected': int((selected >= 48).sum())})
    return lead, diagnostics


def decision_criteria(frame, selected, global_, intervals, lead):
    policy, base = [record['overall']['policies']['pooled_1']
                    for record in [selected, global_]]
    return dict(
        adequate_population=bool(policy['events'] >= 50 and frame.huc.str[:4].nunique() >= 10),
        recall_gain=policy['event_recall'] - base['event_recall'] >= .05,
        recall_intervals=all(v['event_recall_difference_95'][0] > 0 for v in intervals.values()),
        brier_improvement=all(v['brier_difference_95'][1] < 0 for v in intervals.values()),
        pooled_false_alerts=policy['false_alerts_per_100'] <= 1.5,
        regional_false_alerts=all(g['policies']['pooled_1']['false_alerts_per_100'] <= 3
                                  for g in selected['groups']['huc2'].values()),
        paired_lead=lead['median_paired_lead_gain_hours'] is not None
                    and lead['median_paired_lead_gain_hours'] >= 0)


def audit():
    selection = verify_selection()
    old = verify_analog()
    frame = load_data('test').reset_index(drop=True)
    if not frame.role.eq('fresh_test').all() or not frame.decision_date.between(
            '2023-01-08', '2024-09-25').all():
        raise ValueError('Invalid final population')
    combined = json.loads((OUT / 'evaluation.json').read_text())
    require_equal(combined['selected'], selection['selected'], 'selected')
    require_equal(combined['selection_sha256'], digest(OUT / 'selection.json'), 'selection_sha256')
    require_equal(combined['test_sha256'], digest(OUT / 'test.csv.gz'), 'test_sha256')
    evaluation_hashes = {family: digest(OUT / f'{family}.evaluation.json') for family in FAMILIES}
    require_equal(combined['evaluation_hashes'], evaluation_hashes, 'evaluation_hashes')
    records, predictions = [], {}
    for family, names in FAMILIES.items():
        evaluation = json.loads((OUT / f'{family}.evaluation.json').read_text())
        replay = json.loads((OUT / f'{family}.audit.json').read_text())
        require_equal(evaluation['selection_sha256'], combined['selection_sha256'], family + '.selection_sha256')
        require_equal(evaluation['test_provenance_sha256'], digest(OUT / 'test.provenance.json'),
                      family + '.test_provenance_sha256')
        require_equal(replay['status'], 'passed', family + '.audit.status')
        require_equal(replay['evaluation_sha256'], evaluation_hashes[family], family + '.audit.evaluation_sha256')
        require_equal(replay['code_sha256'], digest(ROOT / 'scripts/audit_localcal.py'), family + '.audit.code_sha256')
        require_equal([row['name'] for row in evaluation['candidates']], names, family + '.candidates')
        require_equal(sorted(replay['predictions']), sorted(names), family + '.audit.predictions')
        for name in names:
            path = OUT / 'predictions' / f'{name}.test.csv.gz'
            require_equal(digest(path), replay['predictions'][name], name + '.prediction_sha256')
            saved = pd.read_csv(path, dtype={'station': str, 'huc': str},
                                float_precision='round_trip').fillna({'event_date': ''})
            verify_alignment(frame, saved)
            predictions[name] = saved
        records.extend(evaluation['candidates'])
    # All files are aligned before any cross-model comparison or metric calculation.
    require_equal(combined['candidates'], records, 'candidates')
    recomputed = {}
    for row in records:
        name = row['name']
        source = selection if name in ['global', *METHODS] else old
        cuts = next(candidate['thresholds'] for candidate in source['candidates']
                    if candidate['name'] == name)
        actual = candidate_metrics(frame, predictions[name], cuts)
        for key, value in actual.items():
            require_equal(row[key], value, f'candidates.{name}.{key}')
        recomputed[name] = actual
        if name in METHODS:
            p = local_probabilities(frame, predictions['global'].probability.to_numpy(),
                                    selection['intercepts'][name])
            if not np.array_equal(p, predictions[name].probability):
                raise ValueError(f'{name}: frozen intercept replay differs')
            for _, group in frame.groupby('station'):
                verify_no_inversions(predictions['global'].probability.to_numpy()[group.index],
                                     p[group.index])
    name = selection['selected']
    intervals = recompute_intervals(frame, predictions[name], predictions['global'])
    lead, diagnostics = lead_comparison(predictions[name], predictions['global'])
    criteria = decision_criteria(frame, recomputed[name], recomputed['global'], intervals, lead)
    expected = dict(paired_intervals=intervals, paired_lead=lead, criteria=criteria,
                    passes=all(criteria.values()), gauges=int(frame.station.nunique()),
                    huc4=int(frame.huc.str[:4].nunique()))
    for key, value in expected.items():
        require_equal(combined[key], value, key)

    cal = calibration_rows(load_data('calibration')).reset_index(drop=True)
    registry = json.loads((OUT / 'stations.json').read_text())
    fresh = cal[cal.station.isin({row['station'] for row in registry})]
    if not set(frame.station) <= set(fresh.station):
        raise ValueError('Final station missing from fresh calibration')
    for station, group in fresh.groupby('station'):
        require_equal(selection['calibration'][station], dict(rows=len(group),
            positives=int(group.exceedance.sum()), first=group.decision_date.min(),
            last=group.decision_date.max()), f'calibration.{station}')
    diagnostics['fresh_station_2021_calibration'] = dict(stations=int(fresh.station.nunique()),
        rows=len(fresh), positive_windows=int(fresh.exceedance.sum()),
        zero_positive_stations=int(fresh.groupby('station').exceedance.sum().eq(0).sum()))
    return dict(status='passed', selected=name, candidates=len(records), test_rows=len(frame),
                gauges=expected['gauges'], huc4=expected['huc4'], criteria=criteria,
                study_passes=expected['passes'], diagnostics=diagnostics,
                evaluation_sha256=digest(OUT / 'evaluation.json'),
                evaluation_hashes=evaluation_hashes)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
