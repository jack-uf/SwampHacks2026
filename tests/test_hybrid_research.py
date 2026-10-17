import numpy as np
import pandas as pd
import pytest

from scripts.hybrid_research import (complementarity, fit_combiner, gate, logits, mean_score,
                                     combiner_score, GATING)
from scripts.response_research import SIGNATURE


def synthetic():
    rng = np.random.default_rng(42)
    rows = 600
    group = np.arange(rows) % 2
    y = (rng.random(rows) < .3).astype(int)
    strong = logits(np.where(y, .9, .1).astype(float))
    noise = logits(rng.uniform(.05, .95, rows))
    specialist_a = np.where(group == 0, strong, noise)
    specialist_b = np.where(group == 1, strong, noise)
    padding = [logits(rng.uniform(.05, .95, rows)) for _ in range(3)]
    frame = pd.DataFrame({'station': '10000001', 'huc': '03010101', 'decision_date': '2021-06-01',
                          'event_date': np.where(y, '2021-06-02', ''), 'event_lead_hours': 24,
                          'exceedance': y, 'q_3': np.log1p(rng.uniform(1, 10, rows)),
                          'group': group, 'season_sin': 0., 'season_cos': 1., 'snow_store': 0.,
                          'history_complete': True, 'log_area': 2., 'elevation_m': 100.,
                          'rain_spread_0': 0., 'rain_spread_1': 0., 'rain_mean_0': 1.})
    for name in SIGNATURE:
        frame[name] = 0.
    gating = frame[GATING].copy()
    return frame, np.column_stack([specialist_a, specialist_b] + padding), gating


def test_logits_and_mean_score_are_equal_weighted_and_monotone():
    matrix = np.array([[np.log(.75), np.log(.25)], [np.log(.5), np.log(.5)]])
    scores = mean_score(matrix)
    np.testing.assert_allclose(scores, [.5 * (np.log(.75) + np.log(.25)), np.log(.5)])
    assert np.all(np.diff(logits(np.array([.1, .5, .9]))) > 0)


def test_complementarity_separates_shared_and_single_expert_events():
    frame = pd.DataFrame({
        'station': ['a', 'a', 'a', 'b'], 'event_date': ['x', 'y', 'z', 'w'],
        'exceedance': [1, 1, 1, 1], 'event_lead_hours': [24, 24, 24, 24]})
    probabilities = {'left': np.array([.9, .1, .9, .1]), 'right': np.array([.1, .9, .9, .1])}
    cuts = {'pooled_1': {'left': .5, 'right': .5}}
    result = complementarity(frame, probabilities, cuts)
    assert result['events'] == 4
    assert result['single_expert_events'] == {'left': 1, 'right': 1}
    assert result['shared_events'] == 1
    assert result['missed_by_all'] == 1
    assert result['union_recall'] == .75
    assert result['pairwise_agreement']['left,right'] == .5


def test_stacking_and_gating_beat_the_best_single_expert_on_specialists():
    frame, matrix, gating = synthetic()
    stacked = fit_combiner('stack', frame, matrix, gating)
    gated = fit_combiner('gated', frame, matrix, gating)
    from sklearn.metrics import average_precision_score
    y = frame.exceedance.to_numpy()
    scores = {'a': matrix[:, 0], 'b': matrix[:, 1],
              'stack': combiner_score(stacked, matrix, gating), 'gated': combiner_score(gated, matrix, gating)}
    ap = {name: average_precision_score(y, value) for name, value in scores.items()}
    assert ap['stack'] > max(ap['a'], ap['b'])
    assert ap['gated'] > max(ap['a'], ap['b'])


def test_combiners_refuse_a_frame_from_another_fold():
    frame, matrix, gating = synthetic()
    with pytest.raises(ValueError, match='2021'):
        fit_combiner('stack', frame.assign(decision_date='2022-06-01'), matrix, gating)
    with pytest.raises(ValueError, match='per expert'):
        fit_combiner('stack', frame, matrix[:, :1], gating)
    with pytest.raises(ValueError, match='missing'):
        fit_combiner('gated', frame, matrix, gating.assign(rain_spread_0=np.nan))
    with pytest.raises(ValueError, match='name'):
        fit_combiner('mixture', frame, matrix, gating)


def gate_record(recall=.50, false=1.0, region=2.0, brier=.02, ap=.30, lead=54.0):
    return dict(pooled=dict(ap=ap, brier=brier, policies={'pooled_1': dict(
        event_recall=recall, false_alerts_per_100=false)}),
        regions={'0301': dict(policies={'pooled_1': dict(false_alerts_per_100=region)})},
        median_event_lead_hours=lead)


def test_gate_requires_every_criterion_at_its_boundary():
    control = gate_record(recall=.45, ap=.28, brier=.025, false=1.2, region=2.5)
    intervals = {'huc4': dict(ap_difference_95=[.001, .05], event_recall_difference_95=[.01, .2]),
                 'month': dict(ap_difference_95=[.001, .05], event_recall_difference_95=[.01, .2])}
    passed = gate(gate_record(), control, intervals)
    assert passed['passes'] and all(passed['criteria'].values())
    for broken in [gate_record(recall=.49), gate_record(region=3.2), gate_record(false=1.6),
                   gate_record(ap=.27), gate_record(brier=.03), gate_record(lead=50.0)]:
        result = gate(broken, control, intervals)
        assert not result['passes'], broken['pooled']
    uncertain = {scheme: dict(ap_difference_95=[-.001, .05], event_recall_difference_95=[-.01, .2])
                 for scheme in ['huc4', 'month']}
    assert not gate(gate_record(), control, uncertain)['passes']
    assert gate(gate_record(recall=.50, region=3.0, false=1.5, ap=.281, brier=.024), control, intervals)['passes']
