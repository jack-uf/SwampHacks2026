"""Focused failure cases for the additional, read-only summary verifier."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import audit_localcal_summary as audit
from scripts.localcal_research import adjusted


@pytest.fixture
def sample():
    frame = pd.DataFrame(dict(
        station=['001', '001', '002', '002'],
        huc=['01010001', '01010001', '02020001', '02020001'],
        decision_date=['2023-01-08', '2023-01-09', '2024-02-08', '2024-02-09'],
        event_date=['2023-01-10', '', '2024-02-10', ''],
        event_lead_hours=[48., np.nan, 48., np.nan],
        exceedance=[1, 0, 1, 0], q_3=[0., 1., 0., 0.]))
    prediction = frame[audit.IDENTITY].copy()
    prediction['probability'] = [.8, .6, .4, .1]
    prediction['pooled_1'] = prediction.probability >= .5
    prediction['strict_persistence'] = frame.q_3 > np.log(2)
    return frame, prediction


@pytest.mark.parametrize('change', ['reorder', 'truncate', 'duplicate', 'station',
                                    'huc', 'event_date', 'event_lead_hours', 'exceedance'])
def test_pairing_rejects_misalignment_before_bootstrap(sample, monkeypatch, change):
    frame, prediction = sample
    audit.verify_alignment(frame, prediction)
    bad = prediction.copy()
    if change == 'reorder':
        bad = bad.iloc[::-1].reset_index(drop=True)
    elif change == 'truncate':
        bad = bad.iloc[:-1]
    elif change == 'duplicate':
        bad.iloc[1] = bad.iloc[0]
    else:
        bad.loc[0, change] = dict(station='003', huc='03030001', event_date='2023-01-11',
                                  event_lead_hours=72., exceedance=0)[change]

    def forbidden(*args):
        pytest.fail('Bootstrap ran before alignment verification')

    monkeypatch.setattr(audit, 'paired_intervals', forbidden)
    with pytest.raises(ValueError, match='alignment|duplicate'):
        audit.recompute_intervals(frame, bad, prediction)


def test_clipping_ties_are_valid_but_real_inversions_fail():
    original = np.array([2e-8, 1e-8, .2])
    clipped = adjusted(original, 0)
    assert clipped[0] == clipped[1]
    assert not np.array_equal(np.argsort(original, kind='stable'),
                              np.argsort(clipped, kind='stable'))
    audit.verify_no_inversions(original, clipped)
    audit.verify_no_inversions([.1, .1, .2], [.2, .1, .3])
    with pytest.raises(ValueError, match='inversion'):
        audit.verify_no_inversions([.1, .1, .2], [.4, .1, .3])
    with pytest.raises(ValueError, match='inversion'):
        audit.verify_no_inversions([.1, .2], [.3, .1])


@pytest.mark.parametrize('group,key', [('station', '001'), ('huc2', '01'), ('year', '2023')])
def test_every_group_metric_is_checked(sample, group, key):
    frame, prediction = sample
    actual = audit.candidate_metrics(frame, prediction, {'pooled_1': .5})
    tampered = deepcopy(actual)
    tampered['groups'][group][key]['policies']['pooled_1']['false_alerts'] += 1
    with pytest.raises(ValueError, match=rf'groups.{group}.{key}.*false_alerts'):
        audit.require_equal(tampered, actual, 'candidate')


def test_policy_masks_are_replayed_including_persistence(sample):
    frame, prediction = sample
    for policy in ['pooled_1', 'strict_persistence']:
        bad = prediction.copy()
        bad.loc[0, policy] = not bad.loc[0, policy]
        with pytest.raises(ValueError, match=f'policy mismatch: {policy}'):
            audit.candidate_metrics(frame, bad, {'pooled_1': .5})


def test_lead_diagnostics_use_distinct_events_and_inclusive_48h_boundary():
    # Repeated positive windows must count as one event, at its earliest warning.
    base = pd.DataFrame(dict(station=['a', 'a', 'b', 'c', 'c', 'd', 'e'],
        event_date=['2023-01-10', '2023-01-10', '2023-01-11', '2023-01-12',
                    '2023-01-12', '2023-01-13', '2023-01-14'],
        event_lead_hours=[72., 48., 48., 72., 24., 48., 48.], exceedance=[1]*7,
        pooled_1=[False, True, True, True, True, False, True]))
    selected = base.copy()
    selected['pooled_1'] = [True, True, True, False, True, True, False]
    lead, diagnostics = audit.lead_comparison(selected, base)
    assert lead == dict(added_events=1, lost_events=1, common_events=3,
                       median_paired_lead_gain_hours=0.)
    assert diagnostics == dict(common_event_earlier=1, common_event_same=1,
        common_event_later=1, events_warned_at_least_48h={'global': 4, 'selected': 3})


@pytest.mark.parametrize('field,value', [('paired_intervals', {'huc4': [0., 1.]}),
                                       ('passes', True), ('selection_sha256', 'changed')])
def test_summary_comparison_rejects_changed_fields(field, value):
    expected = dict(paired_intervals={'huc4': [-.1, .2]}, passes=False,
                    selection_sha256='frozen')
    bad = deepcopy(expected)
    bad[field] = value
    with pytest.raises(ValueError, match=field):
        audit.require_equal(bad, expected, 'evaluation')


def test_identical_predictions_have_zero_paired_intervals(sample):
    frame, prediction = sample
    intervals = audit.recompute_intervals(frame, prediction, prediction)
    assert intervals == {cluster: dict(draws=500, ap_difference_95=[0., 0.],
        event_recall_difference_95=[0., 0.], brier_difference_95=[0., 0.])
        for cluster in ['huc4', 'month']}


@pytest.mark.parametrize('failure', ['adequate_population', 'recall_gain', 'recall_intervals',
                                    'brier_improvement', 'pooled_false_alerts',
                                    'regional_false_alerts', 'paired_lead'])
def test_all_decision_criteria_are_recomputed_at_boundaries(failure):
    frame = pd.DataFrame({'huc': [f'{i:04d}0000' for i in range(10)]})
    selected = dict(overall={'policies': {'pooled_1': dict(events=50,
        event_recall=.8, false_alerts_per_100=1.5)}},
        groups={'huc2': {'01': {'policies': {'pooled_1': {'false_alerts_per_100': 3.}}}}})
    global_ = {'overall': {'policies': {'pooled_1': {'event_recall': .5}}}}
    intervals = {key: dict(event_recall_difference_95=[.01, .2],
        brier_difference_95=[-.1, -.001]) for key in ['huc4', 'month']}
    lead = {'median_paired_lead_gain_hours': 0.}
    assert all(audit.decision_criteria(frame, selected, global_, intervals, lead).values())
    if failure == 'adequate_population':
        frame = frame.iloc[:-1]
    elif failure == 'recall_gain':
        selected['overall']['policies']['pooled_1']['event_recall'] = .54
    elif failure == 'recall_intervals':
        intervals['month']['event_recall_difference_95'][0] = 0.
    elif failure == 'brier_improvement':
        intervals['huc4']['brier_difference_95'][1] = 0.
    elif failure == 'pooled_false_alerts':
        selected['overall']['policies']['pooled_1']['false_alerts_per_100'] = 1.5001
    elif failure == 'regional_false_alerts':
        selected['groups']['huc2']['01']['policies']['pooled_1']['false_alerts_per_100'] = 3.0001
    else:
        lead['median_paired_lead_gain_hours'] = -1.
    actual = audit.decision_criteria(frame, selected, global_, intervals, lead)
    assert {key for key, passed in actual.items() if not passed} == {failure}
