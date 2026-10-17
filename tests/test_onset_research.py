import numpy as np
import pandas as pd
import pytest


def test_onset_labels_require_complete_days_and_known_quiet_history():
    from scripts.onset_research import onset_rows
    dates = pd.date_range('2020-01-01', periods=35)
    q = pd.Series(1., index=dates)
    q.loc['2020-01-21':'2020-01-22'] = 11.
    frame = pd.DataFrame({'decision_date': dates.strftime('%Y-%m-%d')}, index=dates)
    rows = onset_rows(frame, q, 10., 'America/New_York')
    assert rows.loc['2020-01-18', 'event_date'] == '2020-01-21'
    assert rows.loc['2020-01-18', 'exceedance'] == 1
    assert rows.loc['2020-01-18', 'event_lead_hours'] == 77
    assert rows.loc['2020-01-19', 'event_lead_hours'] == 53
    assert rows.loc['2020-01-20', 'event_lead_hours'] == 29
    assert rows.loc['2020-01-21', 'exceedance'] == 0
    assert pd.Timestamp('2020-01-24') not in rows.index
    q.loc['2020-01-19'] = np.nan
    assert pd.Timestamp('2020-01-18') not in onset_rows(frame, q, 10., 'America/New_York').index


def test_partitions_reject_fresh_test_and_enforce_year_buffers():
    from scripts.onset_research import partitions
    frame = pd.DataFrame(dict(role=['development'] * 5, huc=['05010001'] * 5,
        decision_date=['2020-12-24', '2020-12-25', '2021-01-07', '2021-01-08', '2022-02-01']))
    parts = partitions(frame)
    assert list(parts['train'].index) == [0]
    assert list(parts['calibration'].index) == [3]
    assert list(parts['validation'].index) == [4]
    frame.loc[0, 'huc'] = '01010001'
    with pytest.raises(ValueError, match='Fresh'):
        partitions(frame)


def test_adaptation_uses_only_delayed_negatives_and_strict_ties():
    from scripts.onset_research import adaptive_alerts
    frame = pd.DataFrame(dict(station=['a'] * 10, decision_date=pd.date_range('2023-01-01', periods=10),
                             exceedance=[0] * 10))
    p = np.array([.9] + [.5] * 9)
    alerts, thresholds = adaptive_alerts(frame, p, np.zeros(100))
    assert thresholds[7] == 0
    assert thresholds[8] == 0  # one local negative is less than the 1% tail
    assert thresholds[9] == .5
    assert alerts[8] and not alerts[9]
    frame.loc[0, 'exceedance'] = 1
    _, changed = adaptive_alerts(frame, p, np.zeros(100))
    assert changed[8] == 0 and changed[9] == 0


def test_event_recall_counts_episodes_once_and_uses_first_warning():
    from scripts.onset_research import alert_metrics
    frame = pd.DataFrame(dict(station=['a'] * 4, event_date=['2023-01-04'] * 2 + ['2023-02-04', ''],
        exceedance=[1, 1, 1, 0], event_lead_hours=[77, 53, 29, np.nan]))
    metrics = alert_metrics(frame, np.array([True, True, False, True]))
    assert metrics['events'] == 2 and metrics['event_recall'] == .5
    assert metrics['false_alerts_per_100'] == 25
    assert metrics['median_event_lead_hours'] == 77


def test_trajectory_quantiles_are_sorted_before_head_features():
    from scripts.onset_worker import trajectory_inputs
    class FixedRegressor:
        def __init__(self, value):
            self.value = value
        def predict(self, frame):
            return np.full(len(frame), self.value)
    bundle = dict(features=['q_3'], regressors=[FixedRegressor(x) for x in [3, 1, 2] * 3])
    frame = pd.DataFrame(dict(q_3=[.2], q_change=[.1]))
    quantiles, head = trajectory_inputs(bundle, frame)
    assert np.allclose(quantiles[0], [[1.2, 2.2, 3.2]] * 3)
    assert head.shape == (1, 14)
    assert np.allclose(head[0, 9:12], [2, 2, 2])


def test_adaptation_cannot_see_future_outcome_revisions():
    from scripts.onset_research import adaptive_alerts
    frame = pd.DataFrame(dict(station=['a'] * 40, decision_date=pd.date_range('2023-01-01', periods=40),
                             exceedance=[0] * 40))
    p = np.linspace(.1, .9, 40)
    before, thresholds = adaptive_alerts(frame, p, np.zeros(100))
    frame.loc[20:, 'exceedance'] = 1
    after, changed = adaptive_alerts(frame, p, np.zeros(100))
    assert np.array_equal(before[:28], after[:28])
    assert np.array_equal(thresholds[:28], changed[:28])


def test_frozen_fresh_watersheds_are_absent_from_all_previous_roles():
    import json
    from scripts.onset_research import OUT, PREVIOUS, verify_protocol
    verify_protocol()
    stations = json.loads((OUT / 'stations.json').read_text())
    previous = json.loads((PREVIOUS / 'stations.json').read_text())
    old_hucs = {s['huc'][:4] for s in previous}
    fresh = [s for s in stations if s['role'] == 'fresh_test']
    assert len(fresh) == 35
    assert len({s['huc'][:4] for s in fresh}) == 11
    assert not old_hucs.intersection(s['huc'][:4] for s in fresh)
