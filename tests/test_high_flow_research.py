"""Synthetic checks for research isolation and operational budget selection."""
import numpy as np
import pandas as pd
import pytest

from scripts.high_flow_research import hydrologic_features, split, threshold
from scripts.shift_hydrology import FEATURES


def test_threshold_matches_exhaustive_search_with_ties_and_zero_alerts():
    rng = np.random.default_rng(42)
    for _ in range(30):
        p = rng.choice([.1, .2, .3, .8], 80)
        f = pd.DataFrame({'target': rng.integers(0, 2, 80), 'huc': ['01010000'] * 40 + ['04010000'] * 40})
        t = threshold(f, p, 10, 15)
        choices = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
        feasible = []
        for cut in choices:
            false = (p >= cut) & f.target.eq(0)
            if 100 * false.mean() <= 10 and all(100 * false[g.index].mean() <= 15 for _, g in f.groupby('huc')):
                feasible.append(cut)
        assert t == min(feasible)
    f = pd.DataFrame({'target': [0, 0, 1], 'huc': ['01010000'] * 3})
    assert threshold(f, [.9, .8, .1], 1, 3) > .9
    with pytest.raises(ValueError):
        threshold(f, [np.nan, .2, .3], 1, 3)


def test_geographic_and_label_boundary_purges():
    times = ['2022-06-01', '2022-12-31T12:00', '2023-03-01', '2023-06-30T12:00',
             '2023-08-01', '2023-12-31T12:00', '2024-03-01', '2025-03-01']
    records = [{'huc': huc, 'station': station, 'decision_time': pd.Timestamp(t, tz='UTC').isoformat()}
               for huc, station in [('01010000', 'one'), ('04010000', 'two')] for t in times]
    parts = split(pd.DataFrame(records), {'0401'})
    assert len(parts['train']) == len(parts['calibration']) == 1
    assert len(parts['threshold']) == 2
    assert len(parts['temporal_validation']) == len(parts['geographic_validation']) == 1
    for key in ['train', 'calibration', 'temporal_validation']:
        assert set(parts[key].station) == {'one'}


def test_hydrologic_features_use_log_two_reference_and_ignore_labels():
    f = pd.DataFrame({key: [.1, .2] for key in FEATURES})
    f['q_2'] = [np.log(2), .5]
    f['horizon'] = 6
    f['target'] = [0, 1]
    f['future_max_log_flow'] = [3, 4]
    a = hydrologic_features(f)
    assert a.margin.iloc[0] == pytest.approx(0)
    assert a.linear_margin.iloc[0] == pytest.approx((np.log(2) - .1) * 8 / 6)
    b = hydrologic_features(f.assign(target=[1, 0], future_max_log_flow=[100, 200]))
    pd.testing.assert_frame_equal(a.drop(columns=['target', 'future_max_log_flow']),
                                  b.drop(columns=['target', 'future_max_log_flow']))


def test_temporal_validation_requires_a_fitted_station_and_annual_schedule():
    times = ['2021-03-01', '2022-03-01', '2022-08-01', '2023-03-01', '2023-08-01', '2024-03-01']
    frame = pd.DataFrame([dict(huc=huc, station=station, decision_time=pd.Timestamp(t, tz='UTC').isoformat())
        for huc, station in [('01010000', 'fitted'), ('04010000', 'geographic')] for t in times]
        + [dict(huc='02010000', station='unfitted', decision_time='2024-03-01T00:00:00+00:00')])
    half, annual = [split(frame, {'0401'}, cycle) for cycle in [3, 4]]
    assert set(annual['temporal_validation'].station) == {'fitted'}
    assert len(annual['calibration']) == 2
    assert len(half['threshold']) == 2
    assert len(annual['threshold']) == 4
    pd.testing.assert_frame_equal(half['train'], annual['train'])
    pd.testing.assert_frame_equal(half['calibration'], annual['calibration'])


def test_common_row_comparison_rejects_changed_outcomes():
    from scripts.report_high_flow_research import common_predictions
    f = pd.DataFrame(dict(station=['one'], huc=['01010000'], decision_time=['2024-01-01T00:00:00+00:00'],
        horizon=[6], target=[0], event_time=[''], probability=[.1]))
    with pytest.raises(ValueError, match='labels'):
        common_predictions([f, f.assign(target=1)])
    a, b = common_predictions([f, f.assign(probability=.9)])
    assert a.probability.iloc[0] != b.probability.iloc[0]


def test_two_way_bootstrap_preserves_paired_zero_differences():
    from scripts.report_high_flow_research import two_way_uncertainty
    dates = pd.date_range('2024-01-01', periods=20, freq='7D', tz='UTC')
    f = pd.DataFrame([dict(station=str(i), huc=huc, decision_time=t.isoformat(), horizon=6,
        target=int(j % 2 == 0), event_time=(t + pd.Timedelta(hours=2)).isoformat() if j % 2 == 0 else '')
        for i, huc in enumerate(['01010000', '02010000', '04010000']) for j, t in enumerate(dates)])
    p = np.linspace(0, 1, len(f))
    result = two_way_uncertainty(f, p, p, .5, .5, draws=20)
    assert result['spatial_clusters'] == 3 and result['weekly_blocks'] == 20
    assert all(value == [0, 0] for value in result['intervals_95'].values())
    with pytest.raises(ValueError, match='six-hour'):
        two_way_uncertainty(f.assign(horizon=12), p, p, .5, .5)
