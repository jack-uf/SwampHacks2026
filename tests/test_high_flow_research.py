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
