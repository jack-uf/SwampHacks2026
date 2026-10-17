import numpy as np
import pytest

from scripts.localcal_research import fit_intercept, adjusted


def test_local_calibration_corrects_bias_and_pooling_limits_sparse_adjustment():
    p = np.full(100, .2)
    y = np.r_[np.ones(50), np.zeros(50)]
    local = fit_intercept(p, y, .1)
    pooled = fit_intercept(p, y, 10)
    assert .49 < adjusted(p, local).mean() < .5
    assert .2 < adjusted(p, pooled).mean() < .49
    assert 0 < pooled < local


def test_calibration_rejects_bad_measurements_and_handles_no_events():
    for p, y, penalty in [([.2, np.nan], [0, 1], 2), ([1.1], [0], 2),
                           ([.2], [2], 2), ([.2], [0, 1], 2), ([], [], 2), ([.2], [0], 0)]:
        with pytest.raises(ValueError):
            fit_intercept(p, y, penalty)
    b = fit_intercept([.1]*20, [0]*20, 2)
    actual = adjusted([.01, .1, .9], b)
    assert np.isfinite(actual).all() and np.all(np.diff(actual) > 0)
    assert 0 < actual[1] < .1


def test_calibration_boundary_prevents_final_labels_from_entering_fit():
    import pandas as pd
    from scripts.localcal_research import calibration_rows
    f = pd.DataFrame({'decision_date': ['2021-12-24', '2023-01-08'], 'station': ['a', 'a']})
    with pytest.raises(ValueError, match='2021'):
        calibration_rows(f)
    assert len(calibration_rows(f.iloc[:1])) == 1


def test_unseen_gauges_require_explicit_calibration_and_keep_local_ranking():
    import pandas as pd
    from scripts.localcal_worker import local_probabilities
    frame = pd.DataFrame({'station': ['a', 'b', 'a']})
    with pytest.raises(ValueError, match='Missing historical calibration'):
        local_probabilities(frame, [.2, .3, .4], {'a': 1})
    result = local_probabilities(frame, [.2, .3, .4], {'a': 1, 'b': -1})
    assert result[0] < result[2]
    assert result[0] > .2 and result[1] < .3


def test_local_probabilities_rejects_truncated_predictions():
    import pandas as pd
    from scripts.localcal_worker import local_probabilities
    with pytest.raises(ValueError, match='length'):
        local_probabilities(pd.DataFrame({'station': ['a', 'a']}), [.2], {'a': 0})
