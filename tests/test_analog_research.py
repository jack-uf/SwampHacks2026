import numpy as np
import pandas as pd

from scripts.analog_research import analog_features


def history():
    days = pd.date_range('2000-01-01', '2024-01-31')
    rng = np.random.default_rng(42)
    return pd.Series(np.exp(2 + np.sin(np.arange(len(days)) / 20) + rng.normal(0, .2, len(days))), index=days)


def test_analog_features_ignore_future_and_post_reference_library_outcomes():
    q = history()
    issues = pd.DatetimeIndex(['2023-06-10', '2023-08-10'])
    expected, metadata = analog_features(q, issues)
    changed = q.copy()
    changed.loc['2016-01-01':'2022-12-31'] *= 100
    changed.loc['2023-08-08':] *= 100
    actual, after = analog_features(changed, issues)
    pd.testing.assert_frame_equal(actual, expected)
    assert metadata == after
    assert np.isfinite(actual.to_numpy()).all()
    assert metadata['last_library_target'] < '2016-01-01'


def test_missing_calendar_observation_is_unavailable_not_previous_row():
    q = history().drop(pd.Timestamp('2023-06-07'))
    result, _ = analog_features(q, pd.DatetimeIndex(['2023-06-10', '2023-08-10']))
    assert result.iloc[0].isna().all()
    assert np.isfinite(result.iloc[1]).all()


def test_offset_inference_preserves_prior_when_trees_learn_no_correction():
    from lightgbm import LGBMClassifier
    from scripts.analog_worker import candidate_score
    frame = pd.DataFrame({'constant': np.ones(100), 'analog_logit': np.repeat([-np.log(4), np.log(4)], 50)})
    y = np.r_[np.ones(10), np.zeros(40), np.ones(40), np.zeros(10)]
    model = LGBMClassifier(n_estimators=1, n_jobs=1, verbosity=-1).fit(frame[['constant']], y, init_score=frame.analog_logit)
    bundle = dict(name='offset', model=model, features=['constant'])
    raw = candidate_score(bundle, frame.iloc[[0, -1]])
    np.testing.assert_allclose(1 / (1 + np.exp(-raw.ravel())), [.2, .8], atol=1e-8)
