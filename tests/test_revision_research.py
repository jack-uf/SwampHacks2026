import numpy as np
import pandas as pd
import pytest

from scripts.revision_research import forecast_revisions


def test_revisions_align_valid_days_across_initializations_and_bridge_flow_delay():
    dates = pd.date_range('2020-01-01', periods=12)
    values = np.arange(12)[:, None] * 100 + np.arange(10)[None, :]
    f = forecast_revisions(dates, values, values + 1000)
    # Decision Jan 6, target Jan 7: Jan 5 lead 3, Jan 4 lead 4, Jan 3 lead 5.
    assert f.loc['2020-01-06', 'rain_mean_1'] == 303
    assert f.loc['2020-01-06', 'rain_revision_1'] == 198
    assert f.loc['2020-01-06', 'rain_spread_1'] == pytest.approx(np.std([402, 303, 204]))
    # Forecast of Jan 5 issued Jan 4, and Jan 4 issued Jan 3.
    assert f.loc['2020-01-06', 'rain_bridge_1'] == 301
    assert f.loc['2020-01-06', 'rain_bridge_2'] == 201


def test_revisions_never_use_decision_day_or_later_runs():
    dates = pd.date_range('2020-01-01', periods=12)
    rain = np.ones((12, 10)); temp = np.zeros_like(rain)
    before = forecast_revisions(dates, rain, temp).loc['2020-01-06'].copy()
    rain[5:] = 1e6; temp[5:] = 1e6
    after = forecast_revisions(dates, rain, temp).loc['2020-01-06']
    pd.testing.assert_series_equal(before, after)


def test_missing_runs_are_unavailable_and_calendar_gaps_are_rejected():
    dates = pd.date_range('2020-01-01', periods=12)
    rain = np.ones((12, 10)); temp = np.zeros_like(rain)
    rain[2, 4] = np.nan
    assert np.isnan(forecast_revisions(dates, rain, temp).loc['2020-01-06', 'rain_mean_1'])
    with pytest.raises(ValueError, match='daily'):
        forecast_revisions(dates.delete(2), rain[:-1], temp[:-1])
    with pytest.raises(ValueError, match='shape'):
        forecast_revisions(dates, rain[:, :9], temp)
    with pytest.raises(ValueError, match='precipitation'):
        forecast_revisions(dates, -rain, temp)


def test_second_resolution_calendar_is_accepted():
    dates = pd.date_range('2020-01-01', periods=12).as_unit('s')
    values = np.arange(12)[:, None] * 100 + np.arange(10)[None, :]
    frame = forecast_revisions(dates, values, values + 1000)
    assert frame.index.equals(pd.date_range('2020-01-01', periods=12))
    assert frame.loc[pd.Timestamp('2020-01-06'), 'rain_mean_1'] == 303
