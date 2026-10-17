import pytest
from scripts.hydrology_research import daily_discharge


def test_daily_discharge_uses_only_exact_approved_measured_values():
    text = '''# Discharge, cubic feet per second (Mean)
agency_cd\tsite_no\tdatetime\t123_00060_00003\t123_00060_00003_cd
5s\t15s\t20d\t14n\t10s
USGS\t01013500\t2010-01-01\t1240\tA:e
USGS\t01013500\t2010-01-02\t1200\tA
USGS\t01013500\t2010-01-03\t1300\tP
USGS\t01013500\t2010-01-04\tIce\tA
'''
    observed = daily_discharge(text, '01013500')
    assert observed.to_dict() == {'2010-01-02': 1200.}
    with pytest.raises(ValueError, match='station'):
        daily_discharge(text, '99999999')
    with pytest.raises(ValueError, match='series'):
        daily_discharge(text.replace('123_00060_00003_cd', '123_00060_00003_cd\t456_00060_00003'), '01013500')


def test_features_cannot_see_future_weather_or_recent_discharge():
    import numpy as np
    import pandas as pd
    from scripts.hydrology_research import basin_features
    dates = pd.date_range('2016-01-01', periods=400).as_unit('s')
    rain = np.ones((400, 10)); temperature = np.full((400, 10), 10.)
    q = pd.Series(20., index=dates)
    station = dict(station='01013500', role='development', huc='01010003', latitude=47.,
                   longitude=-68., area_km2=100., elevation_m=100., timezone='America/New_York')
    a = basin_features(dates, rain, temperature, q, 10., station, 1)
    decision = dates[350]
    rain[350:] = 1000
    q.loc[dates[348]:] = 1000
    b = basin_features(dates, rain, temperature, q, 10., station, 1)
    features = [c for c in a if c.startswith(('rain_', 'temp_', 'q_', 'memory_', 'liquid_', 'snow_'))]
    pd.testing.assert_series_equal(a.loc[decision, features], b.loc[decision, features])
    assert a.loc[decision, 'lead_start_hours'] == 29.
    assert a.loc[decision, 'exceedance'] == 1
    rain[349] = np.nan
    c = basin_features(dates, rain, temperature, q, 10., station, 1)
    assert decision not in c.index


def test_split_enforces_geographic_roles_and_year_boundary_embargo():
    import pandas as pd
    from scripts.hydrology_research import partitions
    frame = pd.DataFrame(dict(role=['development']*3 + ['geographic_test'],
        huc=['01010003']*3 + ['17100205'], target_date=['2020-12-24','2020-12-25','2021-01-08','2023-06-01']))
    parts = partitions(frame)
    assert parts['train'].index.tolist() == [0]
    assert parts['calibration'].index.tolist() == [2]
    assert parts['geographic_test'].index.tolist() == [3]
    frame.loc[3, 'huc'] = '01020001'
    with pytest.raises(ValueError, match='geographic'):
        partitions(frame)


def test_budget_threshold_keeps_ties_and_never_spends_more_than_validation_budget():
    import numpy as np
    from scripts.hydrology_research import budget_threshold
    y = np.array([1, 0, 1, 0]); p = np.array([.9, .9, .8, .2])
    assert budget_threshold(y, p, .25) == .8
    assert not (p >= budget_threshold(y, p, .20)).any()
    with pytest.raises(ValueError, match='probabilities'):
        budget_threshold(y, np.array([1.1, .9, .8, .2]), .25)


def test_small_negative_forecast_totals_are_bounded_and_auditable():
    import numpy as np
    import pandas as pd
    from scripts.hydrology_research import basin_features
    dates = pd.date_range('2016-01-01', periods=300)
    rain = np.full((300, 10), -.03); temp = np.ones_like(rain)
    q = pd.Series(20., index=dates)
    station = dict(station='01013500', role='development', huc='01010003', latitude=47.,
                   longitude=-68., area_km2=100., elevation_m=100., timezone='America/New_York')
    frame = basin_features(dates, rain, temp, q, 10., station, 1)
    assert frame.rain_3.eq(0).all() and len(frame) > 0
    rain[280, 0] = -.2
    with pytest.raises(ValueError, match='Negative precipitation'):
        basin_features(dates, rain, temp, q, 10., station, 1)
