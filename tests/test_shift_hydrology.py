import numpy as np
import pandas as pd
import pytest

from scripts.shift_hydrology import parse_observations, hourly_flow, decision_rows


def observation(time, value='10', **updates):
    p = dict(monitoring_location_id='USGS-12345678', parameter_code='00060',
             statistic_id='00011', time_series_id='series', time=time, value=value,
             unit_of_measure='ft^3/s', approval_status='Approved', qualifier=None)
    p.update(updates)
    return {'properties': p}


def test_quality_and_units_are_checked_before_hourly_aggregation():
    rows = [observation('2023-01-01T00:00:00Z'),
            observation('2023-01-01T00:15:00Z', approval_status='Provisional'),
            observation('2023-01-01T00:30:00Z', qualifier='ice'),
            observation('2023-01-01T02:00:00Z', '20')]
    s = parse_observations(rows, '12345678')
    q = hourly_flow(s)
    assert q.iloc[0] == 10 and np.isnan(q.iloc[1]) and q.iloc[2] == 20
    with pytest.raises(ValueError, match='unit'):
        parse_observations([observation('2023-01-01T00:00:00Z', unit_of_measure='m')], '12345678')


def test_ambiguous_series_and_duplicate_measurements_fail():
    a = observation('2023-01-01T00:00:00Z')
    with pytest.raises(ValueError, match='Duplicate'):
        parse_observations([a, a], '12345678')
    with pytest.raises(ValueError, match='series'):
        parse_observations([a, observation('2023-01-01T01:00:00Z', time_series_id='other')], '12345678')


def fixture_flow():
    index = pd.date_range('2023-01-01', periods=360, freq='h', tz='UTC')
    q = pd.Series(10., index=index)
    q.loc[index[254:258]] = 110
    station = dict(station='12345678', huc='01010001', role='development',
                   latitude=40, longitude=-80, area_km2=100, elevation_m=10)
    return q, station


def test_future_changes_labels_but_never_past_features():
    q, station = fixture_flow()
    rows, features, _ = decision_rows(q, 100., station)
    t = pd.Timestamp('2023-01-11T00:00Z')
    altered = q.copy()
    altered.loc[altered.index > t] = 10
    later, _, _ = decision_rows(altered, 100., station)
    a = rows.loc[rows.decision_time.eq(t.isoformat()) & rows.horizon.eq(24)].iloc[0]
    b = later.loc[later.decision_time.eq(t.isoformat()) & later.horizon.eq(24)].iloc[0]
    assert a.target == 1 and b.target == 0
    np.testing.assert_allclose(a[features].to_numpy(float), b[features].to_numpy(float))
    assert pd.Timestamp(a.event_time) == q.index[254]
    assert a.q_2 == np.log1p(.1)


def test_missing_hours_cannot_create_quiet_history_or_negative_labels():
    q, station = fixture_flow()
    q.iloc[250] = np.nan
    rows, _, events = decision_rows(q, 100., station)
    assert q.index[254].isoformat() not in events
    assert not ((rows.decision_time == q.index[240].isoformat()) & rows.horizon.eq(24)).any()
