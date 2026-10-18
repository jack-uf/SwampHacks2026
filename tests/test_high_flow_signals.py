"""Synthetic leakage and source-contract checks for new predictive information."""
import numpy as np
import pandas as pd
import pytest

from scripts.high_flow_signals import meteorology, USABLE, VARIABLES
from scripts.high_flow_upstream import upstream_features
from scripts.high_flow_hrrr import precipitation_range


def test_rainfall_rolls_before_the_declared_delay_and_rejects_calendar_gaps():
    dates=pd.date_range('2020-01-01',periods=240,freq='h')
    hourly={v:np.ones(240) for v in VARIABLES}
    hourly['snow_depth']=np.full(240,np.nan)
    hourly['time']=[t.isoformat() for t in dates]
    before=meteorology(hourly,6)
    assert before.rain_6_lag6.iloc[200]==6
    assert before.rain_24_lag6.iloc[200]==24
    with pytest.raises(ValueError,match='Snow depth'):
        meteorology({**hourly,'snow_depth':np.ones(240)},6)
    changed={k:list(v) for k,v in hourly.items()}
    for v in USABLE:changed[v][195:]=[1000.]*(240-195)
    after=meteorology(changed,6)
    pd.testing.assert_series_equal(before.iloc[200],after.iloc[200])
    assert before.rain_24_lag6.iloc[201]!=after.rain_24_lag6.iloc[201]
    changed['time'][10]=changed['time'][9]
    with pytest.raises(ValueError,match='calendar'):meteorology(changed,6)


def test_upstream_delay_uses_calendar_hours_and_never_bridges_missing_history():
    dates=pd.date_range('2020-01-01',periods=220,freq='h',tz='UTC')
    q=pd.Series(np.arange(220,dtype=float)+10,index=dates)
    issue=dates[200]
    a=upstream_features(q,100,[issue])
    assert a.up_q_2.iloc[0]==pytest.approx(np.log1p(q.iloc[198]/100))
    changed=q.copy();changed.iloc[199:]=99999
    pd.testing.assert_frame_equal(a,upstream_features(changed,100,[issue]))
    changed=q.copy();changed.iloc[120]=np.nan
    assert upstream_features(changed,100,[issue]).isna().all().all()


def test_grib_range_selects_cumulative_field_and_excludes_next_message():
    text='1:0:d=2020010106:TMP:surface:12 hour fcst:\n2:100:d=2020010106:APCP:surface:11-12 hour acc fcst:\n3:200:d=2020010106:APCP:surface:0-12 hour acc fcst:\n4:500:d=2020010106:WEASD:surface:12 hour fcst:\n'
    assert precipitation_range(text)==(200,499)
    with pytest.raises(ValueError):precipitation_range(text.replace('0-12','0-6'))
    with pytest.raises(ValueError):precipitation_range(text+'5:800:d=2020010106:APCP:surface:0-12 hour acc fcst:\n6:1000:d=2020010106:TMP:surface:12 hour fcst:')
