import numpy as np
import pandas as pd
import pytest


def test_warning_budget_obeys_regional_prefix_bound_and_delays_refunds():
    from scripts.warning_research import budget_alerts
    f = pd.DataFrame(dict(station=['a'] * 240, huc=['01010000'] * 240,
                          decision_date=pd.date_range('2023-01-01', periods=240).strftime('%Y-%m-%d'),
                          exceedance=np.zeros(240, dtype=int)))
    alerts = budget_alerts(f, np.full(240, .9), .5)
    assert np.flatnonzero(alerts).tolist() == [99, 199]
    f.loc[99, 'exceedance'] = 1
    refunded = budget_alerts(f, np.full(240, .9), .5)
    assert np.flatnonzero(refunded).tolist() == [99, 109, 199]
    assert np.array_equal(refunded[:109], alerts[:109])
    assert np.all(np.cumsum(refunded & f.exceedance.eq(0)) <= np.arange(1, 241) / 100)
    changed = f.copy(); changed.loc[220:, 'exceedance'] = 1
    assert np.array_equal(budget_alerts(changed, np.full(240, .9), .5)[:230], refunded[:230])

