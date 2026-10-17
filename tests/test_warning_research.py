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


def test_warning_budget_batches_regions_and_is_order_independent():
    from scripts.warning_research import budget_alerts
    f = pd.DataFrame(dict(station=[f's{i:03}' for i in range(400)],
        huc=['01010000'] * 200 + ['02010000'] * 200,
        decision_date=['2023-01-01'] * 400, exceedance=[0] * 400))
    p = np.linspace(.1, .9, 400)
    alerts = budget_alerts(f, p, .5)
    assert np.flatnonzero(alerts).tolist() == [398, 399]
    perm = np.random.default_rng(7).permutation(400)
    shuffled = budget_alerts(f.iloc[perm], p[perm], .5)
    assert np.array_equal(shuffled[np.argsort(perm)], alerts)
    with pytest.raises(ValueError, match='duplicate'):
        budget_alerts(pd.concat([f, f.iloc[:1]]), np.full(401, .5), .5)
    with pytest.raises(ValueError, match='probabilities'):
        budget_alerts(f, np.full(400, np.nan), .5)


def test_warning_budget_accepts_standard_huc12_subwatersheds():
    from scripts.warning_research import budget_alerts
    f = pd.DataFrame(dict(station=[str(i) for i in range(100)],
        huc=['010100000001'] * 100, decision_date=['2023-01-01'] * 100, exceedance=[0] * 100))
    assert budget_alerts(f, np.full(100, .8), .5).sum() == 1


def test_matched_supervision_losses_preserve_category_information():
    # Torch stays in a separate process from any boosting imported by the suite.
    import subprocess
    from pathlib import Path
    result = subprocess.run([str(Path.cwd() / '.venv/bin/python'), '-c', '''
import torch
from scripts.warning_network import supervision_loss
p = torch.tensor([[.2, .1, .6, .1], [.2, .6, .1, .1]])
y = torch.tensor([2, 2])
binary = supervision_loss(p, y, 'binary', reduction='none')
day = supervision_loss(p, y, 'day', reduction='none')
assert torch.allclose(binary, torch.tensor([.22314355, .22314355]))
assert torch.allclose(day, torch.tensor([.51082562, 2.30258509]))
assert day[0] < day[1]
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

