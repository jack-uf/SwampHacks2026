import numpy as np
import pandas as pd

from scripts.response_research import flow_signature


def test_flow_signature_uses_only_pre2016_measured_history():
    q = pd.Series(np.tile([1., 2., 3., 4.], 1000),
                  index=pd.date_range('2000-01-01', periods=4000))
    expected = flow_signature(q, 86.4 * .028316846592)
    changed = pd.concat([q, pd.Series([1e12], index=pd.to_datetime(['2023-01-01']))])
    assert flow_signature(changed, 86.4 * .028316846592) == expected
    assert np.isclose(expected['sig_mean'], .625)
    assert np.isclose(np.expm1(expected['sig_runoff']), 4.)


def test_watershed_audit_rejects_nested_and_cross_boundary_basins():
    from shapely.geometry import box
    from scripts.response_research import disjoint_basin
    occupied = [box(0, 0, 2, 2)]
    assert not disjoint_basin(box(1, 1, 3, 3), occupied)
    assert not disjoint_basin(box(.5, .5, 1, 1), occupied)
    assert disjoint_basin(box(2, 0, 3, 1), occupied)


def test_onset_categories_keep_event_day_out_of_predictors():
    from scripts.response_research import onset_categories, response_features
    f = pd.DataFrame({'decision_date': ['2020-01-01'] * 4,
                      'event_date': ['', '2020-01-02', '2020-01-03', '2020-01-04'],
                      'exceedance': [0, 1, 1, 1]})
    assert onset_categories(f).tolist() == [0, 1, 2, 3]
    assert not {'event_date', 'exceedance', 'flow_1', 'flow_2', 'flow_3', 'station', 'huc'} & set(response_features())


def test_response_mixture_returns_valid_probabilities_and_reloads_exactly():
    import os
    import subprocess
    import sys
    # Torch shares no process with the suite's boosted-model libraries.
    program = '''
import torch
from scripts.response_network import ResponseMixture
torch.manual_seed(42)
model = ResponseMixture(3, 2, 4, True)
x, context = torch.zeros(5, 3), torch.zeros(5, 2)
p, trajectory = model(x, context)
assert p.shape == (5, 4) and trajectory.shape == (5, 3)
assert torch.all(p >= 0) and torch.allclose(p.sum(1), torch.ones(5))
clone = ResponseMixture(3, 2, 4, True)
clone.load_state_dict(model.state_dict())
assert torch.equal(clone(x, context)[0], p)
'''
    result = subprocess.run([sys.executable, '-c', program],
        capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2'))
    assert result.returncode == 0, result.stderr[-2000:]
