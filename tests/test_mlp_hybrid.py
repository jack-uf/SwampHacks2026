"""Experiment boundaries; fixtures are synthetic and never enter research data.

torch and the boosted libraries cannot share one process here, so network fitting
runs in a subprocess exactly as it does in the experiment.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_ceiling_rule_spends_the_free_budget_the_row_rule_leaves_on_the_table():
    from scripts.hydrology_research import budget_threshold
    from scripts.mlp_hybrid_worker import ceiling_threshold
    generator = np.random.default_rng(0)
    negative_values = np.linspace(.90, .31, 60)
    probabilities = np.r_[negative_values, np.full(20, .95), np.full(10, .65),
                          np.full(1910, .01), generator.uniform(.2, .3, 0)]
    target = np.r_[np.zeros(60), np.ones(20), np.ones(10), np.zeros(1910)].astype(int)
    frame = pd.DataFrame(dict(target=target, huc=['01'] * len(target)))
    row = budget_threshold(target, probabilities, .01)
    ceiling = ceiling_threshold(frame, probabilities)
    assert ceiling < row
    detected_row = int(((probabilities >= row) & (target == 1)).sum())
    detected_ceiling = int(((probabilities >= ceiling) & (target == 1)).sum())
    assert detected_ceiling > detected_row
    false_rate = 100 * ((probabilities >= ceiling) & (target == 0)).sum() / len(target)
    assert false_rate <= 1.5 + 1e-9


def test_ceiling_rule_stops_at_a_regional_violation():
    from scripts.mlp_hybrid_worker import ceiling_threshold
    # Region 02 has no headroom: dropping below .5 would put it above three per hundred.
    probabilities = np.r_[np.full(40, .95), np.full(40, .50), np.full(200, .01),
                          np.full(20, .50), np.full(430, .01)]
    target = np.r_[np.ones(40), np.zeros(40), np.zeros(200), np.zeros(20), np.zeros(430)].astype(int)
    huc = np.array(['01'] * 280 + ['02'] * 450)
    frame = pd.DataFrame(dict(target=target, huc=huc))
    chosen = ceiling_threshold(frame, probabilities)
    assert chosen >= .50
    region_rows = 450
    assert 100 * ((huc == '02') & (probabilities >= chosen) & (target == 0)).sum() / region_rows <= 3 + 1e-9


def test_prior_features_exclude_the_current_decision():
    from scripts.mlp_hybrid_worker import prior_features
    times = pd.date_range('2024-01-01', periods=10, freq='6h', tz='UTC')
    frame = pd.DataFrame(dict(station=['one'] * 10, horizon=[6] * 10,
        decision_time=[t.isoformat() for t in times], q_2=[1.] * 5 + [-1.] * 5))
    features = prior_features(frame).sort_values('decision_time')
    assert features.prior_above.tolist() == [1.] * 5 + [0.] * 5
    # The first row sees no history, and every window stops before the current row.
    assert features.days_since_above.iloc[0] == 365
    assert features.trailing_30d_above.iloc[0] == 0.
    assert features.trailing_30d_above.iloc[5] == pytest.approx(1.)
    assert features.days_since_above.iloc[5] == pytest.approx(.25)
    altered = frame.copy()
    altered.loc[5, 'q_2'] = 99.
    after = prior_features(altered).sort_values('decision_time')
    assert after.trailing_30d_above.iloc[5] == pytest.approx(features.trailing_30d_above.iloc[5])


def test_physical_extrapolation_accounts_for_the_two_hour_delay():
    from scripts.mlp_hybrid_worker import physical_score
    frame = pd.DataFrame(dict(q_2=[.5], q_8=[.2], horizon=[6]))
    assert physical_score('linear', frame)[0] == pytest.approx(.9)
    assert .5 < physical_score('damped', frame)[0] < .9
    assert physical_score('proximity', frame)[0] == .5


HYBRID_SCRIPT = '''
import json

import numpy as np
import pandas as pd

from scripts.mlp_hybrid_network import OFFSET_COLUMN, fit_network, network_raw
from scripts.shift_hydrology import FEATURES


def frame(rows=400, seed=0):
    generator = np.random.default_rng(seed)
    result = pd.DataFrame(generator.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    result[OFFSET_COLUMN] = generator.normal(size=rows)
    result["target"] = (result.q_2 > result.q_2.median()).astype(int)
    return result


train = frame()
report = {}
for label, offset in [("direct", False), ("hybrid", True)]:
    first = fit_network(label, FEATURES, train, train, offset=offset)
    second = fit_network(label, FEATURES, train, train, offset=offset)
    left, right = network_raw(first, train), network_raw(second, train)
    report[label] = dict(deterministic=bool(np.allclose(left, right)), rows=int(left.shape[0]),
                         parameters=int(first["parameters"]), offsets=bool(first["offset"]))
# A fitted hybrid must not reduce to the direct network: the shift term has to reach the logit.
report["differs"] = bool(not np.allclose(network_raw(fit_network("hybrid", FEATURES, train, train, offset=True), train),
                                        network_raw(fit_network("direct", FEATURES, train, train, offset=False), train)))
print(json.dumps(report))
'''


def test_hybrid_offset_is_deterministic_and_reaches_the_logit():
    from scripts.mlp_hybrid import ROOT
    completed = subprocess.run([sys.executable, '-c', HYBRID_SCRIPT], capture_output=True, text=True,
                               cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    assert completed.returncode == 0, completed.stderr[-900:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report['direct']['deterministic'] and report['hybrid']['deterministic']
    assert report['direct']['rows'] == 400 and report['hybrid']['parameters'] > 0
    assert report['hybrid']['offsets'] and not report['direct']['offsets']
    assert report['differs']


def test_development_collection_never_requests_the_final_year(tmp_path, monkeypatch):
    import scripts.mlp_hybrid as research
    from urllib.parse import urlparse, parse_qs
    intervals = []

    def page(path, url, client=None):
        intervals.append(parse_qs(urlparse(url).query)['datetime'][0])
        return {'features': [], 'links': []}

    monkeypatch.setattr(research, 'OUT', tmp_path)
    monkeypatch.setattr(research, 'cached_page', page)
    assert research.observations('12345678', 2024).empty
    assert len(intervals) == 4
    assert pd.Timestamp(intervals[-1].split('/')[1]) < pd.Timestamp('2025-01-01T00:00Z')
