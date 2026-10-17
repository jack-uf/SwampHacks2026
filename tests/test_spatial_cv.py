import numpy as np
import pandas as pd
from scripts import spatial_cv
from scripts.build_training import FEATURES


def synthetic():
    rng = np.random.default_rng(0)
    dates = pd.date_range('2024-01-01', '2026-03-31', freq='2D').strftime('%Y-%m-%d')
    rows = []
    for region, state, role in [('A, FL', 'FL', 'development'), ('B, TX', 'TX', 'development'), ('C, OK', 'OK', 'development'),
                                ('D, AZ', 'AZ', 'geographic_test')]:
        for d in dates:
            row = {f: rng.normal() for f in FEATURES}
            rows.append(dict(row, target_date=d, region=region, state=state, role=role,
                             flood=int(rng.random() < .15), storm=int(rng.random() < .15)))
    return pd.DataFrame(rows)


def test_only_2024_2025_development_rows_are_read():
    kept = spatial_cv.development_rows(synthetic())
    assert set(kept.role) == {'development'} and kept.target_date.max() < '2026-01-01'


def test_left_out_state_never_fits_or_calibrates_its_own_model():
    result = spatial_cv.score_hazard(spatial_cv.development_rows(synthetic()), 'flood')
    scored = result['rows']
    assert scored.target_date.min() >= '2025-07-01' and scored.target_date.max() < '2026-01-01'
    assert set(scored.state) == {'FL', 'TX', 'OK'}
    for state, fold in result['folds'].items():
        assert fold['status'] == 'ok' and state not in fold['fit_states'] and state not in fold['calibration_states']
    assert scored[['seen', 'left_out']].notna().all().all() and not np.allclose(scored.seen, scored.left_out)
