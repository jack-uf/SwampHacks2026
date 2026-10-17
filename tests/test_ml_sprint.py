"""Experiment boundaries; fixtures are synthetic and never enter training files."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.ml_sprint import forecast_features
from scripts import ml_sprint


def forecast():
    times = pd.date_range('2024-01-01', periods=240, freq='h')
    values = {
        'precipitation': [1.] * 240,
        'temperature_2m': [20.] * 240,
        'relative_humidity_2m': [80.] * 240,
        'wind_gusts_10m': [36.] * 240,
        'pressure_msl': [1000. - i // 24 for i in range(240)],
        'cape': [200.] * 240,
    }
    units = dict(precipitation='mm', temperature_2m='°C', relative_humidity_2m='%',
                 wind_gusts_10m='km/h', pressure_msl='hPa', cape='J/kg')
    return dict(utc_offset_seconds=0, timezone='GMT',
                hourly_units={k + '_previous_day2': v for k, v in units.items()},
                hourly=dict(time=times.strftime('%Y-%m-%dT%H:%M').tolist(),
                            **{k + '_previous_day2': v for k, v in values.items()}))


def test_features_use_complete_archived_forecasts_and_past_windows():
    payload = forecast()
    payload['hourly']['precipitation'] = [9999.] * 240
    rows = forecast_features(payload, 40., -100.)
    assert [r['target_date'] for r in rows] == ['2024-01-08', '2024-01-09', '2024-01-10']
    row = rows[0]
    assert row['issue_date'] == '2024-01-07'
    assert row['antecedent_forecast_rain_7d_mm'] == 168.
    assert row['forecast_rain_mm'] == 24.
    assert row['forecast_pressure_min_hpa'] == 993.
    assert row['forecast_pressure_change_24h_hpa'] == -1.
    assert row['forecast_cape_max_jkg'] == 200.
    changed = copy.deepcopy(payload)
    changed['hourly']['precipitation_previous_day2'][9 * 24:] = [100.] * 24
    assert forecast_features(changed, 40., -100.)[0] == row


@pytest.mark.parametrize('bad', [None, np.inf])
def test_incomplete_antecedent_is_withheld_without_imputation(bad):
    payload = forecast()
    payload['hourly']['precipitation_previous_day2'][0] = bad
    assert [r['target_date'] for r in forecast_features(payload, 40., -100.)] == ['2024-01-09', '2024-01-10']


def test_provider_unit_and_utc_mismatch_fail_fast():
    payload = forecast()
    payload['hourly_units']['wind_gusts_10m_previous_day2'] = 'm/s'
    with pytest.raises(ValueError, match='units'):
        forecast_features(payload, 40., -100.)
    payload = forecast()
    payload['utc_offset_seconds'] = -18000
    with pytest.raises(ValueError, match='UTC'):
        forecast_features(payload, 40., -100.)


def test_duplicate_or_missing_forecast_hour_is_not_accepted():
    payload = forecast()
    payload['hourly']['time'][1] = payload['hourly']['time'][0]
    with pytest.raises(ValueError, match='unique'):
        forecast_features(payload, 40., -100.)
    payload = forecast()
    for values in payload['hourly'].values():
        del values[0]
    assert [r['target_date'] for r in forecast_features(payload, 40., -100.)] == ['2024-01-09', '2024-01-10']


def test_partitions_exclude_inspected_tests_and_all_years_of_new_test_states():
    rows = []
    for state, role, origin in [('FL', 'development', 'original'), ('KS', 'development', 'new'),
                                ('IA', 'geographic_validation', 'new'), ('OR', 'geographic_test', 'new')]:
        for date in ['2024-06-01', '2025-03-01', '2025-09-01', '2026-03-01']:
            rows.append(dict(state=state, role=role, origin=origin, target_date=date))
    frame = pd.DataFrame(rows)
    parts = ml_sprint.partition_rows(frame)
    assert set(parts['train'].state) == set(parts['calibration'].state) == {'FL', 'KS'}
    assert set(parts['validation'].state) == {'FL', 'KS'}
    assert set(parts['geographic_validation'].state) == {'IA'}
    assert set(parts['test'].state) == {'KS'}
    assert set(parts['geographic_test'].state) == {'OR'}
    assert parts['train'].target_date.max() < parts['calibration'].target_date.min()
    assert parts['calibration'].target_date.max() < parts['validation'].target_date.min()
    assert parts['validation'].target_date.max() < parts['test'].target_date.min()
    ids = [set(v.index) for v in parts.values()]
    assert sum(map(len, ids)) == len(set.union(*ids))
    bad = frame.copy()
    bad.loc[bad.state.eq('IA'), 'state'] = 'KS'
    with pytest.raises(ValueError, match='state'):
        ml_sprint.partition_rows(bad)


def test_threshold_preserves_reference_recall_and_does_not_relax_alert_budget():
    y = np.array([1] * 5 + [0] * 25)
    p = np.array([.9, .8, .7, .6, .1] + [.2] * 25)
    reference = np.array([.8] * 4 + [.1] * 26)
    threshold, supported = ml_sprint.recall_threshold(y, p, reference, .5)
    assert threshold == .6 and supported
    assert (p[y == 1] >= threshold).sum() == (reference[y == 1] >= .5).sum() == 4
    p[y == 0] = .65
    threshold, supported = ml_sprint.recall_threshold(y, p, reference, .5)
    assert threshold == .6 and not supported
    with pytest.raises(ValueError, match='both classes'):
        ml_sprint.recall_threshold(np.zeros(5), np.ones(5) * .1, np.zeros(5), .5)


def test_promotion_rejects_recall_tradeoff_uncertainty_and_regional_harm():
    updated = dict(positives=30, alert_rate=.15, recall=.6, false_alerts_per_100_county_days=3.)
    baseline = dict(recall=.5, false_alerts_per_100_county_days=5.)
    intervals = dict(available=True, delta_ap_95=[.01, .08], delta_brier_95=[-.02, -.001])
    states = {'KS': dict(updated=dict(positives=10, recall=.6), served=dict(recall=.5))}
    assert ml_sprint.promotion_failures(updated, baseline, intervals, states, True) == []
    assert any('recall' in r.lower() for r in ml_sprint.promotion_failures(dict(updated, recall=.4), baseline, intervals, states, True))
    uncertain = dict(intervals, delta_ap_95=[-.01, .08])
    assert any('AP' in r for r in ml_sprint.promotion_failures(updated, baseline, uncertain, states, True))
    states['KS']['updated']['recall'] = .3
    assert any('KS' in r for r in ml_sprint.promotion_failures(updated, baseline, intervals, states, True))
    assert any('unsupported' in r for r in ml_sprint.promotion_failures(updated, baseline, intervals, {}, False))


def test_paired_intervals_report_no_gain_for_identical_predictions():
    frame = pd.DataFrame(dict(target_date=pd.date_range('2025-01-01', periods=100).strftime('%Y-%m-%d')))
    y = np.arange(100) % 5 == 0
    p = np.linspace(.01, .7, 100)
    result = ml_sprint.paired_intervals(frame, y, p, p, .3, .3)
    assert result['available']
    for name in ['ap', 'brier', 'recall', 'false_alerts_per_100']:
        assert result['delta_' + name + '_95'] == [0., 0.]


def test_frozen_selection_rejects_changed_code_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_sprint, 'ROOT', tmp_path)
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    monkeypatch.setattr(ml_sprint, 'OUT', experiment)
    artifacts = [(experiment / 'selected.joblib', 'model_sha256'),
                 (experiment / 'training.csv.gz', 'dataset_sha256'),
                 (experiment / 'protocol.json', 'protocol_sha256'),
                 (tmp_path / 'models/hazards.joblib', 'served_model_sha256'),
                 (tmp_path / 'docs/ML_SPRINT_PROTOCOL.md', 'protocol_document_sha256'),
                 (tmp_path / 'scripts/ml_sprint.py', 'code_sha256')]
    selection = {}
    for path, field in artifacts:
        path.parent.mkdir(exist_ok=True)
        path.write_text('frozen artifact')
        selection[field] = ml_sprint.digest(path)
    (experiment / 'selection.json').write_text(json.dumps(selection))
    assert ml_sprint.frozen_selection() == selection
    artifacts[-1][0].write_text('changed after selection')
    with pytest.raises(ValueError, match='Frozen selection input changed'):
        ml_sprint.frozen_selection()
