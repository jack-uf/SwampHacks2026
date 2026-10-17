"""Experiment boundaries; fixtures are synthetic and never enter training files."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import wind_sprint
from scripts.wind_sprint import forecast_features, observation_days


def asos_days(rows):
    header = ['station', 'valid', 'gust']
    return '\n'.join([','.join(header)] + [f"{station},{valid},{gust}" for station, valid, gust in rows])


def full_day(day, gusts=None):
    gusts = gusts or ['10'] * 24
    return [('JFK', f'2024-01-{day:02d} {hour:02d}:53', gust) for hour, gust in enumerate(gusts)]


def test_measured_gusts_label_the_day_and_withhold_incomplete_days():
    rows = full_day(1)
    rows += full_day(2, ['12'] * 3 + ['35'] + ['10'] * 20)
    rows += [('JFK', f'2024-01-03 {hour:02d}:53', 'M') for hour in range(24)]
    rows += full_day(4)[:12]
    rows += [('JFK', f'2024-01-05 {hour:02d}:53', '20') for hour in list(range(6)) + list(range(12, 24))]
    days = observation_days(asos_days(rows))
    assert [row['target_date'] for row in days] == ['2024-01-01', '2024-01-02', '2024-01-03']
    quiet, gusty, reported = days
    assert quiet == dict(target_date='2024-01-01', max_gust_kmh=pytest.approx(18.52), exceedance=0, first_exceed_hour_utc=None)
    assert gusty['exceedance'] == 1
    assert gusty['max_gust_kmh'] == pytest.approx(64.82)
    assert gusty['first_exceed_hour_utc'] == 3
    assert reported == dict(target_date='2024-01-03', max_gust_kmh=None, exceedance=0, first_exceed_hour_utc=None)


def test_threshold_uses_the_exact_kmh_conversion_and_duplicate_reports_collapse():
    hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
    rows = [('JFK', f'2024-02-01 {hour:02d}:53', '32') for hour in hours]
    rows.append(('JFK', '2024-02-01 00:53', '33'))
    days = observation_days(asos_days(rows))
    assert len(days) == 1 and days[0]['exceedance'] == 1
    rows = [('JFK', f'2024-02-02 {hour:02d}:53', '32') for hour in hours]
    assert observation_days(asos_days(rows))[0]['exceedance'] == 0
    rows = [('JFK', f'2024-02-03 {hour:02d}:53', '32') for hour in hours[:6]]
    rows += [('JFK', f'2024-02-03 {hour:02d}:53', 'M') for hour in hours[6:]]
    days = observation_days(asos_days(rows))
    assert len(days) == 1 and days[0]['exceedance'] == 0 and days[0]['max_gust_kmh'] == pytest.approx(59.264)


def forecast():
    times = pd.date_range('2024-01-01', periods=240, freq='h')
    values = {
        'precipitation': [1.] * 240,
        'temperature_2m': [20.] * 240,
        'relative_humidity_2m': [80.] * 240,
        'wind_gusts_10m': [70.] * 240,
        'pressure_msl': [1000. - i // 24 for i in range(240)],
        'cape': [200.] * 240,
    }
    units = dict(precipitation='mm', temperature_2m='°C', relative_humidity_2m='%',
                 wind_gusts_10m='km/h', pressure_msl='hPa', cape='J/kg')
    return dict(utc_offset_seconds=0, timezone='GMT',
                hourly_units={k + '_previous_day2': v for k, v in units.items()},
                hourly=dict(time=times.strftime('%Y-%m-%dT%H:%M').tolist(),
                            **{k + '_previous_day2': v for k, v in values.items()}))


STATION = dict(sid='JFK', name='Test', state='NY', latitude=40.6, longitude=-73.8, elevation_m=4.0)


def test_features_use_complete_archived_forecasts_and_past_windows():
    payload = forecast()
    payload['hourly']['precipitation'] = [9999.] * 240
    rows = forecast_features(payload, STATION)
    assert [r['target_date'] for r in rows] == ['2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10']
    row = rows[0]
    assert row['issue_date'] == '2024-01-03'
    assert row['forecast_gust_max_kmh'] == 70. and row['forecast_gust_mean_kmh'] == 70.
    assert row['forecast_rain_mm'] == 24. and row['antecedent_forecast_rain_72h_mm'] == 72.
    assert row['forecast_pressure_min_hpa'] == 997. and row['forecast_pressure_change_24h_hpa'] == -1.
    assert row['forecast_cape_max_jkg'] == 200. and row['forecast_humidity_pct'] == 80.
    assert row['latitude'] == STATION['latitude'] and row['elevation_m'] == 4.
    changed = copy.deepcopy(payload)
    changed['hourly']['precipitation_previous_day2'][-24:] = [100.] * 24
    assert forecast_features(changed, STATION)[0] == row


@pytest.mark.parametrize('bad', [None, np.inf])
def test_incomplete_windows_are_withheld_without_imputation(bad):
    payload = forecast()
    payload['hourly']['precipitation_previous_day2'][0] = bad
    withheld = forecast_features(payload, STATION)
    assert withheld[0]['target_date'] == '2024-01-05' and '2024-01-04' not in [r['target_date'] for r in withheld]


def test_provider_unit_and_utc_mismatch_fail_fast():
    payload = forecast()
    payload['hourly_units']['wind_gusts_10m_previous_day2'] = 'm/s'
    with pytest.raises(ValueError, match='units'):
        forecast_features(payload, STATION)
    payload = forecast()
    payload['utc_offset_seconds'] = -18000
    with pytest.raises(ValueError, match='UTC'):
        forecast_features(payload, STATION)
    payload = forecast()
    payload['hourly']['time'][1] = payload['hourly']['time'][0]
    with pytest.raises(ValueError, match='unique'):
        forecast_features(payload, STATION)


def test_station_claim_order_keeps_roles_apart_and_drops_stranded_counties(tmp_path, monkeypatch):
    counties = [
        dict(state='IA', county_fips=19001, role='geographic_test', latitude=41.0, longitude=-93.0),
        dict(state='IA', county_fips=19002, role='development', latitude=41.05, longitude=-93.05),
        dict(state='IA', county_fips=19003, role='development', latitude=45.0, longitude=-100.0),
    ]
    source = tmp_path / 'county-source.json'
    source.write_text(json.dumps(dict(new_counties=counties)))
    document = tmp_path / 'protocol.md'
    document.write_text('frozen')
    monkeypatch.setattr(wind_sprint, 'OUT', tmp_path)
    monkeypatch.setattr(wind_sprint, 'COUNTY_SOURCE', source)
    monkeypatch.setattr(wind_sprint, 'DOCUMENT', document)
    monkeypatch.setattr(wind_sprint, 'MIN_STATIONS', 2)
    monkeypatch.setattr(wind_sprint, 'network', lambda state: [
        dict(sid='A', name='Shared', state='IA', latitude=41.0, longitude=-93.0, elevation_m=100.),
        dict(sid='B', name='Second', state='IA', latitude=41.5, longitude=-93.5, elevation_m=200.),
    ])
    wind_sprint.stations()
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    assert [(s['sid'], s['role']) for s in protocol['stations']] == [('A', 'geographic_test'), ('B', 'development')]
    assert [(c['county_fips'], c['station'], c['dropped']) for c in protocol['counties']][:2] == [
        (19001, 'A', None), (19002, 'B', None)]
    assert protocol['counties'][2]['dropped'] == 'No unclaimed station within 100 km'
    with pytest.raises(FileExistsError):
        wind_sprint.stations()


def test_partitions_keep_roles_and_periods_disjoint():
    rows = []
    for station, role in [('D1', 'development'), ('V1', 'geographic_validation'), ('T1', 'geographic_test')]:
        for date in ['2024-06-01', '2025-03-01', '2025-09-01', '2026-03-01']:
            rows.append(dict(station=station, role=role, target_date=date))
    parts = wind_sprint.partition_rows(pd.DataFrame(rows))
    assert set(parts['train'].station) == set(parts['calibration'].station) == {'D1'}
    assert set(parts['validation'].station) == {'D1'}
    assert set(parts['geographic_validation'].station) == {'V1'}
    assert set(parts['test'].station) == {'D1'}
    assert set(parts['geographic_test'].station) == {'T1'}
    assert parts['train'].target_date.max() < parts['calibration'].target_date.min()
    assert parts['calibration'].target_date.max() < parts['validation'].target_date.min()
    assert parts['validation'].target_date.max() < parts['test'].target_date.min()
    ids = [set(v.index) for v in parts.values()]
    assert sum(map(len, ids)) == len(set.union(*ids))
    bad = pd.DataFrame(rows)
    bad.loc[bad.station.eq('V1') & bad.target_date.eq('2025-03-01'), 'role'] = 'development'
    with pytest.raises(ValueError, match='crosses'):
        wind_sprint.partition_rows(bad)


def test_integration_failures_require_ap_gain_recall_and_fewer_false_alerts():
    updated = dict(positives=30, alert_rate=.12, recall=.6, false_alerts_per_100_station_days=3.)
    baseline = dict(recall=.5, false_alerts_per_100_station_days=5.)
    intervals = dict(available=True, delta_ap_95=[.01, .08])
    stations = {'JFK': dict(updated=dict(positives=10, recall=.6), baseline=dict(recall=.5))}
    assert wind_sprint.integration_failures(updated, baseline, intervals, stations, True) == []
    assert any('AP' in f for f in wind_sprint.integration_failures(updated, baseline, dict(intervals, delta_ap_95=[-.01, .08]), stations, True))
    assert any('Recall' in f for f in wind_sprint.integration_failures(dict(updated, recall=.4), baseline, intervals, stations, True))
    assert any('False alerts' in f for f in wind_sprint.integration_failures(
        dict(updated, false_alerts_per_100_station_days=4.6), baseline, intervals, stations, True))
    stations['JFK']['updated']['recall'] = .3
    assert any('JFK' in f for f in wind_sprint.integration_failures(updated, baseline, intervals, stations, True))
    assert any('unsupported' in f for f in wind_sprint.integration_failures(updated, baseline, intervals, {}, False))
    assert any('bootstrap' in f for f in wind_sprint.integration_failures(
        updated, baseline, dict(available=False), stations, True))


def test_paired_intervals_report_no_gain_for_identical_rule_and_model():
    frame = pd.DataFrame(dict(target_date=pd.date_range('2025-01-01', periods=100).strftime('%Y-%m-%d')))
    y = np.arange(100) % 5 == 0
    score = np.linspace(20., 90., 100)
    result = wind_sprint.paired_intervals(frame, y, score, score, 60., 60.)
    assert result['available']
    assert result['delta_ap_95'] == [0., 0.] and result['delta_recall_95'] == [0., 0.]
    assert result['delta_false_alerts_per_100_95'] == [0., 0.]


def test_lead_times_measure_from_the_previous_midnight_issue():
    frame = pd.DataFrame(dict(exceedance=[1, 1, 0], first_exceed_hour_utc=[6, 20, None]))
    alert = np.array([True, False, True])
    result = wind_sprint.lead_times(frame, alert)
    assert result == dict(count=1, minimum=30., median=30., maximum=30., all_at_least_24_hours=True)
    assert wind_sprint.lead_times(frame, np.zeros(3, dtype=bool)) == dict(count=0)


def test_frozen_selection_rejects_changed_code_or_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(wind_sprint, 'ROOT', tmp_path)
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    monkeypatch.setattr(wind_sprint, 'OUT', experiment)
    document = tmp_path / 'docs/WIND_SPRINT_PROTOCOL.md'
    document.parent.mkdir()
    document.write_text('frozen')
    monkeypatch.setattr(wind_sprint, 'DOCUMENT', document)
    dependency = tmp_path / 'scripts/train.py'
    dependency.parent.mkdir()
    dependency.write_text('frozen')
    artifacts = [(experiment / 'selected.joblib', 'model_sha256'), (experiment / 'training.csv.gz', 'dataset_sha256'),
                 (experiment / 'protocol.json', 'protocol_sha256'), (document, 'protocol_document_sha256'),
                 (tmp_path / 'scripts/wind_sprint.py', 'code_sha256')]
    selection = dict(dependencies={'scripts/train.py': wind_sprint.digest(dependency)},
                     versions={'numpy': wind_sprint.importlib.metadata.version('numpy')})
    for path, field in artifacts:
        path.parent.mkdir(exist_ok=True)
        path.write_text('frozen artifact')
        selection[field] = wind_sprint.digest(path)
    (experiment / 'selection.json').write_text(json.dumps(selection))
    assert wind_sprint.frozen_selection() == selection
    dependency.write_text('changed after selection')
    with pytest.raises(ValueError, match='Frozen dependency changed'):
        wind_sprint.frozen_selection()


def test_interrupted_final_evaluation_cannot_be_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(wind_sprint, 'OUT', tmp_path)
    monkeypatch.setattr(wind_sprint, 'frozen_selection', lambda: {})
    (tmp_path / 'selection.json').write_text('{}')
    def interrupted():
        raise RuntimeError('interrupted')
    monkeypatch.setattr(wind_sprint, 'load_dataset', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        wind_sprint.evaluate()
    with pytest.raises(FileExistsError):
        wind_sprint.evaluate()

