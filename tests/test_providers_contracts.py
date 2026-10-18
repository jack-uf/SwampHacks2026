import json
from pathlib import Path

from backend import providers

FIXTURES = Path(__file__).parent / 'fixtures'


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_gauge_trend_reports_direction_and_headroom_or_abstains():
    rising = providers.gauge_trend([('2026-10-18T00:00:00+00:00', 2.0), ('2026-10-18T06:00:00+00:00', 2.4),
                                    ('2026-10-18T12:00:00+00:00', 3.1)], flood_stage=4.0)
    assert rising['available'] is True and rising['direction'] == 'rising'
    assert rising['change_6h_m'] == 0.7 and rising['change_24h_m'] is None
    assert rising['headroom_m'] == 0.9 and rising['above_flood_stage'] is False
    falling = providers.gauge_trend([('2026-10-17T12:00:00+00:00', 3.0), ('2026-10-18T12:00:00+00:00', 2.4)])
    assert falling['direction'] == 'falling' and falling['change_24h_m'] == -0.6
    flat = providers.gauge_trend([('2026-10-17T12:00:00+00:00', '3.0'), ('2026-10-18T12:00:00+00:00', '3.01')])
    assert flat['direction'] == 'flat'
    assert providers.gauge_trend([('2026-10-18T12:00:00+00:00', 3.0)])['available'] is False
    assert providers.gauge_trend([('not-a-time', 3.0), ('2026-10-18T12:00:00+00:00', 'n/a')])['available'] is False
    above = providers.gauge_trend([('2026-10-18T00:00:00+00:00', 3.9), ('2026-10-18T06:00:00+00:00', 4.2)], flood_stage=4.0)
    assert above['above_flood_stage'] is True and above['headroom_m'] == -0.2


def test_hourly_rows_drops_ragged_or_non_list_variables_without_failing():
    rows, dropped = providers.hourly_rows(dict(time=['2026-10-18T00:00', '2026-10-18T01:00'],
                                              temperature_2m=[20, 21], precipitation=[0],
                                              wind_gusts_10m='not-a-list'))
    assert dropped == ['precipitation', 'wind_gusts_10m']
    assert rows == [dict(time='2026-10-18T00:00Z', temperature_2m=20),
                    dict(time='2026-10-18T01:00Z', temperature_2m=21)]
    assert providers.stamp_zulu('2026-10-18T00:00Z') == '2026-10-18T00:00Z'
    assert providers.stamp_zulu('2026-10-18T00:00+00:00') == '2026-10-18T00:00+00:00'
    assert providers.hourly_rows({}) == ([], [])


def test_ensemble_columns_keeps_complete_numeric_members_including_naming_variants():
    hourly = fixture('ensemble-response.json')['hourly']
    members, dropped = providers.ensemble_columns(hourly, ['wind_gusts_10m', 'apparent_temperature', 'precipitation'])
    assert set(members) == {'wind_gusts_10m', 'apparent_temperature', 'precipitation'}
    assert len(members['wind_gusts_10m']) == 3  # control plus two complete members
    assert len(members['apparent_temperature']) == 2  # members only, spelling variants accepted
    assert members['precipitation'] == [[0, 0.4], [0, 1.2]]
    assert 'precipitation_member02' in dropped  # ragged, so not trusted


def test_elevation_value_rejects_missing_unparseable_and_implausible_samples():
    values = {name: providers.elevation_value(value) for name, value in fixture('terrain-values.json').items()}
    assert values == dict(centre=14.2, north=15.8, south=12.9, east=None, west=None)
    assert providers.elevation_value(None) is None
    assert providers.elevation_value('nan') is None
    assert providers.elevation_value({'value': '12.5'}) == 12.5


def test_terrain_summary_reports_missing_points_instead_of_inventing_relief():
    samples = {name: providers.elevation_value(value) for name, value in fixture('terrain-values.json').items()}
    samples = {name: value for name, value in samples.items() if value is not None}
    partial = providers.terrain_summary(samples, 'fixture')
    assert partial['available'] is False and sorted(partial['missing']) == ['east', 'west']
    complete = providers.terrain_summary(dict(samples, east=13.0, west=13.6), 'fixture')
    assert complete['available'] is True and complete['elevation_m'] == 14.2
    assert complete['relief_m'] == 2.9 and complete['slope_pct'] > 0 and complete['missing'] == []
