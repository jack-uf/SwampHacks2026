from datetime import timedelta

import pytest

from backend import main, planner
from backend.domain import now_utc
from test_api import client, AUTH  # noqa: F401  (shared fixture with stubbed providers)


def hourly(start, hours, gust, heat, rain, soil=.2, deep=.25):
    return [dict(time=(start+timedelta(hours=i)).isoformat(), apparent_temperature=heat, wind_gusts_10m=gust,
                 precipitation=rain, soil_moisture_0_to_1cm=soil, soil_moisture_0_to_100cm=deep) for i in range(hours)]


def weather(rows, rain_72h=0.0):
    return dict(source='Open-Meteo numerical weather models', forecast=rows, summary=dict(rain_72h_mm=rain_72h),
                feed=dict(available=True, fetched_at=now_utc().isoformat()))


def anchor():
    return now_utc().replace(minute=0, second=0, microsecond=0)


def test_planner_ranks_the_calm_window_first_and_keeps_jha_bands():
    start = anchor()
    calm, rough = hourly(start, 6, 15, 20, 0.0), hourly(start+timedelta(hours=6), 6, 70, 41, 30, soil=.5, deep=.45)
    plan = planner.plan(weather(calm+rough), main.missions.ACTIVITIES['excavation_trenching'], 6, hours=12, step_hours=6)
    assert plan['candidate_count'] == 2 and plan['recommended'] == 0
    best, worst = plan['windows'][0], plan['windows'][-1]
    assert best['risk'] < worst['risk'] and best['label'] == 'low' and best['saturated'] is False
    assert worst['restricted'] is True and 'flood high' in worst['drivers'] and worst['saturated'] is True
    assert plan['recommended_start'] == best['start']


def test_activity_wind_limits_drive_the_restriction():
    start = anchor()
    windy = hourly(start, 6, 50, 22, 0.0)
    plan = planner.plan(weather(windy), main.missions.ACTIVITIES['aerial_lift_crane'], 6, hours=6, step_hours=6)
    assert plan['windows'][0]['restricted'] is True and 'storm high' in plan['windows'][0]['drivers']


def test_ensemble_exceedance_counts_members_crossing_the_activity_limit():
    start = anchor()
    times = [(start+timedelta(hours=i)).isoformat() for i in range(6)]
    members = dict(wind_gusts_10m=[[10]*6, [50]*6, [60]*6], apparent_temperature=[[20]*6]*3, precipitation=[[0]*6]*3)
    out = planner.exceedances(dict(available=True, times=times, members=members, model='gfs_seamless'),
                             start, start+timedelta(hours=6), main.missions.ACTIVITIES['aerial_lift_crane'])
    assert out['members'] == 3 and out['gust_high_probability'] == pytest.approx(round(2/3, 3))
    assert out['gust_limit_kmh'] == 45 and out['gust_p90_kmh'] == 60


def test_windows_without_half_the_forecast_hours_are_skipped():
    plan = planner.plan(weather(hourly(anchor(), 3, 10, 20, 0.0)), main.missions.ACTIVITIES['general_site_work'],
                        6, hours=12, step_hours=6)
    assert plan['candidate_count'] == 1 and plan['skipped'] == 1
    assert plan['windows'][0]['coverage'] == .5 and plan['windows'][0]['unknown'] == []


def test_work_window_endpoint_reports_windows_and_degrades_without_helpers(client):
    site = main.SITES[0]
    start = (now_utc()+timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    created = client.post('/api/missions', json=dict(site_id=site['id'], name='Window check', activity='general_site_work',
        latitude=site['latitude'], longitude=site['longitude'], shift_start=start.isoformat(),
        shift_end=(start+timedelta(hours=8)).isoformat(), crew_size=3, supervisor='A. Rivera'), headers=AUTH)
    assert created.status_code == 200
    mid = created.json()['id']
    body = client.get(f'/api/missions/{mid}/work-windows', params=dict(hours=48, step_hours=3)).json()
    assert body['duration_hours'] == 8 and body['anchor'] and body['mission_id'] == mid
    for key in ['windows', 'recommended', 'skipped', 'candidate_count', 'notes']:
        assert key in body
    assert client.get('/api/missions/nope/work-windows').status_code == 404
    assert client.get(f'/api/missions/{mid}/work-windows', params=dict(hours=1)).status_code == 422
