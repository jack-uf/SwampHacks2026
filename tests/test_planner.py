from datetime import timedelta

import pytest

from backend import main, planner
from backend.domain import now_utc, parse_time
from test_api import client, AUTH  # noqa: F401  (shared fixture with stubbed providers)


def hourly(start, hours, gust, heat, rain, soil=.2, deep=.25):
    return [dict(time=(start+timedelta(hours=i)).isoformat(), apparent_temperature=heat, wind_gusts_10m=gust,
                 precipitation=rain, soil_moisture_0_to_1cm=soil, soil_moisture_0_to_100cm=deep) for i in range(hours)]


def weather(rows, rain_72h=0.0):
    return dict(source='Open-Meteo numerical weather models', forecast=rows, summary=dict(rain_72h_mm=rain_72h),
                feed=dict(available=True, fetched_at=now_utc().isoformat()))


def anchor():
    return now_utc().replace(minute=0, second=0, microsecond=0)


def test_non_aligned_plan_is_compared_and_candidate_hours_survive_routine_refresh():
    start = anchor()
    feed = weather(hourly(start, 50, 20, 24, 0.0))
    planned = start + timedelta(hours=2, minutes=17)
    kwargs = dict(hours=48, step_hours=1, planned_start=planned.isoformat(), timezone='UTC', flex_hours=2)
    a = planner.plan(feed, activity('general_site_work'), 8,
                     now=(start+timedelta(minutes=7, seconds=12)).isoformat(), **kwargs)
    b = planner.plan(feed, activity('general_site_work'), 8,
                     now=(start+timedelta(minutes=7, seconds=22)).isoformat(), **kwargs)
    assert a['planned']['start'] == b['planned']['start'] == planned.isoformat()
    assert a['recommendation']['choice'] == 'keep'
    assert [w['start'] for w in a['windows']] == [w['start'] for w in b['windows']]
    assert a['candidate_count'] == b['candidate_count']
    # A fixed restriction is still applied to the added exact planned window.
    restricted = planner.plan(weather(hourly(start, 50, 80, 24, 0.0)), activity('aerial_lift_crane'), 8,
                              now=(start+timedelta(minutes=7)).isoformat(), **kwargs)
    assert restricted['planned']['restricted'] is True


def activity(name):
    return main.missions.ACTIVITIES[name]


def window(name, risk, heat, flood, share, restricted=False, unknown=()):
    band_of = lambda level: 'low' if level < .5 else 'high' if level >= 1 else 'elevated'
    return dict(start=name, risk=risk, restricted=restricted, unknown=list(unknown),
                ratings=dict(heat=band_of(heat), flood=band_of(flood)), levels=dict(heat=heat, flood=flood),
                uncertainty=dict(restrict_share=share))


def test_severity_is_continuous_and_ranks_by_exposure_not_clock_time():
    assert planner.severity(25, *planner.HEAT) == 0 and planner.severity(20, *planner.HEAT) == 0
    assert .5 < planner.severity(38, *planner.HEAT) < 1 and planner.severity(33, *planner.HEAT) > planner.severity(31, *planner.HEAT)
    start = anchor()
    first = hourly(start, 8, 30, 33, 0.0)
    second = [dict(row, apparent_temperature=22.0) for row in hourly(start+timedelta(hours=8), 8, 30, 22, 0.0)]
    plan = planner.plan(weather(first+second), activity('general_site_work'), 8, hours=16, step_hours=8,
                        planned_start=first[0]['time'])
    assert plan['windows'][0]['start'] == second[0]['time'] and plan['windows'][0]['risk'] < plan['windows'][1]['risk']
    assert plan['recommendation']['choice'] == 'move' and 'heat' in plan['recommendation']['detail']


def test_decide_requires_a_material_band_change_and_no_trade():
    decisive = ['heat', 'flood']
    planned = window('a', 1.0, .9, .2, .6)
    best = window('b', .3, .0, .2, .1)
    assert planner.decide([planned, best], planned, decisive, True)['choice'] == 'move'
    assert planner.decide([planned, best], planned, decisive, False)['choice'] == 'move'
    assert planner.decide([planned, best], planned, decisive, False)['member_veto'] is None
    near = window('c', .95, .85, .2, .55)
    assert planner.decide([planned, near], planned, decisive, True)['choice'] == 'keep'
    trade = window('d', .3, .0, .8, .1)
    assert planner.decide([planned, trade], planned, decisive, True)['choice'] == 'keep'
    assert 'worse' in planner.decide([planned, trade], planned, decisive, True)['reason']
    veto = window('e', .3, .0, .2, .8)
    assert planner.decide([planned, veto], planned, decisive, True)['choice'] == 'keep'
    assert planner.decide([planned, veto], planned, decisive, True)['member_veto'] is True
    restricted_plan = window('a', 1.2, 1.0, .2, .9, restricted=True)
    assert planner.decide([restricted_plan, best], restricted_plan, decisive, True)['choice'] == 'move'
    unknown_plan = window('a', 1.0, .9, .2, .6, unknown=['storm'])
    assert planner.decide([unknown_plan, best], unknown_plan, decisive, True)['choice'] == 'insufficient'
    assert planner.decide([window('f', 1.0, 1.0, .9, .9, restricted=True)], window('f', 1.0, 1.0, .9, .9, restricted=True),
                          decisive, True)['choice'] == 'keep'
    assert planner.decide([], None, decisive, True)['choice'] == 'insufficient'
    assert planner.decide([planned, best], None, decisive, True)['choice'] == 'insufficient'


def test_constraints_exclude_hour_band_and_site_overlapping_windows():
    start = anchor()
    busy_start, busy_end = start+timedelta(hours=26), start+timedelta(hours=34)
    plan = planner.plan(weather(hourly(start, 48, 20, 24, 0.0)), activity('general_site_work'), 8, hours=48, step_hours=1,
                        now=(start+timedelta(hours=6)).isoformat(), planned_start=(start+timedelta(hours=8)).isoformat(),
                        timezone='UTC', flex_hours=1, busy=[(busy_start.isoformat(), busy_end.isoformat())])
    planned_hour = planner.local_hour((start+timedelta(hours=8)).isoformat(), 'UTC')
    assert plan['excluded']['hour_band'] > 0 and plan['excluded']['overlap'] > 0
    assert plan['constraints']['busy_intervals'] == 1
    assert plan['constraints']['planned_local_hour'] == round(planned_hour, 2)
    for item in plan['windows']:
        begin, end = parse_time(item['start']), parse_time(item['end'])
        assert begin >= parse_time(plan['constraints']['timezone'] and (start+timedelta(hours=6)).isoformat())
        assert planner.hour_distance(planner.local_hour(item['start'], 'UTC'), planned_hour) <= 1 + 1e-9
        assert not (begin < busy_end and busy_start < end)


def test_local_hour_band_uses_the_site_timezone():
    start = anchor()
    planned = (start+timedelta(hours=16)).isoformat()
    plan = planner.plan(weather(hourly(start, 48, 20, 24, 0.0)), activity('general_site_work'), 8, hours=48, step_hours=1,
                        now=start.isoformat(), planned_start=planned, timezone='America/Los_Angeles', flex_hours=0)
    assert plan['excluded']['hour_band'] > 0
    local = planner.local_hour(planned, 'America/Los_Angeles')
    for item in plan['windows']:
        assert planner.hour_distance(planner.local_hour(item['start'], 'America/Los_Angeles'), local) <= 1e-9


def test_activity_wind_limits_drive_the_restriction():
    start = anchor()
    plan = planner.plan(weather(hourly(start, 6, 50, 22, 0.0)), activity('aerial_lift_crane'), 6, hours=6, step_hours=6,
                        planned_start=start.isoformat())
    assert plan['windows'][0]['restricted'] is True and 'storm high' in plan['windows'][0]['drivers']
    assert plan['recommendation']['choice'] == 'keep'


def test_ensemble_exceedance_counts_members_crossing_the_activity_limit():
    start = anchor()
    times = [(start+timedelta(hours=i)).isoformat() for i in range(6)]
    members = dict(wind_gusts_10m=[[10]*6, [50]*6, [60]*6], apparent_temperature=[[20]*6]*3, precipitation=[[0]*6]*3)
    out = planner.exceedances(dict(available=True, times=times, members=members, model='gfs_seamless'),
                              start, start+timedelta(hours=6), activity('aerial_lift_crane'))
    assert out['members'] == 3 and out['gust_high_probability'] == pytest.approx(round(2/3, 3))
    assert out['gust_limit_kmh'] == 45 and out['gust_p90_kmh'] == 60 and out['restrict_share'] == pytest.approx(round(2/3, 3))
    assert planner.exceedances(None, start, start+timedelta(hours=6), activity('aerial_lift_crane')) is None


def test_windows_without_half_the_forecast_hours_are_skipped():
    start = anchor()
    plan = planner.plan(weather(hourly(start, 3, 10, 20, 0.0)), activity('general_site_work'), 6, hours=12, step_hours=6,
                        now=start.isoformat(), planned_start=start.isoformat())
    assert plan['candidate_count'] == 0 and plan['excluded']['coverage'] == 2 and plan['skipped'] == 2
    assert plan['recommendation']['choice'] == 'insufficient'


def test_work_window_endpoint_reports_a_decision_and_rejects_bad_parameters(client):
    site = main.SITES[0]
    start = (now_utc()+timedelta(days=1)).replace(minute=0, second=0, microsecond=0)
    created = client.post('/api/missions', json=dict(site_id=site['id'], name='Window check', activity='general_site_work',
        latitude=site['latitude'], longitude=site['longitude'], shift_start=start.isoformat(),
        shift_end=(start+timedelta(hours=8)).isoformat(), crew_size=3, supervisor='A. Rivera'), headers=AUTH)
    assert created.status_code == 200
    mid = created.json()['id']
    body = client.get(f'/api/missions/{mid}/work-windows', params=dict(hours=48, step_hours=1, flex_hours=2)).json()
    assert body['duration_hours'] == 8 and body['mission_id'] == mid
    for key in ['windows', 'planned', 'recommendation', 'excluded', 'constraints', 'candidate_count', 'notes', 'ensemble_available']:
        assert key in body
    assert body['recommendation']['choice'] in ('move', 'keep', 'insufficient')
    assert body['constraints']['planned_local_hour'] is not None
    assert all('model' in window for window in body['windows'])
    assert client.get('/api/missions/nope/work-windows').status_code == 404
    assert client.get(f'/api/missions/{mid}/work-windows', params=dict(flex_hours=99)).status_code == 422
    assert client.get(f'/api/missions/{mid}/work-windows', params=dict(step_hours=9)).status_code == 422


def test_malformed_member_timestamps_are_ignored_not_fatal():
    start = anchor()
    times = [(start+timedelta(hours=i)).isoformat() for i in range(6)]
    times[3] = 'not-a-time'
    members = dict(wind_gusts_10m=[[60]*6]*3, apparent_temperature=[[20]*6]*3, precipitation=[[0]*6]*3)
    out = planner.exceedances(dict(available=True, times=times, members=members, model='gfs_seamless'),
                              start, start+timedelta(hours=6), activity('aerial_lift_crane'))
    assert out['members'] == 3 and out['restrict_share'] == 1.0


def test_missing_forecast_is_reported_instead_of_guessed():
    start = anchor()
    plan = planner.plan(dict(forecast=[], summary={}, feed=dict(available=False)), activity('general_site_work'), 8,
                        hours=48, now=start.isoformat(), planned_start=start.isoformat())
    assert plan['windows'] == [] and plan['candidate_count'] == 0
    assert plan['recommendation']['choice'] == 'insufficient' and 'No hourly forecast' in plan['recommendation']['reason']


def test_county_day_model_context_is_disclosed_without_changing_the_advice():
    start = anchor()
    rows = hourly(start, 24, 30, 26, 0.0)
    model = dict(available=True, target_date=(start+timedelta(hours=8)).date().isoformat(),
                 hazards=dict(flood=dict(probability=.2, threshold=.05, elevated=True, warning_supported=True,
                                         model='balanced logistic',
                                         tested=dict(period='2026 H1', precision=.4, recall=.5))))
    arguments = dict(now=start.isoformat(), hours=24, step_hours=1, planned_start=start.isoformat())
    without = planner.plan(weather(rows), activity('general_site_work'), 8, **arguments)
    with_model = planner.plan(weather(rows), activity('general_site_work'), 8, model=model, **arguments)
    assert without['windows'][0]['model'] is None
    attached = [w for w in with_model['windows'] if w['model']]
    assert attached and attached[0]['model']['granularity'].startswith('county-day')
    assert attached[0]['model']['hazards']['flood']['warning_supported'] is True
    assert [(w['start'], w['risk'], w['restricted']) for w in with_model['windows']] == \
           [(w['start'], w['risk'], w['restricted']) for w in without['windows']]
    assert with_model['recommendation'] == without['recommendation']
