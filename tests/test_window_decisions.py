from datetime import timedelta

from backend import main
from backend.domain import now_utc
from backend.domain import parse_time
from test_api import client, AUTH  # noqa: F401  (shared fixture with stubbed providers)


def wind_then_calm(site, anchor, hours=72, first=85.0, second=18.0, fetched=None):
    """An hourly stub: gusty for the first day, calm afterwards, so a restriction exists."""
    async def weather(requested):
        rows = [dict(time=(anchor+timedelta(hours=index)).isoformat(), apparent_temperature=24.0,
                     wind_gusts_10m=first if index < 24 else second, precipitation=0.0,
                     soil_moisture_0_to_1cm=.2, soil_moisture_0_to_100cm=.25)
                for index in range(hours)]
        return dict(source='test forecast', forecast=rows, summary=dict(rain_72h_mm=0.0),
                    feed=dict(available=True, fetched_at=(fetched or anchor).isoformat()))
    return weather


def setup(client, monkeypatch, activity='aerial_lift_crane'):
    site = main.SITES[0]
    anchor = now_utc().replace(minute=0, second=0, microsecond=0)
    created = client.post('/api/missions', json=dict(site_id=site['id'], name='Window record', activity=activity,
        latitude=site['latitude'], longitude=site['longitude'], shift_start=(anchor+timedelta(hours=6)).isoformat(),
        shift_end=(anchor+timedelta(hours=14)).isoformat(), crew_size=3, supervisor='A. Rivera'), headers=AUTH)
    assert created.status_code == 200
    monkeypatch.setattr(main.providers, 'weather', wind_then_calm(site, anchor))
    monkeypatch.setattr(main, 'now_utc', lambda: anchor)
    with main.db() as c:
        c.execute('DELETE FROM cache')  # an expired assessment cache, so the stub's forecast is what the JHA sees
    return created.json()['id']


def test_window_decision_requires_a_current_snapshot_and_a_listed_window(client, monkeypatch):
    mid = setup(client, monkeypatch)
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    assert plan['recommendation']['choice'] == 'move' and plan['decision'] is None and len(plan['snapshot']) == 64
    restricted = [w for w in plan['windows'] if w['restricted']]
    assert restricted and plan['recommendation']['start'] not in [w['start'] for w in restricted]
    body = dict(choice='keep', snapshot=plan['snapshot'], supervisor='A. Rivera')
    assert client.post(f'/api/missions/{mid}/window-decisions', json=body).status_code == 401
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(body, snapshot='0'*64)).status_code == 409
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(body, choice='move')).status_code == 422
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(body, choice='move', start='2099-01-01T00:00:00+00:00')).status_code == 422
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(body, choice='move', start=restricted[0]['start'])).status_code == 409


def test_recorded_window_decision_flows_into_the_plan_detail_and_audit_log(client, monkeypatch):
    mid = setup(client, monkeypatch)
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    recorded = client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                           json=dict(choice='move', start=plan['recommendation']['start'], note='Crew briefed',
                                     snapshot=plan['snapshot'], supervisor='A. Rivera'))
    assert recorded.status_code == 200, recorded.text
    record = recorded.json()
    assert record['choice'] == 'move' and record['start'] == plan['recommendation']['start']
    assert record['note'] == 'Crew briefed' and record['identity'] == 'self-declared' and record['jha_version'] == 1
    after = client.get(f'/api/missions/{mid}/work-windows').json()
    assert after['decision']['id'] == record['id'] and after['decision']['stale'] is False
    detail = client.get(f'/api/missions/{mid}').json()
    assert [d['id'] for d in detail['window_decisions']] == [record['id']]
    log = client.get('/api/audit', headers=AUTH).json()
    rows = [row for row in log if row['action'] == 'window.decide']
    assert rows and rows[0]['target'] == record['id'] and rows[0]['detail']['choice'] == 'move'


def test_keeping_the_plan_is_recorded_without_a_window(client, monkeypatch):
    mid = setup(client, monkeypatch, activity='general_site_work')
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    recorded = client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                           json=dict(choice='keep', snapshot=plan['snapshot'], supervisor='A. Rivera'))
    assert recorded.status_code == 200 and recorded.json()['start'] is None


def test_routine_refresh_keeps_a_decision_and_a_real_change_marks_it_stale(client, monkeypatch):
    site = main.SITES[0]
    anchor = now_utc().replace(minute=0, second=0, microsecond=0)
    mid = setup(client, monkeypatch)
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    recorded = client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                           json=dict(choice='move', start=plan['recommendation']['start'],
                                     snapshot=plan['snapshot'], supervisor='A. Rivera'))
    assert recorded.status_code == 200
    # A later fetch of the same forecast is freshness, not a change of advice.
    monkeypatch.setattr(main.providers, 'weather', wind_then_calm(site, anchor, fetched=anchor+timedelta(minutes=35)))
    refreshed = client.get(f'/api/missions/{mid}/work-windows').json()
    assert refreshed['weather_fetched_at'] != plan['weather_fetched_at']
    assert refreshed['snapshot'] == plan['snapshot'] and refreshed['decision']['stale'] is False
    # Flip the gusty day so the advice itself changes.
    monkeypatch.setattr(main.providers, 'weather', wind_then_calm(site, anchor, first=18.0, second=85.0))
    with main.db() as c:
        c.execute('DELETE FROM cache')
    changed = client.get(f'/api/missions/{mid}/work-windows').json()
    assert changed['snapshot'] != plan['snapshot'] and changed['decision']['stale'] is True


def test_applying_a_move_reschedules_the_mission_and_versions_the_jha(client, monkeypatch):
    mid = setup(client, monkeypatch)
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    start = plan['recommendation']['start']
    assert plan['recommendation']['choice'] == 'move'
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(choice='keep', apply=True, snapshot=plan['snapshot'], supervisor='A. Rivera')).status_code == 422
    applied = client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                          json=dict(choice='move', start=start, apply=True, note='Crew confirmed',
                                    snapshot=plan['snapshot'], supervisor='A. Rivera'))
    assert applied.status_code == 200, applied.text
    body = applied.json()
    assert body['applied']['start'] == start and body['applied']['end'] > start
    assert body['start'] == start and body['note'] == 'Crew confirmed'
    detail = client.get(f'/api/missions/{mid}').json()
    assert detail['mission']['shift_start'] == start and detail['mission']['shift_end'] == body['applied']['end']
    assert detail['versions'][-1]['trigger'].startswith('Shift moved to')
    if body['applied']['jha_version'] is not None:
        assert body['applied']['jha_version'] == detail['versions'][-1]['version']
    assert [d['id'] for d in detail['window_decisions']] == [body['id']]
    log = client.get('/api/audit', headers=AUTH).json()
    actions = [row['action'] for row in log]
    assert 'window.reschedule' in actions and actions.count('window.decide') >= 1
    rescheduled = next(row for row in log if row['action'] == 'window.reschedule')
    assert rescheduled['detail']['new_start'] == start and rescheduled['detail']['previous_start'] != start
    # The old snapshot no longer matches the moved plan.
    assert client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                       json=dict(choice='move', start=start, apply=True, snapshot=plan['snapshot'],
                                 supervisor='A. Rivera')).status_code == 409


def test_applying_a_move_refuses_a_window_that_now_overlaps_another_mission(client, monkeypatch):
    mid = setup(client, monkeypatch)
    plan = client.get(f'/api/missions/{mid}/work-windows').json()
    start = plan['recommendation']['start']
    site = main.SITES[0]
    original = main.prepare_jha

    def interleaved(mission, site_record, assessment):
        # A second supervisor lands a mission between the comparison and the commit.
        with main.db() as c:
            c.execute('INSERT OR IGNORE INTO missions VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                      ('interleaved', site['id'], 'Second crew', 'general_site_work', site['latitude'], site['longitude'],
                       start, (parse_time(start)+timedelta(hours=4)).isoformat(), 2, 'B. Lee', now_utc().isoformat()))
        return original(mission, site_record, assessment)

    monkeypatch.setattr(main, 'prepare_jha', interleaved)
    refused = client.post(f'/api/missions/{mid}/window-decisions', headers=AUTH,
                          json=dict(choice='move', start=start, apply=True, snapshot=plan['snapshot'], supervisor='A. Rivera'))
    assert refused.status_code == 409 and 'overlaps' in refused.json()['detail']
    assert client.get(f'/api/missions/{mid}').json()['mission']['shift_start'] != start
