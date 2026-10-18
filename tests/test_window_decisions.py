from datetime import timedelta

from backend import main
from backend.domain import now_utc
from test_api import client, AUTH  # noqa: F401  (shared fixture with stubbed providers)


def wind_then_calm(site, anchor, hours=72):
    """An hourly stub: gusty for the first day, calm afterwards, so a restriction exists."""
    async def weather(requested):
        rows = [dict(time=(anchor+timedelta(hours=index)).isoformat(), apparent_temperature=24.0,
                     wind_gusts_10m=85.0 if index < 24 else 18.0, precipitation=0.0,
                     soil_moisture_0_to_1cm=.2, soil_moisture_0_to_100cm=.25)
                for index in range(hours)]
        return dict(source='test forecast', forecast=rows, summary=dict(rain_72h_mm=0.0),
                    feed=dict(available=True, fetched_at=anchor.isoformat()))
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
