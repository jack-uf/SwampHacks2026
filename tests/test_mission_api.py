from datetime import timedelta
from backend import main
from backend.domain import now_utc
from test_api import client, AUTH  # noqa: F401  (shared fixture with stubbed providers)


def tomorrow(hour):
    return (now_utc()+timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def mission_body(site, **overrides):
    return dict(dict(site_id=site['id'], name='DEMO trench repair', activity='excavation_trenching', latitude=site['latitude'],
                     longitude=site['longitude'], shift_start=tomorrow(8), shift_end=tomorrow(16), crew_size=4, supervisor='A. Rivera'), **overrides)


def test_mission_validation_and_supervisor_only(client):
    s = main.SITES[0]
    assert client.post('/api/missions', json=mission_body(s)).status_code == 401
    assert client.post('/api/missions', json=mission_body(s, latitude=0), headers=AUTH).status_code == 422
    assert client.post('/api/missions', json=mission_body(s, shift_end=tomorrow(7)), headers=AUTH).status_code == 422
    assert client.post('/api/missions', json=mission_body(s, activity='skydiving'), headers=AUTH).status_code == 422
    assert client.post('/api/missions', json=mission_body(s, site_id='nope'), headers=AUTH).status_code == 404
    assert {a['id'] for a in client.get('/api/activities').json()} >= {'excavation_trenching', 'aerial_lift_crane'}


def test_crew_report_lifecycle_versions_the_mission_jha_with_attribution(client):
    s = main.SITES[0]
    created = client.post('/api/missions', json=mission_body(s), headers=AUTH)
    assert created.status_code == 200, created.text
    mid = created.json()['id']
    first = client.get('/api/missions/'+mid).json()
    assert [v['version'] for v in first['versions']] == [1] and first['versions'][0]['trigger'] == 'Mission created'
    assert first['conditions']['hours_available'] == 1 and first['conditions']['kind'] == 'predicted'
    flood = next(e for e in first['jha']['entries'] if e['hazard'] == 'flood')
    assert flood['rating'] == 'elevated'
    assert isinstance(first['memory'], list) and isinstance(first['planned_nearby'], list)
    assert [m['id'] for m in client.get('/api/missions', params=dict(site_id=s['id'])).json()] == [mid]

    data = dict(classification='flood', description='DEMO TEST: water over the trench access road.', latitude=s['latitude'], longitude=s['longitude'])
    report = client.post(f"/api/sites/{s['id']}/reports", data=data, headers=AUTH).json()
    pending = client.get('/api/missions/'+mid).json()
    assert len(pending['versions']) == 2 and 'submitted' in pending['versions'][-1]['trigger']
    assert pending['versions'][-1]['diff'][0]['change'] == 'evidence'

    assert client.patch('/api/reports/'+report['id'], json={'status': 'verified'}, headers=AUTH).status_code == 200
    verified = client.get('/api/missions/'+mid).json()
    latest = verified['versions'][-1]
    assert latest['version'] == 3 and report['id'][:8] in latest['trigger'] and 'verified' in latest['trigger']
    assert latest['diff'][0] == dict(hazard='flood', change='rating', before='elevated', after='high',
                                     evidence_added=['Crew report verified by supervisor: DEMO TEST: water over the trench access road.'],
                                     evidence_removed=['Crew report (pending supervisor review): DEMO TEST: water over the trench access road.'])
    cited = [e for e in next(e for e in verified['jha']['entries'] if e['hazard'] == 'flood')['evidence'] if e.get('ref') == report['id']]
    assert cited[0]['kind'] == 'observed'
    assert len(client.get('/api/missions/'+mid).json()['versions']) == 3

    decision = dict(hazard='flood', decision='mitigate', rationale='Pump trench and reroute access before entry.',
                    mitigation='Dewater trench; competent-person inspection.', supervisor='A. Rivera')
    assert client.post(f'/api/missions/{mid}/decisions', json=decision).status_code == 401
    saved = client.post(f'/api/missions/{mid}/decisions', json=decision, headers=AUTH)
    assert saved.status_code == 200, saved.text
    assert saved.json()['version'] == 3
    assert client.post(f'/api/missions/{mid}/decisions', json=dict(decision, decision='override'), headers=AUTH).status_code == 422
    listed = client.get('/api/missions/'+mid).json()['decisions']
    assert listed[0]['supervisor'] == 'A. Rivera' and listed[0]['stale'] is False

    client.patch('/api/reports/'+report['id'], json={'status': 'resolved'}, headers=AUTH)
    after = client.get('/api/missions/'+mid).json()
    assert after['decisions'][0]['stale'] is True and len(after['versions']) == 4


def test_lightning_stop_work_cannot_be_overridden_and_replay_is_simulated(client):
    s = main.SITES[0]
    now = now_utc()
    body = mission_body(s, activity='aerial_lift_crane', shift_start=(now-timedelta(minutes=30)).isoformat(),
                        shift_end=(now+timedelta(hours=4)).isoformat())
    mid = client.post('/api/missions', json=body, headers=AUTH).json()['id']
    client.post(f"/api/sites/{s['id']}/lightning-replay", headers=AUTH)
    detail = client.get('/api/missions/'+mid).json()
    lightning = next(e for e in detail['jha']['entries'] if e['hazard'] == 'lightning')
    assert lightning['rating'] == 'stop_work' and lightning['simulated']
    assert detail['jha']['readiness']['status'] == 'simulated_stop_work'
    assert 'replay' in detail['versions'][-1]['trigger'].lower()
    override = dict(hazard='lightning', decision='override', rating='low', rationale='Crew wants to continue.', supervisor='B. Lee')
    response = client.post(f'/api/missions/{mid}/decisions', json=override, headers=AUTH)
    assert response.status_code == 409 and 'deterministic' in response.json()['detail']


def test_plan_overlaps_compare_entities_with_factors(client):
    result = client.get('/api/plan-overlaps', params=dict(utilities_only='true', limit=5)).json()
    assert result['source']['url'].startswith('https://www.eia.gov/') and result['total'] >= len(result['pairs']) > 0
    pair = result['pairs'][0]
    assert pair['a']['entity_id'] != pair['b']['entity_id'] and pair['a']['sector'] == pair['b']['sector'] == 'Electric Utility'
    assert set(pair['factors']) >= {'proximity', 'timing'}
    excluded = {p['plant_name'] for p in result['suspect_coordinates']}
    assert 'Mullin Creek #1 Generating Station' in excluded
    assert not any(excluded & {p['a']['plant_name'], p['b']['plant_name']} for p in result['pairs'])
