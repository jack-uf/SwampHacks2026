import json
from datetime import timedelta
from backend import main, memory
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
    assert verified['memory_source']['crew'] == 1 and verified['memory_source']['noaa'] == len(memory.site_records(s['id']))
    again = client.get('/api/missions/'+mid).json()
    assert len(again['versions']) == 3
    assert again['checked']['checked_at'] > verified['checked']['checked_at'] > latest['created_at']
    assert again['checked']['assessed_at'] and again['checked']['stale'] is False

    decision = dict(hazard='flood', decision='mitigate', rationale='Pump trench and reroute access before entry.',
                    mitigation='Dewater trench; competent-person inspection.', supervisor='A. Rivera', version=3)
    assert client.post(f'/api/missions/{mid}/decisions', json=decision).status_code == 401
    behind = client.post(f'/api/missions/{mid}/decisions', json=dict(decision, version=2), headers=AUTH)
    assert behind.status_code == 409 and 'version 3' in behind.json()['detail']
    saved = client.post(f'/api/missions/{mid}/decisions', json=decision, headers=AUTH)
    assert saved.status_code == 200, saved.text
    assert saved.json()['version'] == 3
    assert client.post(f'/api/missions/{mid}/decisions', json=dict(decision, decision='override'), headers=AUTH).status_code == 422
    listed = client.get('/api/missions/'+mid).json()['decisions']
    assert listed[0]['supervisor'] == 'A. Rivera' and listed[0]['stale'] is False

    client.patch('/api/reports/'+report['id'], json={'status': 'resolved'}, headers=AUTH)
    after = client.get('/api/missions/'+mid).json()
    assert after['decisions'][0]['stale'] is True and len(after['versions']) == 4


def test_personal_tokens_set_the_recorded_supervisor_and_every_change_is_audited(client, monkeypatch):
    monkeypatch.setenv('PLANIT_SUPERVISOR_TOKENS', 'A. Rivera=tok-rivera, B. Lee=tok-lee')
    rivera = {'Authorization': 'Bearer tok-rivera'}
    s = main.SITES[0]
    assert client.post('/api/missions', json=mission_body(s), headers={'Authorization': 'Bearer tok-nobody'}).status_code == 401
    mid = client.post('/api/missions', json=mission_body(s), headers=rivera).json()['id']
    data = dict(classification='flood', description='DEMO TEST: water over the trench access road.', latitude=s['latitude'], longitude=s['longitude'])
    report = client.post(f"/api/sites/{s['id']}/reports", data=data, headers=rivera).json()
    assert client.patch('/api/reports/'+report['id'], json={'status': 'verified'}, headers=rivera).status_code == 200
    version = client.get('/api/missions/'+mid).json()['versions'][-1]['version']
    body = dict(hazard='flood', decision='accept', rationale='Pumps staged.', supervisor='Somebody Else', version=version)
    named = client.post(f'/api/missions/{mid}/decisions', json=body, headers=rivera).json()
    assert named['supervisor'] == 'A. Rivera' and named['identity'] == 'token'
    shared = client.post(f'/api/missions/{mid}/decisions', json=body, headers=AUTH).json()
    assert shared['supervisor'] == 'Somebody Else' and shared['identity'] == 'self-declared'
    listed = client.get('/api/missions/'+mid).json()['decisions']
    assert sorted((d['supervisor'], d['identity']) for d in listed) == [('A. Rivera', 'token'), ('Somebody Else', 'self-declared')]
    with main.db() as c:
        assert 'identity' not in {r['name'] for r in c.execute('PRAGMA table_info(jha_decisions)')}
    assert client.get('/api/audit').status_code == 401
    log = client.get('/api/audit', headers=AUTH).json()
    actions = [(e['action'], e['actor'], e['identity']) for e in reversed(log)]
    assert actions == [('mission.create', 'A. Rivera', 'token'), ('report.submit', 'A. Rivera', 'token'),
                       ('report.review', 'A. Rivera', 'token'), ('decision.record', 'A. Rivera', 'token'),
                       ('decision.record', 'Somebody Else', 'self-declared')]
    review = next(e for e in log if e['action'] == 'report.review')
    assert review['target'] == report['id'] and review['detail'] == dict(status='verified', previous='pending')
    assert 'water' not in json.dumps(log)


def test_alert_watch_reassesses_only_sites_with_open_missions_when_alerts_change(client, monkeypatch):
    import asyncio
    s = main.SITES[0]
    mid = client.post('/api/missions', json=mission_body(s), headers=AUTH).json()['id']
    asyncio.run(main.watch_alerts())
    assert len(main.versions_for(mid)) == 1
    warning = dict(id='urn:nws:flood-1', event='Flood Warning', severity='Severe', sent=now_utc().isoformat(),
                   expires=(now_utc()+timedelta(days=2)).isoformat())
    checked = []
    async def alerts(site):
        checked.append(site['id'])
        return dict(items=[warning], feed=dict(available=True, fetched_at=now_utc().isoformat()))
    monkeypatch.setattr(main.providers, 'alerts', alerts)
    asyncio.run(main.watch_alerts())
    latest = main.versions_for(mid)[-1]
    assert checked[0] == s['id'] and latest['trigger'] == 'NWS alert set changed'
    assert any(e.get('ref') == 'urn:nws:flood-1' for e in next(e for e in latest['payload']['entries'] if e['hazard'] == 'flood')['evidence'])


def test_concurrent_jha_updates_each_get_their_own_version(client, monkeypatch):
    import threading
    import time
    s = main.SITES[0]
    mid = client.post('/api/missions', json=mission_body(s), headers=AUTH).json()['id']
    mission, assessment = main.mission_or_404(mid), main.cached_assessment(s)
    real_build, real_diff = main.missions.build_jha, main.missions.diff_jha
    def build(m, ctx):
        jha = real_build(m, ctx)
        return dict(jha, fingerprint=threading.current_thread().name)
    def slow_diff(before, after):
        time.sleep(.3)
        return real_diff(before, after)
    monkeypatch.setattr(main.missions, 'build_jha', build)
    monkeypatch.setattr(main.missions, 'diff_jha', slow_diff)
    writers = [threading.Thread(target=main.record_jha, args=(mission, s, assessment, 'race '+n), name=n) for n in 'AB']
    for w in writers:
        w.start()
    for w in writers:
        w.join()
    assert sorted(v['trigger'] for v in main.versions_for(mid)[1:]) == ['race A', 'race B']


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
    override = dict(hazard='lightning', decision='override', rating='low', rationale='Crew wants to continue.', supervisor='B. Lee',
                    version=detail['versions'][-1]['version'])
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
