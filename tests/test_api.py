import io
from datetime import timedelta
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from backend import main
from backend.domain import now_utc
from backend.store import get_cache, put_cache, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('PLANIT_DB',str(tmp_path/'test.sqlite'))
    monkeypatch.setenv('PLANIT_SCHEDULER','false')
    monkeypatch.setenv('PLANIT_ADMIN_TOKEN','test-supervisor')
    monkeypatch.setenv('PLANIT_LIGHTNING_TOKEN','test-strike-provider')
    monkeypatch.setattr(main,'UPLOADS',tmp_path)
    async def weather(site):
        tomorrow=(now_utc()+timedelta(days=1)).replace(hour=12,minute=0,second=0,microsecond=0)
        return dict(summary=dict(apparent_max_c=35,rain_24h_mm=20,gust_max_kmh=40,soil_moisture=.3),
                    forecast=[dict(time=tomorrow.isoformat(),apparent_temperature=35,wind_gusts_10m=40,precipitation=20)],
                    feed=dict(available=True,fetched_at=now_utc().isoformat()))
    async def alerts(site):
        return dict(items=[],feed=dict(available=True,fetched_at=now_utc().isoformat()))
    async def predictions(site):
        return dict(available=False,hazards={},reason='Explicit test provider; no model substitution')
    async def floodplain(site):
        return dict(zones=[],feed=dict(available=False),note='Test unavailable')
    monkeypatch.setattr(main.providers,'weather',weather)
    monkeypatch.setattr(main.providers,'alerts',alerts)
    monkeypatch.setattr(main.providers,'floodplain',floodplain)
    monkeypatch.setattr(main,'predictions',predictions)
    with TestClient(main.app) as c:
        yield c


AUTH={'Authorization':'Bearer test-supervisor'}


def test_inventory_and_trust_boundaries(client):
    sites=client.get('/api/sites').json()['sites']
    assert len(sites)>=50 and len({s['id'] for s in sites})==len(sites)
    assert all(-125<s['longitude']<-66 and 24<s['latitude']<50 for s in sites)
    assert client.get('/api/sites/unknown').status_code==404
    assert client.post('/api/refresh').status_code==401
    assert client.post('/api/sites/'+sites[0]['id']+'/lightning-replay').status_code==401


def test_report_photo_review_and_persistence(client):
    s=main.SITES[0]
    data=dict(classification='flood',description='Water pooling across the equipment access route.',latitude=s['latitude'],longitude=s['longitude'])
    image=io.BytesIO();Image.new('RGB',(20,20),'blue').save(image,format='PNG')
    response=client.post(f"/api/sites/{s['id']}/reports",data=data,files={'image':('photo.png',image.getvalue(),'image/png')},headers=AUTH)
    assert response.status_code==200,response.text
    report=response.json()
    assert report['status']=='pending' and report['image'].endswith('.jpg')
    detail=client.get('/api/sites/'+s['id']).json()
    assert len(detail['reports'])==1 and detail['reports'][0]['latitude']==s['latitude']
    assert next(c for c in detail['index']['components'] if c['name']=='Reviewed crew reports')['points']==0
    assert client.patch('/api/reports/'+report['id'],json={'status':'verified'},headers=AUTH).status_code==200
    detail=client.get('/api/sites/'+s['id']).json()
    assert detail['reports'][0]['status']=='verified'
    assert next(c for c in detail['index']['components'] if c['name']=='Reviewed crew reports')['points']>0
    assert client.patch('/api/reports/'+report['id'],json={'status':'resolved'},headers=AUTH).status_code==200
    assert client.post(f"/api/sites/{s['id']}/reports",data={**data,'latitude':0},headers=AUTH).status_code==422
    assert client.post(f"/api/sites/{s['id']}/reports",data=data,files={'image':('bad.png',b'not an image','image/png')},headers=AUTH).status_code==422


def test_replay_separate_from_live_and_delayed_events(client):
    site=main.SITES[0]
    url=f"/api/sites/{site['id']}/lightning-replay"
    first=client.post(url,headers=AUTH).json()
    second=client.post(url,headers=AUTH).json()
    assert second['last_strike_at']>=first['last_strike_at'] and second['remaining_seconds']>=1798
    detail=client.get('/api/sites/'+site['id']).json()
    assert detail['lightning']['replay']['active'] and not detail['lightning']['live']['active']
    body=dict(id='strike-a',latitude=site['latitude'],longitude=site['longitude'],occurred_at=(now_utc()-timedelta(minutes=5)).isoformat())
    assert client.post('/api/lightning/strikes',json=body).status_code==401
    provider={'Authorization':'Bearer test-strike-provider'}
    assert client.post('/api/lightning/strikes',json=body,headers=provider).status_code==200
    assert client.post('/api/lightning/strikes',json=body,headers=provider).status_code==200
    older={**body,'id':'strike-old','occurred_at':(now_utc()-timedelta(minutes=20)).isoformat()}
    client.post('/api/lightning/strikes',json=older,headers=provider)
    detail=client.get('/api/sites/'+site['id']).json()
    assert 1495<detail['lightning']['live']['remaining_seconds']<=1500
    future={**body,'id':'future','occurred_at':(now_utc()+timedelta(minutes=1)).isoformat()}
    assert client.post('/api/lightning/strikes',json=future,headers=provider).status_code==422
    with db() as c:
        assert c.execute("SELECT count(*) FROM strikes WHERE source='live'").fetchone()[0]==2


def test_briefing_history_is_persistent_and_idempotent(client):
    site=main.SITES[0]
    url=f"/api/sites/{site['id']}/briefings"
    first=client.post(url,headers=AUTH)
    assert first.status_code==200,first.text
    second=client.post(url,headers=AUTH).json()
    assert first.json()['id']==second['id']
    saved=client.get(url).json()
    assert len(saved)==1 and saved[0]['payload']['timezone']==site['timezone']
    assert saved[0]['delivery']=='not configured'


def test_bad_description_and_invalid_review(client):
    s=main.SITES[0]
    response=client.post(f"/api/sites/{s['id']}/reports",data=dict(classification='other',description='short',latitude=s['latitude'],longitude=s['longitude']),headers=AUTH)
    assert response.status_code==422
    assert client.patch('/api/reports/missing',json={'status':'verified'},headers=AUTH).status_code==404
    assert client.patch('/api/reports/missing',json={'status':'bogus'},headers=AUTH).status_code==422


def test_scheduler_local_due_and_same_day_idempotency(client, monkeypatch):
    import asyncio
    from datetime import datetime, timezone
    site={**main.SITES[0], 'timezone':'America/New_York'}
    monkeypatch.setattr(main,'SITES',[site])
    asyncio.run(main.tick(datetime(2026,10,16,21,59,tzinfo=timezone.utc)))
    assert client.get(f"/api/sites/{site['id']}/briefings").json()==[]
    asyncio.run(main.tick(datetime(2026,10,16,22,0,tzinfo=timezone.utc)))
    asyncio.run(main.tick(datetime(2026,10,16,22,1,tzinfo=timezone.utc)))
    saved=client.get(f"/api/sites/{site['id']}/briefings").json()
    assert len(saved)==1 and saved[0]['local_date']=='2026-10-17'


def test_multi_hazard_correction_provider_failure_retains_report(client,monkeypatch):
    async def unavailable(text,*,consent):
        assert consent is True
        return dict(status='unavailable',source='manual',reason='Fixture provider outage')
    monkeypatch.setattr(main.jev,'classify',unavailable)
    s=main.SITES[0]
    data=dict(classification='wildlife',categories='["unstable_terrain","equipment_obstruction"]',jev_consent='true',
              description='HYPOTHETICAL TEST: Gator and saturated east access road; trucks stuck.',latitude=s['latitude'],longitude=s['longitude'])
    response=client.post(f"/api/sites/{s['id']}/reports",data=data,headers=AUTH)
    assert response.status_code==200,response.text
    report=response.json();assert report['status']=='pending'
    assert report['triage']['manual_categories']==['wildlife','unstable_terrain','equipment_obstruction']
    url='/api/reports/'+report['id']+'/triage'
    assert client.patch(url,json={'categories':['flood']}).status_code==401
    assert client.patch(url,json={'categories':[]},headers=AUTH).status_code==422
    assert client.patch(url,json={'categories':['unknown']},headers=AUTH).status_code==422
    result=client.patch(url,json={'categories':['flood','unstable_terrain']},headers=AUTH)
    assert result.status_code==200
    detail=client.get('/api/sites/'+s['id']).json();r=detail['reports'][0]
    assert r['classification']=='flood' and r['status']=='pending'
    assert r['triage']['provider']['reason']=='Fixture provider outage'
    assert r['triage']['manual_categories']==['flood','unstable_terrain']
    with db() as c:
        assert c.execute('SELECT count(*) FROM report_triage').fetchone()[0]==1
    assert client.post(f"/api/sites/{s['id']}/reports",data={**data,'categories':'{}'},headers=AUTH).status_code==422
