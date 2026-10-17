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
    async def river_gauges(site):
        return dict(gauges=[],feed=dict(available=True,fetched_at=now_utc().isoformat()))
    async def wildfires():
        return dict(incidents=[],complete=True,feed=dict(available=True,fetched_at=now_utc().isoformat()))
    monkeypatch.setattr(main.providers,'river_gauges',river_gauges)
    monkeypatch.setattr(main.providers,'wildfires',wildfires)
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


def test_portfolio_ranks_partial_sites_by_what_their_unknown_inputs_could_add(client):
    from backend.domain import hazard_index
    full_site,partial_site=main.SITES[0],main.SITES[1]
    weather=dict(summary=dict(apparent_max_c=33,rain_24h_mm=10,gust_max_kmh=40,soil_moisture=.3,rain_72h_mm=20))
    model=dict(flood=dict(probability=.1,in_domain=True),storm=dict(probability=.1,in_domain=True))
    full=hazard_index(weather,[dict(severity='Minor')],[],model,dict(zones=[dict(SFHA_TF='F')],feed=dict(available=True)))
    partial=hazard_index(None,[dict(severity='Moderate')],[],{},None)
    assert full['score']>partial['score'] and partial['score_max']>full['score_max']
    for site,index in [(full_site,full),(partial_site,partial)]:
        put_cache('assessment:'+main.MODEL_REVISION+':'+site['id'],dict(index=index,assessed_at=now_utc().isoformat()),now_utc().isoformat())
    ranked=[s['id'] for s in client.get('/api/sites').json()['sites']]
    assert ranked.index(partial_site['id'])<ranked.index(full_site['id'])


def test_site_detail_rates_nearby_wildfire_and_keeps_only_regional_incidents(client, monkeypatch):
    s=main.SITES[0]
    def incident(id,dlat):
        return dict(id=id,name=id,acres=40.,contained_pct=0,discovered_at=now_utc().isoformat(),updated_at=now_utc().isoformat(),
                    latitude=s['latitude']+dlat,longitude=s['longitude'])
    async def wildfires():
        return dict(incidents=[incident('near',.05),incident('far',8)],complete=True,feed=dict(available=True,fetched_at=now_utc().isoformat()))
    monkeypatch.setattr(main.providers,'wildfires',wildfires)
    detail=client.get('/api/sites/'+s['id']).json()
    assert [i['id'] for i in detail['wildfires']['incidents']]==['near']
    assert detail['nearby']['wildfire'][0]['rating']=='high' and detail['nearby']['wildfire'][0]['ref']=='near'
    assert detail['nearby']['flood'][0]['kind']=='unavailable'


def test_failed_reassessment_serves_the_last_assessment_marked_stale(client, monkeypatch):
    s=main.SITES[0]
    assert client.get('/api/sites/'+s['id']).status_code==200
    key='assessment:'+main.MODEL_REVISION+':'+s['id']
    old=get_cache(key)['payload']
    put_cache(key,dict(old,assessed_at=(now_utc()-timedelta(hours=2)).isoformat()),now_utc().isoformat())
    async def broken(site):
        raise RuntimeError('model artifact unreadable')
    monkeypatch.setattr(main,'predictions',broken)
    response=client.get('/api/sites/'+s['id'])
    assert response.status_code==200
    body=response.json()
    assert body['stale'] is True and 'RuntimeError' in body['reassessment_error']


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


def test_typed_suggestions_survive_correction_without_approving_or_mutating_text(client,monkeypatch):
    from test_jev import payload
    async def classify(text,*,consent):
        assert consent is True
        return main.jev.interpret(payload({'flood':.95,'unstable_terrain':.9,'immediate_review':.5}))
    monkeypatch.setattr(main.jev,'classify',classify)
    s=main.SITES[0]
    text='SYNTHETIC UNIT TEST: Water and sinking ground at the fictional east gate.'
    data=dict(classification='other',description=text,latitude=s['latitude'],longitude=s['longitude'],jev_consent='true')
    response=client.post(f"/api/sites/{s['id']}/reports",data=data,headers=AUTH)
    assert response.status_code==200,response.text
    report=response.json();provider=report['triage']['provider']
    assert provider['review_queue']=='clarification'
    assert provider['routing']['type']=='choice' and provider['urgency']['type']=='score'
    assert report['triage']['manual_categories']==['other'] and report['status']=='pending'
    url='/api/reports/'+report['id']+'/triage'
    assert client.patch(url,json={'categories':['flood','equipment_obstruction']},headers=AUTH).status_code==200
    saved=client.get('/api/sites/'+s['id']).json()['reports'][0]
    assert saved['description']==text and saved['status']=='pending'
    assert saved['triage']['provider']==provider and saved['triage']['source']=='manual_correction'
    assert saved['triage']['manual_categories']==['flood','equipment_obstruction']
