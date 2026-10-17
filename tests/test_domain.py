from datetime import datetime, timedelta, timezone
from backend.domain import distance_miles, lightning_state, briefing_due, hazard_index

NOW = datetime(2026,10,16,22,tzinfo=timezone.utc)
SITE = dict(latitude=29.65,longitude=-82.32,timezone='America/New_York')


def strike(minutes, latitude=29.65):
    return dict(latitude=latitude,longitude=-82.32,occurred_at=(NOW-timedelta(minutes=minutes)).isoformat())


def test_lightning_distance_reset_and_expiry():
    assert distance_miles(0,0,0,0)==0
    assert 69 < distance_miles(0,0,1,0) < 70
    assert lightning_state([strike(20)],SITE,NOW)['remaining_seconds']==600
    assert lightning_state([strike(20),strike(3)],SITE,NOW)['remaining_seconds']==1620
    assert not lightning_state([strike(30)],SITE,NOW)['active']
    assert not lightning_state([strike(1,latitude=30.65)],SITE,NOW)['active']
    assert not lightning_state([strike(-1)],SITE,NOW)['active']


def test_lightning_ten_mile_boundary():
    offset=10/3958.7613*180/3.141592653589793
    site=dict(latitude=0,longitude=0)
    s=dict(latitude=offset,longitude=0,occurred_at=NOW.isoformat())
    assert lightning_state([s],site,NOW)['active']
    s['latitude']+=.00001
    assert not lightning_state([s],site,NOW)['active']


def test_briefing_local_time_and_dst():
    assert briefing_due(SITE,NOW)==(True,'2026-10-16','2026-10-17')
    assert not briefing_due(SITE,NOW-timedelta(minutes=1))[0]
    winter=datetime(2026,12,16,23,tzinfo=timezone.utc)
    assert briefing_due(SITE,winter)==(True,'2026-12-16','2026-12-17')
    assert not briefing_due(SITE,winter-timedelta(minutes=1))[0]
    az=dict(timezone='America/Phoenix')
    assert briefing_due(az,datetime(2026,7,17,1,tzinfo=timezone.utc))==(True,'2026-07-16','2026-07-17')


def test_unknown_is_not_zero_and_never_renormalized():
    index=hazard_index(None,None,[],{},None)
    assert index['coverage']==.05 and index['score']==0 and not index['complete']
    assert index['components'][0]['severity'] is None
    assert index['components'][-1]['severity']==0
    weather=dict(summary=dict(apparent_max_c=45,rain_24h_mm=150,gust_max_kmh=100,soil_moisture=.5,rain_72h_mm=150))
    index=hazard_index(weather,[dict(severity='Extreme')],[dict(status='verified')]*3,
                       dict(flood=dict(probability=1,in_domain=True),storm=dict(probability=.5,in_domain=True)),dict(zones=[dict(SFHA_TF='T')],feed=dict(available=True)))
    assert index['score']==10 and index['complete']
    outside=hazard_index(weather,[],[],dict(flood=dict(probability=1,in_domain=False)),None)
    assert next(c for c in outside['components'] if c['name']=='County event model')['points'] is None


def test_one_supported_hazard_cannot_complete_combined_model_component():
    for hazard in ('flood','storm'):
        result=hazard_index(None,None,[],{hazard:dict(probability=.9,in_domain=True)},None)
        assert result['coverage']==.05
        assert next(c for c in result['components'] if c['name']=='County event model')['points'] is None


def test_incomplete_rain_inputs_reduce_coverage():
    summary = dict(rain_24h_mm=10, soil_moisture=.3, rain_72h_mm=30)
    for missing in summary:
        partial = dict(summary, **{missing: None})
        result = hazard_index(dict(summary=partial), None, [], {}, None)
        rain = next(c for c in result['components'] if c['name']=='Rain & saturation')
        assert rain['points'] is None and result['coverage'] == .05


def test_partial_index_reports_how_high_its_unknown_components_could_take_it():
    partial=hazard_index(None,[],[],{},None)
    assert partial['score']==0 and partial['coverage']==.25
    assert partial['score_max']==7.5
    assert partial['missing']==['Heat screening','Rain & saturation','Wind & storms','County event model','Mapped flood exposure']
    weather=dict(summary=dict(apparent_max_c=45,rain_24h_mm=150,gust_max_kmh=100,soil_moisture=.5,rain_72h_mm=150))
    full=hazard_index(weather,[dict(severity='Extreme')],[dict(status='verified')]*3,
                      dict(flood=dict(probability=1,in_domain=True),storm=dict(probability=.5,in_domain=True)),dict(zones=[dict(SFHA_TF='T')],feed=dict(available=True)))
    assert full['score_max']==full['score']==10 and full['missing']==[]
