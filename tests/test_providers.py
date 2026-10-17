import asyncio
from datetime import timedelta
import httpx
from backend import providers
from backend.domain import now_utc
from backend.store import initialize, put_cache, get_cache


def test_failed_refresh_preserves_last_success_and_cools_down(tmp_path, monkeypatch):
    monkeypatch.setenv('PLANIT_DB',str(tmp_path/'providers.sqlite'))
    initialize()
    old=(now_utc()-timedelta(hours=2)).isoformat()
    put_cache('test',dict(data={'original':True},last_success_at=old),old)
    calls=[]
    class FailingClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,*args,**kwargs):
            calls.append(True)
            raise httpx.ReadTimeout('')
    monkeypatch.setattr(providers.httpx,'AsyncClient',FailingClient)
    first=asyncio.run(providers.fetch('test','https://example.test',{}))
    second=asyncio.run(providers.fetch('test','https://example.test',{}))
    assert not first['available'] and first['stale']
    assert first['fetched_at']==second['fetched_at']==old
    assert first['error'].startswith('ReadTimeout:') and second['data']=={'original':True}
    assert len(calls)==1 and get_cache('test')['error']


def test_unmapped_flood_exposure_is_unknown():
    from backend.domain import hazard_index
    result=hazard_index(None,None,[],{},dict(zones=[],feed=dict(available=True)))
    assert next(c for c in result['components'] if c['name']=='Mapped flood exposure')['severity'] is None


NWPS = {'gauges': [
    {'lid': 'POPW2', 'name': 'Ohio River at Point Pleasant', 'latitude': 38.8439, 'longitude': -82.1397,
     'status': {'observed': {'primary': 24.93, 'primaryUnit': 'ft', 'floodCategory': 'no_flooding', 'validTime': '2026-10-17T03:00:00Z'},
                'forecast': {'primary': 25.1, 'primaryUnit': 'ft', 'floodCategory': 'no_flooding', 'validTime': '2026-10-18T00:00:00Z'}}},
    {'lid': 'POMO1', 'name': 'Ohio River at Pomeroy', 'latitude': 39.0272, 'longitude': -82.0319,
     'status': {'observed': {'primary': -999, 'primaryUnit': '', 'floodCategory': 'obs_not_current', 'validTime': '0001-01-01T00:00:00Z'},
                'forecast': {'primary': -999, 'primaryUnit': '', 'floodCategory': 'fcst_not_current', 'validTime': '0001-01-01T00:00:00Z'}}}]}


def test_river_gauges_keep_official_categories_and_drop_missing_value_sentinels():
    first, second = providers.parse_gauges(NWPS)
    assert first['observed'] == dict(category='no_flooding', stage=24.93, unit='ft', valid_time='2026-10-17T03:00:00Z')
    assert first['forecast']['stage'] == 25.1 and first['lid'] == 'POPW2'
    assert second['observed'] == dict(category='obs_not_current', stage=None, unit=None, valid_time=None)


WFIGS = {'features': [
    {'attributes': {'UniqueFireIdentifier': '2026-MNMNS-000412', 'IncidentName': 'Becker', 'IncidentSize': 120.5, 'PercentContained': 40,
                    'FireDiscoveryDateTime': 1792175576000, 'ModifiedOnDateTime_dt': 1792207846047},
     'geometry': {'x': -93.9, 'y': 45.4}},
    {'attributes': {'UniqueFireIdentifier': '2026-XX-1', 'IncidentName': 'No size', 'IncidentSize': None, 'PercentContained': None,
                    'FireDiscoveryDateTime': None, 'ModifiedOnDateTime_dt': None}, 'geometry': None}]}


def test_wildfire_incidents_parse_epoch_times_and_skip_records_without_a_location():
    incidents, complete = providers.parse_wildfires(WFIGS)
    assert complete and len(incidents) == 1
    fire = incidents[0]
    assert fire['id'] == '2026-MNMNS-000412' and fire['acres'] == 120.5 and fire['contained_pct'] == 40
    assert fire['latitude'] == 45.4 and fire['discovered_at'].startswith('2026-10-1') and fire['updated_at'] > fire['discovered_at']
    assert providers.parse_wildfires(dict(WFIGS, exceededTransferLimit=True))[1] is False
