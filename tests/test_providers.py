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


# Records from the NWPS river-gauge map service, layers 0 (observed) and 15 (full forecast period), October 17, 2026.
OBSERVED = {'features': [
    {'attributes': {'gaugelid': 'POPW2', 'status': 'no_flooding', 'location': 'Point Pleasant', 'waterbody': 'Ohio River',
                    'obstime': '2026-10-17 03:00:00', 'units': 'ft', 'observed': '24.93', 'latitude': 38.843889, 'longitude': -82.139722}},
    {'attributes': {'gaugelid': 'POMO1', 'status': 'obs_not_current', 'location': 'Pomeroy', 'waterbody': 'Ohio River',
                    'obstime': 'N/A', 'units': '', 'observed': '', 'latitude': 39.027222, 'longitude': -82.031944}}]}
FORECAST = {'features': [
    {'attributes': {'gaugelid': 'POPW2', 'status': 'no_flooding', 'fcsttime': '2026-10-18 00:00:00', 'units': 'ft', 'forecast': '25.1'}},
    {'attributes': {'gaugelid': 'POMO1', 'status': 'fcst_not_current', 'fcsttime': 'N/A', 'units': '', 'forecast': '-999'}}]}


def test_river_gauges_keep_official_categories_and_drop_missing_value_sentinels():
    first, second = providers.parse_gauges(OBSERVED, FORECAST)
    assert first['lid'] == 'POPW2' and first['name'] == 'Ohio River at Point Pleasant'
    assert first['observed'] == dict(category='no_flooding', stage=24.93, unit='ft', valid_time='2026-10-17T03:00:00+00:00')
    assert first['forecast'] == dict(category='no_flooding', stage=25.1, unit='ft', valid_time='2026-10-18T00:00:00+00:00')
    assert second['observed'] == dict(category='obs_not_current', stage=None, unit=None, valid_time=None)
    assert second['forecast'] == dict(category='fcst_not_current', stage=None, unit=None, valid_time=None)


def test_a_gauge_missing_from_the_forecast_layer_has_no_forecast():
    only, = providers.parse_gauges(OBSERVED, {'features': []})[:1]
    assert only['forecast'] == dict(category=None, stage=None, unit=None, valid_time=None)


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
