"""Authoritative providers with explicit cache freshness and visible failures."""
import asyncio
import os
from datetime import datetime, timedelta, timezone
import httpx
from backend.domain import now_utc, parse_time
from backend.store import get_cache, put_cache

LIMIT = asyncio.Semaphore(4)


async def fetch(key, url, params, ttl=1800):
    cached = get_cache(key)
    now = now_utc()
    saved = cached['payload'] if cached else dict(data={}, last_success_at=None)
    if cached and not cached['error'] and now-parse_time(cached['fetched_at']) < timedelta(seconds=ttl):
        return dict(available=True, stale=False, fetched_at=saved['last_success_at'], data=saved['data'], error=None, url=url)
    # Failed fetches are cooled down too: repeated UI polling must not hammer providers.
    if cached and cached['error'] and now-parse_time(cached['fetched_at']) < timedelta(seconds=120):
        return dict(available=False, stale=True, fetched_at=saved['last_success_at'], data=saved['data'], error=cached['error'], url=url)
    try:
        async with LIMIT:
            async with httpx.AsyncClient(timeout=35, follow_redirects=True) as client:
                for attempt in range(3):
                    response = await client.get(url, params=params, headers={
                        'User-Agent': os.getenv('PLANIT_USER_AGENT', 'PlanITControl/0.1 local research prototype'),
                        'Accept': 'application/geo+json, application/json'})
                    if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                        break
                    await asyncio.sleep(2**attempt)
                response.raise_for_status()
                data = response.json()
                if data.get('error'):
                    raise ValueError(str(data['error']))
        put_cache(key, dict(data=data, last_success_at=now.isoformat()), now.isoformat())
        return dict(available=True, stale=False, fetched_at=now.isoformat(), data=data, error=None, url=url)
    except (httpx.HTTPError, ValueError) as exc:
        error = f'{type(exc).__name__}: {exc}'[:400]
        put_cache(key, saved, now.isoformat(), error)
        return dict(available=False, stale=True, fetched_at=saved['last_success_at'],
                    data=saved['data'], error=error, url=url)


async def weather(site):
    result = await fetch('weather:'+site['id'], 'https://api.open-meteo.com/v1/forecast', dict(
        latitude=site['latitude'], longitude=site['longitude'], timezone='GMT', past_days=7, forecast_days=3,
        current='temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_gusts_10m',
        hourly='temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_gusts_10m,soil_moisture_0_to_1cm'))
    if not result['available']:
        return dict(source='Open-Meteo numerical weather models', feed=result, summary={}, forecast=[], current=None)
    data = result['data']
    hourly = data['hourly']
    now = now_utc()
    rows = [dict(time=t+'Z', **{k: v[i] for k,v in hourly.items() if k!='time'}) for i,t in enumerate(hourly['time'])]
    future = [r for r in rows if now <= parse_time(r['time']) < now+timedelta(hours=24)]
    def aggregate(items, key, operation):
        values = [r[key] for r in items if r.get(key) is not None]
        return operation(values) if len(values) == len(items) and values else None
    past72 = [r for r in rows if now-timedelta(hours=72) <= parse_time(r['time']) < now]
    past24 = [r for r in rows if now-timedelta(hours=24) <= parse_time(r['time']) < now]
    past7 = [r for r in rows if now-timedelta(days=7) <= parse_time(r['time']) < now]
    summary = dict(rain_24h_mm=aggregate(future,'precipitation',sum),
        apparent_max_c=aggregate(future,'apparent_temperature',max), gust_max_kmh=aggregate(future,'wind_gusts_10m',max),
        rain_previous_24h_mm=aggregate(past24,'precipitation',sum), rain_72h_mm=aggregate(past72,'precipitation',sum),
        rain_7d_mm=aggregate(past7,'precipitation',sum), soil_moisture=aggregate(future[:1],'soil_moisture_0_to_1cm',max))
    # API 'current' and past hours are model estimates, not site instrument observations.
    current = dict(data['current'], time=data['current']['time']+'Z', kind='Modeled current conditions')
    return dict(source='Open-Meteo numerical weather models', feed={k:v for k,v in result.items() if k!='data'},
                summary=summary, current=current, forecast=rows, units=data['hourly_units'])


async def alerts(site):
    result = await fetch('alerts:'+site['id'], 'https://api.weather.gov/alerts/active',
                         dict(point=f"{site['latitude']},{site['longitude']}"), ttl=300)
    items = []
    if result['available']:
        for feature in result['data'].get('features', []):
            p = feature['properties']
            if p.get('expires') and parse_time(p['expires']) <= now_utc():
                continue
            items.append({k:p.get(k) for k in ['id','event','severity','headline','description','instruction','sent','onset','expires','ends']})
    return dict(source='NOAA / National Weather Service', feed={k:v for k,v in result.items() if k!='data'},
                items=items if result['available'] else None)


# The NWPS JSON API (api.water.noaa.gov) allows 10 requests per 5 minutes, too few for one query per facility.
# The NWPS map service publishes the same gauge statuses without that limit.
NWPS_MAP = 'https://mapservices.weather.noaa.gov/eventdriven/rest/services/water/riv_gauges/MapServer'
WFIGS_CURRENT = ('https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/'
                 'WFIGS_Incident_Locations_Current/FeatureServer/0/query')


def _reading(a, stage_field, time_field):
    """The map service stores values as text: '' or '-999' for no value, 'N/A' for no time, times in UTC."""
    if a is None:
        return dict(category=None, stage=None, unit=None, valid_time=None)
    try:
        stage = float(a.get(stage_field))
    except (TypeError, ValueError):
        stage = None
    time = a.get(time_field) or ''
    return dict(category=a.get('status') or None, stage=None if stage is None or stage <= -999 else stage,
                unit=a.get('units') or None,
                valid_time=time.replace(' ', 'T')+'+00:00' if time[:4].isdigit() else None)


def parse_gauges(observed, forecast):
    forecasts = {f['attributes']['gaugelid']: f['attributes'] for f in forecast.get('features', [])}
    gauges = []
    for f in observed.get('features', []):
        a = f['attributes']
        if a.get('latitude') is None or a.get('longitude') is None:
            continue
        name = ' at '.join(p for p in (a.get('waterbody'), a.get('location')) if p)
        gauges.append(dict(lid=a['gaugelid'], name=name or a['gaugelid'], latitude=a['latitude'], longitude=a['longitude'],
                           observed=_reading(a, 'observed', 'obstime'),
                           forecast=_reading(forecasts.get(a['gaugelid']), 'forecast', 'fcsttime')))
    return gauges


async def river_gauges(site):
    """NWS river gauges in a box about 70 miles around the facility, so missions up to 50 miles away are covered."""
    lat, lon = site['latitude'], site['longitude']
    box = dict(geometry=f'{lon-1.3},{lat-1},{lon+1.3},{lat+1}', geometryType='esriGeometryEnvelope', inSR=4326,
               spatialRel='esriSpatialRelIntersects', returnGeometry='false', f='json')
    observed = await fetch('nwps-map:observed:'+site['id'], NWPS_MAP+'/0/query', dict(
        box, outFields='gaugelid,status,location,waterbody,obstime,units,observed,latitude,longitude'), ttl=900)
    forecast = await fetch('nwps-map:forecast:'+site['id'], NWPS_MAP+'/15/query', dict(
        box, outFields='gaugelid,status,fcsttime,units,forecast'), ttl=900)
    complete = observed['available'] and not observed['data'].get('exceededTransferLimit')
    # A failed forecast query leaves gauges without forecast evidence; observations still count.
    forecast_data = forecast['data'] if forecast['available'] else {}
    return dict(source='NOAA National Water Prediction Service river gauges', feed={k: v for k, v in observed.items() if k != 'data'},
                forecast_feed={k: v for k, v in forecast.items() if k != 'data'},
                gauges=parse_gauges(observed['data'], forecast_data) if complete else None)


def _epoch(value):
    return None if value is None else datetime.fromtimestamp(value/1000, timezone.utc).isoformat()


def parse_wildfires(data):
    """Active WFIGS wildfire incidents. A response cut at the record limit cannot show that no fire is nearby."""
    incidents = []
    for f in data.get('features', []):
        a, g = f.get('attributes') or {}, f.get('geometry')
        if not g or g.get('x') is None or g.get('y') is None:
            continue
        incidents.append(dict(id=a.get('UniqueFireIdentifier'), name=a.get('IncidentName'), acres=a.get('IncidentSize'),
                              contained_pct=a.get('PercentContained'), discovered_at=_epoch(a.get('FireDiscoveryDateTime')),
                              updated_at=_epoch(a.get('ModifiedOnDateTime_dt')), latitude=g['y'], longitude=g['x']))
    return incidents, not data.get('exceededTransferLimit')


async def wildfires():
    result = await fetch('wfigs:current', WFIGS_CURRENT, dict(
        where="IncidentTypeCategory='WF' AND FireOutDateTime IS NULL", returnGeometry='true', outSR=4326, f='json',
        outFields='UniqueFireIdentifier,IncidentName,IncidentSize,PercentContained,FireDiscoveryDateTime,ModifiedOnDateTime_dt'), ttl=900)
    incidents, complete = parse_wildfires(result['data']) if result['available'] else (None, False)
    return dict(source='NIFC WFIGS current wildfire incidents', feed={k: v for k, v in result.items() if k != 'data'},
                incidents=incidents, complete=complete)


async def floodplain(site):
    result = await fetch('fema:'+site['id'], 'https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query',
        dict(f='json', geometry=f"{site['longitude']},{site['latitude']}", geometryType='esriGeometryPoint',
             inSR=4326, spatialRel='esriSpatialRelIntersects', outFields='FLD_ZONE,SFHA_TF', returnGeometry='false'), ttl=604800)
    features = result['data'].get('features', []) if result['available'] else []
    return dict(source='FEMA National Flood Hazard Layer', feed={k:v for k,v in result.items() if k!='data'},
                zones=[f['attributes'] for f in features],
                note='No intersecting polygon means unmapped/unknown, not no flood risk. Exposure is static, not a forecast.')
