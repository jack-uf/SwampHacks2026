"""Authoritative providers with explicit cache freshness and visible failures."""
import asyncio
import os
from datetime import datetime, timedelta, timezone
import httpx
from math import cos, hypot, isfinite, radians
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


def stamp_zulu(value):
    """Open-Meteo returns naive or suffixed stamps; the rest of the app wants explicit UTC."""
    text = str(value)
    return text if text.endswith('Z') or '+' in text[10:] else text + 'Z'


def hourly_rows(hourly):
    """Per-hour rows; a ragged variable is dropped instead of failing the whole request."""
    times = hourly.get('time') or []
    columns, dropped = {}, []
    for key, value in hourly.items():
        if key == 'time':
            continue
        if not isinstance(value, list) or len(value) != len(times):
            dropped.append(key)
            continue
        columns[key] = value
    rows = [dict(time=stamp_zulu(times[index]), **{key: column[index] for key, column in columns.items()})
            for index in range(len(times))]
    return rows, sorted(dropped)


async def weather(site):
    result = await fetch('weather:'+site['id'], 'https://api.open-meteo.com/v1/forecast', dict(
        latitude=site['latitude'], longitude=site['longitude'], timezone='GMT', past_days=7, forecast_days=3,
        current='temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_gusts_10m',
        hourly='temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_gusts_10m,soil_moisture_0_to_1cm,soil_moisture_0_to_100cm'))
    if not result['available']:
        return dict(source='Open-Meteo numerical weather models', feed=result, summary={}, forecast=[], current=None)
    data = result['data']
    hourly = data['hourly']
    now = now_utc()
    rows, dropped = hourly_rows(hourly)
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
    observed = data.get('current') or {}
    current = dict(observed, time=stamp_zulu(observed['time']), kind='Modeled current conditions') if observed.get('time') else None
    return dict(source='Open-Meteo numerical weather models', feed={k:v for k,v in result.items() if k!='data'},
                summary=summary, current=current, forecast=rows, units=data.get('hourly_units') or {},
                dropped_variables=dropped)


def gauge_trend(readings, flood_stage=None):
    """Rate of change of a gauge series, as the hydrologic signal the JHA never had.

    Readings are (ISO timestamp, stage in metres) pairs, oldest first. Values that do not parse
    are skipped rather than filled; a series with fewer than two usable readings has no trend.
    """
    series = []
    for stamp, stage in readings:
        try:
            series.append((parse_time(stamp), float(stage)))
        except (TypeError, ValueError):
            continue
    series = [(moment, stage) for moment, stage in series if isfinite(stage)]
    series.sort(key=lambda item: item[0])
    if len(series) < 2:
        return dict(available=False, note='Fewer than two usable gauge readings; no trend computed')
    latest_at, latest = series[-1]
    def change(hours):
        cutoff = latest_at - timedelta(hours=hours)
        earlier = [stage for moment, stage in series if moment <= cutoff]
        return None if not earlier else round(latest - earlier[-1], 3)
    change_6h, change_24h = change(6), change(24)
    reference = change_24h if change_24h is not None else change_6h
    direction = 'unknown' if reference is None else 'flat' if abs(reference) < .05 else 'rising' if reference > 0 else 'falling'
    result = dict(available=True, readings=len(series), latest_m=round(latest, 3), latest_at=latest_at.isoformat(),
                  change_6h_m=change_6h, change_24h_m=change_24h, direction=direction,
                  note='Observed gauge stage trend; a rise is not a flood forecast')
    if flood_stage is not None and isfinite(flood_stage):
        result['flood_stage_m'] = round(flood_stage, 3)
        result['headroom_m'] = round(flood_stage - latest, 3)
        result['above_flood_stage'] = latest >= flood_stage
    return result


EPQS = 'https://epqs.nationalmap.gov/v1/json'
ELEVATION_RANGE = (-500., 9000.)  # metres; outside this a sample is a provider error, not terrain


def elevation_value(sample):
    """One EPQS answer as metres, or None when it is missing, unparseable or implausible."""
    try:
        value = float(sample['value'] if isinstance(sample, dict) else sample)
    except (KeyError, TypeError, ValueError):
        return None
    if not isfinite(value) or not ELEVATION_RANGE[0] <= value <= ELEVATION_RANGE[1]:
        return None
    return value


def terrain_summary(samples, source):
    """Relief and gradient across the five-point sample, or an unavailable record."""
    if len(samples) < 5:
        return dict(source=source, available=False, samples={k: round(v, 1) for k, v in samples.items()},
                    missing=[name for name in ['centre', 'north', 'south', 'east', 'west'] if name not in samples],
                    note='Terrain samples incomplete; relief not estimated')
    gradient = hypot(abs(samples['north']-samples['south'])/1000, abs(samples['east']-samples['west'])/1000)
    return dict(source=source, available=True, samples={k: round(v, 1) for k, v in samples.items()},
                elevation_m=round(samples['centre'], 1), relief_m=round(max(samples.values())-min(samples.values()), 1),
                slope_pct=round(100*gradient, 2), missing=[],
                note='Five-point sample across about one kilometre; a screening proxy, not a survey')
ENSEMBLE = 'https://ensemble-api.open-meteo.com/v1/ensemble'
TERRAIN_OFFSET_DEGREES = 0.0045  # roughly 500 m at mid latitudes


async def terrain(site):
    """Five-point 3DEP sample: the facility point plus 500 m north, south, east and west."""
    latitude, longitude = site['latitude'], site['longitude']
    scale = TERRAIN_OFFSET_DEGREES / max(cos(radians(latitude)), .2)
    offsets = dict(centre=(0.,0.), north=(TERRAIN_OFFSET_DEGREES,0.), south=(-TERRAIN_OFFSET_DEGREES,0.),
                   east=(0.,scale), west=(0.,-scale))
    samples, missing = {}, []
    for name, (dy, dx) in offsets.items():
        params = dict(x=round(longitude+dx, 6), y=round(latitude+dy, 6), units='Meters', wkid=4326, includeDate='false')
        result = await fetch('terrain:'+site['id']+':'+name, EPQS, params, ttl=604800)
        value = elevation_value(result['data']) if result['available'] else None
        if value is None:
            missing.append(name)
        else:
            samples[name] = value
    source = 'USGS 3DEP Elevation Point Query Service'
    summary = terrain_summary(samples, source)
    summary['missing'] = sorted(set(summary['missing']) | set(missing))
    return summary


def ensemble_columns(hourly, variables):
    """Member columns per variable; incomplete or non-numeric columns are dropped, not trusted."""
    times = hourly.get('time') or []
    result, dropped = {}, []
    for variable in variables:
        columns = []
        for key, value in hourly.items():
            if key != variable and not key.startswith(variable + '_member'):
                continue
            numeric = isinstance(value, list) and any(isinstance(item, (int, float)) and not isinstance(item, bool)
                                                      for item in value)
            if not numeric or len(value) != len(times):
                dropped.append(key)
                continue
            columns.append(value)
        if len(columns) > 1:
            result[variable] = columns
    return result, sorted(dropped)


async def ensemble(site, days=3):
    """Member spread for the thresholds the deterministic planner screens against."""
    variables = ['wind_gusts_10m','apparent_temperature','precipitation']
    result = await fetch('ensemble:'+site['id'], ENSEMBLE, dict(latitude=site['latitude'], longitude=site['longitude'],
        timezone='GMT', forecast_days=max(2, min(days, 7)), models='gfs_seamless', hourly=','.join(variables)))
    if not result['available']:
        return dict(source='NOAA GFS ensemble via Open-Meteo', feed=result, available=False, times=[], members={})
    hourly = result['data'].get('hourly') or {}
    times = [stamp_zulu(t) for t in hourly.get('time', [])]
    members, dropped = ensemble_columns(hourly, variables)
    if not times or not members:
        return dict(source='NOAA GFS ensemble via Open-Meteo', feed=result, available=False, times=[], members={},
                    dropped_columns=dropped, note='Ensemble response carried no usable member columns')
    return dict(source='NOAA GFS ensemble via Open-Meteo', feed={k:v for k,v in result.items() if k!='data'},
                available=True, times=times, members=members, dropped_columns=dropped,
                member_count=min(len(c) for c in members.values()), model='gfs_seamless',
                note='Weather-model ensemble spread, not a calibrated probability of damage')


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
