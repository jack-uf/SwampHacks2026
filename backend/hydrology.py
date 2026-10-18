"""Live river context and isolated, fail-closed high-flow research inference."""
import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import timedelta
from functools import lru_cache

import httpx
import joblib
import numpy as np
import pandas as pd

from backend import providers
from backend.domain import distance_miles, now_utc, parse_time
from backend.store import db, get_cache, put_cache
from scripts.common import ROOT

NLDI = 'https://api.water.usgs.gov/nldi/linked-data'
USGS = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections'
MANIFEST = ROOT / 'data/hydrology-model.json'
MAX_AGE = timedelta(hours=2)
TTL = timedelta(minutes=15)
LOCK = asyncio.Lock()
NOTE = ('Gauge discharge describes a river, not water depth at this work location. '
        'Check drainage, access routes and NWS instructions with the supervisor. '
        'Experimental ML cannot authorize work or change hazard ratings or window rankings.')


def point_key(point):
    return f"hydrology:{point['latitude']:.6f}:{point['longitude']:.6f}"


def cached(point):
    record = get_cache(point_key(point))
    if not record:
        return None
    result = record['payload']
    stale = now_utc() - parse_time(result['checked_at']) > TTL
    if stale:
        return dict(result, stale=True, status='unavailable',
            summary='River context expired; check live data again. Unknown is not low risk.')
    gauges = []
    for gauge in result['gauges']:
        observation = dict(gauge['observation'])
        if observation.get('latest_at'):
            age = (now_utc()-parse_time(observation['latest_at'])).total_seconds()/60
            observation.update(age_minutes=round(age, 1), available=0 <= age <= MAX_AGE.total_seconds()/60)
        experimental = dict(gauge['experimental'])
        if experimental.get('valid_until') and parse_time(experimental['valid_until']) <= now_utc():
            experimental.update(status='withheld', probability=None, baseline_probability=None,
                                reasons=experimental['reasons']+['forecast_horizon_expired'])
        gauges.append(dict(gauge, observation=observation, experimental=experimental))
    available = any(g['observation']['available'] for g in gauges)
    return dict(result, stale=False, gauges=gauges, status='observed' if available else 'unavailable',
                summary=result['summary'] if available else 'No fresh discharge remains in this river context. Check live data again.')


@lru_cache(maxsize=1)
def manifest():
    data = json.loads(MANIFEST.read_text())
    if data['operationally_promoted'] is not False:
        raise ValueError('This integration accepts an unpromoted research reference only')
    return data


@lru_cache(maxsize=2)
def artifact(name):
    record = manifest()[name]
    path = ROOT / record['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Hydrology artifact checksum mismatch; restore the frozen research artifact')
    return joblib.load(path)


def parse_discharge(payload, station, received_at):
    """Keep provisional observations visible; only approved, unqualified values enter ML."""
    if not isinstance(payload, dict) or not isinstance(payload.get('features'), list):
        raise ValueError('USGS response lacks a feature collection')
    if any(link.get('rel') == 'next' for link in payload.get('links', [])):
        raise ValueError('USGS response is paginated; history is incomplete')
    rows, series, times = [], set(), set()
    for feature in payload['features']:
        p = feature['properties']
        if p['monitoring_location_id'] != 'USGS-' + station or p['parameter_code'] != '00060' or \
                p['statistic_id'] != '00011' or p['unit_of_measure'] != 'ft^3/s':
            raise ValueError('Discharge station, parameter, statistic or unit mismatch')
        series.add(p['time_series_id'])
        t = parse_time(p['time'])
        if len(series) != 1 or t in times:
            raise ValueError('Ambiguous discharge series or duplicate timestamp')
        if t > received_at:
            raise ValueError('Future-dated observation')
        times.add(t)
        try:
            value = float(p['value'])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(value) or value < 0:
            continue
        quality = p['approval_status']
        if quality not in ['Approved', 'Provisional']:
            continue
        rows.append(dict(time=t.isoformat(), value_cfs=value, quality=quality,
                         qualified=bool(p.get('qualifier'))))
    return sorted(rows, key=lambda x: x['time'])


def observation_summary(rows, received_at, now):
    usable = [r for r in rows if not r['qualified']]
    if not usable:
        return dict(available=False, reason='No unqualified discharge readings returned')
    latest = usable[-1]
    age = (now - parse_time(latest['time'])).total_seconds() / 60
    def change(hours):
        cutoff = parse_time(latest['time']) - timedelta(hours=hours)
        earlier = [r for r in usable if parse_time(r['time']) <= cutoff]
        if not earlier or cutoff - parse_time(earlier[-1]['time']) > timedelta(hours=1):
            return None
        return round(latest['value_cfs'] - earlier[-1]['value_cfs'], 3)
    return dict(available=0 <= age <= MAX_AGE.total_seconds()/60, latest_at=latest['time'],
        value_cfs=latest['value_cfs'], quality=latest['quality'], age_minutes=round(age, 1),
        received_at=received_at.isoformat(), observation_to_receipt_minutes=round(
            (received_at-parse_time(latest['time'])).total_seconds()/60, 1),
        latency_note='Observation age at retrieval, not first-publication or transmission delay.',
        change_6h_cfs=change(6), change_24h_cfs=change(24),
        provisional_readings=sum(r['quality'] == 'Provisional' for r in rows),
        approved_readings=sum(r['quality'] == 'Approved' for r in rows),
        qualified_readings=sum(r['qualified'] for r in rows), readings=len(rows))


def feature_row(rows, station, issue):
    """Reproduce frozen causal inputs, without future values or eligibility from outcomes."""
    accepted = [r for r in rows if r['quality'] == 'Approved' and not r['qualified']]
    if not accepted:
        raise ValueError('approved_history_missing')
    series = pd.Series([r['value_cfs'] for r in accepted],
                       index=pd.to_datetime([r['time'] for r in accepted], utc=True))
    q = series.resample('h', closed='right', label='right').last()
    cutoff = pd.Timestamp(issue) - pd.Timedelta(hours=2)
    history = q.reindex(pd.date_range(cutoff-pd.Timedelta(hours=168), cutoff, freq='h'))
    if history.isna().any():
        raise ValueError('approved_169_hour_history_incomplete')
    reference = station['reference_cfs']
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('invalid_frozen_reference')
    if history.iloc[-24:].gt(reference).any():
        raise ValueError('not_quiet_for_24_hours')
    x = np.log1p(history / reference)
    row = {f'q_{lag}': float(x.iloc[-(lag-2)-1]) for lag in [2, 8, 14, 26, 50, 74, 170]}
    for hours in [6, 12, 24, 48, 72, 168]:
        row[f'rise_{hours}'] = row['q_2'] - row[f'q_{hours+2}']
    row['acceleration'] = row['rise_6'] - (row['q_8'] - row['q_14'])
    for hours in [24, 72]:
        row[f'range_{hours}'] = float(x.iloc[-hours:].max() - x.iloc[-hours:].min())
    phase = 2*np.pi*pd.Timestamp(issue).dayofyear/365.25
    row.update(season_sin=float(np.sin(phase)), season_cos=float(np.cos(phase)),
        latitude=station['latitude'], longitude=station['longitude'],
        log_area=float(np.log1p(station['area_km2'])), elevation_m=station['elevation_m'])
    return pd.DataFrame([row])


def shadow(station_id, receipt, issue):
    """Score a gauge only from a full response actually received by a scheduled issue."""
    result = dict(status='withheld', operational_status='withheld', probability=None, baseline_probability=None,
        reasons=['model_not_promoted', 'site_flood_applicability_unverified'], issue_time=issue.isoformat(),
        valid_until=(issue+timedelta(hours=6)).isoformat(), threshold=manifest()['candidate']['threshold'])
    station = manifest()['stations'].get(station_id)
    if station is None:
        result['reasons'].append('gauge_has_no_frozen_reference_and_attributes')
    if receipt is None:
        result['reasons'].append('no_response_received_by_issue')
    if issue.hour % 6 or issue.minute or issue.second or issue.microsecond:
        result['reasons'].append('outside_frozen_issue_cadence')
    if issue > now_utc():
        result['reasons'].append('issue_not_reached')
    if receipt and (parse_time(receipt['received_at']) > issue or
                    issue - parse_time(receipt['received_at']) > MAX_AGE):
        result['reasons'].append('receipt_outside_issue_availability_window')
    if len(result['reasons']) > 2:
        return result
    try:
        if receipt['status'] != 200:
            raise ValueError('provider_request_failed')
        if hashlib.sha256(receipt['body']).hexdigest() != receipt['sha256']:
            raise ValueError('receipt_checksum_mismatch')
        rows = parse_discharge(json.loads(receipt['body']), station_id, parse_time(receipt['received_at']))
        frame = feature_row(rows, station, issue)
        candidate, baseline = artifact('candidate'), artifact('baseline')
        p = np.clip(candidate['model'].predict_proba(frame[candidate['features']])[:, 1], 1e-6, 1-1e-6)
        inputs = pd.DataFrame({'score': np.log(p/(1-p))})
        probability = float(candidate['calibrator'].predict_proba(inputs)[0, 1])
        x, slope = frame.q_2.iloc[0], (frame.q_2.iloc[0] - frame.q_8.iloc[0])/6
        base_inputs = pd.DataFrame(dict(score=[x+slope*8], season_sin=frame.season_sin,
            season_cos=frame.season_cos, margin=np.log(2.)-frame.q_2, log_area=frame.log_area))
        control = float(baseline['calibrator'].predict_proba(base_inputs)[0, 1])
        result.update(status='experimental_gauge_score', probability=probability, baseline_probability=control,
            receipt_id=receipt['id'], receipt_sha256=receipt['sha256'],
            note='Retrospective development calibration only. This is not a construction-site flood probability.')
    except (ValueError, KeyError, TypeError) as exc:
        result['reasons'].append(str(exc))
    return result


async def discover(point):
    coords = f"POINT({point['longitude']} {point['latitude']})"
    lookup = await providers.fetch(point_key(point)+':network', NLDI+'/comid/position', dict(coords=coords, f='json'), ttl=86400)
    features = lookup['data'].get('features', []) if lookup['available'] else []
    if len(features) != 1 or not str(features[0].get('properties', {}).get('identifier', '')).isdigit():
        return dict(available=False, reason='No unambiguous NLDI catchment lookup', candidates=[])
    comid = str(features[0]['properties']['identifier'])
    async def direction(mode, relationship):
        url = NLDI+f'/comid/{comid}/navigation/{mode}/nwissite'
        response = await providers.fetch(point_key(point)+':'+mode, url, dict(distance=50, f='json'), ttl=86400)
        candidates = []
        if response['available']:
            for f in response['data'].get('features', []):
                p = f.get('properties', {})
                sid = p.get('identifier', '')
                coordinates = (f.get('geometry') or {}).get('coordinates', [])
                if p.get('source') != 'nwissite' or not re.fullmatch(r'USGS-\d{8,15}', sid) or len(coordinates) != 2:
                    continue
                lon, lat = coordinates
                candidates.append(dict(station=sid[5:], name=p.get('name') or sid,
                    latitude=lat, longitude=lon, relationship=relationship,
                    distance_miles=round(distance_miles(point['latitude'], point['longitude'], lat, lon), 2),
                    source_url=url+'?distance=50&f=json'))
        return response['available'], candidates
    up, down = await asyncio.gather(direction('UT', 'upstream_network'), direction('DM', 'downstream_network'))
    candidates = {}
    for c in up[1] + down[1]:
        if c['station'] in candidates:
            c = dict(c, relationship='same_indexed_reach')
        candidates[c['station']] = c
    ordered = sorted(candidates.values(), key=lambda c: (c['distance_miles'], c['station']))
    return dict(available=up[0] and down[0], comid=comid, candidates=ordered,
        candidate_count=len(ordered), navigation_km=50,
        note='NLDI catchment-level connectivity. Distance is straight-line; no travel time or site inundation is established.')


async def active_gauges(network, point):
    """Screen every returned network candidate for fresh, unambiguous instantaneous discharge."""
    candidates, active, excluded = network['candidates'], set(), {}
    for start in range(0, len(candidates), 40):
        batch = candidates[start:start+40]
        key = point_key(point)+f':discharge-screen:{start}'
        result = await providers.fetch(key, USGS+'/latest-continuous/items', dict(f='json',
            monitoring_location_id=','.join('USGS-'+g['station'] for g in batch),
            parameter_code='00060', statistic_id='00011', limit=1000), ttl=300)
        if not result['available'] or any(x.get('rel') == 'next' for x in result['data'].get('links', [])):
            return [], dict(available=False, reason='Current discharge inventory unavailable or paginated')
        for gauge in batch:
            features = [f for f in result['data'].get('features', [])
                        if f.get('properties', {}).get('monitoring_location_id') == 'USGS-'+gauge['station']]
            try:
                rows = parse_discharge(dict(features=features, links=[]), gauge['station'], now_utc())
                status = observation_summary(rows, parse_time(result['fetched_at']), now_utc())
                if status['available']:
                    active.add(gauge['station'])
                else:
                    excluded[gauge['station']] = status.get('reason', 'Current observation is stale')
            except (ValueError, KeyError, TypeError) as exc:
                excluded[gauge['station']] = str(exc)
    return [g for g in candidates if g['station'] in active], dict(available=True,
        candidates_screened=len(candidates), fresh_discharge_gauges=len(active), excluded=excluded,
        note='Active means fresh unqualified discharge, including provisional data; ML has a separate approved-only contract.')


def latest_receipt(station, before=None):
    with db() as c:
        row = c.execute('SELECT * FROM hydrology_receipts WHERE station=? AND received_at<=? ORDER BY received_at DESC LIMIT 1',
                        (station, (before or now_utc()).isoformat())).fetchone()
    return dict(row) if row else None


async def capture(station):
    """Persist the exact provider response after receipt; never backdate availability."""
    now = now_utc()
    previous = latest_receipt(station)
    if previous and now-parse_time(previous['received_at']) < TTL:
        return previous
    params = dict(f='json', monitoring_location_id='USGS-'+station, parameter_code='00060',
        statistic_id='00011', datetime=(now-timedelta(days=8)).isoformat()+'/'+now.isoformat(), limit=50000, skipGeometry='true')
    headers = {'Accept': 'application/json'}
    if os.getenv('USGS_API_KEY'):
        headers['X-Api-Key'] = os.environ['USGS_API_KEY']
    url = str(httpx.URL(USGS+'/continuous/items', params=params))
    try:
        async with providers.LIMIT:
            async with httpx.AsyncClient(timeout=35, follow_redirects=False) as client:
                response = await client.get(url, headers=headers)
        body, status = response.content, response.status_code
    except httpx.HTTPError as exc:
        body, status = json.dumps({'error': type(exc).__name__}).encode(), None
    received = now_utc()
    record = dict(id=uuid.uuid4().hex, station=station, received_at=received.isoformat(), url=url,
                  status=status, body=body, sha256=hashlib.sha256(body).hexdigest())
    with db() as c:
        c.execute('INSERT INTO hydrology_receipts VALUES (:id,:station,:received_at,:url,:status,:sha256,:body)', record)
    return record


async def inspect(point):
    saved = cached(point)
    if saved and not saved['stale']:
        return saved
    async with LOCK:
        saved = cached(point)
        if saved and not saved['stale']:
            return saved
        network = await discover(point)
        selected, screen = await active_gauges(network, point)
        network['discharge_screen'] = screen
        gauges = []
        # A bounded screen, not a claim that all historic network gauges have current discharge.
        for gauge in selected[:3]:
            receipt = await capture(gauge['station'])
            observation = dict(available=False, reason=f"USGS request returned {receipt['status'] or 'a transport error'}")
            if receipt['status'] == 200:
                try:
                    rows = parse_discharge(json.loads(receipt['body']), gauge['station'], parse_time(receipt['received_at']))
                    observation = observation_summary(rows, parse_time(receipt['received_at']), now_utc())
                except (ValueError, KeyError, TypeError) as exc:
                    observation = dict(available=False, reason=str(exc))
            now = now_utc()
            issue = now.replace(hour=now.hour//6*6, minute=0, second=0, microsecond=0)
            prediction = shadow(gauge['station'], latest_receipt(gauge['station'], issue), issue)
            gauges.append(dict(gauge, observation=observation, experimental=prediction,
                receipt=dict(id=receipt['id'], sha256=receipt['sha256'], url=receipt['url'], received_at=receipt['received_at'])))
        available = any(g['observation']['available'] for g in gauges)
        result = dict(status='observed' if available else 'unavailable', checked_at=now_utc().isoformat(),
            latitude=point['latitude'], longitude=point['longitude'], network=network, gauges=gauges,
            operational_prediction='withheld', stale=False, note=NOTE,
            summary='Live river context; experimental ML withheld from operational decisions.' if available else
                'No fresh discharge in the bounded gauge screen. River conditions remain unknown.',
            model=dict(name='Six-hour LightGBM development reference', promoted=False,
                       reason='Independent validation, input-availability and site-applicability requirements remain open.'))
        put_cache(point_key(point), result, result['checked_at'])
        return result


def jha_evidence(snapshot):
    """Cited observational context carries no rating, regardless of experimental score."""
    if snapshot is None:
        return []
    fresh = [] if snapshot.get('stale') else [g for g in snapshot['gauges'] if g['observation']['available']]
    if not fresh:
        return [dict(id='hydrology:unavailable', kind='unavailable', counts=False, rating=None,
            source='USGS discharge / NLDI', summary='Live discharge context unavailable; no ML site-flood warning is established.')]
    items = []
    for g in fresh:
        o = g['observation']
        items.append(dict(id=f"hydrology:{g['station']}:{o['latest_at']}", kind='observed', counts=False, rating=None,
            source='USGS discharge / NLDI', at=o['latest_at'], ref=g['receipt']['id'],
            summary=f"{g['name']}: {o['value_cfs']:g} ft³/s, {o['quality'].lower()}, {g['relationship'].replace('_', ' ')}. "
                    'Supervisor drainage/access review only; site flooding and ML warning remain unverified.'))
    return items
