"""Freeze a metadata-only, geographically disjoint CAMELS cohort.

Run ``.venv/bin/python -m scripts.localcal_cohort`` with network authorization.
``--self-test`` checks the selection invariants without network access or writes.
Only station inventories, registries, request manifests, gauge points and basin
polygons are read. Outcome payloads and prediction labels are never opened.
Prior station IDs and IDs in observation request manifests are excluded. Every
excluded population has a polygon in the occupied ledger, including upstream
gauges absent from the CAMELS inventory. Existing source caches are immutable.

Candidates follow SHA256('localcal-20261017-' + station), then station ID.
The target is 24 gauges, at most one per HUC4, allowing previously used HUC4s.
Every accepted polygon must have exactly zero intersection area with ALL prior
population polygons and earlier selections. Touching boundaries are permitted.
This certifies separation of the retained simplified geometries, not unsimplified
watersheds, shared weather systems, or statistical independence.

At most 80 candidate HTTP attempts are allowed, with no retries. Obtaining missing
occupied polygons is separately counted. Failed metadata screens may be followed
by later candidates; outcomes never determine selection or replacements. A frozen
audit is never overwritten. At least 10 selected HUC4s are required for success;
an insufficient audit is retained and the command exits unsuccessfully.
"""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sprint-20261017-localcal'
API = 'https://api.water.usgs.gov/nldi/linked-data/'
LIMIT = 80
TARGET = 24
MIN_HUCS = 10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def reference(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def order_key(station):
    return hashlib.sha256(('localcal-20261017-' + station).encode()).hexdigest(), station


def basin(raw):
    if raw.get('type') != 'FeatureCollection' or len(raw.get('features', [])) != 1:
        raise ValueError('Expected exactly one basin feature')
    geometry = shape(raw['features'][0]['geometry'])
    if geometry.geom_type not in ('Polygon', 'MultiPolygon') or geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
        raise ValueError('Invalid basin polygon; no geometry repairs are permitted')
    return geometry


def location(raw, station, latitude, longitude):
    if raw.get('type') != 'FeatureCollection':
        raise ValueError('Expected gauge FeatureCollection')
    matches = [f for f in raw.get('features', [])
               if f.get('properties', {}).get('identifier') == 'USGS-' + station
               and f['properties'].get('source') == 'nwissite']
    if len(matches) != 1 or matches[0]['geometry']['type'] != 'Point':
        raise ValueError('Missing or ambiguous exact USGS identifier and nwissite point')
    feature = matches[0]
    x, y = feature['geometry']['coordinates'][:2]
    if not all(math.isfinite(v) for v in (x, y, latitude, longitude)):
        raise ValueError('Nonfinite gauge coordinates')
    # Haversine distance in kilometers; the limit is inclusive and unrounded.
    dy, dx = math.radians(y - latitude), math.radians(x - longitude)
    a = math.sin(dy / 2) ** 2 + math.cos(math.radians(y)) * math.cos(math.radians(latitude)) * math.sin(dx / 2) ** 2
    distance = 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, a))))
    if distance > 1:
        raise ValueError(f'Gauge point differs from inventory by {distance:.6f} km, above 1 km')
    comid = str(feature['properties'].get('comid', ''))
    if not comid.isdigit():
        raise ValueError('Exact gauge point has no numeric COMID')
    return dict(identifier='USGS-' + station, source='nwissite', comid=comid,
                longitude=x, latitude=y, location_distance_km=distance,
                reachcode=feature['properties'].get('reachcode'))


def station_record(site):
    sid = site['site_no']
    huc = site['huc_cd']
    if site.get('agency_cd') != 'USGS' or not re.fullmatch(r'\d{8}', sid):
        raise ValueError('Not an eight-digit USGS gauge')
    if len(huc) not in (8, 12) or not huc.isdigit():
        raise ValueError('Missing or invalid inventory HUC')
    zone = {'EST': 'America/New_York', 'CST': 'America/Chicago', 'MST': 'America/Denver', 'PST': 'America/Los_Angeles'}
    fixed = {'EST': 'Etc/GMT+5', 'CST': 'Etc/GMT+6', 'MST': 'Etc/GMT+7', 'PST': 'Etc/GMT+8'}
    if site['local_time_fg'] not in ('Y', 'N'):
        raise ValueError('Unknown daylight-saving metadata')
    record = dict(station=sid, name=site['station_nm'], huc=huc[:8],
                  latitude=float(site['dec_lat_va']), longitude=float(site['dec_long_va']),
                  area_km2=float(site['drain_area_va']) * 2.589988110336,
                  elevation_m=float(site['alt_va']) * .3048,
                  timezone=(zone if site['local_time_fg'] == 'Y' else fixed)[site['tz_cd']], role='fresh_test')
    if not all(math.isfinite(record[k]) for k in ('latitude', 'longitude', 'area_km2', 'elevation_m')) or record['area_km2'] <= 0:
        raise ValueError('Invalid numeric station metadata')
    return record


class MetadataCache:
    def __init__(self):
        self.requests = []
        self.reused = {}

    def existing(self, path, expected_url=None, pinned=None):
        manifest_path = Path(str(path) + '.request.json')
        manifest = read(manifest_path)
        actual = digest(path)
        if actual != manifest['sha256'] or (pinned is not None and actual != pinned):
            raise ValueError(f'Cached metadata hash mismatch: {path}')
        if manifest['status'] != 200 or (expected_url and manifest['url'] != expected_url):
            raise ValueError(f'Unexpected cached metadata request: {path}')
        if not manifest['url'].startswith(API):
            raise ValueError('Only NLDI metadata cache is allowed')
        provenance = dict(**reference(path), request=reference(manifest_path), url=manifest['url'])
        self.reused[str(path.relative_to(ROOT))] = provenance
        return read(path), provenance

    def fetch(self, station, kind, url, purpose):
        allowed = (re.fullmatch(re.escape(API) + r'nwissite/USGS-\d{8}(?:/basin\?simplified=true)?', url)
                   or re.fullmatch(re.escape(API) + r'comid/\d+/basin\?simplified=true', url))
        if not allowed or kind not in ('location', 'basin'):
            raise ValueError('Request is not an allowed gauge/basin metadata URL')
        path = OUT / 'basins' / f'{station}.{kind}.json'
        manifest_path = Path(str(path) + '.request.json')
        if path.exists() or manifest_path.exists():
            raise FileExistsError(f'Retained request already exists; audit requires explicit recovery: {path}')
        if purpose == 'candidate' and sum(r['purpose'] == 'candidate' for r in self.requests) >= LIMIT:
            raise RuntimeError('Candidate request budget exhausted')
        manifest = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                        station=station, kind=kind, purpose=purpose, status=None)
        self.requests.append(manifest)
        try:
            response = httpx.get(url, timeout=30, follow_redirects=False,
                                 headers={'User-Agent': 'PlanITControl/0.1 metadata-only cohort audit'})
            payload = response.content
            manifest.update(status=response.status_code, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
            response.raise_for_status()
            raw = json.loads(payload)
        except (httpx.HTTPError, ValueError) as exc:
            manifest['error'] = f'{type(exc).__name__}: {exc}'
            raise ValueError(manifest['error']) from exc
        finally:
            write(manifest_path, manifest)
        return raw, dict(**reference(path), request=reference(manifest_path), url=url)


def prior_population():
    """Read manifests, never the observation payload associated with a manifest."""
    prior, inputs = {}, []
    for path in sorted((ROOT / 'experiments').glob('sprint-20261017-*/stations.json')):
        if OUT in path.parents:
            continue
        provenance = reference(path)
        inputs.append(provenance)
        for item in read(path):
            prior.setdefault(item['station'], []).append(dict(kind='station_registry', **provenance))
    for path in sorted((ROOT / 'experiments').glob('sprint-20261017-*/**/*.request.json')):
        if OUT in path.parents:
            continue
        manifest = read(path)
        url = manifest.get('url', '')
        parsed = urlparse(url)
        if parsed.hostname == 'waterservices.usgs.gov' and parsed.path.rstrip('/') in ('/nwis/dv', '/nwis/iv'):
            params = parse_qs(parsed.query)
            ids = params.get('sites', [])
            if not ids:
                raise ValueError(f'Observation request without explicit station IDs: {path}')
            provenance = dict(kind='observation_request_manifest', **reference(path), url=url)
            for sid in ','.join(ids).split(','):
                if not re.fullmatch(r'\d{8,15}', sid):
                    raise ValueError(f'Unparseable source-request station ID in {path}')
                prior.setdefault(sid, []).append(provenance)
            inputs.append(reference(path))
    return prior, inputs


def run():
    if (OUT / 'cohort.json').exists() or (OUT / 'stations.json').exists() or (OUT / 'basins').exists():
        raise FileExistsError('Cohort or retained request files already exist; no overwrite or automatic retry')
    inventory_path = ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json'
    coordinate_path = inventory_path.with_name('basin.json')
    sites = read(inventory_path)
    if len({s['site_no'] for s in sites}) != len(sites):
        raise ValueError('Duplicate station inventory IDs')
    camels = {s.removeprefix('camels_') for s in read(coordinate_path) if s.startswith('camels_')}
    if {s['site_no'] for s in sites} != camels:
        raise ValueError('Inventory is not the exact CAMELS coordinate population')
    prior, inputs = prior_population()
    geography_path = ROOT / 'experiments/sprint-20261017-response/geography.json'
    geography = read(geography_path)
    inputs.extend([reference(inventory_path), reference(coordinate_path), reference(geography_path)])
    audit = dict(schema_version=1, scope='METADATA ONLY; no outcomes or prediction labels',
                 created_utc=datetime.now(timezone.utc).isoformat(), script=reference(Path(__file__)),
                 status='incomplete', target=TARGET, minimum_huc4=MIN_HUCS, candidate_request_limit=LIMIT,
                 order="sha256(('localcal-20261017-' + station).encode()).hexdigest(), then station",
                 overlap_rule='intersection.area == 0 for every prior population and selected polygon; no tolerance',
                 huc_rule='one station per inventory HUC4; previously used HUC4 allowed',
                 mapping_rule='one exact USGS identifier, nwissite Point, haversine distance <= 1 km; numeric COMID',
                 inputs=inputs, prior_station_provenance=prior, occupied_geometry_hashes=[],
                 candidate_order=[], exclusions=[], selected=[], unattempted=[],
                 limitations=['Simplified NLDI polygons may not reproduce unsimplified catchment boundaries.',
                              'Polygon separation does not establish weather or statistical independence.',
                              'Metadata cannot prove that outcomes were never viewed outside retained project provenance.',
                              'Forecast and outcome coverage is not checked; no outcome-based replacements are allowed.',
                              'HUC4 uses the station inventory; NLDI reachcode disagreement is recorded, not silently substituted.'])
    cache = MetadataCache()
    occupied, selected, records = [], [], []
    failure = None
    try:
        # Fail closed if even one prior population polygon cannot be certified.
        for sid in sorted(prior):
            cached = ROOT / f'experiments/sprint-20261017-response/basins/{sid}.gauge-basin.json'
            upstream = ROOT / f'experiments/sprint-20261017-upstream/basins/{sid}.json'
            if cached.exists():
                if sid not in geography['boundary_hashes']:
                    raise ValueError('Occupied response geometry missing from pinned geography: ' + sid)
                raw, ref = cache.existing(cached, pinned=geography['boundary_hashes'][sid])
            elif upstream.exists():
                raw, ref = cache.existing(upstream, API + f'nwissite/USGS-{sid}/basin?simplified=true')
            else:
                raw, ref = cache.fetch(sid, 'basin', API + f'nwissite/USGS-{sid}/basin?simplified=true', 'occupied')
            geometry = basin(raw)
            entry = dict(station=sid, **ref)
            audit['occupied_geometry_hashes'].append(entry)
            occupied.append((entry, geometry))
        candidates = sorted((s for s in sites if s['site_no'] not in prior), key=lambda s: order_key(s['site_no']))
        audit['candidate_order'] = [dict(station=s['site_no'], sha256=order_key(s['site_no'])[0]) for s in candidates]
        audit['inventory_count'] = len(sites)
        audit['fresh_candidate_count'] = len(candidates)
        for index, site in enumerate(candidates):
            sid = site['site_no']
            entry = dict(station=sid, order_index=index, inventory_huc=site['huc_cd'])
            if len(records) >= TARGET:
                audit['unattempted'] = [s['site_no'] for s in candidates[index:]]
                audit['stop_reason'] = 'target reached'
                break
            if sum(r['purpose'] == 'candidate' for r in cache.requests) >= LIMIT:
                audit['unattempted'] = [s['site_no'] for s in candidates[index:]]
                audit['stop_reason'] = 'candidate request budget exhausted'
                break
            try:
                record = station_record(site)
                if any(s['huc'][:4] == record['huc'][:4] for s in records):
                    entry['reason'] = 'HUC4 already selected'
                    audit['exclusions'].append(entry)
                    continue
                cached = ROOT / f'experiments/sprint-20261017-response/basins/{sid}.gauge-location.json'
                if cached.exists():
                    raw, ref = cache.existing(cached, API + f'nwissite/USGS-{sid}', geography['location_hashes'][sid])
                else:
                    raw, ref = cache.fetch(sid, 'location', API + f'nwissite/USGS-{sid}', 'candidate')
                entry['location'] = ref
                mapped = location(raw, sid, record['latitude'], record['longitude'])
                entry['mapping'] = mapped
                entry['huc_disagreement'] = bool(mapped['reachcode'] and mapped['reachcode'][:4] != record['huc'][:4])
                cached = ROOT / f'experiments/sprint-20261017-response/basins/{sid}.gauge-basin.json'
                upstream = ROOT / f'experiments/sprint-20261017-upstream/basins/{sid}.json'
                if cached.exists():
                    raw, ref = cache.existing(cached, API + f'comid/{mapped["comid"]}/basin?simplified=true', geography['boundary_hashes'][sid])
                elif upstream.exists():
                    raw, ref = cache.existing(upstream, API + f'nwissite/USGS-{sid}/basin?simplified=true')
                else:
                    raw, ref = cache.fetch(sid, 'basin', API + f'comid/{mapped["comid"]}/basin?simplified=true', 'candidate')
                entry['basin'] = ref
                geometry = basin(raw)
                conflicts = []
                for prior_entry, other in occupied + selected:
                    area = geometry.intersection(other).area
                    if area > 0:
                        conflicts.append(dict(station=prior_entry['station'], sha256=prior_entry['sha256'], intersection_area_degrees2=area))
                if conflicts:
                    entry.update(reason='positive-area polygon overlap', conflicts=conflicts)
                    audit['exclusions'].append(entry)
                    continue
                records.append(record)
                audit['selected'].append(dict(**entry, huc4=record['huc'][:4]))
                selected.append((dict(station=sid, **ref), geometry))
                print(f'Selected {len(records):02d}: {sid} HUC4 {record["huc"][:4]}', flush=True)
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                entry.update(reason='metadata screen failed', error=f'{type(exc).__name__}: {exc}')
                audit['exclusions'].append(entry)
        else:
            audit['stop_reason'] = 'candidate inventory exhausted'
        audit['status'] = 'eligible' if len({s['huc'][:4] for s in records}) >= MIN_HUCS else 'insufficient_huc4'
    except Exception as exc:
        audit['status'] = 'failed'
        audit['error'] = f'{type(exc).__name__}: {exc}'
        failure = exc
    finally:
        audit.update(selected_count=len(records), selected_huc4_count=len({s['huc'][:4] for s in records}),
                     occupied_count=len(occupied), prior_station_count=len(prior),
                     candidate_requests=sum(r['purpose'] == 'candidate' for r in cache.requests),
                     occupied_requests=sum(r['purpose'] == 'occupied' for r in cache.requests),
                     requests=cache.requests, reused_metadata=list(cache.reused.values()))
        write(OUT / 'stations.json', records)
        audit['stations'] = reference(OUT / 'stations.json')
        write(OUT / 'cohort.json', audit)
    print(json.dumps({k: audit[k] for k in ('status', 'selected_count', 'selected_huc4_count', 'occupied_count', 'candidate_requests', 'occupied_requests')}), flush=True)
    if failure:
        raise RuntimeError('Occupied metadata audit failed; inspect cohort.json') from failure
    if audit['status'] != 'eligible':
        raise RuntimeError('Fewer than 10 eligible HUC4 groups; cohort is not authorized for evaluation')


def self_test():
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    touch = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    overlap = Polygon([(.99, 0), (2, 0), (2, 1), (.99, 1)])
    assert square.intersection(touch).area == 0
    assert square.intersection(overlap).area > 0
    raw = dict(type='FeatureCollection', features=[dict(properties=dict(identifier='USGS-12345678', source='nwissite', comid=1), geometry=dict(type='Point', coordinates=[0, 0]))])
    assert location(raw, '12345678', 0, 0)['location_distance_km'] == 0
    for sid, latitude in [('00000000', 0), ('12345678', .01)]:
        try:
            location(raw, sid, latitude, 0)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid location accepted')
    raw['features'].append(raw['features'][0])
    try:
        location(raw, '12345678', 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate identifier accepted')
    assert order_key('12345678')[0] == hashlib.sha256(b'localcal-20261017-12345678').hexdigest()
    prior, _ = prior_population()
    assert {'03212980', '07055875', '07375800', '12357500'} <= prior.keys()
    print('Metadata-only self-test passed: strict overlap, exact identifier, distance, duplicate point, hash order, upstream exclusion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else run()
