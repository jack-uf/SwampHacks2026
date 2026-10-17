"""Bounded connected-gauge feasibility study, isolated from application models."""
import json
import re
import csv
import math

from scripts.hydrology_research import ROOT, digest, fetch, write_json

OUT = ROOT / 'experiments/sprint-20261017-upstream'


def upstream_sites(data, downstream):
    """Parse USGS upstream-tributary navigation, without treating proximity as a link."""
    if data.get('type') != 'FeatureCollection':
        raise ValueError('Expected navigation FeatureCollection')
    identifiers = []
    for feature in data['features']:
        props = feature['properties']
        identifier = props['identifier']
        if props['source'] != 'nwissite' or not re.fullmatch(r'[A-Z0-9]+-\d{8,15}', identifier):
            raise ValueError('Invalid official gauge identifier')
        if feature['geometry']['type'] != 'Point':
            raise ValueError('Expected gauge point')
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate navigation identifier')
    return sorted(sid[5:] for sid in identifiers if sid.startswith('USGS-') and sid != 'USGS-' + downstream)


def prepare():
    """Select a bounded metadata pilot before requesting discharge observations."""
    from concurrent.futures import ThreadPoolExecutor
    from shapely.geometry import shape
    from scripts.response_research import OUT as response, disjoint_basin
    if (OUT / 'metadata.json').exists():
        raise FileExistsError('Metadata pilot already frozen')
    previous = {s['station']: s for name in ['hydrology', 'onset', 'response', 'warning']
                for s in json.loads((ROOT / f'experiments/sprint-20261017-{name}/stations.json').read_text())}
    geography = json.loads((response / 'geography.json').read_text())
    occupied = []
    for sid in previous:
        path = response / 'basins' / f'{sid}.gauge-basin.json'
        if digest(path) != geography['boundary_hashes'][sid]:
            raise ValueError('Previous basin changed')
        occupied.append(shape(json.loads(path.read_text())['features'][0]['geometry']))
    sites = json.loads((ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    candidates = [s for s in sites if s['site_no'] not in previous
                  and len(s['huc_cd']) in (8, 12) and s['huc_cd'].isdigit()]
    # Largest unused candidate per HUC4, then largest drainage areas first.
    candidates = [max([s for s in candidates if s['huc_cd'][:4] == h],
                      key=lambda s: (float(s['drain_area_va'] or 0), s['site_no']))
                  for h in sorted({s['huc_cd'][:4] for s in candidates})]
    candidates = sorted(candidates, key=lambda s: (-float(s['drain_area_va'] or 0), s['site_no']))[:48]
    exclusions = []
    def basin(s):
        sid = s['site_no']
        path = OUT / 'basins' / f'{sid}.json'
        raw = json.loads(fetch(path, f'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}/basin?simplified=true'))
        if raw['type'] != 'FeatureCollection' or len(raw['features']) != 1:
            raise ValueError('Ambiguous upstream basin: ' + sid)
        geometry = shape(raw['features'][0]['geometry'])
        return s, geometry
    selected = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for s, geometry in pool.map(basin, candidates):
            if disjoint_basin(geometry, occupied):
                selected.append(s)
                occupied.append(geometry)
            else:
                exclusions.append(dict(station=s['site_no'], reason='overlaps an inspected or earlier selected basin'))
    training = [s for s in json.loads((response / 'stations.json').read_text()) if s['role'] == 'development']
    training = sorted(training, key=lambda s: (-s['area_km2'], s['station']))[:24]
    def navigation(item):
        sid, role = item
        path = OUT / 'navigation' / f'{sid}.json'
        url = f'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}/navigation/UT/nwissite?distance=500'
        data = json.loads(fetch(path, url))
        own = [f for f in data['features'] if f['properties']['identifier'] == 'USGS-' + sid]
        if len(own) != 1:
            raise ValueError('Missing or ambiguous downstream point: ' + sid)
        source = next(s for s in sites if s['site_no'] == sid)
        x, y = own[0]['geometry']['coordinates']
        km = 111.2 * math.hypot((x - float(source['dec_long_va'])) * math.cos(math.radians(y)),
                               y - float(source['dec_lat_va']))
        if km > 1:
            raise ValueError('Indexed gauge location differs from metadata: ' + sid)
        return dict(station=sid, role=role, upstream=upstream_sites(data, sid),
                    non_usgs_records=sum(not f['properties']['identifier'].startswith('USGS-') for f in data['features']),
                    location_distance_km=km, navigation_sha256=digest(path))
    items = [(s['station'], 'development') for s in training] + [(s['site_no'], 'uninspected') for s in selected]
    with ThreadPoolExecutor(max_workers=2) as pool:
        connections = list(pool.map(navigation, items))
    write_json(OUT / 'metadata.json', dict(candidates=candidates, selected=selected,
        exclusions=exclusions, connections=connections, prior_stations=len(previous),
        geography_sha256=digest(response / 'geography.json'),
        script_sha256=digest(ROOT / 'scripts/upstream_research.py')))
    print('Metadata pilot:', len(selected), 'new disjoint basins;', len(connections), 'navigation requests', flush=True)


def expand():
    """Complete the development metadata screen without replacing the first pilot."""
    from concurrent.futures import ThreadPoolExecutor
    path = OUT / 'expanded-metadata.json'
    if path.exists():
        raise FileExistsError('Expanded metadata already recorded')
    metadata = json.loads((OUT / 'metadata.json').read_text())
    seen = {c['station'] for c in metadata['connections']}
    stations = json.loads((ROOT / 'experiments/sprint-20261017-response/stations.json').read_text())
    remaining = [s for s in stations if s['role'] == 'development' and s['station'] not in seen]
    def navigation(s):
        sid = s['station']
        target = OUT / 'navigation' / f'{sid}.json'
        data = json.loads(fetch(target, f'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}/navigation/UT/nwissite?distance=500'))
        own = [f for f in data['features'] if f['properties']['identifier'] == 'USGS-' + sid]
        if len(own) != 1:
            raise ValueError('Missing or ambiguous downstream point: ' + sid)
        x, y = own[0]['geometry']['coordinates']
        km = 111.2 * math.hypot((x - s['longitude']) * math.cos(math.radians(y)), y - s['latitude'])
        if km > 1:
            raise ValueError('Indexed gauge location differs from metadata: ' + sid)
        return dict(station=sid, role='development', upstream=upstream_sites(data, sid),
                    location_distance_km=km, navigation_sha256=digest(target))
    with ThreadPoolExecutor(max_workers=2) as pool:
        metadata['connections'].extend(pool.map(navigation, remaining))
    metadata['pilot_sha256'] = digest(OUT / 'metadata.json')
    metadata['script_sha256'] = digest(ROOT / 'scripts/upstream_research.py')
    write_json(path, metadata)
    print('Additional development metadata:', len(remaining), flush=True)


def catalog(expanded=False):
    """Check daily-discharge coverage and choose one upstream gauge by metadata only."""
    from concurrent.futures import ThreadPoolExecutor
    prefix = 'expanded-' if expanded else ''
    if (OUT / f'{prefix}pairs.json').exists():
        raise FileExistsError('Gauge pairs already frozen')
    metadata = json.loads((OUT / f'{prefix}metadata.json').read_text())
    identifiers = sorted({s for c in metadata['connections'] for s in c['upstream'] if len(s) == 8})
    def rows(raw):
        lines = [l for l in raw.decode().splitlines() if l and not l.startswith('#')]
        if len(lines) < 2:
            raise ValueError('Missing USGS catalog schema')
        return list(csv.DictReader([lines[0]] + lines[2:], delimiter='\t'))
    def chunk(item):
        index, ids = item
        url = ('https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=' + ','.join(ids) +
               '&seriesCatalogOutput=true&parameterCd=00060')
        return rows(fetch(OUT / 'research' / f'{prefix}catalog-{index}.rdb', url))
    batches = [(i // 50, identifiers[i:i+50]) for i in range(0, len(identifiers), 50)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        series = [r for chunk_rows in pool.map(chunk, batches) for r in chunk_rows]
    eligible = {s['site_no'] for s in series if s['agency_cd'] == 'USGS' and s['data_type_cd'] == 'dv'
                and s['parm_cd'] == '00060' and s['stat_cd'] == '00003'
                and s['begin_date'] <= '2000-01-01' and s['end_date'] >= '2024-09-25'
                and int(s['count_nu']) >= 7000}
    expanded = []
    for index, start in enumerate(range(0, len(eligible), 50)):
        ids = sorted(eligible)[start:start+50]
        url = 'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=' + ','.join(ids) + '&siteOutput=expanded'
        expanded.extend(rows(fetch(OUT / 'research' / f'{prefix}sites-{index}.rdb', url)))
    by_id = {s['site_no']: s for s in expanded}
    if len(by_id) != len(expanded) or set(by_id) != eligible:
        raise ValueError('Expanded gauge catalog does not match eligible sites')
    sites = json.loads((ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    downstream = {s['site_no']: s for s in sites}
    pairs, exclusions = [], []
    for connection in metadata['connections']:
        sid = connection['station']; s = downstream[sid]
        area = float(s['drain_area_va']); options = []
        nav = json.loads((OUT / 'navigation' / f'{sid}.json').read_text())
        for uid in connection['upstream']:
            if uid not in by_id:
                continue
            u = by_id[uid]
            try:
                ratio = float(u['drain_area_va']) / area
            except ValueError:
                continue
            if not .05 <= ratio <= .95 or u['site_tp_cd'] != 'ST':
                continue
            f = next(f for f in nav['features'] if f['properties']['identifier'] == 'USGS-' + uid)
            x, y = f['geometry']['coordinates']
            km = 111.2 * math.hypot((x - float(u['dec_long_va'])) * math.cos(math.radians(y)),
                                   y - float(u['dec_lat_va']))
            if km > 1:
                continue
            options.append(dict(upstream=uid, area_ratio=ratio, location_distance_km=km, site=u))
        if options:
            chosen = sorted(options, key=lambda o: (-o['area_ratio'], o['upstream']))[0]
            pairs.append(dict(station=sid, role=connection['role'], downstream=s,
                              eligible_options=len(options), **chosen))
        else:
            exclusions.append(dict(station=sid, role=connection['role'], reason='no metadata-eligible connected discharge gauge'))
    write_json(OUT / f'{prefix}pairs.json', dict(pairs=pairs, exclusions=exclusions,
        navigated_usgs_8digit_sites=len(identifiers), catalog_eligible_sites=len(eligible),
        metadata_sha256=digest(OUT / f'{prefix}metadata.json'), script_sha256=digest(ROOT / 'scripts/upstream_research.py')))
    print('Metadata-eligible pairs:', {role: sum(p['role'] == role for p in pairs)
                                       for role in ['development', 'uninspected']}, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'catalog', 'expand', 'catalog-expanded'])
    args = parser.parse_args()
    {'prepare': prepare, 'catalog': catalog, 'expand': expand,
     'catalog-expanded': lambda: catalog(expanded=True)}[args.command]()
