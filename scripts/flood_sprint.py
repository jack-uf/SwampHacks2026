"""Bounded flood-label feasibility audit. No modeling or production writes.

Run `python -m scripts.flood_sprint collect` once with network access, then
`python -m scripts.flood_sprint summarize` offline to reproduce the audit.
See docs/FLOOD_SPRINT_PROTOCOL.md for the frozen research contract.
"""
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/sprint-20261017-flood'
DOCUMENT = ROOT / 'docs/FLOOD_SPRINT_PROTOCOL.md'
LIDS = ['POPW2', 'HRSP1', 'PORM2', 'RALN7', 'DALT2', 'CRNW1']
USGS = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections'
NOAA = 'https://api.water.noaa.gov/nwps/v1'
SAMPLE = '2024-01-01T00:00:00Z/2024-01-03T00:00:00Z'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def assess_pair(gauge, location):
    """Audit current metadata; it cannot certify a historical threshold interval.

    Historical eligibility deliberately stays closed until documentary threshold
    and gauge-zero histories are reviewed. Current matches cannot open that gate.
    """
    properties = location['properties']
    blockers = ['historical_threshold_interval_unverified', 'historical_gauge_zero_unverified']
    identity = location['id'] == 'USGS-' + gauge['usgsId']
    if not identity:
        blockers.append('station_identifier_mismatch')
    lon, lat = location['geometry']['coordinates'][:2]
    phi1, phi2 = math.radians(gauge['latitude']), math.radians(lat)
    a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        math.radians(lon - gauge['longitude']) / 2) ** 2
    distance = 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., a))))
    if distance > 1:
        blockers.append('station_locations_disagree')
    if properties.get('site_type_code') != 'ST':
        blockers.append('not_a_stream_station')
    flood = gauge.get('flood', {})
    stage = flood.get('categories', {}).get('minor', {}).get('stage')
    if not isinstance(stage, (int, float)) or not math.isfinite(stage) or stage <= -999:
        blockers.append('minor_flood_stage_unavailable')
    if flood.get('stageUnits') != 'ft':
        blockers.append('stage_units_not_feet')
    vertical = gauge.get('datums', {}).get('vertical', {}).get('value', [])
    reference = properties.get('vertical_datum')
    zeros = [row for row in vertical if row.get('abbrev') == reference and reference]
    difference = None
    if len(zeros) != 1 or zeros[0].get('value') is None or properties.get('altitude') is None:
        blockers.append('current_vertical_reference_unresolved')
    else:
        difference = float(zeros[0]['value']) - float(properties['altitude'])
        if abs(difference) > 1e-6:
            blockers.append('current_elevation_difference_requires_review')
    return dict(lid=gauge['lid'], usgs_id=gauge['usgsId'], name=gauge.get('name'),
                identity_matches=identity, distance_km=round(distance, 5),
                huc=properties.get('hydrologic_unit_code'), minor_stage_ft=stage,
                noaa_vertical=vertical, usgs_vertical=reference,
                usgs_altitude_ft=properties.get('altitude'), metadata_elevation_difference_ft=difference,
                eligible=False, blockers=blockers)


def fetch(name, url):
    """Cache exact bytes and provenance. No automatic retries or cache replacement."""
    path = OUT / 'sources' / (name + '.json')
    record = path.with_suffix('.request.json')
    if record.exists():
        metadata = json.loads(record.read_text())
        if metadata['url'] != url:
            raise ValueError('Cached URL changed: ' + name)
        return cached(name)
    started = time.monotonic()
    metadata = dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat())
    payload = None
    try:
        response = httpx.get(url, timeout=40, follow_redirects=True)
        metadata.update(status=response.status_code, final_url=str(response.url),
                        bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        if response.status_code == 200:
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
    metadata['seconds'] = round(time.monotonic() - started, 3)
    write_json(record, metadata)
    print(name, metadata.get('status', metadata.get('error')), flush=True)
    return payload


def collect():
    frozen = OUT / 'protocol.json'
    contract = dict(document='docs/FLOOD_SPRINT_PROTOCOL.md',
                    sha256=hashlib.sha256(DOCUMENT.read_bytes()).hexdigest(),
                    lids=LIDS, sample_interval=SAMPLE)
    if frozen.exists():
        if json.loads(frozen.read_text()) != contract:
            raise ValueError('Frozen protocol changed')
    else:
        write_json(frozen, contract)
    fetch('noaa-openapi', NOAA + '/docs/swagger.json')
    for collection in ['continuous', 'time-series-metadata', 'time-series-revisions', 'time-series-methods']:
        fetch(collection + '-schema', f'{USGS}/{collection}/queryables?f=json')
    previous_noaa = time.monotonic()
    for lid in LIDS:
        if not (OUT / 'sources' / (lid + '-noaa.request.json')).exists():
            time.sleep(max(0, 31 - (time.monotonic() - previous_noaa)))
        gauge = fetch(lid + '-noaa', f'{NOAA}/gauges/{lid}')
        previous_noaa = time.monotonic()
        if not gauge or not gauge.get('usgsId'):
            continue
        sid = 'USGS-' + gauge['usgsId']
        fetch(lid + '-location', f'{USGS}/monitoring-locations/items/{sid}?f=json')
        fetch(lid + '-series', f'{USGS}/time-series-metadata/items?f=json&monitoring_location_id={sid}&limit=100')
        for parameter in ['00065', '00060']:
            fetch(lid + '-' + parameter, f'{USGS}/continuous/items?f=json&monitoring_location_id={sid}'
                  f'&parameter_code={parameter}&datetime={SAMPLE}&limit=1000')
        stage = cached(lid + '-00065')
        for series in sorted({f['properties']['time_series_id'] for f in (stage or {}).get('features', [])}):
            for collection in ['time-series-revisions', 'time-series-methods']:
                fetch(lid + '-' + collection + '-' + series,
                      f'{USGS}/{collection}/items?f=json&time_series_id={series}&limit=100')
    summarize()


def cached(name):
    record = OUT / 'sources' / (name + '.request.json')
    if not record.exists():
        return None
    metadata = json.loads(record.read_text())
    path = OUT / 'sources' / (name + '.json')
    if metadata.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Cached response changed: ' + name)
    return json.loads(path.read_text()) if metadata.get('status') == 200 and not metadata.get('error') else None


def summarize():
    frozen = json.loads((OUT / 'protocol.json').read_text())
    if frozen['sha256'] != hashlib.sha256(DOCUMENT.read_bytes()).hexdigest():
        raise ValueError('Frozen protocol changed')
    rows = []
    for lid in LIDS:
        gauge, location = cached(lid + '-noaa'), cached(lid + '-location')
        if not gauge or not location or not gauge.get('usgsId'):
            rows.append(dict(lid=lid, eligible=False, blockers=['station_metadata_unavailable']))
            continue
        row = assess_pair(gauge, location)
        row['samples'] = {}
        for parameter in ['00065', '00060']:
            sample = cached(lid + '-' + parameter)
            if sample is None:
                row['samples'][parameter] = dict(status='unavailable')
                continue
            points = [f['properties'] for f in sample['features']]
            if any(p['monitoring_location_id'] != 'USGS-' + gauge['usgsId'] or
                   p['parameter_code'] != parameter for p in points):
                raise ValueError('Unexpected station or parameter in ' + lid)
            times = sorted(p['time'] for p in points)
            row['samples'][parameter] = dict(
                status='available' if points else 'empty', observations=len(points),
                first=times[0] if times else None, last=times[-1] if times else None,
                units=sorted({p['unit_of_measure'] for p in points}),
                approval=sorted({p['approval_status'] for p in points}),
                qualifiers=sorted({str(p.get('qualifier')) for p in points}),
                series=sorted({p['time_series_id'] for p in points}),
                truncated=any(link.get('rel') == 'next' for link in sample.get('links', [])))
        rows.append(row)
    requests = [json.loads(p.read_text()) for p in sorted((OUT / 'sources').glob('*.request.json'))]
    summary = dict(decision='stop_historical_label_contract_unverified', gauges=rows,
                   eligible_gauges=sum(row['eligible'] for row in rows),
                   validated_flood_events=None, metrics=None, modeled=False,
                   download_bytes=sum(r.get('bytes', 0) for r in requests),
                   request_seconds=round(sum(r['seconds'] for r in requests), 3),
                   requests=len(requests), request_errors=sum(r.get('status') != 200 or bool(r.get('error')) for r in requests))
    write_json(OUT / 'audit.json', summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['collect', 'summarize'])
    args = parser.parse_args()
    {'collect': collect, 'summarize': summarize}[args.command]()
