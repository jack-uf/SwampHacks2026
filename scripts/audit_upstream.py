"""Recheck connected-gauge feasibility from retained metadata and source bytes."""
import csv
import json

from shapely.geometry import shape

from scripts.hydrology_research import ROOT, digest, write_json
from scripts.upstream_research import OUT, upstream_sites
from scripts.upstream_pilot import verify_protocol, reference_coverage


def audit():
    verify_protocol()
    records = {}
    for manifest in sorted(OUT.rglob('*.request.json')):
        source = manifest.with_name(manifest.name.removesuffix('.request.json'))
        record = json.loads(manifest.read_text())
        assert record['status'] == 200 and source.stat().st_size == record['bytes']
        assert digest(source) == record['sha256'], str(source)
        records[str(source.relative_to(ROOT))] = record
    metadata = json.loads((OUT / 'expanded-metadata.json').read_text())
    assert metadata['pilot_sha256'] == digest(OUT / 'metadata.json')
    pairs = json.loads((OUT / 'expanded-pairs.json').read_text())
    assert pairs['metadata_sha256'] == digest(OUT / 'expanded-metadata.json')
    previous = {s['station'] for name in ['hydrology', 'onset', 'response', 'warning']
                for s in json.loads((ROOT / f'experiments/sprint-20261017-{name}/stations.json').read_text())}
    geography = json.loads((ROOT / 'experiments/sprint-20261017-response/geography.json').read_text())
    assert digest(ROOT / 'experiments/sprint-20261017-response/geography.json') == metadata['geography_sha256']
    occupied = []
    for sid in previous:
        path = ROOT / f'experiments/sprint-20261017-response/basins/{sid}.gauge-basin.json'
        assert digest(path) == geography['boundary_hashes'][sid]
        occupied.append(shape(json.loads(path.read_text())['features'][0]['geometry']))
    for s in metadata['selected']:
        assert s['site_no'] not in previous
        polygon = shape(json.loads((OUT / 'basins' / f"{s['site_no']}.json").read_text())['features'][0]['geometry'])
        assert polygon.is_valid and polygon.area > 0
        assert all(polygon.intersection(other).area == 0 for other in occupied)
        occupied.append(polygon)
    series = []
    for source in (OUT / 'research').glob('expanded-catalog-*.rdb'):
        lines = [s for s in source.read_text().splitlines() if s and not s.startswith('#')]
        series.extend(csv.DictReader([lines[0]] + lines[2:], delimiter='\t'))
    eligible = {s['site_no'] for s in series if s['agency_cd'] == 'USGS' and s['data_type_cd'] == 'dv'
                and s['parm_cd'] == '00060' and s['stat_cd'] == '00003'
                and s['begin_date'] <= '2000-01-01' and s['end_date'] >= '2024-09-25' and int(s['count_nu']) >= 7000}
    assert len(eligible) == pairs['catalog_eligible_sites']
    connections = {s['station']: s for s in metadata['connections']}
    for c in connections.values():
        path = OUT / 'navigation' / f"{c['station']}.json"
        assert digest(path) == c['navigation_sha256']
        assert upstream_sites(json.loads(path.read_text()), c['station']) == c['upstream']
    for pair in pairs['pairs']:
        assert pair['upstream'] in eligible & set(connections[pair['station']]['upstream'])
        ratio = float(pair['site']['drain_area_va']) / float(pair['downstream']['drain_area_va'])
        assert ratio == pair['area_ratio'] and .05 <= ratio <= .95
        assert pair['location_distance_km'] <= 1 and pair['site']['site_tp_cd'] == 'ST'
    coverage = json.loads((OUT / 'coverage.json').read_text())
    assert coverage['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert coverage['code_sha256'] == digest(ROOT / 'scripts/upstream_pilot.py')
    development = {p['upstream'] for p in pairs['pairs'] if p['role'] == 'development'}
    assert {p.stem for p in (OUT / 'sources').glob('*.rdb')} == development
    for row in coverage['pairs']:
        source = OUT / 'sources' / f"{row['station']}.rdb"
        assert dict(downstream=row['downstream'], **reference_coverage(source.read_text(), row['station'])) == row
    assert coverage['reference_eligible_pairs'] == 3 and coverage['required_pairs'] == 4
    assert coverage['status'] == 'aborted_incomplete_frozen_population'
    assert coverage['fitted_models'] == 0 and not coverage['independent_geographic_outcomes_opened']
    assert not (OUT / 'evaluation.json').exists() and not (OUT / 'development.csv.gz').exists()
    assert not list((OUT / 'artifacts').glob('*')) and not list((OUT / 'predictions').glob('*'))
    write_json(OUT / 'audit.json', dict(status='passed', sources=records, source_bytes=sum(r['bytes'] for r in records.values()),
        source_files=len(records), prior_gauges=len(previous), disjoint_new_basins=len(metadata['selected']),
        development_navigation=sum(c['role'] == 'development' for c in connections.values()),
        uninspected_navigation=sum(c['role'] == 'uninspected' for c in connections.values()),
        navigated_eight_digit_usgs_sites=pairs['navigated_usgs_8digit_sites'],
        catalog_eligible_sites=len(eligible), development_metadata_pairs=len(development),
        reference_eligible_development_pairs=coverage['reference_eligible_pairs'],
        unopened_metadata_pairs=sum(p['role'] == 'uninspected' for p in pairs['pairs']),
        coverage_sha256=digest(OUT / 'coverage.json'), code_sha256=digest(ROOT / 'scripts/audit_upstream.py')))
    print('Audited metadata and coverage; reserved geographic outcomes remain unopened', flush=True)


if __name__ == '__main__':
    audit()
