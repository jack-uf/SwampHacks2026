import json
import re
from backend import jev
from scripts.common import ROOT


def test_planned_projects_are_public_multi_entity_records_with_provenance():
    data = json.loads((ROOT/'data/planned-projects.json').read_text())
    assert data['source']['url'].startswith('https://www.eia.gov/') and len(data['source']['sha256']) == 64
    projects = data['projects']
    assert len({p['entity_id'] for p in projects}) >= 2 and len({(p['plant_id'], p['entity_id']) for p in projects}) == len(projects)
    assert all(-180 <= p['longitude'] <= -60 and 18 <= p['latitude'] <= 72 for p in projects)
    assert all(p['planned_month'] is None or re.fullmatch(r'20\d\d-(0[1-9]|1[0-2])', p['planned_month']) for p in projects)


def test_hazard_memory_holds_noaa_records_mapped_to_supported_hazards():
    data = json.loads((ROOT/'data/hazard-memory.json').read_text())
    sites = {s['id'] for s in json.loads((ROOT/'data/sites.json').read_text())['sites']}
    assert set(data['by_site']) == sites
    assert all(len(s['sha256']) == 64 for s in data['sources'].values())
    events = data['events'].values()
    assert all(e['hazard'] in jev.CATEGORIES and e['source'] == 'NOAA Storm Events' for e in events)
    assert all((e['latitude'] is None) == (e['geography'] != 'point') for e in events)
    assert all(i in data['events'] for ids in data['by_site'].values() for i in ids)
