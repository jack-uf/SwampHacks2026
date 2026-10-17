"""Site Hazard Memory: historical NOAA Storm Events and closed crew reports, ranked with visible factors."""
import json
from functools import lru_cache
from math import cos, pi
from backend.domain import distance_miles, parse_time
from backend.missions import ACTIVITIES
from scripts.common import ROOT

PATH = ROOT/'data/hazard-memory.json'
RADIUS_MILES = 20
WEIGHTS = dict(geography=.3, conditions=.3, mission=.2, season=.1, impact=.1)
CONDITION_SCORE = dict(stop_work=1, high=1, elevated=.7, low=.2, unknown=.4)


@lru_cache(maxsize=1)
def dataset():
    return json.loads(PATH.read_text()) if PATH.exists() else None


def site_records(site_id):
    data = dataset()
    if not data:
        return []
    return [data['events'][event_id] for event_id in data['by_site'].get(site_id, [])]


def crew_records(reports):
    """Supervisor-closed reports (verified or resolved) become local memory, one record per labeled hazard."""
    records = []
    for r in reports:
        if r['status'] not in ('verified', 'resolved'):
            continue
        for hazard in (r.get('triage') or {}).get('manual_categories') or [r['classification']]:
            records.append(dict(id='report:'+r['id'], source=f"PlanIT crew report ({r['status']})", event_type='Crew report', hazard=hazard,
                                begin=r['created_at'], latitude=r['latitude'], longitude=r['longitude'], geography='point', area=None,
                                state=None, narrative=r['description'][:300], injuries=0, deaths=0, damage_property=None, magnitude=None))
    return records


def _month(value):
    return parse_time(value if '+' in value[10:] or value.endswith('Z') else value+'+00:00').month


def _geography(site, r):
    if r['geography'] == 'point' and r['latitude'] is not None:
        d = distance_miles(site['latitude'], site['longitude'], r['latitude'], r['longitude'])
        return max(0., 1-d/RADIUS_MILES), f'{d:.1f} miles from the work location'
    if r['geography'] == 'county':
        return .6, f"Same county ({r['area']}); NOAA gives no point location"
    return .45, f"NWS zone {r['area']} matches the facility county name; zone records cover a wider area"


def _impact(r):
    if (r.get('injuries') or 0)+(r.get('deaths') or 0) > 0:
        return 1., f"{r.get('injuries') or 0} injuries, {r.get('deaths') or 0} deaths recorded"
    damage = r.get('damage_property')
    if damage and damage not in ('0', '0.00K', '0K'):
        return .6, f'Property damage {damage}'
    return 0., 'No recorded injuries or damage'


def retrieve(site, activity, entries, start, records, limit=6):
    ratings = {e['hazard']: e['rating'] for e in entries}
    sensitivity = ACTIVITIES[activity]['sensitivity']
    mission_month = _month(start)
    results = []
    for r in records:
        geography, where = _geography(site, r)
        rating = ratings.get(r['hazard'])
        delta = abs(_month(r['begin'])-mission_month) % 12
        season = (1+cos(2*pi*min(delta, 12-delta)/12))/2
        impact, harm = _impact(r)
        factors = dict(
            geography=dict(score=round(geography, 3), detail=where),
            conditions=dict(score=CONDITION_SCORE.get(rating, .1),
                            detail=f"Mission JHA rates {r['hazard']} {rating}" if rating else f"No {r['hazard']} entry in the mission JHA"),
            mission=dict(score=sensitivity.get(r['hazard'], .3), detail=f"{ACTIVITIES[activity]['label']} sensitivity to {r['hazard']}"),
            season=dict(score=round(season, 3), detail=f'{min(delta, 12-delta)} months from the shift month'),
            impact=dict(score=impact, detail=harm))
        results.append(dict(record=r, factors=factors, score=round(sum(WEIGHTS[k]*v['score'] for k, v in factors.items()), 3)))
    return sorted(results, key=lambda x: -x['score'])[:limit]
