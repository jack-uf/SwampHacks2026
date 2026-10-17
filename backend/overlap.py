"""Cross-entity overlap of public planned construction (EIA-860M Planned generators)."""
import json
from collections import defaultdict
from functools import lru_cache
from math import floor
from backend.domain import distance_miles, parse_time
from scripts.common import ROOT

CLOSE_MILES = 25
REGION_MILES = 75
CONCURRENT_MONTHS = 6
WEIGHTS = dict(proximity=.45, timing=.35, cross_state=.1, shared_balancing_authority=.1)
PATH = ROOT/'data/planned-projects.json'


@lru_cache(maxsize=1)
def dataset():
    return json.loads(PATH.read_text()) if PATH.exists() else None


def month_index(value):
    year, month = value[:7].split('-')
    return int(year)*12+int(month)-1


def _factors(distance, months, cross_state, shared_ba):
    timing = None if months is None else max(0., 1-months/(2*CONCURRENT_MONTHS))
    factors = dict(
        proximity=dict(score=round(max(0., 1-distance/REGION_MILES), 3), detail=f'{distance:.1f} miles apart'),
        timing=dict(score=0. if timing is None else round(timing, 3),
                    detail='Planned month unknown' if months is None else f'Planned operation {months} months apart'),
        cross_state=dict(score=1. if cross_state else 0., detail='Different states' if cross_state else 'Same state'),
        shared_balancing_authority=dict(score=1. if shared_ba else 0.,
                                        detail='Same balancing authority' if shared_ba else 'Different or unknown balancing authority'))
    return factors, round(sum(WEIGHTS[k]*v['score'] for k, v in factors.items()), 3)


GENERIC = {'city', 'town', 'village', 'county', 'of', 'the', 'public', 'power', 'energy', 'electric', 'utility', 'utilities',
           'municipal', 'board', 'district', 'solar', 'wind', 'storage', 'project', 'llc', 'inc', 'co', 'company'}


def _stem(name):
    words = [w for w in ''.join(c if c.isalnum() else ' ' for c in name.casefold()).split() if w not in GENERIC and len(w) > 2]
    return words[0] if words else None


def possible_affiliates(a, b):
    """Name heuristic only: EIA-860M does not publish parent companies."""
    stem = _stem(a)
    return stem is not None and stem == _stem(b)


def _classify(distance, months):
    close = distance <= CLOSE_MILES
    concurrent = months is not None and months <= CONCURRENT_MONTHS and distance <= REGION_MILES
    if close and concurrent:
        return 'close_and_concurrent'
    return 'close' if close else 'concurrent_in_region' if concurrent else None


def suspect_coordinates(projects):
    """Plant IDs whose exact coordinates also belong to a plant in another state; one point cannot lie in two states."""
    states = defaultdict(set)
    for p in projects:
        states[(p['latitude'], p['longitude'])].add(p['state'])
    return {p['plant_id'] for p in projects if len(states[(p['latitude'], p['longitude'])]) > 1}


def find_overlaps(projects, utilities_only=False):
    """Pairs of plants owned by different entities that are close, or concurrent within a region."""
    suspect = suspect_coordinates(projects)
    projects = [p for p in projects if (not utilities_only or p['sector'] == 'Electric Utility') and p['plant_id'] not in suspect]
    cell = REGION_MILES/69  # degrees of latitude spanned by the regional radius
    grid = defaultdict(list)
    for i, p in enumerate(projects):
        grid[(floor(p['latitude']/cell), floor(p['longitude']/cell))].append(i)
    pairs = []
    for i, a in enumerate(projects):
        row, col = floor(a['latitude']/cell), floor(a['longitude']/cell)
        # Longitude cells shrink with latitude; two neighbours either side covers the continental US.
        for key in [(row+dr, col+dc) for dr in (-1, 0, 1) for dc in (-2, -1, 0, 1, 2)]:
            for j in grid.get(key, []):
                b = projects[j]
                if j <= i or a['entity_id'] == b['entity_id'] or a['plant_id'] == b['plant_id']:
                    continue
                distance = distance_miles(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
                if distance > REGION_MILES:
                    continue
                months = abs(month_index(a['planned_month'])-month_index(b['planned_month'])) if a['planned_month'] and b['planned_month'] else None
                kind = _classify(distance, months)
                if not kind:
                    continue
                cross_state = a['state'] != b['state']
                shared_ba = bool(a['balancing_authority']) and a['balancing_authority'] == b['balancing_authority']
                factors, score = _factors(distance, months, cross_state, shared_ba)
                pairs.append(dict(a=a, b=b, kind=kind, distance_miles=round(distance, 2), months_apart=months,
                                  cross_state=cross_state, possible_affiliates=possible_affiliates(a['entity_name'], b['entity_name']),
                                  factors=factors, score=score))
    return sorted(pairs, key=lambda p: (-p['score'], p['distance_miles']))


def near_mission(projects, latitude, longitude, start, operator=''):
    """Public plans of other entities that are close to, or concurrent near, a supervisor mission."""
    mission_month = month_index(parse_time(start).date().isoformat())
    suspect = suspect_coordinates(projects)
    matches = []
    for p in projects:
        if p['plant_id'] in suspect or (operator and p['entity_name'].casefold() == operator.casefold()):
            continue
        distance = distance_miles(latitude, longitude, p['latitude'], p['longitude'])
        if distance > REGION_MILES:
            continue
        months = abs(month_index(p['planned_month'])-mission_month) if p['planned_month'] else None
        kind = _classify(distance, months)
        if kind:
            factors, score = _factors(distance, months, False, False)
            matches.append(dict(project=p, kind=kind, distance_miles=round(distance, 2), months_apart=months,
                                factors={k: factors[k] for k in ('proximity', 'timing')}, score=score))
    return sorted(matches, key=lambda m: (-m['score'], m['distance_miles']))
