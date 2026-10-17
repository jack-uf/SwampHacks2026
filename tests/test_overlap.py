from backend import overlap


def project(pid, entity, lat, lon, month, state='FL', ba='FPL', sector='Electric Utility'):
    return dict(plant_id=pid, plant_name='Plant '+pid, entity_id=entity, entity_name='Entity '+entity, sector=sector,
                state=state, county='X', balancing_authority=ba, latitude=lat, longitude=lon, capacity_mw=100.0,
                planned_month=month, technologies=['Solar Photovoltaic'], statuses=['(U) Under construction'])


def test_close_projects_of_different_entities_overlap_with_factor_breakdown():
    # 0.1 degree of latitude is about 6.9 miles.
    pairs = overlap.find_overlaps([project('1', 'A', 30.0, -82.0, '2027-03'), project('2', 'B', 30.1, -82.0, '2027-05')])
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair['kind'] == 'close_and_concurrent'
    assert 6.5 < pair['distance_miles'] < 7.2 and pair['months_apart'] == 2
    assert set(pair['factors']) == {'proximity', 'timing', 'cross_state', 'shared_balancing_authority'}
    assert pair['factors']['shared_balancing_authority']['score'] == 1
    assert 0 < pair['score'] <= 1


def test_same_entity_far_apart_and_unrelated_timing_are_not_flagged():
    projects = [project('1', 'A', 30.0, -82.0, '2027-03'), project('2', 'A', 30.05, -82.0, '2027-03'),
                project('3', 'C', 34.0, -82.0, '2027-03'),  # ~276 miles away, same month
                project('4', 'D', 30.0, -83.2, '2029-01')]  # ~72 miles away, 22 months later
    assert overlap.find_overlaps(projects) == []


def test_regional_concurrency_and_close_but_not_concurrent_are_labeled():
    near_later = overlap.find_overlaps([project('1', 'A', 30.0, -82.0, '2027-03'), project('2', 'B', 30.2, -82.0, '2029-03')])
    assert near_later[0]['kind'] == 'close'
    regional = overlap.find_overlaps([project('1', 'A', 30.0, -82.0, '2027-03', state='FL'),
                                      project('2', 'B', 30.8, -82.0, '2027-06', state='GA', ba='SOCO')])
    assert regional[0]['kind'] == 'concurrent_in_region' and regional[0]['cross_state'] is True
    assert regional[0]['factors']['cross_state']['score'] == 1


def test_co_owners_of_one_plant_are_not_an_overlap_and_affiliates_are_marked():
    a, b = project('7', 'A', 30.0, -82.0, '2027-03'), project('7', 'B', 30.0, -82.0, '2027-03')
    assert overlap.find_overlaps([a, b]) == []
    duke = [dict(project('1', 'A', 30.0, -82.0, '2027-03'), entity_name='Duke Energy Carolinas, LLC'),
            dict(project('2', 'B', 30.1, -82.0, '2027-03'), entity_name='Duke Energy Progress - (NC)'),
            dict(project('3', 'C', 30.1, -82.1, '2027-03'), entity_name='City of Iola - (KS)')]
    flags = {(p['a']['plant_id'], p['b']['plant_id']): p['possible_affiliates'] for p in overlap.find_overlaps(duke)}
    assert flags == {('1', '2'): True, ('1', '3'): False, ('2', '3'): False}


def test_sector_filter_and_ranking():
    projects = [project('1', 'A', 30.0, -82.0, '2027-03'), project('2', 'B', 30.01, -82.0, '2027-03'),
                project('3', 'C', 30.0, -82.3, '2027-03', sector='IPP Non-CHP')]
    utilities = overlap.find_overlaps(projects, utilities_only=True)
    assert [(p['a']['plant_id'], p['b']['plant_id']) for p in utilities] == [('1', '2')]
    everything = overlap.find_overlaps(projects)
    assert len(everything) == 3 and everything[0]['score'] >= everything[-1]['score']


def test_coordinates_shared_across_states_are_suspect_and_never_compared_by_distance():
    # EIA-860M August 2026 gives McNew (Reno, KS) and Mullin Creek #1 (Nodaway, MO) identical coordinates.
    kansas, missouri = project('1', 'A', 38.0, -97.9, '2030-01', state='KS'), project('2', 'B', 38.0, -97.9, '2030-01', state='MO')
    nearby = project('3', 'C', 38.05, -97.9, '2030-01', state='KS')
    colocated = [project('4', 'D', 30.0, -82.0, '2027-03'), project('5', 'E', 30.0, -82.0, '2027-03')]
    assert overlap.suspect_coordinates([kansas, missouri, nearby]+colocated) == {'1', '2'}
    assert overlap.find_overlaps([kansas, missouri, nearby]) == []
    assert len(overlap.find_overlaps(colocated)) == 1
    assert overlap.near_mission([kansas, missouri, nearby], 38.0, -97.9, '2029-12-01T12:00:00+00:00')[0]['project']['plant_id'] == '3'
    assert len(overlap.near_mission([kansas, missouri, nearby], 38.0, -97.9, '2029-12-01T12:00:00+00:00')) == 1


def test_mission_is_compared_against_public_plans_of_other_entities():
    projects = [project('1', 'A', 30.0, -82.0, '2027-03'), project('2', 'B', 30.1, -82.0, '2026-11'),
                project('3', 'B', 32.0, -82.0, '2026-11')]
    matches = overlap.near_mission(projects, latitude=30.0, longitude=-82.05, start='2026-10-20T12:00:00+00:00', operator='Entity A')
    assert [m['project']['plant_id'] for m in matches] == ['2']
    assert matches[0]['months_apart'] == 1 and matches[0]['distance_miles'] < 8


def test_regional_pairs_are_found_at_arctic_latitudes():
    # At 70.2 N, 3.15 degrees of longitude are about 74 miles and here cross three grid-cell boundaries.
    pairs = overlap.find_overlaps([project('1', 'A', 70.2, -148.90, '2027-03', state='AK', ba='AK'),
                                   project('2', 'B', 70.2, -152.05, '2027-04', state='AK', ba='AK')])
    assert len(pairs) == 1 and pairs[0]['kind'] == 'concurrent_in_region' and 73 < pairs[0]['distance_miles'] < 75
