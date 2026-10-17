from backend import memory

SITE = dict(id='s1', latitude=30., longitude=-82., county='Alachua', state='FL')


def record(id, hazard, lat, lon, begin, injuries=0, geography='point'):
    return dict(id=id, source='NOAA Storm Events', event_type=hazard.title(), hazard=hazard, begin=begin, latitude=lat, longitude=lon,
                geography=geography, area='ALACHUA', state='FL', narrative='Test narrative.', injuries=injuries, deaths=0,
                damage_property='0.00K', magnitude=None)


def test_retrieval_ranks_by_explained_factors():
    records = [record('hail', 'storm', 30.25, -82., '2025-01-10T15:00:00'),  # ~17 miles, opposite season
               record('flood', 'flood', 30.04, -82., '2025-07-02T15:00:00', injuries=2),
               record('heat', 'heat', None, None, '2025-07-20T15:00:00', geography='zone')]
    entries = [dict(hazard='flood', rating='high'), dict(hazard='storm', rating='low'), dict(hazard='heat', rating='unknown')]
    results = memory.retrieve(SITE, 'excavation_trenching', entries, '2026-07-15T12:00:00+00:00', records)
    assert [r['record']['id'] for r in results] == ['flood', 'heat', 'hail']
    top = results[0]
    assert set(top['factors']) == {'geography', 'conditions', 'mission', 'season', 'impact'}
    assert top['factors']['conditions']['score'] == 1 and 'high' in top['factors']['conditions']['detail']
    assert top['factors']['impact']['score'] == 1 and top['factors']['season']['score'] > .99
    assert 'miles' in top['factors']['geography']['detail']
    assert results[1]['factors']['geography']['detail'].startswith('NWS zone')
    assert all(0 <= r['score'] <= 1 for r in results)


def test_verified_crew_reports_join_memory_and_rejected_do_not():
    reports = [dict(id='r1', status='resolved', classification='flood', description='Water pooled at gate.', latitude=30.01, longitude=-82.,
                    created_at='2026-05-01T12:00:00+00:00', triage=dict(manual_categories=['flood', 'unstable_terrain'])),
               dict(id='r2', status='rejected', classification='flood', description='False alarm.', latitude=30., longitude=-82.,
                    created_at='2026-05-01T12:00:00+00:00', triage=None)]
    records = memory.crew_records(reports)
    assert [(r['id'], r['hazard']) for r in records] == [('report:r1', 'flood'), ('report:r1', 'unstable_terrain')]
    assert records[0]['source'] == 'PlanIT crew report (resolved)'
