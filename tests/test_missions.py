from datetime import datetime, timedelta, timezone
from backend import missions

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)


def weather(hours=72, apparent=30., gust=20., rain=0., soil=.2, start=NOW):
    rows = [dict(time=(start+timedelta(hours=h)).isoformat().replace('+00:00', 'Z'), apparent_temperature=apparent,
                 wind_gusts_10m=gust, precipitation=rain, soil_moisture_0_to_1cm=soil) for h in range(hours)]
    return dict(forecast=rows, feed=dict(available=True, fetched_at=NOW.isoformat()), summary=dict(rain_72h_mm=0.))


MISSION = dict(id='m1', activity='general_site_work', shift_start=(NOW+timedelta(hours=2)).isoformat(),
               shift_end=(NOW+timedelta(hours=10)).isoformat(), latitude=30., longitude=-82.)
IDLE = dict(active=False, last_strike_at=None, review_after=None, remaining_seconds=0)


def context(mission=MISSION, w=None, **overrides):
    w = w or weather()
    base = dict(now=NOW, conditions=missions.window_conditions(w, mission['shift_start'], mission['shift_end']),
                weather=w, alerts=[], model=dict(available=False, hazards={}), floodplain=dict(zones=[], feed=dict(available=False)),
                reports=[], lightning=dict(live=IDLE, replay=IDLE), live_feed_configured=False)
    return dict(base, **overrides)


def entry(jha, hazard):
    return next(e for e in jha['entries'] if e['hazard'] == hazard)


def test_forecast_heat_and_wind_ratings_cite_the_forecast_and_depend_on_activity():
    hot = missions.build_jha(MISSION, context(w=weather(apparent=40., gust=50.)))
    heat = entry(hot, 'heat')
    assert heat['rating'] == 'high'
    assert heat['evidence'][0]['kind'] == 'predicted' and 'Open-Meteo' in heat['evidence'][0]['source']
    assert heat['evidence'][0]['value'] == 40
    assert entry(hot, 'storm')['rating'] == 'elevated'
    lift = dict(MISSION, activity='aerial_lift_crane')
    assert entry(missions.build_jha(lift, context(lift, w=weather(gust=50.))), 'storm')['rating'] == 'high'
    mild = missions.build_jha(MISSION, context(w=weather(apparent=25., gust=15.)))
    assert entry(mild, 'heat')['rating'] == 'low' and entry(mild, 'storm')['rating'] == 'low'
    assert entry(mild, 'heat')['controls']


def test_unavailable_forecast_is_unknown_never_low():
    late = dict(MISSION, shift_start=(NOW+timedelta(days=6)).isoformat(), shift_end=(NOW+timedelta(days=6, hours=8)).isoformat())
    jha = missions.build_jha(late, context(late))
    assert entry(jha, 'heat')['rating'] == 'unknown' and entry(jha, 'heat')['evidence'][0]['kind'] == 'unavailable'
    assert entry(jha, 'flood')['rating'] == 'unknown'


def hold(minutes_left):
    until = NOW+timedelta(minutes=minutes_left)
    return dict(active=True, last_strike_at=(until-timedelta(minutes=30)).isoformat(), review_after=until.isoformat(),
                remaining_seconds=minutes_left*60)


ACTIVE = dict(MISSION, shift_start=(NOW-timedelta(hours=1)).isoformat())


def test_recorded_strike_timer_is_an_observed_stop_work_that_sets_readiness():
    jha = missions.build_jha(ACTIVE, context(ACTIVE, lightning=dict(live=hold(20), replay=IDLE), live_feed_configured=True))
    lightning = entry(jha, 'lightning')
    assert lightning['rating'] == 'stop_work' and lightning['evidence'][0]['kind'] == 'observed'
    assert lightning['evidence'][0]['summary'] == 'Strike recorded at 11:50 UTC; review after 12:20 UTC'
    assert jha['readiness']['status'] == 'stop_work' and jha['entries'][0]['hazard'] == 'lightning'


def test_replay_timer_is_labeled_simulated_and_does_not_look_like_a_live_hold():
    jha = missions.build_jha(ACTIVE, context(ACTIVE, lightning=dict(live=IDLE, replay=hold(25))))
    lightning = entry(jha, 'lightning')
    assert lightning['rating'] == 'stop_work' and {e['kind'] for e in lightning['evidence']} >= {'simulated', 'unavailable'}
    assert jha['readiness']['status'] == 'simulated_stop_work'
    assert lightning['evidence'][0]['summary'] == 'Simulated strike at 11:55 UTC; scenario timer runs until 12:25 UTC'


def test_missing_strike_feed_is_unknown_and_expired_timers_do_not_reach_a_later_shift():
    assert entry(missions.build_jha(MISSION, context()), 'lightning')['rating'] == 'unknown'
    configured = missions.build_jha(MISSION, context(live_feed_configured=True))
    assert entry(configured, 'lightning')['rating'] == 'low'
    later = missions.build_jha(MISSION, context(lightning=dict(live=hold(20), replay=IDLE), live_feed_configured=True))
    assert entry(later, 'lightning')['rating'] == 'low'


def report(id, status, categories, jev=None, hours_ago=2):
    provider = dict(status='classified', model='jev-1.13.0', categories=jev, review_queue='standard', abstained=False) if jev is not None \
        else dict(status='unavailable', reason='consent not supplied')
    return dict(id=id, status=status, classification=categories[0], description='Water over the access road.',
                created_at=(NOW-timedelta(hours=hours_ago)).isoformat(), latitude=30., longitude=-82.,
                triage=dict(manual_categories=categories, source='manual', provider=provider))


def test_verified_report_is_observed_evidence_and_jev_suggestions_stay_advisory():
    jha = missions.build_jha(MISSION, context(reports=[report('r1', 'verified', ['flood'], jev=['flood', 'wildlife'])]))
    flood = entry(jha, 'flood')
    cited = next(e for e in flood['evidence'] if e.get('ref') == 'r1')
    assert flood['rating'] == 'high' and cited['kind'] == 'observed' and cited['counts']
    wildlife = entry(jha, 'wildlife')
    assert wildlife['rating'] == 'unknown'
    assert [(e['kind'], e['counts'], e['ref']) for e in wildlife['evidence']] == [('ai_suggestion', False, 'r1')]


def test_pending_reports_are_cited_unverified_and_rejected_reports_are_ignored():
    jha = missions.build_jha(MISSION, context(reports=[report('p1', 'pending', ['wildlife']), report('x1', 'rejected', ['flood']),
                                                      report('old', 'verified', ['flood'], hours_ago=60)]))
    wildlife = entry(jha, 'wildlife')
    assert wildlife['rating'] == 'unknown' and wildlife['needs_review']
    assert wildlife['evidence'][0]['kind'] == 'reported' and not wildlife['evidence'][0]['counts']
    assert entry(jha, 'flood')['rating'] == 'low'


def test_nws_alerts_and_county_model_are_cited_and_missing_alert_feed_is_incomplete():
    alert = dict(id='urn:nws:1', event='Severe Thunderstorm Warning', severity='Severe', headline='Severe storms', sent=NOW.isoformat(),
                 expires=(NOW+timedelta(hours=3)).isoformat())
    model = dict(available=True, target_date=(NOW+timedelta(hours=4)).date().isoformat(),
                 hazards=dict(flood=dict(probability=.31, threshold=.2, elevated=True, warning_supported=True, model='original_logistic')))
    jha = missions.build_jha(MISSION, context(alerts=[alert], model=model))
    storm = entry(jha, 'storm')
    assert storm['rating'] == 'high' and any(e['kind'] == 'official_alert' and e['ref'] == 'urn:nws:1' for e in storm['evidence'])
    assert entry(jha, 'lightning')['rating'] == 'elevated'
    flood = next(e for e in entry(jha, 'flood')['evidence'] if e['id'].startswith('model:'))
    assert flood['kind'] == 'predicted' and flood['rating'] == 'elevated' and 'original_logistic' in flood['source']
    assert entry(missions.build_jha(MISSION, context(alerts=None)), 'storm')['incomplete']


def nws(id, event, severity='Moderate', **times):
    return dict(id=id, event=event, severity=severity, sent=NOW.isoformat(), expires=(NOW+timedelta(hours=3)).isoformat(), **times)


def test_every_active_nws_alert_is_cited_and_alerts_outside_the_shift_are_not():
    later = (NOW+timedelta(hours=11)).isoformat()
    jha = missions.build_jha(MISSION, context(alerts=[
        nws('urn:chill', 'Wind Chill Advisory'), nws('urn:winter', 'Winter Storm Warning', 'Severe'),
        nws('urn:over', 'Flood Warning', 'Severe', ends=(NOW+timedelta(hours=1)).isoformat()),
        nws('urn:later', 'Heat Advisory', onset=later, ends=(NOW+timedelta(hours=20)).isoformat())]))
    assert not any(e.get('ref') == 'urn:chill' for e in entry(jha, 'storm')['evidence'])
    other = entry(jha, 'other')
    assert {e['ref'] for e in other['evidence']} == {'urn:chill', 'urn:winter'}
    assert other['rating'] == 'high' and all(e['kind'] == 'official_alert' for e in other['evidence'])
    assert entry(jha, 'flood')['rating'] == 'low' and entry(jha, 'heat')['rating'] == 'low'


def test_model_evidence_states_how_often_a_flag_was_right_in_testing():
    tested = dict(probability=.31, threshold=.2, elevated=True, warning_supported=True, model='original_logistic',
                  tested=dict(precision=.073, recall=.595, period='January to June 2026'))
    model = dict(available=True, target_date=(NOW+timedelta(hours=4)).date().isoformat(), hazards=dict(flood=tested))
    cited = next(e for e in entry(missions.build_jha(MISSION, context(model=model)), 'flood')['evidence'] if e['id'].startswith('model:'))
    assert 'about 7 of every 100 flagged county-days had a NOAA-reported flood' in cited['summary']
    assert 'January to June 2026' in cited['summary']


def test_versions_change_on_band_or_evidence_changes_and_diff_names_them():
    base = missions.build_jha(MISSION, context(w=weather(gust=50.)))
    drift = missions.build_jha(MISSION, context(w=weather(gust=55.)))
    assert base['fingerprint'] == drift['fingerprint']
    assert missions.build_jha(MISSION, context(w=weather(gust=75.)))['fingerprint'] != base['fingerprint']
    verified = missions.build_jha(MISSION, context(w=weather(gust=50.), reports=[report('r1', 'verified', ['flood'])]))
    changes = missions.diff_jha(base['entries'], verified['entries'])
    assert changes == [dict(hazard='flood', change='rating', before='low', after='high',
                            evidence_added=['Crew report verified by supervisor: Water over the access road.'], evidence_removed=[])]
    assert missions.diff_jha([], base['entries'])[0]['change'] == 'added'
    assert missions.diff_jha(base['entries'], base['entries']) == []


def test_shift_window_slices_forecast_and_reports_coverage():
    w = weather(hours=10, apparent=33., gust=41., rain=1.5)
    inside = missions.window_conditions(w, (NOW+timedelta(hours=2)).isoformat(), (NOW+timedelta(hours=6)).isoformat())
    assert inside['hours_requested'] == 4 and inside['hours_available'] == 4 and inside['coverage'] == 1
    assert inside['apparent_max_c'] == 33 and inside['gust_max_kmh'] == 41 and inside['rain_mm'] == 6 and inside['kind'] == 'predicted'
    partial = missions.window_conditions(w, (NOW+timedelta(hours=8)).isoformat(), (NOW+timedelta(hours=12)).isoformat())
    assert partial['hours_available'] == 2 and partial['coverage'] == .5 and partial['kind'] == 'predicted'
    beyond = missions.window_conditions(w, (NOW+timedelta(days=5)).isoformat(), (NOW+timedelta(days=5, hours=8)).isoformat())
    assert beyond['kind'] == 'unavailable' and beyond['apparent_max_c'] is None and beyond['rain_mm'] is None
