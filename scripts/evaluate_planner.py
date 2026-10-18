"""Scenario evaluation for the work-window planner; semantics are in docs/WORK_WINDOW_PLANNER.md.

Every scenario is synthetic weather on a fixed grid, so the numbers measure the decision rule
(margin, member agreement, constraints) rather than any historical forecast quality.
"""
import json
from datetime import timedelta

from scripts.common import ROOT
from backend.domain import now_utc
from backend.missions import ACTIVITIES
from backend import planner

OUT = ROOT / 'experiments/sprint-20261017-planner'
HOURS = 72


def grid(start, profile, rain_72h=0., soil=.2, deep=.25):
    """Hourly forecast rows; profile maps hour index to (apparent, gust, rain)."""
    rows = []
    for index in range(HOURS):
        apparent, gust, rain = profile(index)
        rows.append(dict(time=(start + timedelta(hours=index)).isoformat(), apparent_temperature=apparent,
                         wind_gusts_10m=gust, precipitation=rain, soil_moisture_0_to_1cm=soil,
                         soil_moisture_0_to_100cm=deep))
    return dict(source='scenario', forecast=rows, summary=dict(rain_72h_mm=rain_72h), feed=dict(available=True))


def members(times, gust, heat, rain):
    return dict(available=True, model='scenario', times=times,
                members=dict(wind_gusts_10m=[[value]*len(times) for value in gust],
                             apparent_temperature=[[value]*len(times) for value in heat],
                             precipitation=[[value]*len(times) for value in rain]))


def scenarios():
    start = now_utc().replace(minute=0, second=0, microsecond=0)
    times = [(start + timedelta(hours=i)).isoformat() for i in range(HOURS)]
    day = lambda i: i // 24  # noqa: E731

    def hot_then_mild(i):
        return (38.0 if day(i) == 0 else 24.0), 15.0, 0.0

    def rain_front(i):
        return 26.0, 20.0, (14.0 if day(i) == 0 else 1.0)

    def all_windy(i):
        return 24.0, 90.0, 0.0

    def flat(i):
        return 24.0, 20.0, 0.5

    def windy_morning(i):
        return 22.0, (52.0 if (i % 24) < 10 else 18.0), 0.0

    def near_tie(i):
        return 24.0, (30.0 if day(i) == 0 else 28.0), 0.0

    def missing_gust(i):
        return 24.0, None, 0.0

    plan_start = (start + timedelta(hours=8)).isoformat()
    base = dict(duration_hours=8, now=start.isoformat(), hours=HOURS, step_hours=1, planned_start=plan_start,
                timezone='UTC', flex_hours=4)
    calm_members = members(times, [20, 22, 24], [26, 28, 30], [0, 1, 2])
    return [
        dict(name='hot_then_mild_excavation', expect='move', weather=grid(start, hot_then_mild),
             ensemble=calm_members, activity='excavation_trenching', **base),
        dict(name='rain_front_excavation', expect='move', weather=grid(start, rain_front),
             ensemble=calm_members, activity='excavation_trenching', **base),
        dict(name='all_windows_restricted', expect='keep', weather=grid(start, all_windy),
             ensemble=calm_members, activity='aerial_lift_crane', **base),
        dict(name='flat_exposure', expect='keep', weather=grid(start, flat),
             ensemble=calm_members, activity='general_site_work', **base),
        dict(name='windy_morning_calm_evening', expect='move', weather=grid(start, windy_morning),
             ensemble=calm_members, activity='aerial_lift_crane', **base),
        dict(name='no_ensemble_still_moves_on_large_gain', expect='move', weather=grid(start, hot_then_mild),
             ensemble=None, activity='excavation_trenching', **base),
        dict(name='near_tie_declines', expect='keep', weather=grid(start, near_tie),
             ensemble=calm_members, activity='general_site_work', **base),
        dict(name='unknown_exposure_declines', expect='insufficient', weather=grid(start, missing_gust),
             ensemble=None, activity='aerial_lift_crane', **base),
    ]


def run_scenarios():
    records = []
    for scenario in scenarios():
        weather, ensemble = scenario['weather'], scenario['ensemble']
        result = planner.plan(weather, ACTIVITIES[scenario['activity']], scenario['duration_hours'],
                              ensemble=ensemble, now=scenario['now'], hours=scenario['hours'],
                              step_hours=scenario['step_hours'], planned_start=scenario['planned_start'],
                              timezone=scenario['timezone'], flex_hours=scenario['flex_hours'], limit=24,
                              busy=scenario.get('busy', ()))
        choice = result['recommendation']['choice']
        windows = [dict(w, start=w['start'], end=w['end']) for w in result['windows']]
        planned = result['planned']
        usable = sorted([w for w in windows if not w['restricted']], key=lambda w: w['start'])
        same_hour = [w for w in usable if w['start'][11:] == scenario['planned_start'][11:]
                     and w['start'] > scenario['planned_start']]
        naive = dict(keep=planned['risk'] if planned else None,
                     earliest=usable[0]['risk'] if usable else None,
                     later_same_hour=same_hour[0]['risk'] if same_hour else None)
        chosen = next((w for w in result['windows'] if w['start'] == result['recommendation']['start']), None)
        violations = []
        if chosen:
            if chosen['restricted']:
                violations.append('restricted window recommended')
            if planner.hour_distance(planner.local_hour(chosen['start'], scenario['timezone']),
                                     planner.local_hour(scenario['planned_start'], scenario['timezone'])) > scenario['flex_hours'] + 1e-9:
                violations.append('outside the allowed start band')
            if chosen['start'] < scenario['now']:
                violations.append('starts in the past')
        records.append(dict(name=scenario['name'], expect=scenario['expect'], choice=choice, matched=choice == scenario['expect'],
                            reason=result['recommendation']['reason'], improvement=result['recommendation']['improvement'],
                            planned_risk=planned['risk'] if planned else None, chosen_risk=chosen['risk'] if chosen else None,
                            candidate_count=result['candidate_count'], excluded=result['excluded'],
                            near_ties=result['recommendation']['near_ties'], naive=naive, violations=violations,
                            ensemble_available=bool(ensemble and ensemble.get('available'))))
    moved = [r for r in records if r['choice'] == 'move']
    def beats_naive(record):
        values = [v for v in record['naive'].values() if v is not None]
        return bool(values) and record['chosen_risk'] is not None and record['chosen_risk'] <= min(values) + 1e-9
    metrics = dict(scenarios=len(records), matched=sum(1 for r in records if r['matched']),
                   constraint_violations=sum(len(r['violations']) for r in records),
                   moves=len(moved), declined=sum(1 for r in records if r['choice'] == 'keep'),
                   insufficient=sum(1 for r in records if r['choice'] == 'insufficient'),
                   mean_improvement_on_move=round(sum(r['improvement'] or 0 for r in moved)/len(moved), 3) if moved else None,
                   chosen_at_or_better_than_naive=sum(1 for r in moved if beats_naive(r)))
    return dict(metrics=metrics, records=records)


if __name__ == '__main__':
    report = run_scenarios()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    for record in report['records']:
        print(f"{record['name']:38s} expect {record['expect']:12s} got {record['choice']:12s} "
              f"improvement {record['improvement']} violations {len(record['violations'])}")
    print(json.dumps(report['metrics'], sort_keys=True))
