"""Replay forecast cycles through the work-window planner and report how the advice moves.

No archived point-forecast cycles ship with this repository and the development sandbox has no
network, so the committed fixture is synthetic and says so in every cycle. A cycle carrying a real
`weather` payload (and optionally an `ensemble`) works the same way once someone freezes one with
`scripts.download` or a provider call. Nothing here fits or tunes anything: the planner is the
served rule, and every metric is a comparison against simple fixed baselines.
"""
import argparse
import json
from datetime import timedelta
from pathlib import Path

from scripts.common import ROOT
from backend import planner
from backend.domain import parse_time
from backend.main import plan_snapshot
from backend.missions import ACTIVITIES

FIXTURE = ROOT / 'tests/fixtures/window-replay.jsonl'
BASELINES = ('keep', 'earliest', 'later_same_hour')
PROFILES = dict(windy_then_calm=(85., 18.), calm_then_windy=(18., 85.), all_windy=(85., 85.))


def load(path):
    """Read one JSON object per cycle; each must declare what it is and what it carries."""
    cycles = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not cycles:
        raise ValueError('Replay file carries no cycles')
    for cycle in cycles:
        if 'cycle' not in cycle or not cycle.get('kind'):
            raise ValueError('Every cycle needs a timestamp and a kind (real-archive or synthetic-fixture)')
        if 'weather' not in cycle and cycle.get('profile') not in PROFILES:
            raise ValueError('Every cycle needs a weather payload or a known synthetic profile')
    return cycles


def synthetic_weather(cycle, hours=72):
    first, second = PROFILES[cycle['profile']]
    start = parse_time(cycle['cycle'])
    rows = [dict(time=(start+timedelta(hours=index)).isoformat(), apparent_temperature=24.0,
                 wind_gusts_10m=first if index < 24 else second, precipitation=0.0,
                 soil_moisture_0_to_1cm=.2, soil_moisture_0_to_100cm=.25) for index in range(hours)]
    return dict(source='synthetic fixture', forecast=rows, summary=dict(rain_72h_mm=0.0),
                feed=dict(available=True, fetched_at=cycle['cycle']))


def advice_digest(cycle, plan, planned_start, hours):
    """The endpoint's snapshot digest, expressed in hours from the cycle so cycles are comparable."""
    def offset(stamp):
        return round((parse_time(stamp) - parse_time(cycle['cycle'])).total_seconds()/3600, 3)
    def window(item):
        return dict(start=offset(item['start']), risk=item['risk'], restricted=item['restricted'], ratings=item['ratings'])
    recommendation = plan['recommendation']
    return plan_snapshot(dict(planned_start=offset(planned_start), horizon_hours=hours,
        duration_hours=plan['duration_hours'], constraints=dict(flex_hours=plan['constraints']['flex_hours']),
        recommendation=dict(choice=recommendation['choice'],
                            start=None if recommendation['start'] is None else offset(recommendation['start'])),
        planned=None if plan['planned'] is None else window(plan['planned']),
        windows=[window(item) for item in plan['windows']]))


def run(cycles, activity='aerial_lift_crane', duration_hours=8, hours=48, step_hours=1, flex_hours=4,
        planned_offset_hours=6):
    records, previous = [], None
    for cycle in cycles:
        weather = cycle.get('weather') or synthetic_weather(cycle)
        planned_start = (parse_time(cycle['cycle']) + timedelta(hours=planned_offset_hours)).isoformat()
        plan = planner.plan(weather, ACTIVITIES[activity], duration_hours, ensemble=cycle.get('ensemble'),
                            now=cycle['cycle'], hours=hours, step_hours=step_hours, planned_start=planned_start,
                            timezone=cycle.get('timezone', 'UTC'), flex_hours=flex_hours, limit=24)
        digest = advice_digest(cycle, plan, planned_start, hours)
        chosen = next((w for w in plan['windows'] if w['start'] == plan['recommendation']['start']), None)
        usable = sorted([w for w in plan['windows'] if not w['restricted']], key=lambda w: w['start'])
        same_hour = [w for w in usable if w['start'][11:19] == planned_start[11:19] and w['start'] > planned_start]
        baselines = dict(keep=plan['planned']['risk'] if plan['planned'] else None,
                         earliest=usable[0]['risk'] if usable else None,
                         later_same_hour=same_hour[0]['risk'] if same_hour else None)
        advice = (plan['recommendation']['choice'],
                  None if plan['recommendation']['start'] is None
                  else round((parse_time(plan['recommendation']['start']) - parse_time(cycle['cycle'])).total_seconds()/3600, 3))
        records.append(dict(cycle=cycle['cycle'], kind=cycle['kind'], digest=digest, advice=advice,
                            choice=plan['recommendation']['choice'], start=plan['recommendation']['start'],
                            planned_risk=plan['planned']['risk'] if plan['planned'] else None,
                            chosen_risk=chosen['risk'] if chosen else None,
                            improvement=plan['recommendation']['improvement'], baselines=baselines,
                            advice_unchanged=previous is not None and advice == previous,
                            reversed_advice=previous is not None and advice != previous))
        previous = advice
    moves = [record for record in records if record['choice'] == 'move']
    def beats(record):
        values = [record['baselines'][name] for name in BASELINES if record['baselines'][name] is not None]
        return bool(values) and record['chosen_risk'] is not None and record['chosen_risk'] <= min(values) + 1e-9
    metrics = dict(cycles=len(records), moves=len(moves),
                   declares=len({record['kind'] for record in records}),
                   reversals=sum(1 for record in records if record['reversed_advice']),
                   advice_unchanged_after_update=sum(1 for record in records if record['advice_unchanged']),
                   abstentions=sum(1 for record in records if record['choice'] == 'insufficient'),
                   restricted_recommendations=sum(1 for record in moves if record['chosen_risk'] is None),
                   mean_improvement_on_move=round(sum(record['improvement'] or 0 for record in moves)/len(moves), 3) if moves else None,
                   moves_at_or_better_than_baselines=sum(1 for record in moves if beats(record)))
    return dict(source=[record['kind'] for record in records], activity=activity,
                planned_offset_hours=planned_offset_hours, metrics=metrics, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', default=str(FIXTURE))
    parser.add_argument('--out', default=None)
    arguments = parser.parse_args()
    report = run(load(arguments.path))
    if arguments.out:
        Path(arguments.out).write_text(json.dumps(report, indent=2) + '\n')
    for record in report['records']:
        print(f"{record['cycle']} {record['kind']:18s} {record['choice']:12s} start {record['start']} "
              f"risk {record['chosen_risk']} reversed {record['reversed_advice']}")
    print(json.dumps(report['metrics'], sort_keys=True))
