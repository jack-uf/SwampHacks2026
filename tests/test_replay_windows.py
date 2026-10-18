import json

import pytest

from scripts.replay_windows import FIXTURE, load, run


def test_replay_measures_reversals_abstentions_and_refresh_churn():
    cycles = load(FIXTURE)
    assert len(cycles) == 4 and {cycle['kind'] for cycle in cycles} == {'synthetic-fixture'}
    report = run(cycles)
    metrics = report['metrics']
    assert metrics['cycles'] == 4
    assert metrics['reversals'] == 1  # the swap of the windy day changes the advice once
    assert metrics['advice_unchanged_after_update'] == 2  # the identical refresh, and the two declines
    assert metrics['moves'] == 2 and metrics['abstentions'] == 0
    assert metrics['restricted_recommendations'] == 0
    assert metrics['mean_improvement_on_move'] == 1.0
    assert metrics['moves_at_or_better_than_baselines'] == metrics['moves']


def test_replay_refuses_cycles_that_do_not_declare_themselves(tmp_path):
    good = json.loads(FIXTURE.read_text().splitlines()[0])
    without_kind = tmp_path / 'no-kind.jsonl'
    without_kind.write_text(json.dumps({k: v for k, v in good.items() if k != 'kind'}) + '\n')
    with pytest.raises(ValueError, match='kind'):
        load(without_kind)
    without_payload = tmp_path / 'no-payload.jsonl'
    without_payload.write_text(json.dumps(dict(cycle='2026-10-18T00:00:00+00:00', kind='real-archive',
                                               profile='unknown-profile')) + '\n')
    with pytest.raises(ValueError, match='weather payload'):
        load(without_payload)
    empty = tmp_path / 'empty.jsonl'
    empty.write_text('\n')
    with pytest.raises(ValueError, match='no cycles'):
        load(empty)


def test_replay_accepts_a_real_archive_cycle_without_a_profile():
    cycle = json.loads(FIXTURE.read_text().splitlines()[0])
    real = dict(cycle, kind='real-archive', weather=dict(source='archive', forecast=[
        dict(time='2026-10-18T06:00:00+00:00', apparent_temperature=24.0, wind_gusts_10m=18.0, precipitation=0.0,
             soil_moisture_0_to_1cm=.2, soil_moisture_0_to_100cm=.25) for _ in range(48)],
        summary=dict(rain_72h_mm=0.0), feed=dict(available=True, fetched_at=cycle['cycle'])))
    report = run([real])
    assert report['metrics']['cycles'] == 1 and report['records'][0]['kind'] == 'real-archive'
    assert report['metrics']['reversals'] == 0
