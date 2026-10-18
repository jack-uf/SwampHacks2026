from scripts.evaluate_planner import run_scenarios


def test_scenario_suite_is_consistent_and_constraint_clean():
    report = run_scenarios()
    metrics = report['metrics']
    assert metrics['constraint_violations'] == 0
    assert metrics['matched'] == metrics['scenarios'] == len(report['records']) == 8
    assert metrics['moves'] >= 3 and metrics['insufficient'] >= 1 and metrics['declined'] >= 2
    assert metrics['mean_improvement_on_move'] > .2
    assert metrics['chosen_at_or_better_than_naive'] == metrics['moves']


def test_every_recommendation_avoids_restricted_windows_and_keeps_the_planned_gain():
    report = run_scenarios()
    for record in report['records']:
        assert not record['violations'], record
        if record['choice'] == 'move':
            assert record['chosen_risk'] is not None and record['improvement'] > 0
            assert record['chosen_risk'] <= (record['planned_risk'] or 0)
        if record['choice'] == 'keep':
            assert record['improvement'] is None or record['improvement'] < record['planned_risk']
