import pandas as pd
import pytest

from scripts.revision_worker import folds, gate


def frame():
    rows = []
    windows = {'01-02': 0, '01-08': 0, '03-01': 0, '06-15': 1, '09-01': 1, '12-24': 0, '12-31': 0}
    for role, prefix in [('development', '1000000'), ('geographic_validation', '2000000')]:
        stations = [prefix + str(index) for index in range(1, 4)]
        for year in range(2017, 2023):
            for month_day, exceedance in windows.items():
                for station in stations:
                    rows.append(dict(station=station, huc='0301' + str(year)[-2:], role=role,
                        decision_date=f'{year}-{month_day}', exceedance=exceedance,
                        event_date=f'{year}-{month_day}' if exceedance else '', event_lead_hours=48))
    return pd.DataFrame(rows)


def test_folds_are_forward_only_and_drop_out_of_window_decisions():
    parts = {fold['name']: fold for fold in folds(frame())}
    early = parts['2021']
    assert set(early['train'].decision_date.str[:4]) == {'2017', '2018', '2019'}
    assert set(early['calibration'].decision_date.str[:4]) == {'2020'}
    assert len(set(early['calibration'].decision_date)) == 5
    assert set(early['temporal'].decision_date.str[:4]) == {'2021'}
    assert set(early['temporal'].role) == {'development'}
    assert set(early['geographic'].decision_date.str[:4]) == {'2021'}
    assert set(early['geographic'].role) == {'geographic_validation'}
    late = parts['2022']
    assert set(late['train'].decision_date.str[:4]) == {'2017', '2018', '2019', '2020'}
    assert set(late['calibration'].decision_date.str[:4]) == {'2021'}
    assert set(late['temporal'].decision_date.str[:4]) == {'2022'}
    keys = lambda part: set(zip(part.station, part.decision_date))
    for fold in parts.values():
        seen = set()
        for key in ['train', 'calibration', 'temporal', 'geographic']:
            current = keys(fold[key])
            assert not current & seen
            seen |= current
        assert '2019-01-02' not in set(fold['train'].decision_date)


def test_fresh_outcomes_cannot_enter_fold_selection():
    data = frame()
    data.loc[0, 'role'] = 'fresh_test'
    with pytest.raises(ValueError, match='development populations'):
        folds(data)
    with pytest.raises(ValueError, match='Empty'):
        folds(frame().query('role != "geographic_validation"'))


def records(recall=0.46, region=2.9, ap=0.33, brier=0.03, false=1.2):
    def metrics(value_ap, value_brier, value_recall, value_false):
        return dict(ap=value_ap, brier=value_brier, policies={'pooled_1': dict(
            event_recall=value_recall, false_alerts_per_100=value_false)})
    control = metrics(0.31, 0.031, 0.40, 1.6)
    selected = metrics(ap, brier, recall, false)
    return {name: dict(temporal=control if name == 'control' else selected,
                       geographic=control if name == 'control' else selected,
                       pooled=control if name == 'control' else selected,
                       regions={'03': dict(policies={'pooled_1': dict(false_alerts_per_100=region)}),
                                '12': dict(policies={'pooled_1': dict(false_alerts_per_100=2.0)})})
            for name in ['control', 'bridge']}


def intervals(low=0.001):
    return {scheme: dict(ap_difference_95=[low, .05], event_recall_difference_95=[low, .1]) for scheme in ['huc4', 'month']}


def test_gate_requires_every_precommitted_criterion_and_holds_at_boundaries():
    both = {'2021': records(), '2022': records()}
    passed = gate(both, {fold: intervals() for fold in both}, 'bridge', 'control')
    assert passed['passes'] and all(passed['criteria'].values())
    for change in [dict(recall=0.44), dict(region=3.1), dict(ap=0.30), dict(brier=0.032), dict(false=1.6)]:
        mutated = {'2021': records(**change), '2022': records()}
        result = gate(mutated, {fold: intervals() for fold in mutated}, 'bridge', 'control')
        assert not result['passes'], change
    boundary = {'2021': records(recall=0.45, region=3.0, ap=0.31, brier=0.031, false=1.5), '2022': records()}
    assert gate(boundary, {fold: intervals() for fold in boundary}, 'bridge', 'control')['passes']
    uncertain = gate(both, {fold: intervals(low=-0.002) for fold in both}, 'bridge', 'control')
    assert not uncertain['passes']
    assert not gate(both, {fold: intervals() for fold in both}, 'control', 'control')['passes']
