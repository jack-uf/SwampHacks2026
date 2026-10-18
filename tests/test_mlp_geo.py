"""Experiment boundaries; fixtures are synthetic and never enter research data."""
import json

import pytest


def _site(station, huc, area='100'):
    return dict(site_no=station, station_nm=station, huc_cd=huc, drain_area_va=area, alt_va='10',
                dec_lat_va='44', dec_long_va='-70')


@pytest.mark.parametrize('extra, expected_roles', [
    ([_site('90000002', '01020001')], ['development', 'geographic_test']),
    ([], ['development']),
])
def test_one_development_gauge_per_region_and_one_geographic_only_when_a_second_exists(
        extra, expected_roles, tmp_path, monkeypatch):
    import scripts.mlp_geo as research
    registry = [_site('90000001', '01010001')] + extra
    monkeypatch.setattr(research, 'REGISTRY', tmp_path / 'sites.json')
    (tmp_path / 'sites.json').write_text(json.dumps(registry))
    monkeypatch.setattr(research, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(research, 'exposed_stations', lambda: set())
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    monkeypatch.setattr(research, 'digest', lambda path: 'x')
    monkeypatch.setattr(research, 'write_json',
        lambda path, value: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text(json.dumps(value)))
    (tmp_path / 'docs').mkdir()
    research.freeze()
    cohort = json.loads((tmp_path / 'out/stations.json').read_text())
    assert [s['role'] for s in cohort] == expected_roles
    assert len({s['huc'][:4] for s in cohort}) == len(cohort)
    assert {s['huc'][:2] for s in cohort} == {'01'}


def test_registered_stations_are_excluded_from_the_cohort(tmp_path, monkeypatch):
    import scripts.mlp_geo as research
    registry = [_site('90000001', '01010001'), _site('90000009', '01020001')]
    monkeypatch.setattr(research, 'REGISTRY', tmp_path / 'sites.json')
    (tmp_path / 'sites.json').write_text(json.dumps(registry))
    monkeypatch.setattr(research, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(research, 'exposed_stations', lambda: {'90000001'})
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    monkeypatch.setattr(research, 'digest', lambda path: 'x')
    monkeypatch.setattr(research, 'write_json',
        lambda path, value: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text(json.dumps(value)))
    (tmp_path / 'docs').mkdir()
    research.freeze()
    cohort = json.loads((tmp_path / 'out/stations.json').read_text())
    assert [s['station'] for s in cohort] == ['90000009']
    assert [s['role'] for s in cohort] == ['development']


def test_every_candidate_family_receives_both_threshold_rules():
    from scripts.mlp_geo_worker import FAMILIES, RULES
    assert sorted(RULES) == ['ceiling', 'row']
    # The matched comparison is the point of the experiment, so no family may skip a rule.
    assert set(FAMILIES) == {'baseline', 'logistic', 'lightgbm', 'xgboost', 'mlp'}
