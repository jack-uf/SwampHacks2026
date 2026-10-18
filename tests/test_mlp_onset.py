import json

import numpy as np
import pandas as pd
import pytest


def test_cohort_screen_requires_distinct_huc4_and_two_gauges_per_region(tmp_path, monkeypatch):
    import scripts.mlp_onset as research
    registry = [
        dict(site_no='90000001', station_nm='a', huc_cd='01010001', drain_area_va='100', alt_va='10', dec_lat_va='44', dec_long_va='-70'),
        dict(site_no='90000002', station_nm='b', huc_cd='01020001', drain_area_va='100', alt_va='10', dec_lat_va='44', dec_long_va='-70'),
        dict(site_no='90000003', station_nm='c', huc_cd='01030001', drain_area_va='100', alt_va='10', dec_lat_va='44', dec_long_va='-70'),
        dict(site_no='90000004', station_nm='d', huc_cd='02020001', drain_area_va='5', alt_va='10', dec_lat_va='44', dec_long_va='-70'),
        dict(site_no='90000005', station_nm='e', huc_cd='03030001', drain_area_va='100', alt_va='10', dec_lat_va='44', dec_long_va='-70'),
    ]
    monkeypatch.setattr(research, 'REGISTRY', tmp_path / 'sites.json')
    (tmp_path / 'sites.json').write_text(json.dumps(registry))
    monkeypatch.setattr(research, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(research, 'exposed_stations', lambda: {'90000003'})
    monkeypatch.setattr(research, 'digest', lambda path: 'x')
    monkeypatch.setattr(research, 'write_json', lambda path, value: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text(json.dumps(value)))
    (tmp_path / 'docs').mkdir()
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    research.freeze()
    cohort = json.loads((tmp_path / 'out/stations.json').read_text())
    # Region 01 keeps two distinct HUC4s once the excluded gauge is dropped; region 03 has one gauge only.
    assert [s['station'] for s in cohort] == ['90000001', '90000002']
    assert {s['role'] for s in cohort} == {'development', 'geographic_test'}
    assert len({s['huc'] for s in cohort}) == 2


def test_partitions_keep_final_year_and_roles_separate():
    from scripts.mlp_onset_worker import partitions
    frame = pd.DataFrame(dict(
        decision_time=['2019-12-31T18:00:00+00:00', '2020-06-01T00:00:00+00:00', '2021-06-01T00:00:00+00:00',
                        '2022-06-01T00:00:00+00:00', '2023-06-01T00:00:00+00:00', '2024-06-01T00:00:00+00:00',
                        '2024-06-01T00:00:00+00:00', '2025-06-01T00:00:00+00:00', '2025-06-01T00:00:00+00:00'],
        role=['development', 'development', 'development', 'development', 'development', 'development',
              'geographic_test', 'development', 'geographic_test']))
    p = partitions(frame)
    assert p['train'].index.tolist() == [1, 2, 3]
    assert p['calibration'].index.tolist() == [4]
    assert p['temporal_validation'].index.tolist() == [5]
    assert p['geographic_validation'].index.tolist() == [6]
    assert p['temporal_test'].index.tolist() == [7]
    assert p['geographic_test'].index.tolist() == [8]


def test_physical_extrapolation_accounts_for_the_two_hour_delay():
    from scripts.mlp_onset_worker import physical_score
    frame = pd.DataFrame(dict(q_2=[.5], q_8=[.2], horizon=[6]))
    assert physical_score('linear', frame)[0] == pytest.approx(.9)
    assert .5 < physical_score('damped', frame)[0] < .9
    assert physical_score('proximity', frame)[0] == .5


def _frame(rows=400, seed=0, horizon=6):
    generator = np.random.default_rng(seed)
    from scripts.mlp_onset import FEATURES
    frame = pd.DataFrame(generator.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame['head'] = 0
    frame['target'] = (frame.q_2 > frame.q_2.median()).astype(int)
    frame['horizon'] = horizon
    return frame


@pytest.mark.parametrize('architecture', ['mlp_direct', 'mlp_residual'])
def test_mlp_training_is_deterministic_and_bounded(architecture):
    from scripts.mlp_onset_network import fit_network, network_probability, horizon_heads
    from scripts.mlp_onset import FEATURES
    frame = _frame()
    first = fit_network(architecture, FEATURES, horizon_heads(frame, [6]), horizon_heads(frame, [6]))
    second = fit_network(architecture, FEATURES, horizon_heads(frame, [6]), horizon_heads(frame, [6]))
    p1 = network_probability(first, frame, 6)
    p2 = network_probability(second, frame, 6)
    np.testing.assert_allclose(p1, p2)
    assert p1.shape == (len(frame),) and ((0 <= p1) & (p1 <= 1)).all()
    assert first['parameters'] > 0


def test_multitask_routes_each_horizon_to_its_own_head():
    from scripts.mlp_onset_network import fit_network, horizon_heads, network_probability
    from scripts.mlp_onset import FEATURES
    parts = [_frame(seed=index, horizon=horizon) for index, horizon in enumerate([6, 12, 24])]
    stacked = horizon_heads(pd.concat(parts, ignore_index=True))
    bundle = fit_network('mlp_multitask', FEATURES, stacked, stacked)
    outputs = [network_probability(bundle, part, part.horizon.iloc[0]) for part in parts]
    assert not np.allclose(outputs[0], outputs[1])
    assert all(o.shape == (len(parts[0]),) for o in outputs)


def test_every_gate_is_required():
    from scripts.report_mlp_onset import promotion_gates
    pair = dict(candidate=dict(events=30, event_recall=.7, brier=.01, regions={'01': {'false_alerts_per_100': 1}}),
                baseline=dict(events=30, event_recall=.6, brier=.02),
                uncertainty=dict(ap_difference_95=[.01, .05]),
                lead=dict(jointly_detected_events=15, median_lead_delta_hours=0))
    evidence = dict(temporal_test=dict(pair), geographic_test=dict(pair),
        pooled=dict(candidate=dict(events=60, false_alerts_per_100=1),
                    uncertainty=dict(event_recall_difference_95=[.01, .1])))
    assert promotion_gates(evidence)['statistical_pass']
    assert not promotion_gates(evidence)['integration_ready']
    for mutate in [lambda e: e['geographic_test']['candidate']['regions'].update({'17': {'false_alerts_per_100': 3.01}}),
                   lambda e: e['pooled']['uncertainty'].update(event_recall_difference_95=[0, .1]),
                   lambda e: e['temporal_test']['candidate'].update(events=19),
                   lambda e: e['geographic_test']['lead'].update(median_lead_delta_hours=-.5)]:
        changed = dict(temporal_test=dict(pair), geographic_test=dict(pair),
            pooled=dict(candidate=dict(events=60, false_alerts_per_100=1),
                        uncertainty=dict(event_recall_difference_95=[.01, .1])))
        mutate(changed)
        assert not promotion_gates(changed)['statistical_pass']


def test_coverage_keeps_onsets_at_gauges_with_no_eligible_decisions():
    from scripts.report_mlp_onset import coverage
    event = '2025-02-01T06:00:00+00:00'
    frame = pd.DataFrame(dict(station=['one'], decision_time=['2025-02-01T00:00:00+00:00'],
        target=[1], event_time=[event]))
    ledger = [dict(station='one', role='development', events=[event],
                   event_eligibility=[dict(event_time=event, reasons={'6': 'eligible'})]),
              dict(station='omitted', role='development', events=[event],
                   event_eligibility=[dict(event_time=event, reasons={'6': 'missing_history'})])]
    result = coverage(frame, ledger, 6, 'development')
    assert result['all_certified_onsets'] == 2 and result['event_coverage'] == .5
    assert result['eligibility_reasons'] == {'eligible': 1, 'missing_history': 1}


def test_saved_predictions_must_match_canonical_labels(tmp_path, monkeypatch):
    from scripts import report_mlp_onset as report
    monkeypatch.setattr(report, 'OUT', tmp_path)
    (tmp_path / 'predictions').mkdir()
    expected = pd.DataFrame(dict(station=['001'], huc=['01'], decision_time=['2025-02-01T00:00:00+00:00'],
        horizon=[6], target=[1], event_time=['2025-02-01T06:00:00+00:00']))
    saved = expected.assign(probability=.5)
    path = tmp_path / 'predictions' / 'candidate-temporal_test.csv.gz'
    saved.to_csv(path, index=False)
    evaluations = [dict(name='candidate', partition='temporal_test', prediction_sha256=report.digest(path))]
    report.read_predictions({'name': 'candidate'}, 'temporal_test', evaluations, expected)
    saved.loc[0, 'target'] = 0
    saved.to_csv(path, index=False)
    evaluations[0]['prediction_sha256'] = report.digest(path)
    with pytest.raises(AssertionError):
        report.read_predictions({'name': 'candidate'}, 'temporal_test', evaluations, expected)


def test_development_collection_never_requests_the_final_year(tmp_path, monkeypatch):
    import scripts.mlp_onset as research
    from urllib.parse import urlparse, parse_qs
    intervals = []

    def page(path, url):
        intervals.append(parse_qs(urlparse(url).query)['datetime'][0])
        return {'features': [], 'links': []}

    monkeypatch.setattr(research, 'OUT', tmp_path)
    monkeypatch.setattr(research, 'cached_page', page)
    assert research.observations('12345678', 2024).empty
    assert len(intervals) == 4
    assert pd.Timestamp(intervals[-1].split('/')[1]) < pd.Timestamp('2025-01-01T00:00Z')
