"""Pilot boundaries; fixtures are synthetic and never enter experiment data."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import multimodal_sprint
from scripts import jev_image


def fake_jev_payload():
    return dict(
        model='jev-1.13.0',
        answers=dict(
            water_visible=dict(type='noul', noul=.19),
            water_extent=dict(type='score', score=.2, legend={str(i): text for i, text in enumerate(jev_image.LEGEND)},
                              probabilities={'0': .8, '1': .19, '2': .01}, confidence=.69),
            ground_condition=dict(type='choice', choice='dry_bare',
                                  probabilities={'dry_bare': 1., 'vegetated': 0., 'water_dominant': 0., 'cloud_or_masked': 0.},
                                  confidence=1.)),
        usage=dict(input_tokens=1465, output_tokens=89))


def test_jev_image_interpret_accepts_the_recorded_shape():
    result = jev_image.interpret(fake_jev_payload(), 'jev-1.13.0')
    assert result['water_visible'] == .19 and result['water_extent'] == .2
    assert result['ground_condition'] == 'dry_bare' and result['usage']['input_tokens'] == 1465


@pytest.mark.parametrize('mutate', [
    lambda payload: payload.update(model='jev-9.9.9'),
    lambda payload: payload['answers']['water_extent'].update(score=2.5),
    lambda payload: payload['answers']['water_extent'].update(legend={'0': 'different'}),
    lambda payload: payload['answers']['ground_condition'].update(choice='flooded'),
    lambda payload: payload['answers']['ground_condition']['probabilities'].update(dry_bare=.5),
])
def test_jev_image_interpret_rejects_malformed_answers(mutate):
    payload = fake_jev_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        jev_image.interpret(payload, 'jev-1.13.0')


def test_jev_image_keeps_a_score_that_disagrees_with_its_probabilities():
    payload = fake_jev_payload()
    payload['answers']['water_extent']['score'] = .9          # probabilities imply 0.21
    result = jev_image.interpret(payload, 'jev-1.13.0')
    assert result['water_extent'] == .9 and round(result['score_gap'], 2) == .69


def test_feature_sets_keep_the_weather_only_candidate_clean():
    sets = multimodal_sprint.feature_sets()
    assert multimodal_sprint.SAT_FEATURES[0] not in sets['weather_jev_xgboost']
    assert multimodal_sprint.JEV_FEATURES[0] not in sets['weather_sat_xgboost']
    for name, features in sets.items():
        assert len(features) == len(set(features)), name
    assert sets['weather_xgboost'] == list(multimodal_sprint.EXTENDED)
    combined = sets['weather_sat_jev_xgboost']
    assert all(feature in combined for feature in multimodal_sprint.SAT_FEATURES + multimodal_sprint.JEV_FEATURES)


def test_prepare_frames_refuses_cross_role_stations(tmp_path, monkeypatch):
    monkeypatch.setattr(multimodal_sprint, 'OUT', tmp_path)
    rows = []
    for station, role, day in [('DEV', 'development', '2026-08-01'), ('DEV', 'development', '2026-09-15'),
                               ('GEO', 'geographic_test', '2026-09-15')]:
        row = dict(station=station, role=role, target_date=day, exceedance=0)
        row.update({feature: 1.0 for feature in multimodal_sprint.EXTENDED})
        row.update({feature: 1.0 for feature in multimodal_sprint.SAT_FEATURES + multimodal_sprint.JEV_FEATURES})
        rows.append(row)
    table = pd.DataFrame(rows)
    summary = multimodal_sprint.prepare_frames('weather_xgboost', table)
    assert summary == dict(train=1, test=1, geo_test=1, positives=[0, 0, 0])
    crossed = pd.DataFrame(rows + [dict(rows[-1], target_date='2026-08-15')])
    with pytest.raises(ValueError, match='geographic split'):
        multimodal_sprint.prepare_frames('weather_xgboost', crossed)


def test_matched_threshold_tracks_the_baseline_budget():
    y = np.array([1, 1] + [0] * 8)
    scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    assert multimodal_sprint.matched_threshold(scores, y, .25) == .6
    assert multimodal_sprint.matched_threshold(scores, y, 0) > .9


def test_satellite_rows_keep_missing_indicators(tmp_path, monkeypatch):
    features = tmp_path / 'features'
    (features / 's2').mkdir(parents=True)
    (features / 'jev').mkdir(parents=True)
    monkeypatch.setattr(multimodal_sprint, 'OUT', tmp_path)
    monkeypatch.setattr(multimodal_sprint.wind_sprint, 'load_protocol',
                        lambda: dict(stations=[dict(sid='DEV', role='development')]))
    s2 = dict(item='S2X_20260801', days_since=3, scene_cloud_cover=5., local_cloud_fraction=0., valid_fraction=1.,
              ndvi_mean=.4, ndvi_p10=.1, ndvi_p90=.6, ndwi_mean=-.3, ndwi_p10=-.5, ndwi_p90=-.1,
              water_fraction=.02, vegetation_fraction=.5, bare_fraction=.3)
    (features / 's2' / 'DEV.json').write_text(json.dumps(dict(**{'2026-08-01': s2},
                                                               **{'2026-08-02': dict(station='DEV', decision_date='2026-08-02', item=None, missing=True)})))
    jev = dict(fake_jev_payload())
    jev.update(station='DEV', item='S2X_20260801', day='2026-08-01', status='ok',
               water_visible=.19, water_extent=.2, water_extent_probabilities={'0': .8, '1': .19, '2': .01},
               water_extent_confidence=.69, ground_condition='dry_bare', ground_confidence=1.,
               ground_probabilities={'dry_bare': 1., 'vegetated': 0., 'water_dominant': 0., 'cloud_or_masked': 0.})
    (features / 'jev' / 'DEV_S2X_20260801.json').write_text(json.dumps(jev))
    frame = multimodal_sprint.satellite_rows().sort_values('target_date').reset_index(drop=True)
    assert list(frame.target_date) == ['2026-08-01', '2026-08-02']
    first, second = frame.iloc[0], frame.iloc[1]
    assert first.s2_missing == 0 and first.s2_ndvi_mean == .4 and first.jev_missing == 0
    assert first.jev_water_visible == .19 and first.jev_confidence == .69
    assert second.s2_missing == 1 and second.jev_missing == 1
    assert np.isnan(second.s2_ndvi_mean)

