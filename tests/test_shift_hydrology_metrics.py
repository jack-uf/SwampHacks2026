import numpy as np
import pandas as pd
import pytest


def decisions():
    return pd.DataFrame(dict(
        station=['a', 'a', 'b', 'b'], huc=['01010001'] * 2 + ['03010001'] * 2,
        decision_time=['2024-01-01T00:00:00Z', '2024-01-01T12:00:00Z'] * 2,
        target=[1, 1, 0, 0], event_time=['2024-01-02T00:00:00Z'] * 2 + ['', ''],
        horizon=[24] * 4))


def test_row_metrics_use_inclusive_threshold_and_all_decisions_for_false_alert_rate():
    from scripts.shift_hydrology_metrics import metrics

    result = metrics(decisions(), [.8, .4, .6, .2], .6)
    assert result['rows'] == 4
    assert result['positives'] == 2
    assert result['ap'] == pytest.approx(5 / 6)
    assert result['brier'] == pytest.approx(.2)
    assert result['precision'] == .5
    assert result['row_recall'] == .5
    assert result['false_alerts_per_100'] == 25.


def test_events_are_unique_by_station_and_time_with_earliest_alert_lead():
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions().iloc[[1, 0, 0, 0]].copy().reset_index(drop=True)
    frame['station'] = ['a', 'a', 'b', 'c']
    frame.loc[2, 'event_time'] = '2024-01-01T06:00:00Z'
    result = metrics(frame, [.9, .5, .8, .1], .5)
    assert result['events'] == 3
    assert result['detected_events'] == 2
    assert result['event_recall'] == pytest.approx(2 / 3)
    assert result['median_lead_hours'] == 15.  # Median of 24h and 6h, one per event.
    # An unalerted event remains in the eligible-population denominator.
    result = metrics(frame, [0, 0, 0, 0], .5)
    assert result['events'] == 3
    assert result['event_recall'] == 0.
    assert result['median_lead_hours'] is None


def test_regions_use_local_denominators_and_episodes_merge_up_to_24_hours():
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions()
    result = metrics(frame, [.8, .4, .6, .2], .6)
    assert result['regions']['01']['false_alerts_per_100'] == 0.
    assert result['regions']['03']['false_alerts_per_100'] == 50.
    assert result['regions']['01']['event_recall'] == 1.
    assert result['regions']['03']['event_recall'] is None
    assert result['regions']['03']['ap'] is None
    assert result['regions']['03']['brier'] == pytest.approx(.2)

    # The two 24h gaps merge transitively; 24h + 1s starts a new episode.
    frame = frame.iloc[[2] * 5].copy().reset_index(drop=True)
    frame['decision_time'] = ['2024-01-03T00:00:00Z', '2024-01-01T00:00:00Z',
                              '2024-01-02T00:00:00Z', '2024-01-04T00:00:01Z',
                              '2024-01-02T00:00:00Z']
    frame.loc[4, 'station'] = 'c'
    result = metrics(frame, np.ones(5), .5)
    assert result['false_warning_episodes'] == 3
    assert result['false_alerts_per_100'] == 100.
    assert metrics(frame, np.zeros(5), .5)['false_warning_episodes'] == 0


@pytest.mark.parametrize('event_time', ['', None, 'not-a-time', '2024-01-01T00:00:00Z',
    '2023-12-31T23:00:00Z', '2024-01-02T00:00:01Z', '2024-01-02T00:00:00',
    '2024-01-02T00:00:00+01:00'])
def test_positive_event_must_be_utc_and_strictly_future_within_horizon(event_time):
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions()
    frame.loc[0, 'event_time'] = event_time
    with pytest.raises(ValueError, match='event_time'):
        metrics(frame, [.8, .4, .6, .2], .5)


@pytest.mark.parametrize('column,value', [
    ('target', 2), ('target', np.nan), ('target', '1'),
    ('horizon', 0), ('horizon', 1.5), ('horizon', np.nan),
    ('station', 12), ('station', ''), ('huc', '0101'), ('huc', 'abcdefgh'),
    ('huc', 1010001), ('decision_time', ''), ('decision_time', '2024-01-01'),
    ('decision_time', '2024-01-01T00:00:00-05:00')])
def test_invalid_common_decision_fields_fail_fast(column, value):
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions().astype({column: object})
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        metrics(frame, [.8, .4, .6, .2], .5)


@pytest.mark.parametrize('p', [[.5], [[.5]] * 4, [.5, np.nan, .2, .1],
                              [.5, np.inf, .2, .1], [.5, -.1, .2, .1], [.5, 1.1, .2, .1]])
def test_invalid_probabilities_fail_fast(p):
    from scripts.shift_hydrology_metrics import metrics

    with pytest.raises(ValueError, match='probabilities'):
        metrics(decisions(), p, .5)


def test_negative_events_must_be_empty_and_station_geography_must_be_consistent():
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions()
    frame.loc[2, 'event_time'] = '2024-01-02T00:00:00Z'
    with pytest.raises(ValueError, match='event_time'):
        metrics(frame, [.8, .4, .6, .2], .5)
    frame = decisions()
    frame.loc[1, 'huc'] = '03010001'
    with pytest.raises(ValueError, match='huc'):
        metrics(frame, [.8, .4, .6, .2], .5)


@pytest.mark.parametrize('threshold', [np.nan, np.inf, -np.inf, [.5, .5]])
def test_threshold_must_be_a_finite_scalar(threshold):
    from scripts.shift_hydrology_metrics import metrics

    with pytest.raises(ValueError, match='threshold'):
        metrics(decisions(), [.8, .4, .6, .2], threshold)


def test_utc_spellings_share_event_identity_and_inputs_are_not_modified():
    from scripts.shift_hydrology_metrics import metrics

    frame = decisions()
    frame.loc[1, 'event_time'] = '2024-01-02T00:00:00+00:00'
    frame.index = [5, 5, 3, 8]
    before = frame.copy(deep=True)
    result = metrics(frame, [.8, .4, .6, .2], .5)
    assert result['events'] == 1
    pd.testing.assert_frame_equal(frame, before)
    # Thresholds just above one are emitted by budget_threshold to suppress all alerts.
    result = metrics(frame, [1, 1, 1, 1], np.nextafter(1., np.inf))
    assert result['precision'] is None
    assert result['row_recall'] == 0.
    assert result['detected_events'] == 0


def test_bootstrap_preserves_event_multiplicity_for_repeated_cluster_draws():
    from scripts.shift_hydrology_metrics import paired_uncertainty

    frame = pd.DataFrame(dict(
        station=['a', 'b', 'c', 'd', 'e', 'f'],
        huc=['01010001', '02010001', '02010001', '03010001', '03010001', '03010001'],
        decision_time=['2024-01-01T00:00:00Z'] * 6, target=[1] * 6,
        event_time=['2024-01-02T00:00:00Z'] * 6, horizon=[24] * 6))
    # Seed 2 draws HUC4 indices [2, 0, 0]: three missed events and two detected copies.
    result = paired_uncertainty(frame, np.zeros(6), [1, 0, 0, 0, 0, 0], .5, .5,
                                draws=1, seed=2)
    assert result['event_recall_difference'] == pytest.approx(1 / 6)
    assert result['event_recall_difference_95'] == [.4, .4]
    assert result['ap_difference'] is None
    assert result['ap_difference_95'] is None
    assert result['valid_ap_draws'] == 0


def test_single_class_resamples_do_not_erase_defined_recall_or_false_alert_draws():
    from scripts.shift_hydrology_metrics import paired_uncertainty

    result = paired_uncertainty(decisions(), [.1] * 4, [.9] * 4, .5, .5,
                                draws=100, seed=42)
    assert result['ap_difference'] == 0.
    assert result['ap_difference_95'] == [0., 0.]
    assert 0 < result['valid_ap_draws'] < result['valid_event_recall_draws'] < 100
    assert result['event_recall_difference'] == 1.
    assert result['event_recall_difference_95'] == [1., 1.]
    assert result['false_alerts_per_100_difference'] == 50.
    assert result['false_alerts_per_100_difference_95'] == [0., 100.]
    assert result['valid_false_alerts_per_100_draws'] == 100


def test_all_negative_population_has_no_ap_or_event_recall_but_has_false_alert_interval():
    from scripts.shift_hydrology_metrics import metrics, paired_uncertainty

    frame = decisions().iloc[2:]
    result = metrics(frame, [.8, .9], .5)
    assert result['ap'] is result['event_recall'] is result['row_recall'] is None
    assert result['events'] == result['detected_events'] == 0
    result = paired_uncertainty(frame, [.1, .2], [.8, .9], .5, .5, draws=5)
    assert result['clusters'] == 1
    assert result['ap_difference_95'] is result['event_recall_difference_95'] is None
    assert result['valid_ap_draws'] == result['valid_event_recall_draws'] == 0
    assert result['false_alerts_per_100_difference_95'] == [100., 100.]


@pytest.mark.parametrize('draws', [0, -1, 1.5, True])
def test_bootstrap_requires_positive_integer_draws(draws):
    from scripts.shift_hydrology_metrics import paired_uncertainty

    with pytest.raises(ValueError, match='draws'):
        paired_uncertainty(decisions(), [.1] * 4, [.9] * 4, .5, .5, draws=draws)


@pytest.mark.parametrize('argument', ['p_control', 'p_candidate'])
def test_bootstrap_validates_both_paired_probability_vectors(argument):
    from scripts.shift_hydrology_metrics import paired_uncertainty

    arguments = dict(p_control=[.1] * 4, p_candidate=[.9] * 4,
                     threshold_control=.5, threshold_candidate=.5)
    arguments[argument] = [.5]
    with pytest.raises(ValueError, match='probabilities'):
        paired_uncertainty(decisions(), **arguments)


def test_common_population_requires_columns_rows_and_unique_decisions():
    from scripts.shift_hydrology_metrics import metrics

    with pytest.raises(ValueError, match='horizon'):
        metrics(decisions().drop(columns='horizon'), [.5] * 4, .5)
    with pytest.raises(ValueError, match='nonempty'):
        metrics(decisions().iloc[:0], [], .5)
    with pytest.raises(ValueError, match='Duplicate'):
        metrics(pd.concat([decisions(), decisions().iloc[:1]]), [.5] * 5, .5)


def test_paired_bootstrap_identical_predictions_have_zero_intervals():
    from scripts.shift_hydrology_metrics import paired_uncertainty

    frame = decisions()
    # Both HUC4 clusters contain both classes, so every AP replicate is defined.
    frame.loc[1, ['target', 'event_time']] = [0, '']
    frame.loc[2, ['target', 'event_time']] = [1, '2024-01-02T00:00:00Z']
    before = frame.copy(deep=True)
    result = paired_uncertainty(frame, [.8, .4, .6, .2], [.8, .4, .6, .2], .5, .5,
                                draws=30, seed=42)
    assert result['clusters'] == 2
    assert result['weak_clusters'] is True
    assert result['draws'] == result['valid_ap_draws'] == result['valid_event_recall_draws'] == 30
    for name in ['ap', 'event_recall', 'false_alerts_per_100']:
        assert result[name + '_difference'] == 0.
        assert result[name + '_difference_95'] == [0., 0.]
    assert result == paired_uncertainty(frame, [.8, .4, .6, .2], [.8, .4, .6, .2],
                                        .5, .5, draws=30, seed=42)
    pd.testing.assert_frame_equal(frame, before)


def test_false_warning_station_days_use_eligible_six_hour_decision_exposure():
    from scripts.shift_hydrology_metrics import metrics

    result = metrics(decisions(), [.8, .4, .6, .2], .6)
    assert result['eligible_decision_exposure_station_days'] == 1.
    assert result['false_warnings_per_station_day'] == 1.
    assert result['station_day_denominator'] == 'eligible_decision_exposure_rows_divided_by_4'
    assert result['regions']['03']['eligible_decision_exposure_station_days'] == .5
    assert result['regions']['03']['false_warnings_per_station_day'] == 2.
    # Station subsets can keep arbitrary input indices and cover widely separated dates.
    frame = decisions().iloc[2:].copy()
    frame.loc[3, 'decision_time'] = '2024-02-01T00:00:00Z'
    result = metrics(frame, [.6, .2], .6)
    assert result['eligible_decision_exposure_station_days'] == .5
    assert result['false_warnings_per_station_day'] == 2.


def test_reliability_bins_keep_boundary_probabilities_and_empty_bins():
    from scripts.shift_hydrology_metrics import metrics

    bins = metrics(decisions(), [0., .1, .9, 1.], .5)['reliability_bins']
    assert len(bins) == 10
    assert bins[0] == dict(lower=0., upper=.1, rows=1, mean_probability=0., observed_rate=1.)
    assert bins[1]['rows'] == 1
    assert bins[1]['mean_probability'] == .1
    assert bins[9]['rows'] == 2
    assert bins[9]['mean_probability'] == pytest.approx(.95)
    assert bins[9]['observed_rate'] == 0.
    assert bins[2]['rows'] == 0
    assert bins[2]['mean_probability'] is bins[2]['observed_rate'] is None


def test_joint_lead_comparison_uses_same_events_and_first_alerts_for_both_models():
    from scripts.shift_hydrology_metrics import jointly_detected_lead_delta

    frame = decisions().iloc[[0, 1, 0, 1, 0]].reset_index(drop=True)
    frame['station'] = ['a', 'a', 'b', 'b', 'c']
    # c is candidate-only and must not affect its conditional lead comparison.
    result = jointly_detected_lead_delta(frame, [1, 1, 0, 1, 0], [0, 1, 1, 1, 1], .5, .5)
    assert result == dict(jointly_detected_events=2, control_median_lead_hours=18.,
                          candidate_median_lead_hours=18., median_lead_delta_hours=0.)
    result = jointly_detected_lead_delta(frame, [0, 1, 0, 1, 0], [1, 1, 1, 1, 1], .5, .5)
    assert result['median_lead_delta_hours'] == 12.
    result = jointly_detected_lead_delta(frame, [1, 1, 0, 0, 0], [0, 0, 1, 1, 1], .5, .5)
    assert result == dict(jointly_detected_events=0, control_median_lead_hours=None,
                          candidate_median_lead_hours=None, median_lead_delta_hours=None)


def test_default_bootstrap_uses_huc4_and_candidate_minus_control_sign():
    from scripts.shift_hydrology_metrics import paired_uncertainty

    frame = decisions()
    frame.loc[1, ['target', 'event_time']] = [0, '']
    frame.loc[2, ['target', 'event_time']] = [1, '2024-01-02T00:00:00Z']
    frame.loc[2:, 'huc'] = '01020001'  # One HUC2 region, two HUC4 clusters.
    result = paired_uncertainty(frame, [.1, .9, .1, .9], [.9, .1, .9, .1], .5, .5)
    assert result['draws'] == result['valid_ap_draws'] == 500
    assert result['clusters'] == 2
    assert result['ap_difference'] == .5
    assert result['ap_difference_95'] == [.5, .5]
    assert result['event_recall_difference_95'] == [1., 1.]
    assert result['false_alerts_per_100_difference_95'] == [-50., -50.]
