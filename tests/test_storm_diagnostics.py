"""Small checks for the diagnostic's AP accounting, label joins and evidence guards."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from scripts.diagnose_storm import overlapping_storms, positive_ranks, verify_metrics


@pytest.mark.parametrize(('y', 'p'), [([1, 0, 1, 0], [.8, .8, .3, .1]),
                                    ([0, 0], [.5, .5]), ([1, 1], [.3, .7])])
def test_positive_rank_contributions_match_ap_with_ties_and_empty_positives(y, p):
    result = positive_ranks(y, p)
    expected = average_precision_score(y, p) if sum(y) else 0
    assert sum(r['ap_contribution'] for r in result) == pytest.approx(expected)
    if sum(y) == 2 and y[0] == 1 and len(y) == 4:
        assert result[0]['rank_min'] == 1 and result[0]['rank_max'] == 2
        assert result[0]['tied_rows'] == 2
        assert result[0]['precision_at_score'] == .5


def test_positive_ranks_reject_invalid_labels_scores_and_alignment():
    for labels, scores in [([1], [.2, .3]), ([2], [.2]), ([1], [np.nan])]:
        with pytest.raises(ValueError, match='aligned binary'):
            positive_ranks(labels, scores)


def test_noaa_join_retains_each_record_and_matches_closed_midnight_and_county():
    events = pd.DataFrame([
        dict(EVENT_ID=1, STATE_FIPS=6, CZ_FIPS=37, CZ_TYPE='C', EVENT_TYPE='Thunderstorm Wind',
             start_utc='2026-02-15 23:59', end_utc='2026-02-16 00:00'),
        dict(EVENT_ID=2, STATE_FIPS=6, CZ_FIPS=37, CZ_TYPE='C', EVENT_TYPE='Hail',
             start_utc='2026-02-16 12:00', end_utc='2026-02-16 12:00'),
        dict(EVENT_ID=3, STATE_FIPS=6, CZ_FIPS=37, CZ_TYPE='C', EVENT_TYPE='Tornado',
             start_utc='2026-02-17 00:00', end_utc='2026-02-17 00:00'),
        dict(EVENT_ID=4, STATE_FIPS=6, CZ_FIPS=38, CZ_TYPE='C', EVENT_TYPE='Thunderstorm Wind',
             start_utc='2026-02-16 12:00', end_utc='2026-02-16 12:00'),
        dict(EVENT_ID=5, STATE_FIPS=6, CZ_FIPS=37, CZ_TYPE='Z', EVENT_TYPE='High Wind',
             start_utc='2026-02-16 12:00', end_utc='2026-02-16 12:00')])
    for column in ['start_utc', 'end_utc']:
        events[column] = pd.to_datetime(events[column])
    row = pd.Series(dict(county_fips=6037, target_date='2026-02-16'))
    assert overlapping_storms(events, row).EVENT_ID.tolist() == [1, 2]


def test_saved_metrics_guard_tolerates_roundoff_but_rejects_changed_artifacts():
    verify_metrics({'ap': .15 + 1e-13, 'bins': [None, 2]}, {'ap': .15, 'bins': [None, 2]})
    with pytest.raises(ValueError, match='numeric mismatch'):
        verify_metrics({'ap': .13}, {'ap': .15})
    with pytest.raises(ValueError, match='length mismatch'):
        verify_metrics([1], [1, 2])
