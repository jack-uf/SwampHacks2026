"""Metrics for the common eligible decisions in the hourly onset experiment."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def _utc_times(values, name):
    valid = values.map(lambda value: isinstance(value, str))
    if not valid.all() or not values.str.match(r'^\d{4}-\d{2}-\d{2}T.*(?:Z|\+00:00)$').all():
        raise ValueError(f'{name} must contain UTC ISO timestamps')
    parsed = pd.to_datetime(values, format='ISO8601', utc=True, errors='coerce')
    if parsed.isna().any():
        raise ValueError(f'{name} must contain valid UTC ISO timestamps')
    return parsed


def _validated_frame(frame):
    required = {'station', 'huc', 'decision_time', 'target', 'event_time', 'horizon'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Require nonempty common decisions with columns: ' + ', '.join(sorted(required)))
    frame = frame[list(sorted(required))].reset_index(drop=True).copy()
    if not frame.station.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('station must contain nonempty strings')
    if not frame.huc.map(lambda value: isinstance(value, str) and len(value) == 8 and
                         value.isascii() and value.isdigit()).all():
        raise ValueError('huc must contain eight-digit strings')
    if frame.groupby('station').huc.nunique().gt(1).any():
        raise ValueError('A station must have one consistent huc')
    for column in ['target', 'horizon']:
        if not frame[column].map(lambda value: isinstance(value, (int, np.integer)) and
                                 not isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f'{column} must contain integers')
    if not frame.target.isin([0, 1]).all():
        raise ValueError('target must be binary')
    if not frame.horizon.gt(0).all():
        raise ValueError('horizon must contain positive integer hours')
    frame['decision_time'] = _utc_times(frame.decision_time, 'decision_time')
    if frame.duplicated(['station', 'decision_time', 'horizon']).any():
        raise ValueError('Duplicate common decisions for station, decision_time and horizon')
    positive = frame.target.eq(1)
    if not frame.loc[~positive, 'event_time'].eq('').all():
        raise ValueError('Negative target event_time must be empty')
    events = _utc_times(frame.loc[positive, 'event_time'], 'event_time')
    lead = (events - frame.loc[positive, 'decision_time']).dt.total_seconds() / 3600
    if not (lead.gt(0) & lead.le(frame.loc[positive, 'horizon'])).all():
        raise ValueError('Positive event_time must be in (decision_time, decision_time + horizon]')
    frame['event_time'] = pd.to_datetime(frame.event_time, format='ISO8601', utc=True, errors='coerce')
    return frame


def _probabilities(p, rows):
    try:
        p = np.asarray(p, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid probabilities') from error
    if p.shape != (rows,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Require matching one-dimensional finite probabilities in [0, 1]')
    return p


def _threshold(value):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('threshold must be a finite scalar') from error
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError('threshold must be a finite scalar')
    return float(value)


def metrics(frame, p, threshold):
    """Measure positional probabilities at an inclusive threshold, overall and by HUC2.

    Events are unique (station, event_time) pairs in this eligible decision population;
    callers report raw onsets separately. AP requires both classes. Undefined ratios
    and leads are None. Reliability bins are [lower, upper), with 1 included in the
    final bin. Station-days mean rows / 4 at the frozen six-hour decision cadence,
    not full calendar surveillance. False warnings here count rows, not episodes.
    """
    frame = _validated_frame(frame)
    p = _probabilities(p, len(frame))
    threshold = _threshold(threshold)
    result = _summary(frame, p, threshold)
    result['regions'] = {key: _summary(group, p[group.index], threshold)
                         for key, group in frame.groupby(frame.huc.str[:2])}
    return result


def _summary(frame, p, threshold):
    y = frame.target.to_numpy(dtype=int)
    alerts = p >= threshold
    tp = int(np.sum(alerts & (y == 1)))
    fp = int(np.sum(alerts & (y == 0)))
    leads = _event_leads(frame, alerts)
    event_count = len(frame.loc[y == 1].drop_duplicates(['station', 'event_time']))
    false_times = frame.loc[alerts & (y == 0), ['station', 'decision_time']].copy()
    gaps = false_times.sort_values(['station', 'decision_time']).groupby('station').decision_time.diff()
    edges = np.arange(11) / 10
    bins = np.minimum(np.searchsorted(edges, p, side='right') - 1, 9)
    counts = np.bincount(bins, minlength=10)
    probability_sums = np.bincount(bins, weights=p, minlength=10)
    positive_sums = np.bincount(bins, weights=y, minlength=10)
    reliability = [dict(lower=float(edges[i]), upper=float(edges[i + 1]), rows=int(count),
                        mean_probability=float(probability_sums[i] / count) if count else None,
                        observed_rate=float(positive_sums[i] / count) if count else None)
                   for i, count in enumerate(counts)]
    return dict(rows=len(frame), positives=int(y.sum()),
                ap=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, p)),
                precision=tp / int(alerts.sum()) if alerts.any() else None,
                row_recall=tp / int(y.sum()) if y.sum() else None,
                false_alerts_per_100=100 * fp / len(frame),
                eligible_decision_exposure_station_days=len(frame) / 4,
                station_day_denominator='eligible_decision_exposure_rows_divided_by_4',
                false_warnings_per_station_day=fp / (len(frame) / 4),
                reliability_bins=reliability,
                events=event_count, detected_events=len(leads),
                event_recall=len(leads) / event_count if event_count else None,
                median_lead_hours=float(leads.median()) if len(leads) else None,
                false_warning_episodes=int((gaps.isna() | gaps.gt(pd.Timedelta(hours=24))).sum()))


def _event_leads(frame, alerts):
    hits = frame.loc[frame.target.eq(1) & alerts, ['station', 'event_time', 'decision_time']].copy()
    hits['lead'] = (hits.event_time - hits.decision_time).dt.total_seconds() / 3600
    return hits.groupby(['station', 'event_time']).lead.max()


def jointly_detected_lead_delta(frame, p_control, p_candidate, threshold_control, threshold_candidate):
    """Candidate-minus-control median lead on the same jointly detected events.

    Each model uses its earliest alert per event. The delta is the difference of
    medians on that shared set, not the median of per-event lead differences.
    """
    frame = _validated_frame(frame)
    control = _event_leads(frame, _probabilities(p_control, len(frame)) >= _threshold(threshold_control))
    candidate = _event_leads(frame, _probabilities(p_candidate, len(frame)) >= _threshold(threshold_candidate))
    joint = pd.concat([control.rename('control'), candidate.rename('candidate')], axis=1, join='inner')
    control_median = float(joint.control.median()) if len(joint) else None
    candidate_median = float(joint.candidate.median()) if len(joint) else None
    return dict(jointly_detected_events=len(joint), control_median_lead_hours=control_median,
                candidate_median_lead_hours=candidate_median,
                median_lead_delta_hours=candidate_median - control_median if len(joint) else None)


def paired_uncertainty(frame, p_control, p_candidate, threshold_control, threshold_candidate,
                       draws=500, seed=42):
    """Candidate-minus-control deltas with paired HUC4 percentile 95% intervals.

    Each draw samples the original number of HUC4 clusters with replacement.
    Cluster multiplicities weight both unique event counts and detections, preserving
    a separate event identity per sampled occurrence without copying decision rows.
    Undefined AP and event recall draws are excluded separately and counted. Fewer
    than ten clusters sets weak_clusters; that flag is a heuristic, not a guarantee
    of interval coverage at larger counts.
    """
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, (bool, np.bool_)) or draws < 1:
        raise ValueError('draws must be a positive integer')
    frame = _validated_frame(frame)
    control = _probabilities(p_control, len(frame))
    candidate = _probabilities(p_candidate, len(frame))
    alerts_control = control >= _threshold(threshold_control)
    alerts_candidate = candidate >= _threshold(threshold_candidate)
    codes, groups = pd.factorize(frame.huc.str[:4])
    clusters = len(groups)
    y = frame.target.to_numpy(dtype=int)

    def event_counts(mask):
        indices = frame.loc[mask].drop_duplicates(['station', 'event_time']).index
        return np.bincount(codes[indices], minlength=clusters)

    events = event_counts(y == 1)
    hit_difference = event_counts((y == 1) & alerts_candidate) - event_counts((y == 1) & alerts_control)
    false_difference = (np.bincount(codes[(y == 0) & alerts_candidate], minlength=clusters) -
                        np.bincount(codes[(y == 0) & alerts_control], minlength=clusters))

    def differences(counts):
        weights = counts[codes]
        positives = np.dot(weights, y)
        rows = weights.sum()
        ap = None
        if 0 < positives < rows:
            ap = float(average_precision_score(y, candidate, sample_weight=weights) -
                       average_precision_score(y, control, sample_weight=weights))
        event_total = np.dot(counts, events)
        return (ap, float(np.dot(counts, hit_difference) / event_total) if event_total else None,
                float(100 * np.dot(counts, false_difference) / rows))

    result = dict(clusters=clusters, weak_clusters=clusters < 10, draws=int(draws))
    point = differences(np.ones(clusters, dtype=int))
    rng = np.random.default_rng(seed)
    samples = [differences(np.bincount(rng.integers(clusters, size=clusters), minlength=clusters))
               for _ in range(draws)]
    for i, name in enumerate(['ap', 'event_recall', 'false_alerts_per_100']):
        values = [sample[i] for sample in samples if sample[i] is not None]
        result[name + '_difference'] = point[i]
        result[name + '_difference_95'] = np.quantile(values, [.025, .975]).tolist() if values else None
        result['valid_' + name + '_draws'] = len(values)
    return result
