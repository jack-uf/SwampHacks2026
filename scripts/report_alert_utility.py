"""Operational alert utility for one signal-cycle candidate.

Decision-level AP says how well the ranking sorts rows. This report answers the questions a supervisor
would ask: how many issued alerts are correct, how many separate false episodes there are, how often a
station is warned, how much warning time a detected event actually gets, and how the answer changes by
region, season, catchment size and event severity. Episode grouping is frozen here: consecutive alert
decisions at one station belong to the same episode while the gap between them is at most one decision
interval (six hours).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.flood_sprint import write_json
from scripts.high_flow_signals import OUT
from scripts.hydrology_research import ROOT

PARTITIONS = ['temporal_validation', 'geographic_validation']
EPISODE_GAP_HOURS = 6
STRATA_COLUMNS = ['future_max_log_flow', 'log_area', 'p_mean', 'pet_mean']


def episodes(frame, alerts):
    """Maximal alert runs per station; a gap above the decision interval starts a new episode."""
    rows = frame.loc[alerts, ['station', 'decision_time', 'target']].copy()
    rows['decision_time'] = pd.to_datetime(rows.decision_time, utc=True)
    rows = rows.sort_values(['station', 'decision_time'])
    gaps = rows.groupby('station').decision_time.diff()
    rows['episode'] = ((gaps.isna()) | (gaps > pd.Timedelta(hours=EPISODE_GAP_HOURS))).cumsum()
    grouped = rows.groupby('episode')
    summary = grouped.agg(station=('station', 'first'), start=('decision_time', 'min'),
        end=('decision_time', 'max'), alerts=('target', 'size'), positives=('target', 'sum'))
    summary['duration_hours'] = (summary.end - summary.start).dt.total_seconds()/3600 + EPISODE_GAP_HOURS
    summary['false'] = summary.positives.eq(0)
    return summary.reset_index()


def lead_table(frame, alerts):
    hits = frame.loc[alerts & frame.target.eq(1), ['station', 'event_time', 'decision_time']].copy()
    if hits.empty:
        return dict(detected_events=0)
    hits['decision_time'] = pd.to_datetime(hits.decision_time, utc=True)
    hits['event_time'] = pd.to_datetime(hits.event_time, utc=True)
    hits['lead'] = (hits.event_time - hits.decision_time).dt.total_seconds()/3600
    best = hits.groupby(['station', 'event_time']).agg(lead=('lead', 'max'), warnings=('lead', 'size'))
    events = frame.loc[frame.target.eq(1)].drop_duplicates(['station', 'event_time']).shape[0]
    buckets = {'0-2h': int(best.lead.lt(2).sum()), '2-4h': int(best.lead.ge(2).lt(4).sum()),
        '4-6h': int(best.lead.ge(4).sum())}
    return dict(detected_events=int(len(best)), events=events, buckets=buckets,
        share_at_least_4h=float(best.lead.ge(4).mean()),
        median_lead_hours=float(best.lead.median()), mean_warnings_per_detected_event=float(best.warnings.mean()),
        max_warnings_per_detected_event=int(best.warnings.max()))


def reliability(frame, probability):
    edges = np.linspace(0, 1, 11)
    index = np.clip(np.digitize(probability, edges, right=False) - 1, 0, 9)
    y = frame.target.to_numpy(int)
    bins = []
    for position in range(10):
        mask = index == position
        bins.append(dict(lower=float(edges[position]), upper=float(edges[position+1]), rows=int(mask.sum()),
            positives=int(y[mask].sum()),
            mean_probability=float(probability[mask].mean()) if mask.any() else None,
            observed_rate=float(y[mask].mean()) if mask.any() else None))
    finite = (probability > 1e-6) & (probability < 1-1e-6)
    slope = None
    if frame.target.nunique() > 1 and finite.sum() > 10:
        logit = np.log(probability[finite]/(1-probability[finite])).reshape(-1, 1)
        model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, frame.target.to_numpy(int)[finite])
        slope = float(model.coef_[0, 0])
    return dict(calibration_slope=slope, bins=bins,
        brier=float(np.mean((probability - y)**2)))


def strata(frame, alerts, probability):
    frame = frame.copy()
    frame['aridity'] = frame.p_mean/frame.pet_mean.replace(0, np.nan)
    frame['month'] = pd.to_datetime(frame.decision_time, utc=True).dt.month
    frame['season'] = np.select([frame.month.isin([12, 1, 2]), frame.month.isin([3, 4, 5]),
        frame.month.isin([6, 7, 8])], ['DJF', 'MAM', 'JJA'], 'SON')
    frame['alert'] = alerts
    frame['probability'] = probability
    report = {}
    for label, column, bins in [('log_area', 'log_area', 4), ('aridity', 'aridity', 4)]:
        try:
            frame[label+'_band'] = pd.qcut(frame[column], bins, labels=False, duplicates='drop')
        except ValueError:
            continue
        report[label] = frame.groupby(label+'_band').apply(lambda group: dict(
            rows=int(len(group)), events=int(group.target.sum()), alerts=int(group.alert.sum()),
            precision=float(group.loc[group.alert, 'target'].mean()) if group.alert.any() else None,
            false_alerts_per_100=float(100*((group.alert) & (group.target.eq(0))).mean()),
            median_value=float(group[column].median())), include_groups=False).to_dict()
    report['season'] = frame.groupby('season').apply(lambda group: dict(rows=int(len(group)),
        events=int(group.target.sum()), false_alerts_per_100=float(100*((group.alert) & group.target.eq(0)).mean()),
        precision=float(group.loc[group.alert, 'target'].mean()) if group.alert.any() else None), include_groups=False).to_dict()
    report['region'] = frame.groupby(frame.huc.str[:2]).apply(lambda group: dict(rows=int(len(group)),
        events=int(group.target.sum()), false_alerts_per_100=float(100*((group.alert) & group.target.eq(0)).mean()),
        worst_station=float(100*((group.alert) & group.target.eq(0)).groupby(group.station).mean().max()*100)),
        include_groups=False).to_dict()
    positives = frame.loc[frame.target.eq(1)].drop_duplicates(['station', 'event_time'])
    if len(positives) >= 3:
        positives = positives.assign(severity=pd.qcut(positives.future_max_log_flow, 3, labels=['low', 'mid', 'high'], duplicates='drop'))
        report['severity'] = {str(key): dict(events=len(group), median_peak_exceedance=float(group.future_max_log_flow.median()-np.log(2)))
            for key, group in positives.groupby('severity', observed=True)}
    return report


def run(cycle, candidate, destination=None):
    folder = OUT/f'cycle-{cycle}'
    records = {record['name']: record for record in json.loads((folder/'results.json').read_text())}
    if candidate not in records:
        raise ValueError('Unknown candidate: '+candidate)
    threshold = records[candidate]['threshold']
    population = pd.read_csv(OUT/'development.csv.gz', dtype={'station': str, 'huc': str},
        usecols=['station', 'decision_time']+STRATA_COLUMNS, low_memory=False)
    report = dict(cycle=cycle, candidate=candidate, threshold=threshold,
        episode_rule=f'Alert decisions at one station join an episode while the gap is at most {EPISODE_GAP_HOURS} hours.',
        partitions={})
    for partition in PARTITIONS:
        predictions = pd.read_csv(folder/f'{candidate}-{partition}.csv.gz', dtype={'station': str, 'huc': str},
            keep_default_na=False)
        probability = predictions.pop('probability').to_numpy(float)
        frame = predictions.merge(population, on=['station', 'decision_time'], how='left', validate='one_to_one')
        if frame[STRATA_COLUMNS].isna().any(axis=1).any():
            raise ValueError('Missing physical context for scored rows')
        alerts = probability >= threshold
        cohort = episodes(frame, alerts)
        months = pd.to_datetime(frame.decision_time, utc=True).dt.to_period('M').nunique()
        entry = dict(rows=len(frame), events=int(frame.target.eq(1).sum()), alerts=int(alerts.sum()),
            true_alerts=int((alerts & frame.target.eq(1)).sum()), false_alert_decisions=int((alerts & frame.target.eq(0)).sum()),
            alert_precision=float(frame.loc[alerts, 'target'].mean()) if alerts.any() else None,
            row_recall=float(alerts[frame.target.to_numpy() == 1].mean()),
            stations=int(frame.station.nunique()))
        entry['episodes'] = dict(total=int(len(cohort)), false=int(cohort.false.sum()),
            per_station_month=float(len(cohort)/(entry['stations']*months)), months=int(months),
            median_duration_hours=float(cohort.duration_hours.median()),
            max_duration_hours=float(cohort.duration_hours.max()),
            median_alerts_per_episode=float(cohort.alerts.median()),
            worst_region=str(cohort.assign(huc=cohort.station.map(frame.drop_duplicates('station').set_index('station').huc))
                .groupby('huc').size().idxmax()) if len(cohort) else None,
            false_by_region=cohort.loc[cohort.false].assign(
                huc=cohort.loc[cohort.false].station.map(frame.drop_duplicates('station').set_index('station').huc))
                .groupby('huc').size().to_dict())
        entry['lead'] = lead_table(frame, alerts)
        entry['reliability'] = reliability(frame, probability)
        entry['strata'] = strata(frame, alerts, probability)
        report['partitions'][partition] = entry
    if destination:
        write_json(ROOT/destination, report)
    return report


def summarize(report):
    for partition, entry in report['partitions'].items():
        lead = entry['lead']
        print(f"{report['candidate']} {partition}: alerts={entry['alerts']} precision={entry['alert_precision']:.3f} "
              f"false_decisions={entry['false_alert_decisions']} episodes={entry['episodes']['total']} "
              f"(false {entry['episodes']['false']}) median_duration={entry['episodes']['median_duration_hours']:.1f}h")
        if lead.get('detected_events'):
            print(f"   lead buckets {lead['buckets']} median={lead['median_lead_hours']:.1f}h "
                  f"share>=4h={lead['share_at_least_4h']:.3f} warnings/detected={lead['mean_warnings_per_detected_event']:.2f}")
        print(f"   brier={entry['reliability']['brier']:.5f} slope={entry['reliability']['calibration_slope']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', default=None)
    arguments = parser.parse_args()
    summarize(run(arguments.cycle, arguments.candidate, arguments.out))
