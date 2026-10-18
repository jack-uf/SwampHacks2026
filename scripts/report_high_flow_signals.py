"""Paired development comparisons for the subdaily signal experiments.

Every candidate is compared with its cycle's control on identical rows, using each
model's own validation-selected threshold. Intervals resample whole HUC4 groups.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.flood_sprint import write_json
from scripts.high_flow_signals import OUT
from scripts.hydrology_research import digest
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty

PARTITIONS = ['temporal_validation', 'geographic_validation']
POOLED_CEILING, REGION_CEILING = 1.5, 3.0


def cycle_records(cycle):
    folder = OUT/f'cycle-{cycle}'
    return folder, {record['name']: record for record in json.loads((folder/'results.json').read_text())}


def predictions(folder, name, partition):
    frame = pd.read_csv(folder/f'{name}-{partition}.csv.gz', dtype={'station': str, 'huc': str}, keep_default_na=False)
    probabilities = frame.pop('probability').to_numpy(float)
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite saved probabilities: '+name)
    return frame, probabilities


def covered(frame, probabilities):
    """Cycle 2's once-daily 12 UTC forecast pins the comparable subset."""
    mask = pd.to_datetime(frame.decision_time, utc=True).dt.hour.eq(12).to_numpy()
    if not mask.any():
        raise ValueError('Empty forecast-covered subset')
    return frame.loc[mask].reset_index(drop=True), probabilities[mask]


def compare(cycle, subset=None, draws=500, reference='control'):
    folder, records = cycle_records(cycle)
    if reference not in records:
        raise ValueError('Cycle has no reference candidate: '+reference)
    control = records[reference]
    result = dict(cycle=cycle, subset=subset or 'all', draws=draws, reference=reference,
        results_sha256=digest(folder/'results.json'), candidates={})
    for name, record in records.items():
        entry = {}
        for partition in PARTITIONS:
            frame, probabilities = predictions(folder, name, partition)
            control_frame, control_probabilities = predictions(folder, reference, partition)
            if subset == 'covered':
                frame, probabilities = covered(frame, probabilities)
                control_frame, control_probabilities = covered(control_frame, control_probabilities)
            if not (frame.station.equals(control_frame.station) and
                    frame.decision_time.equals(control_frame.decision_time)):
                raise ValueError('Candidate rows do not align with control')
            threshold, control_threshold = record['threshold'], control['threshold']
            measured = metrics(frame, probabilities, threshold)
            regions = measured.pop('regions')
            warm = max((value['false_alerts_per_100'] for value in regions.values()), default=0.)
            delta = paired_uncertainty(frame, control_probabilities, probabilities, control_threshold, threshold,
                draws=draws)
            lead = jointly_detected_lead_delta(frame, control_probabilities, probabilities, control_threshold, threshold)
            entry[partition] = dict(measured,
                worst_region_false_alerts_per_100=float(warm),
                worst_region=max(regions, key=lambda key: regions[key]['false_alerts_per_100']) if regions else None,
                false_alert_ceiling_met=bool(measured['false_alerts_per_100'] <= POOLED_CEILING and
                    warm <= REGION_CEILING),
                regions={key: dict(rows=value['rows'], events=value['events'],
                    false_alerts_per_100=value['false_alerts_per_100'], ap=value['ap']) for key, value in regions.items()},
                paired=delta, lead=lead,
                brier_delta=float(measured['brier'] - control['metrics'][partition]['brier'])
                    if subset is None else None)
        frames, probabilities, control_probabilities = [], [], []
        for partition in PARTITIONS:
            frame, values = predictions(folder, name, partition)
            _, reference_values = predictions(folder, reference, partition)
            frames.append(frame)
            probabilities.append(values)
            control_probabilities.append(reference_values)
        pooled = pd.concat(frames, ignore_index=True)
        entry['pooled'] = paired_uncertainty(pooled, np.concatenate(control_probabilities),
            np.concatenate(probabilities), control['threshold'], record['threshold'], draws=draws)
        result['candidates'][name] = entry
    return result


def summarize(result):
    print(f"cycle {result['cycle']} subset={result['subset']} deltas against {result['reference']}")
    for name, entry in result['candidates'].items():
        for partition in PARTITIONS:
            values = entry[partition]
            paired = values['paired']
            def bound(key):
                interval = paired.get(key + '_difference_95')
                return 'n/a' if not interval else f"{interval[0]:+.3f}"
            print(f"  {name:18s} {partition:22s} ev={values['detected_events']:3d}/{values['events']:3d} "
                  f"AP={values['ap']:.3f} FA100={values['false_alerts_per_100']:.3f} "
                  f"worst={values['worst_region']}:{values['worst_region_false_alerts_per_100']:.2f} "
                  f"ceiling={'ok' if values['false_alert_ceiling_met'] else 'FAIL'} "
                  f"lead={values['median_lead_hours']} dAP_low={bound('ap')} dRec_low={bound('event_recall')} "
                  f"dBrier={values['brier_delta'] if values['brier_delta'] is None else round(values['brier_delta'], 6)}")
        pooled = entry['pooled']
        interval = pooled.get('event_recall_difference_95')
        print(f"  {name:18s} {'pooled':22s} dRec={pooled['event_recall_difference']:+.4f} "
              f"low={interval[0]:+.4f} clusters={pooled['clusters']}")


def promotion(cycle, name, reference='linear_conditional', draws=500):
    """Evaluate the protocol's promotion gates for one candidate on this cycle's data."""
    comparison = compare(cycle, None, draws, reference)
    if name not in comparison['candidates']:
        raise ValueError('Unknown candidate: '+name)
    entry = comparison['candidates'][name]
    record = {}
    gates = {'at_least_20_events_per_partition': True, 'at_least_50_events_total': True}
    total = 0
    for partition in PARTITIONS:
        values = entry[partition]
        paired = values['paired']
        low = paired['ap_difference_95']
        recall = paired['event_recall_difference']
        events = values['events']
        total += events
        prefix = partition.replace('_validation', '_test')
        gates[prefix+'_events'] = events >= 20
        gates[prefix+'_ap'] = bool(low and low[0] > 0)
        gates[prefix+'_event_gain'] = bool(recall is not None and recall >= .05)
        gates[prefix+'_brier'] = bool(values['brier_delta'] is not None and values['brier_delta'] <= 0)
        gates[prefix+'_false_alerts'] = bool(values['false_alert_ceiling_met'])
        lead = values['lead']['median_lead_delta_hours']
        gates[prefix+'_lead'] = bool(lead is not None and lead >= 0)
        record[partition] = dict(metrics={key: values[key] for key in
            ['rows', 'positives', 'events', 'detected_events', 'event_recall', 'ap', 'brier',
             'precision', 'row_recall', 'false_alerts_per_100', 'median_lead_hours',
             'false_warning_episodes', 'worst_region', 'worst_region_false_alerts_per_100',
             'false_alert_ceiling_met']}, paired=paired, lead=values['lead'], brier_delta=values['brier_delta'])
    gates['total_events'] = total >= 50
    pooled = entry['pooled']
    interval = pooled.get('event_recall_difference_95')
    gates['pooled_event_uncertainty'] = bool(interval and interval[0] > 0)
    record['pooled'] = pooled
    record['gates'] = gates
    record['passed'] = all(gates.values())
    record['candidate'] = name
    record['reference'] = reference
    record['evidence'] = 'Inspected development partitions. Not independent confirmation.'
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--subset', choices=['covered'], default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--reference', default='control', help='candidate name to compare against')
    parser.add_argument('--promote', default=None, help='candidate name to score against the promotion gates')
    arguments = parser.parse_args()
    if arguments.promote:
        record = promotion(arguments.cycle, arguments.promote, arguments.reference, arguments.draws)
        print(json.dumps(record['gates'], indent=1))
        print('passed', record['passed'])
        if arguments.out:
            write_json(Path(arguments.out), record)
            sys.exit()
        sys.exit()
    comparison = compare(arguments.cycle, arguments.subset, arguments.draws, arguments.reference)
    summarize(comparison)
    if arguments.out:
        write_json(Path(arguments.out), comparison)
