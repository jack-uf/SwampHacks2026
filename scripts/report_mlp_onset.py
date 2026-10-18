"""Verify frozen MLP predictions, event coverage and precommitted promotion gates."""
import json
from collections import Counter

import pandas as pd

from scripts.mlp_onset import OUT, digest, write_json, protocol, verify_selection
from scripts.mlp_onset_worker import FAMILIES, load_data, partitions
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty


def promotion_gates(evidence):
    """Every frozen criterion is required; none is relaxed after inspection."""
    gates = {}
    for partition in ['temporal_test', 'geographic_test']:
        pair = evidence[partition]
        candidate, baseline = pair['candidate'], pair['baseline']
        interval = pair['uncertainty']['ap_difference_95']
        gates[partition + '_events'] = candidate['events'] >= 20
        gates[partition + '_ap'] = interval is not None and interval[0] > 0
        gates[partition + '_event_gain'] = (candidate['event_recall'] is not None and
            baseline['event_recall'] is not None and
            candidate['event_recall'] - baseline['event_recall'] >= .05 - 1e-12)
        gates[partition + '_brier'] = candidate['brier'] <= baseline['brier']
        gates[partition + '_regions'] = all(r['false_alerts_per_100'] <= 3 for r in candidate['regions'].values())
        lead = pair['lead']['median_lead_delta_hours']
        gates[partition + '_joint_lead'] = lead is not None and lead >= 0
    pooled = evidence['pooled']
    gates['total_events'] = pooled['candidate']['events'] >= 50
    gates['pooled_false_alerts'] = pooled['candidate']['false_alerts_per_100'] <= 1.5
    interval = pooled['uncertainty']['event_recall_difference_95']
    gates['pooled_event_uncertainty'] = interval is not None and interval[0] > 0
    return dict(gates=gates, statistical_pass=all(gates.values()), integration_ready=False,
        operational_blockers=['No historical telemetry receipt or revision evidence',
                              'No verified construction-site gauge applicability',
                              'High-flow percentile target is not a verified flooding threshold'])


def read_predictions(record, partition, evaluations, expected_labels):
    path = OUT / 'predictions' / (record['name'] + '-' + partition + '.csv.gz')
    expected = next(r['prediction_sha256'] for r in evaluations
                    if r['name'] == record['name'] and r['partition'] == partition)
    if digest(path) != expected:
        raise ValueError('Saved prediction checksum changed')
    saved = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
    pd.testing.assert_frame_equal(saved[expected_labels.columns].reset_index(drop=True),
                                  expected_labels.reset_index(drop=True))
    return saved


def paired_frame(control, candidate):
    columns = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']
    pd.testing.assert_frame_equal(control[columns], candidate[columns])
    frame = control[columns].copy()
    frame['control'] = control.probability.to_numpy()
    frame['candidate'] = candidate.probability.to_numpy()
    return frame


def coverage(frame, ledger, horizon, role):
    represented = frame.loc[frame.target.eq(1), ['station', 'event_time']].drop_duplicates()
    represented_pairs = set(map(tuple, represented.to_numpy()))
    records = [r for r in ledger if r['role'] == role]
    all_events = {(r['station'], event) for r in records for event in r.get('events', [])}
    losses = Counter(event['reasons'][str(horizon)] for r in records for event in r.get('event_eligibility', []))
    expected = {(r['station'], event['event_time']) for r in records for event in r.get('event_eligibility', [])
                if event['reasons'][str(horizon)] == 'eligible'}
    if expected != represented_pairs:
        raise ValueError('Event eligibility diagnostics disagree with evaluated labels')
    return dict(horizon=horizon, all_certified_onsets=len(all_events), eligible_events=len(represented_pairs),
        eligibility_reasons=dict(losses), excluded_gauges=[r['station'] for r in records if 'excluded' in r],
        events_without_eligible_warning_decision=len(all_events - represented_pairs),
        event_coverage=len(represented_pairs) / len(all_events) if all_events else None)


def mlp_comparisons(selected, records, evaluations, canonical, horizon):
    """Frozen-threshold MLP comparisons; the protocol selects on validation, so these are descriptive."""
    per_horizon = selected['selected'][str(horizon)]
    baseline = records[per_horizon['baseline']]
    result = {}
    for kind in ['mlp_direct', 'mlp_residual', 'mlp_multitask']:
        name = f'mlp-{kind}-{horizon}'
        if name not in records:
            continue
        candidate = records[name]
        block = {}
        for partition in ['temporal_test', 'geographic_test']:
            expected = canonical[partition].loc[canonical[partition].horizon.eq(horizon)]
            control = read_predictions(baseline, partition, evaluations, expected)
            test = read_predictions(candidate, partition, evaluations, expected)
            frame = paired_frame(control, test)
            block[partition] = dict(
                threshold=candidate['threshold'], selected_threshold=baseline['threshold'],
                candidate=metrics(frame, frame.candidate, candidate['threshold']),
                baseline=metrics(frame, frame.control, baseline['threshold']),
                uncertainty=paired_uncertainty(frame, frame.control, frame.candidate,
                                               baseline['threshold'], candidate['threshold']),
                lead=jointly_detected_lead_delta(frame, frame.control, frame.candidate,
                                                 baseline['threshold'], candidate['threshold']))
        result[name] = block
    return result


def report():
    protocol()
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final report already recorded')
    selected = verify_selection()
    records = {r['name']: r for r in selected['candidates']}
    labels = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']
    canonical = {key: value[labels] for key, value in partitions(load_data('test')).items()}
    evaluations, summary = [], []
    for family in FAMILIES:
        family_result = json.loads((OUT / (family + '-evaluation.json')).read_text())
        if family_result['selection_sha256'] != digest(OUT / 'selection.json'):
            raise ValueError('Evaluation used a different selection')
        if family_result['dataset_sha256'] != digest(OUT / 'test.csv.gz'):
            raise ValueError('Evaluation used a different test dataset')
        for result in family_result['results']:
            evaluations.append(result)
            expected = canonical[result['partition']]
            read_predictions(records[result['name']], result['partition'], evaluations,
                             expected.loc[expected.horizon.eq(records[result['name']]['horizon'])])
            row = dict(name=result['name'], partition=result['partition'])
            row.update({key: result['metrics'][key] for key in ['rows', 'positives', 'ap', 'brier', 'events',
                'detected_events', 'event_recall', 'median_lead_hours', 'false_alerts_per_100',
                'false_warning_episodes']})
            summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    ledger = json.loads((OUT / 'test-coverage.json').read_text())
    results = {}
    for horizon in [6, 12, 24]:
        pair = selected['selected'][str(horizon)]
        baseline, candidate = records[pair['baseline']], records[pair['candidate']]
        evidence, frames = {}, []
        for partition in ['temporal_test', 'geographic_test']:
            expected = canonical[partition].loc[canonical[partition].horizon.eq(horizon)]
            control = read_predictions(baseline, partition, evaluations, expected)
            test = read_predictions(candidate, partition, evaluations, expected)
            f = paired_frame(control, test)
            frames.append(f)
            evidence[partition] = dict(
                baseline=metrics(f, f.control, baseline['threshold']),
                candidate=metrics(f, f.candidate, candidate['threshold']),
                uncertainty=paired_uncertainty(f, f.control, f.candidate, baseline['threshold'], candidate['threshold']),
                lead=jointly_detected_lead_delta(f, f.control, f.candidate, baseline['threshold'], candidate['threshold']),
                coverage=coverage(f, ledger, horizon, 'development' if partition == 'temporal_test' else 'geographic_test'),
                stations={sid: dict(baseline=metrics(g, g.control, baseline['threshold']),
                                    candidate=metrics(g, g.candidate, candidate['threshold'])) for sid, g in f.groupby('station')})
        pooled = pd.concat(frames, ignore_index=True)
        evidence['pooled'] = dict(baseline=metrics(pooled, pooled.control, baseline['threshold']),
            candidate=metrics(pooled, pooled.candidate, candidate['threshold']),
            uncertainty=paired_uncertainty(pooled, pooled.control, pooled.candidate,
                                           baseline['threshold'], candidate['threshold']))
        evidence.update(selection=pair, promotion=promotion_gates(evidence))
        evidence['mlp_comparisons'] = mlp_comparisons(selected, records, evaluations, canonical, horizon)
        results[str(horizon)] = evidence
        print(horizon, pair, 'statistical pass', evidence['promotion']['statistical_pass'], flush=True)
    write_json(OUT / 'evaluation.json', dict(horizons=results,
        selection_sha256=digest(OUT / 'selection.json'), test_sha256=digest(OUT / 'test.csv.gz')))


if __name__ == '__main__':
    report()
