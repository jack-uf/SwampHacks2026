"""Verify frozen geographic-study predictions, coverage and precommitted promotion gates."""
import json

import pandas as pd

from scripts.mlp_geo import OUT, digest, protocol, verify_selection, write_json
from scripts.mlp_geo_worker import FAMILIES, load_data
from scripts.mlp_hybrid_worker import partitions
from scripts.report_mlp_onset import coverage, paired_frame, promotion_gates
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty


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


def matched_comparison(selected, records, evaluations, canonical, horizon):
    """Every candidate at both rules against the selected baseline, so a rule artifact is visible."""
    baseline = records[selected['selected'][str(horizon)]['baseline']]
    result = {}
    for name, record in records.items():
        if record['horizon'] != horizon or name == baseline['name']:
            continue
        block = {}
        for partition in ['temporal_test', 'geographic_test']:
            expected = canonical[partition].loc[canonical[partition].horizon.eq(horizon)]
            control = read_predictions(baseline, partition, evaluations, expected)
            test = read_predictions(record, partition, evaluations, expected)
            frame = paired_frame(control, test)
            block[partition] = dict(rule=record['rule'],
                candidate=metrics(frame, frame.candidate, record['threshold']),
                baseline=metrics(frame, frame.control, baseline['threshold']),
                uncertainty=paired_uncertainty(frame, frame.control, frame.candidate,
                                               baseline['threshold'], record['threshold']))
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
            row = dict(name=result['name'], partition=result['partition'], rule=records[result['name']]['rule'])
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
            frame = paired_frame(control, test)
            frames.append(frame)
            evidence[partition] = dict(
                baseline=metrics(frame, frame.control, baseline['threshold']),
                candidate=metrics(frame, frame.candidate, candidate['threshold']),
                uncertainty=paired_uncertainty(frame, frame.control, frame.candidate,
                                               baseline['threshold'], candidate['threshold']),
                lead=jointly_detected_lead_delta(frame, frame.control, frame.candidate,
                                                 baseline['threshold'], candidate['threshold']),
                coverage=coverage(frame, ledger, horizon, 'development' if partition == 'temporal_test' else 'geographic_test'),
                stations={sid: dict(baseline=metrics(g, g.control, baseline['threshold']),
                                    candidate=metrics(g, g.candidate, candidate['threshold'])) for sid, g in frame.groupby('station')})
        pooled = pd.concat(frames, ignore_index=True)
        evidence['pooled'] = dict(baseline=metrics(pooled, pooled.control, baseline['threshold']),
            candidate=metrics(pooled, pooled.candidate, candidate['threshold']),
            uncertainty=paired_uncertainty(pooled, pooled.control, pooled.candidate,
                                           baseline['threshold'], candidate['threshold']))
        evidence.update(selection=pair, promotion=promotion_gates(evidence))
        evidence['matched_comparison'] = matched_comparison(selected, records, evaluations, canonical, horizon)
        results[str(horizon)] = evidence
        print(horizon, pair, 'statistical pass', evidence['promotion']['statistical_pass'], flush=True)
    write_json(OUT / 'evaluation.json', dict(horizons=results,
        selection_sha256=digest(OUT / 'selection.json'), test_sha256=digest(OUT / 'test.csv.gz')))


if __name__ == '__main__':
    report()
