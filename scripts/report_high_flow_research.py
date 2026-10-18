"""Audit retained development predictions and compare identical observations across research cycles."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.flood_sprint import write_json
from scripts.high_flow_research import OUT, ROOT, load, predict, split
from scripts.hydrology_research import digest
from scripts.shift_hydrology_metrics import metrics, paired_uncertainty, jointly_detected_lead_delta

KEYS = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']
PARTITIONS = ['temporal_validation', 'geographic_validation']


def prediction(cycle, name, partition):
    return pd.read_csv(OUT / 'predictions' / f'cycle-{cycle}-{name}-{partition}.csv.gz',
        dtype={'station': str, 'huc': str}, keep_default_na=False)


def common_predictions(frames):
    """Join on identity and then certify that labels and geography match."""
    identity = ['station', 'decision_time', 'horizon']
    common = frames[0][identity]
    for f in frames[1:]:
        common = common.merge(f[identity], on=identity, validate='one_to_one')
    if common.empty:
        raise ValueError('No common validation decisions')
    result = [common.merge(f, on=identity, validate='one_to_one') for f in frames]
    if any(not result[0][KEYS].equals(f[KEYS]) for f in result[1:]):
        raise ValueError('Common prediction labels or geography differ')
    return result


def two_way_uncertainty(frame, candidate, control, candidate_cut, control_cut, draws=500):
    """Sensitivity to shared weather: paired HUC4 and seven-day UTC block weights.

    At this six-hour cadence each positive is one unique eligible event. Product
    weights sample both drainage groups and time blocks, retaining shared storms.
    These percentile intervals remain exploratory and assume exchangeable blocks.
    """
    if not frame.horizon.eq(6).all() or frame.loc[frame.target.eq(1)].duplicated(['station', 'event_time']).any():
        raise ValueError('Two-way sensitivity requires one decision per six-hour event')
    metrics(frame, candidate, candidate_cut)
    metrics(frame, control, control_cut)
    y = frame.target.to_numpy(int)
    candidate, control = np.asarray(candidate), np.asarray(control)
    spatial, spatial_groups = pd.factorize(frame.huc.str[:4])
    times = pd.to_datetime(frame.decision_time, utc=True)
    temporal, temporal_groups = pd.factorize(times.astype('int64') // (7 * 86400 * 10**9))
    nc, nt = len(spatial_groups), len(temporal_groups)
    rng, samples = np.random.default_rng(42), []
    for _ in range(draws):
        a = np.bincount(rng.integers(nc, size=nc), minlength=nc)
        b = np.bincount(rng.integers(nt, size=nt), minlength=nt)
        w = a[spatial] * b[temporal]
        positives, rows = np.dot(w, y), w.sum()
        if not 0 < positives < rows:
            continue
        hits = (candidate >= candidate_cut).astype(int) - (control >= control_cut).astype(int)
        samples.append([average_precision_score(y, candidate, sample_weight=w) -
            average_precision_score(y, control, sample_weight=w),
            np.dot(w * y, hits) / positives, 100 * np.dot(w * (1-y), hits) / rows])
    return dict(spatial_clusters=nc, weekly_blocks=nt, draws=draws, valid_draws=len(samples),
        intervals_95={key: np.quantile(np.asarray(samples)[:, i], [.025, .975]).tolist() if samples else None
                      for i, key in enumerate(['ap', 'event_recall', 'false_alerts_per_100'])})


def sensitivity():
    path = OUT / 'shared-weather-sensitivity.json'
    if path.exists():
        raise FileExistsError(path)
    records = json.loads((OUT / 'cycle-4.json').read_text())['records']
    selected = {name: next(r for r in records if r['name'] == name and r['policy'] == 'headroom')
                for name in ['expanded_original', 'linear-conditional']}
    results, pooled = {}, []
    for partition in PARTITIONS:
        a, b = common_predictions([prediction(4, name, partition) for name in selected])
        pooled.append(a.assign(control_probability=b.probability))
        results[partition] = two_way_uncertainty(a, a.probability, b.probability,
            selected['expanded_original']['threshold'], selected['linear-conditional']['threshold'])
    f = pd.concat(pooled, ignore_index=True)
    results['pooled'] = two_way_uncertainty(f, f.probability, f.control_probability,
        selected['expanded_original']['threshold'], selected['linear-conditional']['threshold'])
    write_json(path, results)


def independence_audit():
    from scripts.mlp_onset import exposed_stations
    registry = ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json'
    sites, used = json.loads(registry.read_text()), exposed_stations()
    prior = {s['huc_cd'][:4] for s in sites if s['site_no'] in used and len(s.get('huc_cd', '')) >= 8}
    eligible = []
    for s in sites:
        try:
            if len(s.get('huc_cd', '')) >= 8 and 20 < float(s['drain_area_va']) < 1200 and np.isfinite(float(s['alt_va'])):
                eligible.append(s)
        except (ValueError, KeyError, TypeError):
            continue
    fresh = [s for s in eligible if s['site_no'] not in used]
    return dict(registry_sha256=digest(registry), exposed_station_ids=len(used),
        eligible_metadata_stations=len(eligible), unused_station_ids=len(fresh),
        unused_stations_in_unexposed_huc4=[s['site_no'] for s in fresh if s['huc_cd'][:4] not in prior],
        prior_huc4=len(prior),
        note='Metadata pool only. Reused HUC4 can contain disjoint basins; no complete prior polygon exclusion audit exists for this study.')


def decision():
    """Lock a development reference and apply the existing gates without calling it a final test."""
    from scripts.report_mlp_onset import promotion_gates
    path = OUT / 'decision.json'
    if path.exists():
        raise FileExistsError(path)
    summary = json.loads((OUT / 'summary.json').read_text())
    records = json.loads((OUT / 'cycle-4.json').read_text())['records']
    candidates = {name: next(r for r in records if r['name'] == name and r['policy'] == 'headroom')
                  for name in ['expanded_original', 'linear-conditional']}
    a, b = candidates.values()
    evidence = {}
    pooled = []
    for partition in PARTITIONS:
        control = summary['comparisons']['linear-conditional'][partition]
        evidence[partition.replace('validation', 'test')] = dict(candidate=a['metrics'][partition],
            baseline=b['metrics'][partition], **control)
        frame = prediction(4, 'expanded_original', partition)
        pooled.append(frame)
    frame = pd.concat(pooled, ignore_index=True)
    evidence['pooled'] = dict(candidate=metrics(frame, frame.probability, a['threshold']),
        uncertainty=summary['comparisons']['linear-conditional']['pooled'])
    check = promotion_gates(evidence)
    write_json(path, dict(status='Development only. No model promotion or operational integration.',
        exploratory_gate_check=check, final_confirmation=None,
        candidate=a, comparator=b, selection='Adaptive development selection, strongest calibrated physical baseline by mean AP',
        pipeline=dict(training='2020-2021, pooled retained cohorts excluding reserved HUC4s',
            calibration='2022, scalar standardized logistic on LightGBM log odds',
            threshold='2023, inclusive pooled 1.0 and regional 2.0 false alerts per 100 decisions',
            model='LightGBM, 250 trees, learning rate .04, 15 leaves, depth 4, minimum leaf 80, lambda 5, seed 42',
            features='scripts.shift_hydrology.FEATURES, unchanged', validation='2024, previously inspected development'),
        frozen_evidence={str(p.relative_to(ROOT)): digest(p) for p in [
            OUT / 'summary.json', OUT / 'shared-weather-sensitivity.json', OUT / 'independence-audit.json',
            OUT / 'cycle-4-freeze.json', ROOT / 'docs/HIGH_FLOW_RESEARCH_PROTOCOL.md',
            ROOT / 'scripts/high_flow_research.py', ROOT / 'scripts/report_high_flow_research.py',
            ROOT / 'scripts/report_mlp_onset.py']},
        stopping_reason='Remaining recall uncertainty, exposed geography, missing as-issued inputs and site applicability require new evidence.'))


def main():
    destination = OUT / 'summary.json'
    if destination.exists():
        raise FileExistsError(destination)
    records = {c: json.loads((OUT / f'cycle-{c}.json').read_text())['records'] for c in range(1, 5)}
    audit, hashes = [], {}
    for cycle in range(1, 5):
        lock = OUT / f'cycle-{cycle}-freeze.json'
        frozen = json.loads(lock.read_text())
        if digest(lock) != json.loads((OUT / f'cycle-{cycle}.json').read_text())['freeze_sha256']:
            raise ValueError('Cycle freeze changed')
        for path, checksum in frozen['sources'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Retained source changed')
        for path, checksum in frozen['preserved'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Production artifact changed')
        frame = load(cycle > 1)
        parts = split(frame, set(frozen['reserved_huc4']), cycle)
        for r in records[cycle]:
            if r.get('policy') != 'ceiling':
                continue
            name = r['name']
            path = OUT / 'artifacts' / f'cycle-{cycle}-{name}.joblib'
            if digest(path) != r['artifact_sha256']:
                raise ValueError('Fitted artifact changed')
            bundle = joblib.load(path)
            for partition in PARTITIONS:
                saved = prediction(cycle, name, partition)
                # Cycle 2 predates the strict fitted-station temporal definition.
                # Audit its complete saved population directly against canonical source rows.
                source = saved[['station', 'decision_time', 'horizon']].merge(frame,
                    on=['station', 'decision_time', 'horizon'], validate='one_to_one')
                if not saved[KEYS].equals(source[KEYS]):
                    raise ValueError('Prediction labels differ from source')
                replay = predict(bundle, source)
                error = float(np.max(np.abs(replay - saved.probability)))
                if error > 1e-12:
                    raise ValueError('Saved predictions fail artifact replay')
                output = OUT / 'predictions' / f'cycle-{cycle}-{name}-{partition}.csv.gz'
                hashes[str(output.relative_to(OUT))] = digest(output)
                audit.append(dict(cycle=cycle, name=name, partition=partition, rows=len(saved), max_error=error))
        # Confirm the corrected temporal definition separately from historical frozen scores.
        if not set(parts['temporal_validation'].station) <= set(parts['train'].station):
            raise ValueError('Unfitted station in temporal population')

    # Common rows isolate expansion and features from the changing validation population.
    common = []
    candidates = [(1, 'original'), (1, 'hydrologic'), (1, 'balanced'),
        (2, 'expanded_original'), (2, 'expanded_no_coordinates'), (2, 'expanded_hydrologic'),
        (2, 'expanded_balanced'), (2, 'expanded_conditional'),
        (3, 'expanded_original'), (4, 'expanded_original')]
    for partition in PARTITIONS:
        frames = common_predictions([prediction(c, n, partition) for c, n in candidates])
        for (cycle, name), f in zip(candidates, frames):
            for policy in ['ceiling', 'headroom']:
                r = next(r for r in records[cycle] if r['name'] == name and r['policy'] == policy)
                common.append(dict(cycle=cycle, name=name, policy=policy, partition=partition,
                    metrics=metrics(f, f.probability, r['threshold'])))

    comparisons = {}
    candidate = next(r for r in records[4] if r['name'] == 'expanded_original' and r['policy'] == 'headroom')
    for control in ['damped-platt', 'linear-platt', 'linear-conditional']:
        baseline = next(r for r in records[4] if r['name'] == control and r['policy'] == 'headroom')
        comparisons[control] = {}
        pooled = []
        for partition in PARTITIONS:
            a, b = common_predictions([prediction(4, 'expanded_original', partition), prediction(4, control, partition)])
            pooled.append(a.assign(control_probability=b.probability))
            comparisons[control][partition] = dict(
                uncertainty=paired_uncertainty(a, b.probability, a.probability, baseline['threshold'], candidate['threshold']),
                lead=jointly_detected_lead_delta(a, b.probability, a.probability, baseline['threshold'], candidate['threshold']))
        f = pd.concat(pooled, ignore_index=True)
        comparisons[control]['pooled'] = paired_uncertainty(f, f.control_probability, f.probability,
            baseline['threshold'], candidate['threshold'])

    write_json(destination, dict(evidence='Adaptive development only; no promotion inference',
        common_row_scores=common, comparisons=comparisons, prediction_hashes=hashes, replay_audit=audit,
        candidate=dict(cycle=4, name='expanded_original', policy='headroom',
            threshold=candidate['threshold'], artifact_sha256=candidate['artifact_sha256']),
        promotion=False, reasons=['No untouched temporal and polygon-screened geographic confirmation',
            'No as-issued input-availability evidence', 'No site-to-gauge applicability assessment']))
    print('Audited', len(audit), 'prediction files; common-row comparisons and development uncertainty saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='summary', choices=['summary', 'sensitivity', 'independence', 'decision'])
    command = parser.parse_args().command
    if command == 'summary':
        main()
    elif command == 'sensitivity':
        sensitivity()
    elif command == 'decision':
        decision()
    else:
        print(json.dumps(independence_audit(), indent=2))
