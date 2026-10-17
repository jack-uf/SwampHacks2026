"""Transformer experiment coordinator. See docs/TRANSFORMER_SPRINT_PROTOCOL.md before running.

Model fitting happens in scripts/transformer_worker.py, one process per candidate:
on this machine xgboost/lightgbm and torch cannot initialize OpenMP in the same
process, so this coordinator never imports a model library itself. It prepares
frames, runs the workers, and computes every threshold and metric from the saved
probability arrays.
"""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import ROOT, download
from scripts import wind_sprint
from scripts.wind_sprint import EXTENDED, THRESHOLD_KMH, digest, rule_metrics, station_metrics, write_json

OUT = ROOT / 'experiments/sprint-20261017-transformer'
PROTOCOL_DOCUMENT = ROOT / 'docs/TRANSFORMER_SPRINT_PROTOCOL.md'
EXTENSION_START, EXTENSION_END = '2026-06-27', '2026-10-15'
FRESH_START = '2026-07-01'
WIND_END = '2026-06-30'
CHECKPOINT = Path.home() / 'Library/Caches/tabpfn' / 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'
CHECKPOINT_SHA256 = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
CANDIDATE_NAMES = ('logistic_extended', 'xgboost_extended', 'lightgbm_extended', 'tabpfn_extended', 'ft_transformer_extended')
PACKAGES = ['tabpfn', 'torch', 'mlx', 'scikit-learn', 'xgboost', 'lightgbm', 'numpy', 'pandas', 'joblib', 'scipy']


def verify_checkpoint():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'TabPFN checkpoint missing: {CHECKPOINT}; the frozen experiment requires the cached v2 weights')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('TabPFN checkpoint hash changed; the frozen protocol pins the v2 weights hash')


def frames_dir():
    return OUT / 'frames'


def predictions_dir():
    return OUT / 'probabilities'


def artifacts_digest():
    files = sorted(path for path in predictions_dir().glob('*') if path.is_file())
    if not files:
        raise FileNotFoundError('No frozen candidate artifacts found')
    return hashlib.sha256('|'.join(f'{path.name}:{digest(path)}' for path in files).encode()).hexdigest()


def prepare_frames(frames):
    frames_dir().mkdir(parents=True, exist_ok=True)
    (frames_dir() / 'features.json').write_text(json.dumps(dict(features=EXTENDED), indent=2) + '\n')
    (frames_dir() / 'splits.json').write_text(json.dumps(dict(splits=list(frames)), indent=2) + '\n')
    for name, frame in frames.items():
        frame.to_csv(frames_dir() / f'{name}.csv', index=False)


def run_worker(name, reuse=False, out_dir=None, models_dir=None):
    out_dir = Path(out_dir) if out_dir else predictions_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'scripts.transformer_worker', name, str(frames_dir()), str(out_dir)]
    if reuse:
        command.append('--reuse')
    command += ['--models', str(Path(models_dir) if models_dir else out_dir)]
    environment = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1')
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    if result.returncode:
        raise RuntimeError(f'Worker {name} failed with exit {result.returncode}: ' + result.stderr[-900:])
    return json.loads((out_dir / f'{name}.meta.json').read_text())


def load_probability(name, split):
    return np.load(predictions_dir() / f'{name}.{split}.npy')


def matched_budget_threshold(scores, y, budget_fraction):
    scores, y = np.asarray(scores), np.asarray(y)
    negatives = np.sort(scores[~y.astype(bool)])[::-1]
    if not len(negatives):
        raise ValueError('Matched-budget threshold needs at least one negative validation row')
    budget = int(np.ceil(budget_fraction * len(negatives)))
    if budget <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    if budget >= len(negatives):
        return float(negatives[-1])
    return float(negatives[budget - 1])


def extension_frames():
    frame = pd.read_csv(OUT / 'dataset-extension.csv.gz')
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Invalid extension rows')
    if (pd.to_datetime(frame.target_date) < pd.Timestamp(FRESH_START)).any():
        raise ValueError('Extension must start at the fresh window; earlier days belong to the frozen dataset')
    frozen = wind_sprint.load_dataset()
    if (pd.to_datetime(frozen.target_date) > pd.Timestamp(WIND_END)).any():
        raise ValueError('Frozen dataset changed: it no longer ends where the protocol says')
    if set(frame.station) - set(frozen.station):
        raise ValueError('Extension contains stations outside the frozen protocol')
    return frozen, frame


def selection_inputs():
    frame = wind_sprint.load_dataset()
    parts = wind_sprint.partition_rows(frame[frame.target_date < '2026-01-01'])
    validation = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    return parts, validation


def cohorts_of(frozen, fresh):
    return dict(
        fresh_temporal=fresh[fresh.role.eq('development')],
        fresh_geographic_validation=fresh[fresh.role.eq('geographic_validation')],
        fresh_geographic_test=fresh[fresh.role.eq('geographic_test')],
        inspected_temporal=frozen[frozen.role.eq('development') & (frozen.target_date >= '2026-01-01')],
        inspected_geographic_test=frozen[frozen.role.eq('geographic_test') & (frozen.target_date >= '2026-01-01')])


def extend():
    if (OUT / 'dataset-extension.csv.gz').exists():
        raise FileExistsError('Extension already frozen; use select or verify')
    protocol = wind_sprint.load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, coverage, sources = [], [], []
    for station in protocol['stations']:
        params = [('station', station['sid']), ('data', 'gust'), ('report_type', 3), ('report_type', 4),
                  ('year1', 2026), ('month1', 6), ('day1', 27), ('year2', 2026), ('month2', 10), ('day2', 15),
                  ('tz', 'UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('direct', 'no')]
        observed_path = download('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                                 f"iem-asos-{station['sid']}-202610.csv", params)
        observed = {row['target_date']: row for row in wind_sprint.observation_days(observed_path.read_text())}
        forecast_params = dict(latitude=station['latitude'], longitude=station['longitude'], start_date=EXTENSION_START,
                               end_date=EXTENSION_END, hourly=','.join(v + '_previous_day2' for v in wind_sprint.UNITS),
                               models='gfs_seamless', timezone='GMT')
        forecast_path = download('https://previous-runs-api.open-meteo.com/v1/forecast',
                                 f"wind-forecast-{station['sid']}-202610.json", forecast_params)
        daily = wind_sprint.forecast_features(json.loads(forecast_path.read_text()), station)
        for values in daily:
            if values['target_date'] < FRESH_START or values['target_date'] not in observed:
                continue
            day = observed[values['target_date']]
            rows.append(dict(**values, max_gust_kmh=day['max_gust_kmh'], exceedance=day['exceedance'],
                             first_exceed_hour_utc=day['first_exceed_hour_utc'], station=station['sid'],
                             name=station['name'], state=station['state'], role=station['role']))
        coverage.append(dict(station=station['sid'], role=station['role'], observed_days=len(observed),
                             complete_forecast_days=len(daily),
                             merged_days=len([v for v in daily if v['target_date'] >= FRESH_START and v['target_date'] in observed]),
                             first=min(observed) if observed else None, last=max(observed) if observed else None))
        for path in [observed_path, forecast_path]:
            sources.append(json.loads(path.with_suffix(path.suffix + '.source.json').read_text()))
        print(station['sid'], 'observed days', len(observed), 'merged fresh days', coverage[-1]['merged_days'], flush=True)
        time.sleep(10)
    frame = pd.DataFrame(rows).sort_values(['target_date', 'station'])
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Duplicate or non-finite extension rows')
    frame.to_csv(OUT / 'dataset-extension.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    write_json(OUT / 'provenance-extension.json', dict(created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
        coverage=coverage, dataset_sha256=digest(OUT / 'dataset-extension.csv.gz'),
        protocol_document_sha256=digest(PROTOCOL_DOCUMENT),
        window=dict(requested=[EXTENSION_START, EXTENSION_END], fresh_from=FRESH_START, frozen_through=WIND_END),
        contracts='Same frozen parsing, completeness and issue-time contracts as scripts/wind_sprint.py'))
    print('Extension frozen:', len(frame), 'rows;', frame.station.nunique(), 'stations', flush=True)


def select():
    if (OUT / 'selection.json').exists():
        raise FileExistsError('Selection is already frozen; use evaluate or verify')
    verify_checkpoint()
    parts, validation = selection_inputs()
    reference = (validation.forecast_gust_max_kmh >= THRESHOLD_KMH).astype(float)
    baseline = rule_metrics(validation.exceedance, validation.forecast_gust_max_kmh, THRESHOLD_KMH)
    budget_fraction = baseline['false_alerts_per_100_station_days'] / 100
    prepare_frames(dict(train=parts['train'], calibration=parts['calibration'],
                        validation=parts['validation'], geographic_validation=parts['geographic_validation']))
    results, metadata = {}, {}
    for name in CANDIDATE_NAMES:
        meta = run_worker(name)
        probabilities = {split: load_probability(name, split) for split in ['validation', 'geographic_validation']}
        pooled = np.concatenate([probabilities['validation'], probabilities['geographic_validation']])
        threshold, supported = wind_sprint.recall_threshold(validation.exceedance, pooled, reference, .5)
        scores = {part: station_metrics(parts[part].exceedance, probabilities[part], threshold)
                  for part in ['validation', 'geographic_validation']}
        scores.update(features=list(EXTENDED), threshold=threshold, warning_supported=supported,
                      matched_threshold=matched_budget_threshold(pooled, validation.exceedance, budget_fraction),
                      fit_seconds=meta['fit_seconds'], tool_seconds=meta['total_seconds'])
        scores['selection_ap'] = float(np.mean([scores[k]['pr_auc'] or 0. for k in ['validation', 'geographic_validation']]))
        scores['selection_brier'] = float(np.mean([scores[k]['brier'] for k in ['validation', 'geographic_validation']]))
        results[name], metadata[name] = scores, meta
        print(name, 'fit seconds', meta['fit_seconds'], 'threshold', round(threshold, 4), 'validation AP',
              round(scores['selection_ap'], 4), 'supported', supported, flush=True)
    eligible = [name for name in results if results[name]['warning_supported']]
    winner = sorted(eligible or list(results), key=lambda n: (-results[n]['selection_ap'], results[n]['selection_brier'], n))[0]
    write_json(OUT / 'selection.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selected=winner,
        eligible_candidates=eligible, candidates=results, baseline_validation=baseline, budget_fraction=budget_fraction,
        worker_metadata=metadata, dataset_sha256=digest(wind_sprint.OUT / 'training.csv.gz'),
        protocol_sha256=digest(wind_sprint.OUT / 'protocol.json'), artifacts_sha256=artifacts_digest(),
        protocol_document_sha256=digest(PROTOCOL_DOCUMENT), checkpoint_sha256=CHECKPOINT_SHA256,
        code_sha256=digest(ROOT / 'scripts/transformer_sprint.py'),
        worker_sha256=digest(ROOT / 'scripts/transformer_worker.py'),
        versions={name: importlib.metadata.version(name) for name in PACKAGES},
        selection_rule='Supported candidates first; equal temporal/geographic validation AP, mean Brier tie-break, then name',
        execution='One worker process per candidate; the coordinator never imports a model library',
        test_outcomes_inspected=False))
    print('Selected', winner, flush=True)


def transformer_gate_failures(updated, baseline, intervals, stations, supported, matched):
    failures = wind_sprint.integration_failures(updated, baseline, intervals, stations, supported)
    if matched['recall'] < baseline['recall'] - 1e-12:
        failures.append('Matched-budget recall below the forecast rule')
    return failures


def evaluate():
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists; do not tune on it. Use verify')
    selection = frozen_selection()
    with (OUT / 'evaluation-started.json').open('x') as marker:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(OUT / 'selection.json')), marker)
    verify_checkpoint()
    parts, validation = selection_inputs()
    frozen, fresh = extension_frames()
    cohorts = cohorts_of(frozen, fresh)
    prepare_frames(cohorts)
    probabilities = {}
    for name in CANDIDATE_NAMES:
        run_worker(name, reuse=True)
        probabilities[name] = {cohort: load_probability(name, cohort) for cohort in cohorts}
        print('predicted', name, flush=True)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), selected=selection['selected'],
                  selection_sha256=digest(OUT / 'selection.json'), protocol='docs/TRANSFORMER_SPRINT_PROTOCOL.md',
                  checkpoint_sha256=CHECKPOINT_SHA256, cohorts={}, baseline_contract='Archived forecast maximum gust at 60 km/h',
                  metrics_contract='Selection-threshold metrics plus the matched-budget operating point defined in the protocol')
    prediction_frames = []
    for cohort, data in cohorts.items():
        info = dict(rows=len(data), stations=int(data.station.nunique()), positives=int(data.exceedance.sum()),
                    states=sorted(set(data.state)), models={})
        for name in CANDIDATE_NAMES:
            threshold = selection['candidates'][name]['threshold']
            matched = selection['candidates'][name]['matched_threshold']
            probability = probabilities[name][cohort]
            info['models'][name] = dict(selection=station_metrics(data.exceedance, probability, threshold),
                                        matched_budget=station_metrics(data.exceedance, probability, matched))
            prediction_frames.append(pd.DataFrame(dict(cohort=cohort, station=data.station, target_date=data.target_date,
                label=data.exceedance, model=name, probability=probability,
                selection_alert=probability >= threshold, matched_alert=probability >= matched)))
        info['baseline'] = rule_metrics(data.exceedance, data.forecast_gust_max_kmh, THRESHOLD_KMH)
        info['climatology'] = station_metrics(data.exceedance, np.full(len(data), parts['train'].exceedance.mean()), .5)
        winner_name = selection['selected']
        winner_probability = probabilities[winner_name][cohort]
        info['winner'] = winner_name
        info['paired_uncertainty'] = wind_sprint.paired_intervals(data, data.exceedance, data.forecast_gust_max_kmh,
                                                                  winner_probability, THRESHOLD_KMH,
                                                                  selection['candidates'][winner_name]['threshold'])
        info['lead_time_winner'] = wind_sprint.lead_times(data, winner_probability >= selection['candidates'][winner_name]['threshold'])
        info['stations_detail'] = {station: dict(baseline=rule_metrics(group.exceedance, group.forecast_gust_max_kmh, THRESHOLD_KMH),
                                                 updated=station_metrics(group.exceedance,
                                                                         winner_probability[data.station.eq(station).to_numpy()],
                                                                         selection['candidates'][winner_name]['threshold']))
                                   for station, group in data.groupby('station')}
        report['cohorts'][cohort] = info
        print(cohort, 'rows', len(data), 'positives', info['positives'], 'rule AP', round(info['baseline']['pr_auc'] or 0, 4),
              'winner AP', round(info['models'][winner_name]['selection']['pr_auc'] or 0, 4), flush=True)
    primary = report['cohorts']['fresh_temporal']
    winner = primary['models'][selection['selected']]
    report['gate'] = transformer_gate_failures(winner['selection'], primary['baseline'], primary['paired_uncertainty'],
                                               primary['stations_detail'],
                                               selection['candidates'][selection['selected']]['warning_supported'],
                                               winner['matched_budget'])
    report['decision'] = 'propose_integration_review' if not report['gate'] else 'retain_forecast_threshold'
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / 'predictions.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    report['predictions_sha256'] = digest(OUT / 'predictions.csv.gz')
    write_json(OUT / 'evaluation.json', report)
    print('Decision:', report['decision'], 'failures:', report['gate'], flush=True)


def frozen_selection():
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, field in [(wind_sprint.OUT / 'training.csv.gz', 'dataset_sha256'),
                        (wind_sprint.OUT / 'protocol.json', 'protocol_sha256'), (PROTOCOL_DOCUMENT, 'protocol_document_sha256'),
                        (ROOT / 'scripts/transformer_sprint.py', 'code_sha256'),
                        (ROOT / 'scripts/transformer_worker.py', 'worker_sha256')]:
        if digest(path) != selection[field]:
            raise ValueError('Frozen selection input changed: ' + str(path))
    if artifacts_digest() != selection['artifacts_sha256']:
        raise ValueError('Frozen candidate artifacts changed')
    for name, version in selection['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Frozen package version changed: ' + name)
    if selection['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Frozen checkpoint hash changed')
    return selection


def verify():
    selection = frozen_selection()
    verify_checkpoint()
    report = json.loads((OUT / 'evaluation.json').read_text())
    if digest(OUT / 'selection.json') != report['selection_sha256'] or digest(OUT / 'predictions.csv.gz') != report['predictions_sha256']:
        raise ValueError('Evaluation artifact checksum mismatch')
    assert report['selected'] == selection['selected']
    assert report['decision'] == ('retain_forecast_threshold' if report['gate'] else 'propose_integration_review')
    parts, validation = selection_inputs()
    frozen, fresh = extension_frames()
    cohorts = cohorts_of(frozen, fresh)
    prepare_frames(cohorts)
    predictions = pd.read_csv(OUT / 'predictions.csv.gz')
    with tempfile.TemporaryDirectory() as tmp:
        for name in CANDIDATE_NAMES:
            run_worker(name, reuse=True, out_dir=tmp, models_dir=predictions_dir())
            for cohort, data in cohorts.items():
                fresh_probability = np.load(Path(tmp) / f'{name}.{cohort}.npy')
                saved = predictions[predictions.cohort.eq(cohort) & predictions.model.eq(name)]
                assert len(saved) == len(data)
                assert list(saved.station) == list(data.station) and list(saved.target_date) == list(data.target_date)
                assert np.array_equal(saved.label, data.exceedance)
                assert np.allclose(fresh_probability, saved.probability, atol=1e-12, rtol=0)
                threshold = selection['candidates'][name]['threshold']
                assert station_metrics(data.exceedance, fresh_probability, threshold) == report['cohorts'][cohort]['models'][name]['selection']
    print('Verified frozen inputs, checkpoint, workers, prediction rows and pooled metrics')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extend', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    dict(extend=extend, select=select, evaluate=evaluate, verify=verify)[args.command]()

