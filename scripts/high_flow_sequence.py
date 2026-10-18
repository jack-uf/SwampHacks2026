"""Hourly sequence experiment for the six-hour high-flow onset task.

New files only: this module reads the caches written by earlier sprints and never writes outside
experiments/high-flow-sequence-20261018/. Torch is imported only by
scripts/high_flow_sequence_worker.py, which runs as a subprocess, because loading torch beside
LightGBM in one process hangs or faults in this environment.

Commands, in order:
  freeze    Write the protocol manifest before anything is fitted.
  assemble  Build one hourly four-channel series per gauge from the cached USGS pages and cached ERA5
            weather, then verify that every window slices back to the frozen features.
  fit       Train in a worker subprocess, then calibrate, threshold and score 2024 exactly as the
            frozen cycle-11 tree candidate does.
  evaluate  Compare with the frozen cycle-11 predictions on identical rows.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import high_flow_sequence_windows as sequence
from scripts.flood_sprint import write_json
from scripts.high_flow_research import LATEST, load, split, threshold
from scripts.hydrology_research import ROOT, digest
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty

OUT = ROOT/'experiments/high-flow-sequence-20261018'
SIGNALS = ROOT/'experiments/high-flow-signals-20261018'
CONTROL = SIGNALS/'cycle-11'
PARTITIONS = ['train', 'calibration', 'threshold', 'temporal_validation', 'geographic_validation']
SCORED = ['calibration', 'threshold', 'temporal_validation', 'geographic_validation']
CALIBRATION_INPUTS = ['season_sin', 'season_cos', 'margin', 'log_area']
THRESHOLD_RULE = dict(pooled=1., region=2.)
# The stored series is float16, and one hourly value quantizes to roughly five parts in ten
# thousand of its own size; the discharge checks below see log1p(flow/reference) values near one,
# the weather checks see millimetres and degrees. Any interval or alignment mistake is orders of
# magnitude larger than these bounds, so they still fail an experiment that drifts: the misaligned
# rain comparison caught during development was 42 millimetres against a bound of 0.05.
TOLERANCE = {'flow': 5e-3, 'weather': 5e-2}
# A strided causal stem followed by dilated causal blocks: the window is a dense, equally spaced
# hourly grid whose relevant structure spans days, so dilated convolutions reach across it with
# 16k parameters and no recurrence state, and CPU training stays near 20 seconds an epoch.
ARCHITECTURE = dict(hidden=32, kernel=5, dilations=[1, 2, 4], dropout=.2, stem_stride=3)
TRAINING = dict(epochs=40, patience=5, batch=512, learning_rate=3e-3, weight_decay=1e-4, seed=42,
                validation_tail=.1)
CONTROL_NAME = 'winner_forecast'
CODE = ['scripts/high_flow_sequence.py', 'scripts/high_flow_sequence_windows.py',
        'scripts/high_flow_sequence_worker.py', 'scripts/shift_hydrology.py',
        'scripts/high_flow_research.py', 'scripts/shift_hydrology_metrics.py']
METHOD = ('Per decision time, one hourly window of the %d hours ending at that time inclusive: '
          'log1p(discharge/reference), precipitation, temperature_2m and soil_moisture_0_to_7cm with '
          'the six-hour receipt delay scripts/high_flow_signals.py already applies to the tree '
          'candidates, so the sequence model is not handed fresher weather than its control.') % sequence.WINDOW


def verify_freeze():
    manifest = json.loads((OUT/'freeze.json').read_text())
    current = {name: digest(ROOT/name) for name in CODE}
    if manifest['code_sha256'] != current:
        raise ValueError('Frozen implementation changed')
    for name, checksum in manifest['preserved'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Production artifact changed: '+name)
    return manifest


def signal_weather():
    """The weather columns the signals dataset itself carries, used only to verify the windows."""
    path = SIGNALS/'development.csv.gz'
    if digest(path) != json.loads((SIGNALS/'dataset.json').read_text())['sha256']:
        raise ValueError('Signal dataset changed: '+str(path))
    return pd.read_csv(path, usecols=['station', 'decision_time', 'rain_6_lag6', 'temperature_lag6'],
        dtype={'station': str})


def population():
    """The evaluated decision population and the frozen split, reproduced from the signals dataset."""
    frame = load(True)
    reserved = {s['huc'][:4] for s in json.loads((LATEST/'stations.json').read_text())
                if s['role'] == 'geographic_test'}
    frame = frame.merge(signal_weather(), on=['station', 'decision_time'], how='left', validate='one_to_one')
    return frame, split(frame, reserved, cycle=4), sorted(reserved)


def verify(part, array):
    """Prove the windows are the same series the frozen features were computed from.

    Every comparison re-derives a stored feature from the assembled window: the discharge taps and
    rise shapes, the 24-hour range, the six-hour rain total and the delayed temperature. A mismatch
    means the interval semantics drifted and the experiment must stop.
    """
    last = sequence.WINDOW - 1
    checks, limits = {}, {}
    for lag in [2, 8, 14, 26, 50, 74]:
        checks[f'q_{lag}'] = float(np.nanmax(np.abs(array[:, 0, last - lag] - part[f'q_{lag}'].to_numpy(float))))
        limits[f'q_{lag}'] = TOLERANCE['flow']
    for lag in [6, 24, 72]:
        window = part[f'rise_{lag}'].to_numpy(float)
        rebuilt = array[:, 0, last - 2] - array[:, 0, last - lag - 2]
        checks[f'rise_{lag}'] = float(np.nanmax(np.abs(rebuilt - window)))
        limits[f'rise_{lag}'] = TOLERANCE['flow']
    # range_24 is the 24-hour span ending at t-2 (rolling(24).shift(2)); the newest rain the model
    # sees at t is the hour t-6, so rain_6_lag6 sums the channel's final six entries.
    rebuilt_range = np.nanmax(array[:, 0, last - 25:last - 1], axis=1) - np.nanmin(array[:, 0, last - 25:last - 1], axis=1)
    checks['range_24'] = float(np.nanmax(np.abs(rebuilt_range - part['range_24'].to_numpy(float))))
    limits['range_24'] = TOLERANCE['flow']
    checks['rain_6_lag6'] = float(np.nanmax(np.abs(array[:, 1, last - 5:].sum(axis=1)
                                                   - part['rain_6_lag6'].to_numpy(float))))
    limits['rain_6_lag6'] = TOLERANCE['weather']
    checks['temperature_lag6'] = float(np.nanmax(np.abs(array[:, 2, last] - part['temperature_lag6'].to_numpy(float))))
    limits['temperature_lag6'] = TOLERANCE['weather']
    if [name for name in checks if checks[name] > limits[name]]:
        raise ValueError('Assembled windows disagree with the frozen features: '+json.dumps(checks))
    return checks


def partition_array(part, records, cache):
    """Materialize one partition's windows in memory from the stored per-gauge series."""
    array = np.empty((len(part), len(sequence.CHANNELS), sequence.WINDOW), dtype=np.float32)
    times = pd.to_datetime(part.decision_time, utc=True)
    for station, group in part.groupby('station', sort=True):
        origin, block = sequence.load_series(OUT/'series', records, station, cache)
        array[group.index.to_numpy()] = sequence.windows(block, origin, times.loc[group.index])
    return array


def freeze():
    if (OUT/'freeze.json').exists():
        raise FileExistsError('Protocol already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((SIGNALS/'dataset.json').read_text())
    write_json(OUT/'freeze.json', dict(
        experiment='Hourly sequence model for the six-hour high-flow onset task',
        method=METHOD,
        population=dict(dataset=str((SIGNALS/'development.csv.gz').relative_to(ROOT)),
            dataset_sha256=sources['sha256'], gauges=sources['gauges'],
            window_hours=sequence.WINDOW, channels=sequence.CHANNELS,
            weather_delay_hours=sequence.WEATHER_DELAY,
            weather_note='ERA5 reanalysis; the delay is the retrospective availability assumption the '
                'signals sprint already uses, not an as-issued forecast.',
            unit='One row is one eligible decision at the frozen six-hour cadence, horizon 6 only.'),
        storage=dict(artifact='one float16 hourly four-channel series per gauge under series/, about '
                '31 MB in total, plus one gzipped index per partition and the worker logits',
            note='Windows are sliced from the series on demand; a materialized float32 window per '
                'decision row needs about 760 MB, which this machine cannot spare. Every stored byte '
                'is rebuilt by the frozen code from the cached pages, whose hashes are recorded in '
                'series.json, and no window array is kept after training and scoring.',
            population='Every eligible decision of the 88-gauge signals population is kept. Subsample'
                ' of decision times: none. Storage scales with gauge-hours, not decision rows, so a '
                'one-hour-a-day subsample would free no space and would break the identical-row '
                'comparison with the stored cycle-11 predictions.',
            regeneration='Delete series/ and index/ and rerun assemble; freeze.json then still '
                'describes the run.'),
        split=dict(definition='scripts/high_flow_research.split(cycle=4), reserved HUC4s from '
                'experiments/sprint-20261017-mlp-geo/stations.json role=geographic_test',
            commentary='Train 2020-2021, calibrate 2022, threshold 2023, score 2024. Unchanged.'),
        calibration=dict(inputs=CALIBRATION_INPUTS,
            estimator='StandardScaler + LogisticRegression(C=1, max_iter=2000, random_state=42) on the '
                '2022 partition, identical to scripts/high_flow_signals.py',
            requirement='Calibration slope must be positive.'),
        threshold=dict(rule='scripts/high_flow_research.threshold on the 2023 partition, inclusive',
            pooled=THRESHOLD_RULE['pooled'], region=THRESHOLD_RULE['region']),
        architecture=ARCHITECTURE,
        training=dict(**TRAINING, early_stopping='Contiguous tail of the training period by '
            'decision_time (last %d%% of rows), never calibration, threshold or 2024 rows.'
            % int(TRAINING['validation_tail']*100)),
        comparison=dict(control=str((CONTROL/f'{CONTROL_NAME}-temporal_validation.csv.gz').relative_to(ROOT)),
            draws=500, seed=42, clusters='huc4',
            metrics='scripts/shift_hydrology_metrics.metrics and paired_uncertainty'),
        code_sha256={name: digest(ROOT/name) for name in CODE},
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'models').glob('*') if p.is_file()}))
    print('Frozen sequence protocol', flush=True)


def assemble():
    verify_freeze()
    if (OUT/'series.json').exists():
        raise FileExistsError('Series already assembled')
    (OUT/'series').mkdir(parents=True, exist_ok=True)
    (OUT/'index').mkdir(parents=True, exist_ok=True)
    frame, parts, reserved = population()
    times = pd.to_datetime(frame.decision_time, utc=True)
    records = {}
    total = int(frame.station.nunique())
    started = time.monotonic()
    for index, station in enumerate(sorted(frame.station.unique()), 1):
        rows = frame.loc[frame.station.eq(station)]
        reference = rows.reference_cfs.to_numpy(float)
        if len(np.unique(reference)) != 1 or reference[0] <= 0:
            raise ValueError('Ambiguous reference discharge for '+station)
        origin, block, sources = sequence.station_series(station, times.loc[rows.index], reference[0])
        path = OUT/'series'/(station + '.npy')
        np.save(path, block.astype(np.float16))
        records[station] = dict(file=path.name, origin=origin.isoformat(), hours=int(block.shape[1]),
            dtype='float16', sha256=digest(path), bytes=path.stat().st_size, sources=sources)
        if index % 20 == 0 or index == total:
            print('assembled', index, 'of', total, 'gauges',
                  round(time.monotonic() - started, 1), 'seconds', flush=True)
    report = dict(gauges=sorted(records), series=records,
        series_bytes=sum(record['bytes'] for record in records.values()), partitions={}, checks={},
        scored_rows=0, elapsed_seconds=None)
    write_json(OUT/'series.json', report)
    cache = {}
    for key, part in parts.items():
        array = partition_array(part, records, cache)
        part[['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']].to_csv(
            OUT/'index'/(key + '.csv.gz'), index=False, compression={'method': 'gzip', 'mtime': 0})
        missing = {name: int(np.isnan(array[:, i]).sum()) for i, name in enumerate(sequence.CHANNELS)}
        if missing['log_flow']:
            empty = part.loc[np.isnan(array[:, 0]).any(axis=1), ['station', 'decision_time']]
            raise ValueError('Discharge windows are incomplete for ' + str(empty.to_dict('records')[:20]))
        report['partitions'][key] = dict(rows=len(part), gauges=int(part.station.nunique()),
            positives=int(part.target.sum()), start=part.decision_time.min(), end=part.decision_time.max(),
            index_sha256=digest(OUT/'index'/(key + '.csv.gz')), missing_entries=missing)
        report['checks'][key] = verify(part, array)
        del array
        print(key, report['partitions'][key]['rows'], 'rows,', report['partitions'][key]['gauges'], 'gauges',
              'checks', report['checks'][key], flush=True)
    report['reserved_huc4'] = reserved
    report['scored_rows'] = int(sum(report['partitions'][key]['rows'] for key in SCORED))
    report['elapsed_seconds'] = time.monotonic() - started
    write_json(OUT/'series.json', report)


def fit():
    verify_freeze()
    report = json.loads((OUT/'series.json').read_text())
    for station, record in report['series'].items():
        if digest(OUT/'series'/record['file']) != record['sha256']:
            raise ValueError('Stored series changed: '+station)
    for key, record in report['partitions'].items():
        if digest(OUT/'index'/(key + '.csv.gz')) != record['index_sha256']:
            raise ValueError('Stored index changed: '+key)
    if (OUT/'results.json').exists():
        raise FileExistsError('Sequence candidate already fitted')
    (OUT/'predictions').mkdir(parents=True, exist_ok=True)
    frame, parts, reserved = population()
    order = parts['train'].sort_values('decision_time').index.to_numpy()
    cut = int(len(order)*(1 - TRAINING['validation_tail']))
    boundary = parts['train'].decision_time.iloc[order[cut:]].min()
    job = dict(architecture=ARCHITECTURE, training=TRAINING, window=sequence.WINDOW,
        channels=sequence.CHANNELS, series=str(OUT/'series.json'), directory=str(OUT/'series'),
        index=str(OUT/'index'), partitions=PARTITIONS, scored=SCORED,
        inner_train_rows=order[:cut].tolist(), inner_validation_rows=order[cut:].tolist(),
        output=str(OUT/'sequence-logits.csv.gz'), summary=str(OUT/'worker.json'),
        bundle=str(OUT/'sequence-model.pt'), threads=4)
    write_json(OUT/'fit-job.json', job)
    environment = dict(os.environ, OMP_NUM_THREADS='4', KMP_DUPLICATE_LIB_OK='TRUE', MPLCONFIGDIR='/tmp')
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-m', 'scripts.high_flow_sequence_worker', str(OUT/'fit-job.json')],
        cwd=ROOT, capture_output=True, text=True, env=environment)
    if result.returncode != 0:
        raise ValueError('Sequence worker failed: ' + result.stderr[-2000:])
    print(result.stdout.strip(), flush=True)
    worker = json.loads((OUT/'worker.json').read_text())
    logits = pd.read_csv(OUT/'sequence-logits.csv.gz', dtype={'station': str}, keep_default_na=False)
    def raw(key):
        part = parts[key]
        aligned = part[['station', 'decision_time']].merge(
            logits.loc[logits.partition.eq(key), ['station', 'decision_time', 'logit']],
            on=['station', 'decision_time'], how='left', validate='one_to_one')
        if aligned.logit.isna().any():
            raise ValueError('Missing sequence logits for a scored row of '+key)
        inputs = pd.DataFrame({'score': aligned.logit.to_numpy(float)})
        for column in CALIBRATION_INPUTS:
            inputs[column] = part[column].to_numpy()
        return inputs
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
    calibrator.fit(raw('calibration'), parts['calibration'].target)
    if calibrator[-1].coef_[0, 0] <= 0:
        raise ValueError('Nonpositive calibration slope')
    cut_threshold = threshold(parts['threshold'],
        calibrator.predict_proba(raw('threshold'))[:, 1], THRESHOLD_RULE['pooled'], THRESHOLD_RULE['region'])
    scores = {}
    for key in ['temporal_validation', 'geographic_validation']:
        part = parts[key]
        probability = calibrator.predict_proba(raw(key))[:, 1]
        prediction = part[['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']].copy()
        prediction['probability'] = probability
        prediction.to_csv(OUT/'predictions'/f'sequence-{key}.csv.gz', index=False,
                          compression={'method': 'gzip', 'mtime': 0})
        scores[key] = metrics(prediction, probability, cut_threshold)
    bundle = dict(calibrator=calibrator, threshold=cut_threshold, architecture=ARCHITECTURE,
        features=sequence.CHANNELS, window=sequence.WINDOW, weather_delay=sequence.WEATHER_DELAY,
        worker=worker)
    joblib.dump(bundle, OUT/'sequence.joblib', compress=3)
    record = dict(name='sequence_tcn', threshold=cut_threshold, metrics=scores,
        artifact_sha256=digest(OUT/'sequence.joblib'),
        calibration=dict(coef=calibrator[-1].coef_.tolist(), inputs=CALIBRATION_INPUTS,
            rows=int(len(parts['calibration']))),
        training=dict(epochs_run=worker['epochs_run'], validation_log_loss=worker['validation_log_loss'],
            parameters=worker['parameters'], inner_train_rows=cut, inner_validation_rows=len(order) - cut,
            validation_boundary=str(boundary), seconds=worker['seconds']),
        seconds=time.monotonic() - started)
    write_json(OUT/'results.json', [record])
    print('sequence_tcn', {k: (v['detected_events'], v['events'], round(v['ap'], 4),
          round(v['false_alerts_per_100'], 3)) for k, v in scores.items()}, flush=True)


def evaluate():
    verify_freeze()
    records = json.loads((OUT/'results.json').read_text())
    record = next(r for r in records if r['name'] == 'sequence_tcn')
    if digest(OUT/'sequence.joblib') != record['artifact_sha256']:
        raise ValueError('Sequence artifact changed')
    control = next(r for r in json.loads((CONTROL/'results.json').read_text())
                   if r['name'] == CONTROL_NAME)
    result = dict(draws=500, seed=42, clusters='huc4', candidate='sequence_tcn',
        control=str((CONTROL/f'{CONTROL_NAME}-temporal_validation.csv.gz').relative_to(ROOT)),
        control_results_sha256=digest(CONTROL/'results.json'), partitions={})
    columns = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']
    for key in ['temporal_validation', 'geographic_validation']:
        stored = pd.read_csv(CONTROL/f'{CONTROL_NAME}-{key}.csv.gz', dtype={'station': str, 'huc': str},
            keep_default_na=False)
        mine = pd.read_csv(OUT/'predictions'/f'sequence-{key}.csv.gz', dtype={'station': str, 'huc': str},
            keep_default_na=False)
        left = mine[columns].sort_values(['station', 'decision_time']).reset_index(drop=True)
        right = stored[columns].sort_values(['station', 'decision_time']).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError('Sequence and control are not scored on identical rows: '+key)
        control_probability = stored.probability.to_numpy(float)
        probability = mine.probability.to_numpy(float)
        result['partitions'][key] = dict(rows=len(left),
            control=dict(threshold=control['threshold'], metrics=metrics(
                left.assign(probability=control_probability), control_probability, control['threshold'])),
            candidate=dict(threshold=record['threshold'], metrics=record['metrics'][key]),
            paired=paired_uncertainty(left, control_probability, probability, control['threshold'],
                record['threshold'], draws=result['draws'], seed=result['seed']),
            lead=jointly_detected_lead_delta(left, control_probability, probability,
                control['threshold'], record['threshold']))
    write_json(OUT/'comparison.json', result)
    for key, value in result['partitions'].items():
        print(key, 'sequence', (value['candidate']['metrics']['detected_events'],
              round(value['candidate']['metrics']['ap'], 4)),
              'control', (value['control']['metrics']['detected_events'],
              round(value['control']['metrics']['ap'], 4)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'assemble', 'fit', 'evaluate'])
    arguments = parser.parse_args()
    {'freeze': freeze, 'assemble': assemble, 'fit': fit, 'evaluate': evaluate}[arguments.command]()
