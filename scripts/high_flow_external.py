"""External benchmark for the six-hour high-flow onset task.

Priority-2 research sprint: score the closest honestly runnable external architecture, the Google
Flood Hub encoder-decoder LSTM family from the OpenHydroNet fork at /tmp/planit-ext/flood-forecasting,
against the frozen cycle-11 candidate
(experiments/high-flow-signals-20261018/cycle-11/winner_forecast) on identical rows and operating
points, with the same conditional calibration, the same inclusive pooled-1/regional-2 threshold rule
and the same sixteen promotion gates as scripts/high_flow_signals.py and
scripts/report_high_flow_signals.py.

The pretrained Google weights cannot be used here. All 88 gauges are in their training basin list as
camels_<id>, their training window (1982-01-01 to 2023-09-30) contains our train, calibration and
threshold partitions, and the released daily input contract cannot be built from the caches. See
audit.json and RUN.md. Everything below is trained from scratch on this repository's population and
protocol and is labelled "Google architecture under our protocol", never Google's published
performance.

New files only; existing scripts and experiments are read, never written. Commands, in order:
  audit     Record the overlap audit (released weights, RiverMamba reforecasts) with hashes.
  freeze    Write the protocol manifest before anything is fitted.
  assemble  Build the cached hourly four-channel series and the partition indices.
  fit       Train, calibrate, threshold and score each candidate under the frozen protocol.
  evaluate  Compare each candidate with the frozen candidate on identical rows and budgets.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import high_flow_sequence_windows as sequence
from scripts import high_flow_signals as signals
from scripts.flood_sprint import write_json
from scripts.high_flow_research import LATEST, split, threshold
from scripts.hydrology_research import ROOT, digest
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty

OUT = ROOT/'experiments/high-flow-external-20261018'
SIGNALS = ROOT/'experiments/high-flow-signals-20261018'
CONTROL = SIGNALS/'cycle-11'
CONTROL_NAME = 'winner_forecast'
REFERENCE = Path('/tmp/planit-ext')
FLOOD = REFERENCE/'flood-forecasting'
RIVERMAMBA = REFERENCE/'RiverMamba_code'
PARTITIONS = ['train', 'calibration', 'threshold', 'temporal_validation', 'geographic_validation']
SCORED = ['calibration', 'threshold', 'temporal_validation', 'geographic_validation']
EVALUATED = ['temporal_validation', 'geographic_validation']
CALIBRATION_INPUTS = ['season_sin', 'season_cos', 'margin', 'log_area']
THRESHOLD_RULE = dict(pooled=1., region=2.)
BUDGETS = [0.5, 1.0, 1.5, 2.0]
CEILING = dict(pooled=1.5, region=3.)
STATICS = ['season_sin', 'season_cos', 'latitude', 'longitude', 'log_area', 'elevation_m']
ARCHITECTURE = dict(channel_hidden=16, lstm_hidden=64, engineered_hidden=32, statics_hidden=16,
    head_hidden=32, dropout=.2)
TRAINING = dict(epochs=30, patience=5, batch=512, learning_rate=3e-3, weight_decay=1e-4, seed=42,
    validation_tail=.1)
CANDIDATES = [
    dict(name='google_lstm_engineered', engineered=True, statics=True,
        description='raw hourly window plus the frozen cycle-11 engineered feature set and statics'),
    dict(name='google_lstm_raw', engineered=False, statics=False,
        description='raw hourly window only'),
]
TOLERANCE = {'flow': 5e-3, 'weather': 5e-2}
FLOW_DELAY_HOURS = 2
COLUMNS = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']
CODE = ['scripts/high_flow_external.py', 'scripts/high_flow_external_worker.py',
        'scripts/high_flow_sequence_windows.py', 'scripts/high_flow_signals.py',
        'scripts/high_flow_research.py', 'scripts/shift_hydrology_metrics.py',
        'scripts/shift_hydrology.py', 'scripts/hydrology_research.py']
METHOD = ('Per decision time, one raw hourly window of the %d hours ending at that time inclusive '
    '(log1p(discharge/reference) delayed two hours so its newest value equals the frozen receipt-'
    'delay contract q_2, precipitation, temperature_2m and soil_moisture_0_to_7cm with the six-hour '
    'receipt delay the signals sprint already applies), fed to the Google release encoder '
    'conventions adapted to a single binary six-hour onset head. google_lstm_engineered additionally '
    'receives the frozen candidate feature set and the static attributes through their own '
    'embeddings; google_lstm_raw does not. The released Mean-Embedding-Forecast-LSTM needs daily '
    'hres/graphcast/imerg/cpc arrays, 365-day sequences and 84 Multimet statics the caches do not '
    'contain, so it cannot run here; see audit.json.') % sequence.WINDOW


def verify_freeze():
    manifest = json.loads((OUT/'freeze.json').read_text())
    current = {name: digest(ROOT/name) for name in CODE}
    if manifest['code_sha256'] != current:
        raise ValueError('Frozen implementation changed')
    for name, checksum in manifest['preserved'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Production artifact changed: ' + name)
    return manifest


def population():
    """The signals population with engineered columns and the frozen cycle-4 split."""
    frame = signals.population(13)
    reserved = {s['huc'][:4] for s in json.loads((LATEST/'stations.json').read_text())
                if s['role'] == 'geographic_test'}
    return frame, split(frame, reserved, cycle=4), sorted(reserved)


def delay_discharge(block):
    """Cool the discharge channel by two hours so its newest value equals the frozen q_2.

    The sequence sprint's window ended at the decision hour inclusive, two hours fresher than the
    frozen candidate is allowed to read: its freshest discharge feature is q_2 under the two-hour
    receipt-delay assumption in scripts/shift_hydrology.py. Delaying the channel per gauge block
    makes every window exactly a window ending at the decision minus two hours, while the weather
    channels keep their six-hour delay, so both candidates and the frozen candidate read the same
    hours of discharge and weather.
    """
    delayed = np.array(block, dtype=block.dtype, copy=True)
    delayed[0, FLOW_DELAY_HOURS:] = block[0, :-FLOW_DELAY_HOURS]
    delayed[0, :FLOW_DELAY_HOURS] = np.nan
    return delayed


def partition_array(part, records, cache):
    """Materialize one partition's windows in memory from the stored per-gauge series."""
    array = np.empty((len(part), len(sequence.CHANNELS), sequence.WINDOW), dtype=np.float32)
    times = pd.to_datetime(part.decision_time, utc=True)
    for station, group in part.groupby('station', sort=True):
        origin, block = sequence.load_series(OUT/'series', records, station, cache)
        array[group.index.to_numpy()] = sequence.windows(delay_discharge(block), origin,
            times.loc[group.index])
    return array


def verify_windows(part, array):
    """Prove the assembled windows are the series the frozen features were computed from."""
    last = sequence.WINDOW - 1
    checks, limits = {}, {}
    for lag in [2, 8, 14, 26, 50, 74]:
        # The discharge channel is delayed two hours, so q_lag sits at last - (lag - 2); q_2 is
        # the newest value in the window.
        checks[f'q_{lag}'] = float(np.nanmax(np.abs(array[:, 0, last - (lag - 2)]
                                                     - part[f'q_{lag}'].to_numpy(float))))
        limits[f'q_{lag}'] = TOLERANCE['flow']
    for lag in [6, 24, 72]:
        rebuilt = array[:, 0, last] - array[:, 0, last - lag]
        checks[f'rise_{lag}'] = float(np.nanmax(np.abs(rebuilt - part[f'rise_{lag}'].to_numpy(float))))
        limits[f'rise_{lag}'] = TOLERANCE['flow']
    rebuilt_range = (np.nanmax(array[:, 0, last - 23:last + 1], axis=1)
                     - np.nanmin(array[:, 0, last - 23:last + 1], axis=1))
    checks['range_24'] = float(np.nanmax(np.abs(rebuilt_range - part['range_24'].to_numpy(float))))
    limits['range_24'] = TOLERANCE['flow']
    checks['rain_6_lag6'] = float(np.nanmax(np.abs(array[:, 1, last - 5:].sum(axis=1)
                                                   - part['rain_6_lag6'].to_numpy(float))))
    limits['rain_6_lag6'] = TOLERANCE['weather']
    checks['temperature_lag6'] = float(np.nanmax(np.abs(array[:, 2, last]
                                                        - part['temperature_lag6'].to_numpy(float))))
    limits['temperature_lag6'] = TOLERANCE['weather']
    if [name for name in checks if checks[name] > limits[name]]:
        raise ValueError('Assembled windows disagree with the frozen features: '
                         + json.dumps(checks))
    return checks


def audit():
    """Record the external-method audit: released Google weights and RiverMamba reforecasts.

    Reads the reference clones under /tmp/planit-ext, verifies the load-bearing quotes, and copies
    the small evidence files into audit/ so the record survives the clones.
    """
    if (OUT/'audit.json').exists():
        raise FileExistsError('Audit already recorded')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'audit').mkdir(exist_ok=True)
    import yaml
    basins = FLOOD/'example-configs/multimet-basins-list-without-chirps.txt'
    metrics_csv = (FLOOD/'pretrained-models/google-floodhub-settings-110-epochs/test/model_epoch110/'
                   'test_metrics.csv')
    config_path = FLOOD/'pretrained-models/google-floodhub-settings-110-epochs/config.yml'
    weights_readme = FLOOD/'pretrained-models/Pretrained-Models-README.md'
    river_readme = RIVERMAMBA/'README.md'
    for path in [basins, metrics_csv, config_path, weights_readme, river_readme]:
        if not path.exists():
            raise FileNotFoundError('Reference clone is missing: ' + str(path))
    config = yaml.safe_load(config_path.read_text())
    weights_text = weights_readme.read_text()
    river_text = river_readme.read_text()
    for quote in ['trained on the FULL historical data period (1982-2023)',
                  'There is NO temporal holdout/test split', '15,955', '10,137']:
        if quote not in weights_text:
            raise ValueError('Released-weights README changed: ' + quote)
    for quote in ['7 time steps representing the 7 days lead time',
                  'The data are provided at daily basis and the timestamp is right-labeled',
                  '3366', '2019-2024', '1979-2018']:
        if quote not in river_text:
            raise ValueError('RiverMamba README changed: ' + quote)
    listed = set(line.strip() for line in basins.read_text().splitlines() if line.strip())
    released = pd.read_csv(metrics_csv).set_index('basin')
    stations = sorted(pd.read_csv(SIGNALS/'development.csv.gz', usecols=['station'],
        dtype={'station': str}).station.unique())
    in_training_list = [s for s in stations if 'camels_' + s in listed]
    in_released_metrics = [s for s in stations if 'camels_' + s in released.index]
    nse = released.loc[['camels_' + s for s in in_released_metrics], 'NSE'].to_numpy(float)
    kge = released.loc[['camels_' + s for s in in_released_metrics], 'KGE'].to_numpy(float)
    # The shipped run config uses dd/mm/yyyy.
    day_start, month_start, year_start = str(config['train_start_date']).split('/')
    day_end, month_end, year_end = str(config['train_end_date']).split('/')
    train_start = f'{year_start}-{month_start}-{day_start}'
    train_end = f'{year_end}-{month_end}-{day_end}'
    overlaps, start, end = {}, pd.Timestamp(train_start), pd.Timestamp(train_end)
    for label, low, high in [('train 2020-2021', '2020-01-01', '2021-12-31'),
            ('calibration 2022', '2022-01-01', '2022-12-31'),
            ('threshold 2023', '2023-01-01', '2023-12-31'),
            ('2024 development partitions', '2024-01-01', '2024-12-31'),
            ('2025-2026 confirmation', '2025-01-01', '2026-12-31')]:
        low, high = pd.Timestamp(low), pd.Timestamp(high)
        inside = max(low, start) <= min(high, end)
        overlaps[label] = dict(overlaps_training_window=bool(inside),
            overlap_window=[str(max(low, start).date()), str(min(high, end).date())]
            if inside else None)
    hindcast = sorted(v for values in config['hindcast_inputs'].values() for v in values)
    forecast = sorted(v for values in config['forecast_inputs'].values() for v in values)
    statics = list(config['static_attributes'])
    sample = json.loads(gzip.decompress((SIGNALS/'weather'/(stations[0] + '.json.gz')).read_bytes()))['hourly']
    weather_coverage = [sample['time'][0], sample['time'][-1]]
    decisions = pd.read_csv(SIGNALS/'development.csv.gz', usecols=['decision_time'])
    first_decision = pd.to_datetime(decisions.decision_time, utc=True).min()
    history_days = int((first_decision - pd.Timestamp(weather_coverage[0], tz='UTC')).days)
    matched_statics = sorted(set(statics) & set(signals.ATTRIBUTES))
    evidence = {}
    for source, name in [(basins, 'multimet-basins-list-without-chirps.txt'),
            (metrics_csv, 'google-floodhub-test-metrics.csv'),
            (config_path, 'google-floodhub-config.yml'),
            (weights_readme, 'google-floodhub-pretrained-README.md'),
            (river_readme, 'rivermamba-README.md')]:
        destination = OUT/'audit'/name
        destination.write_bytes(source.read_bytes())
        evidence[str(destination.relative_to(ROOT))] = digest(destination)
    huggingface = None
    tree_source = REFERENCE/'rm-hf-tree.json'
    readme_source = REFERENCE/'rm-hf-readme.txt'
    if tree_source.exists() and readme_source.exists():
        tree_text, readme_text = tree_source.read_text(), readme_source.read_text()
        for quote in ['7 time steps representing the 7 days lead time',
                      'averaged over 24 hours', '3366']:
            if quote not in readme_text:
                raise ValueError('Hugging Face dataset README changed: ' + quote)
        tree = json.loads(tree_text)
        for source, name in [(tree_source, 'rivermamba-huggingface-tree.json'),
                (readme_source, 'rivermamba-dataset-README.txt')]:
            destination = OUT/'audit'/name
            destination.write_bytes(source.read_bytes())
            evidence[str(destination.relative_to(ROOT))] = digest(destination)
        huggingface = dict(files=[dict(path=item.get('path'), size=item.get('size'))
                                  for item in tree],
            note='Public Hugging Face file listing fetched 2026-10-18; metadata only, no archive '
                 'downloaded. The full-map archive alone is 27.2 GB.')
    google = dict(
        clone=str(FLOOD), model=config['model'], head=config['head'],
        package_version=config['package_version'], epochs=config['epochs'],
        hidden_size=config['hidden_size'], n_distributions=config['n_distributions'],
        train_start=train_start, train_end=train_end,
        window_note='Shipped run config.yml trains, validates and tests on 01/01/1982 to 30/09/2023 '
            'with no temporal holdout; the released README describes the same full-period, no-holdout '
            'design and marks temporary benchmarking on it as invalid.',
        listed_basins=dict(file_lines=len(listed), readme_listed=15955, readme_evaluable=10137,
            released_metric_rows=int(len(released))),
        gauges=dict(ours=len(stations), in_training_list=len(in_training_list),
            in_released_metrics=len(in_released_metrics),
            released_in_sample=dict(nse_median=round(float(np.nanmedian(nse)), 4),
                nse_p10=round(float(np.nanpercentile(nse, 10)), 4),
                nse_p90=round(float(np.nanpercentile(nse, 90)), 4),
                kge_median=round(float(np.nanmedian(kge)), 4),
                note='Released full-period metrics are in-sample for every one of our 88 gauges; '
                     'they support no claim about our task or our 2024 rows.')),
        inputs=dict(hindcast_channels=hindcast, forecast_channels=forecast,
            union_mapping=config['union_mapping'], static_attributes=statics,
            static_count=len(statics), seq_length=config['seq_length'],
            lead_time=config['lead_time'], forecast_overlap=config['forecast_overlap'],
            predict_last_n=config['predict_last_n'], target_variables=config['target_variables'],
            timestep='daily'),
        cached_contract=dict(
            cached_weather_variables=list(signals.USABLE), cached_weather_coverage=weather_coverage,
            cached_discharge='USGS instantaneous discharge pages reassembled to an hourly calendar, '
                '2020-2024, for the 88 gauges',
            first_train_decision=str(first_decision.date()),
            required_history_days=int(config['seq_length']),
            available_history_before_first_train_decision=history_days,
            exact_static_name_matches=matched_statics,
            missing_hindcast_channels=hindcast, missing_forecast_channels=forecast,
            note='None of the required hres/graphcast/imerg/cpc channel names exists in the weather '
                 'cache; only ERA5 precipitation, temperature_2m and soil moisture 0-7cm are cached '
                 '(two of the nine hindcast quantities, under different names), no daily as-issued '
                 'forecast product is cached, and ' + str(len(matched_statics)) + ' of the ' +
                 str(len(statics)) + ' required statics match our cached CAMELS attribute names.'),
        year_overlap=overlaps,
        conclusion='The released weights cannot be applied to our 88 gauges under this protocol '
            'without training overlap contamination. All 88 gauges are in the training basin list as '
            'camels_<id> and all 88 appear in the released in-sample metrics, and the training window '
            'contains our train, calibration and threshold partitions. The 2024 development and '
            '2025-2026 confirmation years are outside the training window, but the daily input '
            'contract cannot be built from the caches, so no zero-shot or fine-tuned application '
            'exists here either. This sprint trains the architecture from scratch under our protocol.')
    rivermamba = dict(
        clone=str(RIVERMAMBA), huggingface=huggingface,
        reforecasts='One file per issue day at 00:00 UTC, seven daily lead steps, discharge averaged '
            'over 24 hours (dis24), for 3,366 AIFAS river points indexed to GRDC stations, or a '
            '0.05-degree global map for 2024 only',
        period='2019-2024 reforecasts from models trained 1979-2018 (AIFAS) or 1979-2023 (full map)',
        conclusion='The released reforecasts are daily. A daily cadence with a one-day minimum lead '
            'cannot express a six-hour decision or a six-hour onset horizon, and the AIFAS index '
            'carries GRDC station identifiers rather than our 88 USGS gauge identifiers, so no '
            'comparison was run and none should be forced.')
    write_json(OUT/'audit.json', dict(google=google, rivermamba=rivermamba, evidence=evidence,
        fetched='2026-10-18', reference_clone=str(REFERENCE)))
    print('Audit: gauges in training list', len(in_training_list), 'of', len(stations),
          '| in released metrics', len(in_released_metrics), '| first decision', first_decision.date(),
          '| cached history days', history_days, flush=True)
    print('Audit: RiverMamba reforecasts are daily with 7 daily lead steps:', bool(huggingface),
          flush=True)


def freeze():
    """Write the protocol manifest before anything is fitted."""
    if (OUT/'freeze.json').exists():
        raise FileExistsError('Protocol already frozen')
    audit_record = json.loads((OUT/'audit.json').read_text())
    sources = json.loads((SIGNALS/'dataset.json').read_text())
    control_freeze = json.loads((CONTROL/'freeze.json').read_text())
    engineered = control_freeze['configurations']['winner_forecast']['features']
    write_json(OUT/'freeze.json', dict(
        experiment='External benchmark: Google Flood Hub architecture trained under this repository protocol',
        label='Google architecture under our protocol. Not Google published performance.',
        revision='The first window implementation reproduced the sequence sprint window ending at '
            'the decision hour inclusive, giving the candidates two hours of fresher discharge than '
            'the frozen candidate is allowed to read, and its trained artifacts are preserved under '
            'superseded-fresh-discharge/ as a development note only. This revision delays the '
            'discharge channel by two hours so the newest value in every window equals q_2; the '
            'delivered comparison uses this contract.',
        audit=dict(file='experiments/high-flow-external-20261018/audit.json',
            sha256=digest(OUT/'audit.json'),
            conclusion_google=audit_record['google']['conclusion'],
            conclusion_rivermamba=audit_record['rivermamba']['conclusion']),
        method=METHOD,
        control=dict(name=CONTROL_NAME, experiment='experiments/high-flow-signals-20261018/cycle-11',
            results_sha256=digest(CONTROL/'results.json'), freeze_sha256=digest(CONTROL/'freeze.json'),
            artifact_sha256=digest(CONTROL/f'{CONTROL_NAME}.joblib'),
            predictions={key: digest(CONTROL/f'{CONTROL_NAME}-{key}.csv.gz') for key in EVALUATED}),
        population=dict(dataset='experiments/high-flow-signals-20261018/development.csv.gz',
            dataset_sha256=sources['sha256'], gauges=sources['gauges'],
            forecast_dataset='experiments/high-flow-signals-20261018/hrrr/features.csv.gz',
            forecast_dataset_sha256=json.loads((SIGNALS/'hrrr/features.json').read_text())['sha256'],
            window_hours=sequence.WINDOW, channels=sequence.CHANNELS,
            weather_delay_hours=sequence.WEATHER_DELAY, discharge_delay_hours=FLOW_DELAY_HOURS,
            window_contract='Channel 0 is delayed two hours at window build so the newest discharge '
                'in every window equals the frozen receipt-delay q_2; the weather channels keep the '
                'six-hour receipt delay. Stored series bytes stay un-delayed and byte-identical to '
                'the sequence sprint, so the provenance check keeps its meaning.',
            reference='log1p(discharge/reference_cfs) rebuilt from the cached USGS pages',
            unit='One row is one eligible decision at the frozen six-hour cadence, horizon 6 only.'),
        engineered=dict(source='cycle-11 freeze.json configurations.winner_forecast.features',
            features=engineered, count=len(engineered),
            note='The engineered branch receives exactly the frozen candidate feature set and no '
                 'feature the frozen candidate could not compute at the decision time. Missing '
                 'entries are standardized with inner-training statistics and filled with zero '
                 'after scaling; the frozen candidate let LightGBM handle missing values natively.'),
        statics=STATICS,
        split=dict(definition='scripts/high_flow_research.split(cycle=4), reserved HUC4s from '
                'experiments/sprint-20261017-mlp-geo/stations.json role=geographic_test',
            commentary='Train 2020-2021, calibrate 2022, threshold 2023, score 2024. Unchanged.'),
        calibration=dict(inputs=CALIBRATION_INPUTS,
            estimator='StandardScaler + LogisticRegression(C=1, max_iter=2000, random_state=42) on the '
                '2022 partition, identical to scripts/high_flow_signals.py',
            requirement='Calibration slope must be positive.'),
        threshold=dict(rule='scripts/high_flow_research.threshold on the 2023 partition, inclusive',
            pooled=THRESHOLD_RULE['pooled'], region=THRESHOLD_RULE['region']),
        candidates=CANDIDATES, architecture=ARCHITECTURE,
        training=dict(**TRAINING, early_stopping='Contiguous tail of the training period by '
            'decision_time (last %d%% of rows), never calibration, threshold or 2024 rows.'
            % int(TRAINING['validation_tail']*100)),
        comparison=dict(reference=CONTROL_NAME, budgets=BUDGETS, ceiling=CEILING,
            draws=500, seed=42, clusters='huc4',
            gates='the sixteen promotion gates of scripts/report_high_flow_signals.py, computed '
                'against the frozen candidate',
            rows='identical evaluation rows, proven against the stored cycle-11 prediction files'),
        storage=dict(artifact='one float16 hourly four-channel series per gauge under series/, about '
                '31 MB, plus one gzipped index per partition, per-partition prediction files and one '
                'model bundle per candidate; well under the 300 MB budget',
            regeneration='Delete series/ and index/ and rerun assemble; freeze.json then still '
                'describes the run.'),
        code_sha256={name: digest(ROOT/name) for name in CODE},
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT/'models').glob('*') if p.is_file()}))
    print('Frozen external protocol', flush=True)


def assemble():
    """Build the per-gauge hourly series and the partition indices, with cross-checks."""
    verify_freeze()
    if (OUT/'series.json').exists():
        raise FileExistsError('Series already assembled')
    (OUT/'series').mkdir(parents=True, exist_ok=True)
    (OUT/'index').mkdir(parents=True, exist_ok=True)
    frozen_series = json.loads((ROOT/'experiments/high-flow-sequence-20261018/series.json').read_text())
    frame, parts, reserved = population()
    times = pd.to_datetime(frame.decision_time, utc=True)
    records = {}
    total = int(frame.station.nunique())
    started = time.monotonic()
    for index, station in enumerate(sorted(frame.station.unique()), 1):
        rows = frame.loc[frame.station.eq(station)]
        reference = rows.reference_cfs.to_numpy(float)
        if len(np.unique(reference)) != 1 or reference[0] <= 0:
            raise ValueError('Ambiguous reference discharge for ' + station)
        origin, block, sources = sequence.station_series(station, times.loc[rows.index], reference[0])
        path = OUT/'series'/(station + '.npy')
        np.save(path, block.astype(np.float16))
        checksum = digest(path)
        frozen = frozen_series['series'].get(station)
        if frozen and (frozen['sha256'] != checksum or frozen['origin'] != origin.isoformat()
                or frozen['hours'] != int(block.shape[1])):
            raise ValueError('Rebuilt series differs from the frozen sequence experiment: ' + station)
        records[station] = dict(file=path.name, origin=origin.isoformat(), hours=int(block.shape[1]),
            dtype='float16', sha256=checksum, bytes=path.stat().st_size, sources=sources)
        if index % 20 == 0 or index == total:
            print('assembled', index, 'of', total, 'gauges', round(time.monotonic() - started, 1),
                  'seconds', flush=True)
    report = dict(gauges=sorted(records), series=records,
        series_bytes=sum(record['bytes'] for record in records.values()), partitions={}, checks={},
        scored_rows=0, elapsed_seconds=None)
    report['window_contract'] = ('Channel 0 delayed %d hours at window build so the newest '
        'discharge in every window equals the frozen receipt-delay q_2; the weather channels keep '
        'the six-hour receipt delay.' % FLOW_DELAY_HOURS)
    write_json(OUT/'series.json', report)
    cache = {}
    frozen_index = ROOT/'experiments/high-flow-sequence-20261018/index'
    for key, part in parts.items():
        array = partition_array(part, records, cache)
        part[COLUMNS].to_csv(OUT/'index'/(key + '.csv.gz'), index=False,
            compression={'method': 'gzip', 'mtime': 0})
        if (frozen_index/(key + '.csv.gz')).exists():
            mirror = pd.read_csv(frozen_index/(key + '.csv.gz'), dtype={'station': str, 'huc': str},
                keep_default_na=False)
            left = part[COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            right = mirror[COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            if not left.equals(right):
                raise ValueError('Partition rows differ from the frozen sequence index: ' + key)
        flow_missing = np.isnan(array[:, 0])
        if flow_missing[:, FLOW_DELAY_HOURS:].any():
            raise ValueError('Discharge windows are incomplete for partition ' + key)
        missing = {name: int(np.isnan(array[:, i]).sum()) for i, name in enumerate(sequence.CHANNELS)}
        missing['log_flow_head_delay'] = int(flow_missing[:, :FLOW_DELAY_HOURS].sum())
        report['partitions'][key] = dict(rows=len(part), gauges=int(part.station.nunique()),
            positives=int(part.target.sum()), start=part.decision_time.min(),
            end=part.decision_time.max(), index_sha256=digest(OUT/'index'/(key + '.csv.gz')),
            missing_entries=missing)
        report['checks'][key] = verify_windows(part, array)
        del array
        print(key, report['partitions'][key]['rows'], 'rows,', report['partitions'][key]['gauges'],
              'gauges, checks', report['checks'][key], flush=True)
    report['reserved_huc4'] = reserved
    report['scored_rows'] = int(sum(report['partitions'][key]['rows'] for key in SCORED))
    report['elapsed_seconds'] = time.monotonic() - started
    write_json(OUT/'series.json', report)


def fit():
    """Train, calibrate, threshold and score every candidate under the frozen protocol."""
    verify_freeze()
    report = json.loads((OUT/'series.json').read_text())
    for station, record in report['series'].items():
        if digest(OUT/'series'/record['file']) != record['sha256']:
            raise ValueError('Stored series changed: ' + station)
    for key, record in report['partitions'].items():
        if digest(OUT/'index'/(key + '.csv.gz')) != record['index_sha256']:
            raise ValueError('Stored index changed: ' + key)
    if (OUT/'results.json').exists():
        raise FileExistsError('Candidates already fitted')
    (OUT/'predictions').mkdir(parents=True, exist_ok=True)
    frame, parts, reserved = population()
    control_freeze = json.loads((CONTROL/'freeze.json').read_text())
    engineered = control_freeze['configurations']['winner_forecast']['features']
    missing = [name for name in engineered + STATICS if name not in frame.columns]
    if missing:
        raise ValueError('Engineered feature missing from the population: ' + ', '.join(missing))
    order = parts['train'].sort_values('decision_time').index.to_numpy()
    cut = int(len(order)*(1 - TRAINING['validation_tail']))
    boundary = parts['train'].decision_time.iloc[order[cut:]].min()
    records = []
    for candidate in CANDIDATES:
        name = candidate['name']
        job = dict(name=name, architecture=ARCHITECTURE, training=TRAINING, window=sequence.WINDOW,
            channels=sequence.CHANNELS, series=str(OUT/'series.json'), directory=str(OUT/'series'),
            index=str(OUT/'index'), partitions=['train'], scored=SCORED, engineered_cycle=13,
            engineered_features=engineered if candidate['engineered'] else [],
            statics=STATICS if candidate['statics'] else [],
            inner_train_rows=order[:cut].tolist(), inner_validation_rows=order[cut:].tolist(),
            output=str(OUT/f'{name}-logits.csv.gz'), summary=str(OUT/f'{name}-worker.json'),
            bundle=str(OUT/f'{name}-model.pt'), threads=4)
        write_json(OUT/f'{name}-fit-job.json', job)
        environment = dict(os.environ, OMP_NUM_THREADS='4', KMP_DUPLICATE_LIB_OK='TRUE',
            MPLCONFIGDIR='/tmp')
        started = time.monotonic()
        result = subprocess.run([sys.executable, '-m', 'scripts.high_flow_external_worker',
                str(OUT/f'{name}-fit-job.json')], cwd=ROOT, capture_output=True, text=True,
            env=environment)
        if result.returncode != 0:
            raise ValueError('External worker failed: ' + result.stderr[-2000:])
        print(result.stdout.strip(), flush=True)
        worker = json.loads((OUT/f'{name}-worker.json').read_text())
        logits = pd.read_csv(OUT/f'{name}-logits.csv.gz', dtype={'station': str}, keep_default_na=False)
        def raw(key):
            part = parts[key]
            aligned = part[['station', 'decision_time']].merge(
                logits.loc[logits.partition.eq(key), ['station', 'decision_time', 'logit']],
                on=['station', 'decision_time'], how='left', validate='one_to_one')
            if aligned.logit.isna().any():
                raise ValueError('Missing logits for a scored row of ' + key)
            inputs = pd.DataFrame({'score': aligned.logit.to_numpy(float)})
            for column in CALIBRATION_INPUTS:
                inputs[column] = part[column].to_numpy()
            return inputs
        calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000,
            random_state=42))
        calibrator.fit(raw('calibration'), parts['calibration'].target)
        if calibrator[-1].coef_[0, 0] <= 0:
            raise ValueError('Nonpositive calibration slope: ' + name)
        probabilities = {key: calibrator.predict_proba(raw(key))[:, 1] for key in SCORED}
        cut_threshold = threshold(parts['threshold'], probabilities['threshold'],
            THRESHOLD_RULE['pooled'], THRESHOLD_RULE['region'])
        scores = {}
        for key in SCORED:
            part = parts[key]
            prediction = part[COLUMNS].copy()
            prediction['probability'] = probabilities[key]
            prediction.to_csv(OUT/'predictions'/f'{name}-{key}.csv.gz', index=False,
                compression={'method': 'gzip', 'mtime': 0})
            if key in EVALUATED:
                scores[key] = metrics(prediction, probabilities[key], cut_threshold)
        bundle = dict(calibrator=calibrator, threshold=cut_threshold, architecture=ARCHITECTURE,
            features=sequence.CHANNELS, engineered=engineered if candidate['engineered'] else [],
            statics=STATICS if candidate['statics'] else [], window=sequence.WINDOW,
            weather_delay=sequence.WEATHER_DELAY, candidate=candidate, worker=worker)
        joblib.dump(bundle, OUT/f'{name}.joblib', compress=3)
        record = dict(name=name, candidate=candidate, threshold=cut_threshold, metrics=scores,
            artifact_sha256=digest(OUT/f'{name}.joblib'),
            calibration=dict(coef=calibrator[-1].coef_.tolist(), inputs=CALIBRATION_INPUTS,
                rows=int(len(parts['calibration']))),
            training=dict(epochs_run=worker['epochs_run'],
                validation_log_loss=worker['validation_log_loss'],
                parameters=worker['parameters'], inner_train_rows=cut,
                inner_validation_rows=len(order) - cut, validation_boundary=str(boundary),
                seconds=worker['seconds']),
            seconds=time.monotonic() - started)
        records.append(record)
        write_json(OUT/'results.json', records)
        print(name, {key: (value['detected_events'], value['events'], round(value['ap'], 4),
              round(value['false_alerts_per_100'], 3)) for key, value in scores.items()}, flush=True)
    for name, checksum in json.loads((OUT/'freeze.json').read_text())['preserved'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Production artifact changed: ' + name)


def partition_metrics(frame, probability, cut, regions_out=True):
    """Partition metrics with the worst-region summary the report script computes."""
    measured = metrics(frame.assign(probability=probability), probability, cut)
    regions = measured.pop('regions')
    worst = max(regions, key=lambda key: regions[key]['false_alerts_per_100']) if regions else None
    warm = regions[worst]['false_alerts_per_100'] if regions else 0.
    measured['worst_region'] = worst
    measured['worst_region_false_alerts_per_100'] = float(warm)
    measured['false_alert_ceiling_met'] = bool(
        measured['false_alerts_per_100'] <= CEILING['pooled'] and warm <= CEILING['region'])
    if regions_out:
        measured['regions'] = {key: dict(rows=value['rows'], events=value['events'],
            false_alerts_per_100=value['false_alerts_per_100'], ap=value['ap'])
            for key, value in regions.items()}
    return measured


def evaluate():
    """Compare each candidate with the frozen cycle-11 candidate on identical rows."""
    verify_freeze()
    manifest = json.loads((OUT/'freeze.json').read_text())
    control_meta = manifest['control']
    if digest(CONTROL/'results.json') != control_meta['results_sha256']:
        raise ValueError('Control results changed')
    if digest(CONTROL/f'{CONTROL_NAME}.joblib') != control_meta['artifact_sha256']:
        raise ValueError('Control artifact changed')
    for key in EVALUATED:
        if digest(CONTROL/f'{CONTROL_NAME}-{key}.csv.gz') != control_meta['predictions'][key]:
            raise ValueError('Control predictions changed: ' + key)
    control_record = next(record for record in json.loads((CONTROL/'results.json').read_text())
                          if record['name'] == CONTROL_NAME)
    bundle = joblib.load(CONTROL/f'{CONTROL_NAME}.joblib')
    frame, parts, reserved = population()
    def control_raw(part):
        probability = np.clip(bundle['model'].predict_proba(part[bundle['features']])[:, 1],
            1e-6, 1 - 1e-6)
        inputs = pd.DataFrame({'score': np.log(probability/(1 - probability))})
        for column in CALIBRATION_INPUTS:
            inputs[column] = part[column].to_numpy()
        return inputs
    control_probability = {key: bundle['calibrator'].predict_proba(control_raw(parts[key]))[:, 1]
                           for key in ['threshold'] + EVALUATED}
    reproduced = threshold(parts['threshold'], control_probability['threshold'],
        THRESHOLD_RULE['pooled'], THRESHOLD_RULE['region'])
    if reproduced != bundle['threshold']:
        raise ValueError('Frozen threshold did not reproduce')
    integrity = dict(threshold=dict(stored=bundle['threshold'], reproduced=reproduced),
        max_abs_probability_delta={})
    stored_control = {}
    for key in EVALUATED:
        stored = pd.read_csv(CONTROL/f'{CONTROL_NAME}-{key}.csv.gz', dtype={'station': str, 'huc': str},
            keep_default_na=False)
        stored_control[key] = stored
        aligned = parts[key][['station', 'decision_time']].merge(
            stored[['station', 'decision_time', 'probability']], on=['station', 'decision_time'],
            how='left', validate='one_to_one')
        delta = float(np.nanmax(np.abs(control_probability[key]
                                       - aligned.probability.to_numpy(float))))
        integrity['max_abs_probability_delta'][key] = delta
        if not np.isfinite(delta) or delta > 1e-9:
            raise ValueError('Frozen candidate did not reproduce: ' + key)
    comparison = dict(draws=500, seed=42, clusters='huc4', label=manifest['label'],
        control=str((CONTROL/f'{CONTROL_NAME}-temporal_validation.csv.gz').relative_to(ROOT)),
        control_results_sha256=control_meta['results_sha256'], control_integrity=integrity,
        candidates={}, prediction_sha256={})
    for record in json.loads((OUT/'results.json').read_text()):
        name = record['name']
        if digest(OUT/f'{name}.joblib') != record['artifact_sha256']:
            raise ValueError('Candidate artifact changed: ' + name)
        mine = {key: pd.read_csv(OUT/'predictions'/f'{name}-{key}.csv.gz',
            dtype={'station': str, 'huc': str}, keep_default_na=False) for key in SCORED}
        comparison['prediction_sha256'][name] = {
            key: digest(OUT/'predictions'/f'{name}-{key}.csv.gz') for key in SCORED}
        for key in SCORED:
            rows = mine[key][COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            expected = parts[key][COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            if not rows.equals(expected):
                raise ValueError('Prediction rows drifted from the partition: ' + key + ' ' + name)
        entry = dict(threshold=record['threshold'], partitions={}, pooled={}, matched_budgets={})
        frames, control_probs, candidate_probs = {}, {}, {}
        for key in EVALUATED:
            left = mine[key][COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            right = stored_control[key][COLUMNS].sort_values(['station', 'decision_time']).reset_index(drop=True)
            if not left.equals(right):
                raise ValueError('Candidate and control are not scored on identical rows: ' + key)
            frames[key] = left
            control_probs[key] = stored_control[key].sort_values(
                ['station', 'decision_time']).probability.to_numpy(float)
            candidate_probs[key] = mine[key].sort_values(
                ['station', 'decision_time']).probability.to_numpy(float)
            measured = partition_metrics(left, candidate_probs[key], record['threshold'])
            control_measured = partition_metrics(left, control_probs[key], control_record['threshold'])
            entry['partitions'][key] = dict(rows=len(left),
                candidate=dict(threshold=record['threshold'], metrics=measured),
                control=dict(threshold=control_record['threshold'], metrics=control_measured),
                paired=paired_uncertainty(left, control_probs[key], candidate_probs[key],
                    control_record['threshold'], record['threshold'], draws=comparison['draws'],
                    seed=comparison['seed']),
                lead=jointly_detected_lead_delta(left, control_probs[key], candidate_probs[key],
                    control_record['threshold'], record['threshold']),
                brier_delta=float(measured['brier'] - control_measured['brier']))
        pooled_left = pd.concat([frames[key] for key in EVALUATED], ignore_index=True)
        pooled_control = np.concatenate([control_probs[key] for key in EVALUATED])
        pooled_candidate = np.concatenate([candidate_probs[key] for key in EVALUATED])
        entry['pooled'] = dict(
            candidate=partition_metrics(pooled_left, pooled_candidate, record['threshold'], regions_out=False),
            control=partition_metrics(pooled_left, pooled_control, control_record['threshold'], regions_out=False),
            paired=paired_uncertainty(pooled_left, pooled_control, pooled_candidate,
                control_record['threshold'], record['threshold'], draws=comparison['draws'],
                seed=comparison['seed']))
        stored = mine['threshold'][['station', 'decision_time', 'probability']].copy()
        ordered = parts['threshold'][['station', 'decision_time']].merge(stored,
            on=['station', 'decision_time'], how='left', validate='one_to_one')
        if ordered.probability.isna().any():
            raise ValueError('Missing candidate threshold-partition probabilities: ' + name)
        candidate_cuts = ordered.probability.to_numpy(float)
        for budget in BUDGETS:
            control_cut = threshold(parts['threshold'], control_probability['threshold'], budget,
                THRESHOLD_RULE['region'])
            candidate_cut = threshold(parts['threshold'], candidate_cuts, budget,
                THRESHOLD_RULE['region'])
            budget_entry = dict(control_threshold=control_cut, candidate_threshold=candidate_cut,
                partitions={})
            for key in EVALUATED:
                budget_entry['partitions'][key] = dict(
                    control=partition_metrics(frames[key], control_probs[key], control_cut, regions_out=False),
                    candidate=partition_metrics(frames[key], candidate_probs[key], candidate_cut, regions_out=False),
                    paired=paired_uncertainty(frames[key], control_probs[key], candidate_probs[key],
                        control_cut, candidate_cut, draws=comparison['draws'], seed=comparison['seed']),
                    lead=jointly_detected_lead_delta(frames[key], control_probs[key],
                        candidate_probs[key], control_cut, candidate_cut))
            budget_entry['pooled'] = dict(
                control=partition_metrics(pooled_left, pooled_control, control_cut, regions_out=False),
                candidate=partition_metrics(pooled_left, pooled_candidate, candidate_cut, regions_out=False),
                paired=paired_uncertainty(pooled_left, pooled_control, pooled_candidate, control_cut,
                    candidate_cut, draws=comparison['draws'], seed=comparison['seed']))
            entry['matched_budgets'][str(budget)] = budget_entry
        gates = {'at_least_20_events_per_partition': True, 'at_least_50_events_total': True}
        total = 0
        for key in EVALUATED:
            values = entry['partitions'][key]
            prefix = key.replace('_validation', '_test')
            events = values['candidate']['metrics']['events']
            total += events
            gates[prefix + '_events'] = events >= 20
            interval = values['paired']['ap_difference_95']
            gates[prefix + '_ap'] = bool(interval and interval[0] > 0)
            recall = values['paired']['event_recall_difference']
            gates[prefix + '_event_gain'] = bool(recall is not None and recall >= .05)
            gates[prefix + '_brier'] = bool(values['brier_delta'] is not None and values['brier_delta'] <= 0)
            gates[prefix + '_false_alerts'] = bool(
                values['candidate']['metrics']['false_alert_ceiling_met'])
            lead = values['lead']['median_lead_delta_hours']
            gates[prefix + '_lead'] = bool(lead is not None and lead >= 0)
        gates['total_events'] = total >= 50
        pooled_interval = entry['pooled']['paired'].get('event_recall_difference_95')
        gates['pooled_event_uncertainty'] = bool(pooled_interval and pooled_interval[0] > 0)
        entry['gates'] = gates
        entry['passed'] = all(gates.values())
        entry['candidate'] = name
        entry['reference'] = CONTROL_NAME
        entry['evidence'] = 'Inspected development partitions. Not independent confirmation.'
        comparison['candidates'][name] = entry
    write_json(OUT/'comparison.json', comparison)
    for name, entry in comparison['candidates'].items():
        for key in EVALUATED:
            values = entry['partitions'][key]
            control_metrics = values['control']['metrics']
            candidate_metrics = values['candidate']['metrics']
            paired = values['paired']
            print(name, key, 'candidate', candidate_metrics['detected_events'], '/',
                  candidate_metrics['events'], 'AP', round(candidate_metrics['ap'], 4),
                  'FA100', round(candidate_metrics['false_alerts_per_100'], 3), '| control',
                  control_metrics['detected_events'], '/', control_metrics['events'], 'AP',
                  round(control_metrics['ap'], 4), '| dAP', round(paired['ap_difference'], 4),
                  'low', None if not paired['ap_difference_95'] else round(paired['ap_difference_95'][0], 4),
                  '| dRec', round(paired['event_recall_difference'], 4), flush=True)
        pooled = entry['pooled']['paired']
        print(name, 'pooled dRec', round(pooled['event_recall_difference'], 4), 'low',
              None if not pooled['event_recall_difference_95'] else round(pooled['event_recall_difference_95'][0], 4),
              '| gates passed', entry['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'freeze', 'assemble', 'fit', 'evaluate'])
    arguments = parser.parse_args()
    {'audit': audit, 'freeze': freeze, 'assemble': assemble, 'fit': fit,
     'evaluate': evaluate}[arguments.command]()
