"""Multimodal pilot coordinator: satellite and Jev features, then the ablation.

See docs/MULTIMODAL_SPRINT_PROTOCOL.md. Model fitting happens in
scripts/multimodal_worker.py, one process per candidate. This pilot is exploratory:
the July to October 2026 window was already used by the transformer sprint.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import jev_image, s2_pipeline, wind_sprint
from scripts.common import ROOT
from scripts.wind_sprint import EXTENDED, digest, station_metrics, write_json

OUT = ROOT / 'experiments/sprint-20261017-multimodal'
TRANSFORMER = ROOT / 'experiments/sprint-20261017-transformer'
S2_CACHE = ROOT / 'data/raw/s2'
WINDOW = ('2026-07-01', '2026-10-14')
FIT_END = '2026-08-31'
GEO_TEST_START = '2026-09-01'
WEATHER_XGBOOST = 'weather_xgboost'
CANDIDATES = ('weather_xgboost', 'weather_sat_xgboost', 'weather_jev_xgboost',
              'weather_sat_jev_xgboost', 'weather_sat_jev_tabpfn')
SAT_FEATURES = ['s2_days_since', 's2_scene_cloud', 's2_local_cloud', 's2_valid_fraction', 's2_ndvi_mean',
                's2_ndvi_p10', 's2_ndvi_p90', 's2_ndwi_mean', 's2_ndwi_p10', 's2_ndwi_p90',
                's2_water_fraction', 's2_vegetation_fraction', 's2_bare_fraction', 's2_missing']
JEV_FEATURES = ['jev_water_visible', 'jev_water_extent', 'jev_p_extent_1', 'jev_p_extent_2', 'jev_ground_vegetated',
                'jev_ground_water_dominant', 'jev_ground_cloud_or_masked', 'jev_confidence', 'jev_missing']
MAX_SCENES, MAX_TOKENS = 400, 600000


def feature_sets():
    weather = list(EXTENDED)
    return dict(weather_xgboost=weather,
                weather_sat_xgboost=weather + SAT_FEATURES,
                weather_jev_xgboost=weather + JEV_FEATURES,
                weather_sat_jev_xgboost=weather + SAT_FEATURES + JEV_FEATURES,
                weather_sat_jev_tabpfn=weather + SAT_FEATURES + JEV_FEATURES)


def dates_for(role):
    start = WINDOW[0] if role == 'development' else GEO_TEST_START
    first, last = date.fromisoformat(start), date.fromisoformat(WINDOW[1])
    return [(first + timedelta(days=offset)).isoformat() for offset in range((last - first).days + 1)]


def jev_evidence():
    totals = dict(calls=0, input_tokens=0, output_tokens=0, seconds=0.0)
    for path in sorted((OUT / 'features/jev').glob('*.json')):
        payload = json.loads(path.read_text())
        if payload.get('status') == 'ok':
            totals['calls'] += 1
            totals['seconds'] += payload.get('seconds') or 0
            totals['input_tokens'] += payload['usage']['input_tokens']
            totals['output_tokens'] += payload['usage']['output_tokens']
    return totals


def build():
    """Render satellite windows and ask Jev once per scene; resume from cache."""
    (OUT / 'features/s2').mkdir(parents=True, exist_ok=True)
    (OUT / 'features/jev').mkdir(parents=True, exist_ok=True)
    protocol = wind_sprint.load_protocol()
    stations = [station for station in protocol['stations'] if station['role'] in ('development', 'geographic_test')]
    for station in stations:
        sid = station['sid']
        cache_path = OUT / 'features/s2' / f'{sid}.json'
        rows = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        for day in dates_for(station['role']):
            if day in rows:
                continue
            row = s2_pipeline.scene(sid, station['latitude'], station['longitude'], day)
            rows[day] = row
            if row.get('item'):
                jev_path = OUT / 'features/jev' / f"{sid}_{row['item']}.json"
                if not jev_path.exists():
                    totals = jev_evidence()
                    if totals['calls'] >= MAX_SCENES or totals['input_tokens'] + totals['output_tokens'] >= MAX_TOKENS:
                        raise RuntimeError(f'Budget stop: {totals}')
                    png = S2_CACHE / f"{sid}_{row['item']}.png"
                    result = jev_image.ask(png)
                    jev_path.write_text(json.dumps(dict(result, station=sid, item=row['item'], day=day), indent=2) + '\n')
            cache_path.write_text(json.dumps(rows, indent=2) + '\n')
            print(sid, day, row.get('item') or 'missing', round(row.get('local_cloud_fraction') or -1, 2), flush=True)
    print('evidence', jev_evidence(), flush=True)


def satellite_rows():
    records = []
    protocol = wind_sprint.load_protocol()
    roles = {station['sid']: station['role'] for station in protocol['stations']}
    for path in sorted((OUT / 'features/s2').glob('*.json')):
        sid = path.stem
        for day, row in sorted(json.loads(path.read_text()).items()):
            record = dict(station=sid, role=roles[sid], target_date=day, s2_missing=1.0, jev_missing=1.0)
            if row.get('item'):
                record.update(s2_missing=0.0, s2_days_since=row['days_since'], s2_scene_cloud=row['scene_cloud_cover'],
                              s2_local_cloud=row['local_cloud_fraction'], s2_valid_fraction=row['valid_fraction'],
                              s2_ndvi_mean=row['ndvi_mean'], s2_ndvi_p10=row['ndvi_p10'], s2_ndvi_p90=row['ndvi_p90'],
                              s2_ndwi_mean=row['ndwi_mean'], s2_ndwi_p10=row['ndwi_p10'], s2_ndwi_p90=row['ndwi_p90'],
                              s2_water_fraction=row['water_fraction'], s2_vegetation_fraction=row['vegetation_fraction'],
                              s2_bare_fraction=row['bare_fraction'], s2_item=row['item'])
                jev_path = OUT / 'features/jev' / f"{sid}_{row['item']}.json"
                if jev_path.exists():
                    jev = json.loads(jev_path.read_text())
                    if jev.get('status') == 'ok':
                        record.update(jev_missing=0.0, jev_water_visible=jev['water_visible'],
                                      jev_water_extent=jev['water_extent'],
                                      jev_p_extent_1=jev['water_extent_probabilities']['1'],
                                      jev_p_extent_2=jev['water_extent_probabilities']['2'],
                                      jev_ground_vegetated=jev['ground_probabilities']['vegetated'],
                                      jev_ground_water_dominant=jev['ground_probabilities']['water_dominant'],
                                      jev_ground_cloud_or_masked=jev['ground_probabilities']['cloud_or_masked'],
                                      jev_confidence=min(jev['water_extent_confidence'], jev['ground_confidence']))
            records.append(record)
    return pd.DataFrame(records)


def analysis_table():
    fresh = pd.read_csv(TRANSFORMER / 'dataset-extension.csv.gz')
    weather = fresh[fresh.role.isin(['development', 'geographic_test'])
                    & fresh.target_date.between(*WINDOW)].copy()
    table = weather.merge(satellite_rows(), on=['station', 'target_date', 'role'], how='left', validate='one_to_one')
    for column in SAT_FEATURES + JEV_FEATURES:
        table[column] = table[column].astype(float)
    if table.duplicated(['station', 'target_date']).any():
        raise ValueError('Duplicate station-day after the satellite join')
    return table


def prepare_frames(name, table):
    features_dir = OUT / 'frames' / name
    features_dir.mkdir(parents=True, exist_ok=True)
    development = table.role.eq('development')
    train = table[development & (table.target_date <= FIT_END)]
    test = table[development & (table.target_date >= GEO_TEST_START)]
    if (table[table.role.eq('geographic_test')].target_date < GEO_TEST_START).any():
        raise ValueError('Geographic-test stations must not appear before the geographic split start')
    geo = table[table.role.eq('geographic_test') & (table.target_date >= GEO_TEST_START)]
    if geo.station.isin(train.station).any() or test.station.isin(geo.station).any():
        raise ValueError('Station roles cross the pilot split')
    features = feature_sets()[name]
    (features_dir / 'features.json').write_text(json.dumps(dict(features=features), indent=2) + '\n')
    (features_dir / 'splits.json').write_text(json.dumps(dict(splits=['test', 'geo_test']), indent=2) + '\n')
    for label, frame in [('train', train), ('test', test), ('geo_test', geo)]:
        frame.to_csv(features_dir / f'{label}.csv', index=False)
    return dict(train=len(train), test=len(test), geo_test=len(geo),
                positives=[int(train.exceedance.sum()), int(test.exceedance.sum()), int(geo.exceedance.sum())])


def run_worker(name, features_dir, out_dir):
    command = [sys.executable, '-m', 'scripts.multimodal_worker', name, str(features_dir), str(out_dir)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1'))
    if result.returncode:
        raise RuntimeError(f'Worker {name} failed: ' + result.stderr[-800:])
    return json.loads((out_dir / f'{name}.meta.json').read_text())


def load(name, split, out_dir):
    return np.load(out_dir / f'{name}.{split}.npy')


def matched_threshold(scores, y, budget_fraction):
    negatives = np.sort(np.asarray(scores)[~np.asarray(y).astype(bool)])[::-1]
    if not len(negatives):
        raise ValueError('No negatives to match a budget against')
    budget = int(np.ceil(budget_fraction * len(negatives)))
    if budget <= 0:
        return float(np.nextafter(np.max(scores), np.inf))
    return float(negatives[min(budget, len(negatives)) - 1])


def bootstrap_ap(frame, y, baseline, candidate, replicates=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, baseline, candidate = np.asarray(y), np.asarray(baseline), np.asarray(candidate)
    values = []
    for _ in range(replicates):
        ids = rng.integers(0, len(y), len(y))
        if y[ids].sum() == 0:
            continue
        from sklearn.metrics import average_precision_score
        values.append(average_precision_score(y[ids], candidate[ids]) - average_precision_score(y[ids], baseline[ids]))
    bounds = np.quantile(values, [.025, .975]) if values else [np.nan, np.nan]
    return dict(replicates=len(values), method='station-day percentile bootstrap, seed 42',
                delta_ap_95=[float(bounds[0]), float(bounds[1])])


def analyze():
    table = analysis_table()
    table.to_csv(OUT / 'features.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    predictions_dir = OUT / 'probabilities'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    frame_summary, metadata = {}, {}
    for name in CANDIDATES:
        frame_summary[name] = prepare_frames(name, table)
        metadata[name] = run_worker(name, OUT / 'frames' / name, predictions_dir)
        print(name, metadata[name]['fit_seconds'], 's;', frame_summary[name], flush=True)
    report = dict(created_at=pd.Timestamp.utcnow().isoformat(), candidates={}, frames=frame_summary,
                  jev_evidence=jev_evidence(), protocol='docs/MULTIMODAL_SPRINT_PROTOCOL.md',
                  dataset_sha256=digest(OUT / 'features.csv.gz'),
                  notice='Exploratory pilot: the July to October 2026 window was already inspected by the transformer sprint')
    baseline = WEATHER_XGBOOST
    for name in CANDIDATES:
        entry = dict(metadata=metadata[name])
        for split in ['test', 'geo_test']:
            frame = pd.read_csv(OUT / 'frames' / name / f'{split}.csv')
            probability = load(name, split, predictions_dir)
            entry[split] = dict(rows=len(frame), positives=int(frame.exceedance.sum()),
                                states=sorted(set(frame.state)), stations=int(frame.station.nunique()))
            entry.setdefault('metrics', {})[split] = station_metrics(frame.exceedance, probability, .5)
            if split == 'test' and name != baseline:
                baseline_probability = load(baseline, 'test', predictions_dir)
                entry['delta_vs_weather'] = bootstrap_ap(frame, frame.exceedance, baseline_probability, probability)
        train_frame = pd.read_csv(OUT / 'frames' / name / 'train.csv')
        train_probability = load(name, 'train', predictions_dir)
        entry['thresholds'] = {}
        report['candidates'][name] = entry
    baseline_train = load(baseline, 'train', predictions_dir)
    baseline_frame = pd.read_csv(OUT / 'frames' / baseline / 'train.csv')
    baseline_frame = pd.read_csv(OUT / 'frames' / baseline / 'train.csv')
    negatives = int((baseline_frame.exceedance == 0).sum())
    false_alerts = int(((baseline_train >= .5) & (baseline_frame.exceedance == 0)).sum())
    budget = false_alerts / negatives
    report['budget_fraction'] = float(budget)
    for name in CANDIDATES:
        train_frame = pd.read_csv(OUT / 'frames' / name / 'train.csv')
        train_probability = load(name, 'train', predictions_dir)
        threshold = matched_threshold(train_probability, train_frame.exceedance, budget)
        report['candidates'][name]['matched_threshold'] = threshold
        for split in ['test', 'geo_test']:
            frame = pd.read_csv(OUT / 'frames' / name / f'{split}.csv')
            probability = load(name, split, predictions_dir)
            report['candidates'][name]['metrics'][f'{split}_matched'] = station_metrics(frame.exceedance, probability, threshold)
    failures = []
    best_satellite = None
    for name in ['weather_sat_xgboost', 'weather_jev_xgboost', 'weather_sat_jev_xgboost', 'weather_sat_jev_tabpfn']:
        interval = report['candidates'][name].get('delta_vs_weather', {})
        gain = interval.get('delta_ap_95', [float('nan'), float('nan')])[0] > 0
        recall_ok = (report['candidates'][name]['metrics']['test_matched']['recall']
                     >= report['candidates'][baseline]['metrics']['test_matched']['recall'] - 1e-12)
        if gain and recall_ok:
            best_satellite = name
            break
    report['signal'] = best_satellite
    report['decision'] = ('satellite_features_add_value' if best_satellite else 'no_measurable_satellite_value')
    write_json(OUT / 'evaluation.json', report)
    print('decision', report['decision'], 'signal', best_satellite, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'evidence', 'analyze'])
    args = parser.parse_args()
    dict(build=build, evidence=lambda: print(json.dumps(jev_evidence(), indent=2)), analyze=analyze)[args.command]()

