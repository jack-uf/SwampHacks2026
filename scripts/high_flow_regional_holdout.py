"""Whole-region holdout check for the six-hour signal candidate.

The standard geographic partition reserves HUC4 groups. This experiment removes entire HUC2 regions
from fitting, calibration and threshold selection, then scores the held-out region's 2024 rows. A
region the candidate has never seen is a harder transfer test than a reserved HUC4 inside a region it
has seen, and it is the closest local approximation to the independent cohort that still needs data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.flood_sprint import write_json
from scripts.high_flow_research import LATEST
from scripts.high_flow_signals import OUT, configurations, fit_candidate, population
from scripts.hydrology_research import ROOT, digest
from scripts.shift_hydrology_metrics import paired_uncertainty

REGIONS = ['04', '05', '11', '12', '17', '18']
CANDIDATES = ['control_conditional', 'rain6_ratio_wetness_hydro', 'forecast_scaled_wetness',
    'linear_conditional']
REFERENCE = 'linear_conditional'
DESTINATION = OUT/'regional-holdout'
CYCLE = 8


def partitions(frame, reserved, region):
    times = pd.to_datetime(frame.decision_time, utc=True)
    end = times + pd.Timedelta(hours=24)
    inside = frame.huc.str[:2].eq(region)
    geographic = frame.huc.str[:4].isin(reserved)
    return dict(
        train=frame.loc[~inside & ~geographic & times.lt('2022-01-01') & end.lt('2022-01-01')].reset_index(drop=True),
        calibration=frame.loc[~inside & ~geographic & times.ge('2022-01-01') & end.lt('2023-01-01')].reset_index(drop=True),
        threshold=frame.loc[~inside & times.ge('2023-01-01') & end.lt('2024-01-01')].reset_index(drop=True),
        evaluation=frame.loc[inside & times.ge('2024-01-01') & end.lt('2025-01-01')].reset_index(drop=True))


def run(draws=500):
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    frame = population(CYCLE)
    weather = {delay: [column for column in frame if column.endswith(f'_lag{delay}')] for delay in [6, 24, 144]}
    configs = configurations(CYCLE, weather)
    reserved = {s['huc'][:4] for s in json.loads((LATEST/'stations.json').read_text()) if s['role'] == 'geographic_test'}
    report = dict(code_sha256=digest(ROOT/'scripts/high_flow_regional_holdout.py'),
        dataset_sha256=digest(OUT/'development.csv.gz'), regions=REGIONS, candidates=CANDIDATES,
        reference=REFERENCE,
        design='Train 2020-2021, calibrate 2022, threshold 2023, all with the region removed; score that region in 2024.',
        results={})
    for region in REGIONS:
        parts = partitions(frame, reserved, region)
        if parts['evaluation'].empty or parts['evaluation'].target.nunique() < 2:
            report['results'][region] = dict(skipped='Region has no usable 2024 evaluation rows')
            print(region, 'skipped', flush=True)
            continue
        folder = DESTINATION/region
        folder.mkdir(parents=True, exist_ok=True)
        probabilities = {}
        entry = dict(rows=len(parts['evaluation']), events=int(parts['evaluation'].target.sum()),
            stations=int(parts['evaluation'].station.nunique()), candidates={})
        for name in CANDIDATES:
            record = fit_candidate(name, configs[name], parts, folder, evaluation=parts['evaluation'], cycle=CYCLE)
            predictions = pd.read_csv(folder/(name+'-holdout.csv.gz'), dtype={'station': str, 'huc': str},
                keep_default_na=False)
            probabilities[name] = (predictions.pop('probability').to_numpy(float), predictions, record['threshold'])
            entry['candidates'][name] = record['metrics']['holdout']
        values_reference, frame_reference, threshold_reference = probabilities[REFERENCE]
        for name, (values, rows, cut) in probabilities.items():
            entry['candidates'][name]['paired_against_reference'] = paired_uncertainty(rows, values_reference,
                values, threshold_reference, cut, draws=draws)
        report['results'][region] = entry
        summary = {name: (entry['candidates'][name]['detected_events'], round(entry['candidates'][name]['ap'], 3))
                   for name in CANDIDATES}
        print(region, 'events', entry['events'], summary, flush=True)
    write_json(DESTINATION/'summary.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draws', type=int, default=500)
    arguments = parser.parse_args()
    run(arguments.draws)
