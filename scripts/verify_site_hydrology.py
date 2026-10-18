"""Offline feature and inference parity on inspected historical observations, never an availability replay."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend import hydrology
from backend.domain import now_utc, parse_time
from scripts.common import ROOT
from scripts.flood_sprint import write_json
from scripts.hydrology_research import digest


def verify():
    destination = ROOT/'experiments/site-hydrology-20261018/historical-parity.json'
    if destination.exists():
        raise FileExistsError(destination)
    candidate, baseline = hydrology.artifact('candidate'), hydrology.artifact('baseline')
    source = ROOT/'experiments/high-flow-20261018/predictions'
    control = pd.read_csv(source/'cycle-4-linear-conditional-temporal_validation.csv.gz', dtype={'station': str})
    target = pd.read_csv(source/'cycle-4-expanded_original-temporal_validation.csv.gz', dtype={'station': str})
    common = target.merge(control[['station', 'decision_time', 'probability']],
                          on=['station', 'decision_time'], suffixes=('', '_baseline'), validate='one_to_one')
    results, hashes = [], {}
    # Stable metadata order and evenly spaced existing decisions; no outcome-based sampling.
    for sid in sorted(common.station.unique())[:5]:
        files = sorted(ROOT.glob(f'experiments/sprint-20261017-*/sources/{sid}/202[34]-*.json.gz'))
        if not files:
            raise ValueError('Retained raw history missing for '+sid)
        features = {}
        for path in files:
            certificate = json.loads(path.with_suffix(path.suffix+'.request.json').read_text())
            if digest(path) != certificate['sha256']:
                raise ValueError('Historical source checksum mismatch')
            hashes[str(path.relative_to(ROOT))] = digest(path)
            document = json.loads(gzip.decompress(path.read_bytes()))
            for f in document['features']:
                t = f['properties']['time']
                if t in features and features[t] != f:
                    raise ValueError('Inconsistent source observations at a chunk boundary')
                features[t] = f
        rows = hydrology.parse_discharge(dict(features=list(features.values()), links=[]), sid, now_utc())
        options = common.loc[common.station.eq(sid)].sort_values('decision_time')
        sample = options.iloc[np.linspace(0, len(options)-1, 10, dtype=int)]
        for _, record in sample.iterrows():
            issue = parse_time(record.decision_time)
            frame = hydrology.feature_row(rows, hydrology.manifest()['stations'][sid], issue)
            p = np.clip(candidate['model'].predict_proba(frame[candidate['features']])[:, 1], 1e-6, 1-1e-6)
            probability = float(candidate['calibrator'].predict_proba(pd.DataFrame({'score': np.log(p/(1-p))}))[0, 1])
            x, slope = frame.q_2.iloc[0], (frame.q_2.iloc[0]-frame.q_8.iloc[0])/6
            base_inputs = pd.DataFrame(dict(score=[x+slope*8], season_sin=frame.season_sin,
                season_cos=frame.season_cos, margin=np.log(2.)-frame.q_2, log_area=frame.log_area))
            control_p = float(baseline['calibrator'].predict_proba(base_inputs)[0, 1])
            errors = [abs(probability-record.probability), abs(control_p-record.probability_baseline)]
            if max(errors) > 1e-12:
                raise ValueError(f'Historical inference parity failed for {sid}: {errors}')
            results.append(dict(station=sid, issue_time=record.decision_time,
                candidate_error=errors[0], comparator_error=errors[1]))
    write_json(destination, dict(evidence='Inspected retrospective feature/inference parity only. Historical receipt availability remains unknown.',
        source_hashes=hashes, model_manifest_sha256=digest(hydrology.MANIFEST), results=results,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in [ROOT/'backend/hydrology.py', Path(__file__)]}))
    print('Matched', len(results), 'historical decisions at', len({r['station'] for r in results}), 'gauges')


if __name__ == '__main__':
    verify()
