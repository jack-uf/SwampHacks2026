"""Exploratory development comparison during a USGS quota wait; never promotion evidence."""
import json
import shutil
from pathlib import Path

import pandas as pd
from scripts import shift_hydrology as study

out = Path(__file__).parent
study.protocol()
if (out / 'development.csv.gz').exists():
    raise FileExistsError('Pilot already materialized')
original = study.cached_page


def cache_only(path, url):
    if not path.exists():
        raise FileNotFoundError(path)
    return original(path, url)


study.cached_page = cache_only
references = json.loads((study.OUT / 'references.json').read_text())
frames, coverage = [], []
for station in json.loads((study.OUT / 'stations.json').read_text()):
    sid = station['station']
    if station['role'] == 'geographic_test':
        continue
    years = [2023, 2024] if station['role'] == 'geographic_validation' else [2021, 2022, 2023, 2024]
    try:
        pieces = [study.observations(sid, year) for year in years]
    except FileNotFoundError:
        continue
    q = study.hourly_flow(pd.concat(pieces).sort_index())
    if q.empty:
        coverage.append(dict(station=sid, excluded='No accepted instantaneous discharge'))
        continue
    q = q.reindex(pd.date_range(f'{years[0]}-01-01', f'{years[-1]+1}-01-01', freq='h', tz='UTC', inclusive='left'))
    frame, _, diagnostics = study.decision_rows(q, references[sid]['reference_cfs'], station)
    frames.append(frame)
    coverage.append(dict(station=sid, role=station['role'], rows=len(frame), **diagnostics))
    print(sid, len(frame), flush=True)
assert frames and {f.role.iloc[0] for f in frames} == {'development', 'geographic_validation'}
frame = pd.concat(frames, ignore_index=True)
frame.to_csv(out / 'development.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})
shutil.copyfile(study.OUT / 'protocol.json', out / 'protocol.json')
study.write_json(out / 'development-provenance.json', dict(dataset_sha256=study.digest(out/'development.csv.gz'),
    interpretation='Exploratory acquisition subset, 2024 development validation only; no final test or promotion.',
    source_requests={str(p.relative_to(study.OUT)):json.loads(p.read_text())['sha256']
                     for p in (study.OUT/'sources').rglob('*.request.json')}))
study.write_json(out / 'coverage.json', coverage)
