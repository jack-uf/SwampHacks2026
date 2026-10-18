"""Build the pinned experimental model reference and measure live site coverage without changing production models."""
import argparse
import asyncio
import json
import gzip
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from backend import hydrology
from backend.store import initialize, db
from scripts.common import ROOT
from scripts.flood_sprint import write_json
from scripts.hydrology_research import digest


def build_manifest():
    path = hydrology.MANIFEST
    if path.exists():
        raise FileExistsError(path)
    out = ROOT / 'experiments/high-flow-20261018'
    decision = json.loads((out/'decision.json').read_text())
    if decision['exploratory_gate_check']['integration_ready']:
        raise ValueError('Unexpected promotion status')
    stations, provenance = {}, {}
    for cohort in ['shift-hydrology', 'mlp-onset', 'mlp-hybrid', 'mlp-geo']:
        folder = ROOT / ('experiments/sprint-20261017-'+cohort)
        dataset = folder/'development.csv.gz'
        certificate = json.loads((folder/'development-provenance.json').read_text())
        if digest(dataset) != certificate['dataset_sha256']:
            raise ValueError('Retained reference dataset changed')
        f = pd.read_csv(dataset, dtype={'station': str, 'huc': str}, usecols=[
            'station', 'huc', 'reference_cfs', 'latitude', 'longitude', 'log_area', 'elevation_m'])
        for sid, group in f.groupby('station'):
            if len(group.drop_duplicates()) != 1 or sid in stations:
                raise ValueError('Station reference or metadata is ambiguous')
            row = group.iloc[0]
            import numpy as np
            stations[sid] = dict(reference_cfs=float(row.reference_cfs), huc=row.huc,
                latitude=float(row.latitude), longitude=float(row.longitude),
                area_km2=float(np.expm1(row.log_area)), elevation_m=float(row.elevation_m))
        provenance[str(dataset.relative_to(ROOT))] = digest(dataset)
    artifacts = {}
    for label, name, field in [('candidate', 'expanded_original', 'candidate'), ('baseline', 'linear-conditional', 'comparator')]:
        file = out/'artifacts'/f'cycle-4-{name}.joblib'
        record = decision[field]
        if digest(file) != record['artifact_sha256']:
            raise ValueError('Frozen model artifact changed')
        artifacts[label] = dict(path=str(file.relative_to(ROOT)), sha256=digest(file), threshold=record['threshold'])
    write_json(path, dict(operationally_promoted=False, **artifacts, stations=stations,
        reference='2000-2015 daily-mean USGS discharge 95th percentile, not a flood threshold',
        source_datasets=provenance, decision_sha256=digest(out/'decision.json')))
    print('Pinned experimental reference for', len(stations), 'gauges; production promotion remains false')


async def audit(command, output, site_ids):
    if output.exists():
        raise FileExistsError(output)
    sites = json.loads((ROOT/'data/sites.json').read_text())['sites']
    if site_ids:
        selected = set(site_ids.split(','))
        if selected - {s['id'] for s in sites}:
            raise ValueError('Unknown site ID')
        sites = [s for s in sites if s['id'] in selected]
    initialize()
    records = []
    # Four independent site lookups share the provider's bounded request semaphore.
    limit = asyncio.Semaphore(4)
    async def one(site):
        async with limit:
            result = await (hydrology.inspect(site) if command == 'live' else hydrology.discover(site))
            if command == 'current-coverage':
                result = await hydrology.discover(site)
                _, screen = await hydrology.active_gauges(result, site)
                result['discharge_screen'] = screen
            print(site['id'], site['name'], result.get('status', result.get('candidate_count', 0)), flush=True)
            return dict(site=site, result=result)
    records = await asyncio.gather(*(one(s) for s in sites))
    if command == 'live':
        for record in records:
            for gauge in record['result']['gauges']:
                with db() as c:
                    receipt = dict(c.execute('SELECT * FROM hydrology_receipts WHERE id=?', (gauge['receipt']['id'],)).fetchone())
                folder = output.parent/'receipts'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/(receipt['id']+'.json.gz')
                path.write_bytes(gzip.compress(receipt.pop('body'), mtime=0))
                write_json(path.with_suffix('.request.json'), receipt)
    write_json(output, dict(command=command, checked_at=hydrology.now_utc().isoformat(), records=records,
        source_sites_sha256=digest(ROOT/'data/sites.json'), model_manifest_sha256=digest(hydrology.MANIFEST),
        note='Live access and metadata screening only. No independent predictive outcomes or site flooding labels.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['manifest', 'coverage', 'current-coverage', 'live'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sites', help='Comma-separated EIA facility IDs; otherwise all facilities')
    args = parser.parse_args()
    load_dotenv(ROOT/'.env')
    if args.command == 'manifest':
        build_manifest()
    else:
        if not args.output:
            parser.error('--output is required for an audit')
        asyncio.run(audit(args.command, args.output, args.sites))
