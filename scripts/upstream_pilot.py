"""Frozen development-only upstream transport pilot; no geographic test outcomes."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.hydrology_research import ROOT, daily_discharge, digest, fetch, write_json
from scripts.upstream_research import OUT

TRANSPORT = ['transport_1', 'transport_2', 'transport_3']


def transport_features(down, up, issues):
    """Reference-only local/augmented trajectories evaluated with delayed daily flows."""
    for q in [down, up]:
        if q.index.has_duplicates or not q.index.is_monotonic_increasing or (q.dropna() < 0).any() or not np.isfinite(q.dropna()).all():
            raise ValueError('Invalid flow history')
    dates = pd.date_range(min(down.index.min(), up.index.min()), max(down.index.max(), up.index.max()))
    refs = []
    for q in [down, up]:
        reference = q.loc[q.index < '2016-01-01'].dropna()
        if len(reference) < 3000 or reference.quantile(.95) <= 0:
            raise ValueError('Insufficient reference flow')
        refs.append(float(reference.quantile(.95)))
    d, u = [np.log1p(q.reindex(dates) / ref) for q, ref in zip([down, up], refs)]
    phase = 2 * np.pi * dates.dayofyear / 365.25
    inputs = pd.DataFrame({'d0': d, 'd1': d.shift(1), 'd4': d.shift(4),
                           'sin': np.sin(phase), 'cos': np.cos(phase),
                           'u0': u, 'u1': u.shift(1), 'u4': u.shift(4)}, index=dates)
    local = ['d0', 'd1', 'd4', 'sin', 'cos']
    observed = pd.DatetimeIndex(issues) - pd.Timedelta(days=3)
    usable = inputs.reindex(observed).notna().all(axis=1).to_numpy()
    available = inputs.reindex(observed).loc[usable]
    result = pd.DataFrame(index=issues, columns=['up_3', 'up_4', 'up_7', 'up_change'] + TRANSPORT, dtype=float)
    result.loc[usable, ['up_3', 'up_4', 'up_7']] = available[['u0', 'u1', 'u4']].to_numpy()
    result.loc[usable, 'up_change'] = (available.u0 - available.u1).to_numpy()
    fitted = dict(references=refs, leads={})
    for lead in [1, 2, 3]:
        horizon = lead + 3
        target = d.shift(-horizon)
        mask = inputs.notna().all(axis=1) & target.notna() & (dates + pd.Timedelta(days=horizon) < pd.Timestamp('2016-01-01'))
        if mask.sum() < 2000:
            raise ValueError('Insufficient complete reference trajectories')
        models = []
        for columns in [local, list(inputs)]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(inputs.loc[mask, columns], target.loc[mask])
            models.append(model)
        if len(available):
            result.loc[usable, f'transport_{lead}'] = models[1].predict(available) - models[0].predict(available[local])
        fitted['leads'][lead] = dict(rows=int(mask.sum()), local=models[0], augmented=models[1])
    return result, fitted


def verify_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, checksum in protocol['inputs'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Frozen pilot input changed: ' + path)
    return protocol


def reference_coverage(text, station):
    """A catalog lifetime range does not establish usable reference-period data."""
    if text.startswith('#  No sites found matching all criteria\n'):
        return dict(station=station, eligible=False, approved_days=0,
                    reason='USGS returned no sites for the requested historical interval')
    q = daily_discharge(text, station)
    q.index = pd.to_datetime(q.index)
    reference = q.loc[q.index < '2016-01-01']
    consecutive = reference.reindex(pd.date_range('2000-01-01', '2015-12-31')).notna().rolling(7).sum().eq(7).sum()
    eligible = len(reference) >= 3000 and reference.quantile(.95) > 0 and consecutive >= 2000
    return dict(station=station, eligible=bool(eligible), approved_days=len(q),
                reference_days=len(reference), consecutive_reference_windows=int(consecutive),
                first_approved=str(q.index.min().date()) if len(q) else None,
                last_approved=str(q.index.max().date()) if len(q) else None,
                annual_approved={str(year): len(group) for year, group in q.groupby(q.index.year)},
                reason=None if eligible else 'Insufficient approved reference-period flow')


def coverage():
    """Finish the four-pair source audit after the first collection failed fast."""
    verify_protocol()
    if (OUT / 'coverage.json').exists():
        raise FileExistsError('Source coverage audit already recorded')
    pairs = [p for p in json.loads((OUT / 'expanded-pairs.json').read_text())['pairs'] if p['role'] == 'development']
    if len(pairs) != 4:
        raise ValueError('Frozen pilot requires exactly four development pairs')
    records = []
    for pair in pairs:
        uid = pair['upstream']
        target = OUT / 'sources' / f'{uid}.rdb'
        url = f'https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={uid}&startDT=2000-01-01&endDT=2022-12-31&parameterCd=00060&statCd=00003'
        text = fetch(target, url).decode()
        records.append(dict(downstream=pair['station'], **reference_coverage(text, uid)))
    write_json(OUT / 'coverage.json', dict(
        status='aborted_incomplete_frozen_population' if not all(r['eligible'] for r in records) else 'source_check_complete',
        pairs=records, required_pairs=4, reference_eligible_pairs=sum(r['eligible'] for r in records),
        fitted_models=0, independent_geographic_outcomes_opened=False,
        protocol_sha256=digest(OUT / 'protocol.json'), code_sha256=digest(ROOT / 'scripts/upstream_pilot.py')))
    print('Reference-eligible development pairs:', sum(r['eligible'] for r in records), '/', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['coverage'])
    parser.parse_args()
    coverage()
