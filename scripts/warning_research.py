"""Experimental regional warning budgets, independent of production safety alerts."""
import json
import hashlib

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, digest, write_json

OUT = ROOT / 'experiments/sprint-20261017-warning'


def prepare():
    """Screen previously mapped but unused basins without requesting outcomes."""
    from shapely.geometry import shape
    from scripts.response_research import OUT as response, disjoint_basin
    if (OUT / 'stations.json').exists():
        raise FileExistsError('Registry already frozen')
    previous = {s['station']: s for name in ['hydrology', 'onset', 'response']
                for s in json.loads((ROOT / f'experiments/sprint-20261017-{name}/stations.json').read_text())}
    geography = json.loads((response / 'geography.json').read_text())
    geometries = {}
    for sid, checksum in geography['boundary_hashes'].items():
        path = response / 'basins' / f'{sid}.gauge-basin.json'
        if digest(path) != checksum:
            raise ValueError('Basin changed: ' + sid)
        location = response / 'basins' / f'{sid}.gauge-location.json'
        if digest(location) != geography['location_hashes'][sid]:
            raise ValueError('Gauge mapping changed: ' + sid)
        geometries[sid] = shape(json.loads(path.read_text())['features'][0]['geometry'])
    used_huc = {s['huc'][:4] for s in previous.values()}
    sites = json.loads((ROOT / 'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    zones = dict(EST='America/New_York', CST='America/Chicago', MST='America/Denver', PST='America/Los_Angeles')
    fixed = dict(EST='Etc/GMT+5', CST='Etc/GMT+6', MST='Etc/GMT+7', PST='Etc/GMT+8')
    candidates = []
    for s in sites:
        sid, huc = s['site_no'], s['huc_cd']
        if sid not in geometries or sid in previous or huc[:4] in used_huc:
            continue
        candidates.append(dict(station=sid, name=s['station_nm'], huc=huc[:8],
            latitude=float(s['dec_lat_va']), longitude=float(s['dec_long_va']),
            area_km2=float(s['drain_area_va']) * 2.589988110336,
            elevation_m=float(s['alt_va']) * .3048,
            timezone=(zones if s['local_time_fg'] == 'Y' else fixed)[s['tz_cd']], role='fresh_test'))
    occupied = [geometries[sid] for sid in previous]
    selected, exclusions, groups = [], [], []
    rank = lambda h: hashlib.sha256(('warning-20261017-' + h).encode()).hexdigest()
    for h in sorted({s['huc'][:4] for s in candidates}, key=rank):
        accepted = []
        for s in sorted([s for s in candidates if s['huc'][:4] == h], key=lambda s: s['station']):
            if disjoint_basin(geometries[s['station']], occupied):
                accepted.append(s)
            else:
                exclusions.append(dict(station=s['station'], reason='upstream polygon overlaps previous population'))
        if accepted:
            groups.append(h)
            selected.extend(accepted)
            occupied.extend(geometries[s['station']] for s in accepted)
    write_json(OUT / 'stations.json', selected)
    write_json(OUT / 'geography.json', dict(previous_stations=len(previous), candidates=len(candidates),
        groups=groups, exclusions=exclusions, stations=len(selected),
        parent_geography_sha256=digest(response / 'geography.json'),
        note='Metadata-only selection. Every accepted upstream polygon has zero positive-area intersection with prior gauges and earlier accepted HUC4 groups.'))
    print('Reserved', len(selected), 'stations in', len(groups), 'new HUC4 groups', flush=True)


def budget_alerts(frame, probabilities, threshold):
    """Spend one credit per decision; refund correct warnings after ten days."""
    p = np.asarray(probabilities, dtype=float)
    f = frame.reset_index(drop=True).copy()
    if p.shape != (len(f),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid threshold')
    if not f.exceedance.isin([0, 1]).all() or f.duplicated(['station', 'decision_date']).any():
        raise ValueError('Invalid labels or duplicate decisions')
    if not f.huc.astype(str).str.fullmatch(r'\d{8}').all():
        raise ValueError('Invalid HUC8')
    f['day'] = pd.to_datetime(f.decision_date, format='%Y-%m-%d', errors='raise')
    f['probability'] = p
    alerts = np.zeros(len(f), dtype=bool)
    for _, region in f.groupby(f.huc.str[:2]):
        credits, pending = 0, []
        for day, decisions in region.groupby('day', sort=True):
            mature = [i for i in pending if f.at[i, 'day'] <= day - pd.Timedelta(days=10)]
            credits = min(300, credits + len(decisions) + 100 * int(f.loc[mature, 'exceedance'].sum()))
            pending = [i for i in pending if i not in mature]
            proposals = decisions[decisions.probability >= threshold].sort_values(
                ['probability', 'station'], ascending=[False, True])
            chosen = proposals.index[:credits // 100].tolist()
            alerts[chosen] = True
            credits -= 100 * len(chosen)
            pending.extend(chosen)
    return alerts
