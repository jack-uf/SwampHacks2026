"""Development-only six-hour research. Retained test outcomes never enter fitting or selection."""
import argparse
import importlib.metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.flood_sprint import write_json
from scripts.hydrology_research import ROOT, digest
from scripts.mlp_hybrid_worker import physical_score
from scripts.shift_hydrology import FEATURES
from scripts.shift_hydrology_metrics import metrics, paired_uncertainty, jointly_detected_lead_delta

OUT = ROOT / 'experiments/high-flow-20261018'
COHORTS = ['shift-hydrology', 'mlp-onset', 'mlp-hybrid', 'mlp-geo']
LATEST = ROOT / 'experiments/sprint-20261017-mlp-geo'
LOCAL = [x for x in FEATURES if x not in ['latitude', 'longitude']]
HYDRO = ['margin', 'linear_margin', 'damped_margin', 'time_to_reference',
         'rise_persistence', 'recent_fraction', 'relative_range']


def hydrologic_features(frame):
    """Use only decision inputs, in their actual log1p(flow/reference) units."""
    f = frame.copy()
    f['margin'] = np.log(2.) - f.q_2
    f['linear_margin'] = physical_score('linear', f) - np.log(2.)
    f['damped_margin'] = physical_score('damped', f) - np.log(2.)
    f['time_to_reference'] = np.where(f.rise_6 > 1e-6,
        f.margin / np.maximum(f.rise_6 / 6, 1e-6), 168.).clip(0, 168)
    f['rise_persistence'] = f.rise_6 * f.rise_24
    f['recent_fraction'] = f.rise_6 / (f.range_24 + .01)
    f['relative_range'] = f.range_72 / (f.q_2 + .01)
    if not np.isfinite(f[LOCAL + HYDRO].to_numpy(float)).all():
        raise ValueError('Non-finite causal features')
    return f


def split(frame, reserved):
    """Separate HUC4s and purge every label crossing a fitting boundary."""
    t = pd.to_datetime(frame.decision_time, utc=True)
    end = t + pd.Timedelta(hours=24)
    geographic = frame.huc.str[:4].isin(reserved)
    masks = dict(
        train=~geographic & t.lt('2023-01-01') & end.lt('2023-01-01'),
        calibration=~geographic & t.ge('2023-01-01') & end.lt('2023-07-01'),
        threshold=t.ge('2023-07-01') & end.lt('2024-01-01'),
        temporal_validation=~geographic & t.ge('2024-01-01') & end.lt('2025-01-01'),
        geographic_validation=geographic & t.ge('2024-01-01') & end.lt('2025-01-01'))
    parts = {k: frame.loc[v].reset_index(drop=True) for k, v in masks.items()}
    if any(f.empty for f in parts.values()):
        raise ValueError('Empty research partition')
    if set(parts['train'].huc.str[:4]) & set(parts['geographic_validation'].huc.str[:4]):
        raise ValueError('Geographic leakage')
    return parts


def threshold(frame, probability, pooled, region):
    """Lowest feasible inclusive threshold, preserving score ties and allowing zero alerts."""
    p, y = np.asarray(probability, float), frame.target.to_numpy(int)
    if not len(y) or p.shape != y.shape or not np.isfinite(p).all() or \
            not ((p >= 0) & (p <= 1)).all() or not np.isin(y, [0, 1]).all() or min(pooled, region) < 0:
        raise ValueError('Invalid threshold inputs')
    order = np.argsort(-p, kind='stable')
    stops = np.flatnonzero(np.r_[p[order][1:] != p[order][:-1], True])
    choices = p[order][stops]
    feasible = 100 * np.cumsum(1 - y[order])[stops] / len(y) <= pooled + 1e-9
    labels = frame.huc.str[:2].to_numpy()[order]
    for name in np.unique(labels):
        rows = labels == name
        feasible &= 100 * np.cumsum(rows & (y[order] == 0))[stops] / rows.sum() <= region + 1e-9
    return float(choices[np.flatnonzero(feasible)[-1]]) if feasible.any() else float(np.nextafter(p.max(), np.inf))


def sources(pooled):
    return [ROOT / ('experiments/sprint-20261017-' + name) / 'development.csv.gz'
            for name in (COHORTS if pooled else ['mlp-geo'])]


def load(pooled):
    frames = []
    for path in sources(pooled):
        certificate = json.loads(path.with_name('development-provenance.json').read_text())
        if digest(path) != certificate['dataset_sha256']:
            raise ValueError('Retained dataset checksum changed')
        f = pd.read_csv(path, dtype={'station': str, 'huc': str}, keep_default_na=False)
        f = f.loc[f.horizon.eq(6)].copy()
        if pd.to_datetime(f.decision_time, utc=True).dt.year.ge(2025).any():
            raise ValueError('Previously inspected final outcomes cannot enter development')
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    if f.duplicated(['station', 'decision_time', 'horizon']).any():
        raise ValueError('Overlapping cohorts must be resolved explicitly')
    if f.groupby('station').huc.nunique().gt(1).any():
        raise ValueError('Inconsistent station geography')
    return hydrologic_features(f)


def diagnose():
    """Reanalyse the inspected result, never present it as independent confirmation."""
    path = OUT / 'diagnosis.json'
    if path.exists():
        raise FileExistsError(path)
    selection = json.loads((LATEST / 'selection.json').read_text())
    record = next(x for x in selection['candidates'] if x['name'] == 'lightgbm-dynamics-ceiling-6')
    frame = pd.read_csv(LATEST / 'test.csv.gz', dtype={'station': str, 'huc': str}, keep_default_na=False)
    predictions = pd.read_csv(LATEST / 'predictions/lightgbm-dynamics-ceiling-6-temporal_test.csv.gz',
        dtype={'station': str, 'huc': str}, keep_default_na=False)
    f = frame.loc[frame.horizon.eq(6)].merge(predictions[['station', 'decision_time', 'probability']],
        on=['station', 'decision_time'], validate='one_to_one')
    f = f.loc[f.huc.str[:2].eq('04')].copy()
    f['false_alert'] = (f.probability >= record['threshold']) & f.target.eq(0)
    f['alert'] = f.probability >= record['threshold']
    f['month'] = pd.to_datetime(f.decision_time).dt.month
    fp = f.loc[f.false_alert].copy()
    for h in [12, 24]:
        later = frame.loc[frame.horizon.eq(h), ['station', 'decision_time', 'target']]
        fp = fp.merge(later.rename(columns={'target': f'onset_{h}'}), on=['station', 'decision_time'], validate='one_to_one')
    fp.to_csv(OUT / 'huc04-false-alerts.csv', index=False)
    development = pd.read_csv(LATEST / 'development.csv.gz', dtype={'station': str, 'huc': str}, keep_default_na=False)
    history = development.loc[development.station.isin(f.station) & development.horizon.eq(6)].copy()
    history['year'] = pd.to_datetime(history.decision_time).dt.year
    history['spring'] = pd.to_datetime(history.decision_time).dt.month.isin([3, 4, 5])
    shift = {}
    for year, group in history.groupby('year'):
        shift[str(year)] = dict(rows=len(group), prevalence=float(group.target.mean()),
            q2_quantiles=group.q_2.quantile([.1, .5, .9]).tolist(), spring_fraction=float(group.spring.mean()))
    write_json(path, dict(evidence='Previously inspected 2025 outcomes, development diagnosis only',
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [LATEST / 'test.csv.gz', LATEST / 'selection.json']},
        threshold=record['threshold'], stations=sorted(f.station.unique()),
        metrics=metrics(f, f.probability, record['threshold']),
        monthly=f.groupby('month').agg(rows=('target', 'size'), positives=('target', 'sum'),
            alerts=('alert', 'sum'), false_alerts=('false_alert', 'sum')).reset_index().to_dict('records'),
        false_alert_later_onsets={str(h): int(fp[f'onset_{h}'].sum()) for h in [12, 24]},
        false_alert_above_reference=int(fp.future_max_log_flow.gt(np.log(2.)).sum()),
        development_shift=shift, test_q2_quantiles=f.q_2.quantile([.1, .5, .9]).tolist()))


def calibrator_inputs(bundle, f):
    if bundle['kind'] in ['linear', 'damped', 'proximity']:
        raw = physical_score(bundle['kind'], f)
    else:
        p = np.clip(bundle['model'].predict_proba(f[bundle['features']])[:, 1], 1e-6, 1-1e-6)
        raw = np.log(p / (1-p))
    x = pd.DataFrame({'score': raw})
    if bundle['calibration'] == 'conditional':
        for name in ['season_sin', 'season_cos', 'margin', 'log_area']:
            x[name] = f[name].to_numpy()
    return x


def predict(bundle, frame):
    return bundle['calibrator'].predict_proba(calibrator_inputs(bundle, frame))[:, 1]


def run(cycle):
    destination = OUT / f'cycle-{cycle}.json'
    if destination.exists():
        raise FileExistsError(destination)
    pooled = cycle > 1
    f = load(pooled)
    stations = json.loads((LATEST / 'stations.json').read_text())
    reserved = {s['huc'][:4] for s in stations if s['role'] == 'geographic_test'}
    parts = split(f, reserved)
    # Each tuple changes one factor relative to the named control in the protocol.
    if cycle == 1:
        configurations = [('original', FEATURES, False, 'platt'),
            ('no_coordinates', LOCAL, False, 'platt'),
            ('hydrologic', LOCAL + HYDRO, False, 'platt'),
            ('balanced', LOCAL + HYDRO, True, 'platt')]
    elif cycle == 2:
        configurations = [('expanded_original', FEATURES, False, 'platt'),
            ('expanded_no_coordinates', LOCAL, False, 'platt'),
            ('expanded_hydrologic', LOCAL + HYDRO, False, 'platt'),
            ('expanded_balanced', LOCAL + HYDRO, True, 'platt'),
            ('expanded_conditional', LOCAL + HYDRO, True, 'conditional')]
    else:
        raise ValueError('Unknown research cycle')
    configurations = [(name, features, balanced, cal, 'lightgbm')
                      for name, features, balanced, cal in configurations]
    configurations += [(kind, [], False, cal, kind) for kind in ['proximity', 'linear', 'damped']
                       for cal in ['platt', 'conditional']]
    manifest = dict(cycle=cycle, evidence='Adaptive development only; no untouched final test',
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources(pooled)},
        code={str(p.relative_to(ROOT)): digest(p) for p in [ROOT / 'scripts/high_flow_research.py',
            ROOT / 'scripts/shift_hydrology.py', ROOT / 'scripts/shift_hydrology_metrics.py',
            ROOT / 'scripts/mlp_hybrid_worker.py']}, reserved_huc4=sorted(reserved),
        configurations=configurations, partitions={key: dict(rows=len(g), gauges=g.station.nunique(),
            huc4=g.huc.str[:4].nunique(), positives=int(g.target.sum())) for key, g in parts.items()},
        versions={name: importlib.metadata.version(name) for name in ['lightgbm', 'scikit-learn', 'numpy', 'pandas', 'joblib']},
        preserved={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'models').glob('*') if p.is_file()})
    lock = OUT / f'cycle-{cycle}-freeze.json'
    if lock.exists():
        raise FileExistsError(lock)
    write_json(lock, manifest)
    records = []
    (OUT / 'artifacts').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    for name, features, balanced, cal, kind in configurations:
        started = time.monotonic()
        bundle = dict(kind=kind, features=features, calibration=cal)
        train, calibration = parts['train'], parts['calibration']
        weights = None
        if balanced:
            groups = train.huc.str[:4]
            weights = 1 / groups.map(groups.value_counts()).to_numpy()
            weights /= weights.mean()
        if kind == 'lightgbm':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=250, learning_rate=.04, num_leaves=15, max_depth=4,
                min_child_samples=80, reg_lambda=5, n_jobs=2, random_state=42, verbosity=-1, deterministic=True)
            model.fit(train[features], train.target, sample_weight=weights)
            bundle['model'] = model
        calibrator = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
        calibrator.fit(calibrator_inputs(bundle, calibration), calibration.target)
        if calibrator[-1].coef_[0, 0] <= 0:
            records.append(dict(name=name, calibration=cal, excluded='Nonpositive calibration slope'))
            continue
        bundle['calibrator'] = calibrator
        p_threshold = predict(bundle, parts['threshold'])
        probabilities = {key: predict(bundle, parts[key]) for key in ['temporal_validation', 'geographic_validation']}
        base_name = f'{name}-{cal}' if kind != 'lightgbm' else name
        artifact = OUT / 'artifacts' / f'cycle-{cycle}-{base_name}.joblib'
        joblib.dump(bundle, artifact, compress=3)
        for key, p in probabilities.items():
            pred = parts[key][['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']].copy()
            pred['probability'] = p
            pred.to_csv(OUT / 'predictions' / f'cycle-{cycle}-{base_name}-{key}.csv.gz', index=False,
                        compression={'method': 'gzip', 'mtime': 0})
        for policy, pooled_limit, region_limit in [('ceiling', 1.5, 3.), ('headroom', 1., 2.)]:
            cut = threshold(parts['threshold'], p_threshold, pooled_limit, region_limit)
            scores = {key: metrics(parts[key], p, cut) for key, p in probabilities.items()}
            records.append(dict(name=base_name, policy=policy, threshold=cut, artifact_sha256=digest(artifact),
                seconds=time.monotonic()-started, metrics=scores))
            print(cycle, base_name, policy, {k: (v['detected_events'], round(v['false_alerts_per_100'], 3),
                round(v['ap'], 4)) for k, v in scores.items()}, flush=True)
    for path, checksum in manifest['preserved'].items():
        if digest(ROOT / path) != checksum:
            raise ValueError('Production artifact changed')
    write_json(destination, dict(freeze_sha256=digest(lock), records=records))


def compare(cycle, candidate, control, policy):
    """Paired uncertainty on development, after candidate choice. No promotion inference."""
    records = json.loads((OUT / f'cycle-{cycle}.json').read_text())['records']
    a, b = [next(r for r in records if r['name'] == name and r['policy'] == policy)
            for name in [candidate, control]]
    results = {}
    for partition in ['temporal_validation', 'geographic_validation']:
        frames = [pd.read_csv(OUT / 'predictions' / f'cycle-{cycle}-{name}-{partition}.csv.gz',
            dtype={'station': str, 'huc': str}, keep_default_na=False) for name in [candidate, control]]
        fa, fb = frames
        if not fa.drop(columns='probability').equals(fb.drop(columns='probability')):
            raise ValueError('Prediction rows differ')
        results[partition] = dict(uncertainty=paired_uncertainty(fa, fb.probability, fa.probability,
            b['threshold'], a['threshold']), lead=jointly_detected_lead_delta(fa, fb.probability,
            fa.probability, b['threshold'], a['threshold']))
    path = OUT / f'cycle-{cycle}-{candidate}-vs-{control}-{policy}.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['diagnose', 'run', 'compare'])
    parser.add_argument('--cycle', type=int, default=1)
    parser.add_argument('--candidate')
    parser.add_argument('--control', default='damped-platt')
    parser.add_argument('--policy', default='ceiling', choices=['ceiling', 'headroom'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'diagnose':
        diagnose()
    elif args.command == 'run':
        run(args.cycle)
    else:
        compare(args.cycle, args.candidate, args.control, args.policy)
