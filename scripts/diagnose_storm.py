"""Reproduce storm diagnostics offline from frozen artifacts; never fit or select.

Run from the repository root: .venv/bin/python -m scripts.diagnose_storm
Only output/storm-diagnostics.json is written. See docs/STORM_DIAGNOSTICS.md.
"""
import hashlib
from importlib.metadata import version
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from backend.model import calibrated
from backend.predict import explain
from scripts.build_training import BASE_FEATURES, FEATURES, normalize_events
from scripts.common import ROOT
from scripts.train import block_comparison, metrics, partitions


SEED = 42
PERMUTATIONS = 12
STORM_TYPES = ['Thunderstorm Wind', 'Tornado', 'Hail']
GROUPS = {'geography': ['latitude', 'longitude'],
          'season': ['season_sin', 'season_cos']}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(frame):
    return dict(rows=len(frame), positives=int(frame.storm.sum()),
                prevalence=float(frame.storm.mean()), counties=int(frame.region.nunique()),
                start=str(frame.target_date.min()), end=str(frame.target_date.max()))


def compact_metrics(y, p, threshold):
    return {k: v for k, v in metrics(y, p, threshold).items() if k != 'calibration'}


def positive_ranks(y, probabilities):
    """Each positive contributes precision at its tied score / positive count to AP."""
    y = np.asarray(y)
    p = np.asarray(probabilities)
    if len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('Ranks require aligned binary labels and finite scores')
    result = []
    for index in np.flatnonzero(y):
        above = p >= p[index]
        precision = float(y[above].sum() / above.sum())
        result.append(dict(index=int(index), probability=float(p[index]),
                           rank_min=int((p > p[index]).sum() + 1),
                           rank_max=int(above.sum()), tied_rows=int((p == p[index]).sum()),
                           precision_at_score=precision, ap_contribution=precision / int(y.sum())))
    return result


def overlapping_storms(events, row):
    """Use the dataset's closed NOAA interval / half-open target-day convention."""
    day = pd.Timestamp(row.target_date)
    return events[(events.STATE_FIPS * 1000 + events.CZ_FIPS == row.county_fips)
                  & events.EVENT_TYPE.isin(STORM_TYPES)
                  & events.CZ_TYPE.eq('C')
                  & (events.start_utc < day + pd.Timedelta(days=1))
                  & (events.end_utc >= day)]


def distribution(values):
    quantiles = values.quantile([0, .05, .5, .95, 1]).tolist()
    return dict(n=len(values), mean=float(values.mean()),
                **dict(zip(['min', 'q05', 'median', 'q95', 'max'], map(float, quantiles))))


def feature_comparison(row, train):
    positives = train[train.storm.eq(1)]
    return {f: dict(value=float(row[f]),
                    fit_percentile_le=float(100 * train[f].le(row[f]).mean()),
                    positive_fit_percentile_le=float(100 * positives[f].le(row[f]).mean()),
                    outside_fit_range=bool(row[f] < train[f].min() or row[f] > train[f].max()))
            for f in FEATURES}


def probability_evidence(bundle, frame, fit_prevalence):
    raw = bundle['model'].predict_proba(frame[bundle['features']])[:, 1]
    p = calibrated(bundle, frame)
    observed = float(frame.storm.mean())
    return dict(calibrated=metrics(frame.storm, p, bundle['threshold']),
                raw_average_precision=float(average_precision_score(frame.storm, raw)),
                raw_brier=float(brier_score_loss(frame.storm, raw)),
                mean_probability=float(p.mean()), expected_positives=float(p.sum()),
                mean_positive_probability=float(p[frame.storm.eq(1)].mean()),
                mean_negative_probability=float(p[frame.storm.eq(0)].mean()),
                zero_probability_brier=observed,
                fit_climatology_brier=float(brier_score_loss(frame.storm, np.full(len(frame), fit_prevalence))),
                empirical_constant_brier=observed * (1 - observed))


def reliance(bundle, frame):
    """Fixed paired-column permutations on validation only; no fitting or selection."""
    p = calibrated(bundle, frame)
    ap = average_precision_score(frame.storm, p)
    brier = brier_score_loss(frame.storm, p)
    rng = np.random.default_rng(SEED)
    orders = [rng.permutation(len(frame)) for _ in range(PERMUTATIONS)]
    result = {}
    for group, features in GROUPS.items():
        samples = []
        for order in orders:
            changed = frame.copy()
            changed[features] = frame[features].to_numpy()[order]
            altered = calibrated(bundle, changed)
            samples.append(dict(ap_drop=float(ap - average_precision_score(frame.storm, altered)),
                                brier_increase=float(brier_score_loss(frame.storm, altered) - brier)))
        result[group] = dict(columns=features, replicates=samples,
                             mean_ap_drop=float(np.mean([s['ap_drop'] for s in samples])),
                             mean_brier_increase=float(np.mean([s['brier_increase'] for s in samples])))
    return dict(partition=counts(frame), original_ap=float(ap), original_brier=float(brier), groups=result)


def selection_evidence(saved, parts, bundles):
    candidates = saved['candidates']
    rows = []
    for name, candidate in candidates.items():
        temporal = candidate['validation']
        geographic = candidate['geographic_validation']
        rows.append(dict(name=name, temporal=compact_saved(temporal), geographic=compact_saved(geographic),
                         selection_score=candidate['selection_score'],
                         row_weighted_mean_ap=(temporal['pr_auc'] * temporal['n']
                                               + geographic['pr_auc'] * geographic['n'])
                                              / (temporal['n'] + geographic['n'])))
    sensitivity = {}
    frame = parts['geographic_validation']
    for name, bundle in bundles.items():
        p = calibrated(bundle, frame)
        positives = positive_ranks(frame.storm, p)
        leave_one_out = []
        for rank in positives:
            index = rank['index']
            keep = np.arange(len(frame)) != index
            row = frame.iloc[index]
            ap = float(average_precision_score(frame.storm.to_numpy()[keep], p[keep]))
            leave_one_out.append(dict(region=row.region, target_date=row.target_date,
                                      rank=rank, ap_without_this_row=ap))
        sensitivity[name] = dict(positive_rows=leave_one_out,
                                ap_without_each_positive_range=[min(r['ap_without_this_row'] for r in leave_one_out),
                                                                max(r['ap_without_this_row'] for r in leave_one_out)])
    return dict(candidates=sorted(rows, key=lambda r: -r['selection_score']),
                temporal_ap_winner=max(candidates, key=lambda n: candidates[n]['validation']['pr_auc']),
                row_weighted_mean_ap_winner=max(rows, key=lambda r: r['row_weighted_mean_ap'])['name'],
                geographic_positive_sensitivity=sensitivity,
                warning='Row-weighted mean AP is not pooled AP. Sensitivity deletes one positive row; prevalence changes. No model is reselected.')


def compact_saved(result):
    return {k: result[k] for k in ['n', 'positives', 'prevalence', 'pr_auc', 'brier', 'threshold', 'confusion_matrix']}


def matched_evidence(frame, bundles, saved):
    scores = {n: calibrated(b, frame) for n, b in bundles.items()}
    ranks = {n: {r['index']: r for r in positive_ranks(frame.storm, p)} for n, p in scores.items()}
    positives = []
    for index in np.flatnonzero(frame.storm.to_numpy()):
        row = frame.iloc[index]
        old, new = (ranks[n][index] for n in ['baseline', 'selected'])
        positives.append(dict(region=row.region, target_date=row.target_date, baseline=old, selected=new,
                              delta_ap_contribution=new['ap_contribution'] - old['ap_contribution']))
    delta = float(average_precision_score(frame.storm, scores['selected'])
                  - average_precision_score(frame.storm, scores['baseline']))
    if not np.isclose(sum(p['delta_ap_contribution'] for p in positives), delta, atol=1e-12):
        raise ValueError('Positive-level AP decomposition does not sum to matched AP change')
    groups = {}
    for field, keys in [('county', frame.region), ('quarter', pd.to_datetime(frame.target_date).dt.to_period('Q').astype(str))]:
        groups[field] = {}
        for key in sorted(keys.unique()):
            use = keys.eq(key).to_numpy()
            related = [p for p in positives if (p['region'] == key if field == 'county'
                       else str(pd.Period(p['target_date'], freq='Q')) == key)]
            groups[field][key] = dict(counts=counts(frame[use]),
                models={n: compact_metrics(frame[use].storm, p[use], bundles[n]['threshold']) for n, p in scores.items()},
                contribution_to_pooled_delta_ap=float(sum(p['delta_ap_contribution'] for p in related)))
    return dict(counts=counts(frame),
                models={n: probability_evidence(b, frame, saved[n]) for n, b in bundles.items()},
                delta_ap=delta, delta_brier=float(np.mean((scores['selected'] - frame.storm) ** 2)
                                               - np.mean((scores['baseline'] - frame.storm) ** 2)),
                paired_uncertainty=block_comparison(frame, frame.storm, scores['baseline'], scores['selected']),
                positive_rows=positives, groups=groups)


def verify_metrics(actual, expected, path='metrics'):
    """Fail on stale artifacts; numeric reproduction allows only floating point noise."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            verify_metrics(actual[key], value, path + '/' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Saved evidence length mismatch: ' + path)
        for index, value in enumerate(expected):
            verify_metrics(actual[index], value, path + '/' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ValueError('Saved evidence numeric mismatch: ' + path)
    elif actual != expected:
        raise ValueError('Saved evidence mismatch: ' + path)


def build_report():
    archive = ROOT / 'experiments/baseline'
    manifest = json.loads((archive / 'manifest.json').read_text())
    protected = [ROOT / p for p in ['data/training.csv', 'data/regions.json', 'data/training-provenance.json',
                                  'models/hazards.joblib', 'models/evaluation.json', 'models/selection.json']]
    protected += [archive / p for p in manifest['files']]
    protected += [archive / 'manifest.json']
    hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in protected}
    for name, expected in manifest['files'].items():
        if hashes['experiments/baseline/' + name] != expected:
            raise ValueError('Frozen baseline checksum mismatch: ' + name)
    evaluation = json.loads((ROOT / 'models/evaluation.json').read_text())
    selection = json.loads((ROOT / 'models/selection.json').read_text())['storm']
    artifact = joblib.load(ROOT / 'models/hazards.joblib')
    old_artifact = joblib.load(archive / 'models/hazards.joblib')
    current = artifact['hazards']['storm']
    baseline = dict(old_artifact['hazards']['storm'], features=old_artifact['features'])
    saved = evaluation['hazards']['storm']
    if not (current['name'] == selection['selected'] == saved['selected']
            and current['threshold'] == selection['threshold']
            and evaluation['dataset_sha256'] == artifact['dataset_sha256'] == hashes['data/training.csv']
            and current['features'] == FEATURES and baseline['features'] == BASE_FEATURES):
        raise ValueError('Model, selection, feature order or dataset provenance mismatch; restore the matching saved artifacts')
    frame = pd.read_csv(ROOT / 'data/training.csv').sort_values(['target_date', 'region']).reset_index(drop=True)
    if frame.duplicated(['county_fips', 'target_date']).any() or not np.isfinite(frame[FEATURES]).all().all():
        raise ValueError('Dataset contains duplicate county-days or nonfinite features')
    parts = partitions(frame)
    old_frame = pd.read_csv(archive / 'data/training.csv')
    old_parts = dict(train=old_frame[(old_frame.target_date < '2024-09-01') & old_frame.region.ne('Los Angeles, CA')],
        calibration=old_frame[(old_frame.target_date >= '2024-09-01') & (old_frame.target_date < '2024-11-01') & old_frame.region.ne('Los Angeles, CA')],
        validation=old_frame[(old_frame.target_date >= '2024-11-01') & (old_frame.target_date < '2025-01-01') & old_frame.region.ne('Los Angeles, CA')])
    original = set(old_frame.region) - {'Los Angeles, CA', 'Maricopa, AZ'}
    matched = parts['test'][parts['test'].region.isin(original)]
    for name in original:
        for feature in ['latitude', 'longitude']:
            if not matched[matched.region.eq(name)][feature].eq(old_frame[old_frame.region.eq(name)].iloc[0][feature]).all():
                raise ValueError('Matched representative coordinates changed: ' + name)
    bundles = dict(baseline=baseline, selected=current)
    fit_prevalence = dict(baseline=float(old_parts['train'].storm.mean()), selected=float(parts['train'].storm.mean()))
    matched_report = matched_evidence(matched, bundles, fit_prevalence)
    verify_metrics(matched_report['models']['baseline']['calibrated'], saved['same_final_rows']['original_threshold_baseline'])
    verify_metrics(matched_report['models']['selected']['calibrated'], saved['same_final_rows']['updated'])
    verify_metrics(matched_report['paired_uncertainty'], saved['same_final_rows']['paired_uncertainty'])
    probabilities = {}
    for name, bundle in bundles.items():
        probabilities[name] = {}
        for part in ['calibration', 'validation', 'geographic_validation', 'test', 'geographic_test']:
            result = probability_evidence(bundle, parts[part], fit_prevalence[name])
            if name == 'selected':
                if part != 'calibration':
                    verify_metrics(result['calibrated'], saved['candidates'][current['name']][part])
            elif part in ['validation', 'geographic_validation', 'test', 'geographic_test']:
                expected = saved['candidates']['frozen_baseline'][part]
                verify_metrics(metrics(parts[part].storm, calibrated(bundle, parts[part]), expected['threshold']), expected)
            probabilities[name][part] = result
    provenance = json.loads((ROOT / 'data/training-provenance.json').read_text())
    source = next(s for s in provenance['sources'] if 'StormEvents_details-ftp_v1.0_d2026_' in s['url'])
    raw_path = ROOT / 'data/raw' / source['url'].rsplit('/', 1)[-1]
    if not raw_path.exists():
        raise FileNotFoundError('Restore the exact local NOAA archive recorded in data/training-provenance.json: ' + raw_path.name)
    raw_hash = fingerprint(raw_path)
    if raw_hash != source['sha256']:
        raise ValueError('Local NOAA archive checksum differs from dataset provenance')
    raw = pd.read_csv(raw_path, low_memory=False)
    geographic = parts['geographic_test']
    events = normalize_events(raw[raw.STATE_FIPS.isin(geographic.county_fips.floordiv(1000).unique())])
    geo_prob = calibrated(current, geographic)
    misses = []
    columns = ['EVENT_ID', 'EPISODE_ID', 'EVENT_TYPE', 'STATE_FIPS', 'CZ_FIPS', 'CZ_TYPE', 'CZ_NAME',
               'CZ_TIMEZONE', 'BEGIN_DATE_TIME', 'END_DATE_TIME', 'MAGNITUDE', 'MAGNITUDE_TYPE', 'SOURCE',
               'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'EVENT_NARRATIVE', 'EPISODE_NARRATIVE',
               'start_utc', 'end_utc']
    for position, (index, row) in enumerate(geographic.iterrows()):
        overlap = overlapping_storms(events, row)
        if bool(len(overlap)) != bool(row.storm):
            raise ValueError('NOAA label reproduction mismatch: ' + row.region + ' ' + row.target_date)
        p = float(geo_prob[position])
        if not row.storm or p >= current['threshold']:
            continue
        explanation = explain(current, frame.loc[[index]])
        reconstructed = explanation['explanation_base'] + sum(e['probability_delta'] for e in explanation['explanations'])
        if not np.isclose(reconstructed, p, atol=1e-10, rtol=0):
            raise ValueError('Calibrated SHAP additivity failed')
        records = overlap.sort_values('EVENT_ID')[columns].copy()
        for field in ['start_utc', 'end_utc']:
            records[field] = records[field].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        attribution = {e['feature']: e['probability_delta'] for e in explanation['explanations']}
        misses.append(dict(region=row.region, county_fips=int(row.county_fips), target_date=row.target_date,
            issue_date=row.issue_date, probability=p,
            raw_probability=float(current['model'].predict_proba(frame.loc[[index], FEATURES])[:, 1][0]),
            baseline_probability=float(calibrated(baseline, frame.loc[[index]])[0]),
            threshold=float(current['threshold']), probability_shortfall=float(current['threshold'] - p),
            all_geographic_rows_at_or_above_probability=int((geo_prob >= p).sum()),
            features_vs_fit=feature_comparison(row, parts['train']), calibrated_probability_shap=explanation,
            shap_additivity_residual=float(reconstructed - p),
            grouped_shap={g: float(sum(attribution[f] for f in features)) for g, features in GROUPS.items()},
            noaa_records=json.loads(records.to_json(orient='records'))))
    threshold_rows = {}
    policies = saved['candidates'][current['name']]['threshold_policies']
    verify_metrics(metrics(parts['validation'].storm, calibrated(current, parts['validation']), selection['threshold']), selection['validation'])
    for policy, values in policies.items():
        if values is None:
            raise ValueError('Expected saved storm threshold policy is unavailable: ' + policy)
        threshold_rows[policy] = dict(validation_policy=values,
            populations={n: compact_metrics(f.storm, calibrated(current, f), values['threshold'])
                         for n, f in dict(validation=parts['validation'], test=parts['test'],
                                          matched=matched, geographic_test=geographic).items()})
    # The archived threshold and the saved re-budgeted baseline policy are distinct.
    old_policies = saved['candidates']['frozen_baseline']['threshold_policies']
    threshold_rows['baseline_original'] = dict(threshold=baseline['threshold'],
        matched=compact_metrics(matched.storm, calibrated(baseline, matched), baseline['threshold']))
    threshold_rows['baseline_validation_budget'] = dict(validation_policy=old_policies['budget'],
        matched=compact_metrics(matched.storm, calibrated(baseline, matched), old_policies['budget']['threshold']))
    booster = current['model'].booster_
    gain = booster.feature_importance(importance_type='gain')
    split = booster.feature_importance(importance_type='split')
    importance = {f: dict(gain=float(gain[i]), gain_share=float(gain[i] / gain.sum()), splits=int(split[i]))
                  for i, f in enumerate(FEATURES)}
    report = dict(schema_version=1,
        protocol=dict(training='None; no fit, calibrator fit, model selection, ablation fit or test optimization.',
            permutations='Declared in source: 12 paired-column permutations each of geography and season, seed 42; temporal and geographic validation only; no outcome-dependent choice.',
            explanations='Existing backend.predict.explain: calibrated-probability permutation SHAP, 16 saved background rows, seed 42, four permutations per explanation.',
            ap='Average precision, not trapezoidal PR area; tied-score precision contributions sum to AP.',
            dependence='County-days are not independent events. Calendar-week bootstrap preserves spatial rows; intervals are descriptive.',
            candidate_limit='Only selected and frozen baseline model objects are saved; other candidate validation metrics are used as recorded, never refitted.'),
        runtime=dict(python=platform.python_version(), packages={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'shap', 'joblib']}),
        input_sha256=hashes, raw_noaa=dict(**source, local_path=str(raw_path.relative_to(ROOT)), checksum_verified=True),
        source_code_sha256={p: fingerprint(ROOT / p) for p in ['scripts/diagnose_storm.py', 'scripts/train.py',
            'scripts/build_training.py', 'backend/model.py', 'backend/predict.py']},
        dataset=dict(counts=counts(frame), partitions={n: counts(f) for n, f in parts.items()},
            unused_rows=int(len(frame) - sum(len(f) for f in parts.values())),
            unused_positives=int(frame.storm.sum() - sum(f.storm.sum() for f in parts.values())),
            baseline_development={n: counts(f) for n, f in old_parts.items()},
            baseline_fit_by_county={n: counts(f) for n, f in old_parts['train'].groupby('region')},
            selected_fit_by_county={n: counts(f) for n, f in parts['train'].groupby('region')},
            selected_by_quarter={n: {str(q): counts(f) for q, f in part.groupby(pd.to_datetime(part.target_date).dt.to_period('Q'))}
                                 for n, part in parts.items()},
            geographic_test_by_county={n: counts(f) for n, f in geographic.groupby('region')},
            geographic_validation_by_county={n: counts(f) for n, f in parts['geographic_validation'].groupby('region')}),
        fit_feature_distributions={f: dict(all=distribution(parts['train'][f]),
            positive=distribution(parts['train'].loc[parts['train'].storm.eq(1), f])) for f in FEATURES},
        calibrators={n: dict(model=b['name'], slope=float(b['calibrator'].coef_[0, 0]),
                             intercept=float(b['calibrator'].intercept_[0]), raw_clip=[1e-6, 1 - 1e-6])
                     for n, b in bundles.items()},
        matched=matched_report, probabilities=probabilities,
        selection=selection_evidence(saved, parts, bundles), threshold_policies=threshold_rows,
        reliance=dict(split_gain=importance,
            grouped_gain_share={g: float(sum(importance[f]['gain_share'] for f in columns)) for g, columns in GROUPS.items()},
            validation_permutations={n: reliance(current, parts[n]) for n in ['validation', 'geographic_validation']}),
        geographic_misses=misses,
        geographic_label_audit=dict(rows_checked=len(geographic), positives=int(geographic.storm.sum()),
            missed_positive_county_days=len(misses), source_records=sum(len(m['noaa_records']) for m in misses),
            unique_episodes=sorted({r['EPISODE_ID'] for m in misses for r in m['noaa_records']})),
        verified_against_saved_metrics=True, protected_inputs_unchanged=True)
    for name, expected in hashes.items():
        if fingerprint(ROOT / name) != expected:
            raise ValueError('Input changed while diagnostics were running; rerun from a stable artifact set: ' + name)
    if fingerprint(raw_path) != raw_hash:
        raise ValueError('NOAA archive changed during diagnostics')
    return report


def main():
    report = build_report()
    path = ROOT / 'output/storm-diagnostics.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(path.relative_to(ROOT)), delta_ap=report['matched']['delta_ap'],
                         missed_geographic_days=len(report['geographic_misses']),
                         saved_metrics_verified=report['verified_against_saved_metrics']), indent=2))


if __name__ == '__main__':
    main()
