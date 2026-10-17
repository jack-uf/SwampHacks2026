"""Transformer experiment. See docs/TRANSFORMER_SPRINT_PROTOCOL.md before running."""
import copy
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from backend.model import calibrated
from scripts.common import ROOT, download
from scripts import wind_sprint
from scripts.wind_sprint import EXTENDED, THRESHOLD_KMH, digest, rule_metrics, station_metrics, write_json

OUT = ROOT / 'experiments/sprint-20261017-transformer'
PROTOCOL_DOCUMENT = ROOT / 'docs/TRANSFORMER_SPRINT_PROTOCOL.md'
EXTENSION_START, EXTENSION_END = '2026-06-27', '2026-10-15'
FRESH_START = '2026-07-01'
WIND_END = '2026-06-30'
CHECKPOINT = Path.home() / 'Library/Caches/tabpfn' / 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'
CHECKPOINT_SHA256 = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
DEPENDENCIES = ['scripts/wind_sprint.py', 'scripts/train.py', 'scripts/common.py', 'backend/model.py']
PACKAGES = ['tabpfn', 'torch', 'mlx', 'scikit-learn', 'xgboost', 'lightgbm', 'numpy', 'pandas', 'joblib', 'scipy']
THREADS = 8


def verify_checkpoint():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'TabPFN checkpoint missing: {CHECKPOINT}; the frozen experiment requires the cached v2 weights')
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('TabPFN checkpoint hash changed; the frozen protocol pins the v2 weights hash')


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_blocks=2, n_heads=4, dropout=.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * .02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=2 * d_token,
                                           dropout=dropout, activation='relu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_blocks)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        return self.head(self.norm(self.encoder(tokens)[:, 0])).squeeze(-1)


class FTTransformerClassifier:
    """Frozen recipe from the protocol: no search, seed 42, early stop on the calibration rows."""

    def __init__(self, d_token=32, n_blocks=2, n_heads=4, dropout=.1, epochs=60, patience=10,
                 batch=256, lr=1e-3, weight_decay=1e-5, seed=42):
        self.kwargs = dict(d_token=d_token, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout, epochs=epochs,
                           patience=patience, batch=batch, lr=lr, weight_decay=weight_decay, seed=seed)

    def fit(self, X, y, X_eval, y_eval):
        torch.manual_seed(self.kwargs['seed'])
        torch.set_num_threads(THREADS)
        X = np.asarray(X, dtype=float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        scaled, eval_scaled = self._scale(X), self._scale(np.asarray(X_eval, dtype=float))
        y = np.asarray(y, dtype=float)
        pos_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1.0))
        net = FTTransformer(X.shape[1], self.kwargs['d_token'], self.kwargs['n_blocks'], self.kwargs['n_heads'], self.kwargs['dropout'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.kwargs['lr'], weight_decay=self.kwargs['weight_decay'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        X_tensor, y_tensor = torch.tensor(scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        eval_x, eval_y = torch.tensor(eval_scaled, dtype=torch.float32), torch.tensor(np.asarray(y_eval, dtype=float), dtype=torch.float32)
        rng = np.random.default_rng(self.kwargs['seed'])
        best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0
        for _ in range(self.kwargs['epochs']):
            epochs_run += 1
            net.train()
            for start in range(0, len(X_tensor), self.kwargs['batch']):
                rows = torch.tensor(rng.permutation(len(X_tensor))[start:start + self.kwargs['batch']], dtype=torch.long)
                optimizer.zero_grad()
                loss = criterion(net(X_tensor[rows]), y_tensor[rows])
                loss.backward()
                optimizer.step()
            net.eval()
            with torch.no_grad():
                eval_loss = float(nn.functional.binary_cross_entropy_with_logits(net(eval_x), eval_y))
            if eval_loss < best_loss - 1e-6:
                best_loss, best_state, stale = eval_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.kwargs['patience']:
                    break
        net.load_state_dict(best_state)
        self.net_, self.epochs_trained_, self.best_eval_loss_ = net, epochs_run, best_loss
        return self

    def _scale(self, X):
        return (X - self.mean_) / self.std_

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(self._scale(np.asarray(X, dtype=float)), dtype=torch.float32))
            positive = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - positive, positive])


def tabpfn():
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', random_state=42)


def candidate_models(train):
    for name, model, features in wind_sprint.recipes(train):
        if name.endswith('_extended'):
            yield name, model, features
    yield 'tabpfn_extended', tabpfn(), EXTENDED
    yield 'ft_transformer_extended', FTTransformerClassifier(), EXTENDED


def fitted_experiment(model, features, train, calibration, name):
    if isinstance(model, FTTransformerClassifier):
        model.fit(train[features], train.exceedance, calibration[features], calibration.exceedance)
    else:
        model.fit(train[features], train.exceedance)
    raw = np.clip(model.predict_proba(calibration[features])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration.exceedance)
    return dict(name=name, model=model, calibrator=calibrator, features=list(features))


def matched_budget_threshold(scores, y, budget_fraction):
    scores, y = np.asarray(scores), np.asarray(y)
    negatives = np.sort(scores[~y.astype(bool)])[::-1]
    if not len(negatives):
        raise ValueError('Matched-budget threshold needs at least one negative validation row')
    budget = int(np.ceil(budget_fraction * len(negatives)))
    if budget <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    if budget >= len(negatives):
        return float(negatives[-1])
    return float(negatives[budget - 1])


def extension_frames():
    frame = pd.read_csv(OUT / 'dataset-extension.csv.gz')
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Invalid extension rows')
    if (pd.to_datetime(frame.target_date) < pd.Timestamp(FRESH_START)).any():
        raise ValueError('Extension must start at the fresh window; earlier days belong to the frozen dataset')
    frozen = wind_sprint.load_dataset()
    if (pd.to_datetime(frozen.target_date) > pd.Timestamp(WIND_END)).any():
        raise ValueError('Frozen dataset changed: it no longer ends where the protocol says')
    if set(frame.station) - set(frozen.station):
        raise ValueError('Extension contains stations outside the frozen protocol')
    return frozen, frame


def selection_inputs():
    frame = wind_sprint.load_dataset()
    parts = wind_sprint.partition_rows(frame[frame.target_date < '2026-01-01'])
    validation = pd.concat([parts['validation'], parts['geographic_validation']], ignore_index=True)
    return parts, validation


def extend():
    if (OUT / 'dataset-extension.csv.gz').exists():
        raise FileExistsError('Extension already frozen; use select or verify')
    protocol = wind_sprint.load_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, coverage, sources = [], [], []
    for station in protocol['stations']:
        params = [('station', station['sid']), ('data', 'gust'), ('report_type', 3), ('report_type', 4),
                  ('year1', 2026), ('month1', 6), ('day1', 27), ('year2', 2026), ('month2', 10), ('day2', 15),
                  ('tz', 'UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('direct', 'no')]
        observed_path = download('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                                 f"iem-asos-{station['sid']}-202610.csv", params)
        observed = {row['target_date']: row for row in wind_sprint.observation_days(observed_path.read_text())}
        forecast_params = dict(latitude=station['latitude'], longitude=station['longitude'], start_date=EXTENSION_START,
                               end_date=EXTENSION_END, hourly=','.join(v + '_previous_day2' for v in wind_sprint.UNITS),
                               models='gfs_seamless', timezone='GMT')
        forecast_path = download('https://previous-runs-api.open-meteo.com/v1/forecast',
                                 f"wind-forecast-{station['sid']}-202610.json", forecast_params)
        daily = wind_sprint.forecast_features(json.loads(forecast_path.read_text()), station)
        for values in daily:
            if values['target_date'] < FRESH_START or values['target_date'] not in observed:
                continue
            day = observed[values['target_date']]
            rows.append(dict(**values, max_gust_kmh=day['max_gust_kmh'], exceedance=day['exceedance'],
                             first_exceed_hour_utc=day['first_exceed_hour_utc'], station=station['sid'],
                             name=station['name'], state=station['state'], role=station['role']))
        coverage.append(dict(station=station['sid'], role=station['role'], observed_days=len(observed),
                             complete_forecast_days=len(daily),
                             merged_days=len([v for v in daily if v['target_date'] >= FRESH_START and v['target_date'] in observed]),
                             first=min(observed) if observed else None, last=max(observed) if observed else None))
        for path in [observed_path, forecast_path]:
            sources.append(json.loads(path.with_suffix(path.suffix + '.source.json').read_text()))
        print(station['sid'], 'observed days', len(observed), 'merged fresh days', coverage[-1]['merged_days'], flush=True)
        time.sleep(10)
    frame = pd.DataFrame(rows).sort_values(['target_date', 'station'])
    if frame.duplicated(['station', 'target_date']).any() or not np.isfinite(frame[EXTENDED]).all().all():
        raise ValueError('Duplicate or non-finite extension rows')
    frame.to_csv(OUT / 'dataset-extension.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    write_json(OUT / 'provenance-extension.json', dict(created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
        coverage=coverage, dataset_sha256=digest(OUT / 'dataset-extension.csv.gz'),
        protocol_document_sha256=digest(PROTOCOL_DOCUMENT),
        window=dict(requested=[EXTENSION_START, EXTENSION_END], fresh_from=FRESH_START, frozen_through=WIND_END),
        contracts='Same frozen parsing, completeness and issue-time contracts as scripts/wind_sprint.py'))
    print('Extension frozen:', len(frame), 'rows;', frame.station.nunique(), 'stations', flush=True)


def select():
    if (OUT / 'selection.json').exists() or (OUT / 'candidates.joblib').exists():
        raise FileExistsError('Selection is already frozen; use evaluate or verify')
    verify_checkpoint()
    torch.set_num_threads(THREADS)
    parts, validation = selection_inputs()
    train, calibration = parts['train'], parts['calibration']
    reference = (validation.forecast_gust_max_kmh >= THRESHOLD_KMH).astype(float)
    baseline = rule_metrics(validation.exceedance, validation.forecast_gust_max_kmh, THRESHOLD_KMH)
    budget_fraction = baseline['false_alerts_per_100_station_days'] / 100
    candidates, results = {}, {}
    for name, model, features in candidate_models(train):
        start = time.time()
        bundle = fitted_experiment(model, features, train, calibration, name)
        bundle.update(matched_threshold=matched_budget_threshold(calibrated(bundle, validation), validation.exceedance, budget_fraction),
                      fit_seconds=round(time.time() - start, 1))
        threshold, supported = wind_sprint.recall_threshold(validation.exceedance, calibrated(bundle, validation), reference, .5)
        bundle.update(threshold=threshold, warning_supported=supported)
        scores = {part: station_metrics(parts[part].exceedance, calibrated(bundle, parts[part]), threshold)
                  for part in ['validation', 'geographic_validation']}
        scores.update(features=bundle['features'], threshold=threshold, matched_threshold=bundle['matched_threshold'],
                      fit_seconds=bundle['fit_seconds'], warning_supported=supported)
        scores['selection_ap'] = float(np.mean([scores[k]['pr_auc'] or 0. for k in ['validation', 'geographic_validation']]))
        scores['selection_brier'] = float(np.mean([scores[k]['brier'] for k in ['validation', 'geographic_validation']]))
        candidates[name], results[name] = bundle, scores
        print(name, 'fit seconds', bundle['fit_seconds'], 'threshold', round(threshold, 4), 'validation AP',
              round(scores['selection_ap'], 4), 'supported', supported, flush=True)
    eligible = [name for name in results if results[name]['warning_supported']]
    winner = sorted(eligible or list(results), key=lambda n: (-results[n]['selection_ap'], results[n]['selection_brier'], n))[0]
    joblib.dump(dict(candidates=candidates), OUT / 'candidates.joblib')
    write_json(OUT / 'selection.json', dict(created_at=datetime.now(timezone.utc).isoformat(), selected=winner,
        eligible_candidates=eligible, candidates=results, baseline_validation=baseline,
        budget_fraction=budget_fraction, dataset_sha256=digest(wind_sprint.OUT / 'training.csv.gz'),
        protocol_sha256=digest(wind_sprint.OUT / 'protocol.json'), model_sha256=digest(OUT / 'candidates.joblib'),
        protocol_document_sha256=digest(PROTOCOL_DOCUMENT), checkpoint_sha256=CHECKPOINT_SHA256,
        code_sha256=digest(ROOT / 'scripts/transformer_sprint.py'),
        dependencies={name: digest(ROOT / name) for name in DEPENDENCIES},
        versions={name: importlib.metadata.version(name) for name in PACKAGES}, threads=THREADS,
        selection_rule='Supported candidates first; equal temporal/geographic validation AP, mean Brier tie-break, then name',
        test_outcomes_inspected=False))
    print('Selected', winner, flush=True)


def transformer_gate_failures(updated, baseline, intervals, stations, supported, matched):
    failures = wind_sprint.integration_failures(updated, baseline, intervals, stations, supported)
    if matched['recall'] < baseline['recall'] - 1e-12:
        failures.append('Matched-budget recall below the forecast rule')
    return failures


def evaluate():
    if (OUT / 'evaluation.json').exists():
        raise FileExistsError('Final evaluation already exists; do not tune on it. Use verify')
    selection = frozen_selection()
    with (OUT / 'evaluation-started.json').open('x') as marker:
        json.dump(dict(started_at=datetime.now(timezone.utc).isoformat(), selection_sha256=digest(OUT / 'selection.json')), marker)
    verify_checkpoint()
    torch.set_num_threads(THREADS)
    parts, validation = selection_inputs()
    frozen, fresh = extension_frames()
    cohorts = dict(
        fresh_temporal=fresh[fresh.role.eq('development')],
        fresh_geographic_validation=fresh[fresh.role.eq('geographic_validation')],
        fresh_geographic_test=fresh[fresh.role.eq('geographic_test')],
        inspected_temporal=frozen[frozen.role.eq('development') & (frozen.target_date >= '2026-01-01')],
        inspected_geographic_test=frozen[frozen.role.eq('geographic_test') & (frozen.target_date >= '2026-01-01')])
    candidates = joblib.load(OUT / 'candidates.joblib')['candidates']
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), selected=selection['selected'],
                  selection_sha256=digest(OUT / 'selection.json'), protocol='docs/TRANSFORMER_SPRINT_PROTOCOL.md',
                  checkpoint_sha256=CHECKPOINT_SHA256, cohorts={}, baseline_contract='Archived forecast maximum gust at 60 km/h',
                  metrics_contract='Selection-threshold metrics plus the matched-budget operating point defined in the protocol')
    prediction_frames = []
    for cohort, data in cohorts.items():
        info = dict(rows=len(data), stations=int(data.station.nunique()), positives=int(data.exceedance.sum()),
                    states=sorted(set(data.state)), models={})
        for name, bundle in candidates.items():
            probability = calibrated(bundle, data)
            selection_metrics = station_metrics(data.exceedance, probability, bundle['threshold'])
            matched_metrics = station_metrics(data.exceedance, probability, bundle['matched_threshold'])
            info['models'][name] = dict(selection=selection_metrics, matched_budget=matched_metrics)
            prediction_frames.append(pd.DataFrame(dict(cohort=cohort, station=data.station, target_date=data.target_date,
                label=data.exceedance, model=name, probability=probability,
                selection_alert=probability >= bundle['threshold'], matched_alert=probability >= bundle['matched_threshold'])))
        info['baseline'] = rule_metrics(data.exceedance, data.forecast_gust_max_kmh, THRESHOLD_KMH)
        info['climatology'] = station_metrics(data.exceedance, np.full(len(data), parts['train'].exceedance.mean()), .5)
        winner = candidates[selection['selected']]
        probability = calibrated(winner, data)
        info['winner'] = selection['selected']
        info['paired_uncertainty'] = wind_sprint.paired_intervals(data, data.exceedance, data.forecast_gust_max_kmh,
                                                                  probability, THRESHOLD_KMH, winner['threshold'])
        info['lead_time_winner'] = wind_sprint.lead_times(data, probability >= winner['threshold'])
        info['stations_detail'] = {station: dict(baseline=rule_metrics(group.exceedance, group.forecast_gust_max_kmh, THRESHOLD_KMH),
                                                 updated=station_metrics(group.exceedance, calibrated(winner, group), winner['threshold']))
                                   for station, group in data.groupby('station')}
        report['cohorts'][cohort] = info
        print(cohort, 'rows', len(data), 'positives', info['positives'], 'rule AP', round(info['baseline']['pr_auc'] or 0, 4),
              'winner AP', round(info['models'][info['winner']]['selection']['pr_auc'] or 0, 4), flush=True)
    primary = report['cohorts']['fresh_temporal']
    winner = primary['models'][selection['selected']]
    report['gate'] = transformer_gate_failures(winner['selection'], primary['baseline'], primary['paired_uncertainty'],
                                               primary['stations_detail'],
                                               selection['candidates'][selection['selected']]['warning_supported'],
                                               winner['matched_budget'])
    report['decision'] = 'propose_integration_review' if not report['gate'] else 'retain_forecast_threshold'
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / 'predictions.csv.gz', index=False, compression=dict(method='gzip', mtime=0))
    report['predictions_sha256'] = digest(OUT / 'predictions.csv.gz')
    write_json(OUT / 'evaluation.json', report)
    print('Decision:', report['decision'], 'failures:', report['gate'], flush=True)


def frozen_selection():
    selection = json.loads((OUT / 'selection.json').read_text())
    for path, field in [(OUT / 'candidates.joblib', 'model_sha256'), (wind_sprint.OUT / 'training.csv.gz', 'dataset_sha256'),
                        (wind_sprint.OUT / 'protocol.json', 'protocol_sha256'), (PROTOCOL_DOCUMENT, 'protocol_document_sha256'),
                        (ROOT / 'scripts/transformer_sprint.py', 'code_sha256')]:
        if digest(path) != selection[field]:
            raise ValueError('Frozen selection input changed: ' + str(path))
    for name, checksum in selection['dependencies'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Frozen dependency changed: ' + name)
    for name, version in selection['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Frozen package version changed: ' + name)
    if selection['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Frozen checkpoint hash changed')
    return selection


def verify():
    selection = frozen_selection()
    verify_checkpoint()
    report = json.loads((OUT / 'evaluation.json').read_text())
    if digest(OUT / 'selection.json') != report['selection_sha256'] or digest(OUT / 'predictions.csv.gz') != report['predictions_sha256']:
        raise ValueError('Evaluation artifact checksum mismatch')
    assert report['selected'] == selection['selected']
    assert report['decision'] == ('retain_forecast_threshold' if report['gate'] else 'propose_integration_review')
    predictions = pd.read_csv(OUT / 'predictions.csv.gz')
    parts, validation = selection_inputs()
    frozen, fresh = extension_frames()
    cohorts = dict(
        fresh_temporal=fresh[fresh.role.eq('development')],
        fresh_geographic_validation=fresh[fresh.role.eq('geographic_validation')],
        fresh_geographic_test=fresh[fresh.role.eq('geographic_test')],
        inspected_temporal=frozen[frozen.role.eq('development') & (frozen.target_date >= '2026-01-01')],
        inspected_geographic_test=frozen[frozen.role.eq('geographic_test') & (frozen.target_date >= '2026-01-01')])
    candidates = joblib.load(OUT / 'candidates.joblib')['candidates']
    for cohort, data in cohorts.items():
        saved = predictions[predictions.cohort.eq(cohort)]
        assert len(saved) == len(data) * len(candidates)
        for name, bundle in candidates.items():
            rows = saved[saved.model.eq(name)]
            assert list(rows.station) == list(data.station) and list(rows.target_date) == list(data.target_date)
            assert np.array_equal(rows.label, data.exceedance)
            probability = calibrated(bundle, data)
            assert np.allclose(probability, rows.probability, atol=1e-12, rtol=0)
            assert station_metrics(data.exceedance, probability, bundle['threshold']) == report['cohorts'][cohort]['models'][name]['selection']
    print('Verified frozen inputs, checkpoint, prediction rows and pooled metrics')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extend', 'select', 'evaluate', 'verify'])
    args = parser.parse_args()
    dict(extend=extend, select=select, evaluate=evaluate, verify=verify)[args.command]()

