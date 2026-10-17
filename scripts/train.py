"""Chronological benchmark with disjoint probability calibration, selection, and test."""
import json
import warnings
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scripts.build_training import FEATURES
from scripts.common import ROOT


def metrics(y, p, threshold):
    prediction = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(n=len(y), positives=int(sum(y)), precision=float(precision_score(y, prediction, zero_division=0)),
                recall=float(recall_score(y, prediction, zero_division=0)),
                pr_auc=float(average_precision_score(y, p)) if sum(y) else None,
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                brier=float(brier_score_loss(y, p)), confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                false_alerts_per_100_county_days=float(100*fp/len(y)),
                false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,
                threshold=float(threshold))


def main():
    df = pd.read_csv(ROOT / 'data/training.csv').sort_values(['target_date', 'region'])
    geo = df.region == 'Los Angeles, CA'
    train = df[(df.target_date < '2024-09-01') & ~geo]
    calibration = df[(df.target_date >= '2024-09-01') & (df.target_date < '2024-11-01') & ~geo]
    validation = df[(df.target_date >= '2024-11-01') & (df.target_date < '2025-01-01') & ~geo]
    test = df[(df.target_date >= '2025-01-01') & ~geo]
    geographic = df[(df.target_date >= '2025-01-01') & geo]
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), features=FEATURES,
        partitions={k: dict(rows=len(v), start=str(v.target_date.min()), end=str(v.target_date.max()),
                            flood_positives=int(v.flood.sum()), storm_positives=int(v.storm.sum()))
                    for k, v in [('train',train),('calibration',calibration),('validation',validation),('test',test),('geographic_test',geographic)]},
        hazards={}, limitations=[
            'County-day event reporting is incomplete and spatially coarse; no reported event is not proof of no hazard.',
            'Twelve convenience-sampled US counties in 2024–2025; no nationwide or site-level validation.',
            'GFS forecasts sampled at one representative county point, with >=48-hour lead, not one fixed model run.',
            'Predicts tomorrow UTC (24–48h from issue midnight), not the next rolling 24h or local work shift.',
            'No realized target-day weather or retrospective reanalysis used as a predictor.',
            'Floodplain, antecedent rain and soil moisture excluded from ML until an issue-time-consistent archive is added.',
            'Small validation event counts make threshold, calibration and model ranking uncertain.',
            'Local explanations are reference-feature sensitivity, not causal or additive SHAP attributions.'
        ])
    artifacts = {}
    for hazard in ['flood', 'storm']:
        y = train[hazard]
        ratio = (len(y)-sum(y))/max(sum(y), 1)
        candidates = dict(
            xgboost=XGBClassifier(n_estimators=140, max_depth=3, learning_rate=.05, subsample=.8,
                                   colsample_bytree=.9, scale_pos_weight=ratio, random_state=42, n_jobs=2),
            lightgbm=LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7, learning_rate=.05,
                                   class_weight='balanced', verbosity=-1, random_state=42, n_jobs=2),
            logistic=make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                   LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)))
        results = {}
        bundles = {}
        for name, model in candidates.items():
            model.fit(train[FEATURES], y)
            # Sigmoid calibration is fitted on a later, separate, naturally imbalanced period.
            raw = np.clip(model.predict_proba(calibration[FEATURES])[:,1], 1e-6, 1-1e-6)
            if calibration[hazard].nunique() != 2:
                raise ValueError(f'{hazard}: calibration period needs both classes')
            calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw/(1-raw)).reshape(-1,1), calibration[hazard])
            def probability(frame):
                p = np.clip(model.predict_proba(frame[FEATURES])[:,1], 1e-6, 1-1e-6)
                return calibrator.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
            vp = probability(validation)
            # Optimize F2 on validation: recall weighted higher, never use test to choose.
            thresholds = np.unique(np.r_[np.linspace(.005,.5,100), vp])
            def f2(t):
                precision = precision_score(validation[hazard], vp >= t, zero_division=0)
                recall = recall_score(validation[hazard], vp >= t, zero_division=0)
                return 5*precision*recall/(4*precision+recall) if precision+recall else 0
            threshold = float(max(thresholds, key=f2))
            results[name] = dict(validation=metrics(validation[hazard], vp, threshold),
                                 test=metrics(test[hazard], probability(test), threshold),
                                 geographic_test=metrics(geographic[hazard], probability(geographic), threshold))
            bundles[name] = dict(model=model, calibrator=calibrator, threshold=threshold,
                                 reference=train[FEATURES].median().to_dict(), name=name)
        winner = max(results, key=lambda n: (results[n]['validation']['pr_auc'] or 0, -results[n]['validation']['brier']))
        artifacts[hazard] = bundles[winner]
        report['hazards'][hazard] = dict(selected=winner, selection='highest validation PR-AUC; Brier tie-break', candidates=results)
        report['hazards'][hazard]['climatology_test'] = metrics(test[hazard], np.full(len(test), train[hazard].mean()), .5)
    (ROOT / 'models').mkdir(exist_ok=True)
    joblib.dump(dict(features=FEATURES, hazards=artifacts, trained_at=report['created_at']), ROOT / 'models/hazards.joblib')
    (ROOT / 'models/evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Measured model evaluation', '', 'Generated by `python -m scripts.train` from authentic cached NOAA and GFS records.', '',
             'Selection uses November–December 2024 PR-AUC. January–December 2025 stays held out. Los Angeles is excluded from all fitting and selection.', '',
             '| Hazard | Selected | Test rows (+) | Precision | Recall | PR-AUC | ROC-AUC | Brier | False alerts /100 days |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for h, info in report['hazards'].items():
        m = info['candidates'][info['selected']]['test']
        lines.append(f"| {h} | {info['selected']} | {m['n']} ({m['positives']}) | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.4f} | {m['false_alerts_per_100_county_days']:.2f} |")
    lines += ['', '## Design and limits', '', *['- '+s for s in report['limitations']], '',
              'All candidate results, thresholds, confusion matrices, geographic test and partition sizes are in `models/evaluation.json`. Confusion matrices use [[TN, FP], [FN, TP]]. PR-AUC is average precision. Random seeds are 42. Candidate hyperparameters are fixed before evaluation.', '',
              'Calibration uses September–October 2024 sigmoid regression. Thresholds maximize validation F2. Training uses class weights; evaluation preserves natural prevalence. Missing hourly days are discarded, never labeled dry. NOAA event durations are expanded by UTC overlap; standard timezone offsets are parsed explicitly.', '',
              'The API displays model probabilities as county-event screening estimates, with a regional extrapolation flag. Outside 150 km of training representatives, ML is excluded from the operational index. This radius is a conservative demo guard, not an empirically validated transfer boundary.', '',
              'Model is an experimental retrospective benchmark, not evidence of safe working conditions. No worker-injury labels are available.']
    (ROOT / 'docs/EVALUATION.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))


if __name__ == '__main__':
    main()
