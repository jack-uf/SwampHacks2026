"""Leave-one-state-out diagnostic of the served recipes; protocol in docs/SPATIAL_CV_PROTOCOL.md."""
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from backend.model import calibrated
from scripts.build_training import BASE_FEATURES, FEATURES
from scripts.common import ROOT
from scripts.train import block_comparison, metrics

THRESHOLDS = dict(flood=0.030639765827391585, storm=0.07567978303996738)
RECIPES = dict(
    flood=('balanced logistic, base features (served frozen-baseline recipe)', BASE_FEATURES,
           lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))),
    storm=('balanced LightGBM, enhanced features (served lightgbm_enhanced recipe)', FEATURES,
           lambda: LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7, learning_rate=.05, class_weight='balanced',
                                  verbosity=-1, random_state=42, n_jobs=2)))


def development_rows(df):
    kept = df[df.role.eq('development') & (df.target_date < '2026-01-01')].copy()
    assert kept.target_date.max() < '2026-01-01' and set(kept.role) == {'development'}
    return kept


def periods(df):
    return (df[df.target_date < '2025-01-01'], df[(df.target_date >= '2025-01-01') & (df.target_date < '2025-07-01')],
            df[df.target_date >= '2025-07-01'])


def fit(hazard, fit_rows, calibration_rows):
    _, features, make = RECIPES[hazard]
    if calibration_rows[hazard].nunique() != 2 or fit_rows[hazard].nunique() != 2:
        return None
    model = make().fit(fit_rows[features], fit_rows[hazard])
    raw = np.clip(model.predict_proba(calibration_rows[features])[:, 1], 1e-6, 1-1e-6)
    calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw/(1-raw)).reshape(-1, 1), calibration_rows[hazard])
    return dict(model=model, calibrator=calibrator, features=features)


def score_hazard(df, hazard):
    fit_rows, calibration_rows, scored = periods(df)
    seen = fit(hazard, fit_rows, calibration_rows)
    scored = scored.assign(seen=calibrated(seen, scored), left_out=np.nan)
    folds = {}
    for state in sorted(df.state.unique()):
        others = lambda frame: frame[frame.state != state]
        bundle = fit(hazard, others(fit_rows), others(calibration_rows))
        folds[state] = dict(status='ok' if bundle else 'failed: a fit or calibration period lacks a class',
                            fit_states=sorted(others(fit_rows).state.unique()), calibration_states=sorted(others(calibration_rows).state.unique()))
        if bundle:
            mask = scored.state.eq(state)
            scored.loc[mask, 'left_out'] = calibrated(bundle, scored[mask])
    return dict(rows=scored[scored.left_out.notna()], folds=folds)


def summary(frame, hazard, column):
    m = metrics(frame[hazard], frame[column], THRESHOLDS[hazard])
    return dict(rows=m['n'], positives=m['positives'], average_precision=m['pr_auc'], brier=m['brier'],
                expected_positives=float(frame[column].sum()), recall=m['recall'],
                false_alerts_per_100_county_days=m['false_alerts_per_100_county_days'])


def main():
    path = ROOT/'data/training.csv'
    df = development_rows(pd.read_csv(path))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), protocol='docs/SPATIAL_CV_PROTOCOL.md',
                  dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), states=sorted(df.state.unique()),
                  counties=int(df.region.nunique()), hazards={})
    for hazard in RECIPES:
        result = score_hazard(df, hazard)
        rows = result['rows']
        per_state = {}
        for state, group in rows.groupby('state'):
            per_state[state] = dict(rows=len(group), positives=int(group[hazard].sum()),
                                    seen_ap=float(average_precision_score(group[hazard], group.seen)) if group[hazard].sum() else None,
                                    left_out_ap=float(average_precision_score(group[hazard], group.left_out)) if group[hazard].sum() else None,
                                    seen_brier=float(brier_score_loss(group[hazard], group.seen)),
                                    left_out_brier=float(brier_score_loss(group[hazard], group.left_out)))
        report['hazards'][hazard] = dict(recipe=RECIPES[hazard][0], threshold=THRESHOLDS[hazard], folds=result['folds'],
            seen_state=summary(rows, hazard, 'seen'), left_out_state=summary(rows, hazard, 'left_out'), per_state=per_state,
            paired_difference=block_comparison(rows, rows[hazard], rows.seen.to_numpy(), rows.left_out.to_numpy()))
    (ROOT/'models/spatial-cv.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for hazard, h in report['hazards'].items():
        for arm in ('seen_state', 'left_out_state'):
            print(hazard, arm, {k: round(v, 4) if isinstance(v, float) else v for k, v in h[arm].items()})
        print(hazard, 'left-out minus seen', h['paired_difference'])


if __name__ == '__main__':
    main()
