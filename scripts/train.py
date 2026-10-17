"""Predeclared, disjoint temporal/spatial experiment; final test never selects."""
import hashlib
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scripts.build_training import BASE_FEATURES, FEATURES
from scripts.common import ROOT
from backend.model import calibrated


def metrics(y, p, threshold):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    bins=[]
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        use=(p>=low)&((p<high) if high<1 else (p<=high))
        bins.append(dict(lower=float(low),upper=float(high),n=int(use.sum()),
                         predicted=float(p[use].mean()) if use.any() else None,
                         observed=float(y[use].mean()) if use.any() else None))
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)),pr_auc=float(average_precision_score(y,p)) if y.sum() else None,
        roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,brier=float(brier_score_loss(y,p)),
        confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],false_alerts_per_100_county_days=float(fp*100/len(y)),
        false_positive_rate=float(fp/(fp+tn)) if fp+tn else None,alert_rate=float(pred.mean()),threshold=float(threshold),
        calibration=bins,calibration_ece=float(sum(b['n']*abs(b['predicted']-b['observed']) for b in bins if b['n'])/len(y)))


def threshold_policies(y,p):
    thresholds=np.unique(np.r_[0, p, np.nextafter(max(p),np.inf)])
    rows=[]
    for t in thresholds:
        pred=p>=t;tp=int(np.sum(pred & (np.asarray(y)==1)));fp=int(np.sum(pred & (np.asarray(y)==0)))
        precision=tp/max(tp+fp,1);recall=tp/max(int(sum(y)),1)
        rows.append(dict(threshold=float(t),precision=precision,recall=recall,alert_rate=float(pred.mean()),
                         false_alerts_per_100_county_days=float(fp*100/len(y)),
                         f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                         f2=5*precision*recall/(4*precision+recall) if precision+recall else 0))
    feasible=[r for r in rows if r['alert_rate']<=.20 and r['recall']>=.30]
    policies={k:max(rows,key=lambda r:r[k]) for k in ['f1','f2']}
    # shortcut: budget and recall floor are elicitation hypotheses; replace after prospective supervisor cost study.
    policies['budget']=max(feasible,key=lambda r:r['f1']) if feasible else None
    chosen=policies['budget'] or policies['f1']
    return chosen['threshold'],policies,bool(feasible)


def partitions(df):
    development=df.role.eq('development')
    return dict(train=df[development & (df.target_date<'2025-01-01')],
        calibration=df[development & (df.target_date>='2025-01-01') & (df.target_date<'2025-07-01')],
        validation=df[development & (df.target_date>='2025-07-01') & (df.target_date<'2026-01-01')],
        geographic_validation=df[df.role.eq('geographic_validation') & (df.target_date>='2025-07-01') & (df.target_date<'2026-01-01')],
        test=df[development & (df.target_date>='2026-01-01')],
        geographic_test=df[df.role.eq('geographic_test') & (df.target_date>='2026-01-01')])


def block_comparison(frame, y, old, new, seed=42):
    """Paired calendar-week blocks keep same-week counties together; descriptive CI."""
    blocks=pd.to_datetime(frame.target_date).dt.to_period('W').astype(str).to_numpy()
    unique=np.unique(blocks);rng=np.random.default_rng(seed);values=[]
    for _ in range(300):
        indices=np.concatenate([np.flatnonzero(blocks==week) for week in rng.choice(unique,len(unique),replace=True)])
        if np.asarray(y)[indices].sum()==0:continue
        values.append([average_precision_score(np.asarray(y)[indices],new[indices])-average_precision_score(np.asarray(y)[indices],old[indices]),
                       brier_score_loss(np.asarray(y)[indices],new[indices])-brier_score_loss(np.asarray(y)[indices],old[indices])])
    if len(unique)<10 or len(values)<100:
        return dict(method='paired calendar-week bootstrap',available=False,reason='Insufficient resampled event weeks')
    intervals=np.quantile(values,[.025,.975],axis=0)
    return dict(method='300 paired calendar-week block resamples; spatial rows within week retained; descriptive 95% interval',
                weeks=len(unique),replicates=len(values),delta_ap_95=intervals[:,0].tolist(),delta_brier_95=intervals[:,1].tolist())


def main():
    df=pd.read_csv(ROOT/'data/training.csv').sort_values(['target_date','region']);parts=partitions(df)
    assert parts['train'].target_date.max()<parts['calibration'].target_date.min()<parts['validation'].target_date.min()<parts['test'].target_date.min()
    baseline=joblib.load(ROOT/'experiments/baseline/models/hazards.joblib')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),features=FEATURES,rows=len(df),counties=df.region.nunique(),
        dataset_sha256=hashlib.sha256((ROOT/'data/training.csv').read_bytes()).hexdigest(),
        partitions={k:dict(rows=len(v),start=str(v.target_date.min()),end=str(v.target_date.max()),counties=v.region.nunique(),
                           flood_positives=int(v.flood.sum()),storm_positives=int(v.storm.sum())) for k,v in parts.items()},hazards={},
        protocol='docs/RESEARCH_PLAN.md committed before expanded data; final January–June 2026 evaluated only after selection',
        limitations=['NOAA-reported county events, not site incidents or injuries; unreported does not mean safe.',
          '24 convenience-sampled counties; geographic test states excluded from new fitting; no nationwide validation.',
          'Previous Runs is fixed 48h nominal lead per hour, not one initialization; dissemination timestamps unavailable.',
          'UTC target day begins 24h after midnight decision time; not the next local work shift.',
          'Antecedent precipitation is forecast, not observed; no reanalysis or future observations.',
          'Spatial and seasonal dependence and reporting revisions limit inference; reliability bins retain sample counts.',
          'Warning budget (20%) and recall floor (30%) are unvalidated operational hypotheses.',
          'Index weights and scope restriction are conservative heuristics, not calibrated injury probability or safety boundaries.',
          'SHAP explains calibrated model probability using 16 training-background rows and independent feature masking; correlated inputs can create implausible combinations.',
          'Frozen baseline includes Arizona fitting; its Arizona geographic result is not an independent holdout.'])
    bundles={};selected={};predictions={}
    for hazard in ['flood','storm']:
        train=parts['train'];cal=parts['calibration'];ratio=(len(train)-train[hazard].sum())/train[hazard].sum()
        candidates={}
        for enhanced in [False,True]:
            suffix='enhanced' if enhanced else 'base';features=FEATURES if enhanced else BASE_FEATURES
            candidates['logistic_'+suffix]=(make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)),features)
            candidates['xgboost_'+suffix]=(XGBClassifier(n_estimators=140,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.9,scale_pos_weight=ratio,random_state=42,n_jobs=2),features)
            candidates['lightgbm_'+suffix]=(LGBMClassifier(n_estimators=140,max_depth=3,num_leaves=7,learning_rate=.05,class_weight='balanced',verbosity=-1,random_state=42,n_jobs=2),features)
        candidates['logistic_unweighted']=(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42)),FEATURES)
        results={};hazard_bundles={}
        for name,(model,features) in candidates.items():
            model.fit(train[features],train[hazard]);raw=np.clip(model.predict_proba(cal[features])[:,1],1e-6,1-1e-6)
            if cal[hazard].nunique()!=2:raise ValueError('Calibration must contain both classes')
            calibrator=LogisticRegression(C=1,random_state=42).fit(np.log(raw/(1-raw)).reshape(-1,1),cal[hazard])
            hazard_bundles[name]=dict(model=model,calibrator=calibrator,name=name,features=features,
                background=train[features].sample(min(16,len(train)),random_state=42).reset_index(drop=True))
        frozen=dict(baseline['hazards'][hazard],features=BASE_FEATURES,name='frozen_baseline',
                    background=pd.read_csv(ROOT/'experiments/baseline/data/training.csv').query("target_date < '2024-09-01' and region != 'Los Angeles, CA'")[BASE_FEATURES].sample(16,random_state=42).reset_index(drop=True))
        hazard_bundles['frozen_baseline']=frozen
        for name,bundle in hazard_bundles.items():
            vp=calibrated(bundle,parts['validation']);t,policies,supported=threshold_policies(parts['validation'][hazard],vp)
            bundle.update(threshold=t,warning_supported=supported)
            results[name]={part:metrics(parts[part][hazard],calibrated(bundle,parts[part]),t) for part in ['validation','geographic_validation']}
            results[name]['threshold_policies']=policies;results[name]['warning_supported']=supported
            results[name]['selection_score']=float(np.mean([results[name][part]['pr_auc'] or 0 for part in ['validation','geographic_validation']]))
            if name!='frozen_baseline':
                raw= bundle['model'].predict_proba(parts['validation'][bundle['features']])[:,1]
                results[name]['uncalibrated_validation_brier']=float(brier_score_loss(parts['validation'][hazard],raw))
        winner=max(results,key=lambda n:(results[n]['selection_score'],-results[n]['validation']['brier']))
        selected[hazard]=winner;bundles[hazard]=hazard_bundles;predictions[hazard]=results
    # Freeze winners and threshold policies to disk BEFORE any final-test metric is calculated.
    selection={h:dict(selected=n,threshold=bundles[h][n]['threshold'],validation=predictions[h][n]['validation'],
                       geographic_validation=predictions[h][n]['geographic_validation']) for h,n in selected.items()}
    (ROOT/'models/selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False))
    for hazard,winner in selected.items():
        results=predictions[hazard];hazard_bundles=bundles[hazard];chosen=hazard_bundles[winner]
        for name,b in hazard_bundles.items():
            for part in ['test','geographic_test']:
                frame=parts[part];p=calibrated(b,frame)
                results[name][part]=metrics(frame[hazard],p,b['threshold'])
        frame=parts['test'];p=calibrated(chosen,frame)
        results[winner]['quarters']={str(q):metrics(group[hazard],calibrated(chosen,group),chosen['threshold']) for q,group in frame.groupby(pd.to_datetime(frame.target_date).dt.to_period('Q'))}
        results[winner]['counties']={name:metrics(group[hazard],calibrated(chosen,group),chosen['threshold']) for name,group in frame.groupby('region')}
        original=set(pd.read_csv(ROOT/'experiments/baseline/data/training.csv').region)-{'Los Angeles, CA','Maricopa, AZ'}
        common=frame[frame.region.isin(original)];old=hazard_bundles['frozen_baseline'];op=calibrated(old,common);npred=calibrated(chosen,common)
        original_threshold=baseline['hazards'][hazard]['threshold']
        report['hazards'][hazard]=dict(selected=winner,selection='mean temporal/geographic-validation AP; temporal Brier tie-break',candidates=results,
            climatology_test=metrics(frame[hazard],np.full(len(frame),parts['train'][hazard].mean()),.5),
            same_final_rows=dict(population='Original ten development counties; January–June 2026; same labels/features at original points',
                original_threshold_baseline=metrics(common[hazard],op,original_threshold),updated=metrics(common[hazard],npred,chosen['threshold']),
                paired_uncertainty=block_comparison(common,common[hazard],op,npred)))
        # Scope follows the actual fitting population, never the final-test outcome.
        region_metadata=json.loads((ROOT/'data/regions.json').read_text())
        chosen['supported_regions']=[r for r in region_metadata if r['role']=='development'] if winner!='frozen_baseline' else [r for r in region_metadata if r['region'] in set(pd.read_csv(ROOT/'experiments/baseline/data/training.csv').region)-{'Los Angeles, CA'}]
    joblib.dump(dict(features=FEATURES,hazards={h:bundles[h][n] for h,n in selected.items()},trained_at=report['created_at'],dataset_sha256=report['dataset_sha256']),ROOT/'models/hazards.joblib')
    (ROOT/'models/evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Expanded model evaluation','','See RESEARCH_PLAN.md for the predeclared design. Final test is January–June 2026; 2025 is development data now.','','| Hazard | Selected | County-days (+) | Precision | Recall | AP | Brier | False alerts /100 county-days |','|---|---|---:|---:|---:|---:|---:|---:|']
    for h,info in report['hazards'].items():
        m=info['candidates'][info['selected']]['test'];lines.append(f"| {h} | {info['selected']} | {m['n']} ({m['positives']}) | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.4f} | {m['false_alerts_per_100_county_days']:.2f} |")
    lines+=['','## Compatible comparison: same ten original counties, same 2026 dates','','| Hazard / version | Precision | Recall | AP | Brier | False alerts /100 county-days |','|---|---:|---:|---:|---:|---:|']
    for h,info in report['hazards'].items():
        for version in ['original_threshold_baseline','updated']:
            m=info['same_final_rows'][version];lines.append(f"| {h} / {version} | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.4f} | {m['false_alerts_per_100_county_days']:.2f} |")
    lines+=['','The original threshold is unchanged in this comparison; the updated threshold comes only from 2025 validation. Differences combine expanded fitting, calibration, features and threshold policy; they do not isolate one intervention. Original 2025 metrics cannot be directly compared with a different 2026 population. Paired week-block intervals, all candidate results (including failures), prevalence, reliability counts, geographic tests and quarterly/county variability are in models/evaluation.json. Confusion matrices are [[TN, FP], [FN, TP]]. AP means average precision.','','## Limits','',*['- '+x for x in report['limitations']]]
    (ROOT/'docs/EVALUATION.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:25]))


if __name__=='__main__':main()
