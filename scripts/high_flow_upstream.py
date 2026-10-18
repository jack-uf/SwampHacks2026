"""Subdaily connected-gauge development experiment; reserved geographic pairs stay unopened."""
import argparse
import concurrent.futures
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts import mlp_geo as archive
from scripts.flood_sprint import write_json
from scripts.high_flow_research import threshold
from scripts.high_flow_signals import OUT
from scripts.hydrology_research import ROOT, digest
from scripts.mlp_hybrid_worker import physical_score
from scripts.shift_hydrology import decision_rows, hourly_flow, FEATURES
from scripts.shift_hydrology_metrics import metrics

UP = OUT/'upstream'
PAIRS = ROOT/'experiments/sprint-20261017-upstream/pairs.json'
UP_FEATURES = ['up_q_2','up_q_8','up_q_14','up_q_26','up_q_50','up_q_170',
    'up_rise_6','up_rise_24','up_rise_48','relative_proximity','relative_rise','area_ratio']


def upstream_features(q, reference, times):
    x = np.log1p(q/reference)
    f = pd.DataFrame({f'up_q_{lag}': x.shift(lag) for lag in [2,8,14,26,50,170]})
    for lag in [6,24,48]:
        f[f'up_rise_{lag}'] = x.shift(2)-x.shift(lag+2)
    valid = q.notna().rolling(169).sum().shift(2).eq(169)
    f.loc[~valid] = np.nan
    return f.reindex(pd.to_datetime(times, utc=True)).reset_index(drop=True)


def collect():
    if (UP/'development.csv.gz').exists():
        raise FileExistsError('Upstream development data already materialized')
    pairs = [p for p in json.loads(PAIRS.read_text())['pairs'] if p['role']=='development']
    archive.OUT = UP
    UP.mkdir(parents=True, exist_ok=True)
    write_json(UP/'acquisition.json', dict(pair_registry_sha256=digest(PAIRS), pairs=pairs,
        years=list(range(2020,2025)), geographic_pairs_opened=False,
        purpose='New subdaily development data. Prior daily upstream study stopped before fitting.'))
    identifiers = sorted({sid for p in pairs for sid in [p['station'],p['upstream']]})
    def one(sid):
        try:
            reference = archive.reference(sid)
            years = [archive.observations(sid, year) for year in range(2020,2025)]
            q = hourly_flow(pd.concat(years).sort_index())
            q = q.reindex(pd.date_range('2020-01-01','2025-01-01',freq='h',tz='UTC',inclusive='left'))
            print('Upstream acquisition',sid,int(q.notna().sum()),'accepted hours',flush=True)
            return sid, (reference,q), None
        except ValueError as exc:
            print('Upstream exclusion',sid,str(exc),flush=True)
            return sid,None,str(exc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(one,identifiers))
    observed = {sid:data for sid,data,error in records if data is not None}
    frames, coverage = [], []
    for pair in pairs:
        sid, upstream = pair['station'], pair['upstream']
        if sid not in observed or upstream not in observed:
            coverage.append(dict(station=sid,upstream=upstream,excluded='Pair lacks eligible reference or discharge'))
            continue
        meta = pair['downstream']
        station = dict(station=sid, role='development', huc=meta['huc_cd'],
            latitude=float(meta['dec_lat_va']),longitude=float(meta['dec_long_va']),
            area_km2=float(meta['drain_area_va'])*2.589988110336,elevation_m=float(meta['alt_va'])*.3048)
        reference,q = observed[sid]
        f,_,_ = decision_rows(q,reference,station)
        f=f.loc[f.horizon.eq(6)].reset_index(drop=True)
        uref,uq = observed[upstream]
        u = upstream_features(uq,uref,f.decision_time)
        f=pd.concat([f,u],axis=1)
        f['relative_proximity']=f.q_2-f.up_q_2
        f['relative_rise']=f.rise_6-f.up_rise_6
        f['area_ratio']=pair['area_ratio']
        good=f[FEATURES+UP_FEATURES].notna().all(axis=1)
        coverage.append(dict(station=sid,upstream=upstream,local_rows=len(f),paired_rows=int(good.sum())))
        frames.append(f.loc[good])
    if not frames:
        raise ValueError('No usable paired development observations')
    f=pd.concat(frames,ignore_index=True)
    f.to_csv(UP/'development.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_json(UP/'dataset.json',dict(sha256=digest(UP/'development.csv.gz'),coverage=coverage,
        acquisition_exclusions={sid:error for sid,data,error in records if error},
        sources={str(p.relative_to(ROOT)):digest(p) for p in (UP/'sources').rglob('*.json.gz')},
        availability='Retrospective approved discharge with assumed two-hour lag; not historical receipt evidence.'))


def fit():
    from lightgbm import LGBMClassifier
    path=UP/'development.csv.gz'
    if digest(path)!=json.loads((UP/'dataset.json').read_text())['sha256']:
        raise ValueError('Upstream dataset changed')
    destination=UP/'results.json'
    if destination.exists():raise FileExistsError(destination)
    f=pd.read_csv(path,dtype={'station':str,'huc':str},keep_default_na=False)
    t=pd.to_datetime(f.decision_time,utc=True)
    safe=(t+pd.Timedelta(hours=24)).dt.year.eq(t.dt.year)
    parts={key:f.loc[safe&mask].reset_index(drop=True) for key,mask in dict(
        train=t.dt.year.le(2021),calibration=t.dt.year.eq(2022),threshold=t.dt.year.eq(2023),validation=t.dt.year.eq(2024)).items()}
    if any(p.empty or p.target.nunique()<2 for p in parts.values()):raise ValueError('Insufficient paired population')
    write_json(UP/'fit-freeze.json',dict(dataset_sha256=digest(path), code_sha256=digest(ROOT/'scripts/high_flow_upstream.py'),
        configurations=['linear','damped','upstream_physical','local_lightgbm','upstream_lightgbm','upstream_logistic'],
        partitions={k:dict(rows=len(v),events=int(v.target.sum()),gauges=v.station.nunique()) for k,v in parts.items()},
        evidence='Adaptive development only; uninspected geographic pairs remain unopened.'))
    results=[]
    for name in ['linear','damped','upstream_physical','local_lightgbm','upstream_lightgbm','upstream_logistic']:
        features=FEATURES+(UP_FEATURES if name.startswith('upstream_') else [])
        model=None
        if name.endswith('lightgbm'):
            model=LGBMClassifier(n_estimators=250,learning_rate=.04,num_leaves=15,max_depth=4,
                min_child_samples=80,reg_lambda=5,n_jobs=2,random_state=42,verbosity=-1,deterministic=True)
        elif name.endswith('logistic'):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))
        if model is not None:model.fit(parts['train'][features],parts['train'].target)
        def raw(frame):
            if name in ['linear','damped']:return physical_score(name,frame)
            if name=='upstream_physical':return frame.up_q_2.to_numpy()+frame.up_rise_6.to_numpy()*8/6
            p=np.clip(model.predict_proba(frame[features])[:,1],1e-6,1-1e-6)
            return np.log(p/(1-p))
        cal=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=42))
        cal.fit(raw(parts['calibration']).reshape(-1,1),parts['calibration'].target)
        if cal[-1].coef_[0,0]<=0:
            results.append(dict(name=name,excluded='Nonpositive calibration slope'));continue
        pth=cal.predict_proba(raw(parts['threshold']).reshape(-1,1))[:,1]
        cut=threshold(parts['threshold'],pth,1.,2.)
        p=cal.predict_proba(raw(parts['validation']).reshape(-1,1))[:,1]
        pred=parts['validation'][['station','huc','decision_time','horizon','target','event_time']].copy()
        pred['probability']=p
        pred.to_csv(UP/(name+'-predictions.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
        joblib.dump(dict(model=model,calibrator=cal,features=features,threshold=cut),UP/(name+'.joblib'),compress=3)
        m=metrics(pred,p,cut);results.append(dict(name=name,threshold=cut,metrics=m))
        print(name,m['events'],m['detected_events'],round(m['ap'],4),round(m['false_alerts_per_100'],3),flush=True)
    write_json(destination,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['collect','fit']);args=p.parse_args()
    collect() if args.command=='collect' else fit()
