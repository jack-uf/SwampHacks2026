import asyncio
import copy
import hashlib
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.common import ROOT
from scripts.build_training import FEATURES, VARIABLES, daily_features, normalize_events
from scripts.train import partitions, threshold_policies
from backend import predict
from backend.model import calibrated


def forecast(start='2026-01-01', hours=96):
    times=pd.date_range(start,periods=hours,freq='h')
    return {'hourly':{'time':times.strftime('%Y-%m-%dT%H:%M').tolist(),**{
        v+'_previous_day2':[float(i+1)]*hours for i,v in enumerate(VARIABLES)}}}


def test_hourly_alignment_antecedent_and_no_imputation():
    data=forecast(); rows=daily_features(data,30,-80)
    assert len(rows)==1 and rows[0]['target_date']=='2026-01-04'
    assert rows[0]['issue_date']=='2026-01-03'
    assert rows[0]['forecast_rain_mm']==24 and rows[0]['antecedent_forecast_rain_72h_mm']==72
    assert rows[0]['forecast_wet_hours']==24
    bad=copy.deepcopy(data);bad['hourly']['precipitation_previous_day2'][0]=None
    assert daily_features(bad,30,-80)==[]
    bad=copy.deepcopy(data);bad['hourly']['temperature_2m_previous_day2'][75]=None
    assert daily_features(bad,30,-80)==[]
    bad=copy.deepcopy(data);bad['hourly']['time'][-1]=bad['hourly']['time'][-2]
    with pytest.raises(ValueError,match='unique'):daily_features(bad,30,-80)
    bad=forecast(hours=95)
    assert daily_features(bad,30,-80)==[]


def test_noaa_county_namespace_standard_timezone_and_zero_duration():
    frame=pd.DataFrame([dict(EVENT_TYPE='Flash Flood',CZ_TYPE='C',CZ_TIMEZONE='EST-5',BEGIN_DATE_TIME='01-Jul-25 20:00:00',END_DATE_TIME='01-Jul-25 20:00:00')])
    result=normalize_events(frame)
    assert result.iloc[0].start_utc==pd.Timestamp('2025-07-02 01:00')==result.iloc[0].end_utc
    with pytest.raises(ValueError,match='geography'):normalize_events(frame.assign(CZ_TYPE='Z'))
    with pytest.raises(ValueError,match='timezone'):normalize_events(frame.assign(CZ_TIMEZONE='unknown'))
    with pytest.raises(ValueError,match='before'):normalize_events(frame.assign(END_DATE_TIME='01-Jul-25 19:00:00'))


def test_real_dataset_partitions_artifact_and_validation_selection():
    df=pd.read_csv(ROOT/'data/training.csv');parts=partitions(df)
    assert len(df)==21336 and df.region.nunique()==24 and not df[FEATURES].isna().any().any()
    assert not df.duplicated(['county_fips','target_date']).any()
    assert ((pd.to_datetime(df.target_date)-pd.to_datetime(df.issue_date)).dt.days==1).all()
    assert parts['train'].target_date.max()<parts['calibration'].target_date.min()<=parts['calibration'].target_date.max()<parts['validation'].target_date.min()<=parts['validation'].target_date.max()<parts['test'].target_date.min()
    used=set(parts['train'].county_fips)|set(parts['calibration'].county_fips)|set(parts['validation'].county_fips)
    assert not used & set(parts['geographic_test'].county_fips)
    assert not used & set(parts['geographic_validation'].county_fips)
    assert not set(parts['geographic_test'].state)&set(parts['geographic_validation'].state)
    evaluation=json.loads((ROOT/'models/evaluation.json').read_text())
    assert evaluation['dataset_sha256']==hashlib.sha256((ROOT/'data/training.csv').read_bytes()).hexdigest()
    saved=joblib.load(ROOT/'models/hazards.joblib')
    selection=json.loads((ROOT/'models/selection.json').read_text())
    for h,info in evaluation['hazards'].items():
        winner=max(info['candidates'],key=lambda n:(info['candidates'][n]['selection_score'],-info['candidates'][n]['validation']['brier']))
        assert info['selected']==winner==saved['hazards'][h]['name']==selection[h]['selected']
        assert selection[h]['threshold']==saved['hazards'][h]['threshold']
        assert len(info['candidates'])==8
        result=calibrated(saved['hazards'][h],df.iloc[:10])
        assert ((result>=0)&(result<=1)).all()
        m=info['candidates'][winner]['test'];tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
        assert tn+fp+fn+tp==m['n'] and tp+fn==m['positives']
        assert abs(m['false_alerts_per_100_county_days']-fp/m['n']*100)<1e-8
        assert sum(b['n'] for b in m['calibration'])==m['n']
        for region in saved['hazards'][h]['supported_regions']:
            assert region['role']=='development' or (winner=='frozen_baseline' and region['state']=='AZ')


def test_threshold_budget_is_explicitly_unavailable_if_infeasible():
    t,policies,supported=threshold_policies([0,0,0,1],np.array([.1,.1,.1,.1]))
    assert not supported and policies['budget'] is None
    t,policies,supported=threshold_policies([0]*9+[1],np.array([.1]*9+[.9]))
    assert supported and policies['budget']['alert_rate']<=.2 and t==.9


def test_shap_is_additive_for_calibrated_probability():
    saved=joblib.load(ROOT/'models/hazards.joblib');x=pd.read_csv(ROOT/'data/training.csv').iloc[[100]]
    for b in saved['hazards'].values():
        result=predict.explain(b,x)
        assert result['explanation_base']+sum(e['probability_delta'] for e in result['explanations'])==pytest.approx(calibrated(b,x)[0],abs=1e-7)
        assert len(result['explanations'])==len(b['features'])


def test_inference_restricts_county_and_shares_representative_features(monkeypatch):
    saved=joblib.load(ROOT/'models/hazards.joblib');monkeypatch.setattr(predict,'artifact',lambda:saved)
    calls=[]
    async def fetch(key,url,params,**kwargs):
        calls.append(params);return dict(available=True,data=forecast(),fetched_at='2026-01-03T00:00:00Z')
    monkeypatch.setattr(predict,'fetch',fetch)
    monkeypatch.setattr(predict,'now_utc',lambda:datetime(2026,1,3,tzinfo=timezone.utc))
    outside=asyncio.run(predict.predictions(dict(state='FL',county='Unknown',latitude=29.65,longitude=-82.32)))
    assert not outside['available'] and outside['hazards']=={} and not calls
    site=dict(state='FL',county='Alachua',latitude=29.5,longitude=-82.1)
    result=asyncio.run(predict.predictions(site))
    assert result['available'] and result['target_date']=='2026-01-04'
    assert calls[0]['latitude']==29.65 and calls[0]['longitude']==-82.32
    assert calls[0]['start_date']=='2026-01-01'
    expected=daily_features(forecast(),29.65,-82.32)[0]
    for h,p in result['hazards'].items():
        assert p['probability']==pytest.approx(calibrated(saved['hazards'][h],pd.DataFrame([expected]))[0])
