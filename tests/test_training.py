import json
from pathlib import Path
import joblib
import pandas as pd
from scripts.common import ROOT
from scripts.build_training import FEATURES
from backend.predict import calibrated


def test_real_dataset_contract_and_saved_metrics():
    df=pd.read_csv(ROOT/'data/training.csv')
    assert len(df)>=1000 and not df[FEATURES].isna().any().any()
    assert not df.duplicated(['region','target_date']).any()
    assert ((pd.to_datetime(df.target_date)-pd.to_datetime(df.issue_date)).dt.days==1).all()
    evaluation=json.loads((ROOT/'models/evaluation.json').read_text())
    p=evaluation['partitions']
    assert p['train']['end']<p['calibration']['start']<=p['calibration']['end']<p['validation']['start']<=p['validation']['end']<p['test']['start']
    saved=joblib.load(ROOT/'models/hazards.joblib')
    for h,info in evaluation['hazards'].items():
        winner=max(info['candidates'],key=lambda n:(info['candidates'][n]['validation']['pr_auc'] or 0,-info['candidates'][n]['validation']['brier']))
        assert info['selected']==winner==saved['hazards'][h]['name']
        result=calibrated(saved['hazards'][h],df[FEATURES].iloc[:10])
        assert ((result>=0)&(result<=1)).all()
        m=info['candidates'][winner]['test'];tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
        assert tn+fp+fn+tp==m['n'] and tp+fn==m['positives']
        assert abs(m['false_alerts_per_100_county_days']-fp/m['n']*100)<1e-8
