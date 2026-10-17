"""County-scoped forecasts with the exact evaluated extraction and probability contract."""
import json
from datetime import timedelta
from functools import lru_cache
import joblib
import pandas as pd
import shap
from scripts.common import ROOT
from scripts.build_training import VARIABLES, daily_features
from backend.model import calibrated
from backend.providers import fetch
from backend.domain import now_utc


@lru_cache(maxsize=1)
def artifact():
    path = ROOT / 'models/hazards.joblib'
    return joblib.load(path) if path.exists() else None


@lru_cache(maxsize=1)
def tested():
    """Final-test precision and recall of each served model at its served threshold."""
    path = ROOT / 'models/evaluation.json'
    if not path.exists():
        return {}
    hazards = json.loads(path.read_text())['hazards']
    return {h: dict(precision=i['candidates'][i['selected']]['test']['precision'], recall=i['candidates'][i['selected']]['test']['recall'],
                    period='January to June 2026') for h, i in hazards.items()}


def matching_region(bundle, site):
    return next((r for r in bundle['supported_regions']
                 if r['state'] == site['state'] and r['county'].casefold() == site['county'].casefold()), None)


def explain(bundle, x):
    features=bundle['features']
    def probability(values):
        return calibrated(bundle,pd.DataFrame(values,columns=features))
    explainer=shap.PermutationExplainer(probability,bundle['background'],seed=42)
    result=explainer(x[features],max_evals=(2*len(features)+1)*4,silent=True)
    return dict(explanation_base=float(result.base_values[0]),explanations=sorted([
        dict(feature=f,value=float(x.iloc[0][f]),probability_delta=float(result.values[0,i]))
        for i,f in enumerate(features)],key=lambda e:abs(e['probability_delta']),reverse=True))


async def predictions(site):
    saved=artifact()
    if saved is None:
        return dict(available=False,reason='No trained artifact. Run scripts.build_training and scripts.train.',hazards={})
    regions={h:matching_region(b,site) for h,b in saved['hazards'].items()}
    region=next((r for r in regions.values() if r),None)
    if region is None:
        return dict(available=False,reason='County outside the fitted population. No nationwide or nearby-point extrapolation is supported; ML contribution is unknown.',hazards={})
    tomorrow=now_utc().date()+timedelta(days=1)
    params=dict(latitude=region['latitude'],longitude=region['longitude'],start_date=str(tomorrow-timedelta(days=3)),end_date=str(tomorrow),
                hourly=','.join(v+'_previous_day2' for v in VARIABLES),models='gfs_seamless',timezone='GMT')
    result=await fetch(f"model-weather:{region['state_fips']*1000+region['county_fips']}:{tomorrow}",
                       'https://previous-runs-api.open-meteo.com/v1/forecast',params,ttl=21600)
    if not result['available']:
        return dict(available=False,reason='Archived nominal-lead forecast unavailable: '+str(result['error']),hazards={})
    try:
        rows=daily_features(result['data'],region['latitude'],region['longitude'])
        target=next((r for r in rows if r['target_date']==str(tomorrow)),None)
        if target is None:raise ValueError('Missing complete target or antecedent hours')
        x=pd.DataFrame([target]);hazards={}
        for hazard,bundle in saved['hazards'].items():
            if regions[hazard] is None:continue
            probability=float(calibrated(bundle,x)[0])
            hazards[hazard]=dict(probability=probability,model=bundle['name'],threshold=bundle['threshold'],
                warning_supported=bundle['warning_supported'],elevated=bundle['warning_supported'] and probability>=bundle['threshold'],
                in_domain=True,tested=tested().get(hazard),**explain(bundle,x))
    except (KeyError,ValueError,TypeError) as exc:
        return dict(available=False,reason='Forecast contract rejected; prediction withheld: '+str(exc),hazards={})
    return dict(available=True,hazards=hazards,target_date=str(tomorrow),target_timezone='UTC',target_window=f'{tomorrow} 00:00–24:00 UTC',
        issue_time=f'{tomorrow-timedelta(days=1)}T00:00:00Z',forecast_lead='nominal 48h per target hour; actual publication time unverified',
        resolution='Reported county-day event screening, never a verified site incident',county=region['county']+', '+region['state'],
        forecast_point=dict(latitude=region['latitude'],longitude=region['longitude']),trained_at=saved['trained_at'],
        explanation_method='Permutation SHAP of calibrated probability; baseline + all signed contributions = prediction. Independent masking of 16 training rows; approximate, correlated inputs, not causal.',
        feed={k:v for k,v in result.items() if k!='data'})
