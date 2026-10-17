"""Serve only the saved, evaluated feature contract; no synthetic model substitute."""
from datetime import timedelta
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from scripts.common import ROOT
from scripts.build_training import FEATURES, REGIONS
from backend.providers import fetch
from backend.domain import now_utc, distance_miles


@lru_cache(maxsize=1)
def artifact():
    path = ROOT / 'models/hazards.joblib'
    return joblib.load(path) if path.exists() else None


def calibrated(bundle, x):
    p = np.clip(bundle['model'].predict_proba(x)[:,1], 1e-6, 1-1e-6)
    return bundle['calibrator'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]


async def predictions(site):
    saved = artifact()
    if saved is None:
        return dict(available=False, reason='No trained artifact. Run scripts.build_training and scripts.train.', hazards={})
    tomorrow = now_utc().date()+timedelta(days=1)
    params = dict(latitude=site['latitude'], longitude=site['longitude'], start_date=str(tomorrow), end_date=str(tomorrow),
        hourly=','.join(v+'_previous_day2' for v in ['precipitation','temperature_2m','relative_humidity_2m','wind_gusts_10m']),
        models='gfs_seamless', timezone='GMT')
    result = await fetch('model-weather:'+site['id']+':'+str(tomorrow),
                         'https://previous-runs-api.open-meteo.com/v1/forecast', params, ttl=21600)
    if not result['available']:
        return dict(available=False, reason='Archived issue-time forecast unavailable: '+str(result['error']), hazards={})
    h = result['data']['hourly']
    values = [h.get(v+'_previous_day2', []) for v in ['precipitation','temperature_2m','relative_humidity_2m','wind_gusts_10m']]
    if any(len(v)!=24 or any(x is None for x in v) for v in values):
        return dict(available=False, reason='Incomplete 48-hour-lead archived forecast; prediction withheld.', hazards={})
    doy = tomorrow.timetuple().tm_yday
    features = dict(forecast_rain_mm=sum(values[0]), forecast_temp_c=max(values[1]),
        forecast_humidity_pct=float(np.mean(values[2])), forecast_gust_kmh=max(values[3]), latitude=site['latitude'],
        longitude=site['longitude'], season_sin=np.sin(2*np.pi*doy/365.25), season_cos=np.cos(2*np.pi*doy/365.25))
    x = pd.DataFrame([features], columns=FEATURES)
    nearest = min(distance_miles(site['latitude'],site['longitude'],r[3],r[4])*1.609344 for r in REGIONS if r[0]!='Los Angeles, CA')
    hazards = {}
    for hazard,b in saved['hazards'].items():
        probability = float(calibrated(b,x)[0])
        explanations = []
        for feature in FEATURES:
            changed = x.copy()
            changed.loc[0, feature] = b['reference'][feature]
            delta = probability-float(calibrated(b,changed)[0])
            explanations.append(dict(feature=feature, value=float(features[feature]), probability_delta=round(delta,5)))
        hazards[hazard] = dict(probability=probability, model=b['name'], threshold=b['threshold'],
            elevated=probability >= b['threshold'], in_domain=nearest <= 150,
            explanations=sorted(explanations,key=lambda e: abs(e['probability_delta']),reverse=True))
    return dict(available=True, hazards=hazards, target_date=str(tomorrow), target_timezone='UTC',
        target_window=f'{tomorrow} 00:00–24:00 UTC', issue_time=f'{tomorrow-timedelta(days=1)}T00:00:00Z',
        forecast_lead='at least 48h per target hour', resolution='county-day event screening, not a verified site incident',
        nearest_training_km=round(nearest), extrapolation=nearest>150, trained_at=saved['trained_at'],
        explanation_method='One feature replaced by its training median. Signed probability difference; not additive or causal.',
        feed={k:v for k,v in result.items() if k!='data'})
