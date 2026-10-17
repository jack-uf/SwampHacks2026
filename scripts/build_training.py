"""NOAA county-day outcomes + archived GFS forecasts, never realized target weather.

Issue time is 00:00 UTC on the day before the target day. A target hour's
previous_day2 forecast has at least 48 h lead: its run precedes issue time.
All 24 target hours must be populated. County labels are NOT site incidents.
"""
import json
import re
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from scripts.common import ROOT, download

# Explicit regional sample: representative points, FIPS county and NOAA standard timezone.
REGIONS = [
    ('Alachua, FL', 12, 1, 29.65, -82.32, -5),
    ('Hillsborough, FL', 12, 57, 27.99, -82.35, -5),
    ('Miami-Dade, FL', 12, 86, 25.61, -80.50, -5),
    ('Harris, TX', 48, 201, 29.86, -95.39, -6),
    ('Dallas, TX', 48, 113, 32.77, -96.78, -6),
    ('Travis, TX', 48, 453, 30.33, -97.78, -6),
    ('Oklahoma, OK', 40, 109, 35.55, -97.41, -6),
    ('Tulsa, OK', 40, 143, 36.12, -95.94, -6),
    ('Shelby, TN', 47, 157, 35.18, -89.90, -6),
    ('Davidson, TN', 47, 37, 36.17, -86.78, -6),
    ('Maricopa, AZ', 4, 13, 33.35, -112.49, -7),
    ('Los Angeles, CA', 6, 37, 34.18, -118.26, -8),
]
VARIABLES = ['precipitation', 'temperature_2m', 'relative_humidity_2m', 'wind_gusts_10m']
FEATURES = ['forecast_rain_mm', 'forecast_temp_c', 'forecast_humidity_pct', 'forecast_gust_kmh',
            'latitude', 'longitude', 'season_sin', 'season_cos']


BASE_FEATURES = FEATURES.copy()
FEATURES += ['forecast_rain_max_hour_mm', 'forecast_wet_hours', 'antecedent_forecast_rain_72h_mm',
             'rain_humidity', 'gust_temperature']
ADDITIONAL = [53025, 49015, 8001, 13033, 27053, 17197, 42063, 37019, 45091, 22089, 1071, 39081]
GEO_TEST_STATES = {'CA', 'AZ', 'WA', 'UT'}
GEO_VALIDATION_STATES = {'GA', 'CO'}
TARGET_TYPES = ['Flood', 'Flash Flood', 'Thunderstorm Wind', 'Tornado', 'Hail']


def regions():
    import zipfile
    path = download('https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip',
                    '2024_Gaz_counties_national.zip')
    with zipfile.ZipFile(path) as archive:
        gazetteer = pd.read_csv(archive.open(archive.namelist()[0]), sep='\t')
    gazetteer.columns = gazetteer.columns.str.strip()
    original = {state*1000+county: (name, lat, lon) for name,state,county,lat,lon,_ in REGIONS}
    result = []
    for fips in [*original, *ADDITIONAL]:
        row = gazetteer[gazetteer.GEOID == fips].iloc[0]
        name,lat,lon = original.get(fips, (row.NAME+', '+row.USPS, float(row.INTPTLAT),float(row.INTPTLONG)))
        role = 'geographic_test' if row.USPS in GEO_TEST_STATES else 'geographic_validation' if row.USPS in GEO_VALIDATION_STATES else 'development'
        result.append(dict(region=name,state=row.USPS,county=row.NAME.removesuffix(' County').removesuffix(' Parish'),
                           state_fips=fips//1000,county_fips=fips%1000,latitude=lat,longitude=lon,role=role))
    return result, path


def normalize_events(frame):
    target = frame[frame.EVENT_TYPE.isin(TARGET_TYPES)].copy()
    if not target.CZ_TYPE.eq('C').all():
        raise ValueError('Target event recorded outside county geography; explicit mapping/redefinition required')
    offsets = target.CZ_TIMEZONE.str.extract(r'([+-]\d+)$')[0]
    if offsets.isna().any():
        raise ValueError('Unknown NOAA standard timezone; normalize explicitly')
    for field,out in [('BEGIN_DATE_TIME','start_utc'),('END_DATE_TIME','end_utc')]:
        target[out] = pd.to_datetime(target[field],format='%d-%b-%y %H:%M:%S') - pd.to_timedelta(offsets.astype(int),unit='h')
    if (target.end_utc < target.start_utc).any():
        raise ValueError('NOAA event ends before it starts')
    return target


def daily_features(payload, latitude, longitude):
    hourly = pd.DataFrame(payload['hourly'])
    hourly['time'] = pd.to_datetime(hourly.time)
    if hourly.time.duplicated().any() or not hourly.time.is_monotonic_increasing:
        raise ValueError('Forecast timestamps must be unique and ordered')
    columns = [v+'_previous_day2' for v in VARIABLES]
    observations = []
    for day,group in hourly.groupby(hourly.time.dt.floor('D')):
        expected = pd.date_range(day,periods=24,freq='h')
        if not group.time.reset_index(drop=True).equals(pd.Series(expected)) or group[columns].isna().any().any():
            continue
        antecedent = hourly[(hourly.time >= day-pd.Timedelta(hours=72)) & (hourly.time < day)]
        if len(antecedent)!=72 or antecedent[columns[0]].isna().any():
            continue
        rain=group[columns[0]];temp=float(group[columns[1]].max());humidity=float(group[columns[2]].mean());gust=float(group[columns[3]].max())
        doy=day.dayofyear
        observations.append(dict(target_date=str(day.date()),issue_date=str((day-pd.Timedelta(days=1)).date()),
            forecast_rain_mm=float(rain.sum()),forecast_temp_c=temp,forecast_humidity_pct=humidity,forecast_gust_kmh=gust,
            latitude=latitude,longitude=longitude,season_sin=float(np.sin(2*np.pi*doy/365.25)),season_cos=float(np.cos(2*np.pi*doy/365.25)),
            forecast_rain_max_hour_mm=float(rain.max()),forecast_wet_hours=int((rain>0).sum()),
            antecedent_forecast_rain_72h_mm=float(antecedent[columns[0]].sum()),rain_humidity=float(rain.sum()*humidity/100),
            gust_temperature=gust*temp/100))
    return observations


def main():
    import time
    selected,gazetteer=regions()
    listing = download('https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/', 'noaa-index-research-20261016.html').read_text()
    sources=[json.loads(gazetteer.with_suffix('.zip.source.json').read_text())];frames=[];audit=[]
    for year in [2024,2025,2026]:
        names=sorted(set(re.findall(rf'StormEvents_details-ftp_v1.0_d{year}_c\d+\.csv.gz',listing)))
        if not names: raise RuntimeError(f'NOAA archive missing {year}')
        path=download('https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'+names[-1],names[-1])
        sources.append(json.loads(path.with_suffix(path.suffix+'.source.json').read_text()))
        frame=pd.read_csv(path,low_memory=False)
        for (event,geography),count in frame.groupby(['EVENT_TYPE','CZ_TYPE']).size().items():
            audit.append(dict(year=year,event_type=event,geography=geography,records=int(count),included=event in TARGET_TYPES and geography=='C'))
        keys = {r['state_fips']*1000+r['county_fips'] for r in selected}
        target = frame[frame.EVENT_TYPE.isin(TARGET_TYPES)]
        if not target.CZ_TYPE.eq('C').all():
            raise ValueError('Target category has non-county records; inspect geography audit')
        frames.append(normalize_events(frame[(frame.STATE_FIPS*1000+frame.CZ_FIPS).isin(keys)]))
    events=pd.concat(frames,ignore_index=True)
    if events.EVENT_ID.duplicated().any():raise ValueError('Duplicate NOAA event ID')
    observations=[];coverage=[]
    for r in selected:
        params=dict(latitude=r['latitude'],longitude=r['longitude'],start_date='2024-01-01',end_date='2026-06-30',
                    hourly=','.join(v+'_previous_day2' for v in VARIABLES),models='gfs_seamless',timezone='GMT')
        name=f"expanded-forecast-{r['state_fips']}-{r['county_fips']}-202406.json"
        uncached=not (ROOT/'data/raw'/name).exists()
        path=download('https://previous-runs-api.open-meteo.com/v1/forecast',name,params)
        sources.append(json.loads(path.with_suffix('.json.source.json').read_text()))
        daily=daily_features(json.loads(path.read_text()),r['latitude'],r['longitude'])
        regional=events[(events.STATE_FIPS==r['state_fips']) & (events.CZ_FIPS==r['county_fips'])]
        for values in daily:
            day=pd.Timestamp(values['target_date'])
            overlap=regional[(regional.start_utc<day+pd.Timedelta(days=1)) & (regional.end_utc>=day)]
            observations.append(dict(**values,region=r['region'],state=r['state'],county_fips=r['state_fips']*1000+r['county_fips'],role=r['role'],
                flood=int(overlap.EVENT_TYPE.isin(['Flood','Flash Flood']).any()),storm=int(overlap.EVENT_TYPE.isin(['Thunderstorm Wind','Tornado','Hail']).any())))
        coverage.append(dict(region=r['region'],complete_days=len(daily),role=r['role']))
        print(r['region'],len(daily),'complete county-days',flush=True)
        if uncached:time.sleep(10)
    result=pd.DataFrame(observations)
    assert len(result)>15000 and not result[FEATURES].isna().any().any()
    assert not result.duplicated(['county_fips','target_date']).any()
    result.to_csv(ROOT/'data/training.csv',index=False)
    (ROOT/'data/regions.json').write_text(json.dumps(selected,indent=2))
    (ROOT/'data/noaa-geography-audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'data/training-provenance.json').write_text(json.dumps(dict(sources=sources,coverage=coverage,features=FEATURES,
        target='NOAA-reported Flood/Flash Flood or Thunderstorm Wind/Tornado/Hail overlapping a UTC county-day; unreported != safe',
        interval_policy='Closed reported intervals; zero-duration events belong to their UTC day; midnight end includes that instant in the next day',
        forecast_contract='GFS Previous Runs day2, fixed 48h nominal lead per hour; not a fixed initialization or verified dissemination log',
        issue_time='00 UTC on day before target; all nominal feature initializations precede this time',
        excluded_features={'soil_moisture':'consistent historical operational availability not established','realized_rain':'future/reanalysis leakage forbidden'},
        label_resolution='county-day UTC',forecast_lead_hours=48,created_at=datetime.now(timezone.utc).isoformat()),indent=2))


if __name__ == '__main__':
    main()
