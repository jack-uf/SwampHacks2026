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


def main():
    listing = download('https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/', 'noaa-index.html').read_text()
    events = []
    sources = []
    for year in [2024, 2025]:
        names = sorted(set(re.findall(rf'StormEvents_details-ftp_v1.0_d{year}_c\d+\.csv.gz', listing)))
        if not names:
            raise RuntimeError(f'NOAA archive missing {year}')
        name = names[-1]
        path = download('https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/' + name, name)
        source = json.loads(path.with_suffix(path.suffix + '.source.json').read_text())
        sources.append(source)
        frame = pd.read_csv(path, low_memory=False)
        events.append(frame[(frame.CZ_TYPE == 'C') & frame.EVENT_TYPE.isin(
            ['Flood', 'Flash Flood', 'Thunderstorm Wind', 'Tornado', 'Hail'])].copy())
    events = pd.concat(events, ignore_index=True)
    selected_keys = {(r[1], r[2]) for r in REGIONS}
    events = events[[key in selected_keys for key in zip(events.STATE_FIPS, events.CZ_FIPS)]].copy()
    # NOAA uses standard-time offsets (EST-5, CST-6, etc.), not civil DST.
    offsets = events.CZ_TIMEZONE.str.extract(r'(-\d+)$')[0]
    if offsets.isna().any():
        raise ValueError('Unknown NOAA event timezone; normalize explicitly')
    events['start_utc'] = pd.to_datetime(events.BEGIN_DATE_TIME, format='%d-%b-%y %H:%M:%S') - pd.to_timedelta(offsets.astype(int), unit='h')
    events['end_utc'] = pd.to_datetime(events.END_DATE_TIME, format='%d-%b-%y %H:%M:%S') - pd.to_timedelta(offsets.astype(int), unit='h')
    observations = []
    coverage = []
    for name, state, county, lat, lon, _ in REGIONS:
        params = dict(latitude=lat, longitude=lon, start_date='2024-01-01', end_date='2025-12-31',
                      hourly=','.join(v + '_previous_day2' for v in VARIABLES), models='gfs_seamless', timezone='GMT')
        path = download('https://previous-runs-api.open-meteo.com/v1/forecast', f'forecast-{state}-{county}.json', params)
        sources.append(json.loads(path.with_suffix('.json.source.json').read_text()))
        payload = json.loads(path.read_text())
        hourly = pd.DataFrame(payload['hourly'])
        hourly['time'] = pd.to_datetime(hourly.time)
        hourly['date'] = hourly.time.dt.floor('D')
        regional = events[(events.STATE_FIPS == state) & (events.CZ_FIPS == county)]
        used = 0
        for day, group in hourly.groupby('date'):
            if len(group) != 24 or group[[v + '_previous_day2' for v in VARIABLES]].isna().any().any():
                continue
            overlap = regional[(regional.start_utc < day + pd.Timedelta(days=1)) & (regional.end_utc >= day)]
            doy = day.dayofyear
            observations.append(dict(region=name, target_date=str(day.date()), issue_date=str((day-pd.Timedelta(days=1)).date()),
                forecast_rain_mm=float(group.precipitation_previous_day2.sum()),
                forecast_temp_c=float(group.temperature_2m_previous_day2.max()),
                forecast_humidity_pct=float(group.relative_humidity_2m_previous_day2.mean()),
                forecast_gust_kmh=float(group.wind_gusts_10m_previous_day2.max()),
                latitude=lat, longitude=lon, season_sin=float(np.sin(2*np.pi*doy/365.25)),
                season_cos=float(np.cos(2*np.pi*doy/365.25)),
                flood=int(overlap.EVENT_TYPE.isin(['Flood', 'Flash Flood']).any()),
                storm=int(overlap.EVENT_TYPE.isin(['Thunderstorm Wind', 'Tornado', 'Hail']).any())))
            used += 1
        coverage.append({'region': name, 'complete_days': used})
        print(f'{name}: {used} complete county-days', flush=True)
    result = pd.DataFrame(observations)
    if len(result) < 1000:
        raise ValueError('Insufficient complete archive rows; inspect provider coverage')
    result.to_csv(ROOT / 'data/training.csv', index=False)
    (ROOT / 'data/training-provenance.json').write_text(json.dumps(dict(
        sources=sources, coverage=coverage, features=FEATURES, label_resolution='county-day UTC',
        forecast_lead_hours=48, created_at=datetime.now(timezone.utc).isoformat(),
        target='00:00–24:00 UTC on the day after issue date; a 24h window starting 24h after issue'), indent=2))
    print(result.groupby(result.target_date.str[:4])[['flood', 'storm']].agg(['size', 'sum']))


if __name__ == '__main__':
    main()
