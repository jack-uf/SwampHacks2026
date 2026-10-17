"""Extract NOAA Storm Events (2024 to 2026) near each facility for Site Hazard Memory."""
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scripts.common import ROOT, RAW

FILES = ['StormEvents_details-ftp_v1.0_d2024_c20260728.csv.gz', 'StormEvents_details-ftp_v1.0_d2025_c20260819.csv.gz',
         'StormEvents_details-ftp_v1.0_d2026_c20260918.csv.gz']
HAZARDS = {'Flash Flood': 'flood', 'Flood': 'flood', 'Heavy Rain': 'flood', 'Coastal Flood': 'flood', 'Storm Surge/Tide': 'flood',
           'Thunderstorm Wind': 'storm', 'High Wind': 'storm', 'Strong Wind': 'storm', 'Tornado': 'storm', 'Hail': 'storm',
           'Tropical Storm': 'storm', 'Hurricane (Typhoon)': 'storm', 'Dust Storm': 'storm', 'Lightning': 'lightning',
           'Heat': 'heat', 'Excessive Heat': 'heat', 'Wildfire': 'wildfire', 'Debris Flow': 'unstable_terrain'}
STATES = dict(AL='ALABAMA', AZ='ARIZONA', AR='ARKANSAS', CA='CALIFORNIA', CO='COLORADO', CT='CONNECTICUT', DE='DELAWARE', FL='FLORIDA',
              GA='GEORGIA', ID='IDAHO', IL='ILLINOIS', IN='INDIANA', IA='IOWA', KS='KANSAS', KY='KENTUCKY', LA='LOUISIANA', ME='MAINE',
              MD='MARYLAND', MA='MASSACHUSETTS', MI='MICHIGAN', MN='MINNESOTA', MS='MISSISSIPPI', MO='MISSOURI', MT='MONTANA',
              NE='NEBRASKA', NV='NEVADA', NH='NEW HAMPSHIRE', NJ='NEW JERSEY', NM='NEW MEXICO', NY='NEW YORK', NC='NORTH CAROLINA',
              ND='NORTH DAKOTA', OH='OHIO', OK='OKLAHOMA', OR='OREGON', PA='PENNSYLVANIA', RI='RHODE ISLAND', SC='SOUTH CAROLINA',
              SD='SOUTH DAKOTA', TN='TENNESSEE', TX='TEXAS', UT='UTAH', VT='VERMONT', VA='VIRGINIA', WA='WASHINGTON',
              WV='WEST VIRGINIA', WI='WISCONSIN', WY='WYOMING')
RADIUS_MILES = 20


def number(value):
    return None if pd.isna(value) else int(value)


def record(row, geography):
    begin = datetime.strptime(row['BEGIN_DATE_TIME'], '%d-%b-%y %H:%M:%S').isoformat()
    narrative = row['EVENT_NARRATIVE'] if isinstance(row['EVENT_NARRATIVE'], str) else row['EPISODE_NARRATIVE'] if isinstance(row['EPISODE_NARRATIVE'], str) else ''
    point = geography == 'point'
    return dict(id='noaa:'+str(int(row['EVENT_ID'])), source='NOAA Storm Events', event_type=row['EVENT_TYPE'], hazard=HAZARDS[row['EVENT_TYPE']],
                begin=begin, timezone=row['CZ_TIMEZONE'], latitude=round(float(row['BEGIN_LAT']), 4) if point else None,
                longitude=round(float(row['BEGIN_LON']), 4) if point else None, geography=geography, area=str(row['CZ_NAME']).title(),
                state=str(row['STATE']).title(), narrative=narrative[:300], injuries=number(row['INJURIES_DIRECT']) or 0,
                deaths=number(row['DEATHS_DIRECT']) or 0, damage_property=None if pd.isna(row['DAMAGE_PROPERTY']) else str(row['DAMAGE_PROPERTY']),
                magnitude=None if pd.isna(row['MAGNITUDE']) else float(row['MAGNITUDE']), magnitude_type=None if pd.isna(row['MAGNITUDE_TYPE']) else row['MAGNITUDE_TYPE'],
                episode_id=int(row['EPISODE_ID']), source_file=row['_file'])


def main():
    frames = []
    for name in FILES:
        frame = pd.read_csv(RAW/name, low_memory=False)
        frame['_file'] = name
        frames.append(frame)
    events = pd.concat(frames)
    events = events[events['EVENT_TYPE'].isin(HAZARDS)]
    sites = json.loads((ROOT/'data/sites.json').read_text())['sites']
    lat, lon = np.radians(events['BEGIN_LAT'].to_numpy(float)), np.radians(events['BEGIN_LON'].to_numpy(float))
    chosen, by_site = {}, {}
    for site in sites:
        a, b = np.radians(site['latitude']), np.radians(site['longitude'])
        miles = 3958.7613*2*np.arcsin(np.sqrt(np.sin((lat-a)/2)**2+np.cos(a)*np.cos(lat)*np.sin((lon-b)/2)**2))
        near = np.nan_to_num(miles, nan=1e9) <= RADIUS_MILES
        unlocated = np.isnan(lat)
        named = (events['STATE'] == STATES[site['state']]).to_numpy() & (events['CZ_NAME'].str.casefold() == site['county'].casefold()).to_numpy()
        ids = []
        for mask, geography in [(near, 'point'), (named & unlocated & (events['CZ_TYPE'] == 'C').to_numpy(), 'county'),
                                (named & unlocated & (events['CZ_TYPE'] == 'Z').to_numpy(), 'zone')]:
            for _, row in events[mask].iterrows():
                r = record(row, geography)
                chosen[r['id']] = r
                ids.append(r['id'])
        by_site[site['id']] = ids
    sources = {name: json.loads((RAW/(name+'.source.json')).read_text()) for name in FILES}
    (ROOT/'data/hazard-memory.json').write_text(json.dumps(dict(
        sources=sources, radius_miles=RADIUS_MILES,
        note='NOAA Storm Events are official reports of significant weather, compiled by NWS offices; not crew incidents at these '
             'facilities. Point records use the event begin location. County and zone records have no point location.',
        events=chosen, by_site=by_site), separators=(',', ':')))
    print(f'Wrote {len(chosen)} NOAA events across {sum(bool(v) for v in by_site.values())} facilities')


if __name__ == '__main__':
    main()
