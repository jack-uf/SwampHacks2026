"""Extract 60 real, large operating EIA facilities across the continental US."""
import json
from datetime import datetime, timezone
import pandas as pd
from timezonefinder import TimezoneFinder
from scripts.common import ROOT, download

URL = 'https://www.eia.gov/electricity/data/eia860m/archive/xls/december_generator2024.xlsx'


def main():
    path = download(URL, 'eia860m-december-2024.xlsx')
    frame = pd.read_excel(path, sheet_name='Operating', header=2)
    frame = frame.dropna(subset=['Plant ID', 'Plant Name', 'Latitude', 'Longitude'])
    frame = frame[frame['Latitude'].between(24, 50) & frame['Longitude'].between(-125, -66)]
    rows = []
    for plant_id, group in frame.groupby('Plant ID'):
        r = group.iloc[0]
        rows.append(dict(id=str(int(plant_id)), name=str(r['Plant Name']).strip(), state=r['Plant State'],
                         county=str(r['County']).strip(), latitude=float(r['Latitude']), longitude=float(r['Longitude']),
                         capacity_mw=round(float(group['Nameplate Capacity (MW)'].sum()), 1),
                         technology=str(group.sort_values('Nameplate Capacity (MW)', ascending=False).iloc[0]['Technology']),
                         operator=str(r['Entity Name']), source='EIA-860M December 2024', inventory_date='2024-12-31'))
    # Two per state gives broad geographic coverage without manually invented coordinates.
    selected = pd.DataFrame(rows).sort_values('capacity_mw', ascending=False).groupby('state').head(2)
    selected = selected.sort_values('capacity_mw', ascending=False).head(60)
    finder = TimezoneFinder(in_memory=True)
    sites = selected.to_dict('records')
    for site in sites:
        site['timezone'] = finder.timezone_at(lat=site['latitude'], lng=site['longitude'])
        if not site['timezone']:
            raise ValueError(f"No timezone for EIA facility {site['id']}")
    assert len(sites) >= 50 and len({s['id'] for s in sites}) == len(sites)
    source = json.loads(path.with_suffix('.xlsx.source.json').read_text())
    (ROOT / 'data/sites.json').write_text(json.dumps({'source': source, 'inventory_note':
        'Published inventory coordinates; not a field survey or verified current operating status.', 'sites': sites}, indent=2))
    print(f'Wrote {len(sites)} authentic facilities across {len(selected.state.unique())} states')


if __name__ == '__main__':
    main()
