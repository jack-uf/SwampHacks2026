"""Extract public planned generator construction by owner entity from EIA-860M (August 2026)."""
import json
import pandas as pd
from scripts.common import ROOT, download

URL = 'https://www.eia.gov/electricity/data/eia860m/xls/august_generator2026.xlsx'


def plants(frame):
    frame = frame.dropna(subset=['Plant ID', 'Entity ID', 'Latitude', 'Longitude'])
    frame = frame[frame['Latitude'].between(18, 72) & frame['Longitude'].between(-180, -60)]
    rows = []
    for (plant_id, entity_id), group in frame.groupby(['Plant ID', 'Entity ID']):
        r = group.iloc[0]
        dated = group.dropna(subset=['Planned Operation Year', 'Planned Operation Month'])
        months = sorted(f"{int(y):04d}-{int(m):02d}" for y, m in zip(dated['Planned Operation Year'], dated['Planned Operation Month']))
        rows.append(dict(plant_id=str(int(plant_id)), plant_name=str(r['Plant Name']).strip(), entity_id=str(int(entity_id)),
                         entity_name=str(r['Entity Name']).strip(), sector=str(r['Sector']).strip(), state=str(r['Plant State']).strip(),
                         county=str(r['County']).strip(), balancing_authority=None if pd.isna(r['Balancing Authority Code']) else str(r['Balancing Authority Code']).strip(),
                         latitude=round(float(r['Latitude']), 5), longitude=round(float(r['Longitude']), 5),
                         capacity_mw=round(float(group['Nameplate Capacity (MW)'].fillna(0).sum()), 1), generators=len(group),
                         planned_month=months[0] if months else None,
                         technologies=sorted(group['Technology'].dropna().astype(str).unique().tolist()),
                         statuses=sorted(group['Status'].dropna().astype(str).unique().tolist())))
    return rows


def main():
    path = download(URL, 'eia860m-august-2026.xlsx')
    projects = plants(pd.read_excel(path, sheet_name='Planned', header=2))
    assert len({p['entity_id'] for p in projects}) >= 2
    source = json.loads(path.with_suffix('.xlsx.source.json').read_text())
    (ROOT/'data/planned-projects.json').write_text(json.dumps(dict(
        source=source, inventory='EIA-860M Inventory of Planned Generators as of August 2026',
        note='Owner-reported planned generators. Not transmission or distribution work. Dates and status change monthly; '
             'a listed plan is not a confirmed construction schedule.',
        projects=projects), separators=(',', ':')))
    print(f"Wrote {len(projects)} planned plant records from {len({p['entity_id'] for p in projects})} entities")


if __name__ == '__main__':
    main()
