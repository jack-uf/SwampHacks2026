"""Reserve candidate drainage groups, and screen the unused catchment pool, without outcome access.

The `reserve` command records hand-picked candidates and their metadata exclusions. The `screen`
command walks every unused catchment in the retained CAMELS boundary set and measures positive-area
overlap against every catchment any earlier experiment exposed, which is the independence test the
confirmation protocol asks for. Neither command acquires or inspects an outcome.
"""
import io
import json
import zipfile

import httpx

import numpy as np
import shapefile
from shapely import make_valid
from shapely.geometry import shape
from shapely.strtree import STRtree

from scripts import mlp_geo as archive
from scripts.flood_sprint import write_json
from scripts.high_flow_signals import OUT, fetch
from scripts.hydrology_research import ROOT, digest
from scripts.mlp_onset import exposed_stations

IDS=['02255600','02256500','07235000','08033900','08034500']


def reserve():
    folder=OUT/'fresh-geography';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'reservation.json').exists():raise FileExistsError('Geographic reservation already recorded')
    prior=exposed_stations()
    meta=json.loads((ROOT/'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    prior_huc={s['huc_cd'][:4] for s in meta if s['site_no'] in prior and len(s.get('huc_cd',''))>=8}
    z=zipfile.ZipFile(OUT/'sources/basin_set_full_res.zip')
    base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    r=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),dbf=io.BytesIO(z.read(base+'.dbf')))
    shapes=[make_valid(shape(s.__geo_interface__)) for s in r.iterShapes()]
    tree=STRtree(shapes)
    selected,records=[],[]
    archive.OUT=folder
    # Selection rules precede reference/outcome acquisition. Reference completeness is source eligibility only.
    for sid in IDS:
        url='https://api.waterdata.usgs.gov/ogcapi/v1/collections/monitoring-locations/items/USGS-'+sid
        raw=fetch(folder/(sid+'-location.json'),url,dict(f='json'))
        location=json.loads(raw);p=location['properties'];lon,lat=location['geometry']['coordinates']
        record=dict(station=sid,name=p['monitoring_location_name'],huc=p['hydrologic_unit_code'],
            latitude=lat,longitude=lon,area_km2=float(p['drainage_area'])*2.589988110336,
            elevation_m=float(p['altitude'])*.3048,exclusions=[])
        if sid in prior:record['exclusions'].append('Prior station exposure')
        if record['huc'][:4] in prior_huc:record['exclusions'].append('Previously exposed HUC4')
        if not 20<float(p['drainage_area'])<1200:record['exclusions'].append('Outside fitted drainage-area screen')
        if p['site_type_code']!='ST':record['exclusions'].append('Not a stream site')
        if not np.isfinite([lat,lon,record['elevation_m']]).all():record['exclusions'].append('Invalid static attributes')
        url=f'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}/basin'
        try:
            basin=json.loads(fetch(folder/(sid+'-basin.json'),url,dict(f='json',simplified='false')))
        except httpx.HTTPError as exc:
            record['exclusions'].append('Full-resolution boundary unavailable: '+str(exc))
            records.append(record)
            print('New watershed',sid,'boundary unavailable',flush=True)
            continue
        if len(basin.get('features',[]))!=1:raise ValueError('Ambiguous candidate basin')
        polygon=shape(basin['features'][0]['geometry'])
        if not polygon.is_valid:record['exclusions'].append('Invalid full-resolution basin')
        else:
            overlaps=[int(i) for i in tree.query(polygon,predicate='intersects') if polygon.intersection(shapes[i]).area>0]
            record['camels_polygon_overlaps']=len(overlaps)
            if overlaps:record['exclusions'].append('Positive-area overlap with the full CAMELS population')
        # Do not acquire reference discharge for a geographically ineligible candidate.
        if not record['exclusions']:
            try:record['reference_cfs']=archive.reference(sid)
            except ValueError as exc:record['exclusions'].append('Reference ineligible: '+str(exc))
        if not record['exclusions']:
            if any(polygon.intersection(g).area>0 for g in selected):
                record['exclusions'].append('Nested with an earlier reserved candidate')
            else:selected.append(polygon)
        records.append(record)
        print('New watershed',sid,record['exclusions'] or 'reserved, reference eligible',flush=True)
    write_json(folder/'reservation.json',dict(candidates=records,prior_station_ids=sorted(prior),prior_huc4=sorted(prior_huc),
        outcomes_acquired=False,code_sha256=digest(ROOT/'scripts/reserve_high_flow_geography.py'),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in folder.rglob('*.json') if not p.name.endswith('.request.json')},
        boundary_source_sha256=digest(OUT/'sources/basin_set_full_res.zip'),
        boundary_kind='Full-resolution upstream catchment accumulation; no exact gauge split operation',
        limitation='Zero overlap with full CAMELS polygons does not audit every prior non-CAMELS basin or shared storms. Final independence requires that remaining audit.'))


def basins():
    """Retained full-resolution CAMELS catchments keyed by eight-digit station identifier."""
    archive_path=OUT/'sources/basin_set_full_res.zip'
    z=zipfile.ZipFile(archive_path)
    base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    reader=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),
        dbf=io.BytesIO(z.read(base+'.dbf')))
    fields=[field[0] for field in reader.fields[1:]]
    id_field=next(f for f in fields if f.lower() in ['hru_id','gauge_id'])
    polygons={}
    for record in reader.iterShapeRecords():
        station=str(record.record[fields.index(id_field)]).zfill(8)
        original=shape(record.shape.__geo_interface__)
        normalized=make_valid(original)
        if normalized.geom_type not in ['Polygon','MultiPolygon'] or abs(normalized.area-original.area)>1e-10*original.area:
            raise ValueError('Basin normalization changed area or dimensionality: '+station)
        polygons[station]=(normalized,not original.is_valid)
    return archive_path,polygons


def screen():
    """Positive-area screening of every unused catchment against every previously exposed one."""
    folder=OUT/'fresh-geography';folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'camels-screen.json'
    if destination.exists():raise FileExistsError('Catchment screening already recorded')
    registry=json.loads((ROOT/'experiments/sprint-20261017-hydrology/research/sites.json').read_text())
    meta={s['site_no']:s for s in registry if len(str(s.get('huc_cd','')))>=8}
    prior=exposed_stations()
    archive_path,polygons=basins()
    missing=sorted(s for s in prior if s not in polygons)
    exposed=[s for s in sorted(prior) if s in polygons]
    exposed_geometry=[polygons[s][0] for s in exposed]
    tree=STRtree(exposed_geometry)
    unusable=[s for s in prior if s not in polygons]
    candidates=[]
    for station,(polygon,normalized) in sorted(polygons.items()):
        if station in prior:continue
        overlaps=[int(i) for i in tree.query(polygon,predicate='intersects')
            if polygon.intersection(exposed_geometry[int(i)]).area>0]
        record=dict(station=station,geometry_normalized=normalized,
            fitted_catchment_overlaps=len(overlaps),joined=[exposed[i] for i in overlaps][:5])
        source=meta.get(station)
        if source:
            def numeric(value):
                try:return float(str(value).strip())
                except ValueError:return float('nan')
            record.update(name=source['station_nm'],huc=source['huc_cd'],area_mi2=numeric(source['drain_area_va']),
                elevation_ft=numeric(source['alt_va']))
            record['eligible_screen']=bool(20<record['area_mi2']<1200 and np.isfinite(record['elevation_ft']))
        else:
            record['eligible_screen']=False
        record['independent']=not overlaps
        candidates.append(record)
    independent=[c for c in candidates if c['independent'] and c['eligible_screen']]
    independent.sort(key=lambda c:c['area_mi2'])
    selected=[]
    for candidate in independent:
        polygon=polygons[candidate['station']][0]
        if any(polygon.intersection(chosen['geometry']).area>0 for chosen in selected):continue
        selected.append(dict(station=candidate['station'],name=candidate.get('name'),huc=candidate.get('huc'),
            area_mi2=candidate.get('area_mi2'),geometry=polygon))
    write_json(destination,dict(
        evidence='Boundary geometry only. No discharge, reference or outcome was acquired or inspected.',
        screening='Positive-area intersection of every unused catchment against every catchment an earlier experiment exposed',
        exposed_stations=len(prior),exposed_with_retained_boundary=len(exposed),exposed_absent_from_boundary=unusable,
        boundary_absent_from_prior=missing,
        boundary_source_sha256=digest(archive_path),registry_sha256=digest(ROOT/'experiments/sprint-20261017-hydrology/research/sites.json'),
        candidates_screened=len(candidates),
        independent_and_eligible=[{k:v for k,v in c.items() if k!='geometry'} for c in independent],
        disjoint_selection=[{k:v for k,v in c.items() if k!='geometry'} for c in selected],
        limitation='Polygon independence removes overlap and nesting. It does not remove spatial weather correlation, and these basins still need an outcome period no earlier study inspected.'))
    print('unused screened',len(candidates),'independent and eligible',len(independent),'disjoint selection',len(selected),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['reserve','screen'])
    arguments=parser.parse_args()
    if arguments.command=='reserve':reserve()
    else:screen()
