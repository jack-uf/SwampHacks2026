"""Operational HRRR precipitation-field acquisition for a fixed 12 UTC development pilot."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone

import eccodes
import httpx
import numpy as np
import pandas as pd

from scripts.flood_sprint import write_json
from scripts.high_flow_research import load
from scripts.high_flow_signals import OUT
from scripts.hydrology_research import ROOT, digest

HR = OUT/'hrrr'
BUCKET = 'https://noaa-hrrr-bdp-pds.s3.amazonaws.com'


def precipitation_range(index, end_hour=12):
    lines = index.strip().splitlines()
    matches = [i for i,line in enumerate(lines) if f':APCP:surface:0-{end_hour} hour acc fcst:' in line]
    if len(matches)!=1 or matches[0]+1>=len(lines):
        raise ValueError('Expected exactly one complete cumulative precipitation message')
    i=matches[0]
    start,stop=int(lines[i].split(':')[1]),int(lines[i+1].split(':')[1])-1
    if not 0<=start<stop:raise ValueError('Invalid GRIB byte range')
    return start,stop


def grid():
    """Pin gauge and basin grid cells once, before forecast outcome scoring."""
    path=HR/'grid.json'
    if path.exists():return json.loads(path.read_text())
    import shapefile
    from scipy.spatial import cKDTree
    from shapely import contains_xy
    from shapely.geometry import shape
    frame=load(True)
    stations=frame[['station','latitude','longitude']].drop_duplicates().sort_values('station')
    archive=OUT/'sources/basin_set_full_res.zip'
    z=zipfile.ZipFile(archive)
    base=next(n[:-4] for n in z.namelist() if n.endswith('.shp'))
    reader=shapefile.Reader(shp=io.BytesIO(z.read(base+'.shp')),shx=io.BytesIO(z.read(base+'.shx')),dbf=io.BytesIO(z.read(base+'.dbf')))
    fields=[field[0] for field in reader.fields[1:]]
    id_field=next(f for f in fields if f.lower() in ['hru_id','gauge_id'])
    polygons={str(sr.record[fields.index(id_field)]).zfill(8):shape(sr.shape.__geo_interface__) for sr in reader.iterShapeRecords()}
    probe=HR/'probe.grib'
    h=eccodes.codes_new_from_message(probe.read_bytes())
    try:
        latitude=eccodes.codes_get_array(h,'latitudes')
        longitude=eccodes.codes_get_array(h,'longitudes')
        longitude=np.where(longitude>180,longitude-360,longitude)
        geometry={key:eccodes.codes_get(h,key) for key in ['Ni','Nj','latitudeOfFirstGridPointInDegrees','longitudeOfFirstGridPointInDegrees','DxInMetres','DyInMetres']}
        tree=cKDTree(np.c_[latitude,longitude])
        nearest=tree.query(stations[['latitude','longitude']].to_numpy())[1]
        mapping={}
        for (_,row),index in zip(stations.iterrows(),nearest):
            polygon=polygons[row.station]
            if not polygon.is_valid:raise ValueError('Invalid official basin polygon')
            west,south,east,north=polygon.bounds
            within=np.flatnonzero((longitude>=west)&(longitude<=east)&(latitude>=south)&(latitude<=north))
            cells=within[contains_xy(polygon,longitude[within],latitude[within])]
            if not len(cells):raise ValueError('Basin contains no HRRR cell centres: '+row.station)
            mapping[row.station]=dict(point=int(index),cells=cells.tolist(),latitude=row.latitude,longitude=row.longitude)
    finally:eccodes.codes_release(h)
    result=dict(geometry=geometry,stations=mapping,basin_source_sha256=digest(archive),probe_sha256=digest(probe),
        basin_aggregation='Unweighted 3-km grid-cell-centre mean within full-resolution CAMELS polygons')
    write_json(path,result)
    return result


def extract(body,issue,mapping):
    h=eccodes.codes_new_from_message(body)
    try:
        if eccodes.codes_get(h,'shortName')!='tp' or eccodes.codes_get(h,'units')!='kg m**-2' or eccodes.codes_get(h,'stepRange')!='0-12':
            raise ValueError('Forecast field contract mismatch')
        init=pd.to_datetime(str(eccodes.codes_get(h,'dataDate'))+f"{eccodes.codes_get(h,'dataTime'):04d}",format='%Y%m%d%H%M',utc=True)
        valid=pd.to_datetime(str(eccodes.codes_get(h,'validityDate'))+f"{eccodes.codes_get(h,'validityTime'):04d}",format='%Y%m%d%H%M',utc=True)
        if init!=issue-pd.Timedelta(hours=6) or valid!=issue+pd.Timedelta(hours=6):
            raise ValueError('Forecast cycle or valid interval leaked beyond declared alignment')
        if any(eccodes.codes_get(h,key)!=value for key,value in mapping['geometry'].items()):
            raise ValueError('HRRR grid changed; fixed cell mapping is invalid')
        values=eccodes.codes_get_values(h)
        rows={}
        for sid,indices in mapping['stations'].items():
            point=values[indices['point']]
            basin=values[indices['cells']]
            if not np.isfinite(basin).all() or basin.min()<0 or basin.max()>10000 or not 0<=point<=10000:
                raise ValueError('Invalid precipitation values')
            rows[sid]=dict(hrrr_point12=float(point),hrrr_basin12=float(basin.mean()),hrrr_max12=float(basin.max()))
        return rows
    finally:eccodes.codes_release(h)


def collect():
    mapping=grid()
    dates=pd.date_range('2020-01-01T12:00Z','2024-12-31T12:00Z',freq='D')
    if not (HR/'acquisition.json').exists():
        write_json(HR/'acquisition.json',dict(issue_hour_utc=12,initialization_lag_hours=6,forecast_end_lead=12,
            dates=[t.isoformat() for t in dates],grid_sha256=digest(HR/'grid.json'),code_sha256=digest(ROOT/'scripts/high_flow_hrrr.py'),
            reason='Fixed once-daily development pilot chosen for acquisition cost before model scoring',
            evidence='Operational model archive. Six-hour publication buffer assumed; original receipt evidence unavailable.',
            retention='Retain extracted values, exact ranges, source hashes and ETags. Raw fields are not all retained because local disk has under 7 GiB free.'))
    def one(issue):
        path=HR/'fields'/(issue.strftime('%Y%m%dT%H')+'.json.gz')
        if path.exists():
            x=json.loads(gzip.decompress(path.read_bytes()))
            if x['issue_time']!=issue.isoformat():raise ValueError('Cached issue mismatch')
            return
        init=issue-pd.Timedelta(hours=6)
        url=BUCKET+f'/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf12.grib2'
        for attempt in range(3):
            try:
                idx=httpx.get(url+'.idx',timeout=30);idx.raise_for_status()
                start,stop=precipitation_range(idx.text)
                response=httpx.get(url,headers={'Range':f'bytes={start}-{stop}'},timeout=45)
                response.raise_for_status()
                content_range=response.headers.get('content-range','')
                if response.status_code!=206 or not content_range.startswith(f'bytes {start}-{stop}/') or len(response.content)!=stop-start+1:
                    raise ValueError('Server did not honor the exact GRIB range')
                data=extract(response.content,issue,mapping)
                record=dict(issue_time=issue.isoformat(),initialization_time=init.isoformat(),source_url=url,
                    byte_range=[start,stop],index_sha256=hashlib.sha256(idx.content).hexdigest(),
                    field_sha256=hashlib.sha256(response.content).hexdigest(),etag=response.headers.get('etag'),
                    last_modified=response.headers.get('last-modified'),received_at=datetime.now(timezone.utc).isoformat(),stations=data)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(gzip.compress(json.dumps(record,allow_nan=False).encode(),mtime=0))
                if issue.day==1 and issue.month==1:print('HRRR acquired',str(issue),flush=True)
                return
            except httpx.HTTPError:
                if attempt==2:raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,dates))
    print('Completed',len(dates),'global forecast fields',flush=True)


def materialize():
    target=HR/'features.csv.gz'
    if target.exists():raise FileExistsError(target)
    rows=[]
    for p in sorted((HR/'fields').glob('*.json.gz')):
        x=json.loads(gzip.decompress(p.read_bytes()))
        rows.extend(dict(station=sid,decision_time=x['issue_time'],**v) for sid,v in x['stations'].items())
    f=pd.DataFrame(rows)
    if f.duplicated(['station','decision_time']).any() or len(f)!=1827*len(grid()['stations']):
        raise ValueError('Incomplete once-daily forecast acquisition')
    f.to_csv(target,index=False,compression={'method':'gzip','mtime':0})
    write_json(HR/'features.json',dict(sha256=digest(target),rows=len(f),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (HR/'fields').glob('*.json.gz')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['grid','collect','materialize']);args=p.parse_args()
    HR.mkdir(parents=True,exist_ok=True)
    if args.command=='grid':grid()
    elif args.command=='collect':collect()
    else:materialize()
