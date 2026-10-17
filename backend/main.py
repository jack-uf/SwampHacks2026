"""PlanIT Control local API. Run one Uvicorn worker for the in-process scheduler."""
import asyncio
import hmac
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from backend import providers
from backend.domain import now_utc, parse_time, lightning_state, hazard_index, briefing_due, distance_miles
from backend.store import initialize, db, get_cache, put_cache
from backend.predict import predictions
from scripts.common import ROOT

log = logging.getLogger('planit')
SITES = json.loads((ROOT/'data/sites.json').read_text())['sites']
SITE_BY_ID = {s['id']:s for s in SITES}
UPLOADS = ROOT/'data/uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
REFRESH_LOCK = asyncio.Lock()
refresh_status = {'running':False, 'completed':0, 'total':len(SITES), 'last_finished':None}


def site_or_404(site_id):
    if site_id not in SITE_BY_ID:
        raise HTTPException(404, 'Unknown EIA facility')
    return SITE_BY_ID[site_id]


def admin(authorization: str = Header(default='')):
    token = os.getenv('PLANIT_ADMIN_TOKEN', '')
    if not token:
        raise HTTPException(503, 'Set PLANIT_ADMIN_TOKEN before supervisor actions')
    if not hmac.compare_digest(authorization, 'Bearer '+token):
        raise HTTPException(401, 'Supervisor token required')


def site_reports(site_id):
    with db() as c:
        return [dict(r) for r in c.execute('SELECT * FROM reports WHERE site_id=? ORDER BY created_at DESC', (site_id,))]


def strike_states(site):
    with db() as c:
        rows = [dict(r) for r in c.execute('SELECT * FROM strikes WHERE occurred_at>?', ((now_utc()-timedelta(hours=1)).isoformat(),))]
    return {source:lightning_state([s for s in rows if s['source']==source],site) for source in ['live','replay']}


async def assess(site):
    weather, alerts, model, exposure = await asyncio.gather(
        providers.weather(site), providers.alerts(site), predictions(site), providers.floodplain(site))
    reports = site_reports(site['id'])
    active_reports = [r for r in reports if parse_time(r['created_at']) > now_utc()-timedelta(hours=48)]
    index = hazard_index(weather, alerts['items'], active_reports, model['hazards'], exposure)
    result = dict(site=site, weather=weather, alerts=alerts, model=model, index=index, floodplain=exposure,
                  assessed_at=now_utc().isoformat())
    put_cache('assessment:'+site['id'], result, result['assessed_at'])
    with db() as c:
        c.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                  (site['id'],result['assessed_at'],index['score'],index['coverage']))
    return result


def cached_assessment(site):
    cached = get_cache('assessment:'+site['id'])
    if not cached:
        return None
    result = cached['payload']
    if now_utc()-parse_time(result['assessed_at']) > timedelta(minutes=35):
        result = dict(result, stale=True)
    return result


async def refresh_all():
    if REFRESH_LOCK.locked():
        return
    async with REFRESH_LOCK:
        refresh_status.update(running=True, completed=0)
        try:
            for site in SITES:
                try:
                    await assess(site)
                except Exception:
                    log.exception('Assessment failed for %s',site['id'])
                refresh_status['completed'] += 1
                await asyncio.sleep(.25)
        finally:
            refresh_status.update(running=False,last_finished=now_utc().isoformat())


async def generate_briefing(site, local_date=None):
    now = now_utc()
    local_now = now.astimezone(ZoneInfo(site['timezone']))
    date = local_date or str(local_now.date()+timedelta(days=1))
    with db() as c:
        existing = c.execute('SELECT * FROM briefings WHERE site_id=? AND local_date=?',(site['id'],date)).fetchone()
    if existing:
        return dict(existing, payload=json.loads(existing['payload']))
    result = await assess(site)
    forecast = [r for r in result['weather']['forecast'] if str(parse_time(r['time']).astimezone(ZoneInfo(site['timezone'])).date())==date]
    def maximum(key):
        v = [r[key] for r in forecast if r.get(key) is not None]
        return max(v) if len(v)==len(forecast) and v else None
    rainfall = [r['precipitation'] for r in forecast if r.get('precipitation') is not None]
    payload = dict(site_name=site['name'], target_date=date, timezone=site['timezone'],
        forecast_hours=len(forecast), apparent_max_c=maximum('apparent_temperature'), gust_max_kmh=maximum('wind_gusts_10m'),
        rainfall_mm=round(sum(rainfall),1) if rainfall and len(rainfall)==len(forecast) else None,
        active_alerts=result['alerts']['items'], model=result['model'], index=result['index'],
        sources=dict(weather=result['weather']['feed'],alerts=result['alerts']['feed']),
        actions=['Review NWS warnings and site emergency procedures before dispatch.',
                 'Assess occupational heat exposure using WBGT and the crew acclimatization plan.',
                 'Verify drainage, access routes, and lightning shelter; unavailable feeds require review.'],
        note='Local-day forecast briefing. The ML target is tomorrow UTC and may not match this local day. No autonomous work clearance.')
    briefing_id = str(uuid.uuid4())
    with db() as c:
        c.execute('INSERT OR IGNORE INTO briefings (id,site_id,local_date,generated_at,payload) VALUES (?,?,?,?,?)',
                  (briefing_id,site['id'],date,now.isoformat(),json.dumps(payload)))
        row = c.execute('SELECT * FROM briefings WHERE site_id=? AND local_date=?',(site['id'],date)).fetchone()
    return dict(row,payload=json.loads(row['payload']))


async def deliver_pending():
    url = os.getenv('PLANIT_BRIEFING_WEBHOOK')
    if not url or os.getenv('PLANIT_ENABLE_DELIVERY')!='true':
        return
    if not url.startswith('https://'):
        log.error('Briefing webhook must use HTTPS')
        return
    with db() as c:
        rows = c.execute("SELECT * FROM briefings WHERE delivery != 'delivered' ORDER BY generated_at LIMIT 20").fetchall()
    async with httpx.AsyncClient(timeout=15) as client:
        for row in rows:
            try:
                response = await client.post(url,json=json.loads(row['payload']),headers={'Idempotency-Key':row['id']})
                response.raise_for_status()
                state = 'delivered'
            except httpx.HTTPError as exc:
                state = 'failed: '+str(exc)[:200]
            with db() as c:
                c.execute('UPDATE briefings SET delivery=? WHERE id=?',(state,row['id']))


async def tick(now=None):
    now = now or now_utc()
    for site in SITES:
        due, _, target = briefing_due(site, now)
        if due:
            try:
                await generate_briefing(site,target)
            except Exception:
                log.exception('Briefing failed for %s',site['id'])
    await deliver_pending()


async def scheduler():
    while True:
        try:
            await tick()
        except Exception:
            log.exception('Briefing scheduler failed; retrying next minute')
        await asyncio.sleep(60)


async def refresh_scheduler():
    while True:
        await refresh_all()
        await asyncio.sleep(1800)


@asynccontextmanager
async def lifespan(app):
    initialize()
    tasks = [asyncio.create_task(job()) for job in (scheduler, refresh_scheduler)] if os.getenv('PLANIT_SCHEDULER','true')=='true' else []
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title='PlanIT Control',version='0.1.0',lifespan=lifespan)
app.mount('/uploads',StaticFiles(directory=UPLOADS),name='uploads')


@app.get('/api/health')
def health():
    return dict(status='ok', facilities=len(SITES), lightning_live_configured=bool(os.getenv('PLANIT_LIGHTNING_TOKEN')),
                scheduler=os.getenv('PLANIT_SCHEDULER','true'), refresh=refresh_status)


@app.get('/api/sites')
def sites():
    rows = []
    for s in SITES:
        a = cached_assessment(s)
        strikes = strike_states(s)
        rows.append(dict(s,index=a['index'] if a else None, assessed_at=a['assessed_at'] if a else None,
                         stale=a.get('stale',False) if a else False,lightning=strikes))
    return dict(sites=sorted(rows,key=lambda s: (not s['lightning']['live']['active'], -(s['index']['score'] or 0) if s['index'] else 1)),
                refresh=refresh_status,source='EIA-860M December 2024')


@app.get('/api/sites/{site_id}')
async def detail(site_id):
    s = site_or_404(site_id)
    result = cached_assessment(s)
    if not result or result.get('stale'):
        result = await assess(s)
    with db() as c:
        history = [dict(r) for r in c.execute('SELECT * FROM history WHERE site_id=? ORDER BY recorded_at DESC LIMIT 48',(site_id,))]
    return dict(result, reports=site_reports(site_id), lightning=strike_states(s), history=history)


@app.post('/api/refresh',dependencies=[Depends(admin)])
async def refresh():
    if REFRESH_LOCK.locked():
        return refresh_status
    asyncio.create_task(refresh_all())
    return dict(status='refresh requested')


@app.get('/api/evaluation')
def evaluation():
    path = ROOT/'models/evaluation.json'
    if not path.exists():
        raise HTTPException(503,'Train models to generate measured evaluation')
    return json.loads(path.read_text())


class Strike(BaseModel):
    id: str = Field(min_length=1,max_length=120)
    latitude: float = Field(ge=-90,le=90)
    longitude: float = Field(ge=-180,le=180)
    occurred_at: str


def record_strike(value, source):
    try:
        occurred = parse_time(value.occurred_at)
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    if occurred > now_utc()+timedelta(seconds=5) or occurred < now_utc()-timedelta(hours=1):
        raise HTTPException(422,'Only strikes recorded within the last hour are accepted; future timestamps are rejected')
    with db() as c:
        c.execute('INSERT OR IGNORE INTO strikes VALUES (?,?,?,?,?)',
                  (source+':'+value.id,value.latitude,value.longitude,occurred.isoformat(),source))
    return dict(recorded=True,source=source)


@app.post('/api/lightning/strikes')
def ingest_strike(value: Strike, authorization: str=Header(default='')):
    token = os.getenv('PLANIT_LIGHTNING_TOKEN','')
    if not token:
        raise HTTPException(503,'Live strike ingestion is not configured')
    if not hmac.compare_digest(authorization,'Bearer '+token):
        raise HTTPException(401,'Invalid strike provider token')
    return record_strike(value,'live')


@app.post('/api/sites/{site_id}/lightning-replay',dependencies=[Depends(admin)])
def replay(site_id):
    s = site_or_404(site_id)
    record_strike(Strike(id=str(uuid.uuid4()),latitude=s['latitude'],longitude=s['longitude'],occurred_at=now_utc().isoformat()),'replay')
    return dict(source='LABELED REPLAY — NOT A LIVE OBSERVATION',**strike_states(s)['replay'])


@app.post('/api/sites/{site_id}/reports',dependencies=[Depends(admin)])
async def report(site_id: str, classification: str=Form(...), description: str=Form(...),
                 latitude: float=Form(...), longitude: float=Form(...), image: UploadFile | None=File(default=None)):
    s = site_or_404(site_id)
    if classification not in ['heat','storm','flood','lightning','wildfire','wildlife','other']:
        raise HTTPException(422,'Unknown hazard classification')
    description = description.strip()
    if not 10 <= len(description) <= 2000:
        raise HTTPException(422,'Description must have 10–2000 characters')
    if not (-90<=latitude<=90 and -180<=longitude<=180):
        raise HTTPException(422,'Valid geographic coordinates required')
    if distance_miles(latitude,longitude,s['latitude'],s['longitude']) > 50:
        raise HTTPException(422,'Report location must be within 50 miles of the selected facility')
    identifier = str(uuid.uuid4())
    image_path = None
    if image:
        content = await image.read(5*1024*1024+1)
        if len(content) > 5*1024*1024:
            raise HTTPException(413,'Image limit is 5 MB')
        try:
            Image.MAX_IMAGE_PIXELS = 20_000_000
            with Image.open(io.BytesIO(content)) as photo:
                if photo.format not in ['JPEG','PNG','WEBP'] or photo.width*photo.height > 20_000_000:
                    raise ValueError('Unsupported image or dimensions')
                photo.load()
                photo.thumbnail((1600,1600))
                output = io.BytesIO()
                photo.convert('RGB').save(output,format='JPEG',quality=85)
        except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
            raise HTTPException(422,'Provide a valid JPEG, PNG, or WebP image (maximum 20 megapixels)')
        image_path = identifier+'.jpg'
        (UPLOADS/image_path).write_bytes(output.getvalue())
    try:
        with db() as c:
            c.execute('INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)',
                (identifier,site_id,classification,description,latitude,longitude,now_utc().isoformat(),'pending',image_path))
    except Exception:
        if image_path:
            (UPLOADS/image_path).unlink(missing_ok=True)
        raise
    return dict(id=identifier,status='pending',image=image_path)


class Review(BaseModel):
    status: Literal['verified','rejected','resolved']


@app.patch('/api/reports/{report_id}',dependencies=[Depends(admin)])
async def review(report_id: str, value: Review):
    with db() as c:
        row = c.execute('SELECT site_id FROM reports WHERE id=?',(report_id,)).fetchone()
        if not row:
            raise HTTPException(404,'Report not found')
        c.execute('UPDATE reports SET status=? WHERE id=?',(value.status,report_id))
    await assess(site_or_404(row['site_id']))
    return dict(id=report_id,status=value.status)


@app.post('/api/sites/{site_id}/briefings',dependencies=[Depends(admin)])
async def briefing(site_id):
    return await generate_briefing(site_or_404(site_id))


@app.get('/api/sites/{site_id}/briefings')
def briefing_history(site_id):
    site_or_404(site_id)
    with db() as c:
        return [dict(r,payload=json.loads(r['payload'])) for r in c.execute(
            'SELECT * FROM briefings WHERE site_id=? ORDER BY local_date DESC LIMIT 30',(site_id,))]


if (ROOT/'frontend/dist').exists():
    app.mount('/assets',StaticFiles(directory=ROOT/'frontend/dist/assets'),name='assets')

    @app.get('/')
    def index():
        return FileResponse(ROOT/'frontend/dist/index.html')
