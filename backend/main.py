"""PlanIT Control local API. Run one Uvicorn worker for the in-process scheduler."""
import asyncio
import hmac
import hashlib
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from backend import providers, jev, missions, memory, overlap
from backend.domain import now_utc, parse_time, lightning_state, hazard_index, briefing_due, distance_miles
from backend.store import initialize, db, get_cache, put_cache
from backend.predict import predictions
from scripts.common import ROOT

log = logging.getLogger('planit')
SITES = json.loads((ROOT/'data/sites.json').read_text())['sites']
SITE_BY_ID = {s['id']:s for s in SITES}
MODEL_REVISION = hashlib.sha256((ROOT/'models/hazards.joblib').read_bytes()).hexdigest()
UPLOADS = ROOT/'data/uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
REFRESH_LOCK = asyncio.Lock()
refresh_status = {'running':False, 'completed':0, 'total':len(SITES), 'last_finished':None}


def site_or_404(site_id):
    if site_id not in SITE_BY_ID:
        raise HTTPException(404, 'Unknown EIA facility')
    return SITE_BY_ID[site_id]


def supervisor_tokens():
    """PLANIT_SUPERVISOR_TOKENS holds comma-separated Name=token pairs; names may not contain '=' or ','."""
    pairs = {}
    for item in os.getenv('PLANIT_SUPERVISOR_TOKENS', '').split(','):
        name, sep, token = item.partition('=')
        if sep and name.strip() and token.strip():
            pairs[token.strip()] = name.strip()
    return pairs


def admin(authorization: str = Header(default='')):
    """The acting supervisor: a named personal token, or the shared token whose typed names are self-declared."""
    personal = supervisor_tokens()
    for token, name in personal.items():
        if hmac.compare_digest(authorization, 'Bearer '+token):
            return dict(name=name, identity='token')
    shared = os.getenv('PLANIT_ADMIN_TOKEN', '')
    if not shared and not personal:
        raise HTTPException(503, 'Set PLANIT_ADMIN_TOKEN or PLANIT_SUPERVISOR_TOKENS before supervisor actions')
    if shared and hmac.compare_digest(authorization, 'Bearer '+shared):
        return dict(name=None, identity='self-declared')
    raise HTTPException(401, 'Supervisor token required')


def audit(c, actor, action, target, typed_name=None, **detail):
    """Append-only record of a supervisor change, written in the same transaction. Never stores report text."""
    c.execute('INSERT INTO audit (at, actor, identity, action, target, detail) VALUES (?,?,?,?,?,?)',
              (now_utc().isoformat(), actor['name'] or typed_name, actor['identity'], action, target, json.dumps(detail)))


def site_reports(site_id):
    with db() as c:
        rows = c.execute('SELECT reports.*, report_triage.payload AS triage FROM reports LEFT JOIN report_triage ON reports.id=report_triage.report_id WHERE site_id=? ORDER BY created_at DESC', (site_id,)).fetchall()
        return [dict(r, triage=json.loads(r['triage']) if r['triage'] else None) for r in rows]


def strike_states(site):
    with db() as c:
        rows = [dict(r) for r in c.execute('SELECT * FROM strikes WHERE occurred_at>?', ((now_utc()-timedelta(hours=1)).isoformat(),))]
    return {source:lightning_state([s for s in rows if s['source']==source],site) for source in ['live','replay']}


def mission_context(mission, site, assessment):
    weather = assessment['weather']
    work_point = dict(site, latitude=mission['latitude'], longitude=mission['longitude'])
    return dict(now=now_utc(), conditions=missions.window_conditions(weather, mission['shift_start'], mission['shift_end']),
                weather=weather, alerts=assessment['alerts']['items'], model=assessment['model'], floodplain=assessment['floodplain'],
                reports=site_reports(site['id']), lightning=strike_states(work_point),
                gauges=assessment.get('gauges'), wildfires=assessment.get('wildfires'),
                live_feed_configured=bool(os.getenv('PLANIT_LIGHTNING_TOKEN')))


def record_jha(mission, site, assessment, trigger):
    """Append a JHA version only when hazards, ratings or cited evidence changed."""
    ctx = mission_context(mission, site, assessment)
    jha = missions.build_jha(mission, ctx)
    with db() as c:
        c.execute('BEGIN IMMEDIATE')  # read-then-insert must not interleave with another writer
        c.execute('INSERT OR REPLACE INTO jha_checks VALUES (?,?,?,?)',
                  (mission['id'], now_utc().isoformat(), assessment['assessed_at'], int(bool(assessment.get('stale')))))
        prior = c.execute('SELECT * FROM jha_versions WHERE mission_id=? ORDER BY version DESC LIMIT 1', (mission['id'],)).fetchone()
        if prior and prior['fingerprint'] == jha['fingerprint']:
            return
        diff = missions.diff_jha(json.loads(prior['payload'])['entries'] if prior else [], jha['entries'])
        payload = dict(jha, conditions=ctx['conditions'], computed_at=now_utc().isoformat(),
                       inputs=dict(assessed_at=assessment['assessed_at'], weather_feed=assessment['weather'].get('feed'),
                                   alerts_feed=assessment['alerts'].get('feed'), model_available=assessment['model'].get('available')))
        c.execute('INSERT OR IGNORE INTO jha_versions VALUES (?,?,?,?,?,?,?)',
                  (mission['id'], (prior['version'] if prior else 0)+1, now_utc().isoformat(), trigger, jha['fingerprint'],
                   json.dumps(payload), json.dumps(diff)))


def open_missions(site_id):
    with db() as c:
        rows = [dict(r) for r in c.execute('SELECT * FROM missions WHERE site_id=?', (site_id,))]
    return [m for m in rows if parse_time(m['shift_end']) > now_utc()]


def update_missions(site, trigger, assessment=None):
    assessment = assessment or cached_assessment(site)
    if not assessment:
        return
    for mission in open_missions(site['id']):
        record_jha(mission, site, assessment, trigger)


async def assess(site, trigger='Environmental refresh'):
    weather, alerts, model, exposure, gauges, fires = await asyncio.gather(
        providers.weather(site), providers.alerts(site), predictions(site), providers.floodplain(site),
        providers.river_gauges(site), providers.wildfires())
    if fires.get('incidents') is not None:
        fires = dict(fires, incidents=[f for f in fires['incidents']
                                       if distance_miles(site['latitude'], site['longitude'], f['latitude'], f['longitude']) <= 150])
    reports = site_reports(site['id'])
    active_reports = [r for r in reports if parse_time(r['created_at']) > now_utc()-timedelta(hours=48)]
    index = hazard_index(weather, alerts['items'], active_reports, model['hazards'], exposure)
    result = dict(site=site, weather=weather, alerts=alerts, model=model, index=index, floodplain=exposure,
                  gauges=gauges, wildfires=fires, assessed_at=now_utc().isoformat())
    put_cache('assessment:'+MODEL_REVISION+':'+site['id'], result, result['assessed_at'])
    with db() as c:
        c.execute('INSERT OR IGNORE INTO history VALUES (?,?,?,?)',
                  (site['id'],result['assessed_at'],index['score'],index['coverage']))
    update_missions(site, trigger, result)
    return result


def cached_assessment(site):
    cached = get_cache('assessment:'+MODEL_REVISION+':'+site['id'])
    if not cached:
        return None
    result = cached['payload']
    if now_utc()-parse_time(result['assessed_at']) > timedelta(minutes=35):
        result = dict(result, stale=True)
    return result


async def current_assessment(site, trigger='Environmental refresh'):
    """A fresh assessment, or the last one marked stale with the error when reassessment fails."""
    result = cached_assessment(site)
    if result and not result.get('stale'):
        return result
    try:
        return await assess(site, trigger)
    except Exception as exc:
        log.exception('Reassessment failed for %s', site['id'])
        if not result:
            raise HTTPException(503, 'Assessment failed and no earlier assessment exists; see the server log')
        return dict(result, stale=True, reassessment_error=f'{type(exc).__name__}: {exc}'[:300])


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
        # Never-attempted snapshots first, so failures that keep failing cannot starve newer briefings.
        rows = c.execute("SELECT * FROM briefings WHERE delivery != 'delivered' "
                         "ORDER BY delivery LIKE 'failed%', generated_at LIMIT 20").fetchall()
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


async def watch_alerts():
    """Re-check NWS alerts at sites with open missions; reassess when the set of in-effect alerts changes."""
    for site in SITES:
        if not open_missions(site['id']):
            continue
        try:
            latest = await providers.alerts(site)
            cached = cached_assessment(site)
            known = (cached or {}).get('alerts', {}).get('items')
            if latest['items'] is not None and (known is None or {a['id'] for a in known} != {a['id'] for a in latest['items']}):
                await assess(site, 'NWS alert set changed')
        except Exception:
            log.exception('Alert watch failed for %s', site['id'])


async def alert_scheduler():
    while True:
        await asyncio.sleep(300)
        await watch_alerts()


@asynccontextmanager
async def lifespan(app):
    initialize()
    tasks = [asyncio.create_task(job()) for job in (scheduler, refresh_scheduler, alert_scheduler)] if os.getenv('PLANIT_SCHEDULER','true')=='true' else []
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
    def upper(index):
        """Score with every unknown component at full severity; missing data must not rank a site as calm."""
        return (index['score'] or 0)+sum(10*c['weight'] for c in index['components'] if c['points'] is None)
    return dict(sites=sorted(rows,key=lambda s: (not s['lightning']['live']['active'], -upper(s['index']) if s['index'] else 1,
                                                 -(s['index']['score'] or 0) if s['index'] else 0)),
                refresh=refresh_status,source='EIA-860M December 2024')


@app.get('/api/sites/{site_id}')
async def detail(site_id):
    s = site_or_404(site_id)
    result = await current_assessment(s)
    with db() as c:
        history = [dict(r) for r in c.execute('SELECT * FROM history WHERE site_id=? ORDER BY recorded_at DESC LIMIT 48',(site_id,))]
    point = (s['latitude'], s['longitude'])
    nearby = dict(flood=missions.gauge_evidence(point, result.get('gauges'), now_utc()),
                  wildfire=missions.wildfire_evidence(point, result.get('wildfires'), now_utc()))
    return dict(result, reports=site_reports(site_id), lightning=strike_states(s), history=history, nearby=nearby)


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
    result = record_strike(value,'live')
    for site in SITES:
        if distance_miles(value.latitude,value.longitude,site['latitude'],site['longitude']) <= 60:
            update_missions(site,'Recorded lightning strike (live feed)')
    return result


@app.post('/api/sites/{site_id}/lightning-replay')
def replay(site_id, actor: dict = Depends(admin)):
    s = site_or_404(site_id)
    record_strike(Strike(id=str(uuid.uuid4()),latitude=s['latitude'],longitude=s['longitude'],occurred_at=now_utc().isoformat()),'replay')
    with db() as c:
        audit(c, actor, 'lightning.replay', site_id)
    update_missions(s,'Labeled lightning replay at the facility (simulated, not a live observation)')
    return dict(source='LABELED REPLAY — NOT A LIVE OBSERVATION',**strike_states(s)['replay'])


@app.post('/api/sites/{site_id}/reports')
async def report(site_id: str, classification: str=Form(...), description: str=Form(...),
                 latitude: float=Form(...), longitude: float=Form(...), image: UploadFile | None=File(default=None),
                 categories: str=Form('[]'), jev_consent: bool=Form(False), actor: dict=Depends(admin)):
    s = site_or_404(site_id)
    if classification not in jev.CATEGORIES:
        raise HTTPException(422,'Unknown hazard classification')
    try:
        extra = json.loads(categories)
        if not isinstance(extra,list) or len(extra)>len(jev.CATEGORIES) or any(not isinstance(v,str) or v not in jev.CATEGORIES for v in extra):
            raise ValueError()
    except (ValueError,TypeError):
        raise HTTPException(422,'Categories must be a JSON array of supported hazards')
    manual_categories = list(dict.fromkeys([classification,*extra]))
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
        triage = dict(manual_categories=manual_categories,source='manual',provider=await jev.classify(description,consent=jev_consent))
        with db() as c:
            c.execute('INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?)',
                (identifier,site_id,classification,description,latitude,longitude,now_utc().isoformat(),'pending',image_path))
            c.execute('INSERT INTO report_triage VALUES (?,?,?)',(identifier,json.dumps(triage),now_utc().isoformat()))
            audit(c, actor, 'report.submit', identifier, site_id=site_id, categories=manual_categories, jev_consent=jev_consent,
                  jev_status=triage['provider']['status'], image=bool(image_path))
    except Exception:
        if image_path:
            (UPLOADS/image_path).unlink(missing_ok=True)
        raise
    update_missions(s,f'Crew report {identifier[:8]} submitted (pending review)')
    return dict(id=identifier,status='pending',image=image_path,triage=triage)


class TriageCorrection(BaseModel):
    categories: list[Literal['heat','storm','flood','lightning','wildfire','wildlife','unstable_terrain','equipment_obstruction','other']] = Field(min_length=1,max_length=9)


@app.patch('/api/reports/{report_id}/triage')
def correct_triage(report_id: str, value: TriageCorrection, actor: dict = Depends(admin)):
    with db() as c:
        row=c.execute('SELECT id,site_id FROM reports WHERE id=?',(report_id,)).fetchone()
        if not row:raise HTTPException(404,'Report not found')
        prior=c.execute('SELECT payload FROM report_triage WHERE report_id=?',(report_id,)).fetchone()
        triage=json.loads(prior['payload']) if prior else {}
        audit(c, actor, 'report.relabel', report_id, categories=list(dict.fromkeys(value.categories)), previous=triage.get('manual_categories'))
        triage.update(manual_categories=list(dict.fromkeys(value.categories)),source='manual_correction',corrected_at=now_utc().isoformat())
        c.execute('UPDATE reports SET classification=? WHERE id=?',(value.categories[0],report_id))
        c.execute('INSERT OR REPLACE INTO report_triage VALUES (?,?,?)',(report_id,json.dumps(triage),now_utc().isoformat()))
    update_missions(site_or_404(row['site_id']),f'Crew report {report_id[:8]} labels corrected by supervisor')
    return triage


class Review(BaseModel):
    status: Literal['verified','rejected','resolved']


@app.patch('/api/reports/{report_id}')
async def review(report_id: str, value: Review, actor: dict = Depends(admin)):
    with db() as c:
        row = c.execute('SELECT site_id, status FROM reports WHERE id=?',(report_id,)).fetchone()
        if not row:
            raise HTTPException(404,'Report not found')
        c.execute('UPDATE reports SET status=? WHERE id=?',(value.status,report_id))
        audit(c, actor, 'report.review', report_id, status=value.status, previous=row['status'])
    await assess(site_or_404(row['site_id']),f'Crew report {report_id[:8]} marked {value.status} by supervisor')
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


@app.get('/api/activities')
def activities():
    return [dict(id=k, label=v['label'], sensitivity=v['sensitivity'], wind_kmh=v.get('wind', missions.DEFAULT_WIND))
            for k, v in missions.ACTIVITIES.items()]


class MissionBody(BaseModel):
    site_id: str
    name: str = Field(min_length=3, max_length=120)
    activity: Literal[tuple(missions.ACTIVITIES)]
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    shift_start: str
    shift_end: str
    crew_size: int = Field(ge=1, le=200)
    supervisor: str = Field(min_length=2, max_length=80)


@app.post('/api/missions')
async def create_mission(value: MissionBody, actor: dict = Depends(admin)):
    site = site_or_404(value.site_id)
    try:
        start, end = parse_time(value.shift_start), parse_time(value.shift_end)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if not start < end <= start+timedelta(hours=24):
        raise HTTPException(422, 'Shift must end after it starts and last at most 24 hours')
    if end <= now_utc():
        raise HTTPException(422, 'Shift has already ended; missions plan current or future work')
    if distance_miles(value.latitude, value.longitude, site['latitude'], site['longitude']) > 50:
        raise HTTPException(422, 'Work location must be within 50 miles of the selected facility')
    mission = dict(value.model_dump(), id=str(uuid.uuid4()), name=value.name.strip(), supervisor=value.supervisor.strip(),
                   shift_start=start.isoformat(), shift_end=end.isoformat(), created_at=now_utc().isoformat())
    with db() as c:
        c.execute('INSERT INTO missions VALUES (:id,:site_id,:name,:activity,:latitude,:longitude,:shift_start,:shift_end,:crew_size,:supervisor,:created_at)', mission)
        audit(c, actor, 'mission.create', mission['id'], name=mission['name'], site_id=mission['site_id'], activity=mission['activity'],
              responsible=mission['supervisor'])
    record_jha(mission, site, await current_assessment(site, 'Mission created'), 'Mission created')
    return mission


def mission_or_404(mission_id):
    with db() as c:
        row = c.execute('SELECT * FROM missions WHERE id=?', (mission_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Mission not found')
    return dict(row)


def versions_for(mission_id):
    with db() as c:
        return [dict(r, payload=json.loads(r['payload']), diff=json.loads(r['diff'])) for r in
                c.execute('SELECT * FROM jha_versions WHERE mission_id=? ORDER BY version', (mission_id,))]


@app.get('/api/missions')
def list_missions(site_id: str | None = None):
    with db() as c:
        rows = [dict(r) for r in (c.execute('SELECT * FROM missions WHERE site_id=? ORDER BY shift_start', (site_id,)) if site_id
                                  else c.execute('SELECT * FROM missions ORDER BY shift_start'))]
        for m in rows:
            latest = c.execute('SELECT version, payload FROM jha_versions WHERE mission_id=? ORDER BY version DESC LIMIT 1', (m['id'],)).fetchone()
            m.update(site_name=SITE_BY_ID[m['site_id']]['name'], version=latest['version'] if latest else None,
                     readiness=json.loads(latest['payload'])['readiness'] if latest else None,
                     ended=parse_time(m['shift_end']) <= now_utc())
    return rows


@app.get('/api/missions/{mission_id}')
async def mission_detail(mission_id: str):
    mission = mission_or_404(mission_id)
    site = SITE_BY_ID[mission['site_id']]
    ended = parse_time(mission['shift_end']) <= now_utc()
    if not ended:
        record_jha(mission, site, await current_assessment(site), 'Conditions re-checked when the mission was opened')
    versions = versions_for(mission_id)
    jha = versions[-1]['payload']
    current = {e['hazard']: missions.entry_fingerprint(e) for e in jha['entries']}
    with db() as c:
        # Decisions from before identity was recorded have no identity row and were made under the shared token.
        decisions = [dict(r, identity=r['identity'] or 'self-declared', stale=current.get(r['hazard']) != r['entry_fingerprint']) for r in
                     c.execute('SELECT d.*, i.identity FROM jha_decisions d LEFT JOIN jha_decision_identity i ON i.decision_id=d.id '
                               'WHERE d.mission_id=? ORDER BY d.created_at DESC', (mission_id,))]
        check = c.execute('SELECT checked_at, assessed_at, stale FROM jha_checks WHERE mission_id=?', (mission_id,)).fetchone()
    work_point = dict(site, latitude=mission['latitude'], longitude=mission['longitude'])
    noaa, crew = memory.site_records(site['id']), memory.crew_records(site_reports(site['id']))
    records = noaa+crew
    plans = overlap.dataset()
    return dict(mission=dict(mission, ended=ended), site=site, activity=missions.ACTIVITIES[mission['activity']],
                conditions=jha['conditions'], jha=jha, decisions=decisions,
                checked=dict(check, stale=bool(check['stale'])) if check else None,
                versions=[dict(version=v['version'], created_at=v['created_at'], trigger=v['trigger'], diff=v['diff'],
                               readiness=v['payload']['readiness']) for v in versions],
                memory=memory.retrieve(work_point, mission['activity'], jha['entries'], mission['shift_start'], records),
                memory_source=dict(noaa=len(noaa), crew=len(crew), note=(memory.dataset() or {}).get('note')),
                planned_nearby=overlap.near_mission(plans['projects'], mission['latitude'], mission['longitude'], mission['shift_start'],
                                                    site['operator'])[:6] if plans else [],
                planned_source=dict(inventory=plans['inventory'], note=plans['note'], url=plans['source']['url']) if plans else None)


class Decision(BaseModel):
    hazard: Literal[tuple(missions.LABELS)]
    decision: Literal['accept', 'override', 'mitigate']
    rationale: str = Field(min_length=5, max_length=1000)
    supervisor: str = Field(min_length=2, max_length=80)
    rating: Literal['low', 'elevated', 'high'] | None = None
    mitigation: str | None = Field(default=None, max_length=1000)
    version: int = Field(ge=1)


@app.post('/api/missions/{mission_id}/decisions')
def decide(mission_id: str, value: Decision, actor: dict = Depends(admin)):
    mission_or_404(mission_id)
    latest = versions_for(mission_id)[-1]
    if value.version != latest['version']:
        raise HTTPException(409, f"The JHA is now version {latest['version']}; review the changes before deciding (you decided on version {value.version}).")
    entry = next((e for e in latest['payload']['entries'] if e['hazard'] == value.hazard), None)
    if entry is None:
        raise HTTPException(422, 'Hazard is not in the current JHA version')
    if value.decision == 'override' and value.rating is None:
        raise HTTPException(422, 'An override needs the supervisor rating')
    if value.decision == 'mitigate' and not (value.mitigation or '').strip():
        raise HTTPException(422, 'A mitigation decision needs the controls applied')
    if value.decision == 'override' and entry['rating'] == 'stop_work':
        raise HTTPException(409, 'Lightning stop work is a deterministic rule and cannot be overridden. Record a mitigation or wait for the timer review.')
    record = dict(id=str(uuid.uuid4()), mission_id=mission_id, version=latest['version'], hazard=value.hazard, decision=value.decision,
                  rating=value.rating if value.decision == 'override' else None, mitigation=(value.mitigation or '').strip() or None,
                  rationale=value.rationale.strip(), supervisor=actor['name'] or value.supervisor.strip(),
                  entry_fingerprint=missions.entry_fingerprint(entry), created_at=now_utc().isoformat(), identity=actor['identity'])
    with db() as c:
        c.execute('INSERT INTO jha_decisions VALUES (:id,:mission_id,:version,:hazard,:decision,:rating,:mitigation,:rationale,:supervisor,:entry_fingerprint,:created_at)', record)
        c.execute('INSERT INTO jha_decision_identity VALUES (?,?)', (record['id'], record['identity']))
        audit(c, actor, 'decision.record', record['id'], typed_name=record['supervisor'], mission_id=mission_id, hazard=value.hazard,
              decision=value.decision, version=latest['version'])
    return record


@lru_cache(maxsize=2)
def cached_overlaps(utilities_only):
    data = overlap.dataset()
    return overlap.find_overlaps(data['projects'], utilities_only=utilities_only) if data else None


@app.get('/api/audit', dependencies=[Depends(admin)])
def audit_log(limit: int = 200):
    with db() as c:
        rows = c.execute('SELECT * FROM audit ORDER BY seq DESC LIMIT ?', (max(1, min(limit, 1000)),)).fetchall()
    return [dict(r, detail=json.loads(r['detail'])) for r in rows]


@app.get('/api/plan-overlaps')
def plan_overlaps(utilities_only: bool = False, kind: str | None = None, cross_state: bool = False, limit: int = 100):
    data = overlap.dataset()
    if not data:
        raise HTTPException(503, 'Run scripts.build_planned_projects to extract EIA-860M planned generators')
    pairs = [p for p in cached_overlaps(utilities_only) if (not kind or p['kind'] == kind) and (not cross_state or p['cross_state'])]
    projects = [p for p in data['projects'] if not utilities_only or p['sector'] == 'Electric Utility']
    suspect = overlap.suspect_coordinates(data['projects'])
    return dict(source=data['source'], inventory=data['inventory'], note=data['note'], projects=len(projects),
                entities=len({p['entity_id'] for p in projects}), total=len(pairs), pairs=pairs[:max(1, min(limit, 500))],
                suspect_coordinates=[p for p in projects if p['plant_id'] in suspect],
                thresholds=dict(close_miles=overlap.CLOSE_MILES, region_miles=overlap.REGION_MILES, concurrent_months=overlap.CONCURRENT_MONTHS))


if (ROOT/'frontend/dist').exists():
    app.mount('/assets',StaticFiles(directory=ROOT/'frontend/dist/assets'),name='assets')

    @app.get('/')
    def index():
        return FileResponse(ROOT/'frontend/dist/index.html')
