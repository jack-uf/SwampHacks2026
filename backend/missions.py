"""Mission shift windows and the deterministic, evidence-cited Job Hazard Analysis."""
import hashlib
import json
from datetime import timedelta, timezone
from math import ceil
from backend.domain import parse_time


RANK = dict(unknown=0, low=1, elevated=2, high=3, stop_work=4)
ACTIVITIES = {
    'general_site_work': dict(label='General site work', sensitivity=dict(heat=.5, storm=.5, flood=.5, lightning=.5)),
    'excavation_trenching': dict(label='Excavation / trenching', terrain=True,
                                 sensitivity=dict(flood=1, unstable_terrain=1, heat=.6, lightning=.6, storm=.4)),
    'aerial_lift_crane': dict(label='Aerial lift / crane work', wind=(32, 45), sensitivity=dict(storm=1, lightning=1, heat=.5)),
    'energized_line_work': dict(label='Energized line work', wind=(40, 60), sensitivity=dict(lightning=1, storm=.8, heat=.7, flood=.4)),
    'vegetation_clearing': dict(label='Vegetation clearing', sensitivity=dict(heat=1, wildfire=.8, wildlife=.8, lightning=.6, storm=.5)),
    'substation_maintenance': dict(label='Substation maintenance', sensitivity=dict(lightning=.8, heat=.7, flood=.6, storm=.5)),
}
DEFAULT_WIND = (45, 72)
ALERT_HAZARDS = [('heat', ['heat']), ('flood', ['flood']), ('storm', ['thunderstorm', 'tornado', 'wind', 'hurricane', 'tropical']),
                 ('lightning', ['thunderstorm']), ('wildfire', ['fire', 'red flag', 'smoke'])]
CONTROLS = {
    'heat': ['Water, rest and shade schedule; acclimatize new or returning workers (OSHA heat guidance).',
             'Measure WBGT on site; apparent temperature is only a screening forecast.'],
    'storm': ['Secure loose material; stop elevated work when wind exceeds the equipment rating.',
              'Monitor NWS warnings during the shift and name a shelter location.'],
    'flood': ['Confirm drainage and access routes before dispatch; do not drive through water over the road.',
              'Stage equipment above the mapped flood zone where possible.'],
    'lightning': ['When thunder roars, go indoors: stop work, shelter, wait 30 minutes after the last thunder or strike.',
                  'Timer expiry needs a supervisor review before restart.'],
    'unstable_terrain': ['Competent-person inspection of excavations after rain (29 CFR 1926.651(k)).',
                         'Keep spoil and equipment back from saturated edges; use protective systems.'],
    'wildfire': ['Check NWS fire-weather products and local fire agency status; keep egress routes clear.'],
    'wildlife': ['Brief the crew on the reported animal; keep distance and contact local wildlife authority if needed.'],
    'equipment_obstruction': ['Re-route or clear access before dispatch; confirm emergency vehicle access.'],
    'other': ['Supervisor to define controls for the reported hazard.'],
}
LABELS = dict(heat='Heat', storm='Storm / wind', flood='Flood / water', lightning='Lightning', unstable_terrain='Unstable ground',
              wildfire='Wildfire / smoke', wildlife='Wildlife', equipment_obstruction='Access / equipment obstruction', other='Other')


def evidence(id, kind, source, summary, rating=None, counts=True, **extra):
    return dict(id=id, kind=kind, source=source, summary=summary, rating=rating, counts=counts, **extra)


def utc_clock(value):
    return parse_time(value).astimezone(timezone.utc).strftime('%H:%M UTC')


def band(value, elevated, high):
    return None if value is None else 'high' if value >= high else 'elevated' if value >= elevated else 'low'


def forecast_evidence(hazard, conditions, key, unit, rating, label):
    value = conditions[key]
    if value is None:
        return evidence(f'forecast:{hazard}:unavailable', 'unavailable', conditions['source'],
                        f'No forecast hours cover this shift ({label} unknown, not zero)')
    return evidence(f'forecast:{hazard}:{rating}', 'predicted', conditions['source'],
                    f'{label} {value:g} {unit} over {conditions["hours_available"]} of {conditions["hours_requested"]} shift hours',
                    rating=rating, value=value, unit=unit, at=conditions['fetched_at'])


def build_jha(mission, ctx):
    activity = ACTIVITIES[mission['activity']]
    c = ctx['conditions']
    found = {}
    def add(hazard, item):
        found.setdefault(hazard, []).append(item)
    add('heat', forecast_evidence('heat', c, 'apparent_max_c', '°C', band(c['apparent_max_c'], 32.2, 39.4), 'Apparent temperature max'))
    elevated, high = activity.get('wind', DEFAULT_WIND)
    add('storm', forecast_evidence('storm', c, 'gust_max_kmh', 'km/h', band(c['gust_max_kmh'], elevated, high), 'Gust max'))
    add('flood', forecast_evidence('flood', c, 'rain_mm', 'mm', band(c['rain_mm'], 10, 25), 'Rainfall'))
    if activity.get('terrain'):
        prior = ((ctx.get('weather') or {}).get('summary') or {}).get('rain_72h_mm')
        wet = None if c['soil_moisture_max'] is None and c['rain_mm'] is None else \
            (c['soil_moisture_max'] or 0) >= .4 or (c['rain_mm'] or 0) >= 10 or (prior or 0) >= 25
        add('unstable_terrain', evidence('forecast:terrain:unavailable', 'unavailable', c['source'], 'Soil moisture and rain unknown for this shift')
            if wet is None else evidence(f'forecast:terrain:{"elevated" if wet else "low"}', 'predicted', c['source'],
                f'Soil moisture max {c["soil_moisture_max"]} m³/m³, shift rain {c["rain_mm"]} mm, prior 72 h {prior} mm',
                rating='elevated' if wet else 'low', at=c['fetched_at']))
    start, end = parse_time(mission['shift_start']), parse_time(mission['shift_end'])
    def reaches_shift(state):
        return bool(state and state.get('active') and parse_time(state['review_after']) > start and ctx['now'] < end)
    live, replay = ctx['lightning'].get('live'), ctx['lightning'].get('replay')
    if reaches_shift(live):
        add('lightning', evidence(f"strike:live:{live['last_strike_at']}", 'observed', 'Recorded strike feed (deterministic 10-mile / 30-minute rule)',
            f"Strike recorded at {utc_clock(live['last_strike_at'])}; review after {utc_clock(live['review_after'])}", rating='stop_work',
            at=live['last_strike_at']))
    if reaches_shift(replay):
        add('lightning', evidence(f"strike:replay:{replay['last_strike_at']}", 'simulated', 'Labeled replay scenario (not a live observation)',
            f"Simulated strike at {utc_clock(replay['last_strike_at'])}; scenario timer runs until {utc_clock(replay['review_after'])}",
            rating='stop_work', simulated=True))
    if ctx['live_feed_configured']:
        if not reaches_shift(live):
            add('lightning', evidence('strike:live:none', 'observed', 'Recorded strike feed',
                'No recorded strike timer reaches this shift. Feed gaps are possible.', rating='low'))
    else:
        add('lightning', evidence('strike:feed:unavailable', 'unavailable', 'Recorded strike feed',
            'No live strike feed configured. Missing records do not mean no lightning.'))
    alerts = ctx['alerts']
    if alerts is None:
        for hazard in ('heat', 'storm', 'flood'):
            add(hazard, evidence('nws:unavailable', 'unavailable', 'NOAA / National Weather Service', 'Alert feed unavailable; advisory status unknown'))
    for a in alerts or []:
        if a.get('expires') and parse_time(a['expires']) <= start:
            continue
        event = (a.get('event') or '').casefold()
        severe = a.get('severity') in ('Extreme', 'Severe')
        for hazard, words in ALERT_HAZARDS:
            if any(w in event for w in words):
                rating = 'elevated' if hazard == 'lightning' or not severe else 'high'
                add(hazard, evidence(f"nws:{a['id']}", 'official_alert', 'NOAA / National Weather Service (issued alert)',
                                     f"{a.get('event')} ({a.get('severity')}) · expires {a.get('expires')}", rating=rating, ref=a['id'], at=a.get('sent')))
    model = ctx['model']
    target = model.get('target_date') if model.get('available') else None
    if target and start.date().isoformat() <= target <= (end-timedelta(seconds=1)).date().isoformat():
        for hazard, p in model['hazards'].items():
            supported = p.get('warning_supported')
            add(hazard, evidence(f"model:{hazard}:{'elevated' if p.get('elevated') else 'below'}", 'predicted',
                f"County event model · {p.get('model')}",
                f"{p['probability']*100:.1f}% county-day probability for {target} UTC (threshold {p['threshold']*100:.1f}%)"
                + ('' if supported else '; warning threshold unsupported'),
                rating=('elevated' if p.get('elevated') else 'low') if supported else None, counts=bool(supported),
                value=round(p['probability'], 4)))
    for r in ctx['reports']:
        if r['status'] not in ('verified', 'pending') or parse_time(r['created_at']) < ctx['now']-timedelta(hours=48):
            continue
        triage = r.get('triage') or {}
        manual = triage.get('manual_categories') or [r['classification']]
        provider = triage.get('provider') or {}
        suggested = provider.get('categories') or [] if provider.get('status') == 'classified' else []
        verified = r['status'] == 'verified'
        for hazard in manual:
            add(hazard, evidence(f"report:{r['id']}:{r['status']}", 'observed' if verified else 'reported',
                'Crew report verified by supervisor' if verified else 'Crew report (pending supervisor review)',
                r['description'][:160], rating='high' if verified else None, counts=verified, ref=r['id'], at=r['created_at'],
                ai_agrees=hazard in suggested))
        for hazard in suggested:
            if hazard not in manual:
                add(hazard, evidence(f"jev:{r['id']}:{hazard}", 'ai_suggestion', f"Jev suggestion ({provider.get('model')}), unverified",
                    f"Jev suggested {hazard} for report {r['id'][:8]}; supervisor labels did not include it", counts=False, ref=r['id']))
    entries = []
    for hazard, items in found.items():
        rated = [e['rating'] for e in items if e['counts'] and e['rating']]
        rating = max(rated, key=RANK.get) if rated else 'unknown'
        simulated = rating == 'stop_work' and all(e.get('simulated') for e in items if e['rating'] == 'stop_work')
        entries.append(dict(hazard=hazard, label=LABELS[hazard], rating=rating, simulated=simulated, evidence=items,
                            incomplete=any(e['kind'] == 'unavailable' for e in items),
                            needs_review=any(e['kind'] in ('reported', 'ai_suggestion') for e in items),
                            sensitivity=activity['sensitivity'].get(hazard, .3), controls=CONTROLS[hazard]))
    entries.sort(key=lambda e: (-RANK[e['rating']], -e['sensitivity']))
    return dict(entries=entries, readiness=readiness(entries), fingerprint=fingerprint(entries))


def readiness(entries):
    stops = [e for e in entries if e['rating'] == 'stop_work']
    unknown = [e['hazard'] for e in entries if e['rating'] == 'unknown']
    if any(not e['simulated'] for e in stops):
        status, headline = 'stop_work', 'STOP WORK: recorded lightning timer reaches this shift.'
    elif stops:
        status, headline = 'simulated_stop_work', 'SIMULATED STOP WORK: labeled replay scenario, not a live observation.'
    else:
        top = max((e['rating'] for e in entries), key=RANK.get, default='unknown')
        status = top
        headline = dict(high='High-rated hazards need a supervisor decision before dispatch.',
                        elevated='Elevated hazards: confirm controls before dispatch.',
                        low='No elevated hazard from available inputs. This is not work clearance.',
                        unknown='Inputs unavailable. Unknown is not safe.')[top]
    return dict(status=status, headline=headline, unknown_hazards=unknown)


def diff_jha(before, after):
    old, new = {e['hazard']: e for e in before}, {e['hazard']: e for e in after}
    def label(item):
        return f"{item['source']}: {item['summary']}"
    changes = []
    for hazard in sorted(set(old) | set(new), key=lambda h: list(LABELS).index(h)):
        a, b = old.get(hazard), new.get(hazard)
        if a is None or b is None:
            changes.append(dict(hazard=hazard, change='added' if a is None else 'removed', before=a and a['rating'], after=b and b['rating'],
                                evidence_added=[label(x) for x in b['evidence']] if b else [], evidence_removed=[label(x) for x in a['evidence']] if a else []))
            continue
        ids_a, ids_b = {x['id'] for x in a['evidence']}, {x['id'] for x in b['evidence']}
        if a['rating'] == b['rating'] and ids_a == ids_b:
            continue
        changes.append(dict(hazard=hazard, change='rating' if a['rating'] != b['rating'] else 'evidence', before=a['rating'], after=b['rating'],
                            evidence_added=[label(x) for x in b['evidence'] if x['id'] not in ids_a],
                            evidence_removed=[label(x) for x in a['evidence'] if x['id'] not in ids_b]))
    return changes


def entry_fingerprint(e):
    return hashlib.sha256(json.dumps([e['hazard'], e['rating'], sorted(x['id'] for x in e['evidence'])]).encode()).hexdigest()[:16]


def fingerprint(entries):
    return hashlib.sha256(json.dumps(sorted(entry_fingerprint(e) for e in entries)).encode()).hexdigest()[:16]


def window_conditions(weather, start, end):
    """Forecast hours inside [start, end). Hours outside the provider horizon stay unavailable."""
    start, end = parse_time(start), parse_time(end)
    requested = max(1, ceil((end-start)/timedelta(hours=1)))
    rows = [r for r in (weather or {}).get('forecast', []) if start <= parse_time(r['time']) < end]
    def values(key):
        return [r[key] for r in rows if r.get(key) is not None]
    def aggregate(key, operation):
        v = values(key)
        return round(operation(v), 2) if v else None
    feed = (weather or {}).get('feed', {})
    return dict(hours_requested=requested, hours_available=len(rows), coverage=round(min(1, len(rows)/requested), 3),
                kind='predicted' if rows else 'unavailable',
                apparent_max_c=aggregate('apparent_temperature', max), gust_max_kmh=aggregate('wind_gusts_10m', max),
                rain_mm=aggregate('precipitation', sum), wet_hours=len([v for v in values('precipitation') if v >= .1]) if rows else None,
                soil_moisture_max=aggregate('soil_moisture_0_to_1cm', max),
                first_hour=rows[0]['time'] if rows else None, last_hour=rows[-1]['time'] if rows else None,
                source='Open-Meteo numerical weather models', fetched_at=feed.get('fetched_at'), feed_available=bool(feed.get('available')),
                note='Model forecast for the facility point, not a site measurement. Hours before now are model estimates.')
