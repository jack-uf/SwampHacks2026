"""Mission shift windows and the deterministic, evidence-cited Job Hazard Analysis."""
import hashlib
import json
from datetime import timedelta, timezone
from math import ceil, pi, sqrt
from backend.domain import distance_miles, parse_time


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


GAUGE_SOURCE = 'NOAA National Water Prediction Service river gauge'
GAUGE_MILES, GAUGE_MAX_AGE = 10, timedelta(hours=6)
FLOOD_CATEGORY = dict(no_flooding='low', action='elevated', minor='high', moderate='high', major='high')
CATEGORY_LABEL = dict(no_flooding='no flooding', action='action stage', minor='minor flooding', moderate='moderate flooding',
                      major='major flooding')
FIRE_SOURCE = 'NIFC WFIGS current wildfire incidents (agency-reported)'
FIRE_HIGH_MILES, FIRE_ELEVATED_MILES, FIRE_MAX_AGE = 10, 25, timedelta(days=14)


def gauge_evidence(point, feed, now):
    if not feed or feed.get('gauges') is None:
        return [evidence('gauge:feed:unavailable', 'unavailable', GAUGE_SOURCE, 'River gauge feed unavailable; river stage unknown')]
    near = sorted(((distance_miles(*point, g['latitude'], g['longitude']), g) for g in feed['gauges'] if g.get('lid')),
                  key=lambda pair: (pair[0], pair[1]['lid']))
    near = [(d, g) for d, g in near if d <= GAUGE_MILES][:3]
    if not near:
        return [evidence('gauge:none', 'unavailable', GAUGE_SOURCE,
                         f'No NWS river gauge within {GAUGE_MILES} miles of the work location; river stage is unmonitored, not safe')]
    items = []
    for d, g in near:
        where = f"{g['name']} ({g['lid']}), {d:.1f} miles away"
        o, f = g['observed'], g['forecast']
        rating = FLOOD_CATEGORY.get(o['category'])
        if rating is None:
            why = 'no flood thresholds apply' if o['category'] == 'not_defined' else 'not rated'
            reading = (f"stage {o['stage']:g} {o['unit']} at {utc_clock(o['valid_time'])}, but NWS reports {o['category']}: {why}"
                       if o['stage'] is not None and o['valid_time']
                       else f"no current observation ({o['category'] or 'no category'})")
            items.append(evidence(f"gauge:{g['lid']}:observed:{o['category'] or 'unknown'}", 'unavailable', GAUGE_SOURCE,
                                  f'{where}: {reading}'))
        elif not o['valid_time'] or now-parse_time(o['valid_time']) > GAUGE_MAX_AGE:
            items.append(evidence(f"gauge:{g['lid']}:observed:stale", 'unavailable', GAUGE_SOURCE,
                                  f"{where}: last observation {o['valid_time'] or 'unknown'} is older than 6 hours"))
        else:
            items.append(evidence(f"gauge:{g['lid']}:observed:{o['category']}", 'observed', GAUGE_SOURCE,
                                  f"{where}: {CATEGORY_LABEL[o['category']]} at {o['stage']:g} {o['unit']}, observed {utc_clock(o['valid_time'])}",
                                  rating=rating, ref=g['lid'], at=o['valid_time']))
        forecast = FLOOD_CATEGORY.get(f['category'])
        if forecast and f['valid_time']:
            items.append(evidence(f"gauge:{g['lid']}:forecast:{f['category']}", 'predicted', 'NWS river forecast at ' + g['lid'],
                                  f"{where}: forecast {CATEGORY_LABEL[f['category']]}, {f['stage']:g} {f['unit']} at {f['valid_time'][:16]} UTC",
                                  rating=forecast, ref=g['lid']))
    return items


def wildfire_evidence(point, feed, now):
    if not feed or feed.get('incidents') is None or not feed.get('complete'):
        return [evidence('wfigs:unavailable', 'unavailable', FIRE_SOURCE,
                         'Wildfire incident feed unavailable or cut off at its record limit; nearby fires unknown')]
    rows = []
    for f in feed['incidents']:
        center = distance_miles(*point, f['latitude'], f['longitude'])
        radius = sqrt((f['acres'] or 0)/640/pi)
        current = bool(f['updated_at']) and now-parse_time(f['updated_at']) <= FIRE_MAX_AGE
        rows.append((max(0., center-radius), center, current, f))
    rows.sort(key=lambda r: r[0])
    items = []
    for edge, center, current, f in [r for r in rows if r[0] <= FIRE_ELEVATED_MILES][:3]:
        size = f"{f['acres']:g} acres" if f['acres'] else 'size not reported'
        contained = f", {f['contained_pct']:g}% contained" if f['contained_pct'] is not None else ''
        where = f"origin {center:.1f} miles away" + (f", edge about {edge:.1f} miles if the fire were a circle of its reported size" if f['acres'] else '')
        if current:
            rating = 'high' if edge < FIRE_HIGH_MILES else 'elevated'
            if (f['contained_pct'] or 0) >= 100:
                rating, contained = ('elevated' if rating == 'high' else 'low'), ', fully contained (rated one band lower)'
            items.append(evidence(f"wfigs:{f['id']}:{rating}", 'observed', FIRE_SOURCE,
                                  f"{f['name']}: {size}{contained}, {where}; updated {f['updated_at'][:10]}", rating=rating, ref=f['id'],
                                  at=f['updated_at']))
        else:
            items.append(evidence(f"wfigs:{f['id']}:stale", 'observed', FIRE_SOURCE,
                                  f"{f['name']}: {where}; record not updated since {(f['updated_at'] or 'unknown')[:10]}, may be out of date",
                                  counts=False, ref=f['id']))
    if not any(e['counts'] for e in items):
        fresh = [center for _, center, current, _ in rows if current]
        nearest = f'; nearest current record {min(fresh):.0f} miles away' if fresh else ''
        items.append(evidence(f'wfigs:none-within-{FIRE_ELEVATED_MILES}mi', 'observed', FIRE_SOURCE,
                              f'No current wildfire incident within {FIRE_ELEVATED_MILES} miles{nearest}. New or unreported fires can be missing.',
                              rating='low', at=(feed.get('feed') or {}).get('fetched_at')))
    return items


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
    point = (mission['latitude'], mission['longitude'])
    for item in gauge_evidence(point, ctx.get('gauges'), ctx['now']):
        add('flood', item)
    from backend.hydrology import jha_evidence
    for item in jha_evidence(ctx.get('hydrology')):
        add('flood', item)
    for item in wildfire_evidence(point, ctx.get('wildfires'), ctx['now']):
        add('wildfire', item)
    if activity.get('terrain'):
        prior = ((ctx.get('weather') or {}).get('summary') or {}).get('rain_72h_mm')
        wet = None if c['soil_moisture_max'] is None and c['rain_mm'] is None else \
            (c['soil_moisture_max'] or 0) >= .4 or (c['rain_mm'] or 0) >= 10 or (prior or 0) >= 25
        add('unstable_terrain', evidence('forecast:terrain:unavailable', 'unavailable', c['source'], 'Soil moisture and rain unknown for this shift')
            if wet is None else evidence(f'forecast:terrain:{"elevated" if wet else "low"}', 'predicted', c['source'],
                f'Soil moisture max {c["soil_moisture_max"]} m³/m³, shift rain {c["rain_mm"]} mm, prior 72 h {prior} mm',
                rating='elevated' if wet else 'low', at=c['fetched_at']))
        terrain = ctx.get('terrain') or {}
        if terrain.get('available'):
            add('unstable_terrain', evidence('terrain:relief', 'predicted', terrain['source'],
                f"Sampled relief {terrain['relief_m']} m and {terrain['slope_pct']}% gradient across about one kilometre; "
                f"deep soil moisture (0-100 cm) max {c.get('soil_deep_max')} m\u00b3/m\u00b3. "
                f"{terrain['note']}.", counts=False))
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
        until = a.get('ends') or a.get('expires')
        if until and parse_time(until) <= start or a.get('onset') and parse_time(a['onset']) >= end:
            continue
        event = (a.get('event') or '').casefold()
        severe = a.get('severity') in ('Extreme', 'Severe')
        hazards = [h for h, words in ALERT_HAZARDS if any(w in event for w in words) and not (h == 'storm' and 'chill' in event)]
        for hazard in hazards or ['other']:
            rating = 'elevated' if hazard == 'lightning' or not severe else 'high'
            add(hazard, evidence(f"nws:{a['id']}", 'official_alert', 'NOAA / National Weather Service (issued alert)',
                                 f"{a.get('event')} ({a.get('severity')}) · in effect until {until}", rating=rating, ref=a['id'], at=a.get('sent')))
    model = ctx['model']
    target = model.get('target_date') if model.get('available') else None
    if target and start.date().isoformat() <= target <= (end-timedelta(seconds=1)).date().isoformat():
        for hazard, p in model['hazards'].items():
            supported = p.get('warning_supported')
            tested = p.get('tested')
            add(hazard, evidence(f"model:{hazard}:{'elevated' if p.get('elevated') else 'below'}", 'predicted',
                f"County event model · {p.get('model')}",
                f"{p['probability']*100:.1f}% county-day probability for {target} UTC (threshold {p['threshold']*100:.1f}%)"
                + ('' if supported else '; warning threshold unsupported')
                + (f". In the {tested['period']} test, about {round(tested['precision']*100)} of every 100 flagged county-days had a "
                   f"NOAA-reported {hazard} and the flag caught {round(tested['recall']*100)}% of reported {hazard} days" if tested else ''),
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
                soil_deep_max=aggregate('soil_moisture_0_to_100cm', max),
                first_hour=rows[0]['time'] if rows else None, last_hour=rows[-1]['time'] if rows else None,
                source='Open-Meteo numerical weather models', fetched_at=feed.get('fetched_at'), feed_available=bool(feed.get('available')),
                note='Model forecast for the facility point, not a site measurement. Hours before now are model estimates.')
