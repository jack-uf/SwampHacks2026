"""Pure, deterministic safety rules; independent of trained models and transports."""
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, asin, sqrt
from zoneinfo import ZoneInfo

UTC = timezone.utc


def now_utc():
    return datetime.now(UTC)


def parse_time(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone-aware timestamp required')
    return result.astimezone(UTC)


def distance_miles(lat1, lon1, lat2, lon2):
    a, b, c, d = map(radians, [lat1, lon1, lat2, lon2])
    return 3958.7613 * 2 * asin(min(1, sqrt(sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2)))


def lightning_state(strikes, site, now=None):
    now = now or now_utc()
    qualifying = [parse_time(s['occurred_at']) for s in strikes
                  if distance_miles(site['latitude'], site['longitude'], s['latitude'], s['longitude']) <= 10
                  and parse_time(s['occurred_at']) <= now]
    last = max(qualifying) if qualifying else None
    until = last + timedelta(minutes=30) if last else None
    remaining = max(0, int((until-now).total_seconds())) if until else 0
    return dict(active=bool(until and until > now), last_strike_at=last.isoformat() if last else None,
                review_after=until.isoformat() if until else None, remaining_seconds=remaining,
                instruction='STOP WORK — seek established lightning shelter. Timer expiry requires supervisor review.' if remaining else
                            'No active recorded-strike timer. Missing feed does not mean no lightning.')


def briefing_due(site, now):
    local = now.astimezone(ZoneInfo(site['timezone']))
    return local.hour >= 18, local.date().isoformat(), (local.date()+timedelta(days=1)).isoformat()


def hazard_index(weather, alerts, reports, predictions, exposure):
    """Fixed weights, no renormalizing missing inputs into an apparently complete score."""
    components = []
    def add(name, value, weight, detail):
        normalized = None if value is None else max(0, min(1, value))
        components.append(dict(name=name, severity=normalized, weight=weight,
                               points=None if normalized is None else round(10*weight*normalized, 2), detail=detail))
    d = weather.get('summary', {}) if weather else {}
    add('Heat screening', None if d.get('apparent_max_c') is None else (d['apparent_max_c']-27)/18, .15,
        'Forecast apparent temperature; not occupational WBGT.')
    rain = d.get('rain_24h_mm')
    previous = d.get('rain_72h_mm')
    soil = d.get('soil_moisture')
    flood = None if any(v is None for v in (rain, soil, previous)) else max(rain/75, (soil/.45)*(rain/50), (previous/150)*(rain/50))
    add('Rain & saturation', flood, .20, 'Rain screening, not hydraulic inundation modeling; Static FEMA exposure is a separate component.')
    add('Wind & storms', None if d.get('gust_max_kmh') is None else (d['gust_max_kmh']-25)/65, .15,
        'Maximum next-24h forecast gust; consult active warnings.')
    valid = [predictions[h]['probability'] for h in ('flood', 'storm') if h in predictions and predictions[h].get('in_domain') and predictions[h].get('probability') is not None]
    add('County event model', max(valid) if len(valid)==2 else None, .20,
        'Maximum calibrated flood/storm county-event estimate for tomorrow UTC. Both estimates must be supported; an absent hazard keeps this combined component unknown.')
    add('Active NWS alerts', None if alerts is None else max([{'Extreme':1, 'Severe':.8, 'Moderate':.5, 'Minor':.2}.get(a.get('severity'),.3) for a in alerts] or [0]),
        .20, 'Point-matched NWS alerts. An unavailable alert feed is unknown.')
    zones = exposure.get('zones', []) if exposure and exposure.get('feed', {}).get('available') else []
    mapped = [z['SFHA_TF'] for z in zones if z.get('SFHA_TF') in ('T', 'F')]
    add('Mapped flood exposure', (1 if 'T' in mapped else .2) if mapped else None, .05,
        'FEMA special flood hazard area: mapped exposure, not a flood prediction. Other mapped zones retain residual exposure.')
    add('Reviewed crew reports', min(1, len([r for r in reports if r['status']=='verified'])/3), .05,
        'Verified, unresolved reports from the last 48 hours only.')
    known = [c for c in components if c['points'] is not None]
    points = round(sum(c['points'] for c in known), 1) if known else None
    coverage = round(sum(c['weight'] for c in known), 2)
    return dict(score=points, coverage=coverage, complete=coverage >= .999, components=components,
                label='Partial screening index' if coverage < .999 else 'Operational screening index',
                interpretation='0–10 weighted environmental severity, not injury probability or work clearance.')
