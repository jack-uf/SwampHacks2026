"""Work-window comparison and the decision it supports, from the forecast the JHA cites.

Four rules keep the output defensible. Exposure is continuous so equal-looking windows do not
order themselves by clock time. A schedule change needs a hazard this activity weights to move a
whole band, and no weighted hazard may get worse. Weather-model members can veto but cannot
promote: they block a move when they put more crossings on the recommended window than on the
planned one. Unknown exposure and thin coverage block recommendations instead of hiding in them.
Nothing here is a work authorization: lower exposure is a lower screening score, never safety.
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

from backend.domain import parse_time
from backend.missions import DEFAULT_WIND, band, window_conditions

HEAT = (32.2, 39.4)          # the JHA screening bands, in apparent degrees Celsius
RAIN = (10, 25)              # shift rainfall in millimetres
DECISIVE = .5                # sensitivity at or above this makes a hazard able to restrict work
TOLERANCE = .125             # a quarter of a band step: differences below this are noise
BAND_STEP = .5               # severity units from one band to the next
MIN_BAND_GAIN = .25          # a band change must also move severity by at least this much
COVERAGE_FLOOR = .75         # windows with more unknown hours than this cannot drive a decision
MEMBER_VETO = .05            # member crossing share the recommended window may exceed by this

RANKS = dict(low=0, elevated=1, high=2)
NOTES = ['Windows come from the same single-run forecast the JHA cites; the member sample is a separate product, not a calibrated probability.',
         'Windows with more than a quarter of their hours unforecast are skipped rather than scored.',
         'Candidate starts keep the planned local clock hour within the allowed flexibility and never overlap another mission at the same site.',
         'A move needs one weighted hazard to improve by a whole band step and no weighted hazard to worsen; members can veto it.',
         'Bands, wind limits and controls are the deterministic ones from the Job Hazard Analysis. Lower exposure is not guaranteed safety.']


def severity(value, elevated, high):
    """Continuous exposure where .5 is the elevated band and 1.0 the high band, anchored one band below."""
    if value is None:
        return None
    lower = elevated - (high - elevated)
    if high <= elevated:
        raise ValueError('Invalid bands')
    return round(max(0., min(1., (value - lower) / (high - lower))), 4)


def saturated(conditions, weather):
    """The unstable-ground rule the JHA uses, evaluated for one candidate window."""
    prior = ((weather or {}).get('summary') or {}).get('rain_72h_mm')
    if conditions['soil_moisture_max'] is None and conditions['rain_mm'] is None and prior is None:
        return None
    return ((conditions['soil_moisture_max'] or 0) >= .4 or (conditions['rain_mm'] or 0) >= 10 or (prior or 0) >= 25)


def score(activity, conditions, weather):
    """Per-hazard bands for display, continuous levels for ranking, and the weighted exposure."""
    elevated, high = activity.get('wind', DEFAULT_WIND)
    ratings = dict(heat=band(conditions['apparent_max_c'], *HEAT),
                   storm=band(conditions['gust_max_kmh'], elevated, high),
                   flood=band(conditions['rain_mm'], *RAIN))
    levels = dict(heat=severity(conditions['apparent_max_c'], *HEAT),
                  storm=severity(conditions['gust_max_kmh'], elevated, high),
                  flood=severity(conditions['rain_mm'], *RAIN))
    wet = saturated(conditions, weather)
    if activity.get('terrain') and wet is not None:
        ratings['unstable_terrain'] = 'elevated' if wet else 'low'
        levels['unstable_terrain'] = BAND_STEP if wet else 0.
    risk, drivers, unknown, restricted = 0., [], [], False
    for hazard, rating in ratings.items():
        weight = activity['sensitivity'].get(hazard, 0)
        if not weight:
            continue
        if rating is None or levels.get(hazard) is None:
            unknown.append(hazard)
            continue
        risk += weight * levels[hazard]
        if rating != 'low':
            drivers.append(f'{hazard} {rating}')
        if rating == 'high' and weight >= DECISIVE:
            restricted = True
    label = 'restricted' if restricted else 'elevated' if drivers else 'low'
    return dict(risk=round(risk, 3), label=label, drivers=drivers, unknown=sorted(unknown), restricted=restricted,
                saturated=wet, ratings={h: r for h, r in ratings.items() if activity['sensitivity'].get(h, 0)},
                levels={h: v for h, v in levels.items() if activity['sensitivity'].get(h, 0) and v is not None})


def exceedances(ensemble, begin, end, activity):
    """Member statistics for one window: limit crossings and the share that would restrict work."""
    if not ensemble or not ensemble.get('available'):
        return None
    times = []
    for stamp in ensemble['times']:
        try:
            times.append(parse_time(stamp))
        except ValueError:
            times.append(None)  # a malformed member timestamp is not evidence about this window
    index = [i for i, t in enumerate(times) if t is not None and begin <= t < end]
    if not index:
        return None
    elevated, high = activity.get('wind', DEFAULT_WIND)
    def extremes(variable, operation):
        values = []
        for column in ensemble['members'].get(variable, []):
            points = [column[i] for i in index if i < len(column) and column[i] is not None]
            if points:
                values.append(operation(points))
        return values
    def share(mask, values):
        return round(sum(1 for keep in mask if keep)/len(values), 3) if values else None
    def quantile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered)-1, int(q*len(ordered)))], 1)
    gust = extremes('wind_gusts_10m', max)
    heat = extremes('apparent_temperature', max)
    rain = extremes('precipitation', sum)
    decisive_heat = activity['sensitivity'].get('heat', 0) >= DECISIVE
    decisive_storm = activity['sensitivity'].get('storm', 0) >= DECISIVE
    decisive_flood = activity['sensitivity'].get('flood', 0) >= DECISIVE
    members = max(len(gust), len(heat), len(rain))
    crossings = [bool(decisive_storm and position < len(gust) and gust[position] >= high)
                 or bool(decisive_heat and position < len(heat) and heat[position] >= HEAT[1])
                 or bool(decisive_flood and position < len(rain) and rain[position] >= RAIN[1])
                 for position in range(members)]
    return dict(members=members, model=ensemble.get('model'),
                restrict_share=round(sum(crossings)/len(crossings), 3) if crossings else None,
                gust_high_probability=share([value >= high for value in gust], gust), gust_p90_kmh=quantile(gust, .9),
                gust_limit_kmh=high, heat_high_probability=share([value >= HEAT[1] for value in heat], heat),
                heat_p90_c=quantile(heat, .9), rain_high_probability=share([value >= RAIN[1] for value in rain], rain),
                rain_p90_mm=quantile(rain, .9),
                note='Member share crossing this activity screening limits; spread is not a calibrated risk')


def local_hour(when, timezone):
    local = parse_time(when.isoformat() if hasattr(when, 'isoformat') else when).astimezone(ZoneInfo(timezone))
    return local.hour + local.minute / 60


def hour_distance(left, right):
    distance = abs(left - right) % 24
    return min(distance, 24 - distance)


def model_context(model, begin, end):
    """County-day model probabilities for the days a window covers, copied as context only.

    The served models predict a reported hazard for a county-day, not an exposure for a shift.
    Nothing here feeds the risk score, the restriction flag or the recommendation.
    """
    if not model or not model.get('available') or not model.get('target_date'):
        return None
    if not (begin.date().isoformat() <= model['target_date'] <= end.date().isoformat()):
        return None
    hazards = {}
    for hazard, value in sorted((model.get('hazards') or {}).items()):
        hazards[hazard] = dict(probability=value.get('probability'), threshold=value.get('threshold'),
                               elevated=value.get('elevated'), warning_supported=value.get('warning_supported'),
                               model=value.get('model'), tested=value.get('tested'))
    return dict(target_date=model['target_date'], granularity='county-day, not a shift window',
                note='Disclosed for the supervisor. This never sets a rating or changes a window recommendation.',
                hazards=hazards)


def material(planned, best, decisive):
    """One weighted hazard must improve a whole band while none of them gets meaningfully worse."""
    for hazard in decisive:
        before, after = planned['levels'].get(hazard), best['levels'].get(hazard)
        if before is None or after is None:
            continue
        if after > before + TOLERANCE:
            return False, f'{hazard} would get worse in the alternative'
    for hazard in decisive:
        before_band = RANKS.get(planned['ratings'].get(hazard), 0)
        after_band = RANKS.get(best['ratings'].get(hazard), 0)
        before, after = planned['levels'].get(hazard), best['levels'].get(hazard)
        if before is None or after is None or before_band < 1:
            continue
        if (before_band - after_band >= 1 and before - after >= MIN_BAND_GAIN) or before - after >= BAND_STEP:
            return True, f'{hazard} improves from {planned["ratings"].get(hazard)} to {best["ratings"].get(hazard)}'
    return False, 'no weighted hazard improves by a whole band'


def decide(windows, planned, decisive, ensemble_available, tolerance=TOLERANCE, member_veto=MEMBER_VETO):
    """Recommend a change only when materiality, no-trade and member checks all pass."""
    if not windows:
        return dict(choice='insufficient', reason='No candidate window has enough forecast coverage to compare.',
                    start=None, improvement=None, material=None, member_veto=None, near_ties=0)
    usable = [w for w in windows if not w['restricted']]
    if not usable:
        return dict(choice='keep', reason='Every candidate window in this horizon carries a restriction; no move is advised.',
                    start=None, improvement=None, material=None, member_veto=None, near_ties=0)
    best = min(usable, key=lambda w: (w['risk'], (w['uncertainty'] or {}).get('restrict_share') if
        (w['uncertainty'] or {}).get('restrict_share') is not None else 1, w['start']))
    if planned is None:
        return dict(choice='insufficient', start=None, improvement=None, material=None, member_veto=None, near_ties=0,
                    reason='The planned shift is outside the compared horizon, so there is nothing to compare it against.')
    if planned['unknown']:
        return dict(choice='insufficient', start=None, improvement=None, material=None, member_veto=None, near_ties=0,
                    reason='Exposure is unknown for ' + ', '.join(planned['unknown']) +
                           ' in the planned shift, so no alternative can be shown to be better.')
    improvement = round(planned['risk'] - best['risk'], 3)
    near_ties = len([w for w in usable if w['start'] != best['start'] and w['risk'] <= best['risk'] + tolerance])
    if planned['restricted']:
        return dict(choice='move', start=best['start'], improvement=improvement, material=True, member_veto=None,
                    near_ties=near_ties, detail='the planned shift carries a restriction',
                    reason='The planned shift carries a restriction. This alternative is the lowest-exposure window that does not.')
    ok, why = material(planned, best, decisive)
    if not ok:
        return dict(choice='keep', start=None, improvement=improvement, material=False, member_veto=None,
                    near_ties=near_ties, detail=why, reason=f'Moving is not supported: {why}.')
    planned_share = (planned['uncertainty'] or {}).get('restrict_share')
    best_share = (best['uncertainty'] or {}).get('restrict_share')
    if ensemble_available and planned_share is not None and best_share is not None and best_share > planned_share + member_veto:
        return dict(choice='keep', start=None, improvement=improvement, material=True, member_veto=True,
                    near_ties=near_ties, detail='members put more restriction on the alternative',
                    reason='The weather-model members put more restriction on the alternative than on the plan, so no move is advised.')
    return dict(choice='move', start=best['start'], improvement=improvement, material=True,
                member_veto=False if ensemble_available and best_share is not None and planned_share is not None else None,
                near_ties=near_ties, detail=why,
                reason='One weighted hazard improves a whole band, nothing weighed gets worse, and the members do not object. '
                       'Lower exposure is not guaranteed safety.')


def plan(weather, activity, duration_hours, terrain=None, ensemble=None, now=None, hours=48, step_hours=1, limit=8,
         planned_start=None, timezone=None, flex_hours=None, busy=(), model=None, tolerance=TOLERANCE,
         member_veto=MEMBER_VETO):
    """Compare candidate windows of one shift length and decide whether any of them is worth a change."""
    rows = (weather or {}).get('forecast') or []
    base = dict(windows=[], planned=None, candidate_count=0, skipped=0, horizon_hours=hours, duration_hours=duration_hours,
                step_hours=step_hours, terrain=terrain or dict(available=False),
                excluded=dict(past=0, hour_band=0, overlap=0, coverage=0),
                constraints=dict(timezone=timezone, flex_hours=flex_hours, planned_local_hour=None, busy_intervals=len(busy)),
                recommendation=dict(choice='insufficient', reason='No hourly forecast is available for this facility.',
                                    start=None, improvement=None, material=None, member_veto=None, near_ties=0),
                notes=NOTES)
    if not rows or duration_hours <= 0 or step_hours <= 0 or hours < duration_hours:
        return base
    start = parse_time(now) if now else parse_time(rows[0]['time'])
    horizon = start + timedelta(hours=hours)
    busy = [(parse_time(a), parse_time(b)) for a, b in busy]
    planned_local = local_hour(planned_start, timezone) if planned_start and timezone else None
    base['constraints']['planned_local_hour'] = round(planned_local, 2) if planned_local is not None else None
    decisive = sorted(h for h, weight in activity['sensitivity'].items() if weight >= DECISIVE)
    windows, planned = [], None
    # Stable UTC candidate hours keep routine refreshes comparable; include the exact plan too.
    begin = start.replace(minute=0, second=0, microsecond=0)
    if begin < start:
        begin += timedelta(hours=1)
    starts = []
    while begin + timedelta(hours=duration_hours) <= horizon:
        starts.append(begin)
        begin += timedelta(hours=step_hours)
    if planned_start:
        exact = parse_time(planned_start)
        if start <= exact and exact + timedelta(hours=duration_hours) <= horizon and exact not in starts:
            starts.append(exact)
    for begin in sorted(starts):
        end = begin + timedelta(hours=duration_hours)
        if begin < start - timedelta(seconds=1):
            base['excluded']['past'] += 1
        elif flex_hours is not None and planned_local is not None and timezone and \
                hour_distance(local_hour(begin, timezone), planned_local) > flex_hours + 1e-9:
            base['excluded']['hour_band'] += 1
        elif any(begin < busy_end and busy_start < end for busy_start, busy_end in busy):
            base['excluded']['overlap'] += 1
        else:
            conditions = window_conditions(weather, begin.isoformat(), end.isoformat())
            if conditions['hours_available'] == 0 or conditions['coverage'] < COVERAGE_FLOOR:
                base['excluded']['coverage'] += 1
            else:
                window = dict(start=begin.isoformat(), end=end.isoformat(), hours_available=conditions['hours_available'],
                    model=model_context(model, begin, end),
                    coverage=conditions['coverage'], apparent_max_c=conditions['apparent_max_c'],
                    gust_max_kmh=conditions['gust_max_kmh'], rain_mm=conditions['rain_mm'], wet_hours=conditions['wet_hours'],
                    soil_moisture_max=conditions['soil_moisture_max'], soil_deep_max=conditions.get('soil_deep_max'),
                    uncertainty=exceedances(ensemble, begin, end, activity), **score(activity, conditions, weather))
                windows.append(window)
                if planned_start and begin == parse_time(planned_start):
                    planned = window
    ordered = sorted(windows, key=lambda w: (w['restricted'], w['risk'],
        (w['uncertainty'] or {}).get('restrict_share') if (w['uncertainty'] or {}).get('restrict_share') is not None else 1,
        w['start']))
    return dict(base, windows=ordered[:limit], planned=planned, candidate_count=len(windows),
                skipped=base['excluded']['coverage'],
                recommendation=decide(windows, planned, decisive, bool(ensemble and ensemble.get('available')),
                                      tolerance=tolerance, member_veto=member_veto))
