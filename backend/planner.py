"""Deterministic comparison of candidate work windows from the forecast the JHA already cites."""
from datetime import timedelta

from backend.domain import parse_time
from backend.missions import DEFAULT_WIND, band, window_conditions

HEAT = (32.2, 39.4)          # same screening bands the JHA uses, in apparent degrees Celsius
RAIN = (10, 25)              # shift rainfall in millimetres
WEIGHT = dict(low=0, elevated=1, high=2)
DECISIVE = .5                # sensitivity at or above this makes a high band a restriction


def exposure(conditions, activity):
    elevated, high = activity.get('wind', DEFAULT_WIND)
    return dict(heat=band(conditions['apparent_max_c'], *HEAT),
                storm=band(conditions['gust_max_kmh'], elevated, high),
                flood=band(conditions['rain_mm'], *RAIN))


def saturated(conditions, weather):
    """The unstable-ground rule the JHA uses, evaluated for one candidate window."""
    prior = ((weather or {}).get('summary') or {}).get('rain_72h_mm')
    if conditions['soil_moisture_max'] is None and conditions['rain_mm'] is None and prior is None:
        return None
    return ((conditions['soil_moisture_max'] or 0) >= .4 or (conditions['rain_mm'] or 0) >= 10 or (prior or 0) >= 25)


def score(activity, conditions, weather):
    ratings = exposure(conditions, activity)
    wet = saturated(conditions, weather)
    if activity.get('terrain') and wet is not None:
        ratings['unstable_terrain'] = 'elevated' if wet else 'low'
    risk, drivers, unknown, restricted = 0.0, [], [], False
    for hazard, rating in ratings.items():
        weight = activity['sensitivity'].get(hazard, 0)
        if not weight:
            continue
        if rating is None:
            unknown.append(hazard)
            continue
        risk += weight * WEIGHT[rating]
        if rating != 'low':
            drivers.append(f'{hazard} {rating}')
        if rating == 'high' and weight >= DECISIVE:
            restricted = True
    label = 'restricted' if restricted else 'elevated' if risk >= 1 else 'low'
    return dict(risk=round(risk, 2), label=label, drivers=drivers, unknown=sorted(unknown), restricted=restricted, saturated=wet)


def exceedances(ensemble, begin, end, activity):
    """Share of ensemble members crossing the activity limits inside the window."""
    if not ensemble or not ensemble.get('available'):
        return None
    times = [parse_time(t) for t in ensemble['times']]
    index = [i for i, t in enumerate(times) if begin <= t < end]
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
    def share(values, limit):
        return round(sum(1 for v in values if v >= limit)/len(values), 3) if values else None
    def quantile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered)-1, int(q*len(ordered)))], 1)
    gust = extremes('wind_gusts_10m', max)
    heat = extremes('apparent_temperature', max)
    rain = extremes('precipitation', sum)
    return dict(members=max(len(gust), len(heat), len(rain)), model=ensemble.get('model'),
                gust_high_probability=share(gust, high), gust_p90_kmh=quantile(gust, .9), gust_limit_kmh=high,
                heat_high_probability=share(heat, HEAT[1]), heat_p90_c=quantile(heat, .9),
                rain_high_probability=share(rain, RAIN[1]), rain_p90_mm=quantile(rain, .9),
                note='Share of weather-model members crossing the activity limit; spread is not a calibrated risk')


def plan(weather, activity, duration_hours, terrain=None, ensemble=None, now=None, hours=48, step_hours=3, limit=8):
    """Rank candidate windows of one shift length across the forecast horizon."""
    rows = (weather or {}).get('forecast') or []
    if not rows or duration_hours <= 0 or step_hours <= 0:
        return dict(windows=[], recommended=None, recommended_start=None, skipped=0,
                    unavailable='No hourly forecast is available for this facility.')
    start = parse_time(now) if now else parse_time(rows[0]['time'])
    horizon = start + timedelta(hours=hours)
    windows, skipped = [], 0
    begin = start
    while begin + timedelta(hours=duration_hours) <= horizon:
        end = begin + timedelta(hours=duration_hours)
        conditions = window_conditions(weather, begin.isoformat(), end.isoformat())
        if conditions['hours_available'] == 0 or conditions['coverage'] < .5:
            skipped += 1
        else:
            result = score(activity, conditions, weather)
            windows.append(dict(start=begin.isoformat(), end=end.isoformat(), hours_available=conditions['hours_available'],
                coverage=conditions['coverage'], apparent_max_c=conditions['apparent_max_c'],
                gust_max_kmh=conditions['gust_max_kmh'], rain_mm=conditions['rain_mm'], wet_hours=conditions['wet_hours'],
                soil_moisture_max=conditions['soil_moisture_max'], soil_deep_max=conditions.get('soil_deep_max'),
                uncertainty=exceedances(ensemble, begin, end, activity), **result))
        begin += timedelta(hours=step_hours)
    ordered = sorted(windows, key=lambda w: (w['restricted'], w['risk'],
        (w['uncertainty'] or {}).get('gust_high_probability') if w['uncertainty'] else 0, w['start']))
    best = next((w for w in ordered if not w['restricted']), None)
    return dict(windows=ordered[:limit], recommended=ordered.index(best) if best else None,
                recommended_start=best['start'] if best else None, skipped=skipped, candidate_count=len(windows),
                horizon_hours=hours, duration_hours=duration_hours, step_hours=step_hours,
                terrain=terrain or dict(available=False),
                notes=['Windows come from the same single-run forecast the JHA cites; ensemble spread is a separate member sample.',
                       'Coverage below half the window hours is skipped rather than scored.',
                       'Bands and controls are the deterministic ones from the Job Hazard Analysis; this is screening, not a work authorization.'])
