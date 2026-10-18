# Work-window planner

The planner answers two questions for a planned shift: is there a better window inside the forecast horizon, and is the evidence strong enough to move. It is decision support. It does not authorize work, and it never changes a rating; the deterministic rules in the Job Hazard Analysis stay authoritative.

## What it reads

The same Open-Meteo hourly forecast the JHA cites, the NOAA GFS ensemble member sample, the five-point 3DEP terrain sample, the mission's own activity and shift length, the site's timezone, and the other open missions at the same site.

## Exposure model

Bands are the JHA's screening bands, unchanged: apparent temperature 32.2 and 39.4 degrees Celsius, shift rainfall 10 and 25 millimetres, and the activity's gust limits (45 and 72 km/h by default). Ranking uses a continuous severity instead of band counts, with the lower anchor one band below the elevated threshold, so 0.5 is the elevated band and 1.0 the high band. Severity is weighted by the activity's per-hazard sensitivity and summed. This removes the tie behaviour of integer band sums, where equal-looking windows used to order themselves by clock time.

Unstable ground uses the JHA rule for terrain-sensitive activities: 0-1 cm soil moisture at or above 0.4, shift rain at or above 10 mm, or 72 hour rain at or above 25 mm. The terrain sample appears in the response as context and never sets a rating.

## Constraints applied before ranking

- A candidate never starts in the past.
- A candidate's local clock hour, in the site's timezone, stays within the requested flexibility (default two hours) of the planned start.
- A candidate never overlaps another open mission at the same site.
- A candidate is skipped when more than a quarter of its hours are outside the forecast.

Each exclusion is counted in the response, so a supervisor can see why a window is absent rather than guessing.

## The decision rule

The planner returns one of four outcomes and the reason for it:

1. `insufficient`: no candidate has enough coverage, the planned shift is outside the horizon, or exposure for a weighted hazard is unknown. Unknown exposure blocks a recommendation rather than hiding inside one.
2. `keep` with every candidate restricted, so no move is advised on this horizon.
3. `keep` when nothing material is on offer: a move requires at least one weighted hazard whose band improves by a whole level and whose severity moves by at least a quarter band, with no weighted hazard getting worse. A near-tie (the test suite uses a 0.018 weighted difference) declines.
4. `move` when the planned shift carries a restriction, or when the material improvement holds and the weather-model members do not object.

Members can veto but never promote. The veto fires when the members put more restriction crossings on the recommended window than on the planned one by more than 0.05 of members. When no member sample is available the response says so, the veto cannot be applied, and the advice rests on the single deterministic run.

## Uncertainty language

The member share is the fraction of ensemble members crossing a screening limit for this activity. It is model spread, not a calibrated probability, and the API says so in every response. Lower exposure means a lower screening score, not safety, and the response repeats that. Lightning is not in the ensemble product, so lightning risk is not part of the comparison.

## Measured behaviour

## Recording a decision

`POST /api/missions/{id}/window-decisions` records what a supervisor decided, not what the tool advised. The client echoes the snapshot digest of the comparison it rendered; the server recomputes the comparison and answers 409 when the forecast moved, so a record can never be attached to a view that no longer exists. A move must name a window the current comparison listed and that window must not be restricted; anything else is 422 or 409. The stored record holds the actor, the token identity, the choice, the window, an optional note bounded to 280 characters, the snapshot digest and the JHA version current at the time, and the same transaction writes an audit row. The mission detail returns the recorded decisions and the planner marks the latest one stale when the forecast has since changed. The tool never reschedules the crew; the record states what the comparison showed.

`scripts/evaluate_planner.py` runs eight synthetic scenarios across heat, rain, wind, flat and degenerate conditions, checks the decision rule, and compares the chosen window against three simple baselines: keep the plan, take the earliest usable window, take the same hour the next day. Current results, recorded in `experiments/sprint-20261017-planner/evaluation.json`:

| Metric | Value |
| --- | ---: |
| Scenarios matching their expected outcome | 9 of 9 |
| Constraint violations | 0 |
| Moves recommended | 4 |
| Mean weighted improvement on a move | .787 |
| Moves at or better than the best simple baseline | 4 of 4 |
| Declines (immaterial, all restricted) | 3 |
| Insufficient-evidence outcomes | 2 |
| Recommendations unchanged after a +1 degree, +5 percent gust refresh | 9 of 9 |

The scenarios are synthetic weather on a fixed grid. They test the decision rule, its constraints and its stability under an ordinary forecast update, not forecast skill; nothing here claims the planner improves real outcomes, only that it declines when the evidence is thin, holds its recommendation when the forecast moves by a plausible amount, and clears the stated bars against simple baselines.

## Limits

Work hours are constrained only by the planned start hour and the flexibility parameter; there is no calendar, crew identity, union rule or daylight model, and no route, mud or access model. Coverage comes from one forecast product and one ensemble family. The bands are screening thresholds, not validated work limits. API failures degrade the response to `unavailable` fields instead of substituting data.
