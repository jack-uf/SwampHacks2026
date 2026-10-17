# Audit, priorities and this round's changes

This note records what the repository actually contains, how the four suggested opportunities survive contact with the code, and the two capabilities implemented after that review. It is written against the tree at the commit that adds it, not against intentions.

## What is implemented today

The application is a FastAPI service (`backend/`) with a React and Leaflet frontend (`frontend/src/`) over a SQLite store. Sites are real power-plant locations (`data/sites.json`). Per site the service caches Open-Meteo hourly weather, NWS active alerts, NWPS river gauges within ten miles, WFIGS wildfire incidents, and the FEMA floodplain zone, then exposes an assessment with a hazard index and a coverage number. Supervisor-only flows create missions, crew reports pass through Jev triage, verified reports enter the Job Hazard Analysis as observed evidence, and every JHA change is a stored version with a diff, a trigger and an append-only audit row. Lightning runs on a deterministic ten-mile, thirty-minute rule with a labelled replay path. A separate view compares public generator construction plans across owners.

The served machine learning is two county-day classifiers, flood (balanced logistic) and storm (gradient boosting), trained on 21,336 county-days across 24 counties with temporal and geographic partitions, leave-one-state-out spatial cross-validation in `models/spatial-cv.json`, and a JHA evidence line that states the probability, the threshold, the test precision and recall, and flags a threshold the test set does not support. Nine research experiments sit under `experiments/` and are deliberately outside serving; the repository passes 235 tests.

Two properties are worth protecting because they are the project's real edge: every line of analysis carries an evidence kind, and a missing feed is `unavailable`, never `low`. No weather dashboard does that.

## The four opportunities, assessed

**Site-specific intelligence: mostly present, terrain was the hole.** The shift conditions already aggregate apparent temperature, gusts, rainfall, wet hours and 0-1 cm soil moisture for the exact work point, and gauges, wildfire and floodplain add site context. What was missing was any terrain description and the deeper soil layers, so an excavation or access decision had no ground information at all. This is cheap and authoritative to fix: USGS 3DEP answers point elevation queries and the weather provider already returns a 0-100 cm soil layer. The honest limit is that a handful of samples across a kilometre is a screening proxy, not a survey, and it must be labelled as predicted context rather than a measurement.

**Decision optimization: absent, and half of it is not feasible.** The application reports risk; it never compares options, which is where a supervisor still does arithmetic by hand on the morning of the shift. Comparing schedules is feasible immediately because the hourly forecast and the deterministic bands already exist. Comparing routes is not: the repository has no routing service, no road-condition data, and no model of mud, access or staging, and inventing one from a straight-line distance would be cosmetic. Mitigation is already represented as per-hazard controls, so the useful addition is ranking candidate shift windows and naming which driver moved a window down.

**ML improvements: real, but the bottleneck is not a new model.** Three recent studies converge on the same diagnosis, now recorded in `docs/HYBRID_RESEARCH_RESULTS.md`, `docs/REVISION_RESEARCH_RESULTS.md` and `docs/LOCALCAL_RESEARCH_RESULTS.md`: ranking transfers acceptably, the alert operating point does not, roughly 42 percent of events are missed by every expert in the pool, and pairwise alert agreement between six different models runs from .974 to .989, so expert disagreement carries very little operational leverage. Two of the suggested directions are therefore weak as stated. Missed-event recovery cannot come from re-weighting the same experts; those events are missed because no member carries the signal. Regional calibration remains the strongest candidate, but it belongs to the county-day models that actually serve, and promoting it requires the build-and-evaluate pipeline plus the untouched temporal and geographic test partitions, which hold only 42 and 4 flood positives. That study is worth doing properly and is not worth rushing inside this round.

**Additional innovation: the ensemble planner is the overlooked one.** The service already fetches a forecast that supports real uncertainty quantification through Open-Meteo's GFS ensemble endpoint, and nothing used it. Member spread turns "gust max 48 km/h" into "two of three members cross this activity's limit", which is the difference between a number and a decision. Beyond that, the highest-value additions are cheaper than new models: an upstream gauge trend, and honest lead-time language on alert lines.

## What this round implements

A five-point USGS 3DEP terrain sample (the facility point plus roughly 500 m north, south, east and west) and the 0-100 cm soil moisture layer now travel with every assessment. The unstable-ground line in the JHA keeps its existing saturation rule and gains a cited terrain context row with sampled relief, gradient and deep soil moisture; context rows never set a rating.

A work-window planner (`backend/planner.py`, `GET /api/missions/{id}/work-windows`) ranks candidate windows of the mission's own shift length across the next two days using the same bands, wind limits and terrain rule as the JHA, one day either side of the planned start so a supervisor can move a shift earlier as well as later. Each window reports its exposure, its drivers, whether it is restricted, and the share of GFS ensemble members crossing the activity's gust, heat and rain limits. Windows with less than half their hours covered are skipped instead of scored, and the response states that spread is not calibrated risk.

The frontend renders the ranked windows inside the mission detail, above the Job Hazard Analysis, with the lowest-exposure window marked.

## Remaining weaknesses and the next steps

The planner's bands are the existing screening thresholds, not validated work limits, and its ensemble probabilities come from one weather model family. Terrain sampling does not see micro-topography, drainage or soil type, and the deep soil layer is a model field, not a probe. The county-day hazard models still use a single global threshold each; a regional operating-point study on their own partitions is the highest-value ML work left, followed by an upstream gauge trend and a second geographic cohort for the window planner.

Deliberately not done: route optimization, satellite or Jev fusion into hazard ratings, and any promotion of research models into serving. None has evidence yet, and shipping them would trade the project's credibility for surface area.
