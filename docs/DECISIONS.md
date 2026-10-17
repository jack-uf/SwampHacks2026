# Decision record: mission planning round (October 16, 2026)

I grilled the plan myself before building, since the user asked for autonomous work. Each question below has the answer I settled on and the evidence behind it. Facts come from the code, the stored data, and the sponsor pages retrieved tonight.

## Q1. Does PlanIT qualify for the Sperry challenge today?

No. The [MLH Sperry Tech page](https://www.mlh.com/challenges/01a10a1c-915e-fd71-bf16-9742d7d36285) asks for a tool that compares at least two utilities' public future construction plans and flags work that is physically close or scheduled around the same time. PlanIT had no construction plans at all.

Decision: build plan overlap detection on the EIA-860M "Planned" sheet. The August 2026 edition is public, downloadable, and lists 2,311 planned generators at 1,581 plants owned by 974 entities, with coordinates, state, balancing authority, status and planned operation month. The repo already uses EIA-860M for the 60 facilities, so the data path and provenance pattern exist.

Limits I accept: EIA-860M covers planned generators, not transmission or distribution work, and most owners are independent power producers rather than electric utilities. The UI shows the sector of every entity and can restrict to electric utilities. Planned dates are owner-reported and change month to month.

## Q2. What counts as an overlap?

A pair of plants owned by different entities, flagged when either:

- they are within 25 miles of each other (physically close), or
- their planned operation months are within 6 months and they are within 75 miles (concurrent in the same region).

Pure time overlap across the whole country is noise for labor and equipment sharing, so the timing rule needs a regional radius. Every flagged pair carries a per-factor score (proximity, timing, cross-state, shared balancing authority) so the reason is visible. Supervisor missions are compared against the same plans, which ties Sperry to the mission workflow.

## Q3. What is a mission?

A supervisor-defined unit of planned work: a name, an activity type from a fixed catalog, an EIA facility, a work location within 50 miles of it (same rule as crew reports), a shift window, crew size and the responsible supervisor's name. Missions live in new SQLite tables created with `CREATE TABLE IF NOT EXISTS`, so the existing database is untouched.

## Q4. How does a mission connect to the environmental data?

The site assessment already holds hourly Open-Meteo rows, active NWS alerts, FEMA zones, the county models and lightning timers. The mission slices the hourly rows to its shift window and reports coverage. Hours past the 3-day forecast horizon are unavailable, not zero. The county model only applies when its target UTC day intersects the shift.

## Q5. How is the Job Hazard Analysis rated, and who is in charge?

Deterministic rules rate each hazard as stop work, high, elevated, low or unknown. The thresholds are published screening heuristics (NWS heat index bands, the ANSI A92.20 12.5 m/s MEWP wind rating, rainfall bands), not validated injury models. The rules are code with tests, not AI output.

- Lightning: an active recorded-strike timer overlapping the shift is stop work. A replay timer produces a separately labeled simulated stop work. No supervisor override can lower a lightning stop work entry; the API rejects it.
- Jev suggestions appear as AI advisories next to the report they came from. They never set a rating. Only supervisor-verified report labels count as observed evidence.
- Pending reports are cited as unverified and flagged for review, matching the existing index rule.

## Q6. When does the JHA change, and how is history kept?

A new version is stored only when the analysis fingerprint changes: a hazard appears or disappears, a rating changes, or the set of cited evidence changes. Numeric drift inside the same band does not create a version. Each version records its trigger (mission created, report verified, lightning replay, environmental refresh) and a diff against the previous version. Versions are append-only.

Supervisor decisions (accept, override, mitigate) record the version, hazard, supervisor name, time and rationale. When a later version changes that hazard, the decision is shown as made under older conditions so the supervisor reviews it again. The name is self-declared because the deployment has one shared supervisor token. Real identity needs Auth0 or similar (see Q9).

## Q7. What is Site Hazard Memory built from?

NOAA Storm Events records from 2024 to 2026, already downloaded under `data/raw/` for model labels. They are real historical reports with narratives, locations and impacts. A build script extracts the non-marine events within 20 miles of each facility, plus county and zone records whose name matches the facility county, into a committed JSON file with source checksums. Verified local crew reports join the same retrieval.

Each retrieved record gets a per-factor score: geography, match with the mission's current conditions, mission activity sensitivity, season, and recorded impact. Memory is context. It never changes a rating.

## Q8. How do I reduce Jev abstention honestly?

Inspection of `output/jev-evaluation.jsonl` shows the abstention rule mixes three things: hazard uncertainty, uncertainty in auxiliary questions (location detail, clarification), and the definition that a confident "no hazard" counts as abstention. The review queue already routes the auxiliary uncertainty to clarification.

Decision: a v2 rule that abstains on the hazard label set only when a specific hazard question is uncertain, when the residual "other" question is uncertain and no specific hazard was accepted, or when the inbox choice confidence is under 0.7. The review queue logic stays exactly as it was, so human review is not reduced.

The ten frozen cases were seen while designing the rule, so they are development data. I froze the rule and a new set of author-labeled held-out cases in a protocol file before any held-out call, then ran them once. Both rules are scored on the same held-out responses. Author labels are still not field ground truth, and the docs will say so.

## Q9. Which sponsor integrations?

None of Gemini, Tiger Data, Auth0 or Vultr has credentials in `.env`, the environment, or the usual CLI config locations. Without credentials none of them can be demonstrated end to end tonight, so I skip all four. Auth0 is the most valuable next step because decision records need real identity. Tiger Data (Timescale) would suit the score history and JHA versions. Vultr would host a public demo, which the deployment boundary in the README currently forbids without auth.

## Q10. Retrain models?

No. There is no new research objective, and the storm investigation concluded that refitting against known tests cannot be validated. Missions consume the saved models as they are.

## Skipped on purpose

- Image inference with Jev: gated experimental API, photos stay local.
- Automatic mission clearance: no rule or model clears work.
- Seeded demo missions in the persistent database: missions created during browser checks carry a DEMO label in their names.

# Decision record: technical improvement round (October 17, 2026)

Same method as above: I questioned the plan myself, read the code first, and checked every provider contract against a live response before building on it.

## R1. What was weakest once I read the code?

The JHA could stay silent where it should speak. A vegetation-clearing mission had no wildfire entry at all unless NWS issued a fire alert, and river flooding had no source besides rain forecasts. NWS alerts that matched no keyword (Winter Storm, Dense Fog, Extreme Cold) were dropped, and "Wind Chill" matched the storm keyword "wind". The ML component reported "elevated" with no hint that most flagged county-days in testing had no event. A supervisor decision recorded the latest version even if the supervisor had been reading an older one. Two writers (a sync endpoint thread and the async refresh) could both read version N and race to insert N+1, and `INSERT OR IGNORE` hid the loser.

Order of work, by impact on a crew decision: JHA correctness and traceability first, then the two missing hazard feeds, then index uncertainty, identity and audit, then the ML diagnostic.

## R2. Which flood and wildfire feeds?

- Flood: NOAA National Water Prediction Service gauges (`api.water.noaa.gov/nwps/v1/gauges`, bounding-box query, no key). Each gauge carries an observed and a forecast NWS flood category with a valid time, so the rating uses the official thresholds instead of a raw stage I would have to interpret. USGS instantaneous values give stage without flood thresholds, so I left them out. A national box timed out after 60 seconds; a box around one facility returns in about a second.
- Wildfire: NIFC WFIGS current incident locations (public ArcGIS layer, no key). One request returns every active incident (241 wildfires at check time). NASA FIRMS needs a registered MAP_KEY, which this machine does not have.
- Lightning: no free, keyless strike feed with redistribution terms exists that I could verify, and `PLANIT_LIGHTNING_TOKEN` is blank. The ingestion contract stays; live lightning stays unavailable.
- Briefing delivery: `PLANIT_ENABLE_DELIVERY` and `PLANIT_BRIEFING_WEBHOOK` are blank, so external delivery cannot be verified. I fix a queue bug in it instead (R7).

Rating rules, all deterministic and documented as screening heuristics:

- Gauge within 10 miles of the work point: observed minor, moderate or major flooding rates high; action stage rates elevated; no flooding rates low. The forecast category uses the same bands as predicted evidence. An observation older than 6 hours, or a category the API marks not current, is unavailable for that gauge. No gauge within 10 miles is unavailable: river stage unmonitored, not safe.
- Wildfire: distance to the reported origin minus the radius of a circle with the reported acreage. Under 10 miles rates high, under 25 miles elevated, otherwise low. A response cut off by the server's record limit makes "no fire nearby" unavailable.

The first live run changed two rules. Sherburne County and Grand Coulee both rated high from fires reported 100% contained, so a fully contained fire now rates one band lower. Gauges without NWS flood thresholds (`not_defined`) still report a current stage; the evidence says so instead of claiming there is no observation. WFIGS also lists a "WILDFIRE TRAINING" record last discovered in January, so records not updated for 14 days are shown but not counted.

## R3. Does missing data still look safe anywhere?

Yes, in the portfolio ranking. A site with 45% coverage and a score of 2.0 sorts below a fully covered site at 3.0. Decision: every index also reports the highest score its unknown components could add (each unknown component at full severity), and the leaderboard sorts by that upper bound. Weights stay unchanged so stored history remains comparable.

## R4. Identity: Auth0 or something local?

Auth0 has no credentials here. Decision: optional per-supervisor bearer tokens in `PLANIT_SUPERVISOR_TOKENS`. When the token names a supervisor, decisions and reviews record that name and ignore the typed one. The shared token still works, and records made with it say the name was self-declared. Every supervisor mutation goes into an append-only audit table.

## R5. Can the ML evaluation say anything new without touching inspected holdouts?

Yes, about geographic generalization. The question is how much worse the selected recipes do in a state they never saw. Design, frozen in `docs/SPATIAL_CV_PROTOCOL.md` before the first run: leave one state out across the 18 development counties, fit on 2024, calibrate on January to June 2025, score July to December 2025, all from other states. Compare against the same rows scored by the same recipe fitted on all states. No 2026 row and no geographic-holdout state is read. The result is diagnostic only; it changes no model, threshold or serving scope whatever it shows.

## R6. What about Jev?

The 16 held-out cases are inspected, so they cannot tune a new rule. A new abstention rule would need fresh frozen cases and paid calls. That costs money and still leaves author-written labels. I leave the rule alone and record the limit. Prioritization is a different matter. The pending queue now lists urgent reports first. Clarification reports and reports Jev never screened come next, and standard reports last, oldest first within each group. Closed reports follow, newest first. That is a display order, so it needs no accuracy claim. An unscreened report ranks with clarification because nobody has triaged it yet.

## R7. Smaller reliability fixes

- JHA writes take an immediate SQLite write lock around read-then-insert. I left the journal mode alone: WAL would change the user's live database file for a gain nobody has measured.
- A decision must name the version it was made on; a newer version returns 409.
- Each mission records when its JHA was last re-checked, separately from when its last version was created.
- A failed reassessment returns the last assessment marked stale instead of a 500.
- Briefing delivery tries never-attempted snapshots before failed ones, so 20 permanent failures can no longer block every newer briefing. Failed snapshots still retry every minute; a per-snapshot backoff needs an attempt timestamp the schema does not have.
