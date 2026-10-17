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
