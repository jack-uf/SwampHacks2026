# Local operations and API

## Process and storage

Run from the repository root with one Uvicorn worker. Lifespan starts separate briefing and assessment tasks. SQLite transactions persist cache, reports, optional report-triage entities, strikes, briefings and index history. Assessment cache keys include the serving artifact SHA-256, so an artifact replacement cannot reuse an assessment computed by a different model; restart after replacing the artifact. Historical briefings remain immutable snapshots. `PLANIT_DB` selects the database (relative paths are relative to the working directory); uploads are always in `data/uploads/`. Runtime database, uploads, raw source downloads and `.env` are ignored by Git. The included inventory and training/evaluation artifacts are versioned.

Each startup refreshes the portfolio using current assessment logic. After code-only changes, wait for that refresh or request authenticated `POST /api/refresh`; the model fingerprint does not identify Python source revisions. The combined ML index component remains unknown unless both county-scoped hazard estimates are available. Individual supported estimates can still appear in the forecast panel.

Back up the database and uploads together while the server is stopped. Preserve `.env` separately in secure storage. There is one canonical schema and no automatic migration or historical-format compatibility. A schema/artifact mismatch should be diagnosed explicitly; do not delete the database to repair a cache problem. Preserve reports and briefings before any deliberate reset. The app currently has no retention/deletion UI or audit history.

## Configuration

| Variable | Purpose |
|---|---|
| `PLANIT_DB` | SQLite path, default repository `data/planit.sqlite` |
| `PLANIT_ADMIN_TOKEN` | Required shared bearer secret for supervisor mutations; missing configuration returns 503, incorrect/missing request token returns 401 |
| `PLANIT_USER_AGENT` | Identifying public-provider request header; include your contact |
| `PLANIT_LIGHTNING_TOKEN` | Independent secret for real strike ingestion; blank disables ingestion |
| `PLANIT_ENABLE_DELIVERY` | Exact `true` enables optional delivery; default disabled |
| `PLANIT_BRIEFING_WEBHOOK` | HTTPS destination, used only with delivery enabled |
| `PLANIT_JEV_ENABLED` | Exact `true` authorizes optional paid Jev requests; disabled unless an operator explicitly approves costs |
| `TYPESAFE_API_KEY` | Server-only TypeSafe bearer credential; never exposed to the browser |
| `PLANIT_JEV_MODEL` | Pinned documented stable version, default `jev-1.13.0`; reject unexpected response model |
| `PLANIT_SCHEDULER` | Exact `true` starts jobs; defaults true, set false for isolated tests/manual operation |

The browser stores the supervisor secret in session storage. It is a local shared secret, not a user account or authorization hierarchy. Reads and uploaded images are public to anyone who can reach the server. Keep loopback binding. Photos, descriptions and coordinates can expose sensitive site details; do not use real personal/sensitive material in a demo. Image EXIF stripping does not anonymize visible image content.

## Refresh, staleness and failures

A portfolio refresh assesses sites sequentially, with four concurrent provider requests at most and a 250 ms pause between sites. The next scheduled portfolio pass begins 30 minutes after completion. Assessment timestamps older than 35 minutes are marked stale; requesting a stale detail triggers reassessment. The leaderboard polls every 15 seconds and selected detail every minute, but polling does not bypass assessment/provider caches.

Provider TTLs: weather 30 minutes; NWS five minutes; NWPS river gauges and WFIGS wildfire incidents 15 minutes; inference archive six hours per county/target date; static FEMA seven days. Every five minutes an alert watch re-fetches NWS alerts for each site with an open mission and reassesses that site when the set of in-effect alert IDs changes, so mission JHAs pick up new or cancelled alerts within about five minutes. Sites without open missions still follow the 30-minute assessment cycle. This is polling decision support, not guaranteed immediate warning delivery.

River gauges come from one NWPS bounding-box query per facility (about 70 miles each way). The JHA uses the three nearest gauges within 10 miles of the work point: observed minor, moderate or major flooding rates high, action stage elevated, no flooding low, and the gauge's forecast category the same way as predicted evidence. Observations older than six hours, gauges without flood thresholds (`not_defined`) and gauges the API marks not current are unavailable. No gauge within 10 miles is unavailable, not safe. Wildfire uses one national WFIGS query for active wildfire incidents (`IncidentTypeCategory='WF'`, no fire-out time); each assessment keeps incidents within 150 miles of its facility. The distance is origin distance minus the radius of a circle with the reported acreage: under 10 miles rates high, under 25 miles elevated, fully contained fires one band lower, records not updated for 14 days are listed but not counted. A response cut at the server record limit makes wildfire unavailable. Both are deterministic screening heuristics, not validated hazard models. NASA FIRMS satellite detections need a MAP_KEY and are not used.

Requests retry HTTP 429/500/502/503/504 up to three attempts with one- and two-second waits, a 35-second timeout, and a shared concurrency limit. A failure retains the last successful fetch timestamp and records the error; repeated failures have a two-minute cooldown. Missing sources stay unavailable and do not feed old/synthetic values into the index. Empty successful NWS results mean no matching active alert at fetch time; failed NWS results mean unknown.

If startup shows no scores, inspect `/api/health`, allow the first refresh to finish and inspect visible provider errors. After restoring network access, use authenticated Refresh feeds or wait for the next pass. Check identifying User-Agent, public API limits, system clock, artifact presence and exact dependency versions. Restart after replacing the saved model. Do not interpret a missing score as low risk.

## Screening index

Each available component contributes `10 × weight × clamp(severity, 0, 1)`, rounded to two decimals; the displayed sum is rounded to one decimal. Weights are fixed and never redistributed. Coverage is the sum of known component weights. An unknown component contributes no numeric points but remains visibly missing; compare scores only with their coverage and timestamps. Coverage is data completeness, not confidence or safety.

| Component | Weight | Severity before clamping |
|---|---:|---|
| Heat | 0.15 | `(maximum next-24h apparent °C − 27) / 18` |
| Rain and saturation | 0.20 | max of `forecast rain / 75`, `(soil / .45) × (forecast rain / 50)`, `(past-72h rain / 150) × (forecast rain / 50)`; rain in mm, soil volumetric fraction |
| Wind/storm | 0.15 | `(maximum next-24h gust km/h − 25) / 65` |
| County model | 0.20 | Maximum available calibrated in-region flood/storm probability |
| NWS alerts | 0.20 | Maximum severity: Extreme 1, Severe .8, Moderate .5, Minor .2, other .3; successful empty response 0 |
| FEMA mapped exposure | 0.05 | Special flood hazard flag T: 1; mapped F: .2; no recognized mapping: unknown |
| Reviewed reports | 0.05 | Verified, unresolved reports created within 48 hours, divided by three |

Rain requires all three rain/soil inputs; any missing input makes the component unknown. These thresholds and weights are transparent heuristic screening choices, not empirically validated injury-risk coefficients. Storm ML probability is not the same as the storm index contribution. Unverified, rejected and resolved reports contribute zero. Heat is not WBGT. Flood screening is not a hydraulic forecast. Wildfire/wildlife reports contribute through the reviewed-report component and relevant NWS alerts.

**Lightning is a mandatory independent override.** Active recorded live-strike holds take precedence in ranking and banners, regardless of the numeric index. It is not diluted into the weighted sum.

## Lightning rules and provider contract

A Haversine great-circle calculation uses Earth radius 3958.7613 miles. Strikes at distance ≤10 miles qualify. The latest qualifying occurrence time defines `review_after = occurrence + 30 minutes`. Newer qualifying strikes reset that time; older/out-of-order strikes cannot shorten or extend it incorrectly. A timestamp at or before now is required to start a timer. At expiry, supervisor review and established procedures are still required; this is not an autonomous all-clear.

`POST /api/lightning/strikes` accepts a JSON object with `id`, `latitude`, `longitude`, and timezone-aware ISO `occurred_at`. Send `Authorization: Bearer <PLANIT_LIGHTNING_TOKEN>` from an authenticated provider bridge. IDs are idempotent within the live namespace; duplicate IDs do not update a prior strike. Accepted ingestion timestamps must be within the past hour and no more than five seconds ahead of server time; future accepted records cannot start a hold until their occurrence time. Maintain a synchronized system clock. Strike queries consider the last hour. There is no vendor-specific external client, feed heartbeat or coverage validation in this MVP: a configured token only enables ingestion, not proof of a functioning upstream feed.

`POST /api/sites/{id}/lightning-replay` records an explicitly synthetic strike at that facility using the supervisor token. Replay and live are separate database namespaces and separate UI countdowns. Replay does not produce a live hold or affect the live-hold metric. No lightning records is not evidence of no lightning.

## Reports

`POST /api/sites/{id}/reports` uses multipart fields `classification`, `description`, `latitude`, `longitude` and optional `image`. Classes: heat, storm, flood, lightning, wildfire, wildlife, unstable_terrain, equipment_obstruction, other. `classification` is the primary manual class; optional `categories` is a JSON array of additional manual classes (at most nine). `jev_consent` defaults false. Primary and additional labels are deduplicated and persisted in an associated report-triage entity. A report without requested enrichment remains manually classified; no historical schema conversion occurs. Descriptions trim to 10–2,000 characters; coordinates must be valid and within 50 miles of the selected facility. Initial map coordinates are the facility's location and must be confirmed or replaced by actual observation coordinates. Browser geolocation is optional and permission-dependent.

Images are limited to 5 MB, JPEG/PNG/WebP, and 20 megapixels. They are decoded, resized within 1600×1600, converted to RGB JPEG and saved under generated UUID names. Re-encoding removes embedded metadata. If persistence fails after image creation, the new file is removed. Reports start pending. `PATCH /api/reports/{id}` with `{"status":"verified"}`, `rejected`, or `resolved` changes review status and reassesses the site. Invalid identifiers return 404. Review has no user identity or status-transition audit log.

### Optional Jev enrichment and correction

The operator must separately authorize API spending and configure `PLANIT_JEV_ENABLED=true` plus `TYPESAFE_API_KEY`. Each submitted report must additionally include explicit text-sharing consent. The dashboard checkbox is unchecked by default. With both gates satisfied, `backend/jev.py` makes one 15-second HTTP request to the documented `POST https://api.typesafe.ai/v1/systemone` endpoint, pinned to `PLANIT_JEV_MODEL`. The state contains only `observation` text: no photo, coordinates, site/report identifiers or supervisor credentials. Descriptions may still contain sensitive details; obtain appropriate authorization and redact before consent. Official image support is experimental and **not enabled or live-verified**.

Twelve independent Noul questions produce nine possible hazard suggestions plus immediate-review, location-sufficiency and clarification judgments. A Choice question selects `hazard_review`, `clarify_report` or `routine_log`. A Score question assesses urgency of human follow-up using explicit levels 0 routine, 1 prompt review of a current hazard, 2 immediate review of explicit current exposure. These are fourteen questions in one request, not fourteen requests. The complete wording and typed response contract are in `backend/jev.py` and [Jev validation](JEV_VALIDATION.md).

Noul is the provider's yes probability, not an independently calibrated confidence score. Hand-chosen .7 acceptance/.3 rejection cutoffs leave intermediate answers uncertain; Choice/Score confidence below .7 also requires review. Any uncertain answer or no accepted hazard marks abstention, retaining accepted labels. Immediate-review Noul or urgency-level-2 probability ≥.7 enters the urgent queue; otherwise ambiguity or a clarification judgment enters clarification. All reports remain pending for human review regardless of inbox or score. Cutoffs and question wording have not been field-calibrated. The UI displays the model, original typed judgments/distributions, uncertainty, usage and separate manual/AI source.

Missing consent, disabled configuration or missing credentials makes **zero provider calls**. Timeout, connection failure, 401/403, 429, 5xx, malformed/missing answer IDs, wrong answer types/ranges/usage, inconsistent distributions/Score legend or unexpected model all yield a persisted unavailable reason with manual labels intact. There is no automatic retry or queue that might duplicate paid inference. Report and triage are committed together; photo cleanup remains in place on failed persistence. Provider bodies, arbitrary response headers and secrets are not logged or echoed. Twelve real authorized text requests succeeded; failures were tested with intercepted HTTP rather than paid deliberate errors.

AI suggestions are retained as suggestions, never copied into human classification or verification. `PATCH /api/reports/{id}/triage` with `{"categories":["wildlife","unstable_terrain"]}` requires supervisor authorization, validates at least one supported category, saves source `manual_correction` and timestamp, updates the report's primary classification, and preserves the original provider suggestion. The map and timeline show manual categories with separate AI suggestions; hazard filtering matches either source. Pending queue filtering offers urgent, clarification and unavailable/manual enrichment. Both filters apply to map pins and timeline; counts and empty states show when reports are hidden. Rejected reports have no map pin. Correction does not change pending/verified/rejected/resolved status; failed saves leave the editor open. Human review is the separate existing PATCH operation; only verified recent reports affect the heuristic index. Jev never clears a lightning hold or authorizes work.

`tests/fixtures/jev-hypothetical.json` has six hypothetical observations and hand-authored Noul probabilities for offline interpretation checks; other typed test responses are generated locally. These **do not measure Jev semantic accuracy**. The separately frozen `tests/fixtures/jev-scenarios.json` contains ten author-labeled synthetic observations. Real responses in `output/jev-evaluation.jsonl` measured micro precision .75, recall .90, abstention 1.0 and uncertainty .90. See [Jev validation](JEV_VALIDATION.md) for costs, per-class counts, disagreements, reproduction and the distinction from independently labeled field evidence, which is unavailable. The runner defaults to a no-network plan; `--live` is an explicit paid operation and never runs in CI. A new evaluation requires a new protocol/output rather than overwriting evidence or tuning against these known cases.

## Briefings and optional delivery

Every minute the briefing task checks each site's IANA timezone using `zoneinfo`. At local hour ≥18 it creates one immutable snapshot for the next local calendar date, protected by a database uniqueness constraint. Winter/summer offsets and Arizona's no-DST behavior are tested. A restart after 18:00 catches up the current day's next-day briefing; it does not backfill missed historical days. Jobs are sequential and may be delayed by provider timeouts; exact-to-the-second delivery is not guaranteed. Keep the process running. Multiple workers would duplicate in-process scheduling and are unsupported.

The next local date selects forecast hours in that timezone, allowing 23/24/25-hour DST days. The briefing summarizes forecast apparent temperature, gust and rainfall, NWS alerts, current index/model context and procedural review actions. Its model target remains tomorrow UTC, which can differ from the local briefing date. A saved snapshot is not overwritten by a repeat request or later weather update. `/api/sites/{id}/briefings` GET returns up to 30 saved snapshots; POST immediately generates or retrieves tomorrow's snapshot.

Delivery is disabled unless both explicit opt-in and an HTTPS webhook are configured. Each tick attempts up to 20 undelivered snapshots. Payload is the briefing JSON and `Idempotency-Key` is the briefing UUID. Success marks delivered; failures remain recorded and retry on later ticks. Delivery is at least once: receivers must deduplicate. Enabling delivery can send accumulated undelivered history; review the queue and destination first. No external delivery was configured or exercised. Approve any consequential external delivery before enabling it.

## API surface

| Method | Path | Access / result |
|---|---|---|
| GET | `/api/health` | Public process, inventory and refresh status |
| GET | `/api/sites` | Public ranked inventory with scores and independent timers |
| GET | `/api/sites/{id}` | Public source-aware assessment, reports and last 48 index samples |
| GET | `/api/evaluation` | Public measured benchmark; 503 if absent |
| POST | `/api/refresh` | Supervisor; starts a locked background portfolio assessment |
| POST | `/api/lightning/strikes` | Separate provider secret; real recorded strike contract |
| POST | `/api/sites/{id}/lightning-replay` | Supervisor; labeled replay |
| POST | `/api/sites/{id}/reports` | Supervisor; validated multipart report |
| PATCH | `/api/reports/{id}/triage` | Supervisor; correct multihazard labels while retaining AI evidence |
| PATCH | `/api/reports/{id}` | Supervisor; review status |
| GET / POST | `/api/sites/{id}/briefings` | Public history / supervisor generation |
| GET | `/uploads/{generated-name}.jpg` | Public local report photo |

Use `/docs` for interactive request schemas. Site IDs are string EIA IDs (for example `3470`), not names. Never put bearer secrets in URLs, screenshots or committed files.
