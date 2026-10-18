# AGENTS.md

PlanIT Control is a SwampHacks entry: a field-safety console for power-plant crews that turns live environmental data, crew reports and public construction plans into a cited, versioned Job Hazard Analysis (JHA). Three prose deliverables are still open: the demo script (`docs/DEMO.md`), a README rewrite and the Devpost text. This file holds the rules for writing them and the verified facts they draw on.

## Writing prose in this repository

1. Load `/Users/jack/.claude/skills/technical-writing/SKILL.md` and `/Users/jack/.claude/skills/unslop/SKILL.md`, and apply both to every paragraph. The Devpost text is closer to marketing copy; the technical-writing truth rules still bind it.
2. Take every number from "Verified facts" below or from the file it cites. When the producing command is cheap (tests, the overlap counts), run it again and use the fresh number.
3. Before delivery, search the draft for em dashes, en dashes and ` -- ` and rewrite each sentence that has one.

A draft is done when every number and capability in it traces to a path or command in this file.

## Truth rules

- **Evidence kinds travel with the data.** The app labels every JHA line with one of `observed`, `reported`, `predicted`, `official_alert`, `simulated`, `unavailable`, `ai_suggestion` or `historical` (`backend/missions.py`). Prose uses the same words: forecasts are predicted, a missing feed is unavailable, a replay strike is simulated.
- **Deterministic rules are authoritative.** The lightning rule (a strike within 10 miles starts a 30-minute stop-work timer) and the rating bands in `backend/missions.py` set every rating. Jev suggests labels and the supervisor decides. A stop-work entry rejects an override with HTTP 409 (`tests/test_mission_api.py`), and the UI disables that option. Describe Jev as advisory in every sentence that mentions it.
- **Demo data says it is demo data.** Missions named `DEMO: ...`, reports that start `FICTIONAL DEMO REPORT` and every lightning replay are fictional or simulated. Keep that wording on screen, in narration and in screenshots.
- **Jev numbers come with their method and limits.** Quote the held-out results below with the sentence on how they were produced. The ten frozen cases tuned the v2 rule, so they are development data and support no claim.
- **Claims stop at what was run.** PlanIT has not been used by a utility crew; there are no users, pilots, customers or incident reductions to report. Write "designed for" or "verified locally" instead.

## Deliverables

### Demo script, `docs/DEMO.md`

The current `docs/DEMO.md` is a five-minute verification walkthrough written before missions existed. `docs/JEV_VALIDATION.md` points to it as "the complete regression verification record", so move that record into a section or file of its own and update the pointer before replacing the walkthrough.

The script runs 2-3 minutes, built on the loop below. The memorable moment is step 4: a crew report the supervisor verifies changes a rating, and the JHA history names the report that did it.

1. Plan a mission (Missions & JHA, "Plan mission"). JHA v1 appears with predicted shift conditions and a cited source on every line. Without `PLANIT_LIGHTNING_TOKEN`, lightning shows unavailable and the rating stays unknown.
2. Submit a crew report from the mission with text-sharing consent. Jev returns typed suggestions; the report stays pending and counts for nothing yet.
3. The JHA moves to v2: the report appears as `reported`, not counted.
4. Verify the report. The hazard goes from Low to High, readiness turns High, and v3's trigger reads "Crew report <id> marked verified by supervisor" with an evidence diff.
5. Record a mitigate decision. The line shows who, when, which version and why.
6. Optional, about 20 seconds: a lightning replay on an active-shift mission shows simulated stop work and the disabled override.
7. Plan overlaps: other utilities' planned plants near each other in space and time (the Sperry angle).

Target prizes and the angle for each: Sperry, cross-entity comparison of public construction plans (`backend/overlap.py`); Most Creative Use of AI, typed Jev questions with honest abstention, where AI never sets a rating; Overall, the end-to-end loop on real public data.

After each run, reject the fictional report, because verified reports feed the facility's screening index (`backend/domain.py`, "Reviewed crew reports"). The script is done when a timed read-through fits in 3 minutes (2 for video) and every on-screen claim matches a workflow under "Verified browser workflows".

### README rewrite

Keep the setup, run and test commands working as written; re-run them before changing them. Link model metrics to `docs/EVALUATION.md` and `docs/MODEL_CARD.md` rather than copying tables, so each number has one home. Add missions and the living JHA, Site Hazard Memory, plan overlaps and Jev abstention v2, each with its key file. End with limitations and the credits below.

### Devpost text

Use Devpost's standard sections (what it does, how we built it, challenges, accomplishments, what we learned, what's next) and a "Built with" list. Name the target prizes. Carry the limitations and the AI credits into the text itself; a link alone is not enough.

## Credits for AI tools

Credit exactly these, in both the README and Devpost, and ask the team whether they used others:

- **OpenAI Codex with GPT-6.1 Sol (`openai/gpt-6.1-sol`).** Local Codex session logs for this project start 2026-10-16 19:58 EDT (`~/.codex/sessions/2026/10/16/`). The commits from `1f5b438` through `8cac1a6` fall inside those sessions. The user refers to this as ChatGPT/GPT-6.1 Sol.
- **Cursor agent running Anthropic Claude.** Commits `1159e84`, `11592a5`, `405ca32` and `73fdf80` carry `Co-authored-by: Cursor`, and so do the commits that add missions UI, this file and `docs/demo/`. Cursor's built-in browser captured `docs/demo/`. Playwright captured the earlier evidence in `output/playwright/`.
- **TypeSafe Jev (`jev-1.13.0`)** is a runtime component, the crew-report classifier, not a coding tool. Credit it as part of the product.

## Verified facts

### Features and where they live

| Feature | Backend | Frontend | Tests |
| --- | --- | --- | --- |
| Missions with a versioned, cited JHA and supervisor decisions | `backend/missions.py`, `backend/main.py` (`/api/missions*`), tables in `backend/store.py` | `frontend/src/missions.tsx` (`MissionsView`) | `tests/test_missions.py`, `tests/test_mission_api.py` |
| Site Hazard Memory with per-factor relevance | `backend/memory.py`, `scripts/build_hazard_memory.py` | Mission detail, "Site Hazard Memory" | `tests/test_memory.py` |
| Cross-entity plan overlaps (Sperry) | `backend/overlap.py`, `scripts/build_planned_projects.py`, `/api/plan-overlaps` | `frontend/src/missions.tsx` (`OverlapsView`) | `tests/test_overlap.py` |
| Work-window comparison with ensemble spread | `backend/planner.py`, `/api/missions/{id}/work-windows`, terrain and ensemble providers in `backend/providers.py` | Mission detail, "Work windows" | `tests/test_planner.py` |
| Window decisions recorded against the comparison a supervisor saw | `/api/missions/{id}/window-decisions`, `window_decisions` table in `backend/store.py` | Mission detail, "Work windows" actions | `tests/test_window_decisions.py` |

Its decision rule, constraint set and uncertainty language are documented in `docs/WORK_WINDOW_PLANNER.md` and measured by `scripts/evaluate_planner.py`. A move needs one weighted hazard to improve a whole band with no weighted hazard worse, members can only veto, windows below 75 percent forecast coverage are skipped, and lower exposure is never described as safety.

A window decision echoes the snapshot digest of the comparison the supervisor was looking at; the server recomputes the windows and answers 409 when the forecast has moved since, so no record can be attached to a stale view. A move must name a listed, unrestricted window, and every record writes an audit row carrying the JHA version current at the time.
| Jev abstention v2 and held-out evaluation | `backend/jev.py` (`abstention`), `scripts/evaluate_jev.py` | Report triage text in `frontend/src/main.tsx` | `tests/test_jev.py`, `tests/test_jev_evaluation.py` |

The reasoning behind each design choice is in `docs/DECISIONS.md`.

Behavior worth stating precisely:

- A new JHA version is stored only when a hazard's rating or set of evidence IDs changes; forecast drift inside a rating band adds no version. Each version records its trigger and a diff. A decision goes stale when its hazard entry changes. A decision must name the JHA version it was made on; if a newer version exists the API refuses it with HTTP 409.
- Flood and wildfire lines come from live public feeds (`backend/missions.py`, `gauge_evidence` and `wildfire_evidence`): NWS river gauges within 10 miles via the NWPS map service, and active WFIGS wildfire incidents within 25 miles. Both rules are deterministic screening heuristics, not validated models. A missing, stale or cut-off feed is unavailable, never low. NASA FIRMS needs a key and is not used.
- Every five minutes an alert watch reassesses each site with an open mission when its set of NWS alerts changes, so a mission JHA picks up a new or cancelled alert within about five minutes.
- The facility watchlist ranks by the highest index score unknown inputs could reach (`score_max` in `backend/domain.py`), so missing data never sorts a site as safer.
- Supervisor identity: with `PLANIT_SUPERVISOR_TOKENS` set, a personal token records its owner's name and the decision shows "personal token"; under the shared `PLANIT_ADMIN_TOKEN` the name is typed and shown as self-declared. Every supervisor mutation writes an append-only audit row (`GET /api/audit`). Tokens have no expiry or revocation.
- Hazard Memory ranks records by geography .3, conditions .3, mission activity .2, season .1 and impact .1. It informs the supervisor and never changes a rating.
- An overlap pair is two planned plants of different owners within 25 miles, or within 75 miles with planned operation within 6 months. Co-owners of one plant are excluded. "Possible affiliates" is a name heuristic; EIA does not publish parent companies.

### Data

- Planned construction: EIA-860M August 2026 planned generators, 1,581 plant-owner records from 974 entities (`data/planned-projects.json`, source hash in `data/raw/eia860m-august-2026.xlsx.source.json`). These are generators, not transmission or distribution work.
- Overlap counts from `overlap.find_overlaps`: 8,558 pairs across all sectors, 75 among electric utilities (198 records, 88 entities). The EIA file gives McNew (Reno, KS) and Mullin Creek #1 (Nodaway, MO) identical coordinates; `overlap.suspect_coordinates` drops both from distance comparisons and the UI names them.
- Hazard Memory: 5,075 NOAA Storm Events records for 2024-2026 across 59 facilities (`data/hazard-memory.json`). They are official weather reports, not crew incidents at the plants.

### Run and test

Setup and run commands are in `README.md`. The backend runs without auto-reload, so restart it after backend changes or a frontend build. Supervisor actions need `PLANIT_ADMIN_TOKEN` entered under Supervisor access; read it from `.env` and keep it out of chat, screenshots and commits.

- `.venv/bin/python -m pytest -q`: 151 passed. The measured wind-hazard experiment adds 13 of those, the transformer experiment 6, and the multimodal pilot 11; their artifacts stay under `experiments/sprint-20261017-*`, they leave the served models untouched, and the wind and transformer sprints retain the forecast threshold (`docs/WIND_SPRINT_RESULTS.md`, `docs/TRANSFORMER_SPRINT_RESULTS.md`, `docs/MULTIMODAL_SPRINT_RESULTS.md`). Before the October 17 improvement round the suite had 88; before missions and overlaps, at `8cac1a6`, 57.
- `npm --prefix frontend run build`: passes.

### Verified browser workflows

Run on 2026-10-16 between 23:39 and 23:48 EDT against a local server with live Open-Meteo, NWS and Jev. Screenshots are 1440x900 except the mobile one.

| Workflow | Result | Screenshot |
| --- | --- | --- |
| Create `DEMO: drainage trench repair (fictional mission)` at Gavin Power, OH, excavation | JHA v1 Low; 8 of 8 shift hours predicted; lightning unavailable | `docs/demo/01-mission-jha-v1.png` |
| Submit a fictional trench-wall report with consent | Jev accepted unstable terrain, routed to hazard review (confidence .92), abstained under v2 because equipment obstruction was uncertain; JHA v2 shows it as reported, not counted | `docs/demo/02-crew-report-pending-v2.png` |
| Verify the report | Unstable ground Low to High, readiness High, v3 attributed to the verification | `docs/demo/03-verified-report-jha-v3.png` |
| Record a mitigate decision | Who, when, version, rationale and controls shown | `docs/demo/04-supervisor-decision.png` |
| Version history and Hazard Memory | Three versions with diffs; NOAA cards with factor bars | `docs/demo/05-version-history-and-memory.png` |
| Lightning replay on an active-shift mission at Peach Bottom, PA | Simulated stop work; override disabled in the UI and refused by the API with 409 | `docs/demo/06-simulated-lightning-stop-work.png` |
| Plan overlaps, utilities only | 75 pairs, map and ranked cards, suspect coordinates listed | `docs/demo/07-plan-overlaps.png` |
| Mobile, 390 px | No horizontal overflow on missions or overlaps | `docs/demo/08-mobile-mission-jha.png` |

Rejecting the report afterward produced v4 (back to Low), marked the decision stale and removed the report from Hazard Memory. No browser console errors; the server log showed only the expected 409. Both DEMO missions remain in `data/planit.sqlite`.

These screenshots predate the river-gauge and wildfire lines, so a JHA on screen today has more lines than they show. A re-check on 2026-10-17 between 00:26 and 00:33 EDT ran against a copy of the database, with live feeds and a personal supervisor token. The Gavin mission moved to v6 on an environmental refresh, and its flood lines read Point Pleasant (POPW2) 6.4 miles away, observed and forecast no flooding. Grand Coulee's site panel listed the Kaiser Canyon fire, fully contained, as elevated. A decision on v6 recorded the token owner's name instead of the typed one; a decision sent for v5 got 409. Submitting and rejecting a fictional report wrote two audit rows without the report text.

### Jev before and after

The v1 rule abstained whenever any answer was uncertain, including confident "no hazard" answers, so it abstained on 14 of 14 held-out responses. The v2 rule `hazard-decisive-v2` abstains only on an uncertain specific hazard, uncertain `other` with nothing accepted, or inbox confidence below 0.7. It abstained on 5 of 14; 7 of its 9 decisive answers matched the author labels and 2 were false positives (`resolved-water`, `hearsay-bear`). No review queue changed. Label metrics: micro precision .722, recall .867; routing 14 of 14.

Method: 16 new fictional cases, with corpus, model and rule hashes frozen in `docs/jev-heldout-protocol.json` and committed in `405ca32` before any call; one pass, no retries. 15 were sent; one failed strict validation and the runner stopped, so 14 responses count. Limits: one author wrote cases and labels, 14 responses support no confidence interval, and the cases are fictional. Full write-up: `docs/JEV_VALIDATION.md`.

### Sponsor integrations

None of Gemini, Tiger Data, Auth0 or Vultr is integrated: no credentials for them exist in `.env` or the environment, so none could be shown end to end. Auth0 would add the most: supervisor identity is a typed name behind a shared token unless static personal tokens are configured (`docs/DECISIONS.md`). Name no sponsor prize for these.

### SwampHacks eligibility flags

Confirm each against the current official rules before submitting:

- All commits are dated 2026-10-16, the first at 20:06 EDT. Check that this falls within the event's hacking window.
- Every commit is authored by one account, `jack-uf`; team-size rules may require 2-4 members.
- AI use must be disclosed; the credits section above covers it.
- Demo length: 3 minutes live, 2 minutes for video.
- TypeSafe's sponsor status is unknown; claim no sponsor prize for Jev unless the organizers list it.
- The Sperry fit rests on generator plans only (see Data).
