# Dashboard visual and responsive verification

Verified October 17, 2026 in the Codex in-app Browser against the production frontend build. This iteration keeps the existing routes, data contracts and actions. It introduces no frontend dependency.

## Changes

The interface uses a restrained forest, moss and warm off-white palette. Portfolio metrics share a divided strip, the watchlist has a clearer reading order, and mission memory and construction overlaps use flatter rows with explicitly named scores. Overlap factors remain available in a disclosure. Forms, condition summaries and detail panels stack at narrow widths. Active navigation and selected rows expose their state to assistive technology.

The portfolio map previously rebuilt its layers during recurring renders, including the one-second clock update, and could invalidate its pixel origin during a zoom. It now retains site and report marker identity, updates their data in place, and defers projection-affecting updates until movement finishes. Filtered reports are memoized. A shared resize observer coalesces callbacks and waits for map movement to finish before invalidating size without panning. Both maps use that observer and prevent repeated world tiles. Selecting a facility stops the previous flight and respects reduced-motion preference.

Browser verification also caught a long mobile breadcrumb pushing the settings button outside the page. The breadcrumb can now shrink and truncate while the settings control retains its space. Sidebar status text wraps rather than clipping at intermediate widths.

## Verification environment

The QA backend used a SQLite backup at `/tmp/planit-ui-sprint.sqlite`, with the scheduler disabled on port 8001. No crew report, mission, briefing, feed refresh, replay or Jev request was submitted during these UI checks. Existing copied assessments aged normally; stale/missing labels in screenshots are actual state, not fabricated fresh observations. The application's persistent database was not the QA write target.

The browser had 110% page zoom. Viewport overrides were adjusted to exercise the CSS widths below, confirmed using `innerWidth` and document scroll width. The overrides are temporary and were reset after verification.

| CSS viewport width | Checks | Result |
| ---: | --- | --- |
| 320 | Loaded mission detail and hazard memory; overlaps; model table; long navigation header | No horizontal page overflow after the header fix; evaluation table retains its own horizontal scrolling |
| 354 | Overview, selected facility, crew report dialog | Readable metric grid and vertically scrollable single-column form; controls fit |
| 390 | Overview and map after desktop resize | Document width 390; 60 site marker paths retained |
| 768 | Overlap summary, filters and map | Document width 768; no control/card overflow |
| 1024 | Evaluation view and sidebar | Document width 1024; sidebar status fits its bounds |
| 1280 | Overview, watchlist and map | Document width 1280; 60 site marker paths; no horizontal page overflow |

Repeated map zoom-in/out actions, facility selection, navigation away/back and resizing after a zoom showed settled pins aligned with the map. No transient displacement was seen during this manual pass. DOM checks confirmed all 60 site paths remained present; this is not a pixel-by-pixel animation test. The QA tab recorded no console warnings or errors.

## Reproducible checks

```sh
npm --prefix frontend test
npm --prefix frontend run build
.venv/bin/python -m pytest -q
```

The frontend test exercises the actual shared resize helper: coalescing, deferral through map movement, flushing after movement, observer disconnection and cancellation after removal. It uses Node's built-in test runner and type stripping. The final production build passes TypeScript and Vite compilation. The full backend suite passes 121 tests, including the 13 new isolated ML experiment checks.

For a manual repeat, open the production build, inspect each view at the widths above, select a facility, open and close the report form, zoom the portfolio map repeatedly, and resize during movement. Verify that controls remain reachable and page width matches the viewport. Do not submit forms merely to check layout.

## Screenshots and remaining limits

![Desktop overview](../output/ui-sprint/desktop.jpg)

[Full mobile overview](../output/ui-sprint/mobile.jpg)

Safari, Firefox, physical touch devices and prolonged feed polling during gestures were not tested. Reduced-motion handling is implemented but was not exercised with an operating-system preference change. OSM tiles still require connectivity. No cross-browser or prospective safety claim follows from these layout checks.

## Follow-up: navigation and error cleanup

The second pass on October 17, 2026 removes the placeholder workspace card, fake profile, static navigation badges, generic monitoring text, research badge, page eyebrow and duplicate footer identified in the annotated screenshot. Supervisor access remains in the sidebar and header. Evidence labels, data limitations and safety guidance remain beside the information they qualify. The existing palette and navigation structure are retained.

Missions and plan overlaps now show padded alerts with retry controls. Mission loading, a successful empty list and a failed request have distinct states. A failed overlap request stops its loading indicator, and responses from obsolete filters are ignored. Mission detail errors have their own retry control. When a mission detail refresh succeeds, its list row receives the same version, readiness and mission fields so the selected row agrees with the analysis being displayed.

### Live-server recovery and record preservation

The original `Not Found` came from a stale process on port 8000: its OpenAPI document had no mission or activity routes although the checked-out backend defined them. Restarting that process with the current code resolved the error. No compatibility route or alternate backend was added. For this local server, backend edits require a restart because it runs without auto-reload:

```sh
.venv/bin/python -m uvicorn backend.main:app --env-file .env --host 127.0.0.1 --port 8000
```

A consistent SQLite backup was made at `/tmp/planit-before-ui-restart.sqlite` before restarting against the existing `data/planit.sqlite`. Read-only comparisons confirmed that every original row in `missions` (2), `reports` (5), `jha_versions` (8) and `jha_decisions` (1) remained byte-for-byte equal at the SQLite value level. The current database contained 11 JHA versions after normal condition refreshes and opening missions. No forms, decisions, reports, replay scenarios or manual feed refreshes were submitted during this pass. The canonical startup scheduler ran normally; live feed caches and additional JHA versions are expected operational writes. The temporary backup is local recovery evidence, not a repository artifact.

### Focused verification

The production build was checked in the in-app Browser on port 8000 at 100% page zoom. Error-state checks used the actual stale-server 404 before the restart; pressing the same Missions retry control after the restart loaded both existing missions and the selected analysis.

| State | CSS viewport widths | Result |
| --- | --- | --- |
| Mission list request failure | 320, 390, 768, 1024, 1920 | Alert left and right insets both 21px; page width equals viewport; no false empty state or perpetual loading |
| Loaded mission detail | 320, 390, 768, 1024, 1280, 1920 | Detail present, no horizontal page overflow, removed chrome absent |
| Overlap request failure | 320 | One alert, retry available, no loading indicator or horizontal page overflow |
| Supervisor settings | 1280, 320 | Sidebar and header controls open the dialog; at 320px its bounds are 14px to 306px; closed without submitting |

After the final version synchronization change, the selected mission row and detail both showed `v6`; a DOM assertion confirmed equality. Plan overlaps loaded its summary and project pairs after the backend restart. The final TypeScript/Vite build passed. The frontend map test passed, and the focused mission API and overlap suite passed 15 tests with one existing Starlette/httpx deprecation warning. `git diff --check` passed. Models, training datasets and frozen evaluations were not changed or rerun in this cleanup.

```sh
npm --prefix frontend test
npm --prefix frontend run build
.venv/bin/python -m pytest -q tests/test_mission_api.py tests/test_overlap.py
git diff --check
```

[Desktop after cleanup](../output/ui-cleanup/missions-desktop.jpg) · [Mobile after cleanup](../output/ui-cleanup/missions-mobile.jpg) · [Desktop failure state](../output/ui-cleanup/missions-error-desktop.jpg) · [Mobile failure state](../output/ui-cleanup/missions-error-mobile.jpg)

This pass did not repeat the full backend suite, map animation checks or tests in other browsers. The successful empty-workspace and mission-detail failure states were reviewed in code but were not independently exercised through browser fault injection. Browser overrides were reset after verification.
