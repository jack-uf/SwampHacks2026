# Five-minute demonstration and verification

## Preparation

Follow README setup, start the single-worker server, open port 8000, and let the first portfolio assessment populate. Set Supervisor access using your private `.env` token. Use real current provider results; the presence of a particular alert is not guaranteed. A failed or missing feed is a useful uncertainty demonstration, not a reason to manufacture a value.

## Demonstration

1. **Portfolio, 45 seconds.** Show 60 source-attributed EIA facilities, map markers, ranked screening scores, coverage and timestamps. Explain that a partial score can look low because inputs are missing. A gray marker is unknown/stale, not safe. The inventory is a historical December 2024 snapshot.
2. **Site, 60 seconds.** Search “Parish” and select W A Parish (EIA ID 3470, Texas), or any facility with available feeds. Inspect modeled current conditions, forecast rain/gusts, shallow soil moisture, NWS details, FEMA exposure and contribution bars. Describe the difference between a model estimate, a forecast, a warning and a crew observation. No guarantee that the Wind Advisory observed during development is still active.
3. **Actual ML, 45 seconds.** Open Model intelligence. Show the six measured candidate results and validation-selected logistic winners. State flood precision 5.2%/recall 78.2% and storm precision 7.4%/recall 67.0% on the regional 2025 test. Show geographic holdout and low precision honestly. On a nearby site, inspect forecast target date and signed reference-feature sensitivities; on distant sites, show the extrapolation warning and missing index component.
4. **Lightning, 30 seconds.** Return to the selected site. Click Replay lightning. The labeled replay countdown starts near 30 minutes; click again to reset. Show that live holds remain separate. Explain the 10-mile recorded-strike rule, latest-timestamp reset and human review after expiry. Do not call replay a live strike or timer expiry an all-clear.
5. **Crew report, 60 seconds.** Submit a photo with class Other and description beginning “DEMO ONLY — no field hazard observed; testing reporting.” Confirm demonstration coordinates, submit, and show its map marker, image and pending status. Verify it to show the reviewed-report contribution, then resolve it so the demo leaves no false active hazard. Reports/photos persist locally. Use nonsensitive demonstration imagery.
6. **Briefing, 40 seconds.** Open Safety briefings for the site. Generate tomorrow's local-date briefing, inspect forecast hours/source times/actions, and generate again to show the same immutable saved snapshot. Explain automatic 18:00 facility-local scheduling, DST handling and separate UTC model date. Delivery remains unconfigured.

## Verification record

Verified locally with authentic public feeds on October 16, 2026, with UTC artifacts crossing into October 17. Tests use isolated temporary databases and deterministic fixture feeds; the browser session separately exercised real provider responses.

- **14 automated tests pass** (one upstream Starlette/httpx deprecation warning). Tests cover 10-mile boundary, timer reset/expiry/out-of-order/future strikes, local 18:00 and DST, fixed-weight missing-input behavior, report/photo validation and persisted review, authenticated mutations, live/replay separation and duplicate ingestion, briefing idempotence and due-time scheduling, failed-fetch timestamps/cooldown, unknown FEMA mapping, and actual dataset/artifact split contracts.
- Production frontend type-check/build passed. Desktop and 390-pixel mobile views were inspected, with all 60 markers and no mobile horizontal page overflow. Browser console showed no errors or warnings after final UI fixes.
- Actual EIA-derived inventory populated the map. At W A Parish, live queries returned modeled weather, an NWS Wind Advisory, and FEMA zone X / SFHA F. Values are time-dependent, not fixtures.
- Supervisor authentication, two replay clicks/reset, photo upload, pending → verified → resolved review, report map marker and index response were exercised in the browser. The stored test report explicitly says no field hazard was observed.
- Next-day briefing generation/history was exercised; repeating generation retrieved the existing snapshot. Persistent state was checked after server restart. Automated due-time tests establish 18:00 behavior without waiting for a wall-clock scheduled run.
- Screenshots are under `output/playwright/`. Runtime demonstration records and photos are local ignored data, not shipped as real incidents.

Not exercised: external webhook delivery, a commercial lightning feed, browser geolocation permission on a physical mobile device, public/multi-user deployment, prospective model performance, or operational safety certification. There is no claim of those capabilities being validated.

For current automated results, run `.venv/bin/python -m pytest -q` and `npm --prefix frontend run build`. The generated model report is `docs/EVALUATION.md`; full measured results are `models/evaluation.json`. Source caveats and licensing are in SOURCES.md, and refresh/delivery/storage behavior is in OPERATIONS.md.
