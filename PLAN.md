# PlanIT Control implementation plan

This repository starts empty. The canonical implementation uses FastAPI, SQLite,
React/TypeScript, and Leaflet. It is a local single-supervisor research MVP.

- [x] Data foundation: validated, deduplicated EIA-860M inventory; cached weather and NWS feeds with freshness.
- [x] ML: reproducible NOAA county-day labels and archived forecast features, chronological benchmark and held-out evaluation.
- [x] Dashboard: map, ranked facilities, source-aware details, decomposed operational index.
- [x] Operations: deterministic lightning, reviewed photo reports, persistent timezone-aware 18:00 briefings.
- [x] Verification: critical tests, production frontend build, local end-to-end demo, documentation.

Scope reaches ingestion scripts and cached source manifests; backend domain rules,
API validation, persistence and scheduler; model features, artifacts and evaluation;
frontend API callers, map and accessible forms; tests and setup/demo documentation.
No existing callers or local state exist to preserve. Safety risks are stale feeds,
county/site resolution mismatch, forecast leakage and unsupported clearance claims.

## Completion evidence

- 60 genuine, deduplicated EIA facilities across 33 states; source/provenance manifests committed.
- 8,532 complete archived-forecast county-days; six calibrated candidate classifiers benchmarked, with 2025 temporal and Los Angeles geographic holdouts. Both validation winners are logistic regression.
- Real Open-Meteo, NWS and FEMA queries exercised; missing inputs remain unknown.
- Live/replay separation, photo upload/review, index updates and immutable briefing history exercised in the browser and persisted across server restart.
- 14 automated tests pass; production TypeScript/Vite build passes; desktop/mobile browser verification has no console errors or warnings.
- README and source, model, operations, evaluation and demonstration documents completed. Screenshots retained in output/playwright/.

Scope boundaries: local single-supervisor MVP; no public deployment, external delivery,
commercial lightning feed or verified Jev integration. Models are experimental county-event
screens, have high false-alert burdens, and do not certify safe work. See docs for exact
feed cadence, index formulas, reproducibility, authentication and storage limits.
