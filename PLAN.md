# PlanIT Control implementation plan

This repository starts empty. The canonical implementation uses FastAPI, SQLite,
React/TypeScript, and Leaflet. It is a local single-supervisor research MVP.

- [ ] Data foundation: validated, deduplicated EIA-860M inventory; cached weather and NWS feeds with freshness.
- [ ] ML: reproducible NOAA county-day labels and archived forecast features, chronological benchmark and held-out evaluation.
- [ ] Dashboard: map, ranked facilities, source-aware details, decomposed operational index.
- [ ] Operations: deterministic lightning, reviewed photo reports, persistent timezone-aware 18:00 briefings.
- [ ] Verification: critical tests, production frontend build, local end-to-end demo, documentation.

Scope reaches ingestion scripts and cached source manifests; backend domain rules,
API validation, persistence and scheduler; model features, artifacts and evaluation;
frontend API callers, map and accessible forms; tests and setup/demo documentation.
No existing callers or local state exist to preserve. Safety risks are stale feeds,
county/site resolution mismatch, forecast leakage and unsupported clearance claims.
