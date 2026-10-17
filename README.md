# PlanIT Control

Environmental safety intelligence for utility construction supervisors. This local research MVP maps **60 genuine EIA power plants across 33 states**, combines public environmental feeds into an inspectable screening index, runs historically trained flood/storm classifiers, and persists crew reports and next-day briefings.

![Operations dashboard](output/ui-sprint/desktop.jpg)

This is decision support for human review. The index is not an injury probability, a low score is not clearance, and missing lightning observations do not establish safe conditions. Live provider records and explicitly labeled replay strikes are separate.

## Start locally

Requirements: Python 3.12, Node.js 22.12 or newer, npm, and internet access for environmental feeds and map tiles. The selected model, evaluation, training table, and facility inventory are included. Downloading the historical archives is unnecessary to launch.

Run from the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
```

On macOS, LightGBM may require the OpenMP runtime (`brew install libomp`) if importing it reports a missing `libomp.dylib`. Do not substitute a different classifier to hide an installation failure.

Create configuration once; this command deliberately fails if `.env` already exists:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
import secrets
configuration = Path('.env.example').read_text().replace('change-this-local-token', secrets.token_urlsafe(32))
with open('.env', 'x') as output:
    output.write(configuration)
Path('.env').chmod(0o600)
PY
```

Edit `PLANIT_USER_AGENT` in `.env` to include your real contact address for NWS requests. Keep the supervisor token private. Then run **one worker**, from this directory:

```sh
.venv/bin/python -m uvicorn backend.main:app --env-file .env --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. First-time assessments populate progressively; unavailable sources are shown explicitly. Open **Supervisor access** and enter your local `PLANIT_ADMIN_TOKEN` to enable reports, review, refresh, replay, and manual briefing generation. It is stored in browser session storage. API documentation is at `/docs`; health and refresh progress are at `/api/health`.

For frontend development, keep the backend running and run `npm --prefix frontend run dev` in another terminal. Vite proxies `/api` and `/uploads` to port 8000. Rebuild the frontend before using the backend's production page; restart the backend after its first build so it mounts the compiled assets.

## What works

- Leaflet map, ranked facilities, source timestamps, warning details, responsive site panels, index contributions, and locally accumulated score history.
- Open-Meteo modeled weather, forecast rainfall and shallow soil moisture; active point-matched NWS alerts; FEMA mapped flood exposure.
- Separate flood and severe-storm classifiers benchmarked against XGBoost, LightGBM, and logistic regression, calibrated on a separate period. The original flood estimator was retained; enhanced LightGBM won storm validation. Final evidence is mixed.
- Recorded-strike stop-work timers within 10 miles, reset to 30 minutes after the latest qualifying strike; independent labeled replay controls.
- Geolocated multi-hazard photo reports, persistent review status and manual corrections, consent-gated optional Jev text suggestions, and verified-report index contributions.
- Persistent, idempotent next-day briefings scheduled after 18:00 in each facility's IANA timezone, plus an immediately testable generation button.

Heat, rain, and soil values are screening inputs. Wildfire coverage is limited to relevant NWS alerts and crew reports; wildlife comes from crew reports. There is no active-fire perimeter feed. A commercial live lightning feed has not been connected. Jev text inference is live-verified with the version-pinned official API and tested manual handling of outages. A ten-case synthetic evaluation measured precision .75 and recall .90; authentic field reliability remains unvalidated. Text sharing and API costs are disabled by default, and photos are never sent.

## Evidence and reproduction

The original 8,532-row benchmark was reproduced and preserved under `experiments/baseline/`. The current benchmark has **21,336 county-days, 24 counties, Jan 2024–Jun 2026**. Model/threshold selection uses 2025; the final test is Jan–Jun 2026. On the final temporal development population (18 counties):

| Selected model | Precision | Recall | AP | Brier | False alerts /100 county-days |
|---|---:|---:|---:|---:|---:|
| Flood · retained original logistic | .073 | .595 | .104 | .0123 | 9.76 |
| Storm · enhanced LightGBM | .088 | .613 | .130 | .0234 | 15.56 |

These numbers are a **new benchmark**, not directly comparable with the original 2025 scores. On identical original ten counties and Jan–Jun 2026 dates, flood AP stays .120; a higher threshold reduces false alerts (41.49→13.37) and recall (.875→.625). Storm AP worsens .150→.130 while false alerts fall 40.55→19.45 and recall .812→.667. Paired uncertainty does not establish an improved storm ranking. All three independent geographic storm events were missed at its threshold. Geographic selection has only one flood and three storm events. See [full measured evaluation](docs/EVALUATION.md) and [model card](docs/MODEL_CARD.md).

The [storm investigation](docs/STORM_DIAGNOSTICS.md) reproduces the regression without refitting or changing selection. Positive-slope calibration preserves AP ranking. Selection relies on only three geographic-validation positives in one county, and the three missed test county-days represent two NOAA episodes west of all selected-model fitting coordinates. Feature/SHAP diagnostics identify plausible transfer and event-proxy problems, without establishing a single cause or an improvement. Existing artifacts and serving restrictions are preserved.

The [October 17 geographic sprint](docs/ML_SPRINT_RESULTS.md) adds an isolated 44,250-row, 50-county experiment, with new states reserved for validation and independent evaluation. Logistic regression, XGBoost and LightGBM were compared with forecast pressure, CAPE and antecedent-rain features. Neither selected candidate passed the frozen promotion gates: flood had sparse positive days and mixed temporal results; storm increased false alerts. Both served models remain unchanged. See the [source eligibility audit](docs/ML_DATA_RESEARCH.md), [frozen protocol](docs/ML_SPRINT_PROTOCOL.md) and complete matched-row results before interpreting geographic improvements.

Serving is restricted to the actual fitted county/state inventory per hazard: eleven original flood counties and eighteen storm development counties. It uses the evaluated county representative point and genuine SHAP of calibrated probability. Unsupported locations stay unknown. No national or plant-level validity is claimed. No index weight was statistically validated or changed to make a score look better.

```sh
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
# Replay the isolated sprint's frozen predictions, metrics and integrity checks:
.venv/bin/python -m scripts.ml_sprint verify
# Verify the immutable original benchmark in temporary storage:
.venv/bin/python -m scripts.reproduce_baseline
# Retrain using the included training table; overwrites the saved model and evaluation:
.venv/bin/python -m scripts.train
# Rebuild public source datasets, requiring network access:
.venv/bin/python -m scripts.ingest_sites
.venv/bin/python -m scripts.build_training
.venv/bin/python -m scripts.train
```

Raw downloads are cached under `data/raw/` with URL, retrieval timestamp, byte count, and SHA-256 manifests. Committed provenance describes the artifacts used for the measured result. A fresh NOAA archive listing can select a newer source revision and change results; retain raw files and manifests for byte-for-byte source reproducibility. Training uses fixed seeds and pinned direct Python dependencies, but numerical results can vary slightly across platforms/native libraries. Load only trusted joblib artifacts. Restart the backend after replacing a model because it caches the artifact in process.

## Optional Jev text triage

Jev is disabled by default. Keep `PLANIT_JEV_ENABLED=false` until an operator has explicitly authorized provider costs; then configure server-only `TYPESAFE_API_KEY`, pinned `PLANIT_JEV_MODEL=jev-1.13.0`, and enable it. A report's separate unchecked consent box must authorize sharing its observation text. Twelve authorized text-only inference calls were verified: one connection check, ten frozen synthetic scenarios, and one browser report. Returned token usage implies an estimated $0.00064659 at the retrieved price, not an invoice. Images and coordinate fields are never sent.

Nine independent hazard judgments use Noul, inbox routing uses Choice, and a defined 0–2 urgency scale uses Score. The dashboard filters manual labels or accepted Jev suggestions, and pending urgent/clarification/manual queues. Corrections preserve original text, AI evidence and separate review status. All ten synthetic cases abstained under the current conservative rule, which also counts no accepted hazard; 9/10 contained an uncertain judgment. These are small author-labeled examples, not field validation or calibrated safety confidence. See [implementation and measured validation](docs/JEV_VALIDATION.md) and [operations](docs/OPERATIONS.md).

Six hypothetical workflow fixtures cover normal, ambiguous, multi-hazard, lightning, negated and instruction-containing observations. Their probabilities are hand-authored API doubles: they test schema/abstention/display contracts, **not semantic classification quality**. Actual outputs, latency, usage and synthetic accuracy are recorded separately in the live evaluation above; authentic field accuracy still requires an authorized prospective evaluation. AI suggestions never verify a report, alter lightning holds, authorize work or enter the index automatically.

## Project guide

| Location | Responsibility |
|---|---|
| `frontend/src/main.tsx`, `styles.css` | Dashboard, map, forms, evaluation and briefing views |
| `backend/main.py` | API, validated mutations, assessments, scheduler, optional delivery |
| `backend/domain.py` | Pure index, distance, lightning and local-time rules |
| `backend/providers.py` | Public feeds, retries, freshness and failed-fetch cooldown |
| `backend/predict.py` | County-scoped inference and calibrated-probability SHAP |
| `backend/jev.py` | Consent-gated, pinned TypeSafe typed text decisions; manual on failure |
| `backend/store.py` | Canonical SQLite schema and transactions |
| `scripts/ingest_sites.py` | Genuine EIA inventory extraction and validation |
| `scripts/build_training.py`, `train.py` | Archived predictors, NOAA labels and measured benchmark |
| `scripts/ml_sprint.py`, `experiments/sprint-20261017/` | Isolated geographic experiment, frozen selection, independent results and integrity verification |
| `frontend/src/map.ts`, `frontend/tests/` | Shared animation-safe map resize handling and behavioral check |
| `data/`, `models/` | Committed inventory, training data, provenance and trained artifacts |
| `tests/` | Critical domain, API, persistence, provider and dataset checks |

Read [research findings](docs/RESEARCH.md), [frozen experiment protocol](docs/RESEARCH_PLAN.md), [Jev validation](docs/JEV_VALIDATION.md), [earlier Jev contract audit](docs/JEV_RESEARCH.md), [storm diagnostics](docs/STORM_DIAGNOSTICS.md), [sources and licensing](docs/SOURCES.md), [model methodology](docs/MODEL_CARD.md), [operations and API](docs/OPERATIONS.md), [generated evaluation](docs/EVALUATION.md), and the [demo and verification record](docs/DEMO.md). `PLAN.md` records milestone completion.

The restrained visual update and responsive/map checks are recorded in [UI verification](docs/UI_VERIFICATION.md), including current desktop and mobile screenshots.

## Deployment boundary

The canonical deployment is a local single-supervisor process bound to loopback. Reads, uploaded photos and API docs have no authentication; the shared supervisor token is not a multi-user identity system. There is no TLS termination, account management, audit log, distributed scheduler, retention policy, or production abuse protection. Do not expose this service publicly without designing those controls.

Free Open-Meteo service is for noncommercial use and has request limits; commercial utility use requires appropriate service terms. Public OSM tiles are a community service, not a production SLA. Webhook delivery and real strike ingestion require separately configured services and have not been tested against an external provider. Paid Jev usage was limited to the twelve authorized test calls above. No public deployment was performed.
