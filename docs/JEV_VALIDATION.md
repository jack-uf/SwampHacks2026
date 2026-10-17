# Jev text triage: implementation and validation

This document describes the current text-only report enrichment workflow and how to reproduce its checks. It separates provider documentation, offline workflow tests, synthetic live evaluation and field validation. None establishes work clearance or a probability of worker injury.

## Request and trust boundary

`backend/jev.py` uses the official `POST https://api.typesafe.ai/v1/systemone` HTTP API with a server-only bearer credential. The current default is the versioned stable model `jev-1.13.0`. It makes one request per consented submission, with a 15-second timeout and no automatic retries. `state` contains only the observation text. Photos, coordinate fields, site/report IDs and supervisor credentials are not included. Text itself can contain sensitive information: consent must cover its contents.

The operator enables `PLANIT_JEV_ENABLED=true` and supplies `TYPESAFE_API_KEY` in the backend runtime or ignored project `.env`. A terminal `export` does not update an already running Codex/backend process; restart that process from the configured environment or use `.env`. Uvicorn loads `.env` when started with `--env-file .env`. Never paste a key into chat, a URL, a screenshot, a tracked file or a command argument.

Fourteen independent questions share that text:

| Primitive | Question(s) | Interpretation |
| --- | --- | --- |
| Noul | Nine hazard-presence questions | Concurrent heat, storm, flood, lightning, wildfire, wildlife, unstable/saturated terrain, equipment/access obstruction and other hazards |
| Noul | Immediate review, location sufficiency, clarification | Separate evidence/ambiguity checks, never a final safety decision |
| Choice | Supervisor inbox | Exactly one of `hazard_review`, `clarify_report`, `routine_log`; labels above remain independent |
| Score | Urgency of human follow-up | Ordered, descriptive levels 0 routine follow-up, 1 prompt review of a current hazard, 2 immediate review of explicit current exposure |

Typed responses require all expected question IDs, correct discriminated answer types, finite probabilities in [0,1], complete normalized distributions, a highest-probability Choice selection, the exact Score legend, a score consistent with the weighted level mean, nonnegative integer usage and the pinned model. Distribution sums allow .001 rounding error; the weighted score allows .01. Invalid responses retain manual classification and store an unavailable reason. Provider bodies and arbitrary response headers are not echoed or logged.

Noul acceptance ≥.7 and rejection ≤.3 leave intermediate values uncertain. Choice/Score confidence below .7 also requires clarification. These cutoffs are hand-selected and unvalidated. Accepted labels are retained even when another question is uncertain. `abstained` means at least one question is uncertain or no hazard was accepted; a routine observation with no hazard therefore counts as abstention, even with confident negative judgments. That rate must not be read as a pure confidence measure.

The review queue is `urgent` when immediate-review Noul or level-2 urgency probability reaches .7; otherwise it is `clarification` for uncertainty, a clarification judgment or the clarification inbox, and `standard` otherwise. Urgent reports can also require clarification; both are displayed. Every report remains pending until a separate supervisor review. A routine inbox or urgency zero never dismisses a report, clears a lightning hold or authorizes work.

## Persistence and dashboard behavior

The original text and manual category set are saved with the provider's validated judgments, distributions, model and token usage in the report/triage transaction. A correction replaces only manual categories, records its timestamp/source, and leaves provider judgments and review status intact. Correction failure keeps the editor open. There is no automatic promotion of suggestions to verified observations.

Hazard filtering matches either a manual category or an accepted Jev suggestion. The timeline shows each source separately. The same filter applies to report map pins; rejected reports never have a pin. Pending queues filter urgent, clarification and unavailable/manual enrichment. Resolved/rejected/verified reports remain accessible under all statuses. Filters can hide reports, so the dashboard states matching and total counts. Lightning timers, geographic model eligibility, missing-feed coverage and briefings use their existing independent paths.

## Frozen synthetic evaluation

`tests/fixtures/jev-scenarios.json` contains ten nonsensitive, fictional scenarios with labels written before inference. It includes routine and irrelevant text, ambiguous hearsay, resolved/negated hazards, simultaneous water/wildlife/terrain/equipment concerns, current lightning/heat/fire exposure, another hazard, and an instruction embedded in the observation. The author supplied the labels; they are neither independent adjudication nor authentic field observations. No field corpus exists in this repository.

`docs/jev-evaluation-protocol.json` freezes the corpus and question hashes, pinned model, cutoffs, planned calls and cost estimate. Changes fail the live runner before spending; a changed experiment needs a newly declared protocol and separate output. Do not tune the prompt against these ten scenarios and present a rerun as independent validation.

```sh
# Offline plan only; no network or spending.
.venv/bin/python -m scripts.evaluate_jev

# Explicitly paid; requires operator enablement and local credential.
.venv/bin/python -m scripts.evaluate_jev --live --output output/jev-evaluation.jsonl
```

The runner makes at most ten sequential calls, stops on the first unavailable/invalid response, never retries and refuses to overwrite evidence. JSONL is flushed after every response so completed paid calls survive interruption. It records the protocol, fictional text, expected labels, request hash, validated actual answers, usage, latency and summary. No authorization headers or keys are stored. CI never invokes `--live`.

The pre-call estimate is 5,000 input tokens per scenario × 10 × $0.042/million = **$0.0021**, not a hard token ceiling or an invoice. Actual usage and the same published unit price produce a separate estimate. Output is documented as free. Unknown usage for failed requests cannot be reconstructed. Pricing and account permissions must be rechecked for future runs or larger datasets.

Metrics include multi-label micro/per-class precision and recall, macro averages excluding undefined class ratios, disagreements, response coverage, abstention, uncertainty, clarification routing, inbox accuracy, ordinal urgency mean absolute error and latency. Classification metrics include accepted labels on abstained responses; failed requests are counted separately and excluded from those classification denominators. Null ratios mean undefined, not zero. Ten synthetic examples cannot establish calibration, robust confidence intervals, generalization or operational safety. Urgency disagreement is against subjective author expectations, not measured incident severity.

## Validation status

The typed contract, persistence/corrections and failure handling have passing offline tests. Live inference, measured synthetic results and browser evidence are recorded here after execution; no mocked output is a semantic performance measurement.

## Official documentation checked

The following official pages were retrieved again on October 16, 2026 (America/New_York):

- https://docs.typesafe.ai/api.md — endpoint, authentication and envelope.
- https://docs.typesafe.ai/primitives/choice.md — Choice criteria, distribution, confidence and selected option.
- https://docs.typesafe.ai/primitives/score.md — ordered levels, string-indexed HTTP legend/distribution and weighted mean.
- https://docs.typesafe.ai/models.md — versioned 1.13 stable pin, 1.14 release October 14, pricing and context limits.
- https://docs.typesafe.ai/images.md — separately gated experimental images, asset references and upload limits.

Images require `TypeSafe-Danger-Zone-Images: true`, PNG/JPEG assets and documented size constraints; Box requires Jev 1.14. The models overview labels input “Text” while the dedicated image guide documents gated image support. No image inference is implemented or authorized by the text-only sharing checkbox. No crew photograph has been transmitted. Image behavior, capability quality and account access remain unverified. See `docs/JEV_RESEARCH.md` for the earlier detailed documentation snapshot, whose statements about no live calls describe that earlier research pass.
