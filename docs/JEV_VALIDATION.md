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

Verified on **October 16, 2026 EDT** (the UTC records are October 17). Twelve actual authenticated `systemone` requests succeeded with `jev-1.13.0`: one small connection check, the ten frozen scenarios, and one consented fictional browser report. Each returned validated Noul, Choice and Score answers. None sent a photo or coordinate field. No automatic retry, model-list probe or deliberate paid error request was made.

| Evidence | Calls | Returned input / output tokens | Estimated USD |
| --- | ---: | ---: | ---: |
| `output/jev-connection.json` | 1 | 1,270 / 274 | 0.00005334 |
| `output/jev-evaluation.jsonl` | 10 | 12,815 / 2,747 | 0.00053823 |
| `output/jev-browser-verification.json` | 1 | 1,310 / 274 | 0.00005502 |
| Total | 12 | 15,395 / 3,295 | **0.00064659** |

These are estimates from returned usage and the retrieved input-token price, not account billing verification. The connection check and browser report are workflow evidence and are excluded from the frozen evaluation metrics.

### Measured synthetic results

All ten planned responses succeeded. The accepted hazard labels produced 9 true positives, 3 false positives and 1 false negative: **micro precision .75, recall .90**. Macro precision was .7963 and recall .8889. Choice routing agreed with all ten author labels; urgency mean absolute error was .006 on the 0–2 scale. Mean request latency was .2634 seconds in this single sequential run.

| Hazard | TP / FP / FN | Precision | Recall |
| --- | ---: | ---: | ---: |
| Heat | 1 / 1 / 0 | .50 | 1.00 |
| Storm | 1 / 0 / 0 | 1.00 | 1.00 |
| Flood | 1 / 0 / 0 | 1.00 | 1.00 |
| Lightning | 1 / 0 / 0 | 1.00 | 1.00 |
| Wildfire | 1 / 0 / 0 | 1.00 | 1.00 |
| Wildlife | 1 / 0 / 0 | 1.00 | 1.00 |
| Unstable terrain | 1 / 0 / 0 | 1.00 | 1.00 |
| Equipment/access obstruction | 2 / 1 / 0 | .6667 | 1.00 |
| Other | 0 / 1 / 1 | .00 | .00 |

The disagreements were an additional `other` label on the multi-hazard report, an additional `heat` label on advancing wildfire, and `equipment_obstruction` instead of `other` for broken glass across a pedestrian doorway. Category boundaries and urgency labels were authored locally and not independently adjudicated; some overlap is arguable. Expected labels remain unchanged after inference.

**Abstention was 10/10, uncertainty 9/10, and the clarification queue 6/10**; the other four entered urgent review and also required clarification. The no-hazard cases contribute to abstention by definition. Vague fictional locations, the broad `other` question and additional review judgments contributed to uncertainty. Thus apparent routing agreement and very small urgency error do not establish useful field reliability. Do not adjust cutoffs or question wording against these cases and reuse their scores as independent evidence. A future experiment needs new preregistered cases, independently adjudicated authentic field observations with sharing authorization, and a separately held-out evaluation.

### Abstention rule v2 and the held-out run

On 14 valid held-out responses, the v2 rule cut abstention from 14/14 to 5/14 without changing any review queue. Of the 9 decisive answers, 7 matched the author labels exactly and 2 over-labeled a no-hazard report. Both of those stayed in the clarification queue and pending.

The v1 rule above treats a confident "no hazard" as abstention and lets location or clarification uncertainty block the hazard labels. `jev.abstention` (rule `hazard-decisive-v2`) abstains only when a specific hazard question sits between 0.3 and 0.7, when `other` is uncertain and no specific hazard was accepted, or when inbox confidence is below 0.7. The review queue, cutoffs, prompts and pinned model are unchanged. `abstained_v1` is still returned for comparison. `tests/test_jev.py` replays the ten stored live responses and checks that labels, uncertainty lists, queues and the v1 flag reproduce exactly.

I designed v2 after reading the ten frozen responses, so those ten are development data. On them, v2 abstains on 3/10 and 5 of 7 decisive answers match. That number is not evidence.

The held-out method: 16 new fictional, author-labeled cases (`tests/fixtures/jev-heldout.json`) and `docs/jev-heldout-protocol.json`, which pins the corpus hash, question hash, model and a SHA-256 of the `jev.abstention` source. Both were committed in `405ca32` before any held-out call. The runner made one sequential pass with no retries (`output/jev-heldout-evaluation.jsonl`). Both rules are scored on the same responses by `scripts.evaluate_jev --compare`.

| Rule, held-out responses | Abstained | Decisive | Decisive exact | Decisive wrong | Abstained but exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (original) | 14 / 14 | 0 | n/a | 0 | 8 |
| v2 `hazard-decisive-v2` | 5 / 14 | 9 | 7 (.778) | 2 | 1 |

What happened in the run:

- 15 of 16 cases were sent. The 15th, a prompt-injection case (`injection-hornets`), returned a response that failed strict contract validation, so PlanIT kept manual classification. The runner stops on the first failure, so `meeting-reminder` was never sent. Raw provider bodies are not stored, so the exact validation failure is not recorded.
- Label metrics over the 14 valid responses: micro precision .722, recall .867 (13 TP, 5 FP, 2 FN). Routing matched all 14 author labels; urgency mean absolute error .095.
- The two decisive errors: `resolved-water` (drained water, Jev accepted flood) and `hearsay-bear` (secondhand rumor, Jev accepted wildlife and other). Both are false positives on reports a human would close.
- v2 still abstained on the wind miss (`wind-lift`, storm not accepted) and on the `other`-only case (`damaged-cord`).
- Returned usage was 17,952 input and 3,848 output tokens, about $0.00075 at the retrieved input price. That is an estimate, not an invoice.

Limits: the same author wrote the cases and labels, 14 responses cannot support confidence intervals, and the cases are fictional. The rule now presents more answers as decisive, which makes over-labeling visible as confident suggestions. That is acceptable only because suggestions never verify a report, set a JHA rating or skip the review queue.

### Workflow and failure checks

The browser report returned five hazard suggestions, a low-confidence routine inbox and a clarification queue. Its explicit fictional framing may affect routing; that observation is not part of the measured ten-case suite. The UI retained all accepted suggestions, displayed the different inbox and urgency judgments, and left it pending. A supervisor added flood and equipment obstruction to the manual labels. Reloading preserved the correction, original text and exact original provider object while status remained pending. Filtering on `other`, which was only an AI suggestion, included the report and its map pin; wildfire gave an explicit empty state. Clarification and manual-only queues were exercised separately. The fictional test reports were rejected afterward, retaining their audit records and removing their map pins.

The consent box started unchecked. A separate no-consent browser submission preserved manual labels and an unavailable reason without a provider result. Offline tests intercept HTTP and verify zero requests for missing consent, disabled configuration or missing credentials; strict parsing and one-shot handling of 401/403, 429, 529, connection failures, timeouts and malformed/unexpected-model responses; and atomic report/triage persistence and corrections. These simulated failures verify application behavior, not the provider's deployed error behavior. No failure was deliberately induced against the paid service.

Desktop (1,440 pixels) and mobile (390 pixels) views fit their viewport widths, and browser console checks returned no errors or warnings. Evidence is in `output/playwright/jev-live-desktop.png`, `jev-live-mobile.png` and `jev-live-mobile-detail.png`; `output/jev-browser-verification.json` retains before/after provider and human records. See `docs/DEMO.md` for the complete regression verification record and commands.

## Official documentation checked

The following official pages were retrieved again on October 16, 2026 (America/New_York):

- https://docs.typesafe.ai/api.md — endpoint, authentication and envelope.
- https://docs.typesafe.ai/primitives/choice.md — Choice criteria, distribution, confidence and selected option.
- https://docs.typesafe.ai/primitives/score.md — ordered levels, string-indexed HTTP legend/distribution and weighted mean.
- https://docs.typesafe.ai/models.md — versioned 1.13 stable pin, 1.14 release October 14, pricing and context limits.
- https://docs.typesafe.ai/images.md — separately gated experimental images, asset references and upload limits.

Images require `TypeSafe-Danger-Zone-Images: true`, PNG/JPEG assets and documented size constraints; Box requires Jev 1.14. The models overview labels input “Text” while the dedicated image guide documents gated image support. No image inference is implemented or authorized by the text-only sharing checkbox. No crew photograph has been transmitted. Image behavior, capability quality and account access remain unverified. See `docs/JEV_RESEARCH.md` for the earlier detailed documentation snapshot, whose statements about no live calls describe that earlier research pass.
