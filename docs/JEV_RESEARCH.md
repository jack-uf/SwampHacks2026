# Jev / TypeSafe public API research

Research date: **October 16, 2026** (America/New_York). Public documentation was retrieved during this session; local time was checked at 21:01 EDT. This is an integration research note, not an implementation or a successful API test.

## Scope and evidence standard

Only public HTTPS documentation at `https://docs.typesafe.ai/` was requested. The root HTML identified the canonical introduction and Markdown representation; the public `llms.txt` index supplied the page URLs below. The root and Markdown retrievals succeeded using curl after sandbox DNS resolution initially failed and network escalation was approved. No credentials were attached to these GET requests.

No inference request, authenticated model-list request, console session, or deliberately invalid API request was made. No secrets, local reports, or local photos were read or transmitted. No application code, configuration, tests, or data were changed for this research.

The evidence labels in this note mean:

- **Retrieved:** the public documentation page was actually downloaded and read.
- **Documented:** the retrieved page describes this contract; this does not establish deployed behavior.
- **Unverified live:** no request to the corresponding API operation was performed.
- **Recommendation:** an application design decision proposed here, not a provider guarantee.

Every API behavior below remains **unverified live**, including authentication, model availability, response validation, errors, image acceptance, billing, and latency. Example values are illustrative rather than measured.

## Retrieved primary sources

All sources in this table were retrieved on **2026-10-16 EDT**. The `.md` URLs are public representations of the official documentation, not third-party summaries. Retrieval date is not a publication date; most pages provide no publication or last-review date.

| ID | Actual retrieved URL | Evidence used |
| --- | --- | --- |
| S1 | https://docs.typesafe.ai/ | Root HTML, canonical introduction and Markdown link |
| S2 | https://docs.typesafe.ai/llms.txt | Official page index |
| S3 | https://docs.typesafe.ai/introduction.md | Typed decisions, shared state, independent questions |
| S4 | https://docs.typesafe.ai/api.md | Endpoint, auth, request/answer fields, limits, errors |
| S5 | https://docs.typesafe.ai/models.md | IDs, aliases, modalities, context, pricing, model listing |
| S6 | https://docs.typesafe.ai/images.md | Image activation, assets, transport, limits, errors |
| S7 | https://docs.typesafe.ai/primitives/box.md | Box schema, coordinates, absent-target behavior |
| S8 | https://docs.typesafe.ai/introduction/quickstart.md | Raw HTTP example and complete response envelope |
| S9 | https://docs.typesafe.ai/primitives.md | Question design and batching |
| S10 | https://docs.typesafe.ai/primitives/noul.md | Probability semantics; absence of separate confidence |
| S11 | https://docs.typesafe.ai/primitives/score.md | Ordered rubric semantics |
| S12 | https://docs.typesafe.ai/confidence.md | Confidence derived from distributions; thresholds |
| S13 | https://docs.typesafe.ai/model-jaggedness/jev-1.13.md | Provider's limitations, explicitly last reviewed 2026-10-02 |
| S14 | https://docs.typesafe.ai/sdk/python/api/exceptions.md | SDK HTTP, connection, timeout, and response-validation errors |
| S15 | https://docs.typesafe.ai/sdk/python/api/retries.md | Python retry defaults, retry headers, total retry budget |

## Endpoint and authentication

The documented evaluation operation is:

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

The endpoint is `systemone`, without a hyphen or underscore. `state`, `model`, and `questions` are required in raw HTTP requests. Question IDs are chosen by the caller and returned unchanged as answer keys; the provider says those IDs do not enter inference. This is a typed evaluation API, not an OpenAI chat-completions API: no `messages`, `response_format`, generated prose, or conversational role envelope is documented for this operation. [S3, S4, S8]

`TYPESAFE_API_KEY` is the environment-variable name used in official examples. It is an example of credential configuration, not an extra request field. Missing or invalid bearer credentials are documented as HTTP 401. Credential validity, account permissions, and deployment reachability were not tested. [S4, S8]

## Models and model discovery

The following is the **documentation's October 16 snapshot**, not a live account inventory. [S5]

| Name | Documented meaning or target |
| --- | --- |
| `jev-1.13.0` | Explicit versioned model ID |
| `jev-1.14.0` | Newest release, available October 14, 2026 |
| `jev-latest` | Stable alias pointing to `jev-1.13.0` |
| `jev-preview` | Preview alias pointing to `jev-1.14.0` |

Consequently, “newest release” does not mean the version behind `jev-latest`. Aliases may move; the response `model` identifies the version that answered. The provider recommends a versioned ID when thresholds have been tuned to a version. Box requires 1.14. [S5, S7]

Both versions have documented default context limits of **64k tokens per request** and **32k per question**. Request context is state plus all questions; per-question context is state plus that single question. No separate question-count ceiling is documented, provided both context limits are met. The optional `TypeSafe-Danger-Zone-Long-Context: true` header raises both limits to the provider's stated **262k tokens**, with lower speed and accuracy; this research did not enable it. Do not silently translate these rounded labels into an exact token boundary absent a provider clarification. [S5]

The models page quotes **$0.042 per million input tokens**, with output tokens free, for both versions. This is published pricing, not a verified invoice or a spending authorization. Organization rate limits are in the authenticated console and are expressed in tokens/second and requests/second; no numeric account limit was obtained. [S5]

The documented discovery operation is:

```http
GET https://api.typesafe.ai/v1/models
Authorization: Bearer <API_KEY>
```

Its documented response structure is `{"models":[{"name":string,"description":string,"release_date":string}]}` (structural notation, not literal JSON). The reference specifies strings, not a precise date format. It says the list currently contains aliases and that versioned IDs may be accepted even when absent from the list. No live list was fetched; list membership should therefore not be treated as an exhaustive validation rule for versioned IDs. [S5]

## Exact minimal text request and response shape

This is the smallest documented question form: a Noul with no optional criteria. The following JSON is a **constructed example** using the documented fields and pinned stable model. It was not sent.

```json
{
  "model": "jev-1.13.0",
  "state": "The customer asks for a refund.",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the state explicitly request a refund?"
    }
  }
}
```

The corresponding documented success shape is below. The numerical values are **synthetic placeholders**, not a prediction, expected fixture output, token estimate, or observed response. All envelope and Noul fields shown are required by the HTTP reference; Noul has no `confidence` property. [S4, S10]

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "refund_requested": {
      "type": "noul",
      "noul": 0.95
    }
  },
  "usage": {
    "input_tokens": 100,
    "output_tokens": 20
  }
}
```

The exact structural contract, expressed as a schema inventory rather than an independently validated provider JSON Schema, is:

| Location | Documented type / requirement |
| --- | --- |
| Request `model` | Required string |
| Request `state` | Required string, object, or array |
| Request `questions` | Required map of question ID to question object |
| Every question `type` | Required discriminator: `noul`, `choice`, `score`; experimental `box` is described separately |
| Every question `instructions` | Required; HTTP reference allows string, object, or array for questions; the Box guide demonstrates a string |
| Response `model` | Required string identifying evaluated model |
| Response `answers` | Required map; one answer per input question under the same ID |
| Every answer `type` | Required, matching the input question type |
| Response `usage` | Required object containing documented integer `input_tokens` and `output_tokens` fields |

The reference requires the `usage` object but does not mark each nested token field with a separate required flag. Examples include both token fields. No claim is made here about unknown-field rejection, empty question maps, duplicate JSON key handling, exact minimum instruction length, or additional response metadata. [S4]

## Question and answer contracts

### Noul

Question: `{"type":"noul","instructions":...}`; optional `criteria` is an object with `"true"` and `"false"` descriptions. Each description may be a string, object, or array. The criteria keys are JSON strings, not JSON boolean object keys. Answer: `{"type":"noul","noul":number}` with `noul` in [0,1], interpreted as the probability that the answer is yes. Values near 0.5 express uncertainty. It is not a degree/severity measurement and has no separate returned confidence. [S4, S10]

### Choice

Question: `{"type":"choice","instructions":...,"criteria":{"option_id":"description"}}`. Criteria are required and map labels to a string, object, array, or null description. The documented maximum is **255 options**; the HTTP reference does not establish the minimum number of accepted options. Answer fields are `type:"choice"`, `choice:string`, `probabilities:map<string,number>`, and `confidence:number`. The selected label has the highest probability, all criterion labels appear in the distribution, probabilities sum to one, and confidence is in [0,1]. Tie-breaking is not specified. [S4, S12]

Question IDs do not carry semantic meaning into inference, so put the full task in instructions. Choice option names are the selectable labels; stable application labels with explicit descriptions keep the output bounded. [S4]

### Score

Question: `{"type":"score","instructions":...,"criteria":["low","medium","high"]}`. Criteria are an ordered array of string/object/array descriptions. The provider recommends at least two levels and states that the API accepts up to **10**. Answer fields are `type:"score"`, `score:number`, `legend`, `probabilities`, and `confidence:number`. Probability and legend keys are stringified zero-based indices. Score is the probability-weighted level index, can be fractional, and spans 0 through N−1; a three-level score is not automatically on a 0–1 scale. [S4, S11]

There is a documentation typing gap: the HTTP answer table calls `legend` a map of strings, while request criteria allow structured values, and the image guide says returned legends repeat criteria references. A future client should use the exact response contract needed by its chosen rubric, preferably plain-string criteria initially, rather than assume every structured legend will be a string. This is a documentation clarification issue, not justification for multiple legacy response parsers. [S4, S6]

Confidence is derived from the probability distribution, not an independent verification of truth. Choice and Score use different formulas; the largest probability is not generally equal to confidence. Thresholds must be evaluated on the intended task and pinned model. [S12]

### Box (experimental)

Question: ``{"type":"box","instructions":"the target in `photo`"}``. No criteria are accepted according to the Box guide. It needs an image, model `jev-1.14.0` (or its preview alias), and the images activation header. The answer shape is:

```json
{
  "type": "box",
  "box": {"top": 0.25, "bottom": 0.8, "left": 0.2, "right": 0.75},
  "confidence": 0.9
}
```

This is an illustrative answer object only; it belongs under the requested ID in the normal `answers` envelope. Edges are fractions in [0,1] measured from the top-left; left < right and top < bottom. Multiply horizontal edges by image width and vertical edges by height to obtain pixels. [S7]

There is **no not-found Box result**: even an absent target yields a box. A separate presence Noul is needed before acting on it. Each question produces one box, not an array of detections. The provider explicitly describes images and Box as experimental and allows their formats, limits, and accuracy to change between releases. [S6, S7]

## Modalities and image transport

### What is actually documented

- **Text and structured state:** supported as strings, objects, and arrays. Nested JSON may contain ordinary data values; the HTTP reference restricts the top-level state to the three listed types. [S4]
- **Still images:** the image guide explicitly says Jev 1.13 and 1.14 accept images, gated by `TypeSafe-Danger-Zone-Images: true`. All question types can use images; Box additionally requires 1.14. [S6, S7]
- **Audio/video:** the models page explicitly says these are not accepted directly. Transcript text or extracted still frames are distinct preprocessing steps, not native audio/video support. [S5, S6]
- **PDF/document bytes, remote image URLs, arbitrary attachments:** no native input contract for these was established in the retrieved HTTP/image references. Do not infer support from the fact that state can contain a URL string. [S4, S6]
- **Output:** typed decisions, probabilities, confidence where documented, or a bounding box; no generated explanation, OCR transcript, image, audio, or report output contract was established. [S3, S4, S7]

**Documentation inconsistency:** S5 labels both models' input as “Text,” while S6 explicitly documents image support for both. The defensible reading is a text baseline with separately gated experimental images. It would be incorrect to report that images are impossible, or that image support has been live-verified or production-qualified. Box is also absent from S4's main question-type catalog, despite its dedicated guide. [S4, S5, S6, S7]

### Exact asset fields and placement

Images belong in a top-level `assets` map keyed by reference string. Use the reference as a string value within state, instructions, or criteria. The documented base64 asset form is:

```text
assets[reference] = {
  "type": "image",                    // optional, only this value if present
  "format": "png" | "jpg" | "jpeg",   // required with data_base64
  "data_base64": string               // standard padded base64; no data: prefix or whitespace
}
```

This is annotated structural notation, not runnable JSON. A multipart asset instead uses `{"type":"image","part":"file_part_name"}`; format is optional for multipart. The provider says `part` takes precedence if both transport fields appear, but a new implementation should send exactly one. [S6]

For multipart, the **first** part is named `request`, contains the full JSON envelope, and has `application/json` content type. Each image is a separate named file part referenced by one asset's `part`. Names must be nonempty and unique; `request` is reserved. Accepted image-part content types are `image/png`, `image/jpeg`, `image/jpg`, `application/octet-stream`, or absent. Missing parts, unclaimed parts, and two assets pointing to one part return 400. The guide recommends multipart because base64 adds roughly one third to size. [S6]

Asset references match plain substrings in string values, including within sentences, and are not replaced in object keys. Whole-string matches take priority; longer strings are scanned left to right without overlap. Every asset must be referenced or the request fails. References repeated in several places expand into several copies for model input and expanded-size accounting, even though the file uploads once. UUID-like references prevent accidental matches in ordinary text. [S6]

### Image limits

| Constraint | Documented boundary |
| --- | --- |
| Format | Still PNG or JPEG; animated PNG rejected |
| Per-file size | At most 16 MiB after base64 decoding |
| Each dimension | 280–1568 pixels inclusive |
| Pixel count | 78,400–2,458,624 inclusive |
| Wire request body | At most 32 MiB, including JSON/base64/multipart framing |
| Processed distinct images combined | At most 32 MiB; provider re-encoding may increase size |
| Expanded request | At most 32 MiB including each substituted image occurrence |
| Reference key | Nonblank, at most 1,024 UTF-8 bytes |
| All reference keys | At most 65,536 UTF-8 bytes combined |

Images are rejected rather than automatically resized. Any rejected image fails the entire request, with no partial answers. Token context limits remain separate from these byte/pixel limits. [S6]

## General input limits and error contracts

The HTTP reference specifies a **32 MiB** request body, UTF-8 JSON without a BOM, at most **64 nested object/array levels** separately in state/instructions/criteria, and finite numeric values. Exact large numeric identifiers should be strings. It supports one gzip/x-gzip or zstd compression layer, with zstd window at most 8 MiB; the body limit applies before and after decompression. Compression does not bypass image limits. [S4, S6]

| HTTP status | Documented condition | Recommended application response |
| --- | --- | --- |
| 400 | Encoding/nesting/nonfinite-number violations, malformed compression, `max_tokens_exceeded`; invalid/disabled image input and malformed assets/multipart | Fail explicitly; correct or reduce input before retrying |
| 401 | Missing/invalid API key | Fail configuration/authentication; do not retry unchanged |
| 413 | Request exceeds size limit; image guide also includes oversized file/dimensions/pixels/aggregate/expanded/reference sizes | Report the exceeded category when available; resize or split explicitly |
| 415 | Unsupported or layered content encoding | Correct transport configuration |
| 422 | Required field or question validation failure; details identify offending field | Treat as integration/input defect |
| 429 | Organization rate limit | Bounded exponential backoff; honor retry header if present |
| 529 | Provider overload | Bounded exponential backoff |

The image guide distinguishes below-minimum dimensions/pixel counts (400) from above-maximum size limits (413). It says a request with several problems reports one problem, not necessarily all. These statuses were read from documentation, not induced against the service. [S4, S6]

**Exact error JSON schema is not established.** S4 promises a JSON error body describing the problem but provides no normative envelope, required `error`/`message`/`code` keys, field-path layout, or complete example. `max_tokens_exceeded` is a documented marker, but its exact nesting is not specified. Do not fabricate an `{"error":{"code":...,"message":...}}` contract or assume framework-default 422 `detail` arrays. Header presence, request IDs, error content types for intermediary failures, and a successful response status beyond normal HTTP success semantics were not independently verified. [S4, S6]

The Python SDK exception reference distinguishes HTTP errors (with body/status/headers), connection failure, timeout, and successful HTTP responses with structurally invalid required response data; the latter includes an offending `field_path`. These are SDK exception contracts, not a wire-format definition for server error JSON. [S14]

The Python retry policy documents two retries after the initial attempt; retryable statuses 408, 429, and all 5xx; connection/timeout retries; initial backoff 0.5 seconds capped at 5 seconds with 0.25 fractional jitter; support for `Retry-After` and `retry-after-ms`; and a 30-second total retry budget. These are Python SDK defaults, not an API latency SLA or proof that an installed SDK has the same version. Avoid layering a second retry loop over the SDK. The effect of replayed requests on billing and any idempotency-key contract were not established. [S15]

## Concrete implementation recommendations (no code changed)

1. **Use one explicit contract.** For a text decision integration, pin `jev-1.13.0`, send the required raw HTTP fields, and consume the `answers` map by stable IDs. Choose either the existing project HTTP client or an intentionally selected SDK; do not add a compatibility bridge, alternate endpoint, legacy parser, or model fallback. This research did not inspect application code or installed dependency versions.
2. **Keep the task narrow.** Use Choice for bounded categories, Score for a described ordinal rubric, and Noul for a single proposition. Include an explicit “insufficient evidence” Choice option when appropriate. Batch independent questions sharing state; implement arithmetic, date comparisons, weighting, and final decisions in code. The provider notes literal-reading errors, weak numeric precision, option-order bias, distraction from irrelevant state, and susceptibility to adversarial content for 1.13. Those limitations were not retested on 1.14. [S3, S9, S13]
3. **Validate the chosen contract at the boundary.** Check required answer IDs and discriminators, finite numbers in documented ranges, expected option/index keys, Choice membership, and distribution totals with a deliberate floating-point tolerance. For Score, enforce the chosen N-level range. A missing/malformed answer is an error, not a zero probability or invented default. Report a diagnostic identifying the field and recovery action without echoing private state.
4. **Treat uncertain and unavailable as different outcomes.** Preserve model probability/confidence when valid. A timeout, authentication failure, or invalid response is a service failure, not a low-confidence judgment. Display explicit unavailability and a recovery step; do not silently substitute an old result, heuristic answer, another model, or a mock response.
5. **Keep credentials and private content server-side.** Read the key only in the authorized runtime integration, never in browser bundles. An implementation should log selected/returned model IDs, safe status categories, elapsed time, and token counts, rather than bearer headers, reports, photos, or raw state. This research neither inspected existing secret handling nor authorized sending any content.
6. **Start with text unless experimental image work is explicitly selected.** If images are needed, choose one transport (multipart is the provider recommendation), validate format/dimensions/size first, send the images header, and use collision-resistant references. Pin 1.14.0 if Box is required. Gate boxes with a presence question and evaluate on authorized task-specific samples before presenting them as reliable detections. No universal Box confidence threshold is documented. [S6, S7]
7. **Do not invent generation or arbitrary file ingestion.** For full report prose, unconstrained extraction, or OCR transcription, define a separate supported component only if that work is requested. Jev's documented role is selecting/evaluating typed decisions; sending a report filename or URL as state does not establish document ingestion. [S3, S4, S13]
8. **Handle failures with a bounded policy.** Do not retry deterministic 400/401/413/415/422 failures unchanged. Retry transient conditions within a visible timeout budget. Surface a safe generic provider failure when an error body cannot be interpreted; this is explicit failure reporting, not an alternate success path. Obtain a real sanctioned error example before depending on provider error JSON keys. [S4, S6, S15]

## Remaining verification and maintenance

Before claiming a working integration, a separately authorized exercise would need to confirm bearer authentication, the pinned model's availability, a synthetic text request/response, actual error envelopes, and the selected timeout/retry policy. Image work would additionally need a nonprivate approved synthetic image and boundary checks for the experimental contract. None of those API exercises occurred here, and this document is not permission to run them.

Revisit the model aliases, image activation/limits, Box schema, and error contract when the provider releases a model or SDK change. Resolve the text-versus-image table and legend typing gaps with the provider rather than implementing speculative dual behavior. Maintain the retrieved-source date alongside updates so readers can distinguish a fresh verification from an old assumption.

This research's only repository deliverable is `docs/JEV_RESEARCH.md`. It does not certify model quality, confidentiality terms, medical or other domain suitability, latency, actual token costs, account rate limits, or production readiness.
