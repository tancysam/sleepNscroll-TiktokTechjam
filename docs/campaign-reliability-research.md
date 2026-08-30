# Campaign reliability research

_Primary-source review, 2026-08-29. “OpenAI-compatible” is treated as an HTTP-shape claim, not proof that a third-party endpoint implements OpenAI’s schema subset, token accounting, retry semantics, or error envelope._

## Decision summary

1. Do not treat a JSON Schema `maxLength` as evidence that a model can return that much file content. Generated source embedded in JSON is bounded first by the selected model/provider’s output-token ceiling, the request’s completion-token ceiling, and the client’s response-byte ceiling. JSON syntax, escapes, and—where applicable—reasoning tokens consume the same completion allowance.
2. Keep strict local parsing even when `json_schema` is enabled. Bound bytes before parsing, decode strict UTF-8, reject duplicate object names and non-finite numbers, require one top-level object, validate exact fields/types/ranges, and only then produce a deterministic local serialization and digest.
3. Make retries deadline-aware and error-specific. A real rate-limit response can be retried with bounded `Retry-After`/exponential backoff and jitter; billing, spend, and quota 429s must fail closed. A timed-out or disconnected POST is an ambiguous side effect, not proof that the provider did no work.
4. Make provider attempts durable state transitions, not in-memory telemetry. Persist the attempt identity and budget reservation before dispatch; persist the raw response digest and outcome after dispatch; recover an interrupted in-flight attempt as `UNKNOWN` unless the provider supplies an idempotency/reconciliation contract.

## 1. Structured output when strings contain file contents

### Verified limits and edge cases

- OpenAI Structured Outputs guarantees conformance only to its supported JSON Schema subset. The root must be an object, every field must be required (nullable unions emulate optional fields), and every object must set `additionalProperties: false`. A schema may contain at most 5,000 object properties and 10 nesting levels; the 120,000-character limit covers **schema strings** such as property names, definition names, enum values, and const values—not generated string values. [OpenAI Structured Outputs: supported schemas and limits](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)
- `minLength`, `maxLength`, `pattern`, and `format` are explicitly unsupported for strings on fine-tuned models. Supplying an unsupported keyword with `strict: true` causes an API error. Therefore the exact generated-package schema must be capability-tested against every configured endpoint/model, and local validation remains authoritative even if a provider accepts the request. [OpenAI Structured Outputs: unsupported keywords](https://developers.openai.com/api/docs/guides/structured-outputs#some-type-specific-keywords-are-not-yet-supported)
- `max_completion_tokens` is an upper bound over visible output **and reasoning tokens**. The model itself also has a maximum-output limit; for example, OpenAI currently advertises 128,000 maximum output tokens for the GPT-5.6 family. A locally accepted configuration value cannot enlarge the endpoint/model ceiling. [OpenAI Python request type for `max_completion_tokens`](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py), [OpenAI model limits](https://developers.openai.com/api/docs/models)
- A JSON string must escape quotation marks, backslashes, and U+0000–U+001F control characters. Source code with quotes, backslashes, tabs, and newlines therefore occupies more serialized bytes than the decoded file content. [RFC 8259, Sections 7 and 9](https://www.rfc-editor.org/rfc/rfc8259.html#section-7)
- A strict outer schema that declares `content` as a string validates only that it is a string (and any supported string assertions). It does not prove that the string is complete Python, valid JSON, valid TOML, or another syntactically valid file. JSON Schema’s `contentEncoding`, `contentMediaType`, and `contentSchema` vocabulary is annotation-oriented; implementations must not automatically decode and validate embedded content by default, and failure to parse an embedded document does not by itself invalidate the containing string. [JSON Schema 2020-12 Validation, Section 8](https://json-schema.org/draft/2020-12/json-schema-validation#section-8)
- A completion stopped for `length` is incomplete. A refusal need not follow the requested schema, and `content_filter` can interrupt output. OpenAI’s examples explicitly branch on all of these conditions; only an unrefused completion with `finish_reason == "stop"` is a candidate for parsing. [OpenAI Structured Outputs: incomplete responses and refusals](https://developers.openai.com/api/docs/guides/structured-outputs#how-to-use-structured-outputs)
- JSON mode (`response_format: {"type":"json_object"}`) guarantees valid JSON only in the non-edge case; it does not guarantee schema adherence, and the application must detect incomplete output. [OpenAI Structured Outputs: JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs#json-mode)

### Consequence for this repository

The current generated-package contract permits up to 12 files and up to 512 Ki characters in each `content` value ([`schemas.py`, lines 31 and 1556–1574](../src/kuairand_agent/research/schemas.py)), while the production configuration requests 65,536 completion tokens and caps the HTTP response at 8 MiB ([`full-pure.toml`, lines 33–39](../configs/full-pure.toml)). Those are independent ceilings. The theoretical schema maximum is not a deliverable response-size promise, and a response can exhaust its token allowance long before reaching either decoded-character or response-byte limits.

The safe contract is therefore:

- capability-probe each endpoint at startup with the **exact** `Proposal`, `GeneratedPackage`, and `Reflection` schemas and configured token parameter;
- set an operation-level total decoded-source budget substantially below the verified visible-output budget, not merely a per-file `maxLength`;
- request one bounded atomic patch or a small file set per call; for larger work, use an explicit controller-owned chunk protocol with request ID, file path, chunk index, and terminal chunk count, then assemble and validate only after every chunk is durably present;
- represent a known-shape JSON artifact as an actual nested JSON value and serialize it locally where the Structured Outputs subset permits that closed shape; otherwise independently parse/validate embedded JSON, and syntax-check or compile other generated file types before admission;
- reject and record incomplete/refused/content-filtered outputs; never salvage a truncated JSON suffix or materialize a partial package;
- keep the response-byte limit before parsing and independently cap decoded file count, each file, and the aggregate decoded source size.

These are design inferences from the documented limits. No fixed character-to-token conversion is safe across models or source languages.

## 2. Safe JSON parsing and normalization

### What strict parsing must defend against

- RFC 8259 says object member names should be unique because receiver behavior for duplicates is unpredictable: implementations variously keep the last value, reject the object, or expose every pair. It also permits parsers to impose their own text-size, nesting, numeric, and string limits. [RFC 8259, Sections 4 and 9](https://www.rfc-editor.org/rfc/rfc8259.html#section-4)
- Python’s `json` decoder is deliberately permissive by default: it accepts `NaN` and infinities and keeps only the last duplicate object member. `object_pairs_hook` can detect duplicates and `parse_constant` can reject non-finite literals. Python also warns that untrusted JSON can consume substantial CPU and memory and recommends limiting input size before parsing. [Python `json` documentation: compliance and hooks](https://docs.python.org/3/library/json.html#standard-compliance-and-interoperability), [Python `json` security warning](https://docs.python.org/3/library/json.html)
- I-JSON requires UTF-8, forbids duplicate object member names, and limits interoperable exact integers to the IEEE-754 range unless the application uses an explicitly understood string representation. [RFC 7493, Sections 2.1–2.3](https://www.rfc-editor.org/rfc/rfc7493.html#section-2)
- Canonicalization is not generic “cleanup.” JCS requires I-JSON input, deterministic property sorting and primitive serialization, rejection of invalid Unicode/non-finite numbers, and preservation of parsed string data without Unicode normalization. [RFC 8785, Sections 3.1–3.2](https://www.rfc-editor.org/rfc/rfc8785.html#section-3)

### Recommended acceptance pipeline

1. Read at most `max_response_bytes + 1`; reject over-limit bodies before decoding or parsing.
2. Decode as UTF-8 with strict error handling. Do not strip Markdown fences, search for the first `{`, or otherwise “repair” the provider envelope.
3. Parse exactly one JSON value with duplicate-key rejection and non-finite-number rejection. Require the envelope and nested model content to be objects at their respective boundaries.
4. Validate the provider envelope separately from model content: exact choice count/index, completion status, assistant role, no tool/function call, refusal shape, token-usage consistency, and bounded identifiers.
5. Validate nested model content with exact field sets and exact scalar types. In Python, remember that `bool` is a subclass of `int`; numeric validators must reject booleans explicitly. Enforce counts, aggregate decoded-source bytes/characters, path policy, and cross-field identities locally even after Structured Outputs.
6. Preserve generated file strings exactly through validation. Any line-ending or Unicode normalization is a source transformation and needs a new content digest and explicit policy; it is not JSON canonicalization.
7. Serialize accepted metadata through one versioned deterministic profile with finite-number rejection, then hash the resulting bytes. Do not label a `json.dumps(sort_keys=True, ...)` profile “JCS” unless it implements RFC 8785’s exact UTF-16 property ordering and number/string serialization.

The repository already implements the two most important Python hardenings—duplicate-key rejection through `object_pairs_hook`, non-finite rejection through `parse_constant`, and `allow_nan=False` on deterministic output ([`schemas.py`, lines 44–85](../src/kuairand_agent/research/schemas.py)). Preserve those checks in both the outer provider envelope and nested content. The remaining reliability gate is to ensure the aggregate decoded source budget and all semantic/cross-field checks are applied before any filesystem materialization.

## 3. HTTP retries, deadlines, timeouts, and cancellation

### Retry classification

OpenAI’s official Python SDK retries connection errors, 408, 409, 429, and 5xx twice by default with short exponential backoff. It also defaults to a ten-minute timeout, supports granular connect/read/write/pool timeouts, raises `APITimeoutError`, and retries timeouts by default. These defaults are evidence of OpenAI SDK policy, not a guarantee that every OpenAI-compatible provider has identical semantics. [OpenAI Python SDK: retries and timeouts](https://github.com/openai/openai-python#retries)

OpenAI’s current error guide adds two critical distinctions:

- For a genuine rate-limit 429, honor `Retry-After`; if absent, use exponential backoff with jitter and a bounded retry count.
- Credit-balance, organization/project spend-limit, and organization usage-limit 429s are not repaired by retrying. The client must inspect the error code rather than classifying every 429 as transient. [OpenAI error codes and retry guidance](https://developers.openai.com/api/docs/guides/error-codes#api-errors)

HTTP itself defines `Retry-After` as either an HTTP date or a non-negative integer delay. RFC 9110 also says a client should not automatically retry a non-idempotent method unless it knows the request semantics are idempotent or can determine that the original request was never applied. Chat Completions uses POST, so a connection loss or timeout after dispatch can mean “provider may have completed and billed this request, but the client did not receive the response.” [RFC 9110, Sections 9.2.2 and 10.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)

A safe table for the campaign controller is:

| Condition | Default action |
| --- | --- |
| 400, 401, 403, 404, 422 | Fail closed; do not retry unchanged input. |
| 429 with a rate-limit code | Bounded retry; honor capped `Retry-After`, otherwise exponential backoff plus jitter. |
| 429 with credit/spend/usage/quota code | Non-retryable until external account state changes. |
| 408, 409, 5xx | Bounded retry only under the endpoint’s verified policy and remaining campaign deadline. Honor a valid `Retry-After` on eligible responses, including 503. |
| Connection loss or timeout after POST dispatch | Record `UNKNOWN`/unaccounted attempt. Retry only if the configured provider contract explicitly accepts duplicate generation/cost risk. |
| HTTP 200 with malformed or schema-invalid content | Count usage if known, persist the invalid response, and allow at most the configured semantic repair; this is a new generation, not a transport retry. |
| `length`, refusal, or `content_filter` | Do not accept. For `length`, retry only with a deliberately smaller/chunked contract, not an unchanged request. |

Every attempt record should include endpoint slot, model, operation, stable attempt ID, request digest, response status, provider response/request ID where available, `Retry-After`, start/end monotonic timestamps, timeout budget, response digest, known usage, estimated cost, and outcome. OpenAI exposes `x-request-id` on success and failures specifically for debugging; retain it without retaining credentials. [OpenAI Python SDK: request IDs](https://github.com/openai/openai-python#request-ids)

### Deadline and cancellation semantics

`urllib.request.urlopen(timeout=...)` documents its timeout as applying to blocking operations such as connection attempts; it is not documented as a single absolute wall-clock deadline for the entire upload plus response stream. A slow sequence of successful blocking reads can therefore outlive a campaign’s nominal remaining time. [Python `urllib.request.urlopen`](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlopen)

Python threads also cannot be destroyed, stopped, or interrupted, and a running `concurrent.futures.Future` cannot be cancelled. Wrapping blocking `urlopen` in a thread and setting an event therefore does not provide hard cancellation. [Python `threading` limitations](https://docs.python.org/3/library/threading.html), [Python `Future.cancel`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel)

The controller should use one absolute monotonic deadline and derive every wait from it:

- before dispatch, persist the attempt and reject if no useful time remains;
- set each transport phase/attempt timeout to no more than remaining time minus a bounded persistence/finalization reserve;
- check cancellation before dispatch, during backoff, immediately after response receipt, and before any retry or materialization;
- cap `Retry-After` by both policy and remaining time; do not sleep through the finalization reserve;
- if prompt cancellation during an active request is required, use a transport with cooperative cancellation (for example, an async task whose I/O is cancellable) or an isolated process that owns no campaign database locks/shared queues. Hard process termination is itself hazardous when shared locks, pipes, or queues are involved. [Python `asyncio` task cancellation and timeouts](https://docs.python.org/3/library/asyncio-task.html#task-cancellation), [Python `multiprocessing` termination warning](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.terminate)

The current adapter correctly clips its configured attempt timeout to reported remaining research time and rejects non-`stop` completions ([`provider.py`, lines 669–705 and 789–825](../src/kuairand_agent/research/provider.py)). Its remaining gaps are that `urllib` does not establish a hard end-to-end deadline, transport failures are retried immediately, `Retry-After` is only used for 429, and all 429 responses are initially classified as retryable ([`provider.py`, lines 844–922](../src/kuairand_agent/research/provider.py)).

## 4. Durable campaign progress and state

### Storage facts

- SQLite transactions provide atomic commit for one database: all changes in a transaction occur or none do. [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html)
- In WAL mode, readers and a writer can proceed concurrently, but there is only one writer at a time. `synchronous=FULL` adds a WAL sync after each commit and is documented as ACID in WAL mode. [SQLite WAL concurrency](https://www.sqlite.org/wal.html#concurrency), [SQLite `PRAGMA synchronous`](https://www.sqlite.org/pragma.html#pragma_synchronous)
- The `-wal` file is part of the persistent database state. Copying or moving the `.sqlite3` file without its live WAL can lose committed transactions or corrupt the copy. Use SQLite’s backup mechanism or close/checkpoint correctly; do not treat the main file alone as a checkpoint bundle. [SQLite WAL persistent state](https://www.sqlite.org/wal.html#the_wal_file)
- WAL transactions spanning multiple attached databases are atomic per database but not atomic across the databases as a set. Two independent campaign/ledger databases are likewise not one atomic transaction. [SQLite WAL overview](https://www.sqlite.org/wal.html#overview)

The repository’s `CampaignStore` already chooses a strong base: one intended writer, append-only evidence, optimistic revisions, reservation before external action, `journal_mode=WAL`, and `synchronous=FULL` ([`store.py`, lines 1–16, 1033–1049, and 1346–1418](../src/kuairand_agent/campaign/store.py)). Its conservative project-ledger-first reservation correctly prefers consuming a scarce slot over accidentally spending it twice when two independent databases cannot commit atomically.

### Required state protocol around external side effects

Database atomicity does not extend across an HTTPS POST, a spawned training process, or a filesystem artifact write. The reliable pattern is therefore a state machine with explicit uncertainty:

1. In one SQLite transaction, reserve budget and append `ATTEMPT_PREPARED` with immutable attempt ID, request digest, endpoint/model, operation, deadline, and parent revision.
2. Dispatch the external action. Do not hold a SQLite write transaction or database lock while waiting on network/process I/O.
3. Persist the bounded raw response/artifact first (or its content-addressed bytes plus verified digest), then atomically append one terminal attempt event: `SUCCEEDED`, `FAILED_KNOWN`, or `UNKNOWN`.
4. Parse and validate from the persisted bytes. Record acceptance/rejection as another revision linked to the raw response digest; never let an in-memory parsed object be the only evidence.
5. On restart, reconcile every nonterminal attempt. Without provider-supported idempotency or a queryable remote operation ID, treat `ATTEMPT_PREPARED`/`DISPATCHED` as `UNKNOWN`, charge the conservative budget, and require policy approval before generating again.
6. Derive progress, usage, retry ordinals, and remaining budget from durable events/projections, not mutable in-memory counters. Make every transition idempotent by stable keys and reject payload drift on replay.

This is an architectural inference from SQLite’s transaction boundary and HTTP’s non-idempotent retry rule. It deliberately chooses auditable at-most-once budget accounting over pretending that the database and provider share an exactly-once transaction.

Today, provider transcripts accumulate in an in-memory list in the adapter ([`provider.py`, lines 526–555 and 626–667](../src/kuairand_agent/research/provider.py)) and are incorporated into a durable artifact later by the campaign runtime ([`full_campaign_runtime.py`, lines 2448–2477](../src/kuairand_agent/campaign/full_campaign_runtime.py)). A process crash between dispatch and that later artifact write can therefore erase attempt-level evidence. Moving attempt intent/outcome persistence into the dispatch boundary is the main durability improvement.

### Fault-injection acceptance gates

The implementation is not reliability-complete until tests kill or interrupt the controller at each boundary below and resume from disk:

- immediately before and after attempt reservation;
- during DNS/connect, request upload, response headers, and response body read;
- after a complete provider response but before parse;
- after raw-response persistence but before schema validation;
- after accepted package persistence but before source materialization;
- between project-ledger and campaign-database reservations;
- after artifact temp-file write, after content digest verification, and before/after atomic publication;
- during backoff and at the campaign deadline.

For every case, assert: no silent duplicate budget spend, no acceptance of partial/unvalidated source, durable `UNKNOWN` state where the remote outcome cannot be proven, preservation of the last eligible incumbent/fallback, monotonic revisions, and deterministic resumption from persisted identities.
