# Campaign attempts 1-10: reliability postmortem

_Evidence reviewed 2026-08-29. Credential/key failures are intentionally excluded. “Completed”
below means the controller durably finalized; it does not imply meaningful autonomous science._

## Outcome summary

No attempt produced a successful autonomous candidate evaluation. Attempts 1, 3, 6, and 8
durably finalized the official-FM fallback after research failed. Attempts 2, 4, 5, 7, 9, and 10
were stopped while their databases still said `RUNNING`; no campaign process remains active.
Attempt 7 was the only attempt to admit a generated candidate and start scientific training.

| Attempt | Durable result | Autonomous progress | Principal non-credential issue |
| --- | --- | --- | --- |
| 1 | `COMPLETED`, revision 18 | 0 proposals, 0 iterations | Provider transport exhausted; fallback-only finalization |
| 2 | stale `RUNNING`, revision 15 | 1 rejected pre-execution branch | The response over-declared unchanged material symbols |
| 3 | `COMPLETED`, revision 18 | 0 proposals, 0 iterations | Provider transport exhausted; fallback-only finalization |
| 4 | stale `RUNNING`, revision 15 | Research call in flight | Long provider call, no durable lineage checkpoint before stop |
| 5 | stale `RUNNING`, revision 15 | Research call in flight | Long provider call, no durable lineage checkpoint before stop |
| 6 | `COMPLETED`, revision 18 | 1 proposal, 0 admitted candidates | Provider HTTP failure after proposal; fallback-only finalization |
| 7 | stale `RUNNING`, revision 31 | 1 proposal, 1 admitted candidate, 1 scientific iteration | Generated loader treated a `.npy` ndarray as a context manager; training failed in 0.11 seconds |
| 8 | `COMPLETED`, revision 18 | 3 proposals, 2 implementations, 2 repairs, 0 admissions | Embedded `config.json` syntax failures, unchanged repairs, then a length-truncated implementation |
| 9 | stale `RUNNING`, revision 15 | 1 rejected pre-execution branch | Embedded `config.json` stayed invalid; repairs over-declared a non-code symbol |
| 10 | stale `RUNNING`, revision 15 | 1 rejected pre-execution branch | Embedded `config.json` stayed a Python literal after two repairs |

The repeated `RUNNING` rows are stale durable state from external stops, not evidence that the old
processes are live. Each of these attempts had completed data preparation, two fold-control
executions, and feature construction before becoming stuck in research.

## Root causes

### 1. The wrong reasoning parameter made generation extremely slow and incomplete

For OpenRouter-compatible gateway calls the adapter sent the OpenAI-style top-level
`reasoning_effort` field. OpenRouter documents a nested `reasoning` object. Attempt 8 shows the
consequence: 102,727 of 120,069 completion tokens (85.6%) were reasoning tokens. Nine provider
attempts consumed 1,543.21 seconds of a 1,675.64-second run. The last implementation ended with
`finish_reason=length` after 32,768 completion tokens.

### 2. An outer structured-output schema did not validate documents inside strings

`GeneratedPackage.files[*].content` is a JSON string. A strict outer response can prove that this
field is a string, but cannot prove that the string contains valid `config.json` or valid Python.
Attempts 8, 9, and 10 repeatedly returned single-quoted Python dictionaries or otherwise invalid
JSON inside that string. The local static gate correctly rejected them, but only after expensive
implementation/repair generations.

### 3. The model regenerated too much stable infrastructure

The original seed put protocol parsing, capability loading, NumPy I/O, CLI handling, checkpoint
I/O, model training, and prediction in one 16,129-byte `candidate.py`. Implementations therefore
tended to regenerate the entire file. Attempt 7 replaced the known-safe array loader and introduced
`with np.load(...)` for both `.npy` and `.npz`; `.npy` returns an ndarray, which is not a context
manager. Other responses exhausted their output allowance before producing a complete file.

### 4. Material-symbol declarations were too brittle

The admission gate rejected an entire real change if the model listed one extra unchanged or
missing symbol. Attempt 2 hit this directly. Several repairs in Attempts 8 and 9 restored unchanged
parent code and declared metadata or non-code names. The gate must still reject a package with no
real executable change, but an extra declaration should not erase a genuine declared change.

### 5. Provider evidence was durable too late

Provider transcripts accumulated in memory and were copied into an artifact only when the whole
research stage closed. Stops during long calls left generated directories but no lineage checkpoint
or durable provider-attempt journal, which made Attempts 2, 4, 5, 9, and 10 look idle even when a
provider call or repair had occurred.

### 6. Cooperative cancellation could wait for a full socket timeout

The signal handler only set an event. A blocking `urllib` call cannot observe that event, so a stop
could wait as long as the provider timeout. This explains why operators sometimes needed a hard
kill and why the database retained `RUNNING`.

### 7. Restart setup is material but not the dominant cost

Cold feature preparation took about 72.65 seconds in observed runs, while a verified warm replay
inside the same run took about 19.45 seconds. Cross-attempt sharing could save roughly 53 seconds,
but the current cache commit can be left incomplete by a crash and has no cross-process lock.
Sharing it globally before adding repair/locking would turn one interrupted write into a repeated
failure across every later attempt. Provider generation—not feature preparation—was the dominant
Attempt 8 cost.

## Implemented repairs

- Gateway reasoning now uses `reasoning: {effort, exclude}` for OpenRouter and TokenRouter hostnames;
  direct OpenAI endpoints retain `reasoning_effort`.
- The full autonomous config defaults to low reasoning, a 420-second per-attempt timeout, one
  transport retry, and a 32,768-token local output ceiling.
- OpenRouter model limits are discovered from the current model metadata endpoint. Each request is
  clamped to the advertised context and completion limits after conservative prompt headroom. An
  endpoint without compatible metadata retains the configured ceiling instead of using a guessed
  model-name table. Compatible `/models` responses are also used when they publish the same exact
  fields. Discovered limits are included in provider-usage evidence. OpenRouter routing additionally
  requires support for every supplied parameter before an endpoint is selected.
- The seed now separates a stable 11,228-byte `candidate.py` protocol entrypoint from a 5,953-byte
  mutable `model_impl.py`. Prompts tell the model to return only changed files and normally modify
  `model_impl.py` plus `config.json`, avoiding regeneration of capability and NumPy I/O plumbing.
- A bounded AST decoder converts only JSON-equivalent Python literals into canonical strict JSON.
  It never evaluates calls or accepts tuples, sets, bytes, duplicate keys, non-finite numbers, or
  non-object roots. This directly repairs the captured Attempt 10 response shape.
- Static validation now rejects the captured unsafe `np.load` context-manager pattern before a
  launch is reserved or training begins.
- Materiality admission accepts the real intersection of declared and changed reachable symbols,
  while still rejecting a package when none of its declarations identify a real code change.
- Every completed provider attempt is immediately written as an immutable, fsynced, digest-named
  journal entry before the call returns or a retry begins. This substantially narrows the former
  in-memory-only evidence window.
- The first SIGINT/SIGTERM remains cooperative; a second signal raises `KeyboardInterrupt` to
  interrupt a blocked provider call. Campaign remaining-time checks now also observe cancellation.
- Parent snapshot loading ignores Python bytecode cache directories, matching source provenance and
  preventing a harmless prior import from poisoning a later run.
- Prompt contract version 5 explicitly distinguishes the final manifest from the changed-file
  overlay and directs repairs toward the small mutable model surface.

## Remaining work before claiming end-to-end success

These changes remove the observed deterministic failures, but they do not prove a completed live
campaign. The next live attempt should begin only after the full static/unit/integration suite is
green. It should then be monitored for: discovered context-limit evidence, low reasoning-token
share, immutable provider-attempt journal growth, at least one admitted candidate, a real training
start, metrics, and durable finalization. A baseline-only fallback finalization should be reported
as safe degradation, not meaningful autonomous success.

Cross-attempt feature caching remains deliberately unimplemented until it has a lock, incomplete
entry repair, corruption quarantine, and crash/fault-injection tests. Exact prompt token counting
also remains provider/model-specific; the adaptive limiter uses conservative byte-derived headroom
rather than pretending one tokenizer fits every future model.
