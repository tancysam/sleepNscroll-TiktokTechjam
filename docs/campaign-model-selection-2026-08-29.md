# Model selection for the KuaiRand autonomous campaign

_Researched 2026-08-29. Prices, availability, and OpenRouter performance figures are
time-sensitive snapshots. This note uses first-party vendor documentation and the providers'
own catalog/telemetry pages. Vendor descriptions of model quality are treated as positioning, not
as independent proof that a model will succeed on this repository's workload._

## Decision

For the current implementation, which deliberately stays on `POST /v1/chat/completions`, use:

| Slot | Gateway | Exact model ID | Why |
| --- | --- | --- | --- |
| Main | OpenRouter | `openai/gpt-5.6-sol` | Best overall fit: flagship coding/agentic positioning, strict structured-output support, 128K maximum completion, multiple upstream providers, and direct Chat Completions support. |
| Fallback | OpenRouter | `openai/gpt-5.6-terra` | Live-qualified backup: strict proposal and implementation responses passed through Chat Completions, and the generated implementation cleared the repository's parser, manifest, syntax, import, JSON, and material-change gates. |

Recommended request settings remain **low reasoning**, strict JSON Schema, and the current local
32,768-token ceiling. The larger advertised model limits are safety headroom, not a reason to
regenerate the stable campaign infrastructure or to increase the local ceiling immediately.

### Live qualification result (same day)

The live campaign-sized qualification supersedes the catalog-only TokenRouter recommendation
below:

- OpenRouter `openai/gpt-5.6-sol` completed the captured proposal in 28.18 seconds and the captured
  implementation in 36.20 seconds. Both strict schemas passed; the implementation package passed
  every local static and material-change gate.
- OpenRouter `openai/gpt-5.6-terra` completed the same proposal in 21.13 seconds and implementation
  in 31.22 seconds. Both strict schemas passed; its implementation package also passed every local
  static and material-change gate.
- TokenRouter `qwen/qwen3.8-max` passed a tiny strict-schema probe but timed out after 180 seconds on
  the captured proposal.
- TokenRouter `z-ai/glm-5.3-flash` exhausted all 8,192 output tokens on hidden reasoning and returned
  empty content at low effort; a second request with reasoning set to `none` timed out after 180
  seconds.
- TokenRouter `MiniMax-M3` returned quickly but ignored strict structured output. The documented
  `reasoning_split` option removed the visible think block but still returned Markdown-fenced JSON,
  so the exact production parser correctly rejected it.

These are one-request functional qualifications, not ten-request latency or uptime samples. They
are sufficient to reject the three failed TokenRouter routes and to admit Sol and Terra for a
controlled campaign attempt; they do not establish p95 reliability.

Do **not** configure TokenRouter's `openai/gpt-5.6-sol`, `openai/gpt-5.6-terra`,
`openai/gpt-5.6-luna`, `anthropic/claude-sonnet-5`, `anthropic/claude-opus-5`, or
`google/gemini-3.7-flash` in the current fallback slot. TokenRouter currently advertises those via
Responses, Anthropic Messages, or Gemini-native APIs rather than Chat Completions. The present
adapter always dispatches to `/chat/completions`, so matching model names do not make those endpoint
contracts compatible. TokenRouter does explicitly advertise `qwen/qwen3.8-max` through both
`POST /v1/chat/completions` and `POST /v1/responses`.
[TokenRouter model catalog](https://www.tokenrouter.com/models/),
[TokenRouter Qwen3.8 Max page](https://www.tokenrouter.com/models/qwen/qwen3.8-max/),
[TokenRouter GPT-5.6 Sol page](https://www.tokenrouter.com/models/openai/gpt-5.6-sol/),
[TokenRouter Claude Sonnet 5 page](https://www.tokenrouter.com/models/anthropic/claude-sonnet-5/),
[TokenRouter Gemini 3.7 Flash page](https://www.tokenrouter.com/models/google/gemini-3.7-flash/)

## Why this workload needs a different selection standard

This campaign does not merely need a model that writes plausible prose. A useful response must:

1. follow a strict outer JSON Schema;
2. place complete Python and JSON artifacts inside that response;
3. avoid spending most of its completion allowance on hidden reasoning;
4. finish quickly enough to leave time for training, evaluation, reflection, and repair; and
5. remain available for a one-hour unattended sprint.

The previous DeepSeek configuration failed precisely on those boundaries: the local
[attempt postmortem](campaign-attempt-postmortem.md) records extremely high reasoning-token use,
length truncation, malformed embedded `config.json`, and no schema-enforced recovery. Therefore,
general chat quality or a large context window is not enough.

OpenRouter states that structured outputs are an **endpoint-level**, not merely model-level,
capability. It recommends `response_format.type = "json_schema"`, `strict: true`, and
`provider.require_parameters: true` so that routing excludes upstream endpoints that cannot honor
the request. Those requirements match the repaired adapter.
[OpenRouter structured outputs documentation](https://openrouter.ai/docs/guides/features/structured-outputs)

OpenRouter also documents one unified `POST /api/v1/chat/completions` interface for its catalog,
including automatic routing and fallbacks. Thus an OpenAI, Anthropic, Google, or Qwen model can be
used through the current Chat Completions adapter on **OpenRouter**, even when that model's native
vendor API uses a different shape. That statement must not be carried over to TokenRouter; its own
catalog lists the API formats per model.
[OpenRouter quickstart](https://openrouter.ai/docs/quickstart),
[OpenRouter Chat Completions reference](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)

## Ranked shortlist

### 1. Best balance and recommended main: `openai/gpt-5.6-sol` on OpenRouter

This is the most defensible default for this campaign.

- OpenAI identifies GPT-5.6 Sol as its flagship model for complex reasoning and coding. OpenAI
  documents a 1,050,000-token context window, 128,000 maximum output tokens, support for
  `none` through `max` reasoning effort, Chat Completions, and structured outputs.
  [OpenAI GPT-5.6 Sol model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
  [OpenAI model-selection guidance](https://developers.openai.com/api/docs/models)
- OpenRouter's exact ID is `openai/gpt-5.6-sol`. Its page also reports JSON-Schema structured
  outputs, three upstream providers with automatic failover, and—at the time of research—roughly
  55 output tokens/second at the listed OpenAI endpoint. The displayed promotional OpenRouter price
  was $2/M input and $10/M output; the normal direct OpenAI price was $4/M and $20/M, so the
  promotional figure should not be treated as permanent.
  [OpenRouter GPT-5.6 Sol page](https://openrouter.ai/openai/gpt-5.6-sol/)
- OpenAI explicitly recommends testing the same reasoning level and one level lower when migrating
  to GPT-5.6, because the family can often maintain quality with fewer tokens. That supports `low`
  as the campaign's latency-conscious starting point, but the repository qualification harness
  should compare `none` and `low` rather than assume either is universally best.
  [OpenAI GPT-5.6 guidance](https://developers.openai.com/api/docs/guides/latest-model)

**Inference, not a vendor guarantee:** the combination of direct Chat Completions support, strict
schema support, large completion headroom, multi-provider OpenRouter routing, and flagship coding
positioning makes Sol the lowest-risk first model to qualify on this exact task.

### 2. Quality challenger: `anthropic/claude-opus-5` on OpenRouter

Use this as the high-quality comparison candidate, not as the first production default.

- Anthropic positions Opus 5 for complex agentic coding and enterprise work; OpenRouter describes
  it as a flagship for end-to-end software tasks, code review, bug finding, and long-horizon agents.
  [Anthropic model overview](https://platform.claude.com/docs/en/models/overview),
  [OpenRouter Claude Opus 5 page](https://openrouter.ai/anthropic/claude-opus-5)
- OpenRouter documents the exact ID `anthropic/claude-opus-5`, a 1M context window, 128K maximum
  completion, JSON-Schema structured outputs, five upstream providers, and $5/M input plus $25/M
  output.
  [OpenRouter Claude Opus 5 page](https://openrouter.ai/anthropic/claude-opus-5)

**Trade-off:** it costs 2.5 times as much per output token as the current promotional Sol price and
does not have repository-specific evidence of a higher admitted-candidate rate. Test it only if Sol
still produces semantically weak or repeatedly non-runnable `model_impl.py` files after strict
schema conformance is already working.

### 3. Balanced cross-vendor challenger: `anthropic/claude-sonnet-5` on OpenRouter

Sonnet 5 is a credible alternative to Sol when code-generation quality and instruction following
need a second family at a similar price.

- Anthropic calls `claude-sonnet-5` its best combination of speed and intelligence and says its
  largest gains over Sonnet 4.6 are in coding and agentic tasks. It supports 1M context and 128K
  maximum output.
  [Anthropic Sonnet 5 changes](https://platform.claude.com/docs/en/docs/about-claude/models/whats-new-sonnet-5)
- OpenRouter's exact ID is `anthropic/claude-sonnet-5`, and its catalog currently lists $2/M input
  and $10/M output.
  [OpenRouter Claude Sonnet 5 page](https://openrouter.ai/anthropic/claude-sonnet-5/providers)

**Reliability caveat:** adaptive thinking is on by default in Sonnet 5; `max_tokens` covers both
thinking and visible output. Anthropic also reports that its new tokenizer can produce about 30%
more tokens for the same input text than Sonnet 4.6. The OpenRouter `reasoning` control therefore
must be verified to keep the reasoning share bounded on a captured campaign request.
[Anthropic Sonnet 5 changes](https://platform.claude.com/docs/en/docs/about-claude/models/whats-new-sonnet-5)

### 4. Fast economical production candidate: `google/gemini-3.7-flash` on OpenRouter

This is the strongest speed-first candidate in the researched set.

- Google says Gemini 3.7 Flash is generally available and built for complex coding, agentic
  workflows, and reliable multi-step execution. Google documents 1,048,576 input tokens, 65,536
  output tokens, structured outputs, and low/medium/high thinking.
  [Google Gemini 3.7 Flash overview](https://ai.google.dev/gemini-api/docs/latest-model),
  [Google Gemini 3.7 Flash specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)
- OpenRouter's exact ID is `google/gemini-3.7-flash`. At the research snapshot, its page listed
  about 110 output tokens/second for the fastest upstream, two routed upstreams, 99.88% three-day
  availability, a 65,536-token maximum completion, and $0.75/M input plus $3.75/M output.
  [OpenRouter Gemini 3.7 Flash page](https://openrouter.ai/google/gemini-3.7-flash)

**Trade-off:** the output ceiling is half the 128K-class models' ceiling, although still twice the
campaign's current local 32,768-token cap. More importantly, speed does not prove that complete
Python packages are semantically correct. Qualify it on implementation and repair operations, not
only on the smaller proposal response.

### 5. Cheapest OpenRouter candidates: `openai/gpt-5.6-luna` and `qwen/qwen3.8-flash`

These are appropriate for inexpensive qualification or eventually for small proposal/reflection
operations. They are not yet the recommended single model for all campaign operations.

- `openai/gpt-5.6-luna` costs $0.20/M input and $1.20/M output on OpenRouter, has 1.05M context,
  128K maximum output, structured outputs, and three routed providers. OpenAI positions Luna for
  cost-sensitive, high-volume workloads rather than the hardest coding work.
  [OpenRouter GPT-5.6 Luna page](https://openrouter.ai/openai/gpt-5.6-luna-20260709),
  [OpenAI model-selection guidance](https://developers.openai.com/api/docs/models)
- `qwen/qwen3.8-flash` is even cheaper at the current OpenRouter listing of $0.15/M input and
  $0.47/M output, with 1M context, 131,072 maximum completion, and JSON-Schema structured outputs.
  However, it was released only on 2026-08-26, three days before this research, so there is not yet
  enough operating history to call it a reliable unattended-run default.
  [OpenRouter Qwen3.8 Flash page](https://openrouter.ai/qwen/qwen3.8-flash)

**Inference:** if the controller later supports per-operation routing, a fast model can handle
`PROPOSE` and `REFLECT`, while Sol or Sonnet handles `IMPLEMENT` and `REPAIR`. With today's sticky
main/fallback architecture, choosing one weaker model for every operation creates more risk than
the price saving justifies.

## TokenRouter-compatible fallback ranking under the Chat Completions constraint

### 1. `qwen/qwen3.8-max` — recommended fallback

TokenRouter explicitly exposes this exact model through `POST /v1/chat/completions` and lists a
price of $1/M input and $3/M output. The same exact model on OpenRouter is documented with 1M
context, 131,072 maximum completion, and JSON-Schema structured outputs. The TokenRouter page,
however, does **not** itself publish context/output limits or assert strict JSON-Schema enforcement.
Do not assume that a model-level capability proves the TokenRouter endpoint's implementation.
[TokenRouter Qwen3.8 Max page](https://www.tokenrouter.com/models/qwen/qwen3.8-max/),
[OpenRouter Qwen3.8 Max page](https://openrouter.ai/qwen/qwen3.8-max)

At the research snapshot, OpenRouter reported 40 output tokens/second and 99.72% three-day
availability for its single Qwen3.8 Max upstream. Those figures describe OpenRouter's route, not
TokenRouter's service, and therefore cannot be used as TokenRouter reliability evidence.
[OpenRouter Qwen3.8 Max page](https://openrouter.ai/qwen/qwen3.8-max)

### 2. `openai/gpt-5` — conservative older alternative

TokenRouter advertises `openai/gpt-5` through `POST /v1/chat/completions` at $1.25/M input and
$10/M output. It is older and less attractive than the current Sol family, but it is a possible
fallback if Qwen3.8 Max fails the live strict-schema probe.
[TokenRouter GPT-5 page](https://www.tokenrouter.com/models/openai/gpt-5/)

This is not automatically qualified: TokenRouter's model page does not state its structured-output
contract, output ceiling, or observed uptime. It must pass the same live probe before use.

### Excluded from TokenRouter fallback today

| Model | TokenRouter-advertised API format | Why excluded from current adapter |
| --- | --- | --- |
| `openai/gpt-5.6-sol` | `POST /v1/responses` | No Chat Completions endpoint advertised. |
| `openai/gpt-5.6-terra` | `POST /v1/responses` | No Chat Completions endpoint advertised. |
| `openai/gpt-5.6-luna` | `POST /v1/responses` | No Chat Completions endpoint advertised. |
| `anthropic/claude-sonnet-5` | `POST /v1/messages` | Anthropic Messages shape, not Chat Completions. |
| `anthropic/claude-opus-5` | `POST /v1/messages` | Anthropic Messages shape, not Chat Completions. |
| `google/gemini-3.7-flash` | Gemini `generateContent` | Gemini-native shape, not Chat Completions. |

Sources:
[TokenRouter model catalog](https://www.tokenrouter.com/models/),
[GPT-5.6 Sol](https://www.tokenrouter.com/models/openai/gpt-5.6-sol/),
[GPT-5.6 Luna](https://www.tokenrouter.com/models/openai/gpt-5.6-luna/),
[Claude Sonnet 5](https://www.tokenrouter.com/models/anthropic/claude-sonnet-5/),
[Claude Opus 5](https://www.tokenrouter.com/models/anthropic/claude-opus-5/),
[Gemini 3.7 Flash](https://www.tokenrouter.com/models/google/gemini-3.7-flash/)

## Reliability qualification gate before a full sprint

Do not select from catalog claims alone. Run a small fixed qualification set through the exact
gateway, model ID, request body, and credentials that the campaign will use. A model/gateway pair
passes only if it satisfies all of the following:

1. Ten fixed requests complete without transport or provider errors.
2. Ten of ten outer responses pass the strict operation JSON Schema.
3. Ten of ten embedded `config.json` documents parse as strict JSON.
4. Ten of ten generated Python overlays compile and pass the existing static safety gates.
5. At least eight of ten implementation/repair responses make a real reachable code change instead
   of restating the parent.
6. No response ends because of the length limit.
7. Reasoning tokens remain below 25% of total completion tokens on each request, with low effort.
8. Record p50 and p95 latency, cost per valid response, and cost per admitted candidate. The last
   measure—not price per million tokens—is the useful economic comparison.

For the current architecture, the first test matrix should be:

| Priority | Gateway | Model | Purpose |
| --- | --- | --- | --- |
| 1 | OpenRouter | `openai/gpt-5.6-sol` | Main-candidate qualification. |
| 2 | TokenRouter | `qwen/qwen3.8-max` | Cross-gateway fallback qualification. |
| 3 | OpenRouter | `google/gemini-3.7-flash` | Speed/cost challenger. |
| 4 | OpenRouter | `anthropic/claude-sonnet-5` | Cross-family quality challenger at similar list price. |
| 5 | OpenRouter | `anthropic/claude-opus-5` | Expensive quality-ceiling check only if the earlier models fail semantically. |

## Recommended environment selection

The exact variable names already used by the campaign should resolve to:

```dotenv
INFERENCE_MAIN_BASE_URL=https://openrouter.ai/api/v1
INFERENCE_MAIN_MODEL=openai/gpt-5.6-sol

INFERENCE_FALLBACK_BASE_URL=https://openrouter.ai/api/v1
INFERENCE_FALLBACK_MODEL=openai/gpt-5.6-terra
```

Keep the credentials only in `.env.local`; do not add them to this document or any committed
configuration. The model IDs above should be pinned for a stability run. Avoid `latest` aliases in
the acceptance attempt because they can change behavior without a repository change.

## Bottom line

- **Recommended main:** OpenRouter `openai/gpt-5.6-sol`.
- **Qualified fallback:** OpenRouter `openai/gpt-5.6-terra`.
- **Independence caveat:** main and fallback now share the OpenRouter gateway. This protects against
  model-specific failures but not a gateway-wide OpenRouter outage. None of the tested TokenRouter
  Chat Completions models passed the production-sized qualification.
- **Fastest serious challenger:** OpenRouter `google/gemini-3.7-flash`.
- **Comparable quality challenger:** OpenRouter `anthropic/claude-sonnet-5`.
- **Highest-cost quality check:** OpenRouter `anthropic/claude-opus-5`.
- **Do not return to `deepseek/deepseek-v4-pro-0813` for this role** unless a future endpoint proves
  strict schema enforcement and bounded reasoning on the repository qualification suite.

The final production choice should be the model/gateway pair with the highest valid-artifact and
admitted-candidate rate under the one-hour wall clock, not the model with the largest context window
or the strongest marketing label.
