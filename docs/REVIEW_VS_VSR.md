# Review: llm-router vs vLLM Semantic Router

A structured comparison of this project against [vLLM Semantic Router][vsr]
(hereafter **vSR**), and a list of features worth adopting.

[vsr]: https://github.com/vllm-project/semantic-router

---

## 1. TL;DR

vSR and llm-router solve overlapping but different problems:

| Dimension                 | vSR                                              | llm-router                                         |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| **Primary purpose**       | Gateway in front of vLLM **pools**               | Routing layer for **mixed-provider** (chat + agent)|
| **Architecture**          | Polyglot sidecar (Go 45% / Python 18% / TS / Rust)| Python library + optional FastAPI service         |
| **Routing brain**         | BERT classifier + LoRA + multi-signal pipeline   | Rules + calibrated LR over embeddings              |
| **Data plane**            | Decision-only (delegates to vLLM)                | Decision-only (delegates to LiteLLM / direct SDK)  |
| **First-class features**  | PII, jailbreak, semantic cache, MCP, dashboard   | Agent step-types, tool whitelists, outcome cascade |
| **Caller integration**    | HTTP / OpenAI-compatible proxy                   | Python lib OR HTTP                                 |
| **Config**                | Declarative DSL                                  | YAML + pydantic                                    |
| **Target deployment**     | K8s gateway / sidecar                            | embed-in-app OR uvicorn service                    |

**Honest verdict**: vSR is a more mature *gateway*. llm-router is a more focused
*agent-aware policy layer*. They overlap on routing, but vSR carries weight we
don't need (multimodal, K8s artifacts, Go service) while we carry weight they
don't (per-step-type stickiness, outcome cascade, tenant policy as first-class).

For the user's stated goal — integrate into a **TS+Bun coding agent (reel)** —
llm-router is the better starting point because it is *agent-step-aware by
design* and trivially embeddable behind a single HTTP endpoint. We should
**adopt three things from vSR** (PII detection, jailbreak detection, semantic
cache) and ignore the rest.

---

## 2. Architecture: where they differ

### vSR

```
Client ──HTTP──▶ [vSR Go service]
                    │
                    ├─ classifier (BERT, LoRA-extensible)
                    ├─ PII tagger (token-level BIO)
                    ├─ jailbreak detector
                    ├─ semantic cache (bi-encoder)
                    ├─ cross-encoder reranker (optional)
                    │
                    └──▶ vLLM pool A / B / C
```

vSR is a *fat gateway*: signal extraction, policy decision, AND request
forwarding all in one binary, designed for inserting between a client and one
or more vLLM-served model pools. It standardizes on Hugging Face encoder models
behind a Rust/candle binding for speed.

### llm-router

```
Client (Python lib or HTTP) ──▶ Router
                                   │
                                   ├─ tenant policy
                                   ├─ agent rules  (failure cascade, tools, ...)
                                   ├─ chat rules
                                   ├─ classifier (sklearn LR over ST embedding)
                                   ├─ stickiness (per session × step type)
                                   │
                                   └──▶ Gateway adapter
                                         ├─ noop (decision-only)
                                         ├─ LiteLLM (multi-provider)
                                         └─ Anthropic / OpenAI / ...
```

llm-router is a *thin policy layer*: it decides which **tier** should serve
the request and (optionally) delegates the actual call to a gateway adapter.
The caller can also choose to bypass the gateway entirely and just consume
the decision.

### Implication

vSR's design assumes you OWN the inference stack (vLLM pools you operate).
llm-router's design assumes you CALL providers via SDKs and want one of them
selected per request. For an agent product that's calling Anthropic / OpenAI
through SDKs, vSR's gateway model adds an extra hop you don't need; for a
self-hosted multi-pool setup, vSR is the right shape.

---

## 3. Feature parity matrix

Status: ✅ shipped here · 🟡 partial / different impl · ❌ absent here

| Capability                                          | vSR | llm-router | Should we adopt? |
| --------------------------------------------------- | --- | ---------- | ---------------- |
| Rule-based routing                                  | ✅   | ✅          | —                |
| ML classifier routing                                | ✅ (BERT+LoRA) | ✅ (sklearn LR) | Defer encoder upgrade — calibrated LR is enough at our scale |
| Tier/model resolution                               | ✅   | ✅          | —                |
| Decision-only data plane                            | ✅   | ✅          | —                |
| OpenAI-compatible proxy mode                        | ✅   | ❌          | Yes — small wrapper around `/route` + LiteLLM forwarding |
| PII detection (regex + token-level NER)             | ✅   | ❌          | **Yes — table stakes** (`adopt #1`) |
| Jailbreak / prompt-injection detection              | ✅   | ❌          | **Yes — table stakes** (`adopt #2`) |
| Semantic cache                                      | ✅   | ❌          | **Yes — high ROI** (`adopt #3`) |
| Cross-encoder reranking                             | ✅   | ❌          | No — overkill for tier routing |
| Multimodal routing (image/audio)                    | ✅   | ❌          | No — not in scope for coding agent |
| Embedding dimension truncation (MRL / 2DMSE)        | ✅   | ❌          | Nice-to-have — small latency win |
| Agent step types (planning / tool / edit / ...)     | 🟡 (generic) | ✅      | We're ahead here |
| Tool whitelists (safe / strong)                     | ❌   | ✅          | We're ahead      |
| Outcome-driven cascade (`recent_outcomes`)          | 🟡 (verification) | ✅ | We're ahead      |
| Per-step-type session stickiness                    | ❌   | ✅          | We're ahead      |
| Per-tenant policy (forced / blocked tiers)          | 🟡 (basic) | ✅      | We're ahead      |
| Conversational memory / sticky upgrades             | ✅   | ✅          | Parity           |
| Latency-strict per-tenant mode                      | ❌   | ✅          | We're ahead      |
| Declarative DSL for policies                        | ✅   | 🟡 (YAML)   | No — YAML is enough at our scale |
| Prometheus / Grafana                                | ✅   | ❌ (backlog L5) | Yes — already on backlog |
| Dashboard / UI                                       | ✅   | ❌          | No — defer; logs + Prometheus cover it |
| K8s artifacts (CRDs / operators)                    | ✅   | ❌          | No — premature  |
| Eval harness (offline Pareto curve)                 | ❌   | ✅          | We're ahead      |
| Shadow traffic                                      | 🟡 (mentioned) | ❌ (backlog M1) | Yes — already on backlog |
| Decision-log replay                                  | ❌   | ❌ (backlog M2) | Yes — already on backlog |
| MCP server / MCP-tool detection                     | ✅   | ❌          | Yes — for reel-agent integration (`adopt #4`) |
| Fleet simulator for capacity planning               | ✅   | ❌          | No — we don't own the inference fleet |
| Polyglot Go binary                                  | ✅   | ❌          | No — premature  |

---

## 4. What vSR does better — and what we should adopt

These are the four features that, after honest assessment, move us forward
**for the coding-agent use case** without overreaching.

### Adopt #1 — PII detection

**Why**: agents in production handle user code, credentials, internal URLs,
PII in tool outputs. Routing a PII-laden prompt to the cheap model is fine for
quality but can create a data-handling compliance problem (different providers,
different DPAs). Detection at the router lets us:
- Force STRONG (which may be self-hosted) when PII is present
- Force a tenant-isolated pool
- Annotate the decision log for audit

**vSR approach**: token-level BIO tagger (small encoder), specialized
classifier for ~10 PII categories.

**Recommended for us**:
- v0: regex patterns for high-signal PII (email, IP, AWS keys, SSH keys,
  JWT, credit cards). 30-50 lines.
- v1: optional spaCy NER for names/orgs.
- v2: encoder model if we hit precision limits.

**Where to start**: new module `src/llm_router/safety/pii.py` with a
`PIIDetector` returning a `PIIReport` (categories + spans). New rule
`PiiPresentRule` that consults it. Configurable per tenant (some won't care).

**Effort**: S — half a day to v0, with a clean upgrade path.

### Adopt #2 — Prompt-injection / jailbreak detection

**Why**: agents are particularly vulnerable. Tool outputs are user-influenceable;
a malicious file content can carry "ignore previous instructions and ..." and
the agent will execute it on the next turn. Detection lets us:
- Force STRONG (better at resisting injection)
- Refuse / mask the input
- Annotate the decision log

**vSR approach**: prompt-guard classifier (small encoder).

**Recommended for us**:
- v0: rules over known injection patterns ("ignore previous", "you are now ...",
  base64-only payloads, suspicious system-prompt structure).
- v1: HuggingFace prompt-guard model behind the same interface.

**Where to start**: `src/llm_router/safety/injection.py` with `InjectionDetector`
returning a confidence score. New rule `InjectionDetectedRule` in agent rules.

**Effort**: S — half a day to v0.

### Adopt #3 — Semantic cache

**Why**: in an agent loop, the same `read_file` of the same path with the
same context is asked many times. Even outside loops, similar user prompts
hit the same models. Caching at the **prompt embedding** level (not exact-match)
recovers a chunk of cost AND latency that no amount of routing alone can.

**vSR approach**: bi-encoder embedding + similarity threshold + LRU. They
report 96% cost reduction in some benchmarks; that's marketing, but
**directionally the win is real** for high-repetition agent workloads.

**Recommended for us**:
- Cache key: `embedding(prompt)` rounded to a similarity-aware bucket, plus
  `(tenant_id, tool_name)` namespace so different tenants don't share answers.
- Storage: in-process LRU for v0; Redis for shared cache later.
- Read-side: BEFORE routing, lookup; if hit above threshold, return cached
  response with a `cache_hit` decision layer.
- Invalidation: TTL + per-tool override (read_file results invalidate when
  file mtime changes — caller-driven).
- **Skip for non-deterministic step types**: don't cache `planning` or `edit`
  steps where we want fresh thinking.

**Where to start**: `src/llm_router/cache/semantic_cache.py`. Optional layer
that wraps `Router.route()` and the gateway call. Default off; tenants opt in.

**Effort**: M — 1-2 days, mostly testing the right similarity threshold.

### Adopt #4 — MCP awareness (for agent tool routing)

**Why**: MCP (Model Context Protocol) is becoming the standard for tool servers.
If reel-agent uses MCP, the router should be able to:
- Recognize MCP tool names from the canonical registry
- Apply tool-class routing without manual `weak_safe_tools` config
- Receive structured tool failure outcomes from MCP error envelopes

**vSR approach**: built-in MCP integration (declared in repo topics).

**Recommended for us**:
- Add an optional MCP tool registry loader: read MCP servers' tool definitions
  at startup, auto-classify into safe / strong based on heuristics
  (read-only verbs → safe; write/exec verbs → strong) with config override.
- Wire MCP error envelope → `Outcome(kind=TOOL_EXECUTION_ERROR)`.

**Where to start**: `src/llm_router/integrations/mcp.py`.

**Effort**: M — needs concrete reel-agent MCP shape to design against.

---

## 5. What we do better for the coding-agent use case

These are differentiations to keep, not gaps to close.

### Agent step-type as a first-class routing axis

vSR routes per-request based on prompt content. We route per-request based on
prompt content **plus** the agent step type. For a coding agent:

- A `tool_call` for `grep` and a `tool_call` for `edit_file` should route
  differently regardless of prompt text. Tool-class signal beats prompt signal
  for cost/quality.
- A `planning` step that's *short in text* still needs `STRONG`. vSR's
  "short prompt → cheap" heuristics work against you here.
- A `tool_result` interpretation step is *atomic* — we mark it non-sticky so
  the next planning step isn't pinned by it.

### Outcome cascade

vSR has "real-time verification" but the failure signal flows back via their
own verifier. Ours uses the caller-curated `recent_outcomes` list, which:
- Keeps the router stateless across calls
- Lets the agent framework own the (often domain-specific) failure detection
- Works for both structured failures (schema, exec) and behavioral signals
  (retry, thumbs-down)

### Per-tenant policy depth

We support `forced_tier`, `blocked_tiers`, `latency_strict`, and per-tenant
classifier threshold override. This is essential for enterprise customers
(SaaS coding agents will get these requests on day one) and is heavier in
vSR's DSL but absent at the API surface here without DSL ceremony.

### Library form

vSR runs as a service. llm-router runs as either a library (in-process,
sub-ms overhead) or a service. For an agent framework that wants
ultra-tight integration on the same node, the library form is a real win.
For multi-language callers (TS+Bun reel-agent), HTTP works too.

---

## 6. Integration plan for reel-agent (TS+Bun)

The user is building [reel-agent](/opt/nvme/home/mondaylord/reel-agent) in
TypeScript on Bun. Concrete recommendation:

### Topology

```
┌─────────────────────────────────────────────────────┐
│         reel-agent (TS+Bun)                         │
│                                                     │
│   loop:                                             │
│     1. classify step (plan / tool_call / ...)       │
│     2. POST /route → llm-router (Python service)    │
│     3. resolve tier → model id locally              │
│     4. invoke provider SDK directly                 │
│     5. observe outcome → next /route carries it     │
└─────────────────────────────────────────────────────┘
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────┐
│         llm-router (FastAPI, Python)                │
│   /route → RoutingDecision                          │
│   /healthz                                          │
└─────────────────────────────────────────────────────┘
```

### Why HTTP, not embed

- reel-agent is TS+Bun; llm-router is Python. No Bun ↔ Python FFI worth
  shipping at this scale.
- Routing decision is ~1-100ms; that's well under the model call latency,
  so the HTTP overhead is in the noise.
- Service form lets multiple agent processes / workers share one classifier
  load (the embedding model is the biggest memory footprint).

### Minimum required schema from reel-agent

Every `/route` call from reel-agent should include:

- `prompt` or `messages` (one of)
- `session_id` (the conversation / tape id)
- `agent_step_type` (`planning` / `tool_call` / `tool_result` / `edit` / ...)
- `planned_tool` (when known at call time — most agents know one step ahead)
- `available_tools` (what could be called this turn)
- `recent_outcomes` (failures from prior turns — drives cascade)
- `total_context_tokens` (the assembled context size, for the long-context rule)

The router returns:
- `tier` (`weak` / `mid` / `strong`)
- `layer`, `reason`, `confidence`, `classifier_score`
- `inferred_step_type` (for debugging when caller didn't specify)
- `elapsed_ms`

### Tier → concrete model mapping in reel-agent

reel-agent owns the provider/model mapping. The router stays provider-agnostic:

```ts
// reel-agent side
const MODEL_FOR_TIER = {
  weak:   "claude-haiku-4-5",
  mid:    "claude-sonnet-4-6",
  strong: "claude-opus-4-7",
} as const;
```

This is also where API keys, retry, fallback, and budget caps live.

### Failure signal pipeline (essential for cascade)

reel-agent's verifier (the "in-loop verifier" from the project notes) is
exactly the right place to emit outcomes. Map verifier verdicts to
`OutcomeKind`:

| reel-agent verifier signal           | llm-router OutcomeKind          |
| ------------------------------------ | ------------------------------- |
| Tool args fail JSON schema validation| `TOOL_SCHEMA_ERROR`             |
| Tool returns error / non-zero exit   | `TOOL_EXECUTION_ERROR`          |
| Patch fails to apply                  | `PARSE_ERROR`                   |
| Tests fail after edit                | `VALIDATION_ERROR`              |
| User clicked retry                   | `RETRY_ATTEMPT`                 |
| User thumbs-down                     | `USER_NEGATIVE_FEEDBACK`        |
| Anything else verifier flagged       | `GENERIC_FAILURE`               |

reel-agent maintains a rolling window of the last 1-3 outcomes per session
and passes them into the next `/route` call. Older outcomes are stale and
should be dropped by the caller — the router doesn't dedupe.

### Recommended config for reel-agent's tenant

Start from [examples/agent_preset.example.yaml](../examples/agent_preset.example.yaml)
and:

1. Replace `weak_safe_tools` / `requires_strong_tools` with reel's actual
   tool names (the ones from your MCP / built-in registry).
2. Tune `classifier.p_high` from 0.55 toward 0.50 if you observe too many
   cheap-tier failures, or toward 0.70 if quality is fine and cost is the
   bigger problem.
3. Keep `non_sticky_step_types: [tool_call, tool_result, summarize]` — these
   are right for agent flows.
4. Enable `failure_escalation` — this is the cascade.

A TypeScript client stub is provided at [examples/clients/typescript/llm_router_client.ts](../examples/clients/typescript/llm_router_client.ts).

---

## 7. Recommended backlog updates

After this review, append the following to [PROGRESS.md#Backlog](PROGRESS.md#6-backlog):

- **[H5] PII detection module** (S) — regex v0, NER upgrade path
- **[H6] Prompt-injection detection module** (S) — pattern v0, encoder upgrade path
- **[H7] Semantic cache** (M) — bi-encoder + bucket-keyed LRU/Redis
- **[H8] reel-agent TS client + integration guide** (S) — already started in
  this PR; expand into a full guide
- **[M8] OpenAI-compatible proxy mode** (M) — for callers that want to swap
  llm-router in transparently
- **[M9] MCP registry awareness** (M) — auto-classify MCP tools by verb when
  `weak_safe_tools` is empty
- **[L8] MRL embedding dim truncation** (S) — small latency win on the
  classifier embedding step

Items deliberately rejected after review:
- Multimodal routing (out of scope for coding agent)
- Cross-encoder reranking (overkill for binary tier decision)
- DSL for policy (YAML + pydantic is sufficient at this complexity)
- K8s CRDs / operators (premature)
- Polyglot rewrite (Go service buys nothing for our pattern)

---

## 8. Honest things I'm uncertain about

- **vSR's "96% cost reduction with 8B vs 235B"** claim. The reproducible
  win is closer to 30-50% in published RouteLLM-style benchmarks. Don't
  size capacity expecting 96%.
- **vSR's encoder classifier vs our calibrated LR**. The encoder is more
  expressive but ops overhead (model serving, GPU, monitoring) is real.
  At our scale and given calibrated LR's interpretability, the LR is
  the right default. Re-evaluate if/when we have >100k labeled prompts.
- **Semantic cache hit rate in agent loops**. Highly workload-dependent.
  Same-file `read_file` repeats are likely hits; same-text planning calls
  are likely misses. Recommend a 2-week shadow measurement before relying
  on cached cost savings in projections.
- **Whether MCP awareness is worth investing in now**. Depends entirely on
  whether reel-agent commits to MCP. If yes, do it; if reel-agent uses a
  custom tool registry, build to that instead. Ask before implementing M9.
