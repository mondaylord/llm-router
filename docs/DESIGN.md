# Design

Architecture, design decisions, and tradeoffs for `llm-router`.

This document is the source of truth for *why* things are built this way.
For *what's done* see [PROGRESS.md](PROGRESS.md). For *what's next* see
[PLAN.md](PLAN.md).

---

## 1. Goals & non-goals

**Goals**

- Route each request to the cheapest model tier that meets quality, with
  imperceptible UX impact and bounded worst-case quality loss.
- Production-grade from day one: low latency overhead (<50ms p99 added),
  high concurrency, observable, safe defaults.
- A clean policy layer that can evolve (rules → classifier → cascade →
  bandit) without rewriting downstream integrations.
- Strong evaluation infrastructure — offline eval set + shadow traffic +
  decision logging — because every threshold needs evidence.

**Non-goals**

- Replacing a model gateway. We delegate provider/retry/fallback to LiteLLM
  (or any OpenAI-compatible endpoint).
- Replacing serving infrastructure (vLLM, SGLang, etc.).
- Becoming a general-purpose RAG / agent framework. This is a router only.
- Training new foundation models. We train small classifiers on top of
  off-the-shelf embeddings.

---

## 2. High-level architecture

```
                      Request: prompt, session_id, tenant_id, metadata
                                       │
                                       ▼
                      ┌────────────────────────────────────┐
                      │  Pre-Router                        │
                      │  - tenant policy lookup            │
                      │  - session state lookup            │
                      └─────────────────┬──────────────────┘
                                        │
                                        ▼
       ┌────────────────────────────────────────────────────────┐
       │  Layer 1: Rule Engine (zero-latency)                   │
       │  short_query / structured_request / code_block / ...   │
       └─────────────────┬──────────────────────────────────────┘
                         │ (no rule fired → continue)
                         ▼
       ┌────────────────────────────────────────────────────────┐
       │  Layer 2: Classifier (embedding + calibrated LR)       │
       │  output: p(needs_strong_model), latency ~30-80ms       │
       └─────────────────┬──────────────────────────────────────┘
                         │
                         ▼
       ┌────────────────────────────────────────────────────────┐
       │  Layer 3: Decision Synthesizer                         │
       │  - confidence threshold check                          │
       │  - session stickiness (NEVER downgrade mid-session)    │
       │  - tenant override                                     │
       │  - safe default → strong tier on uncertainty           │
       └─────────────────┬──────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │ RoutingDecision  │ ────▶│  Gateway         │ ──▶ Provider API
   │ (tier, reason,   │      │  (LiteLLM /      │
   │  confidence,     │      │   OpenAI-compat) │
   │  layer, meta)    │      └──────────────────┘
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────────────────┐
   │  Decision Logger                     │
   │  → eval pipeline / observability     │
   └──────────────────────────────────────┘
```

A request flows through the layers in order. Each layer can either
**emit a decision** (early-exit) or **pass through** with annotations.

---

## 3. Open-source landscape

### 3.1 RouteLLM (lm-sys)

The closest existing project. Ships four routers:

- Matrix Factorization (MF)
- BERT classifier
- Causal-LLM classifier
- Similarity-Weighted Ranking (SW-Ranking)

Trained on Chatbot Arena preferences, primarily GPT-4 vs Mixtral-8x7B.

**Useful for us as**: classifier inspiration; their BERT/causal-LLM heads
could even be used as starting weights if we wanted to skip cold-start.

**Not usable as a base because**:

- Binary strong/weak only. Production needs ≥3 tiers (cheap / mid / strong).
- Single-turn focused. No multi-turn session stickiness.
- English-centric Arena training data. Poor on Chinese / multilingual.
- No tenant overrides, no decision logging, no shadow traffic, no eval
  harness with the kinds of quality scores production teams need.
- Tightly couples the classifier with the OpenAI-proxy server — hard to
  swap a layer without rewriting plumbing.

### 3.2 LiteLLM (BerriAI)

Mature **gateway** library. Their concept of "router" is a load-balancer +
fallback policy, NOT quality-aware routing. We use LiteLLM as a downstream
*from* our quality router. Optional dep (`pip install ".[gateway]"`).

### 3.3 FrugalGPT (Stanford)

Cascade approach: small model → scorer → escalate. Mostly research code.
We borrow the technique idea (Stage 3 in PLAN.md), not the code.

### 3.4 Hybrid LLM (Microsoft)

Cascade with deferral, paper-grade. Same: technique reference.

### 3.5 Not Diamond / Martian / OpenRouter

Closed-source routing models, or aggregators that aren't quality routers.

### 3.6 Conclusion

No OSS project ships a production base. The work that is *not* in any of
them — sticky sessions, tenant policy, decision logging, eval harness — is
the work that determines whether the router is safe to roll out to large
customers. So we build a thin layer ourselves and integrate the pieces we
can borrow (LiteLLM downstream, possibly RouteLLM weights).

---

## 4. Key design decisions

### 4.1 Tiers, not concrete models

The router outputs `tier ∈ {weak, mid, strong}`, not a concrete model id.
A separate `ModelResolver` maps tier → provider/model. This decouples
policy from provider config: swapping `claude-haiku-4-5` for a cheaper
provider doesn't touch the router.

### 4.2 Hybrid pipeline (rules + classifier), not single model

Pure rules are brittle on long-tail queries; pure classifiers are wasteful
on obvious traffic and unaccountable on edge cases. Hybrid lets us handle
30-50% of traffic at zero latency with full traceability, and use the
classifier only where it adds value.

### 4.3 Calibrated probabilities, not raw scores

Calibrated classifier (sklearn `CalibratedClassifierCV` over LR) gives
probabilities that mean what they say. Production thresholds depend on
calibrated `p(strong)` so cost/quality tradeoffs are tunable.

### 4.4 Session stickiness: upgrade allowed, downgrade forbidden

Once a session has been served by `strong`, subsequent turns stay on
`strong` (or upgrade) unless the tenant explicitly opts out. Downgrading
mid-conversation produces visible capability drop and is never worth the
savings on a small tail of turns.

Stickiness state lives in a `SessionStore` interface — in-memory by
default, swap to Redis in production with no code change.

### 4.5 Safe defaults: uncertainty → strong

When the classifier is uncertain (probability inside `(low_thresh, high_thresh)`)
or the router service errors, we default to `strong`. Costing a bit more
on ambiguous traffic is acceptable; degrading quality silently is not.

### 4.6 Tenant policy as a first-class concept

Enterprise customers routinely demand "all my traffic on strong" or
"never use vendor X". `TenantPolicy` is checked before any routing
logic and can short-circuit the entire pipeline.

### 4.7 Decision logging is not optional

Every decision is logged with: input fingerprint, layer, reason, scores,
final tier. Without this, we cannot do shadow evaluation or root-cause a
quality regression. We use structlog with JSON output, which connects to
ELK/Loki/Datadog without adapter code.

### 4.8 Library + optional service, not service-only

The core `Router` is a Python library. The FastAPI server is a thin
wrapper. Library form supports embedded use (in-process for ultra-low
latency); service form supports polyglot clients and independent scaling.

### 4.9 Multilingual embedding default

`intfloat/multilingual-e5-small` (118M params, 384-dim). Trade ~10% English
quality vs `all-MiniLM-L6-v2` for usable Chinese/multilingual support out
of the box. Configurable.

---

## 5. Latency budget

| Stage                   | Target (p99) | Mechanism                          |
| ----------------------- | ------------ | ---------------------------------- |
| Pre-router (tenant/sess)| < 2ms        | in-memory cache; Redis MGET batch  |
| Rule engine             | < 1ms        | regex/heuristic over prompt        |
| Embedding (CPU)         | ~30-60ms     | sentence-transformers MiniLM/E5    |
| Embedding (GPU)         | ~5-15ms      | batched if QPS ≥ 200               |
| Classifier              | < 1ms        | sklearn LR                         |
| Decision synth + log    | < 2ms        |                                    |
| **Total p99 added**     | **< 80ms**   | (well below typical TTFB of 500ms+)|

If 80ms is too much for a low-cost-per-query app, run the embedding model
co-located on GPU with batching, or skip Layer 2 for tenants with a strict
latency SLA (per-tenant config).

---

## 6. Failure modes & mitigations

| Failure                              | Mitigation                                    |
| ------------------------------------ | --------------------------------------------- |
| Embedding service slow / down        | Timeout → skip Layer 2 → default to strong    |
| Classifier model file corrupt        | Health check at startup; fail closed to strong|
| Session store partition              | Lookup timeout → treat as new session         |
| Tenant config change race            | Versioned config; logged at decision time     |
| Distribution drift (new query types) | Shadow eval on rolling 7d; alert on Δ>X%      |
| Threshold misconfigured              | Per-decision score logged; replayable offline |

The recurring theme: every layer is fail-open *toward stronger models*,
never silently downgrades.

---

## 7. Evaluation

The router's quality is fundamentally an evaluation problem. We require:

1. **Offline eval set** — labeled prompts × per-model outputs × quality scores.
2. **Pareto curve** — cost vs quality at different thresholds, vs the
   "always strong" baseline.
3. **Shadow traffic** — run a candidate policy in parallel on N% of live
   traffic without acting on it; diff decisions.
4. **Decision-log replay** — every logged decision can be re-evaluated
   against a new policy without rerunning models.

See [EVAL.md](EVAL.md) for the full methodology.

---

## 8. What is intentionally NOT in v0

- **Cascade** (Stage 3): try-cheap-then-escalate. Implemented later because
  reliable quality detection on partial outputs is its own research problem.
- **KNN long-tail router** (Stage 3): collect labeled bad-case samples,
  route by similarity. Needs a feedback pipeline first.
- **Bandit threshold tuning** (Stage 4): online learning over thresholds.
  Needs reward signal infra before it makes sense.
- **GPU embedding service**: planned but not bundled. v0 runs CPU embedding
  in-process, which is fine up to a few hundred QPS per box.
- **Multi-region / failover topology**: deployment concern, not router
  code concern.
