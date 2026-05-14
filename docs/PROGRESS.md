# Progress & Handoff

Single-source-of-truth for picking this project up later. If you're
returning to this codebase after a break, read this top-to-bottom.

- For *why* things are built this way → [DESIGN.md](DESIGN.md)
- For the long-term phased plan → [PLAN.md](PLAN.md)
- For evaluation methodology → [EVAL.md](EVAL.md)

---

## 1. TL;DR

The router has two complete phases implemented end-to-end:

- **Stage 1 — chat rules** (8 builtin heuristics, EN+ZH)
- **Stage 2 — embedding+classifier** (calibrated LR; HashingEmbedder
  for dev, sentence-transformers wired in via `[ml]` extra)
- **Stage 2.5 — agent-mode** (8 agent rules, step-type auto-detection,
  outcome-driven cascade, per-step-type stickiness, tenant overrides)

It runs as a library (`Router`) or as a FastAPI service. 52/52 tests pass.

What it does **NOT** do yet:
- Actually invoke models (gateway is `noop` by default; LiteLLM
  adapter is a stub).
- Persist decision logs anywhere except stdout.
- Have a real evaluation dataset — only synthetic.
- Auto-tune thresholds.

Repo: <https://github.com/mondaylord/llm-router>

---

## 2. Verify it works (smoke test)

From a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/pytest tests/ -q                          # 52 tests pass
.venv/bin/python scripts/seed_data.py               # synthetic data
.venv/bin/python examples/train_classifier.py       # train classifier
.venv/bin/python examples/basic_usage.py            # chat demo
.venv/bin/python examples/agent_usage.py            # agent demo (8 steps)
.venv/bin/python examples/eval_run.py               # eval harness
.venv/bin/uvicorn llm_router.server.app:app         # HTTP service
```

For the real sentence-transformer (downloads ~120 MB on first use):

```bash
.venv/bin/pip install -e ".[ml]"
# Then edit train_classifier.py to use load_default_embedder() instead
# of HashingEmbedder(dim=256).
```

---

## 3. Implementation matrix

Status legend: ✅ done & tested · 🟡 scaffolded (interface exists, impl
is a stub) · ❌ not started.

| Area                              | Status | Notes                                                                    |
| --------------------------------- | ------ | ------------------------------------------------------------------------ |
| Core types (`core/decision.py`)   | ✅     | `RoutingRequest`/`Decision`, `Message`, `ToolCall`, `Outcome`, `Tier`    |
| YAML config (`core/config.py`)    | ✅     | All layers configurable; pydantic-validated                              |
| Router pipeline (`core/router.py`)| ✅     | Tenant → agent-rules → chat-rules → classifier → default                 |
| Step-type detection (`core/messages.py`) | ✅ | Heuristic; explicit caller value always wins                          |
| Chat rules (`rules/builtin.py`)   | ✅     | 8 rules covering common chat patterns                                    |
| Agent rules (`rules/agent.py`)    | ✅     | 8 rules; tool whitelists driven by config                                |
| Classifier trainer                | ✅     | k-fold + isotonic calibration; outputs `p(strong)` ∈ [0,1]               |
| Classifier predictor              | ✅     | Lazy-loads model + embedder; single-call hot path                        |
| Embedding (Hashing)               | ✅     | For tests/CI; no torch dep                                               |
| Embedding (Sentence-Transformer)  | ✅     | `intfloat/multilingual-e5-small` default; `[ml]` extra                   |
| Session store (in-memory)         | ✅     | TTL eviction; per-`(session, step_type)` tier                            |
| Session store (Redis)             | ❌     | Interface ready; impl not written                                        |
| Stickiness policy                 | ✅     | Upgrade-only by default; `non_sticky_step_types` for atomic agent steps  |
| Tenant policy                     | ✅     | Forced tier, blocked tiers, latency-strict, per-tenant p_high override   |
| Eval harness                      | ✅     | Pareto curve, baseline, threshold sweep                                  |
| Synthetic data generator          | ✅     | `scripts/seed_data.py` — labels + (model, quality, cost) triples         |
| FastAPI server                    | ✅     | `/route` + `/healthz`; lifespan-managed router                           |
| Gateway: noop                     | ✅     | Returns synthetic envelope; never calls a provider                       |
| Gateway: LiteLLM adapter          | 🟡     | Working stub; needs auth/credential wiring                               |
| Gateway: Anthropic SDK adapter    | ❌     | Not started; recommended for native Claude tool-calling                  |
| Decision logger (stdout JSON)     | ✅     | One event per decision via structlog                                     |
| Decision-log persistence          | ❌     | Sink to file/JSONL/Kafka — not implemented                               |
| Shadow traffic runner             | ❌     | Stub raises NotImplementedError                                          |
| Decision-log replay               | ❌     | Stub raises NotImplementedError                                          |
| Drift detector                    | ❌     | No metric collection yet                                                 |
| Output-driven cascade (Phase 3)   | ❌     | Pattern is "post-score weak output, escalate"; not built                 |
| KNN long-tail router (Phase 3)    | ❌     | Needs feedback signal pipeline first                                     |
| Auto threshold tuning (Phase 4)   | ❌     | Bandit / grid search over decision logs                                  |
| Tests (rules, router, classifier, agent, server) | ✅ | 52 tests; runs in ~1.3s; no GPU required                          |
| Examples (basic, train, eval, agent) | ✅ | All runnable                                                            |
| CI (GitHub Actions)               | ❌     | Not configured                                                           |
| Dockerfile                        | ❌     | Not written                                                              |
| `/metrics` Prometheus endpoint    | ❌     | Not written                                                              |

---

## 4. Where things live

```
llm-router/
├── docs/
│   ├── DESIGN.md           architecture, decisions, OSS comparison
│   ├── PLAN.md             phased roadmap (Phase 0 → 5)
│   ├── PROGRESS.md         this file
│   └── EVAL.md             evaluation methodology
├── src/llm_router/
│   ├── __init__.py         public API surface (Router, RoutingRequest, ...)
│   ├── core/
│   │   ├── decision.py     types: Tier, RoutingRequest, RoutingDecision,
│   │   │                          Message, ToolCall, Outcome, AgentStepType
│   │   ├── config.py       RouterConfig (+AgentConfig, StickinessConfig, ...)
│   │   ├── router.py       the pipeline — start here for any flow question
│   │   └── messages.py     agent step-type auto-detection heuristics
│   ├── rules/
│   │   ├── base.py         Rule ABC, FunctionRule
│   │   ├── builtin.py      8 chat rules
│   │   └── agent.py        8 agent rules + default_agent_ruleset()
│   ├── classifier/
│   │   ├── embedding.py    Embedder protocol + HashingEmbedder + ST wrapper
│   │   ├── trainer.py      ClassifierTrainer.fit_and_save()
│   │   └── predictor.py    ClassifierPredictor.load() / predict_proba_strong()
│   ├── session/
│   │   └── store.py        SessionStore ABC + InMemorySessionStore
│   ├── policy/
│   │   ├── stickiness.py   per-(session, step_type) upgrade-only stickiness
│   │   └── tenant.py       tenant policy resolver
│   ├── eval/
│   │   ├── dataset.py      JSONL I/O for labeled triples
│   │   └── harness.py      Pareto curve, threshold sweep, evaluate_router()
│   ├── gateway/
│   │   ├── base.py         Gateway ABC + NoopGateway
│   │   └── litellm_adapter.py  LiteLLM stub
│   ├── server/
│   │   ├── app.py          FastAPI app + /route + /healthz
│   │   └── schemas.py      HTTP request/response shapes
│   └── observability/
│       └── logger.py       structlog config
├── tests/
│   ├── test_rules.py       chat rules
│   ├── test_router.py      pipeline integration
│   ├── test_classifier.py  train/predict roundtrip
│   ├── test_agent.py       agent rules, step detection, per-step stickiness
│   └── test_server.py      FastAPI smoke tests
├── examples/
│   ├── basic_usage.py
│   ├── train_classifier.py
│   ├── eval_run.py
│   ├── agent_usage.py
│   ├── router_config.example.yaml
│   └── agent_preset.example.yaml
├── scripts/
│   └── seed_data.py        synthesizes data/train.jsonl + data/eval.jsonl
└── pyproject.toml
```

---

## 5. Common recipes

### 5.1 Add a new chat rule

1. Subclass `Rule` in [src/llm_router/rules/builtin.py](../src/llm_router/rules/builtin.py).
2. Implement `evaluate(request) -> RuleResult`. Return `RuleResult(tier=None)` to pass through.
3. Append to the list in `default_ruleset()`. Order matters: strong-tier rules first.
4. Add a unit test in [tests/test_rules.py](../tests/test_rules.py).

Rules of thumb: keep each rule < 0.5ms; err on the side of **not firing** (let the classifier handle ambiguity); add only when you can prove the precision is high.

### 5.2 Add a new agent rule

Same as above but in [src/llm_router/rules/agent.py](../src/llm_router/rules/agent.py) and append to `default_agent_ruleset()`. Agent rules read `request.metadata['detected_step']` (set by the router) for the step type.

### 5.3 Train classifier on real data

```python
from llm_router.classifier import ClassifierTrainer, TrainingConfig
from llm_router.classifier.embedding import load_default_embedder

trainer = ClassifierTrainer(
    config=TrainingConfig(embedding_model="intfloat/multilingual-e5-small"),
    embedder=load_default_embedder(),  # downloads ST model
)
prompts = [...]  # your real labeled prompts
labels  = [...]  # 1 if needs strong, 0 otherwise
meta = trainer.fit_and_save(prompts, labels, "artifacts/classifier.joblib")
```

Then point `RouterConfig.classifier.artifact_path` at the new file.

### 5.4 Swap the embedding model

Edit `RouterConfig.classifier.embedding_model` (YAML or in code).
**You must re-train the classifier head** after swapping — the dim and feature space change.

### 5.5 Wire a real gateway (LiteLLM)

1. `pip install -e ".[gateway]"`
2. Set provider credentials per LiteLLM docs.
3. Configure `RouterConfig.gateway.tier_to_model`:
   ```yaml
   gateway:
     backend: litellm
     tier_to_model:
       weak: claude-haiku-4-5
       strong: claude-opus-4-7
   ```
4. In your call site:
   ```python
   from llm_router.gateway.litellm_adapter import LiteLLMGateway
   gw = LiteLLMGateway(config.gateway)
   decision = router.route(request)
   response = gw.invoke(decision, messages=[...])
   ```

See [Backlog item #1](#backlog) for the recommended Anthropic-direct alternative.

### 5.6 Add agent failure detection (cascade)

In your agent loop, when a tool call fails:

```python
from llm_router import Outcome, OutcomeKind

next_request = RoutingRequest(
    prompt=user_prompt,
    session_id=session_id,
    agent_step_type=AgentStepType.TOOL_CALL,
    planned_tool="edit_file",
    recent_outcomes=[
        Outcome(kind=OutcomeKind.TOOL_SCHEMA_ERROR, tool_name="edit_file"),
    ],
)
```

The `RecentFailureRule` will force `STRONG` on this turn. The router stays stateless; the caller curates the outcome list (typically the last 1-3 outcomes, dropping older ones).

### 5.7 Per-tenant override

```yaml
tenants:
  overrides:
    customer-acme:
      forced_tier: strong            # always strong, no exceptions
    customer-budget:
      latency_strict: true           # skip classifier (saves embedding latency)
      classifier_p_high_override: 0.55
    customer-paranoid:
      blocked_tiers: [weak]          # weak is forbidden; auto-upgrade
```

---

## 6. Backlog

Prioritized list of next-up work. Each item: what, why, where to start, effort estimate.

### High priority — do these before scaling traffic

**[H1] Anthropic-direct gateway adapter** · effort: S
Why: native Claude tool-call schema is cleaner than going through LiteLLM for an Anthropic-first stack; lower latency; better error surface for `Outcome` translation.
Where: create `src/llm_router/gateway/anthropic_adapter.py` mirroring `litellm_adapter.py`. Map `RoutingDecision.tier` → concrete model id; pass through messages + tools; return canonical response. Add `Outcome` synthesizer that inspects responses for schema errors / refusals and produces the right `OutcomeKind`.

**[H2] Decision-log persistence sink** · effort: S
Why: today logs go to stdout only. Production needs durable logs for shadow eval, replay, drift detection.
Where: add `observability/sinks.py` with `JsonlFileSink`, `KafkaSink` (stub), `MultiSink`. Wire via `LoggingConfig` (extend with `sinks: list[SinkConfig]`).

**[H3] Real-data evaluation pipeline (Phase 0)** · effort: M
Why: every threshold in classifier/agent rules is guessed until we evaluate on real prompts. This is the prerequisite I've been flagging from day one.
Where: write `scripts/build_eval_set.py` that takes a JSONL of prompts, runs each through 2+ models (use the gateway from H1), and emits the `EvalRecord` JSONL format expected by `eval/harness.py`. Add an LLM-as-judge scorer (`eval/scorer_llm_judge.py`) that pairwise-grades outputs with a stronger model.

**[H4] Redis-backed session store** · effort: S
Why: in-memory store doesn't survive restart and doesn't share across workers. Needed for multi-worker `uvicorn` deployments.
Where: `src/llm_router/session/redis_store.py`. Same `SessionStore` interface. Use `redis-py` (sync) or `redis.asyncio`. Key schema: `llmr:sess:{session_id}` → JSON of `SessionState`.

### Medium priority — for production hardening

**[M1] Shadow traffic runner** · effort: M
Why: lets you test a new policy on live traffic without acting on it.
Where: `src/llm_router/eval/shadow.py`. Take two `Router` instances. On each request, route through both; emit a `shadow_decision` log event with both decisions + agreement flag. The "shadow" never reaches a gateway. Stub already exists at `eval/harness.shadow_run()`.

**[M2] Decision-log replay tool** · effort: M
Why: re-evaluate a candidate policy against the labeled-triple dataset using historical decision logs (no model re-runs).
Where: `src/llm_router/eval/replay.py`. Reads JSONL decision logs + the eval dataset. For each logged decision, replace the original policy's tier with the candidate's, look up quality/cost from the dataset, aggregate. Stub at `eval/harness.replay_decisions()`.

**[M3] Output-driven cascade scanner** · effort: M
Why: extends Phase 2.5 cascade to cases where caller can't observe failure but the model output is detectably bad (parse failure, schema mismatch, lint error, test failure).
Where: `src/llm_router/gateway/cascade.py`. After the gateway returns, run a list of validators on the output. If any fail, synthesize an `Outcome`, re-call `router.route(...)` with the failure, and re-invoke at the higher tier.

**[M4] Drift detector** · effort: M
Why: query distribution shifts over time; we need to know when rules / classifier need re-tuning.
Where: `src/llm_router/observability/drift.py`. Consume the decision-log sink. Compute, on a rolling 1h/24h/7d window:
- rule fire-rate per rule
- classifier output distribution (histogram of `p_strong`)
- session stickiness escalation rate
Emit alerts when distributions deviate from baseline by configurable thresholds.

**[M5] Integration guide** · effort: S
Why: smooth onboarding for teams using OpenAI Agents SDK / LangGraph / Anthropic SDK.
Where: `docs/INTEGRATION.md`. Three sections: OpenAI Agents SDK, LangGraph, custom loops. For each: where to call `router.route(...)`, how to pass `recent_outcomes`, how to wrap the gateway.

**[M6] Add tool args as a classifier feature** · effort: S
Why: huge tool args (e.g. 2k-char JSON) signal complex calls that may need strong even on otherwise-safe tools.
Where: `src/llm_router/classifier/features.py` (new). Concatenate prompt embedding with structural numeric features: `[log(len(prompt)), log(tool_arg_chars), n_tools, has_code_block, ...]`. Re-train classifier head. Update `predictor.py` to accept the structural inputs.

**[M7] Per-tenant config hot-reload** · effort: S
Why: enterprise customers expect their forced-tier / blocked-tier config changes to take effect without restarting workers.
Where: extend `TenantPolicyResolver` to watch a config file (inotify) or pull from a remote source (HTTP poll). Versioning + atomic swap. Be careful: don't introduce a request-time blocking I/O.

### Lower priority — nice to have

**[L1] KNN long-tail router** · effort: L
Why: catch the cases where rules + classifier are wrong, by storing user-feedback bad cases in a vector DB and routing similar future prompts to strong.
Where: new module `src/llm_router/classifier/knn_longtail.py`. Embedding + cosine similarity over a Redis/Qdrant-backed store. Read feedback signals from the decision-log sink. Phase 3.

**[L2] Bandit threshold tuning** · effort: L
Why: automatically tune `p_high` / `p_low` based on observed cost-quality trade-offs.
Where: new module `src/llm_router/eval/bandit_tuner.py`. Thompson sampling over a small grid of threshold pairs; reward = (quality × user_weight) − (cost × cost_weight). Phase 4. Defer until offline tuning hits diminishing returns.

**[L3] CI/CD (GitHub Actions)** · effort: S
Why: catch regressions on PRs.
Where: `.github/workflows/ci.yml`. Run pytest, ruff, mypy on 3.10/3.11/3.12. Don't pull `[ml]` extra in CI (slow); rely on HashingEmbedder.

**[L4] Dockerfile + compose** · effort: S
Why: standard deploy artifact.
Where: `Dockerfile`, `docker-compose.yml`. Multi-stage: build wheel, slim runtime. Include `[ml]` extra in a separate "with-ml" image.

**[L5] `/metrics` Prometheus endpoint** · effort: S
Why: standard ops integration.
Where: add to `server/app.py`. Use `prometheus_client`. Counters: decisions by (tier, layer, tenant); histograms: elapsed_ms, p_strong; gauges: session count.

**[L6] A/B testing harness** · effort: M
Why: route hash(tenant + session) to N policies in production for online comparison.
Where: `src/llm_router/eval/ab.py`. Wraps multiple `Router` instances behind a deterministic-hash assignment. Logs assignment alongside decision.

**[L7] Multi-region session partitioning** · effort: M
Why: cross-region session lookups have unacceptable latency; partition by tenant or session-id prefix.
Where: a `PartitionedSessionStore` that routes by tenant_id → Redis cluster. Depends on H4.

---

## 7. Open questions for the user

These need human input — usually a product/business call, not a code decision.

1. **Real eval data source.** Do we have access to production queries with labels, or do we need to bootstrap with LLM-as-judge on a sampled subset? Determines the timeline for H3.
2. **Tier vocabulary.** Is binary (weak/strong) enough, or is there a `mid` model (e.g. Sonnet between Haiku and Opus)? Schema supports 3 tiers; rules and classifier currently emit binary.
3. **Multilingual mix.** What's the actual language distribution? Default embedding is multilingual; if traffic is >95% English, swap to `all-MiniLM-L6-v2` (≈10% accuracy gain on English).
4. **Streaming constraint.** Do downstream calls stream? Streaming + output-driven cascade (M3) need speculative-decoding tricks to avoid latency penalty. Affects M3 scope.
5. **Session id source.** Who issues session ids? Affects whether the stickiness store can be shared across regions or must be per-region (L7).
6. **Tool whitelist truth source.** Where does the canonical list of tool names live in your agent framework? We need to keep `AgentConfig.weak_safe_tools` / `requires_strong_tools` in sync with reality.
7. **Failure signal pipeline.** Which of these failure signals are you confident you can produce in production? schema-validation, tool-execution-error, parse-error (JSON/diff), validation-error (test/lint), retry-attempt, user-thumbs-down. Determines how many `OutcomeKind`s actually fire.

---

## 8. Decision log (append-only)

- **2026-05-06** — started repo. Python 3.10+, FastAPI, pydantic v2,
  sentence-transformers, sklearn LR. OSS comparison documented in
  DESIGN.md §3.
- **2026-05-06** — chose `multilingual-e5-small` over
  `all-MiniLM-L6-v2` as default embedding because the target audience
  needs Chinese support.
- **2026-05-06** — tier vocabulary fixed at `{weak, mid, strong}`.
  v0 rule/classifier outputs are binary; `mid` slot exists for future
  use without a schema migration.
- **2026-05-06** — stickiness defaults to upgrade-only. Per-tenant
  override is supported but discouraged.
- **2026-05-06** — classifier output is calibrated probability, not
  raw margin. Required for principled threshold tuning.
- **2026-05-14** — added agent-mode support.
  - `RoutingRequest` keeps backward compat: `prompt` and `messages`
    both optional, validator requires at least one.
  - Tool whitelists are config-driven, not hardcoded.
  - Cascade is outcome-driven from the caller, not stateful in router.
  - Stickiness is keyed by `(session_id, step_type)`. Default
    non-sticky step types in agent preset: `tool_call`, `tool_result`,
    `summarize`.
  - Step-type auto-detection is heuristic; explicit `agent_step_type`
    from the caller always wins.
- **2026-05-14** — `RoutingRequest.metadata['detected_step']` is
  mutated by the router so agent rules can read step type without
  passing config explicitly. Intentional shared mutation; documented
  in `core/router.py`.

---

## 9. Smoke-test results

**Base build (2026-05-06)**

- `pytest tests/` — 30/30 pass.
- `examples/eval_run.py` on synthetic eval set:
  - always-weak baseline:   q=0.701, cost=9.17
  - always-strong baseline: q=0.920, cost=275.05
  - rules + classifier:     q=0.881, cost=268.35

**Agent build (2026-05-14)**

- `pytest tests/` — 52/52 pass (added 22 agent tests covering step
  detection, all 8 agent rules, per-step-type stickiness, schema
  validation, backward compat).
- `examples/agent_usage.py` — full 8-step Cursor-style agent flow
  routes correctly: planning→strong, safe tool→weak, tool result→weak,
  requires-strong tool→strong, failure escalation→strong, non-sticky
  step types verified.

Numbers reflect synthetic data, not production. Real Pareto
improvements depend entirely on the Phase-0 labeled set (H3).
