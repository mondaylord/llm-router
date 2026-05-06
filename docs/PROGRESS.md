# Progress

Living log of what is implemented, what is scaffolded, and key decisions
made along the way. Append-only timeline at bottom.

---

## Implementation matrix

| Component                            | Status       | Notes                                          |
| ------------------------------------ | ------------ | ---------------------------------------------- |
| `core/decision.py` types             | ✅ done      | `RoutingRequest`, `RoutingDecision`, `Tier`    |
| `core/config.py`                     | ✅ done      | YAML-loadable, validated by pydantic           |
| `core/router.py` pipeline            | ✅ done      | Pre-router → rules → classifier → synthesizer  |
| `rules/base.py` rule interface       | ✅ done      | `Rule` ABC, `RuleResult`                       |
| `rules/builtin.py` rules             | ✅ done      | 8 rules covering common patterns               |
| `classifier/embedding.py`            | ✅ done      | sentence-transformers wrapper, cached          |
| `classifier/predictor.py`            | ✅ done      | Loads joblib artifact, predicts `p(strong)`    |
| `classifier/trainer.py`              | ✅ done      | k-fold + isotonic calibration                  |
| `session/store.py`                   | ✅ done      | In-memory impl + abstract base                 |
| `policy/stickiness.py`               | ✅ done      | Upgrade-only by default                        |
| `policy/tenant.py`                   | ✅ done      | Forced tier, blocked tier, latency mode        |
| `eval/harness.py`                    | ✅ done      | Pareto curve over labeled (prompt, model, q)   |
| `eval/dataset.py`                    | ✅ done      | JSONL loader; sample synthetic generator       |
| `server/app.py` FastAPI              | ✅ done      | `/route` + `/healthz`                          |
| `server/schemas.py`                  | ✅ done      | request/response pydantic models               |
| `gateway/base.py` provider abstract  | ✅ scaffolded| LiteLLM adapter is left as a stub              |
| `observability/logger.py`            | ✅ done      | structlog JSON output                          |
| Tests (unit, rule, router, classifier) | ✅ done    | sklearn/numpy only — no GPU needed             |
| Synthetic data generator             | ✅ done      | `scripts/seed_data.py`                         |
| Examples (basic / train / eval)      | ✅ done      | runnable                                       |
| Cascade                              | ❌ later     | Phase 3                                        |
| KNN long-tail                        | ❌ later     | Phase 3                                        |
| Shadow traffic runner                | ❌ later     | Phase 5                                        |
| Decision-log replay                  | ❌ later     | Phase 5                                        |
| Drift detector                       | ❌ later     | Phase 5                                        |
| Auto-threshold tuning                | ❌ later     | Phase 4                                        |

---

## What's "real" vs "scaffolded"

**Real** (works end-to-end with included data):

- Rule engine with 8 rules; exercised by tests.
- Classifier training and inference. The synthetic-data trainer
  (`examples/train_classifier.py`) produces a working model file.
- Session-stickiness logic (in-memory).
- Tenant policy lookup.
- Decision logging (structured JSON).
- Eval harness Pareto-curve computation.
- FastAPI server with `/route`.

**Scaffolded** (interface present, swap in real impl):

- Embedding backend defaults to sentence-transformers (`ml` extra).
  A `HashingEmbedder` fallback exists so unit tests don't pull torch.
- Gateway is an abstract base. The `LiteLLMGateway` adapter is a stub
  showing the call shape — wire it up when you connect downstream.
- Quality scorer in eval module is pluggable; ships with a "load from
  JSONL" impl. Plug your own LLM-as-judge if needed.

---

## Smoke-test results (2026-05-06, base build)

- `pytest tests/` — 30/30 pass (rules, router pipeline, classifier
  train/predict roundtrip, FastAPI route + healthz).
- `examples/train_classifier.py` — trains on 800 synthetic prompts in
  ~1s using `HashingEmbedder`; cv accuracy = 1.000 (synthetic data is
  intentionally separable; do not interpret this as production quality).
- `examples/eval_run.py` on synthetic eval set:
  - always-weak baseline:   q=0.701, cost=9.17
  - always-strong baseline: q=0.920, cost=275.05
  - rules + classifier:     q=0.881, cost=268.35
    (rules carried 86% of decisions, classifier 14%)
- FastAPI `/route` and `/healthz` return correct decisions in-process
  via `httpx.TestClient`.

The eval numbers reflect the toy dataset, not production. They confirm
the harness math is correct; real Pareto improvements depend entirely
on the Phase-0 labeled set.

---

## Decision log

- **2026-05-06**: started repo. Settled on Python, FastAPI, pydantic v2,
  sentence-transformers, sklearn LR. Documented OSS comparison in
  DESIGN.md §3.
- **2026-05-06**: chose `multilingual-e5-small` over `all-MiniLM-L6-v2`
  as default embedding because the target audience requires Chinese
  support out of the box.
- **2026-05-06**: tier vocabulary fixed at `{weak, mid, strong}`. v0
  rule/classifier outputs are binary (weak vs strong); `mid` slot
  exists for future use without a schema migration.
- **2026-05-06**: stickiness defaults to upgrade-only. Per-tenant
  override is supported but discouraged.
- **2026-05-06**: classifier output is calibrated probability, not raw
  margin. Required for principled threshold tuning.

---

## Open questions for the user

1. **Eval data**: do we have access to real production queries with
   labels, or should we plan to bootstrap from LLM-as-judge on a
   sampled subset? This determines Phase 0 timeline.
2. **Tier count**: is binary (weak / strong) enough, or do you have a
   `mid` model already (e.g. Sonnet between Haiku and Opus)? The schema
   supports 3 tiers; rules and classifier currently emit binary.
3. **Multilingual mix**: what's the language distribution? Default
   embedding is multilingual; if traffic is 99% English we can swap to
   `all-MiniLM-L6-v2` for ~10% accuracy gain on English.
4. **Streaming constraint**: do downstream calls stream? Streaming +
   cascade are incompatible without speculative decoding tricks. Affects
   Phase 3.
5. **Session id source**: who issues session ids? Affects whether the
   stickiness store can be shared across regions or must be per-region.
