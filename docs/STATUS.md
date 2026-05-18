# Status — read this when you come back

A short "where am I, what's next" pointer. Updated each session.

**Last updated**: 2026-05-14
**HEAD**: [`cc78d26`](https://github.com/mondaylord/llm-router/commit/cc78d26)
**Tests**: 52 / 52 passing
**Stages shipped**: Stage 1 (chat rules) + Stage 2 (classifier) + Stage 2.5 (agent mode) + vSR-comparison review + TS client for reel-agent

---

## 1. Snapshot

The router routes both **chat** and **agent** traffic to weak / mid / strong tiers using:

- 8 chat rules + 8 agent rules (high precision, run before the classifier)
- A calibrated logistic-regression classifier over multilingual embeddings (trained, but only on synthetic data so far)
- Per-(session × step-type) stickiness, tenant overrides, outcome-driven cascade
- FastAPI service exposing `/route` and `/healthz`
- A zero-dep TypeScript client at [examples/clients/typescript/](../examples/clients/typescript/) for reel-agent

What's missing for production:
- A real labeled eval set (everything is tuned against synthetic data)
- A real gateway adapter that actually invokes models (Anthropic / OpenAI / LiteLLM)
- Durable decision-log persistence
- PII + jailbreak detection (added to backlog after vSR review)
- Semantic cache (same)

---

## 2. Session log (most recent first)

### 2026-05-14 — Session 3

- Did a structured review against **vLLM Semantic Router** ([docs/REVIEW_VS_VSR.md](REVIEW_VS_VSR.md))
- Added to backlog: H5 (PII), H6 (injection), H7 (semantic cache), H8 (reel-agent integration), M8 (OpenAI-compatible proxy mode), M9 (MCP awareness), L8 (MRL truncation)
- Wrote zero-dep **TypeScript client** + agent-loop demo + README in [examples/clients/typescript/](../examples/clients/typescript/) so reel-agent can call the router over HTTP without schema drift
- Re-confirmed: 52/52 tests still pass

### 2026-05-14 — Session 2

- Added **agent-mode routing** ([commit `fa12fa8`](https://github.com/mondaylord/llm-router/commit/fa12fa8))
- Extended `RoutingRequest` with `messages`, `agent_step_type`, `available_tools`, `planned_tool`, `recent_outcomes`, `total_context_tokens`
- Wrote 8 agent rules: failure escalation, requires-strong tool whitelist, long-context, planning, safe-tool whitelist, tool-result interpretation, edit, summarize
- Per-(session × step-type) stickiness with `non_sticky_step_types` for atomic steps
- 22 new tests; full agent flow demo in [examples/agent_usage.py](../examples/agent_usage.py)

### 2026-05-06 — Session 1

- Base implementation: chat rules + calibrated LR classifier + FastAPI + eval harness
- 30 tests; smoke-tested end to end on synthetic data
- Initial commit pushed to <https://github.com/mondaylord/llm-router>

---

## 3. Next session — do these in this order

Each item links to the detailed entry in the backlog. **Pick one and start.**

### 🟢 #1 (start here) — Wire a real gateway: Anthropic-direct adapter

[Backlog H1](PROGRESS.md#6-backlog) · effort: **S**

Without this, the router can't actually invoke models — `/route` only emits a decision. For an Anthropic-first stack (which reel-agent appears to be), the direct SDK is cleaner than going through LiteLLM.

**Concretely**:
- Create [src/llm_router/gateway/anthropic_adapter.py](../src/llm_router/gateway/anthropic_adapter.py).
- Mirror the shape of [litellm_adapter.py](../src/llm_router/gateway/litellm_adapter.py).
- Map `RoutingDecision.tier` → concrete model id from `GatewayConfig.tier_to_model`.
- Pass messages + tools through; return the raw Anthropic response.
- Add a small post-call helper that inspects the response for tool-schema / refusal errors and synthesizes an `Outcome`.

When done: reel-agent can call llm-router and get an end-to-end answer with no other glue.

### 🟢 #2 — Phase 0 eval data pipeline

[Backlog H3](PROGRESS.md#6-backlog) · effort: **M**

Every threshold today is guessed. Until we evaluate on real prompts, the cost-quality numbers in [examples/eval_run.py](../examples/eval_run.py) are about a toy dataset, not production.

**Concretely**:
- Write `scripts/build_eval_set.py`: take a JSONL of real prompts, run each through ≥2 tiers via the gateway from #1, save outputs.
- Write `eval/scorer_llm_judge.py`: pairwise LLM-as-judge with a stronger model (e.g. opus judging haiku-vs-sonnet outputs), with order randomization to fight position bias.
- Emit the canonical `EvalRecord` JSONL format already consumed by [eval/harness.py](../src/llm_router/eval/harness.py).
- Re-run `examples/eval_run.py` against the real eval set; tune `classifier.p_high` from the resulting curve.

When done: we have *evidence* for every threshold.

### 🟢 #3 — Safety: PII + prompt-injection detection

[Backlog H5](PROGRESS.md#6-backlog) + [H6](PROGRESS.md#6-backlog) · effort: **S + S** (can do in one session)

Borrowed from vSR. **Table stakes for shipping to enterprise customers** and especially for agent contexts where tool outputs can carry injection payloads.

**Concretely**:
- `src/llm_router/safety/pii.py` — regex detector covering email / IP / AWS keys / SSH keys / JWT / credit cards. Return `PIIReport(categories, spans)`.
- `src/llm_router/safety/injection.py` — pattern detector for known prompt-injection shapes ("ignore previous", "you are now ...", suspicious base64).
- Two new agent rules: `PiiPresentRule` (force STRONG if PII present, tenant-configurable) and `InjectionDetectedRule` (force STRONG when detected).
- Tests in `tests/test_safety.py`.

Both have a clean upgrade path to encoder models later. v0 ships in a half-day.

### 🟡 #4 (parallel-track) — Decision-log persistence

[Backlog H2](PROGRESS.md#6-backlog) · effort: **S**

Today every decision goes to stdout via structlog. For shadow eval / replay / drift detection we need durable logs.

**Concretely**: add `src/llm_router/observability/sinks.py` with `JsonlFileSink`, wire via a `sinks` list in `LoggingConfig`.

### Defer until something forces it

- **H4** Redis session store — needed only when you run >1 uvicorn worker
- **H7** Semantic cache — high ROI but needs a chunk of design work; do after #1-3
- **H8** reel-agent integration guide — already started with the TS client; expand into a full doc once reel-agent has real code calling it
- **M9** MCP awareness — only if reel-agent commits to MCP
- All "L*" items — only after the above land

---

## 4. Where to look

| If you want to ...                            | Read                                                  |
| --------------------------------------------- | ----------------------------------------------------- |
| Refresh memory on this whole project          | [README.md](../README.md)                             |
| Pick up coding immediately                    | this file, then [PROGRESS.md §6](PROGRESS.md#6-backlog)|
| Understand WHY a design decision was made     | [DESIGN.md](DESIGN.md)                                |
| Look at the phased plan                       | [PLAN.md](PLAN.md)                                    |
| Understand the evaluation methodology         | [EVAL.md](EVAL.md)                                    |
| See how we compare to vLLM Semantic Router    | [REVIEW_VS_VSR.md](REVIEW_VS_VSR.md)                  |
| Integrate from reel-agent (TS+Bun)            | [examples/clients/typescript/](../examples/clients/typescript/) |
| Run the agent demo                             | `python examples/agent_usage.py`                      |
| Run the chat demo                              | `python examples/basic_usage.py`                      |
| Run the eval harness                           | `python examples/eval_run.py`                         |
| Run tests                                      | `pytest tests/ -q`                                    |

---

## 5. Open questions you owe yourself an answer to

(Copied from [PROGRESS.md §7](PROGRESS.md#7-open-questions-for-the-user) for visibility — answer these before the eval pipeline matters.)

1. **Real eval data source**: do you have access to real prompts + labels, or do we bootstrap with LLM-as-judge?
2. **Tier vocabulary**: binary (weak/strong) or three tiers with `mid` (Sonnet)?
3. **Multilingual mix**: language distribution? Default embedding is multilingual.
4. **Streaming**: does the downstream stream? Affects whether output-driven cascade is even possible.
5. **Tool whitelist source of truth**: where does the canonical tool list live in reel-agent?
6. **Failure signal pipeline**: which `OutcomeKind`s can reel-agent's verifier actually emit?

---

## 6. Boot procedure

When you actually sit down to start a session:

```bash
cd /opt/nvme/home/mondaylord/llm-router

# 1. Make sure tests still pass before you change anything.
.venv/bin/pytest tests/ -q

# 2. Pull anything you forgot to push from another machine.
git status
git pull --rebase origin main

# 3. Pick an item from §3 above. Start a feature branch.
git checkout -b feat/<short-name>

# 4. When done, run tests, commit, push, open PR (or merge to main).
.venv/bin/pytest tests/ -q
git add -A && git commit -m "feat(...): ..."
git push origin <branch>
```
