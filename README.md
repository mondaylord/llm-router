# llm-router

A production-oriented model routing layer that decides, for each incoming request,
which model tier (e.g. `weak` / `strong`) should serve it — so cheap models handle
easy traffic and expensive models are reserved for queries that need them.

This is a **base implementation** intended for further iteration. It implements
**Stage 1 (rule-based early-exit)** and **Stage 2 (embedding + calibrated classifier)**
of the staged plan in [docs/PLAN.md](docs/PLAN.md), plus **Stage 2.5 (agent-mode
routing)** for Cursor-style auto modes where the same router handles planning
steps, tool calls, tool-result interpretation, code edits, and failure-driven
escalation across an agent loop. Hooks are left in for later stages (output-driven
cascade, KNN long-tail, bandit threshold tuning).

## Picking up where you left off

Returning to this repo after a break? Start at [docs/STATUS.md](docs/STATUS.md)
— it's the short "where am I, what's next" pointer. [docs/PROGRESS.md](docs/PROGRESS.md)
is the deep reference (file map, recipes, full backlog).

## Why not just use an existing OSS router?

See [docs/DESIGN.md](docs/DESIGN.md#open-source-landscape) for a full comparison.
Short version:

- **RouteLLM** (lm-sys) is the closest, but is binary strong/weak, single-turn,
  English-centric, and has no session stickiness, tenant overrides, or
  observability hooks. Useful as classifier inspiration; not as a base.
- **LiteLLM** is a great downstream gateway (provider abstraction, retries,
  fallback). We use it as an *optional* downstream — it is not a quality router.
- **FrugalGPT / Hybrid LLM** are research references for cascade techniques.

We need to own the policy layer because production correctness lives in the
parts those projects don't ship: per-tenant config, session stickiness, shadow
evaluation, decision logging, and clean interfaces for swapping classifiers.

## Layout

```
llm-router/
├── docs/
│   ├── DESIGN.md     architecture, decisions, tradeoffs, OSS comparison
│   ├── PLAN.md       phased roadmap
│   ├── PROGRESS.md   what's done, what's next, decisions log
│   └── EVAL.md       evaluation methodology (the part most teams skip)
├── src/llm_router/
│   ├── core/         router pipeline, decision/config types
│   ├── rules/        Stage 1: rule-based early-exit
│   ├── classifier/   Stage 2: embedding + calibrated classifier
│   ├── session/      session-state store (sticky routing)
│   ├── policy/       stickiness + per-tenant overrides
│   ├── eval/         offline + shadow evaluation harness
│   ├── server/       FastAPI HTTP service
│   ├── gateway/      downstream model invocation (LiteLLM-compatible)
│   └── observability/structured logging
├── tests/
├── examples/         basic_usage / train_classifier / eval_run
└── scripts/          seed synthetic data
```

## Quick start

```bash
pip install -e ".[dev]"

# 1. Generate synthetic labeled data and train a classifier
python examples/train_classifier.py

# 2. Run the router as a library
python examples/basic_usage.py

# 3. Run the offline eval harness
python examples/eval_run.py

# 4. Start the HTTP service
uvicorn llm_router.server.app:app --reload
```

## Core API (chat)

```python
from llm_router import Router, RouterConfig, RoutingRequest

router = Router.from_config(RouterConfig.load("config.yaml"))

decision = router.route(RoutingRequest(
    prompt="What is 2 + 2?",
    session_id="sess-123",
    tenant_id="customer-acme",
))
# decision.tier == "weak"
# decision.reason == "rule:very_short_query"
# decision.confidence == 1.0
```

## Core API (agent — Cursor-style auto mode)

```python
from llm_router import (
    Router, RouterConfig, RoutingRequest,
    AgentStepType, Outcome, OutcomeKind,
)

router = Router.from_config(RouterConfig.load("agent_preset.yaml"))

# Planning step → strong
router.route(RoutingRequest(
    prompt="Plan how to refactor the auth middleware.",
    session_id="agent-123",
))

# Safe tool call → weak
router.route(RoutingRequest(
    prompt="Read auth.py",
    session_id="agent-123",
    agent_step_type=AgentStepType.TOOL_CALL,
    planned_tool="read_file",
))

# High-stakes tool → strong
router.route(RoutingRequest(
    prompt="Apply this edit",
    session_id="agent-123",
    agent_step_type=AgentStepType.TOOL_CALL,
    planned_tool="edit_file",
))

# Last turn failed → escalate
router.route(RoutingRequest(
    prompt="Try again",
    session_id="agent-123",
    agent_step_type=AgentStepType.TOOL_CALL,
    planned_tool="edit_file",
    recent_outcomes=[Outcome(kind=OutcomeKind.TOOL_SCHEMA_ERROR, tool_name="edit_file")],
))
```

See [examples/agent_usage.py](examples/agent_usage.py) for a full run, and
[examples/agent_preset.example.yaml](examples/agent_preset.example.yaml) for
the agent-mode YAML config.

## Status

Stage 1 + Stage 2 + Stage 2.5 (agent) implemented end-to-end with synthetic
data. See [docs/PROGRESS.md](docs/PROGRESS.md) for what is real vs scaffolded.
