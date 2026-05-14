# Plan

Phased roadmap for taking `llm-router` from base implementation to
production-grade. Each phase builds on the prior one — do not skip.

The base repo (this commit) covers **Phase 0 + Phase 1 + Phase 2**.
Phases 3-5 are designed-in but not implemented; interfaces exist where
relevant so adding them does not require restructuring.

---

## Phase 0 — Evaluation infrastructure (prereq for everything)

**Goal**: prove how much there is to gain before changing routing logic.

- Sample N=1k–5k real prompts from production (or use synthetic data).
- Run each prompt through every candidate model, store outputs.
- Score (model, prompt) → quality, via:
  - LLM-as-judge (strong model pairwise), or
  - Reference matching (for tasks with golden answers), or
  - Human eval (gold subset).
- Compute **ideal-router Pareto curve**: for each (cost, quality)
  budget, the optimal per-prompt assignment.

**Exit criterion**: a chart showing potential cost reduction at
acceptable quality loss (e.g. "−40% cost at −2% quality"). If the
ceiling is too low, stop here and don't ship a router.

**Status in repo**: `src/llm_router/eval/` provides a harness that
ingests labeled `(prompt, model, quality)` triples and emits Pareto
curves. The actual model invocation must be done outside (use whatever
you use in production — we do not bundle that step here).

---

## Phase 1 — Rule-based early-exit (in production within ~2 weeks)

**Goal**: capture obvious cheap-tier traffic with zero ML risk.

- Implement 10–20 **high-precision** rules. Examples:
  - `len(prompt) < 40 chars` → `weak`
  - `is_pure_classification` (yes/no, multiple choice patterns) → `weak`
  - `is_translation_short` (single sentence, language detected) → `weak`
  - `contains_code_block AND len(code) > N` → `strong`
  - `mentions_finance_legal_medical_keywords` → `strong`
- Default for non-matched prompts: **`strong`** (no quality regression).
- Per-tenant override layer: enterprise customers can disable rules.

**Exit criterion**: shadow run for 7d shows ≥30% of traffic gets a
non-default decision; quality on that traffic is within 2% of "always
strong"; latency p99 added < 5ms.

**Status in repo**: implemented end-to-end with 8 builtin rules. See
`src/llm_router/rules/builtin.py`.

---

## Phase 2 — Embedding + calibrated classifier (1–2 months)

**Goal**: handle the ambiguous middle that rules cannot capture.

- Train data: Phase-0 labeled set, augmented with rule-engine traces.
- Embed prompts with `multilingual-e5-small` (default; configurable).
- Classifier: calibrated logistic regression (sklearn).
- Output: `p(needs_strong)` ∈ [0, 1].
- Decision: low_thresh / high_thresh band; uncertainty → strong (safe).
- Run **after** rules, **before** default.
- Train with k-fold + isotonic calibration so probabilities are usable
  thresholds, not arbitrary scores.

**Exit criterion**: combined Phase 1+2 router achieves ≥70% of the
ideal Pareto curve from Phase 0.

**Status in repo**: implemented with synthetic-data trainer and
predictor. Real production deployment needs Phase-0 data substituted.

---

## Phase 2.5 — Agent-mode routing (implemented 2026-05-14)

**Goal**: make the same router serve both chat and agent (Cursor-style)
flows. Different traffic class, same primitives.

- `RoutingRequest` extended with `messages`, `agent_step_type`,
  `available_tools`, `planned_tool`, `recent_outcomes`,
  `total_context_tokens`.
- New rule layer `rules/agent.py` runs before chat rules.
- Cascade realized as a rule that fires on `recent_outcomes` —
  router stays stateless; the caller observes failures and passes
  them back next turn.
- Stickiness keyed by `(session_id, step_type)`; default non-sticky
  for `tool_call`, `tool_result`, `summarize`.

**Why early**: agent traffic is where mis-routing is most expensive
(a bad tool-call routing breaks the run, not just costs more). The
work was originally slotted in Phase 3 cascade but split out because
it needs a schema change.

**Exit criterion**: same as Phase 2 (within 70% of ideal Pareto curve)
but evaluated separately on agent-stratified traffic.

---

## Phase 3 — Long-tail and output-driven cascade

**Goal**: catch the cases where rules + classifier fail, AND where the
failure signal is only available in the model output.

- **Output-driven cascade** (extends Phase 2.5): for the non-streaming
  paths, the gateway adapter can post-score weak-model output (parse,
  schema-validate, lint) and call `router.route(...)` again with a
  fresh `Outcome` to escalate. This is a thin wrapper around what
  Phase 2.5 already supports.
- **KNN long-tail**: collect user-feedback bad cases (👎, retries,
  copy-then-edit signals). Embed and store in a vector DB. At route
  time, if any neighbor of the incoming prompt has a known-bad-on-weak
  label, force `strong`.

**Status**: interfaces designed, not implemented. The decision logger
already emits the data the KNN store needs to consume.

---

## Phase 4 — Per-tenant policy & threshold tuning

**Goal**: meet enterprise contract requirements without forking code.

- Tenant-level config: forced tiers, allowed tiers, custom thresholds,
  rule disabling, latency-strict mode (skip Layer 2).
- Threshold auto-tuning: bandit or grid search over recent decision
  logs + quality scores.

**Status**: tenant policy is implemented. Auto-tuning is not.

---

## Phase 5 — Operations & observability

**Goal**: make the router safe to leave alone.

- Shadow-traffic mode: candidate policy runs in parallel, emits diffs;
  no acting on its decisions.
- Decision-log replay tool: re-route historical logs through a new
  policy without rerunning models, compare metric impact.
- Drift alerting: rolling distribution monitor over rule-fire-rates and
  classifier output distribution.
- A/B harness: split traffic by hash(tenant + session) to N policies,
  collect quality metrics, decide.

**Status**: decision logger exists; replay tool, shadow runner, and
drift detector are scaffolded with TODOs in eval module.

---

## What we explicitly do NOT plan to do

- Train a foundation model. Off-the-shelf embeddings are sufficient.
- Build a model gateway. LiteLLM is the right tool.
- Build a serving stack. vLLM/SGLang etc. own that.
- Add a UI. Configuration is files; observability is logs to existing
  systems.
