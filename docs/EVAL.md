# Evaluation Methodology

The router is only as good as the data we use to tune and verify it.
This document describes the evaluation approach the codebase is built
around. Skipping any step puts production quality at risk.

---

## 1. The labeled triple

The atomic unit of evaluation is a triple:

```
(prompt, model, quality_score)
```

- `prompt` — the user input we want to route.
- `model` — one of the candidate tiers (or concrete models).
- `quality_score` — a number in [0, 1] representing how good the model's
  response is for this prompt.

A dataset is a collection of these triples covering every (prompt, model)
pair you want to compare. You need this dataset *before* you tune any
threshold.

JSONL schema in `src/llm_router/eval/dataset.py`:

```json
{"prompt_id": "p_001", "prompt": "...", "model": "weak",   "quality": 0.71, "cost": 0.0003}
{"prompt_id": "p_001", "prompt": "...", "model": "strong", "quality": 0.94, "cost": 0.0091}
```

---

## 2. How to obtain quality scores

Pick the cheapest method that meets your accuracy bar:

| Method                     | Cost     | Accuracy | When to use                   |
| -------------------------- | -------- | -------- | ----------------------------- |
| Reference matching         | $0       | High     | Tasks with golden answers     |
| Rule-based (length, etc.)  | $0       | Low      | Sanity checks only            |
| LLM-as-judge (pairwise)    | $$       | Medium-H | Open-ended generation         |
| LLM-as-judge (rubric)      | $$       | Medium   | Mixed tasks                   |
| Human annotation (sampled) | $$$      | Highest  | Calibration of LLM judges     |

In practice: **LLM-as-judge pairwise**, with a small human-labeled
calibration set, is the workable default. Use a STRONGER model than
either candidate as the judge (e.g. judge with Opus when comparing
Haiku vs Sonnet). Beware position bias — randomize order.

---

## 3. The Pareto-optimal baseline

Given the labeled triples, you can compute the **ideal-router curve**:
for each desired quality budget, the minimum total cost achievable by
an oracle that picks the cheapest model that meets quality per prompt.

This is the ceiling. Every real router operates strictly below it.

`harness.compute_pareto_curve()` returns this curve.

---

## 4. Router evaluation

For each candidate routing policy:

1. Run it on every prompt in the eval set → predicted tier per prompt.
2. Look up `(prompt, predicted_tier).quality` and `.cost`.
3. Sum across the dataset.
4. Plot point (total_cost, mean_quality) on the Pareto chart.

A "good" router lies on or close to the ideal curve. A "bad" router
is far below it (less quality at same cost) or far to the right (more
cost at same quality).

---

## 5. Threshold tuning

For Stage-2 classifier with a single threshold:

- Sweep threshold from 0 to 1 in 0.01 steps.
- For each, compute (cost, quality) on the eval set.
- Pick the threshold that gives the cost-quality point closest to your
  business target (e.g. "max cost reduction with quality drop ≤ 2%").

For two-threshold (low/high uncertainty band):

- Grid search over (low, high) where 0 ≤ low ≤ high ≤ 1.
- Same selection logic.

`harness.sweep_thresholds()` does this; output is a CSV you can plot.

---

## 6. Shadow traffic (production)

Once you have an offline-validated policy, before you ship:

1. Deploy the new policy in **shadow mode** — it runs in parallel with
   the live policy on the same requests but its decision is **not**
   acted on.
2. For each request, log both decisions and the actual outcome (which
   live policy was used + quality signals like user feedback, retry
   rate, copy rate).
3. Diff the decisions: how often does shadow disagree, and on what
   slices?
4. Estimate counterfactual quality on the disagreed-on slice. This is
   the hardest step; LLM-as-judge again works.

`shadow.py` is scaffolded for Phase 5; its interface accepts two
`Router` instances and a stream of requests.

---

## 7. Decision-log replay

Every routing decision is logged with the prompt, scores, layer, and
predicted tier. Given that log + the labeled-triple dataset (or any new
quality scores), you can re-route historical traffic through a new
policy without re-running models.

This is how you evaluate threshold changes, new rules, or new
classifier weights cheaply. It's the most-used eval tool in steady
state.

`harness.replay_decisions()` is scaffolded.

---

## 8. Drift monitoring

In production, watch:

- **Rule fire-rate distribution**: a sudden change in % of traffic
  hitting each rule means query distribution shifted.
- **Classifier output distribution**: KL-divergence vs training-time
  distribution. Big shift → retrain.
- **Stickiness escalation rate**: % of sessions that ended on `strong`
  after starting on `weak`. Tells you if Stage 2 is too aggressive in
  picking `weak`.
- **Tier reversal flags**: any decision logger entry where downstream
  feedback (👎, retry) suggests the tier was wrong.

These metrics are derivable from the decision log; the drift detector
is Phase 5 work.

---

## 9. Anti-patterns to avoid

- **Tuning thresholds on the same set you evaluate on**. Hold out a
  test set; only ever tune on dev.
- **Using only single-turn data**. Multi-turn behaviour is fundamentally
  different; collect multi-turn samples even if rare.
- **Ignoring tail latency**. The router itself adds latency; measure
  p50/p95/p99 explicitly, not just averages.
- **Treating "quality drop" as a single number**. Stratify by tenant,
  language, task type — a 2% mean drop can hide a 20% drop on one
  customer.
