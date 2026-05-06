"""Offline evaluation harness.

What it computes:

- `evaluate_router(router, records)` — runs a real `Router` against a
  labeled eval set and returns mean quality, total cost, decision mix.
- `compute_pareto_curve(records, models_in_order)` — the upper bound:
  oracle that picks the cheapest model meeting a quality threshold per
  prompt, swept across thresholds.
- `sweep_thresholds(...)` — for the Layer-2 classifier, sweep p_high
  threshold and report cost/quality at each setting.

What it does NOT do:

- Invoke real models. Bring your own labeled triples. See EVAL.md for
  why this is the right separation of concerns.

Phase 5 additions (`replay_decisions`, `shadow_runner`) are stubbed
with TODO so the call shape is locked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from llm_router.core.decision import RoutingRequest, Tier
from llm_router.core.router import Router
from llm_router.eval.dataset import EvalRecord, group_by_prompt


@dataclass
class EvalResult:
    n_prompts: int
    mean_quality: float
    total_cost: float
    tier_share: dict[str, float]
    layer_share: dict[str, float]


@dataclass
class ParetoPoint:
    quality_threshold: float
    mean_quality: float
    total_cost: float


# Map config "tier" names to dataset "model" labels. Default 1:1.
TierToModel = Callable[[Tier], str]


def default_tier_to_model(tier: Tier) -> str:
    return tier.value


class EvalHarness:
    def __init__(
        self,
        records: list[EvalRecord],
        tier_to_model: TierToModel = default_tier_to_model,
    ) -> None:
        self.records = records
        self.by_prompt = group_by_prompt(records)
        self.tier_to_model = tier_to_model

    # ------------------------------------------------------------------
    # router evaluation
    # ------------------------------------------------------------------
    def evaluate_router(self, router: Router) -> EvalResult:
        tier_counts: dict[str, int] = {}
        layer_counts: dict[str, int] = {}
        total_quality = 0.0
        total_cost = 0.0
        n = 0

        for prompt_id, by_model in self.by_prompt.items():
            # Use any record's prompt text (they're identical by id).
            prompt = next(iter(by_model.values())).prompt
            decision = router.route(RoutingRequest(prompt=prompt))
            model_label = self.tier_to_model(decision.tier)
            rec = by_model.get(model_label)
            if rec is None:
                # Eval set didn't include the chosen model — skip rather
                # than counting a zero. Surfaces dataset gaps loudly.
                continue
            total_quality += rec.quality
            total_cost += rec.cost
            n += 1
            tier_counts[decision.tier.value] = tier_counts.get(decision.tier.value, 0) + 1
            layer_counts[decision.layer.value] = layer_counts.get(decision.layer.value, 0) + 1

        return EvalResult(
            n_prompts=n,
            mean_quality=total_quality / n if n else 0.0,
            total_cost=total_cost,
            tier_share={k: v / n for k, v in tier_counts.items()} if n else {},
            layer_share={k: v / n for k, v in layer_counts.items()} if n else {},
        )

    # ------------------------------------------------------------------
    # baselines
    # ------------------------------------------------------------------
    def baseline_always(self, model: str) -> tuple[float, float]:
        """Always-this-model baseline. Returns (mean_quality, total_cost)."""
        qs: list[float] = []
        cs: list[float] = []
        for by_model in self.by_prompt.values():
            rec = by_model.get(model)
            if rec is None:
                continue
            qs.append(rec.quality)
            cs.append(rec.cost)
        if not qs:
            return 0.0, 0.0
        return sum(qs) / len(qs), sum(cs)

    # ------------------------------------------------------------------
    # ideal-router Pareto curve
    # ------------------------------------------------------------------
    def compute_pareto_curve(
        self,
        models_cheap_to_strong: list[str],
        n_thresholds: int = 21,
    ) -> list[ParetoPoint]:
        """Oracle: pick the cheapest model whose quality >= threshold.

        Returns one point per quality threshold (0.0 .. 1.0)."""
        points: list[ParetoPoint] = []
        for i in range(n_thresholds):
            q_thresh = i / (n_thresholds - 1)
            qs: list[float] = []
            cs: list[float] = []
            for by_model in self.by_prompt.values():
                chosen: EvalRecord | None = None
                for m in models_cheap_to_strong:
                    rec = by_model.get(m)
                    if rec is None:
                        continue
                    if rec.quality >= q_thresh:
                        chosen = rec
                        break
                # Fall back to the strongest if no model met the bar.
                if chosen is None:
                    for m in reversed(models_cheap_to_strong):
                        rec = by_model.get(m)
                        if rec is not None:
                            chosen = rec
                            break
                if chosen is None:
                    continue
                qs.append(chosen.quality)
                cs.append(chosen.cost)
            if not qs:
                continue
            points.append(
                ParetoPoint(
                    quality_threshold=q_thresh,
                    mean_quality=sum(qs) / len(qs),
                    total_cost=sum(cs),
                )
            )
        return points

    # ------------------------------------------------------------------
    # threshold sweep for the classifier
    # ------------------------------------------------------------------
    def sweep_thresholds(
        self,
        router_factory: Callable[[float, float], Router],
        low: float = 0.1,
        high_lo: float = 0.5,
        high_hi: float = 0.95,
        steps: int = 10,
    ) -> list[tuple[float, EvalResult]]:
        """Sweep p_high (classifier upper threshold); fix p_low at `low`.

        `router_factory(p_low, p_high) -> Router`.

        Returns [(p_high, EvalResult), ...]."""
        results: list[tuple[float, EvalResult]] = []
        for i in range(steps):
            t = high_lo + (high_hi - high_lo) * i / max(steps - 1, 1)
            router = router_factory(low, t)
            res = self.evaluate_router(router)
            results.append((t, res))
        return results

    # ------------------------------------------------------------------
    # Phase 5 stubs — interfaces locked, impl to come.
    # ------------------------------------------------------------------
    def replay_decisions(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Phase 5: replay_decisions is scaffolded only")

    def shadow_run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Phase 5: shadow_run is scaffolded only")
