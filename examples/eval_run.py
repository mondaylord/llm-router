"""Run the offline eval harness on the synthetic eval set.

Compares:
  - always-weak baseline
  - always-strong baseline
  - the actual Stage 1+2 router
  - the ideal-router Pareto curve

This is the kind of report you should produce at every threshold change
or rule edit in production.
"""

from __future__ import annotations

from pathlib import Path

from llm_router import Router, RouterConfig
from llm_router.core.config import ClassifierConfig
from llm_router.eval.dataset import load_jsonl
from llm_router.eval.harness import EvalHarness
from llm_router.observability.logger import configure_logging


def main() -> None:
    configure_logging(level="WARNING", json=False)

    eval_path = Path("data/eval.jsonl")
    if not eval_path.exists():
        from scripts.seed_data import main as seed_main

        seed_main()

    records = load_jsonl(eval_path)
    harness = EvalHarness(records)

    q_w, c_w = harness.baseline_always("weak")
    q_s, c_s = harness.baseline_always("strong")
    print(f"baseline always_weak   : quality={q_w:.3f}  cost={c_w:.4f}")
    print(f"baseline always_strong : quality={q_s:.3f}  cost={c_s:.4f}")

    print("\nideal-router Pareto curve:")
    print(f"  {'q_thresh':>8}  {'mean_q':>8}  {'cost':>8}")
    for pt in harness.compute_pareto_curve(
        models_cheap_to_strong=["weak", "strong"], n_thresholds=11
    ):
        print(f"  {pt.quality_threshold:>8.2f}  {pt.mean_quality:>8.3f}  "
              f"{pt.total_cost:>8.4f}")

    artifact = Path("artifacts/classifier.joblib")
    if not artifact.exists():
        print("\n(no classifier artifact; run train_classifier.py first to "
              "evaluate the Stage-2 router)")
        return

    cfg = RouterConfig.default()
    cfg.classifier = ClassifierConfig(
        enabled=True,
        artifact_path=str(artifact),
        embedding_model="hashing:256",
        p_low=0.3,
        p_high=0.7,
    )
    router = Router.from_config(cfg)
    res = harness.evaluate_router(router)
    print(f"\nrouter (rules + classifier):")
    print(f"  n_prompts    = {res.n_prompts}")
    print(f"  mean_quality = {res.mean_quality:.3f}")
    print(f"  total_cost   = {res.total_cost:.4f}")
    print(f"  tier_share   = {res.tier_share}")
    print(f"  layer_share  = {res.layer_share}")

    print("\nthreshold sweep on p_high (p_low fixed at 0.3):")

    def factory(p_low: float, p_high: float) -> Router:
        c = RouterConfig.default()
        c.classifier = ClassifierConfig(
            enabled=True,
            artifact_path=str(artifact),
            embedding_model="hashing:256",
            p_low=p_low,
            p_high=p_high,
        )
        return Router.from_config(c)

    for p_high, r in harness.sweep_thresholds(
        factory, low=0.3, high_lo=0.4, high_hi=0.9, steps=6
    ):
        print(f"  p_high={p_high:.2f}  mean_q={r.mean_quality:.3f}  "
              f"cost={r.total_cost:.4f}  weak%={r.tier_share.get('weak', 0):.2f}")


if __name__ == "__main__":
    main()
