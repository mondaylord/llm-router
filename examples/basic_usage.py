"""Minimal end-to-end example.

Runs a few prompts through the router and prints the decisions. The
classifier is loaded only if a prior run of `train_classifier.py` left
an artifact at the expected path.

Run after `python examples/train_classifier.py`.
"""

from __future__ import annotations

from pathlib import Path

from llm_router import Router, RouterConfig, RoutingRequest
from llm_router.core.config import ClassifierConfig
from llm_router.observability.logger import configure_logging

ARTIFACT = Path("artifacts/classifier.joblib")


def main() -> None:
    configure_logging(level="INFO", json=False)

    config = RouterConfig.default()
    if ARTIFACT.exists():
        config.classifier = ClassifierConfig(
            enabled=True,
            artifact_path=str(ARTIFACT),
            embedding_model="hashing:256",  # matches train_classifier.py default
        )
    else:
        config.classifier.enabled = False
        print(f"(no classifier artifact at {ARTIFACT}; running with rules only)\n")

    router = Router.from_config(config)

    test_prompts = [
        "hi",
        "你好",
        "Is the sky blue?",
        "What is 2 + 2?",
        "Translate 'good morning' to Spanish.",
        "Walk me step by step through why this code has a race condition... (long)",
        "Review this contract clause for ambiguity and potential dispute risk.",
        "请逐步证明 1+2+...+n = n(n+1)/2 并说明每一步。",
    ]

    for i, p in enumerate(test_prompts):
        decision = router.route(
            RoutingRequest(prompt=p, session_id=f"sess-{i}", tenant_id=None)
        )
        print(f"[{decision.tier.value:>6}] [{decision.layer.value:>18}] "
              f"{decision.reason:<45} :: {p[:60]}")


if __name__ == "__main__":
    main()
