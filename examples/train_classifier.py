"""Train a classifier on synthetic data.

For real production: replace `data/train.jsonl` with the Phase-0 labeled
set from EVAL.md and switch the embedder to the multilingual-e5-small
sentence-transformer (default in `load_default_embedder()`).
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_router.classifier.embedding import HashingEmbedder
from llm_router.classifier.trainer import ClassifierTrainer, TrainingConfig
from llm_router.observability.logger import configure_logging


def main() -> None:
    configure_logging(level="INFO", json=False)

    train_path = Path("data/train.jsonl")
    if not train_path.exists():
        # Lazy: regenerate synthetic data on the fly.
        from scripts.seed_data import main as seed_main

        seed_main()

    prompts: list[str] = []
    labels: list[int] = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            labels.append(int(r["label"]))

    # For the example we use HashingEmbedder so the demo runs without
    # downloading torch/transformers. In production swap this for
    # `load_default_embedder()` to use multilingual-e5-small.
    embedder = HashingEmbedder(dim=256)
    cfg = TrainingConfig(embedding_model="hashing:256")
    trainer = ClassifierTrainer(config=cfg, embedder=embedder)
    artifact_path = Path("artifacts/classifier.joblib")
    meta = trainer.fit_and_save(prompts, labels, artifact_path)
    print(f"\ntrained on {meta.n_samples} prompts. "
          f"cv accuracy={meta.cv_accuracy:.3f}, brier={meta.cv_brier:.3f}, "
          f"auroc={meta.cv_auroc}")
    print(f"artifact -> {artifact_path}")


if __name__ == "__main__":
    main()
