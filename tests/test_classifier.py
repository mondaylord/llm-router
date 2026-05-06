"""Test the embedding-and-classifier path with HashingEmbedder.

Avoids torch/transformers as a test dependency by using HashingEmbedder.
The synthetic separation between hard/easy templates is large enough
that even a hashing-based classifier reaches very high accuracy here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_router.classifier.embedding import HashingEmbedder
from llm_router.classifier.predictor import ClassifierPredictor
from llm_router.classifier.trainer import ClassifierTrainer, TrainingConfig

EASY = [
    "hi", "hello", "thanks", "ok", "你好", "谢谢",
    "Is the sky blue?", "What is 2 + 2?", "Translate good morning to Spanish",
    "Respond with a single word: capital of France?",
    "What day comes after Monday?",
    "Convert 5 km to miles.",
] * 10

HARD = [
    "Walk through step by step why this code has a race condition: ... " + ("def f(): pass\n" * 20),
    "Prove the sum of first n odd numbers is n^2 by induction. Show every step.",
    "Derive the closed form for ridge regression starting from the loss function and explain each step.",
    "Review this contract clause for ambiguity and dispute risk in an enterprise SaaS context.",
    "Given this clinical history, list differentials and explain reasoning for each candidate.",
    "请逐步证明 1+2+...+n = n(n+1)/2 并说明数学归纳法每一步。",
    "请权衡 Lagrangian 与 Eulerian 流体仿真在游戏引擎中的使用，给出至少三个考量。",
] * 18  # ~126 examples to get above min sklearn cv quorum


def test_train_predict_roundtrip(tmp_path: Path):
    prompts = EASY + HARD
    labels = [0] * len(EASY) + [1] * len(HARD)

    cfg = TrainingConfig(embedding_model="hashing:256", n_splits=3)
    trainer = ClassifierTrainer(config=cfg, embedder=HashingEmbedder(dim=256))
    artifact = tmp_path / "clf.joblib"
    meta = trainer.fit_and_save(prompts, labels, artifact)

    assert meta.n_samples == len(prompts)
    assert meta.cv_accuracy >= 0.85
    assert artifact.exists()

    # Load + predict
    pred = ClassifierPredictor.load(artifact)
    p_easy = pred.predict_proba_strong("hi how are you")
    p_hard = pred.predict_proba_strong(
        "请逐步证明 1+2+...+n = n(n+1)/2 并说明数学归纳法每一步。"
    )
    # The hashing embedder is weak on linguistically novel inputs but
    # this canned hard prompt is in-distribution; assert relative order.
    assert p_hard > p_easy


def test_predict_returns_probability_in_unit_interval():
    pred = ClassifierPredictor(
        clf=_DummyClf(),
        embedder=HashingEmbedder(dim=8),
    )
    p = pred.predict_proba_strong("anything")
    assert 0.0 <= p <= 1.0


class _DummyClf:
    """sklearn-shaped dummy estimator returning constant probabilities."""

    def predict_proba(self, X):
        import numpy as np

        n = X.shape[0]
        return np.tile([0.4, 0.6], (n, 1))


@pytest.mark.parametrize("text", ["", "x", "x" * 10000])
def test_hashing_embedder_handles_edge_cases(text: str):
    e = HashingEmbedder(dim=64)
    out = e.encode([text])
    assert out.shape == (1, 64)
