"""Inference-side classifier wrapper.

Loads a joblib bundle written by `ClassifierTrainer.save()` and exposes
a single hot-path method `predict_proba_strong(prompt) -> float`.

The embedder is constructed at load time and reused for every call.
sentence-transformers handles thread-safe inference internally; for
extreme QPS, run multiple worker processes rather than threads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from llm_router.classifier.embedding import Embedder, HashingEmbedder, load_default_embedder
from llm_router.observability.logger import get_logger

log = get_logger(__name__)


class ClassifierPredictor:
    def __init__(
        self,
        clf,
        embedder: Embedder,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._clf = clf
        self._embedder = embedder
        self.metadata = metadata or {}

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        embedding_model: str | None = None,
    ) -> ClassifierPredictor:
        bundle = joblib.load(artifact_path)
        if not isinstance(bundle, dict) or "classifier" not in bundle:
            raise ValueError(f"Unexpected artifact format at {artifact_path}")

        meta = bundle.get("metadata", {})
        recorded_model = meta.get("embedding_model")
        chosen_model = embedding_model or recorded_model

        # If the saved metadata says the model used a HashingEmbedder
        # (i.e. tests / no-ML environment), prefer HashingEmbedder at load
        # time rather than auto-installing torch silently.
        if recorded_model and recorded_model.startswith("hashing:"):
            dim = meta.get("embedding_dim", 256)
            embedder: Embedder = HashingEmbedder(dim=dim)
        else:
            embedder = load_default_embedder(chosen_model)
            recorded_dim = meta.get("embedding_dim")
            if recorded_dim and embedder.dim != recorded_dim:
                log.warning(
                    "embedding_dim_mismatch",
                    expected=recorded_dim,
                    actual=embedder.dim,
                    model=chosen_model,
                )
        return cls(clf=bundle["classifier"], embedder=embedder, metadata=meta)

    def predict_proba_strong(self, prompt: str) -> float:
        """Return calibrated `p(needs_strong)` for a single prompt."""
        x = self._embedder.encode([prompt])
        proba = self._clf.predict_proba(x)[0]
        # Class order: sklearn sorts classes ascending → index 1 is class 1 = strong.
        return float(proba[1])

    def predict_batch(self, prompts: list[str]) -> np.ndarray:
        """Return calibrated `p(needs_strong)` for a batch.

        Useful for offline eval and shadow-traffic replay; not on the
        hot path of single-request routing."""
        x = self._embedder.encode(prompts)
        return self._clf.predict_proba(x)[:, 1]
