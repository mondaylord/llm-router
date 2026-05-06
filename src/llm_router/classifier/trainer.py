"""Train a calibrated probabilistic classifier head on top of embeddings.

Classifier head choice: `LogisticRegression` wrapped in
`CalibratedClassifierCV` with isotonic calibration.

Why:
- Calibrated probabilities are a hard requirement for principled
  threshold tuning. Raw LR margins are NOT calibrated when class
  imbalance is non-trivial, and they are uninterpretable as `p(strong)`.
- Isotonic > Platt when you have ≥1k samples per class (typical here).
- Linear model is enough on top of strong embeddings; XGBoost adds
  complexity without measurable gain at this data scale.

Output: a joblib artifact bundling:
  - the trained `ClassifierMixin` (sklearn estimator)
  - the embedding model name (for runtime sanity checking)
  - the embedding dim
  - training metadata (date, n_samples, cv accuracy)

The artifact is what `ClassifierPredictor.load(...)` consumes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from llm_router.classifier.embedding import Embedder, load_default_embedder
from llm_router.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    n_splits: int = 5
    """k for StratifiedKFold inside CalibratedClassifierCV."""

    lr_C: float = 1.0
    """Inverse regularization strength for LR."""

    lr_max_iter: int = 1000
    random_state: int = 42

    embedding_model: str = "intfloat/multilingual-e5-small"
    """Recorded into artifact metadata. NOTE: must match the embedder
    actually passed to fit()."""


@dataclass
class TrainingMetadata:
    embedding_model: str
    embedding_dim: int
    n_samples: int
    n_pos: int
    n_neg: int
    cv_accuracy: float
    cv_log_loss: float
    cv_brier: float
    cv_auroc: float | None
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClassifierTrainer:
    """Fit a calibrated head on (prompts, labels)."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.embedder = embedder or load_default_embedder(self.config.embedding_model)

    def fit(
        self,
        prompts: list[str],
        labels: list[int],
    ) -> tuple[CalibratedClassifierCV, TrainingMetadata]:
        """`labels`: 1 if the prompt needs the strong tier, 0 otherwise."""
        if len(prompts) != len(labels):
            raise ValueError("prompts and labels must be same length")
        if not prompts:
            raise ValueError("empty training set")

        y = np.asarray(labels, dtype=np.int64)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("labels must be binary {0, 1}")

        log.info("embedding_train_set", n=len(prompts))
        X = self.embedder.encode(prompts)

        base = LogisticRegression(
            C=self.config.lr_C,
            max_iter=self.config.lr_max_iter,
            random_state=self.config.random_state,
            class_weight="balanced",
        )
        cv = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        clf = CalibratedClassifierCV(
            estimator=base,
            method="isotonic",
            cv=cv,
        )
        clf.fit(X, y)

        # Out-of-fold-ish quick eval: predict on training set is biased,
        # but for a quick sanity number, run a fresh CV manually.
        from sklearn.model_selection import cross_val_predict

        proba = cross_val_predict(
            CalibratedClassifierCV(
                estimator=LogisticRegression(
                    C=self.config.lr_C,
                    max_iter=self.config.lr_max_iter,
                    random_state=self.config.random_state,
                    class_weight="balanced",
                ),
                method="isotonic",
                cv=cv,
            ),
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        preds = (proba >= 0.5).astype(int)
        meta = TrainingMetadata(
            embedding_model=self.config.embedding_model,
            embedding_dim=int(X.shape[1]),
            n_samples=int(len(prompts)),
            n_pos=int(y.sum()),
            n_neg=int(len(y) - y.sum()),
            cv_accuracy=float(accuracy_score(y, preds)),
            cv_log_loss=float(log_loss(y, proba, labels=[0, 1])),
            cv_brier=float(brier_score_loss(y, proba)),
            cv_auroc=float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None,
        )
        log.info("classifier_trained", **meta.to_dict())
        return clf, meta

    def save(
        self,
        clf: CalibratedClassifierCV,
        meta: TrainingMetadata,
        artifact_path: str | Path,
    ) -> None:
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "version": 1,
            "classifier": clf,
            "metadata": meta.to_dict(),
        }
        joblib.dump(bundle, artifact_path)
        # Also write a sidecar JSON for human inspection.
        sidecar = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("artifact_saved", path=str(artifact_path))

    def fit_and_save(
        self,
        prompts: list[str],
        labels: list[int],
        artifact_path: str | Path,
    ) -> TrainingMetadata:
        clf, meta = self.fit(prompts, labels)
        self.save(clf, meta, artifact_path)
        return meta
