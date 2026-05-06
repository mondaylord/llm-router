"""Embedding backends.

Two concrete impls ship:

- `SentenceTransformerEmbedder` (production default): lazy-loads
  sentence-transformers + torch on first use. Requires the `[ml]` extra.
- `HashingEmbedder` (test fallback): pure-numpy feature hashing. Useful
  for unit tests and CI where torch is unwanted, and to keep the rule
  layer testable without ML deps.

Both implement the `Embedder` protocol. Swap freely via config.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), self.dim) float32 array, L2-normalized."""
        ...


class HashingEmbedder:
    """Deterministic, dependency-light embedder for tests and CI.

    Uses character n-gram hashing. Quality is poor compared to a real
    transformer encoder, but it produces stable vectors of fixed dim
    so the classifier head can be trained and tested end-to-end.
    """

    def __init__(self, dim: int = 256, ngram: tuple[int, int] = (1, 3)) -> None:
        self.dim = dim
        self.ngram_low, self.ngram_high = ngram
        self._token_re = re.compile(r"\w+", re.UNICODE)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = self._token_re.findall(t.lower())
            for tok in tokens:
                for n in range(self.ngram_low, self.ngram_high + 1):
                    if n == 1:
                        grams = [tok]
                    else:
                        grams = [tok[j : j + n] for j in range(len(tok) - n + 1)]
                    for g in grams:
                        h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
                        idx = h % self.dim
                        sign = 1.0 if (h >> 63) & 1 else -1.0
                        out[i, idx] += sign
            # length feature, helps classifier even if features are weak
            out[i, 0] += min(len(t) / 1000.0, 5.0)
            # L2 normalize
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:
    """Wraps `sentence-transformers`. Lazy import so the package can be
    installed without torch when only the rule layer is used."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        max_seq_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with `pip install 'llm-router[ml]'`."
            ) from e

        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        # Most modern encoders expose this directly.
        self.dim = self._model.get_sentence_embedding_dimension()
        # E5 family expects a "query: " prefix at inference time.
        self._needs_e5_prefix = "e5" in model_name.lower()

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._needs_e5_prefix:
            texts = [f"query: {t}" for t in texts]
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32, copy=False)


def load_default_embedder(model_name: str | None = None) -> Embedder:
    """Tries sentence-transformers first, falls back to HashingEmbedder.

    Useful in environments where the ML extra may or may not be present
    (e.g. unit-test image vs production image)."""
    try:
        return SentenceTransformerEmbedder(
            model_name=model_name or "intfloat/multilingual-e5-small"
        )
    except ImportError:
        return HashingEmbedder()
