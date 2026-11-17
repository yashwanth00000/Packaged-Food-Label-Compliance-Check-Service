"""
Embedding provider abstraction.
Supports sentence-transformers for production and a simple hash-based
fallback for testing without heavy ML dependencies.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return vectors."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embeddings."""
        ...


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Production embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("embedding_model_loaded", model=model_name, dimension=self._dim)
        except Exception as e:
            raise RuntimeError(
                f"sentence-transformers failed to load ({e})."
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode(query, show_progress_bar=False).tolist()

    @property
    def dimension(self) -> int:
        return self._dim


class HashEmbedding(EmbeddingProvider):
    """
    Deterministic hash-based 'embedding' for testing.
    NOT semantically meaningful — only for unit/integration tests.
    """

    _DIM = 384  # Match all-MiniLM-L6-v2

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha384(text.lower().encode()).digest()
        # Unpack digest as 48 x float32 ≈ 384 dims via cycling
        vec: list[float] = []
        for i in range(self._DIM):
            byte_idx = i % len(digest)
            vec.append((digest[byte_idx] - 128) / 128.0)
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    @property
    def dimension(self) -> int:
        return self._DIM


def get_embedding_provider(model_name: str = "all-MiniLM-L6-v2", use_mock: bool = False) -> EmbeddingProvider:
    """Factory: return the appropriate embedding provider."""
    if use_mock:
        logger.info("using_hash_embedding", reason="mock mode")
        return HashEmbedding()
    try:
        return SentenceTransformerEmbedding(model_name)
    except Exception as e:
        logger.warning("sentence_transformers_not_available", error=str(e), fallback="hash_embedding")
        return HashEmbedding()
