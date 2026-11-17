"""
RAG retrieval: query ChromaDB for top-k regulation chunks with scores.
"""

from __future__ import annotations

from typing import Any

import chromadb

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import RAG_LATENCY, RAG_QUERIES
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.ingestion import COLLECTION_NAME, get_chroma_client

logger = get_logger(__name__)


class RetrievalResult:
    """A single retrieved chunk with metadata and score."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        score: float,
        regulation_name: str = "",
        clause: str = "",
        source_file: str = "",
        version: str = "",
        is_sample: bool = False,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.regulation_name = regulation_name
        self.clause = clause
        self.source_file = source_file
        self.version = version
        self.is_sample = is_sample

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "regulation_name": self.regulation_name,
            "clause": self.clause,
            "source_file": self.source_file,
            "version": self.version,
            "is_sample": self.is_sample,
        }


class RAGRetriever:
    """Retrieves regulation chunks from ChromaDB."""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        use_mock: bool = False,
    ):
        self._persist_dir = persist_dir or get_settings().chroma_persist_dir
        if embedding_provider is None:
            settings = get_settings()
            use_mock_flag = use_mock or (getattr(settings, "embedding_provider", "sentence-transformers").lower() in ("mock", "hash"))
            self._embedder = get_embedding_provider(
                model_name=settings.embedding_model,
                use_mock=use_mock_flag,
            )
        else:
            self._embedder = embedding_provider
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    def _ensure_collection(self):
        """Lazily initialise the ChromaDB collection."""
        if self._collection is None:
            self._client = get_chroma_client(self._persist_dir)
            try:
                self._collection = self._client.get_collection(COLLECTION_NAME)
            except Exception:
                logger.warning("collection_not_found", collection=COLLECTION_NAME)
                self._collection = None

    def is_ready(self) -> bool:
        """Check if the vector store has an indexed collection."""
        self._ensure_collection()
        return self._collection is not None

    def get_chunk_by_id(self, chunk_id: str) -> RetrievalResult | None:
        """Look up a specific chunk by its ID."""
        self._ensure_collection()
        if self._collection is None:
            return None
        try:
            result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if result and result["ids"] and result["ids"][0]:
                meta = result["metadatas"][0] if result["metadatas"] else {}
                return RetrievalResult(
                    chunk_id=chunk_id,
                    text=result["documents"][0] if result["documents"] else "",
                    score=1.0,
                    regulation_name=meta.get("regulation_name", ""),
                    clause=meta.get("clause", ""),
                    source_file=meta.get("source_file", ""),
                    version=meta.get("version", ""),
                )
        except Exception as e:
            logger.warning("chunk_lookup_failed", chunk_id=chunk_id, error=str(e))
        return None

    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk_id exists in the index."""
        return self.get_chunk_by_id(chunk_id) is not None

    def is_valid_citation(self, chunk_id: str, evidence_quote: str) -> bool:
        """
        Citation validity: chunk_id exists in the index AND
        evidence_quote is a literal substring of that chunk's text.
        """
        chunk = self.get_chunk_by_id(chunk_id)
        if chunk is None:
            return False
        if not evidence_quote:
            return False
        return evidence_quote.strip() in chunk.text

    @RAG_LATENCY.time()
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k regulation chunks for a query.
        Returns results sorted by relevance score (highest first).
        """
        self._ensure_collection()
        if self._collection is None:
            RAG_QUERIES.labels(status="error").inc()
            logger.warning("retrieval_no_collection")
            return []

        settings = get_settings()
        if top_k is None:
            top_k = settings.rag_top_k
        if min_score is None:
            min_score = settings.rag_min_score

        try:
            query_embedding = self._embedder.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            RAG_QUERIES.labels(status="error").inc()
            logger.error("retrieval_failed", error=str(e))
            return []

        RAG_QUERIES.labels(status="success").inc()

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        retrieval_results: list[RetrievalResult] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results["distances"][0][i] if results["distances"] else 0.0
            score = 1.0 - distance  # cosine similarity = 1 - cosine distance

            if score < min_score:
                continue

            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            text = results["documents"][0][i] if results["documents"] else ""

            is_sample_doc = bool(meta.get("is_sample", False)) or (meta.get("source_file", "").endswith(".txt"))
            retrieval_results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=round(score, 4),
                regulation_name=meta.get("regulation_name", ""),
                clause=meta.get("clause", ""),
                source_file=meta.get("source_file", ""),
                version=meta.get("version", ""),
                is_sample=is_sample_doc,
            ))

        # Down-rank sample documents when real PDFs exist in the retrieved results
        has_real_docs = any(not r.is_sample for r in retrieval_results)
        if has_real_docs:
            for r in retrieval_results:
                if r.is_sample:
                    # Penalize sample document score by 0.15 so real official PDFs rank above samples
                    r.score = round(max(0.0, r.score - 0.15), 4)

        # Sort by score descending
        retrieval_results.sort(key=lambda r: r.score, reverse=True)
        return retrieval_results

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search endpoint helper — returns dicts."""
        results = self.retrieve(query, top_k=top_k, min_score=0.0)
        return [r.to_dict() for r in results]
