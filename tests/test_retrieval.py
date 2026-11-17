"""
RAG retrieval tests — ingestion, retrieval, and citation validity.
"""

import tempfile

import pytest

from app.core.config import PROJECT_ROOT
from app.rag.embeddings import HashEmbedding
from app.rag.ingestion import ingest_regulations
from app.rag.retrieval import RAGRetriever


@pytest.fixture
def rag_setup():
    """Ingest regulations with hash embeddings and return a retriever."""
    tmpdir = tempfile.mkdtemp()
    embedder = HashEmbedding()

    stats = ingest_regulations(
        regulations_dir=PROJECT_ROOT / "data" / "regulations",
        persist_dir=tmpdir,
        embedding_provider=embedder,
    )

    retriever = RAGRetriever(
        persist_dir=tmpdir,
        embedding_provider=embedder,
    )
    return retriever, stats


class TestIngestion:
    def test_ingestion_processes_files(self, rag_setup):
        retriever, stats = rag_setup
        assert stats["files"] >= 1
        assert stats["chunks"] >= 1

    def test_ingestion_creates_collection(self, rag_setup):
        retriever, stats = rag_setup
        assert retriever.is_ready()


class TestRetrieval:
    def test_retrieve_returns_results(self, rag_setup):
        retriever, _ = rag_setup
        results = retriever.retrieve("allergen declaration requirements", top_k=3)
        assert isinstance(results, list)
        # With hash embeddings results may not be semantically meaningful
        # but the retrieval mechanism should work

    def test_retrieve_with_min_score(self, rag_setup):
        retriever, _ = rag_setup
        results = retriever.retrieve("labelling requirements", top_k=5, min_score=0.0)
        assert isinstance(results, list)

    def test_retrieve_returns_metadata(self, rag_setup):
        retriever, _ = rag_setup
        results = retriever.retrieve("food labelling", top_k=1, min_score=0.0)
        if results:
            r = results[0]
            assert r.chunk_id
            assert r.text
            assert isinstance(r.score, float)

    def test_search_returns_dicts(self, rag_setup):
        retriever, _ = rag_setup
        results = retriever.search("additives", top_k=3)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], dict)
            assert "chunk_id" in results[0]


class TestChunkLookup:
    def test_get_chunk_by_id_existing(self, rag_setup):
        retriever, _ = rag_setup
        # First retrieve to get a valid chunk_id
        results = retriever.retrieve("labelling", top_k=1, min_score=0.0)
        if results:
            chunk = retriever.get_chunk_by_id(results[0].chunk_id)
            assert chunk is not None
            assert chunk.text == results[0].text

    def test_get_chunk_by_id_nonexistent(self, rag_setup):
        retriever, _ = rag_setup
        chunk = retriever.get_chunk_by_id("nonexistent_chunk_id_12345")
        assert chunk is None

    def test_chunk_exists(self, rag_setup):
        retriever, _ = rag_setup
        results = retriever.retrieve("labelling", top_k=1, min_score=0.0)
        if results:
            assert retriever.chunk_exists(results[0].chunk_id) is True
        assert retriever.chunk_exists("fake_id") is False


class TestCitationValidity:
    def test_valid_citation(self, rag_setup):
        """chunk_id exists in index AND evidence_quote is literal substring of chunk text."""
        retriever, _ = rag_setup
        results = retriever.retrieve("labelling", top_k=1, min_score=0.0)
        if results:
            chunk = results[0]
            # Use a substring of the chunk text as evidence
            if len(chunk.text) > 20:
                quote = chunk.text[:20]
                assert retriever.is_valid_citation(chunk.chunk_id, quote) is True

    def test_invalid_citation_wrong_quote(self, rag_setup):
        """chunk_id exists but evidence_quote is NOT a substring."""
        retriever, _ = rag_setup
        results = retriever.retrieve("labelling", top_k=1, min_score=0.0)
        if results:
            chunk = results[0]
            assert retriever.is_valid_citation(
                chunk.chunk_id, "THIS STRING DOES NOT EXIST IN ANY CHUNK 12345"
            ) is False

    def test_invalid_citation_wrong_chunk_id(self, rag_setup):
        """chunk_id does NOT exist in index."""
        retriever, _ = rag_setup
        assert retriever.is_valid_citation("fake_chunk_id", "any text") is False

    def test_invalid_citation_empty_quote(self, rag_setup):
        """Empty evidence_quote should be invalid."""
        retriever, _ = rag_setup
        results = retriever.retrieve("labelling", top_k=1, min_score=0.0)
        if results:
            assert retriever.is_valid_citation(results[0].chunk_id, "") is False

    def test_no_collection_returns_false(self):
        """No indexed collection should return False for all citation checks."""
        tmpdir = tempfile.mkdtemp()
        retriever = RAGRetriever(
            persist_dir=tmpdir,
            embedding_provider=HashEmbedding(),
        )
        assert retriever.is_valid_citation("any_id", "any_quote") is False
