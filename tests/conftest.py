"""
Pytest configuration and shared fixtures.
"""

import tempfile
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import PROJECT_ROOT, get_settings
from app.models.database import Base, get_db_session
from app.models.schemas import SAMPLE_LABELS
from app.services.llm_provider import LLMService, MockLLM, reset_llm_service

# ---------------------------------------------------------------------------
# Ensure mock LLM during tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def force_mock_llm(monkeypatch):
    """Force mock LLM provider and mock embeddings for all tests, and reset rate limiter."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MOCK_FALLBACK", "false")
    monkeypatch.setenv("REGULATORY_REQUIRE_AUTHORITATIVE_CORPUS", "false")
    monkeypatch.setenv("VISION_PROVIDER", "mock")
    monkeypatch.setenv("EXPERT_API_KEY", "")
    monkeypatch.setenv("EXPERT_API_KEYS_JSON", '{"expert-key-dr-sharma-2024":{"expert_id":"TEST-01","name":"Dr. Anita Sharma","role":"Senior Food Safety Specialist"},"expert-key-rkumar-2024":{"expert_id":"TEST-02","name":"Rajesh Kumar","role":"FSSAI Regulatory Compliance Officer"},"expert-key-fssai-2024":{"expert_id":"TEST-03","name":"FSSAI Authorized Technical Officer","role":"Technical Evaluation Officer"}}')
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    get_settings.cache_clear()
    reset_llm_service()
    from app.core.rate_limit import _rate_limit_store
    _rate_limit_store.clear()
    yield
    reset_llm_service()
    get_settings.cache_clear()
    _rate_limit_store.clear()


# ---------------------------------------------------------------------------
# Test database (in-memory SQLite)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_db():
    """Create a fresh in-memory async SQLite database for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        from sqlalchemy import text
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_log_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is immutable: UPDATE not allowed');
            END;
        """))
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_log_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(ABORT, 'audit_log is immutable: DELETE not allowed');
            END;
        """))

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def get_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    yield get_test_session

    await engine.dispose()


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture
def app(test_db):
    """Create a test FastAPI application with overridden dependencies."""
    from app.api.routes import get_pipeline, get_retriever
    from app.main import create_app
    from app.orchestrator.pipeline import CompliancePipeline
    from app.rag.retrieval import RAGRetriever

    test_app = create_app()
    test_app.dependency_overrides[get_db_session] = test_db
    mock_retriever = RAGRetriever(use_mock=True)
    mock_pipeline = CompliancePipeline(retriever=mock_retriever)
    test_app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    test_app.dependency_overrides[get_retriever] = lambda: mock_retriever
    return test_app


@pytest.fixture
def client(app):
    """Synchronous test client for FastAPI."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Sample submissions
# ---------------------------------------------------------------------------

@pytest.fixture
def compliant_label():
    """A fully compliant label that should pass checks."""
    return SAMPLE_LABELS["compliant_biscuit"]


@pytest.fixture
def missing_declarations_label():
    """A label missing mandatory declarations."""
    return SAMPLE_LABELS["missing_declarations"]


@pytest.fixture
def prohibited_claims_label():
    """A label with prohibited claims."""
    return SAMPLE_LABELS["prohibited_claims"]


# ---------------------------------------------------------------------------
# Mock retriever
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retriever():
    """A RAGRetriever that works without a real ChromaDB index."""
    from app.rag.embeddings import HashEmbedding
    from app.rag.retrieval import RAGRetriever

    tmpdir = tempfile.mkdtemp()
    retriever = RAGRetriever(
        persist_dir=tmpdir,
        embedding_provider=HashEmbedding(),
        use_mock=True,
    )
    return retriever


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(mock_retriever):
    """A CompliancePipeline with mock LLM and mock retriever."""
    from app.orchestrator.pipeline import CompliancePipeline
    llm = LLMService(provider=MockLLM())
    return CompliancePipeline(llm_service=llm, retriever=mock_retriever)


# ---------------------------------------------------------------------------
# Ingested retriever (for retrieval tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def ingested_retriever():
    """A RAGRetriever with regulations actually ingested."""
    from app.rag.embeddings import HashEmbedding
    from app.rag.ingestion import ingest_regulations
    from app.rag.retrieval import RAGRetriever

    tmpdir = tempfile.mkdtemp()
    embedder = HashEmbedding()

    ingest_regulations(
        regulations_dir=PROJECT_ROOT / "data" / "regulations",
        persist_dir=tmpdir,
        embedding_provider=embedder,
    )

    return RAGRetriever(
        persist_dir=tmpdir,
        embedding_provider=embedder,
    )
