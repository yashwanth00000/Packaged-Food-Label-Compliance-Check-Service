"""
SQLAlchemy ORM models and async session management for SQLite.
Tables: submissions, compliance_results, expert_decisions, audit_log.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class SubmissionRecord(Base):
    """A label submission."""
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    check_id = Column(String(32), unique=True, nullable=False, index=True)
    product_name = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    submission_data = Column(Text, nullable=False)  # JSON blob
    submitted_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ComplianceResultRecord(Base):
    """Compliance check result — the full report."""
    __tablename__ = "compliance_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    check_id = Column(String(32), unique=True, nullable=False, index=True)
    recommendation = Column(String(30), nullable=True)
    report_data = Column(Text, nullable=False)  # Full ComplianceReport JSON
    overall_confidence = Column(Float, default=0.0)
    findings_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float, default=0.0)
    has_errors = Column(Boolean, default=False)
    completed_at = Column(DateTime, default=datetime.utcnow)


class ExpertDecisionRecord(Base):
    """Expert decision on a compliance check."""
    __tablename__ = "expert_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(32), unique=True, nullable=False, index=True)
    check_id = Column(String(32), nullable=False, index=True)
    action = Column(String(20), nullable=False)
    original_recommendation = Column(String(30), nullable=False)
    reason = Column(Text, default="")
    expert_name = Column(String(200), default="")
    decided_at = Column(DateTime, default=datetime.utcnow)


class AuditLogRecord(Base):
    """Immutable audit trail."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    check_id = Column(String(32), nullable=True, index=True)
    action = Column(String(100), nullable=False)
    actor = Column(String(200), default="system")
    details = Column(Text, default="")
    request_id = Column(String(32), nullable=True)


class ExtractionRecord(Base):
    """Label photo extraction record with user confirmation status."""
    __tablename__ = "extractions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    extraction_id = Column(String(32), unique=True, nullable=False, index=True)
    image_paths = Column(Text, default="[]")  # JSON list of stored relative image paths
    extracted_data = Column(Text, nullable=False)  # Raw ExtractedLabelData JSON
    user_corrected_data = Column(Text, nullable=True)  # Confirmed fields JSON
    user_confirmed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


from sqlalchemy import event, text


@event.listens_for(AuditLogRecord, "before_update")
def _prevent_audit_log_update(mapper, connection, target):
    raise RuntimeError("AuditLogRecord is immutable: UPDATE is strictly prohibited.")


@event.listens_for(AuditLogRecord, "before_delete")
def _prevent_audit_log_delete(mapper, connection, target):
    raise RuntimeError("AuditLogRecord is immutable: DELETE is strictly prohibited.")


# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

_async_engine = None
_async_session_factory = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
        )
    return _async_engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides an async DB session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables and triggers if they don't exist."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # SQLite immutability triggers for audit_log
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


async def close_db() -> None:
    """Close the database engine."""
    global _async_engine, _async_session_factory
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
