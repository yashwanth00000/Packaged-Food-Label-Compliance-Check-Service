"""
API routes for the FSSAI Label Compliance Check Service.
All endpoints are prefixed with /api/v1/.
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.extraction import LabelExtractionAgent
from app.core.auth import verify_expert_api_key
from app.core.config import UPLOAD_DIR, get_disclaimer, get_settings
from app.core.logging import get_logger
from app.models.database import (
    AuditLogRecord,
    ComplianceResultRecord,
    ExpertDecisionRecord,
    ExtractionRecord,
    SubmissionRecord,
    get_db_session,
)
from app.models.enums import CheckStatus, ExpertAction, Recommendation
from app.models.extraction import ExtractionResult
from app.models.schemas import (
    SAMPLE_LABELS,
    CheckListItem,
    CheckListResponse,
    CheckResponse,
    ComplianceReport,
    ExpertDecisionRequest,
    ExpertDecisionResponse,
    LabelSubmission,
)
from app.orchestrator.pipeline import CompliancePipeline
from app.rag.retrieval import RAGRetriever

logger = get_logger(__name__)

api_router = APIRouter(prefix="/api/v1", tags=["compliance"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_pipeline() -> CompliancePipeline:
    return CompliancePipeline()


def get_retriever() -> RAGRetriever:
    return RAGRetriever()


# ---------------------------------------------------------------------------
# POST /api/v1/extract — Stage 0 Label Extraction from Photos
# ---------------------------------------------------------------------------

@api_router.post("/extract", response_model=ExtractionResult)
async def extract_label_from_photos(
    images: list[UploadFile] = File(default=[]),
    files: list[UploadFile] = File(default=[]),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Stage 0: Extract structured label data from 1-4 uploaded label images.
    Validates MIME by content, strips EXIF metadata, invokes vision LLM,
    and returns extracted fields with confidences.
    Does NOT run the compliance check.
    """
    settings = get_settings()
    all_uploads = [f for f in (list(images) + list(files)) if f.filename]
    if not all_uploads:
        return JSONResponse(status_code=400, content={"error": "At least one label photo is required."})
    if len(all_uploads) > settings.upload_max_images:
        return JSONResponse(
            status_code=400,
            content={"error": f"Too many images. Maximum allowed is {settings.upload_max_images}."},
        )

    raw_bytes_list: list[bytes] = []
    filenames_list: list[str] = []
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    for f in all_uploads:
        content = await f.read(max_bytes + 1)
        if len(content) > max_bytes:
            return JSONResponse(status_code=413, content={"error": f"Each image must be {settings.upload_max_size_mb} MB or smaller."})
        raw_bytes_list.append(content)
        filenames_list.append(f.filename or "")

    try:
        agent = LabelExtractionAgent()
    except (RuntimeError, ValueError, ImportError) as exc:
        return JSONResponse(status_code=503, content={"error": f"Vision provider is not configured: {exc}"})
    result = await agent.extract_from_images(raw_bytes_list, filenames=filenames_list)
    if result.fallback_to_manual:
        return result

    # Persist only sanitized image bytes; the original uploads are discarded.
    from app.agents.image_processing import sanitize_and_prepare_image
    saved_rel_paths: list[str] = []
    upload_folder = UPLOAD_DIR / result.extraction_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    for idx, raw_bytes in enumerate(raw_bytes_list):
        processed = sanitize_and_prepare_image(raw_bytes, filename=f"label_photo_{idx+1}",
                                                max_size_bytes=settings.upload_max_size_mb * 1024 * 1024)
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[processed.mime_type]
        filename = f"label_photo_{idx+1}{suffix}"
        (upload_folder / filename).write_bytes(processed.image_bytes)
        saved_rel_paths.append(f"/label-image/{result.extraction_id}/{filename}")

    # Persist extraction record
    rec = ExtractionRecord(
        extraction_id=result.extraction_id,
        image_paths=json.dumps(saved_rel_paths),
        extracted_data=result.fields.model_dump_json(),
        user_confirmed=False,
    )
    db.add(rec)
    await db.flush()

    logger.info(
        "extraction_completed",
        extraction_id=result.extraction_id,
        image_count=len(files),
        warnings_count=len(result.image_quality_warnings),
        fallback=result.fallback_to_manual,
    )
    return result


# ---------------------------------------------------------------------------
# POST /api/v1/check — run compliance check
# ---------------------------------------------------------------------------

@api_router.post("/check", response_model=CheckResponse)
async def run_check(
    submission: LabelSubmission,
    pipeline: CompliancePipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db_session),
):
    """Run the full compliance pipeline on a label submission."""
    report: ComplianceReport = pipeline.run(submission)

    # Server-side enforcement: CLEAR_FOR_PRINT must be rejected if user_confirmed is false
    if submission.user_confirmed is False and report.recommendation == Recommendation.CLEAR_FOR_PRINT:
        report.recommendation = Recommendation.REVISE_AND_RECHECK
        report.reasons.insert(
            0,
            "Server-side enforcement: CLEAR_FOR_PRINT rejected because user_confirmed is false. "
            "Operator must visually verify all extracted fields before clearance can be recommended.",
        )

    # Link extraction record if check originated from photo extraction
    if submission.extraction_id:
        ext_query = await db.execute(
            select(ExtractionRecord).where(ExtractionRecord.extraction_id == submission.extraction_id)
        )
        ext_rec = ext_query.scalar_one_or_none()
        if ext_rec:
            ext_rec.user_corrected_data = submission.model_dump_json()
            ext_rec.user_confirmed = submission.user_confirmed or False
            db.add(ext_rec)

    # Persist submission
    sub_record = SubmissionRecord(
        check_id=report.check_id,
        product_name=submission.product_name,
        status=CheckStatus.COMPLETED.value,
        submission_data=submission.model_dump_json(),
    )
    db.add(sub_record)

    # Persist result
    result_record = ComplianceResultRecord(
        check_id=report.check_id,
        recommendation=report.recommendation.value,
        report_data=report.model_dump_json(),
        overall_confidence=report.overall_confidence,
        findings_count=len(report.findings),
        processing_time_seconds=report.processing_time_seconds,
        has_errors=report.has_errors,
    )
    db.add(result_record)

    # Audit log
    audit = AuditLogRecord(
        check_id=report.check_id,
        action="compliance_check_completed",
        actor="system",
        details=json.dumps({
            "recommendation": report.recommendation.value,
            "findings_count": len(report.findings),
            "processing_time": report.processing_time_seconds,
            "user_confirmed": submission.user_confirmed,
        }),
    )
    db.add(audit)
    await db.flush()

    return CheckResponse(
        check_id=report.check_id,
        status=CheckStatus.COMPLETED,
        recommendation=report.recommendation,
        report=report,
        disclaimer=get_disclaimer(),
        message="Compliance check completed. This is a recommendation only — a human expert must make the final decision.",
        is_sample=report.is_sample,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/check/{check_id} — fetch a specific report
# ---------------------------------------------------------------------------

@api_router.get("/check/{check_id}", response_model=CheckResponse)
async def get_check(
    check_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Retrieve a previously completed compliance check report."""
    result = await db.execute(
        select(ComplianceResultRecord).where(ComplianceResultRecord.check_id == check_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        return JSONResponse(status_code=404, content={"error": "Check not found", "check_id": check_id})

    report = ComplianceReport.model_validate_json(record.report_data)

    # Check for expert decision
    decision_result = await db.execute(
        select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == check_id)
    )
    decision_record = decision_result.scalar_one_or_none()
    expert_decision = None
    if decision_record:
        expert_decision = ExpertDecisionResponse(
            decision_id=decision_record.decision_id,
            check_id=decision_record.check_id,
            action=ExpertAction(decision_record.action),
            original_recommendation=Recommendation(decision_record.original_recommendation),
            reason=decision_record.reason,
            expert_name=decision_record.expert_name,
            decided_at=decision_record.decided_at,
        )

    return CheckResponse(
        check_id=check_id,
        status=CheckStatus.COMPLETED,
        recommendation=report.recommendation,
        report=report,
        expert_decision=expert_decision,
        disclaimer=get_disclaimer(),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/checks — list recent checks
# ---------------------------------------------------------------------------

@api_router.get("/checks", response_model=CheckListResponse)
async def list_checks(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    """List recent compliance checks with pagination."""
    offset = (page - 1) * per_page

    # Count total
    count_result = await db.execute(select(SubmissionRecord.id))
    total = len(count_result.all())

    # Fetch page
    result = await db.execute(
        select(SubmissionRecord)
        .order_by(desc(SubmissionRecord.submitted_at))
        .offset(offset)
        .limit(per_page)
    )
    records = result.scalars().all()

    items = []
    for rec in records:
        # Check for expert decision
        dec_result = await db.execute(
            select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == rec.check_id)
        )
        has_decision = dec_result.scalar_one_or_none() is not None

        items.append(CheckListItem(
            check_id=rec.check_id,
            product_name=rec.product_name,
            status=CheckStatus(rec.status),
            recommendation=None,
            submitted_at=rec.submitted_at,
            has_expert_decision=has_decision,
        ))

    # Fill in recommendations from results table
    for item in items:
        res = await db.execute(
            select(ComplianceResultRecord.recommendation)
            .where(ComplianceResultRecord.check_id == item.check_id)
        )
        rec_val = res.scalar_one_or_none()
        if rec_val:
            item.recommendation = Recommendation(rec_val)

    return CheckListResponse(items=items, total=total, page=page, per_page=per_page)


# ---------------------------------------------------------------------------
# POST /api/v1/check/{check_id}/decision — expert override
# ---------------------------------------------------------------------------

@api_router.post("/check/{check_id}/decision", response_model=ExpertDecisionResponse)
async def submit_expert_decision(
    check_id: str,
    decision: ExpertDecisionRequest,
    x_expert_key: str | None = Header(None, alias="X-Expert-Key"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Submit an expert decision (Accept or Override with reason) for a compliance check.
    Requires per-expert API key matching hashed credentials. X-User-Role bypass is removed.
    """
    is_authorized, expert_identity = verify_expert_api_key(x_expert_key)
    if not is_authorized or not expert_identity:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized: Valid X-Expert-Key header is required to record expert decisions. X-User-Role header alone is not accepted."},
        )

    expert_identity_str = f"{expert_identity.name} ({expert_identity.role})"

    # Verify the check exists
    result = await db.execute(
        select(ComplianceResultRecord).where(ComplianceResultRecord.check_id == check_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        return JSONResponse(status_code=404, content={"error": "Check not found", "check_id": check_id})

    # Check for existing decision
    existing = await db.execute(
        select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == check_id)
    )
    if existing.scalar_one_or_none():
        return JSONResponse(
            status_code=409,
            content={"error": "Expert decision already exists for this check", "check_id": check_id},
        )

    decision_id = uuid4().hex[:12]
    original_rec = Recommendation(record.recommendation)
    now = datetime.utcnow()

    dec_record = ExpertDecisionRecord(
        decision_id=decision_id,
        check_id=check_id,
        action=decision.action.value,
        original_recommendation=original_rec.value,
        reason=decision.reason,
        expert_name=expert_identity_str,
        decided_at=now,
    )
    db.add(dec_record)

    # Audit log with verified expert identity
    audit = AuditLogRecord(
        check_id=check_id,
        action=f"expert_decision_{decision.action.value}",
        actor=expert_identity_str,
        details=json.dumps({
            "original_recommendation": original_rec.value,
            "action": decision.action.value,
            "reason": decision.reason,
            "expert_id": expert_identity.expert_id,
        }),
    )
    db.add(audit)
    await db.flush()

    logger.info(
        "expert_decision_recorded",
        check_id=check_id,
        action=decision.action.value,
        expert=expert_identity_str,
    )

    return ExpertDecisionResponse(
        decision_id=decision_id,
        check_id=check_id,
        action=decision.action,
        original_recommendation=original_rec,
        reason=decision.reason,
        expert_name=expert_identity_str,
        decided_at=now,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/regulations/search — search regulation chunks
# ---------------------------------------------------------------------------

@api_router.get("/regulations/search")
async def search_regulations(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(5, ge=1, le=20),
    retriever: RAGRetriever = Depends(get_retriever),
):
    """Search the indexed FSSAI regulation chunks."""
    results = retriever.search(q, top_k=top_k)
    return {
        "query": q,
        "results": results,
        "count": len(results),
        "disclaimer": get_disclaimer(),
    }


# ---------------------------------------------------------------------------
# GET /api/v1/samples — list sample labels
# ---------------------------------------------------------------------------

@api_router.get("/samples")
async def list_samples():
    """Return the sample labels available for testing."""
    return {
        "samples": {
            name: label.model_dump() for name, label in SAMPLE_LABELS.items()
        },
        "disclaimer": get_disclaimer(),
    }
