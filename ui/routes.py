"""
UI routes for the web interface.
Renders Jinja2 templates with HTMX progressive enhancement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import UPLOAD_DIR, get_disclaimer, get_settings
from app.core.logging import get_logger
from app.models.database import (
    ComplianceResultRecord,
    ExpertDecisionRecord,
    SubmissionRecord,
    get_db_session,
)
from app.models.enums import CheckStatus
from app.models.schemas import (
    SAMPLE_LABELS,
    AdditiveInfo,
    ComplianceReport,
    LabelSubmission,
    NutrientValue,
    NutritionalInfo,
)
from app.orchestrator.pipeline import CompliancePipeline

logger = get_logger(__name__)

ui_router = APIRouter(tags=["ui"])
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


def get_pipeline() -> CompliancePipeline:
    return CompliancePipeline()


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------

@ui_router.get("/", response_class=HTMLResponse)
async def ui_home(request: Request):
    """Render the home page with label submission form."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "disclaimer": get_disclaimer(),
            "samples": list(SAMPLE_LABELS.keys()),
        },
    )


# ---------------------------------------------------------------------------
# Submit label for checking
# ---------------------------------------------------------------------------

def parse_form_to_submission(
    form: Any,
    extraction_id: str | None = None,
    user_confirmed: bool = True,
) -> LabelSubmission:
    """Parse unified web form data into standard LabelSubmission."""
    import re

    # Parse ingredients
    ingredients_raw = form.get("ingredients_list", "")
    if isinstance(ingredients_raw, str):
        ingredients = [i.strip() for i in ingredients_raw.split(",") if i.strip()]
    else:
        ingredients = []

    # Auto-detect INS food additives from ingredients
    additives: list[AdditiveInfo] = []
    for ing in ingredients:
        ins_matches = re.findall(r"(?:INS\s*|E\s*)?(\d{3,4}(?:\([a-z0-9]+\))?)", ing, re.IGNORECASE)
        for m in ins_matches:
            additives.append(AdditiveInfo(name=ing, ins_number=f"INS {m}"))

    # Parse claims
    claims_raw = form.get("claims", "")
    claims = [c.strip() for c in claims_raw.split(",") if c.strip()] if claims_raw else []

    # Parse nutritional info table
    nutrients: list[NutrientValue] = []
    nutrient_names = form.getlist("nutrient_name[]") if hasattr(form, "getlist") else []
    nutrient_100gs = form.getlist("nutrient_100g[]") if hasattr(form, "getlist") else []
    nutrient_servings = form.getlist("nutrient_serving[]") if hasattr(form, "getlist") else []
    nutrient_units = form.getlist("nutrient_unit[]") if hasattr(form, "getlist") else []

    if nutrient_names:
        for i in range(len(nutrient_names)):
            n_name = nutrient_names[i].strip()
            if not n_name:
                continue
            val_100g = None
            if i < len(nutrient_100gs) and str(nutrient_100gs[i]).strip():
                try:
                    val_100g = float(nutrient_100gs[i])
                except ValueError:
                    pass
            val_serving = None
            if i < len(nutrient_servings) and str(nutrient_servings[i]).strip():
                try:
                    val_serving = float(nutrient_servings[i])
                except ValueError:
                    pass
            unit = nutrient_units[i].strip() if i < len(nutrient_units) and str(nutrient_units[i]).strip() else "g"
            nutrients.append(NutrientValue(name=n_name, per_100g=val_100g, per_serving=val_serving, unit=unit))

    energy_kcal = None
    if form.get("energy_kcal_100g"):
        try:
            energy_kcal = float(form.get("energy_kcal_100g"))
        except (ValueError, TypeError):
            pass

    nutritional_info = None
    if nutrients or energy_kcal is not None or form.get("serving_size"):
        nutritional_info = NutritionalInfo(
            serving_size=form.get("serving_size") or None,
            energy_kcal_per_100g=energy_kcal,
            nutrients=nutrients,
        )
    elif form.get("nutritional_info"):
        try:
            ni_data = json.loads(form.get("nutritional_info"))
            if isinstance(ni_data, dict):
                nutritional_info = NutritionalInfo(**ni_data)
        except Exception:
            pass

    return LabelSubmission(
        product_name=form.get("product_name", "Unknown Product"),
        product_category=form.get("product_category", ""),
        ingredients_list=ingredients,
        allergen_declaration=form.get("allergen_declaration") or None,
        net_quantity=form.get("net_quantity") or None,
        nutritional_info=nutritional_info,
        veg_nonveg_mark=form.get("veg_nonveg_mark") or None,
        fssai_licence_number=form.get("fssai_licence_number") or None,
        manufacturer_details=form.get("manufacturer_details") or None,
        batch_lot=form.get("batch_lot") or None,
        manufacturing_date=form.get("manufacturing_date") or None,
        expiry_date=form.get("expiry_date") or None,
        best_before=form.get("best_before") or None,
        storage_instructions=form.get("storage_instructions") or None,
        country_of_origin=form.get("country_of_origin") or "India",
        additives=additives,
        claims=claims,
        free_text_content=form.get("free_text_content") or None,
        extraction_id=extraction_id,
        user_confirmed=user_confirmed,
    )


# ---------------------------------------------------------------------------
# Submit label for checking (Manual)
# ---------------------------------------------------------------------------

@ui_router.post("/check", response_class=HTMLResponse)
async def ui_check(
    request: Request,
    pipeline: CompliancePipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db_session),
):
    """Handle manual form submission, run pipeline, and render results."""
    form = await request.form()
    submission = parse_form_to_submission(form, user_confirmed=True)

    report: ComplianceReport = pipeline.run(submission)

    # Persist to DB
    from app.models.database import AuditLogRecord, ComplianceResultRecord, SubmissionRecord
    sub_record = SubmissionRecord(
        check_id=report.check_id,
        product_name=submission.product_name,
        status=CheckStatus.COMPLETED.value,
        submission_data=submission.model_dump_json(),
    )
    db.add(sub_record)
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
    audit = AuditLogRecord(
        check_id=report.check_id,
        action="compliance_check_completed_via_ui",
        actor="ui_user",
        details=json.dumps({"recommendation": report.recommendation.value}),
    )
    db.add(audit)
    await db.flush()

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "report": report,
            "disclaimer": get_disclaimer(),
        },
    )


# ---------------------------------------------------------------------------
# Photo Scan & Review Workflow
# ---------------------------------------------------------------------------

@ui_router.post("/scan", response_class=HTMLResponse)
async def ui_scan(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Handle photo upload, run Stage 0 vision extraction, and redirect to review."""
    form = await request.form()
    files = form.getlist("files")
    if not files or not any(getattr(f, "filename", "") for f in files):
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error": "Please select at least 1 photo of the food label to scan.",
                "disclaimer": get_disclaimer(),
            },
            status_code=400,
        )

    settings = get_settings()
    if len(files) > settings.upload_max_images:
        return templates.TemplateResponse(request=request, name="error.html",
            context={"error": f"A scan can contain at most {settings.upload_max_images} photos.",
                     "disclaimer": get_disclaimer()}, status_code=400)
    raw_bytes_list: list[bytes] = []
    for f in files:
        if hasattr(f, "read"):
            content = await f.read(settings.upload_max_size_mb * 1024 * 1024 + 1)
            if len(content) > settings.upload_max_size_mb * 1024 * 1024:
                return templates.TemplateResponse(request=request, name="error.html",
                    context={"error": f"Each photo must be {settings.upload_max_size_mb} MB or smaller.",
                             "disclaimer": get_disclaimer()}, status_code=413)
            if content:
                raw_bytes_list.append(content)

    from app.agents.extraction import LabelExtractionAgent
    from app.core.config import UPLOAD_DIR
    from app.models.database import ExtractionRecord

    try:
        agent = LabelExtractionAgent()
    except (RuntimeError, ValueError, ImportError) as exc:
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={"disclaimer": get_disclaimer(), "samples": list(SAMPLE_LABELS.keys()),
                     "error_banner": f"Scanner AI is not configured: {exc}. Set VISION_PROVIDER and its API key in .env."},
            status_code=503,
        )
    result = await agent.extract_from_images(raw_bytes_list)

    if result.fallback_to_manual:
        # Fall back gracefully to manual entry with clear message
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "disclaimer": get_disclaimer(),
                "samples": list(SAMPLE_LABELS.keys()),
                "error_banner": f"Vision extraction was unable to read the photo ({result.error_message}). Please enter the label details manually below.",
            },
        )

    # Persist only sanitized image bytes (the raw camera upload is never retained).
    from app.agents.image_processing import sanitize_and_prepare_image
    saved_rel_paths = []
    upload_folder = UPLOAD_DIR / result.extraction_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    for idx, raw_bytes in enumerate(raw_bytes_list):
        processed = sanitize_and_prepare_image(
            raw_bytes, filename=f"label_photo_{idx+1}",
            max_size_bytes=get_settings().upload_max_size_mb * 1024 * 1024,
        )
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

    return RedirectResponse(url=f"/review/{result.extraction_id}", status_code=303)


@ui_router.get("/label-image/{extraction_id}/{filename}")
async def ui_label_image(extraction_id: str, filename: str, db: AsyncSession = Depends(get_db_session)):
    """Serve a sanitized label image only when its extraction record exists."""
    from app.models.database import ExtractionRecord
    if not filename.startswith("label_photo_") or Path(filename).name != filename:
        return HTMLResponse("Not found", status_code=404)
    res = await db.execute(select(ExtractionRecord).where(ExtractionRecord.extraction_id == extraction_id))
    record = res.scalar_one_or_none()
    requested_path = f"/label-image/{extraction_id}/{filename}"
    if record is None or requested_path not in json.loads(record.image_paths or "[]"):
        return HTMLResponse("Not found", status_code=404)
    path = UPLOAD_DIR / extraction_id / filename
    if not path.is_file():
        return HTMLResponse("Not found", status_code=404)
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@ui_router.get("/review/{extraction_id}", response_class=HTMLResponse)
async def ui_review(
    extraction_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    """Render the Review & Confirm screen with pre-filled fields and image thumbnails."""
    from app.models.database import ExtractionRecord
    from app.models.extraction import ExtractedLabelData

    res = await db.execute(
        select(ExtractionRecord).where(ExtractionRecord.extraction_id == extraction_id)
    )
    rec = res.scalar_one_or_none()
    if not rec:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={"error": "Extraction record not found.", "disclaimer": get_disclaimer()},
            status_code=404,
        )

    fields_data = ExtractedLabelData.model_validate_json(rec.extracted_data)
    image_paths = json.loads(rec.image_paths or "[]")

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "extraction_id": extraction_id,
            "fields": fields_data,
            "images": image_paths,
            "warnings": [],
            "disclaimer": get_disclaimer(),
        },
    )


@ui_router.post("/review/{extraction_id}/submit", response_class=HTMLResponse)
async def ui_review_submit(
    extraction_id: str,
    request: Request,
    pipeline: CompliancePipeline = Depends(get_pipeline),
    db: AsyncSession = Depends(get_db_session),
):
    """Process user-verified label fields from the Review screen and run compliance check."""
    form = await request.form()
    user_confirmed = form.get("user_confirmed") in ("true", "True", "on", "1")

    submission = parse_form_to_submission(form, extraction_id=extraction_id, user_confirmed=user_confirmed)
    report: ComplianceReport = pipeline.run(submission)

    # Update extraction record in DB
    from app.models.database import AuditLogRecord, ComplianceResultRecord, ExtractionRecord, SubmissionRecord
    res = await db.execute(
        select(ExtractionRecord).where(ExtractionRecord.extraction_id == extraction_id)
    )
    ext_rec = res.scalar_one_or_none()
    if ext_rec:
        ext_rec.user_corrected_data = submission.model_dump_json()
        ext_rec.user_confirmed = user_confirmed
        db.add(ext_rec)

    # Persist submission & result
    sub_record = SubmissionRecord(
        check_id=report.check_id,
        product_name=submission.product_name,
        status=CheckStatus.COMPLETED.value,
        submission_data=submission.model_dump_json(),
    )
    db.add(sub_record)
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
    audit = AuditLogRecord(
        check_id=report.check_id,
        action="compliance_check_completed_via_photo_extraction",
        actor="verified_user" if user_confirmed else "unverified_user",
        details=json.dumps({"recommendation": report.recommendation.value, "extraction_id": extraction_id}),
    )
    db.add(audit)
    await db.flush()

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "report": report,
            "disclaimer": get_disclaimer(),
        },
    )


# ---------------------------------------------------------------------------
# Load sample label (returns JSON for HTMX)
# ---------------------------------------------------------------------------

@ui_router.get("/sample/{sample_name}")
async def ui_get_sample(sample_name: str):
    """Return a sample label as JSON for the 'load sample' button."""
    label = SAMPLE_LABELS.get(sample_name)
    if label is None:
        return {"error": "Sample not found"}
    return label.model_dump()


# ---------------------------------------------------------------------------
# History page
# ---------------------------------------------------------------------------

@ui_router.get("/history", response_class=HTMLResponse)
async def ui_history(
    request: Request,
    page: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db_session),
):
    """Render the history page listing past checks."""
    per_page = 20
    offset = (page - 1) * per_page

    count_result = await db.execute(select(SubmissionRecord.id))
    total = len(count_result.all())

    result = await db.execute(
        select(SubmissionRecord)
        .order_by(desc(SubmissionRecord.submitted_at))
        .offset(offset)
        .limit(per_page)
    )
    records = result.scalars().all()

    items = []
    for rec in records:
        # Get recommendation
        res = await db.execute(
            select(ComplianceResultRecord.recommendation)
            .where(ComplianceResultRecord.check_id == rec.check_id)
        )
        rec_val = res.scalar_one_or_none()

        # Check for expert decision
        dec = await db.execute(
            select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == rec.check_id)
        )
        has_decision = dec.scalar_one_or_none() is not None

        items.append({
            "check_id": rec.check_id,
            "product_name": rec.product_name,
            "status": rec.status,
            "recommendation": rec_val,
            "submitted_at": rec.submitted_at,
            "has_expert_decision": has_decision,
        })

    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "items": items,
            "page": page,
            "total_pages": total_pages,
            "total": total,
            "disclaimer": get_disclaimer(),
        },
    )


# ---------------------------------------------------------------------------
# View a specific check result
# ---------------------------------------------------------------------------

@ui_router.get("/check/{check_id}", response_class=HTMLResponse)
async def ui_view_check(
    request: Request,
    check_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """View a specific compliance check result with expert review panel."""
    result = await db.execute(
        select(ComplianceResultRecord).where(ComplianceResultRecord.check_id == check_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error": "Check not found",
                "disclaimer": get_disclaimer(),
            },
        )

    report = ComplianceReport.model_validate_json(record.report_data)

    # Check for expert decision
    dec_result = await db.execute(
        select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == check_id)
    )
    decision_record = dec_result.scalar_one_or_none()
    expert_decision = None
    if decision_record:
        expert_decision = {
            "decision_id": decision_record.decision_id,
            "action": decision_record.action,
            "original_recommendation": decision_record.original_recommendation,
            "reason": decision_record.reason,
            "expert_name": decision_record.expert_name,
            "decided_at": decision_record.decided_at,
        }

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "report": report,
            "expert_decision": expert_decision,
            "disclaimer": get_disclaimer(),
        },
    )


# ---------------------------------------------------------------------------
# Expert decision form submission
# ---------------------------------------------------------------------------

@ui_router.post("/check/{check_id}/decision", response_class=HTMLResponse)
async def ui_expert_decision(
    request: Request,
    check_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Handle expert decision form submission."""
    form = await request.form()
    action = form.get("action", "accept")
    reason = form.get("reason", "")
    expert_name = form.get("expert_name", "")

    # Verify check exists
    result = await db.execute(
        select(ComplianceResultRecord).where(ComplianceResultRecord.check_id == check_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "error": "Check not found",
                "disclaimer": get_disclaimer(),
            },
        )

    # Check for existing decision
    existing = await db.execute(
        select(ExpertDecisionRecord).where(ExpertDecisionRecord.check_id == check_id)
    )
    if existing.scalar_one_or_none():
        return RedirectResponse(url=f"/check/{check_id}", status_code=303)

    from datetime import datetime
    from uuid import uuid4

    from app.models.database import AuditLogRecord

    decision_id = uuid4().hex[:12]
    now = datetime.utcnow()

    dec_record = ExpertDecisionRecord(
        decision_id=decision_id,
        check_id=check_id,
        action=action,
        original_recommendation=record.recommendation,
        reason=reason,
        expert_name=expert_name,
        decided_at=now,
    )
    db.add(dec_record)

    audit = AuditLogRecord(
        check_id=check_id,
        action=f"expert_decision_{action}_via_ui",
        actor=expert_name or "anonymous_expert",
        details=json.dumps({
            "original_recommendation": record.recommendation,
            "action": action,
            "reason": reason,
        }),
    )
    db.add(audit)
    await db.flush()

    return RedirectResponse(url=f"/check/{check_id}", status_code=303)
