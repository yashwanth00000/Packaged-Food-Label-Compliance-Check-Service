"""
Stage 0: Label Extraction Agent
Extracts label fields from 1-4 food label photos using Vision LLMs.
Enforces null-not-guess rule, prompt injection defence, image quality gating,
circuit-breaker resilience, and fallback to manual entry on failure.
"""

from __future__ import annotations

import time
from uuid import uuid4

from app.agents.image_processing import ProcessedImage, sanitize_and_prepare_image
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import (
    EXTRACTION_CONFIDENCE,
    EXTRACTION_LATENCY,
    EXTRACTION_REQUESTS,
)
from app.models.extraction import (
    ExtractedField,
    ExtractedLabelData,
    ExtractionResult,
)
from app.models.schemas import LabelSubmission
from app.services.vision_provider import VisionProvider, get_vision_provider

logger = get_logger(__name__)


def enforce_null_not_guess(data: ExtractedLabelData) -> ExtractedLabelData:
    """
    Strictly enforce: if a field's value is None, empty list, or empty string,
    its confidence MUST be 0.0. Never infer or guess unseen fields.
    """
    for field_name, field_val in data.__dict__.items():
        if isinstance(field_val, ExtractedField):
            if field_val.value is None or field_val.value == "" or field_val.value == []:
                field_val.value = None
                field_val.confidence = 0.0
    return data


class LabelExtractionAgent:
    """
    Stage 0 Agent: Ingests 1-4 photos, validates and sanitizes them, runs vision extraction,
    applies quality gates and security defenses, and outputs structured ExtractedLabelData.
    """

    def __init__(self, vision_provider: VisionProvider | None = None):
        self.vision_provider = vision_provider or get_vision_provider()
        self.settings = get_settings()

    async def extract_from_images(
        self,
        raw_images: list[bytes],
        filenames: list[str] | None = None,
    ) -> ExtractionResult:
        start_time = time.time()
        extraction_id = f"ext-{uuid4().hex}"

        if not raw_images:
            return ExtractionResult(
                extraction_id=extraction_id,
                fallback_to_manual=True,
                error_message="No images uploaded. Please provide 1 to 4 label photos.",
            )

        if len(raw_images) > self.settings.upload_max_images:
            return ExtractionResult(
                extraction_id=extraction_id,
                fallback_to_manual=True,
                error_message=f"Too many images. Maximum allowed is {self.settings.upload_max_images}.",
            )

        # Step 1: Validate, sanitize, and check quality for each image
        processed_images: list[ProcessedImage] = []
        all_warnings: list[str] = []

        for idx, img_bytes in enumerate(raw_images):
            try:
                fn = filenames[idx] if filenames and idx < len(filenames) else ""
                proc = sanitize_and_prepare_image(
                    img_bytes,
                    filename=fn,
                    max_size_bytes=self.settings.upload_max_size_mb * 1024 * 1024,
                )
                processed_images.append(proc)
                for w in proc.quality_warnings:
                    all_warnings.append(f"Image {idx+1}: {w}")
            except ValueError as e:
                logger.warning("image_sanitization_failed", image_index=idx, error=str(e))
                EXTRACTION_REQUESTS.labels(status="error").inc()
                return ExtractionResult(
                    extraction_id=extraction_id,
                    fallback_to_manual=True,
                    error_message=f"Image {idx+1} rejected: {e!s}",
                )

        # Step 2: Invoke Vision Model with retry / circuit breaker
        try:
            logger.info("running_vision_extraction", extraction_id=extraction_id, count=len(processed_images))
            raw_data = await self.vision_provider.extract_label_data(processed_images)

            # Step 3: Enforce null-not-guess rule
            cleaned_data = enforce_null_not_guess(raw_data)

            # Record metrics
            duration = time.time() - start_time
            EXTRACTION_LATENCY.observe(duration)
            EXTRACTION_REQUESTS.labels(status="success").inc()

            for f_name, f_val in cleaned_data.__dict__.items():
                if isinstance(f_val, ExtractedField):
                    EXTRACTION_CONFIDENCE.observe(f_val.confidence)

            return ExtractionResult(
                extraction_id=extraction_id,
                fields=cleaned_data,
                image_quality_warnings=all_warnings,
                image_count=len(processed_images),
                processing_time_seconds=round(duration, 3),
                fallback_to_manual=False,
            )

        except Exception as e:
            logger.error("vision_extraction_failed", extraction_id=extraction_id, error=str(e))
            duration = time.time() - start_time
            EXTRACTION_REQUESTS.labels(status="error").inc()
            return ExtractionResult(
                extraction_id=extraction_id,
                image_quality_warnings=all_warnings,
                image_count=len(processed_images),
                processing_time_seconds=round(duration, 3),
                fallback_to_manual=True,
                error_message=f"Vision model extraction error: {e!s}. Falling back to manual entry.",
            )

    @staticmethod
    def convert_to_label_submission(data: ExtractedLabelData) -> LabelSubmission:
        """
        Convert confirmed extracted data into standard pipeline LabelSubmission.
        All fields are treated as untrusted strings.
        """
        nutritional_info = None
        if data.nutritional_info.value:
            nutritional_info = data.nutritional_info.value.to_nutritional_info()

        return LabelSubmission(
            product_name=data.product_name.value or "Unknown Product",
            product_category=data.product_category.value or "",
            ingredients_list=data.ingredients_list.value or [],
            allergen_declaration=data.allergen_declaration.value,
            net_quantity=data.net_quantity.value,
            nutritional_info=nutritional_info,
            veg_nonveg_mark=data.veg_nonveg_mark.value,
            fssai_licence_number=data.fssai_licence_number.value,
            manufacturer_details=data.manufacturer_details.value,
            batch_lot=data.batch_lot.value,
            manufacturing_date=data.manufacturing_date.value,
            expiry_date=data.expiry_date.value,
            best_before=data.best_before.value,
            storage_instructions=data.storage_instructions.value,
            country_of_origin=data.country_of_origin.value,
            claims=data.claims.value or [],
            free_text_content=data.other_visible_text.value,
        )

    @classmethod
    def to_submission(
        cls,
        data: ExtractedLabelData,
        extraction_id: str | None = None,
        user_confirmed: bool | None = None,
    ) -> LabelSubmission:
        """Alias for convert_to_label_submission with extraction_id and confirmation."""
        sub = cls.convert_to_label_submission(data)
        if extraction_id:
            sub.extraction_id = extraction_id
        if user_confirmed is not None:
            sub.user_confirmed = user_confirmed
        elif getattr(data, "user_confirmed", None) is not None:
            sub.user_confirmed = getattr(data, "user_confirmed")
        return sub
