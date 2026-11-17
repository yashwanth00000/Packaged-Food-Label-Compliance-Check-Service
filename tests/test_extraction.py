"""
Unit and integration tests for Stage 0 (Label Extraction Agent),
Review-and-Confirm workflow, and Decision Policy enforcement.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.agents.decision import DecisionAgent
from app.agents.extraction import LabelExtractionAgent
from app.models.database import ExtractionRecord, SubmissionRecord
from app.models.enums import Recommendation
from app.models.extraction import ExtractedField, ExtractedLabelData
from app.models.schemas import SAMPLE_LABELS, LabelSubmission
from app.services.vision_provider import MockExtractor


def _make_test_image_bytes(text: str = "Test Label", width: int = 400, height: int = 300) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestLabelExtractionAgent:
    @pytest.mark.asyncio
    async def test_extract_from_images_returns_extracted_label_data(self):
        agent = LabelExtractionAgent(vision_provider=MockExtractor(profile="compliant_biscuit"))
        raw_bytes = _make_test_image_bytes()
        result = await agent.extract_from_images([raw_bytes], ["test.png"])

        assert result.extraction_id.startswith("ext-")
        assert result.image_count == 1
        assert result.extracted_data.product_name.value is not None
        assert result.extracted_data.product_name.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_null_not_guess_enforcement(self):
        """Unseen fields must strictly be null with confidence 0.0."""
        agent = LabelExtractionAgent(vision_provider=MockExtractor(profile="missing_declarations"))
        raw_bytes = _make_test_image_bytes()
        result = await agent.extract_from_images([raw_bytes], ["missing.png"])

        assert result.extracted_data.net_quantity.value is None
        assert result.extracted_data.net_quantity.confidence == 0.0
        assert result.extracted_data.manufacturer_details.value is None
        assert result.extracted_data.manufacturer_details.confidence == 0.0

    def test_to_submission_conversion(self):
        agent = LabelExtractionAgent(vision_provider=MockExtractor())
        extracted_data = ExtractedLabelData(
            product_name=ExtractedField(value="Organic Oats", confidence=0.9),
            ingredients_list=ExtractedField(value=["Rolled Oats"], confidence=0.9),
            net_quantity=ExtractedField(value="500 g", confidence=0.9),
            veg_nonveg_mark=ExtractedField(value="veg", confidence=0.95),
            fssai_licence_number=ExtractedField(value="10012011000123", confidence=0.95),
        )
        submission = agent.to_submission(extracted_data, extraction_id="ext-999", user_confirmed=True)
        assert submission.product_name == "Organic Oats"
        assert submission.net_quantity == "500 g"
        assert submission.veg_nonveg_mark == "veg"
        assert submission.extraction_id == "ext-999"
        assert submission.user_confirmed is True


class TestExtractionDecisionPolicy:
    def test_unconfirmed_photo_extraction_cannot_be_clear_for_print(self):
        """If submission is from photo extraction but unconfirmed by user, cannot be CLEAR_FOR_PRINT."""
        decision_agent = DecisionAgent()
        submission = LabelSubmission(
            product_name="Good Biscuit",
            product_category="Bakery",
            ingredients_list=["Wheat", "Sugar"],
            net_quantity="100 g",
            veg_nonveg_mark="veg",
            fssai_licence_number="10012011000123",
            extraction_id="ext-1234",
            user_confirmed=False,  # Unconfirmed!
        )
        context = {
            "all_findings": [],
            "mandatory_checks_passed": True,
            "is_photo_extraction": True,
            "user_confirmed": False,
        }
        result = decision_agent.process(submission, context)

        assert result["recommendation"] == Recommendation.REVISE_AND_RECHECK
        assert any("confirmed" in r.lower() for r in result["reasons"])

    def test_confirmed_photo_extraction_with_passes_can_be_clear_for_print(self):
        """If user confirmed details against physical label and checks pass, CLEAR_FOR_PRINT is allowed."""
        decision_agent = DecisionAgent()
        submission = LabelSubmission(
            product_name="Good Biscuit",
            product_category="Bakery",
            ingredients_list=["Wheat", "Sugar"],
            net_quantity="100 g",
            veg_nonveg_mark="veg",
            fssai_licence_number="10012011000123",
            extraction_id="ext-1234",
            user_confirmed=True,  # Confirmed!
        )
        context = {
            "all_findings": [],
            "mandatory_checks_passed": True,
            "is_photo_extraction": True,
            "user_confirmed": True,
        }
        result = decision_agent.process(submission, context)
        assert result["recommendation"] == Recommendation.CLEAR_FOR_PRINT

    def test_unconfirmed_low_confidence_fields_block_clear_for_print(self):
        """Unconfirmed low confidence fields must escalate."""
        decision_agent = DecisionAgent()
        submission = LabelSubmission(
            product_name="Good Biscuit",
            product_category="Bakery",
            ingredients_list=["Wheat", "Sugar"],
            net_quantity="100 g",
            veg_nonveg_mark="veg",
            fssai_licence_number="10012011000123",
            extraction_id="ext-1234",
            user_confirmed=True,
        )
        context = {
            "all_findings": [],
            "mandatory_checks_passed": True,
            "is_photo_extraction": True,
            "user_confirmed": True,
            "unconfirmed_low_confidence_fields": ["allergen_declaration"],
        }
        result = decision_agent.process(submission, context)
        assert result["recommendation"] == Recommendation.REVISE_AND_RECHECK
        assert any("low" in r.lower() and "confidence" in r.lower() for r in result["reasons"])


class TestExtractionApiFlow:
    def test_extract_endpoint_success(self, client: TestClient):
        img_bytes = _make_test_image_bytes()
        response = client.post(
            "/api/v1/extract",
            files=[("images", ("test_label.png", img_bytes, "image/png"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert "extraction_id" in data
        assert "fields" in data
        assert data["image_count"] == 1

    def test_extract_endpoint_rejects_empty(self, client: TestClient):
        response = client.post("/api/v1/extract", files=[])
        # Either 400 or 422 depending on FastAPI form binding
        assert response.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_check_with_extraction_id_links_record(self, client: TestClient, test_db):
        # 1. Create an extraction record in DB
        async for session in test_db():
            ext_record = ExtractionRecord(
                extraction_id="ext-test-link-99",
                image_paths="[]",
                extracted_data="{}",
                user_confirmed=True,
            )
            session.add(ext_record)
            await session.commit()
            break

        # 2. Submit check with this extraction_id
        payload = SAMPLE_LABELS["compliant_biscuit"].model_dump()
        payload["extraction_id"] = "ext-test-link-99"
        payload["user_confirmed"] = True

        response = client.post("/api/v1/check", json=payload)
        assert response.status_code == 200
        check_id = response.json()["check_id"]

        # 3. Verify SubmissionRecord in DB has linked extraction_id
        async for session in test_db():
            from sqlalchemy import select
            res = await session.execute(select(SubmissionRecord).filter_by(check_id=check_id))
            sub_in_db = res.scalar_one_or_none()
            assert sub_in_db is not None
            assert "ext-test-link-99" in sub_in_db.submission_data
            break
