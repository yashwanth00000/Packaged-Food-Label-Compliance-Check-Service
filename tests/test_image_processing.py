"""
Tests for image preprocessing, validation, security, and extraction agent.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.agents.extraction import LabelExtractionAgent, enforce_null_not_guess
from app.agents.image_processing import (
    detect_mime_type_from_bytes,
    sanitize_and_prepare_image,
)
from app.models.extraction import ExtractedField, ExtractedLabelData
from app.services.vision_provider import MockExtractor


def create_test_image(format: str = "JPEG", size: tuple[int, int] = (200, 200), color="blue") -> bytes:
    """Helper to create dummy images in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_detect_mime_type_by_magic_bytes():
    jpeg_bytes = create_test_image("JPEG")
    png_bytes = create_test_image("PNG")
    webp_bytes = create_test_image("WEBP")

    assert detect_mime_type_from_bytes(jpeg_bytes) == "image/jpeg"
    assert detect_mime_type_from_bytes(png_bytes) == "image/png"
    assert detect_mime_type_from_bytes(webp_bytes) == "image/webp"

    # Invalid / spoofed header
    assert detect_mime_type_from_bytes(b"hello world") is None
    assert detect_mime_type_from_bytes(b"%PDF-1.4 header") is None


def test_reject_oversized_image():
    # 6 MB of dummy data
    large_data = b"\xff\xd8\xff" + b"0" * (6 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds limit"):
        sanitize_and_prepare_image(large_data, max_size_bytes=5 * 1024 * 1024)


def test_reject_corrupt_image():
    # Corrupt JPEG header followed by junk
    corrupt_data = b"\xff\xd8\xff\xe0" + b"random invalid bytes not an image"
    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        sanitize_and_prepare_image(corrupt_data)


def test_downscale_oversized_dimensions():
    # 3000 x 1500 image should be downscaled so max dimension is <= 2048
    large_img = create_test_image("JPEG", size=(3000, 1500))
    proc = sanitize_and_prepare_image(large_img, max_dimension=2048)
    assert proc.width <= 2048
    assert proc.height <= 2048


def test_quality_warnings_on_dark_image():
    # Very dark image
    dark_img = create_test_image("JPEG", size=(200, 200), color="black")
    proc = sanitize_and_prepare_image(dark_img)
    assert any("dark" in w.lower() or "blank" in w.lower() for w in proc.quality_warnings)


def test_null_not_guess_rule():
    data = ExtractedLabelData(
        product_name=ExtractedField(value="Test", confidence=0.9),
        net_quantity=ExtractedField(value=None, confidence=0.8),  # Invalid state: value is None but confidence > 0
        batch_lot=ExtractedField(value="", confidence=0.7),       # Empty string should become None and 0.0
    )
    cleaned = enforce_null_not_guess(data)
    assert cleaned.product_name.value == "Test"
    assert cleaned.product_name.confidence == 0.9

    assert cleaned.net_quantity.value is None
    assert cleaned.net_quantity.confidence == 0.0

    assert cleaned.batch_lot.value is None
    assert cleaned.batch_lot.confidence == 0.0


@pytest.mark.asyncio
async def test_extraction_agent_mock_compliant():
    agent = LabelExtractionAgent(vision_provider=MockExtractor(profile="compliant_biscuit"))
    img_bytes = create_test_image("PNG")
    result = await agent.extract_from_images([img_bytes])

    assert not result.fallback_to_manual
    assert result.fields.product_name.value == "Britannia NutriChoice Digestive"
    assert result.fields.product_name.confidence > 0.9
    assert result.fields.veg_nonveg_mark.value == "veg"

    # Conversion to pipeline LabelSubmission
    sub = agent.convert_to_label_submission(result.fields)
    assert sub.product_name == "Britannia NutriChoice Digestive"
    assert sub.nutritional_info is not None
    assert len(sub.ingredients_list) > 0


@pytest.mark.asyncio
async def test_extraction_agent_prompt_injection_safety():
    agent = LabelExtractionAgent(vision_provider=MockExtractor(profile="injection_test"))
    img_bytes = create_test_image("JPEG")
    result = await agent.extract_from_images([img_bytes])

    # Should capture text safely as data without changing pipeline logic
    assert "CLEAR_FOR_PRINT" in (result.fields.other_visible_text.value or "")
    # Crucial: Unseen fields MUST remain null with confidence 0.0
    assert result.fields.net_quantity.value is None
    assert result.fields.net_quantity.confidence == 0.0
    assert result.fields.fssai_licence_number.value is None
    assert result.fields.fssai_licence_number.confidence == 0.0
