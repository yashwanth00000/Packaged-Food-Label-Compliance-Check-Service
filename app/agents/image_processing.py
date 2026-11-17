"""
Image validation, sanitation, EXIF stripping, and quality gate.
All operations inspect content bytes — never file extensions alone.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.core.logging import get_logger

logger = get_logger(__name__)

# Constants
ALLOWED_MIME_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_DIMENSION = 2048


@dataclass
class ProcessedImage:
    """Sanitized, downscaled image ready for vision processing."""
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    quality_warnings: list[str]
    filename: str = ""


def detect_mime_type_from_bytes(data: bytes) -> str | None:
    """
    Validate MIME type strictly by file header magic bytes (not file extension).
    """
    if len(data) < 12:
        return None
    # JPEG magic: \xff\xd8\xff
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # PNG magic: \x89PNG\r\n\x1a\n
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # WebP magic: RIFF....WEBP
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def calculate_image_quality(img: Image.Image) -> list[str]:
    """
    Quality gate: inspect image for blur, extreme darkness, and cropped dimensions.
    Returns list of user-facing quality warnings, each prompting a retake.
    """
    warnings: list[str] = []
    w, h = img.size

    # Check minimum resolution
    if w < 160 or h < 160:
        warnings.append(
            f"Photo resolution is very low ({w}x{h}). Text may be unreadable; please retake photo closer to the packaging."
        )

    # Check for cropped / abnormal aspect ratios
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    if aspect_ratio > 3.5:
        warnings.append(
            f"Photo appears cropped or cut off (aspect ratio {aspect_ratio:.1f}:1). "
            "Critical label details may be missing; please retake photo showing the full label."
        )

    # Convert to grayscale for illumination & contrast analysis
    gray = img.convert("L")
    stat = gray.getextrema()
    # Check if image is blank / solid color
    if stat[0] == stat[1]:
        warnings.append("Photo appears blank or solid color. Please retake photo of the actual package.")
        return warnings

    # Calculate average brightness
    histogram = gray.histogram()
    total_pixels = sum(histogram)
    if total_pixels > 0:
        avg_brightness = sum(i * count for i, count in enumerate(histogram)) / total_pixels
        if avg_brightness < 30:
            warnings.append("Photo is too dark. Please retake photo in brighter lighting.")
        elif avg_brightness > 240:
            warnings.append("Photo is overexposed/washed out. Please retake photo without direct glare.")

    # Blur check using pixel variance
    small = gray.resize((128, 128))
    pixels = list(small.getdata())
    mean_val = sum(pixels) / len(pixels)
    variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)
    if variance < 80:
        warnings.append("Photo appears blurry or out of focus. Please retake photo holding camera steady.")

    return warnings


def sanitize_and_prepare_image(
    data: bytes,
    filename: str = "",
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_size_bytes: int = MAX_IMAGE_SIZE_BYTES,
) -> ProcessedImage:
    """
    1. Size check
    2. Content-based MIME detection
    3. Corrupt file detection
    4. EXIF / GPS stripping
    5. Downscaling
    6. Quality gate warnings
    """
    if len(data) > max_size_bytes:
        raise ValueError(f"Image file size ({len(data) / (1024*1024):.1f} MB) exceeds limit of {max_size_bytes / (1024*1024):.0f} MB")

    mime_type = detect_mime_type_from_bytes(data)
    if not mime_type or mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError("Invalid image format. Only JPEG, PNG, and WebP images are allowed (validated by content).")

    # Verify image integrity
    try:
        with Image.open(io.BytesIO(data)) as test_img:
            test_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable image file: {e!s}") from e

    # Re-open for actual processing (verify closes/invalidates the image object)
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Transpose orientation according to EXIF before stripping
            img = ImageOps.exif_transpose(img)

            quality_warnings = calculate_image_quality(img)

            # Ensure RGB mode (handles RGBA, P, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            # Downscale if exceeding max_dimension while preserving aspect ratio
            w, h = img.size
            if max(w, h) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                w, h = img.size

            # Save clean image without EXIF metadata
            output_buffer = io.BytesIO()
            target_format = "JPEG" if mime_type == "image/jpeg" else ("PNG" if mime_type == "image/png" else "WEBP")
            img.save(output_buffer, format=target_format, quality=88, optimize=True)
            clean_bytes = output_buffer.getvalue()

            logger.info(
                "image_processed",
                width=w,
                height=h,
                mime_type=mime_type,
                size_bytes=len(clean_bytes),
                warning_count=len(quality_warnings),
            )

            return ProcessedImage(
                image_bytes=clean_bytes,
                mime_type=mime_type,
                width=w,
                height=h,
                quality_warnings=quality_warnings,
                filename=filename,
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to process image: {e!s}") from e
