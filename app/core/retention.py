"""
File Retention Management.
Automatically cleans up uploaded label images and temporary artifacts older than the configured retention threshold.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.core.config import UPLOAD_DIR, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def cleanup_old_uploads(
    upload_dir: Path | str | None = None,
    retention_days: int | None = None,
    max_age_seconds: float | None = None,
) -> dict[str, int]:
    """
    Delete files in the upload directory older than retention threshold.
    Returns summary stats: {"deleted": int, "retained": int, "errors": int}.
    """
    settings = get_settings()
    if upload_dir is None:
        target_dir = UPLOAD_DIR
    else:
        target_dir = Path(upload_dir)

    if max_age_seconds is None:
        days = retention_days if retention_days is not None else settings.upload_retention_days
        max_age_seconds = float(days * 86400)

    if not target_dir.exists():
        return {"deleted": 0, "retained": 0, "errors": 0}

    now = time.time()
    deleted_count = 0
    retained_count = 0
    error_count = 0

    for root, dirs, files in os.walk(target_dir, topdown=False):
        for fname in files:
            fpath = Path(root) / fname
            try:
                mtime = os.path.getmtime(fpath)
                file_age = now - mtime
                if file_age > max_age_seconds:
                    fpath.unlink()
                    deleted_count += 1
                    logger.info("retention_deleted_old_upload", file=fpath.name, age_days=round(file_age / 86400, 2))
                else:
                    retained_count += 1
            except Exception as e:
                error_count += 1
                logger.error("retention_cleanup_file_error", file=str(fpath), error=str(e))
        for dirname in dirs:
            folder = Path(root) / dirname
            try:
                folder.rmdir()  # only removes extraction folders that are now empty
            except OSError:
                pass

    logger.info(
        "retention_cleanup_completed",
        deleted=deleted_count,
        retained=retained_count,
        errors=error_count,
    )
    return {
        "deleted": deleted_count,
        "retained": retained_count,
        "errors": error_count,
    }
