"""
Unit tests for upload retention deletion (Item 8).
Verifies that files older than the retention threshold are purged,
while recent files are safely retained.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.core.retention import cleanup_old_uploads


def test_cleanup_old_uploads_deletes_old_file(tmp_path: Path):
    # Create two test files: one old (8 days ago), one fresh (now)
    old_file = tmp_path / "old_label_upload.png"
    old_file.write_bytes(b"dummy_old_image_data")

    fresh_file = tmp_path / "recent_label_upload.png"
    fresh_file.write_bytes(b"dummy_recent_image_data")

    now = time.time()
    # Set old_file mtime to 8 days in the past (8 * 86400 seconds)
    eight_days_ago = now - (8 * 86400)
    os.utime(old_file, (eight_days_ago, eight_days_ago))

    # Run cleanup with 7-day retention
    stats = cleanup_old_uploads(upload_dir=tmp_path, retention_days=7)

    assert stats["deleted"] == 1
    assert stats["retained"] == 1
    assert not old_file.exists(), "Old file should have been deleted by retention cleanup"
    assert fresh_file.exists(), "Recent file must be preserved by retention cleanup"


def test_cleanup_old_uploads_handles_empty_dir(tmp_path: Path):
    stats = cleanup_old_uploads(upload_dir=tmp_path / "non_existent")
    assert stats["deleted"] == 0
    assert stats["retained"] == 0
