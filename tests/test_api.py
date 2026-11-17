"""
API endpoint tests.
"""

import pytest

from app.models.enums import Recommendation
from app.models.schemas import SAMPLE_LABELS


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "disclaimer" in data

    def test_ready(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestComplianceAPI:
    def test_check_compliant_label(self, client):
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        assert "check_id" in data
        assert "report" in data
        assert data["disclaimer"]
        # Should have a recommendation
        assert data["report"]["recommendation"] in [
            "CLEAR_FOR_PRINT", "REVISE_AND_RECHECK", "REGULATORY_REVIEW"
        ]

    def test_check_missing_declarations(self, client):
        label = SAMPLE_LABELS["missing_declarations"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        report = data["report"]
        assert report["recommendation"] in ["REVISE_AND_RECHECK", "REGULATORY_REVIEW"]
        assert len(report["findings"]) > 0

    def test_check_prohibited_claims(self, client):
        label = SAMPLE_LABELS["prohibited_claims"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        assert resp.status_code == 200
        data = resp.json()
        report = data["report"]
        assert report["recommendation"] == "REGULATORY_REVIEW"

    def test_check_returns_disclaimer(self, client):
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        data = resp.json()
        assert "DISCLAIMER" in data["disclaimer"].upper() or "disclaimer" in data["disclaimer"].lower()

    def test_get_nonexistent_check(self, client):
        resp = client.get("/api/v1/check/nonexistent_id")
        assert resp.status_code == 404

    def test_list_checks_empty(self, client):
        resp = client.get("/api/v1/checks")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 0

    def test_samples_endpoint(self, client):
        resp = client.get("/api/v1/samples")
        assert resp.status_code == 200
        data = resp.json()
        assert "samples" in data
        assert len(data["samples"]) == 3

    def test_invalid_submission(self, client):
        """Submitting garbage should return a 422."""
        resp = client.post("/api/v1/check", json={"not_a_real_field": "value"})
        assert resp.status_code == 422


class TestExpertDecisionAPI:
    def test_expert_decision_unauthorized_without_key(self, client):
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        check_id = resp.json()["check_id"]

        decision = {"action": "accept", "reason": "OK", "expert_name": "Test"}
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision)
        assert resp2.status_code == 401
        assert "Unauthorized" in resp2.json()["error"]

    def test_submit_check_then_expert_decision(self, client):
        # First, run a check
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        assert resp.status_code == 200
        check_id = resp.json()["check_id"]

        # Now submit expert decision with valid per-expert key
        decision = {
            "action": "accept",
            "reason": "Reviewed and confirmed compliance.",
        }
        headers = {"X-Expert-Key": "expert-key-dr-sharma-2024"}
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["check_id"] == check_id
        assert data["action"] == "accept"
        assert "Dr. Anita Sharma" in data["expert_name"]

    def test_expert_decision_rejects_x_user_role_alone(self, client):
        """Item 1: X-User-Role alone MUST return 401 Unauthorized."""
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        check_id = resp.json()["check_id"]

        headers = {"X-User-Role": "expert"}
        decision = {"action": "accept", "reason": "Attempting header bypass"}
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)
        assert resp2.status_code == 401
        assert "X-Expert-Key" in resp2.json()["error"]

    def test_expert_decision_rejects_wrong_key(self, client):
        """Item 1: An invalid or wrong expert key MUST return 401 Unauthorized."""
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        check_id = resp.json()["check_id"]

        headers = {"X-Expert-Key": "invalid-wrong-secret-key-999"}
        decision = {"action": "accept", "reason": "Attempting bad key"}
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)
        assert resp2.status_code == 401

    @pytest.mark.asyncio
    async def test_expert_identity_written_to_audit_log(self, client, test_db):
        """Item 1: Confirm expert identity is strictly written to the audit log actor."""
        from sqlalchemy import select

        from app.models.database import AuditLogRecord

        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        check_id = resp.json()["check_id"]

        headers = {"X-Expert-Key": "expert-key-rkumar-2024"}
        decision = {"action": "override", "reason": "Special clearance granted by compliance officer."}
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)
        assert resp2.status_code == 200

        # Query audit log
        async for session in test_db():
            result = await session.execute(
                select(AuditLogRecord).where(
                    AuditLogRecord.check_id == check_id,
                    AuditLogRecord.action == "expert_decision_override",
                )
            )
            audit_entry = result.scalar_one_or_none()
            assert audit_entry is not None
            assert "Rajesh Kumar" in audit_entry.actor

    def test_server_enforcement_rejects_clear_for_print_when_user_confirmed_false(self, client):
        """Item 2: API must reject CLEAR_FOR_PRINT when user_confirmed is false."""
        label_dict = SAMPLE_LABELS["compliant_biscuit"].model_dump(mode="json")
        label_dict["user_confirmed"] = False  # Explicitly unconfirmed

        resp = client.post("/api/v1/check", json=label_dict)
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendation"] != Recommendation.CLEAR_FOR_PRINT.value
        assert data["recommendation"] == Recommendation.REVISE_AND_RECHECK.value
        assert any("user_confirmed" in r or "not been confirmed" in r for r in data["report"]["reasons"])

    def test_duplicate_expert_decision_rejected(self, client):
        # Run a check
        label = SAMPLE_LABELS["compliant_biscuit"]
        resp = client.post("/api/v1/check", json=label.model_dump(mode="json"))
        check_id = resp.json()["check_id"]

        headers = {"X-Expert-Key": "expert-key-fssai-2024"}
        decision = {"action": "accept", "reason": "OK"}
        client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)

        # Try to submit again
        resp2 = client.post(f"/api/v1/check/{check_id}/decision", json=decision, headers=headers)
        assert resp2.status_code == 409

    def test_expert_decision_on_nonexistent_check(self, client):
        headers = {"X-Expert-Key": "expert-key-fssai-2024"}
        decision = {"action": "accept", "reason": "OK"}
        resp = client.post("/api/v1/check/fake_id/decision", json=decision, headers=headers)
        assert resp.status_code == 404


class TestUIEndpoints:
    def test_home_page(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "FSSAI" in resp.text or "Compliance" in resp.text

    def test_history_page(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200


class TestExtractAPI:
    def test_extract_endpoint_success(self, client):
        import io

        from PIL import Image

        buf = io.BytesIO()
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        files = [("files", ("label.jpg", image_bytes, "image/jpeg"))]
        resp = client.post("/api/v1/extract", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "extraction_id" in data
        assert "fields" in data
        assert not data["fallback_to_manual"]
        assert data["fields"]["product_name"]["value"]

    def test_extract_endpoint_no_files(self, client):
        # Empty multipart request
        resp = client.post("/api/v1/extract", files=[])
        assert resp.status_code in (400, 422)


class TestAuditImmutability:
    @pytest.mark.asyncio
    async def test_audit_log_cannot_be_updated_or_deleted(self, test_db):
        from sqlalchemy import text

        from app.models.database import AuditLogRecord

        async for session in test_db():
            # Insert an audit record
            rec = AuditLogRecord(
                check_id="immut_test_123",
                action="test_action",
                actor="test_user",
                details="test_details",
            )
            session.add(rec)
            await session.commit()
            record_id = rec.id

            # 1. ORM update prohibited via SQLAlchemy event listener
            with pytest.raises(RuntimeError, match="immutable: UPDATE is strictly prohibited"):
                rec.details = "tampered"
                await session.commit()

            await session.rollback()

            # 2. ORM delete prohibited via SQLAlchemy event listener
            with pytest.raises(RuntimeError, match="immutable: DELETE is strictly prohibited"):
                await session.delete(rec)
                await session.commit()

            await session.rollback()

            # 3. Direct SQL trigger prevents UPDATE
            with pytest.raises(Exception):
                await session.execute(
                    text("UPDATE audit_log SET details = 'tampered' WHERE id = :id"),
                    {"id": record_id},
                )
                await session.commit()

            await session.rollback()

            # 4. Direct SQL trigger prevents DELETE
            with pytest.raises(Exception):
                await session.execute(
                    text("DELETE FROM audit_log WHERE id = :id"),
                    {"id": record_id},
                )
                await session.commit()
