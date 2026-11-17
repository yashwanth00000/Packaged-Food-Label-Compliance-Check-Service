"""
Decision policy tests — ensure the deterministic policy works correctly.
"""

import pytest

from app.agents.decision import DecisionAgent
from app.models.enums import IssueType, Recommendation, Severity
from app.models.schemas import SAMPLE_LABELS, Finding
from app.services.llm_provider import LLMService, MockLLM


@pytest.fixture
def agent():
    return DecisionAgent(llm_service=LLMService(provider=MockLLM()))


class TestDecisionPolicy:
    """Tests for the decision policy's deterministic rules."""

    def test_no_findings_no_errors_is_clear(self, agent):
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"verified_findings": [], "claim_verifications": [], "has_errors": False, "removed_count": 0},
        )
        assert result["recommendation"] == Recommendation.CLEAR_FOR_PRINT

    def test_error_always_regulatory_review(self, agent):
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"verified_findings": [], "claim_verifications": [], "has_errors": True, "removed_count": 0},
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_security_finding_triggers_review(self, agent):
        security_finding = Finding(
            field="security",
            issue_type=IssueType.REGULATORY_VIOLATION,
            severity=Severity.CRITICAL,
            description="Prompt injection detected",
            confidence=1.0,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {
                "verified_findings": [security_finding],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_prohibited_claim_triggers_review(self, agent):
        finding = Finding(
            field="claims",
            issue_type=IssueType.PROHIBITED_CLAIM,
            severity=Severity.CRITICAL,
            description="Prohibited claim",
            confidence=0.95,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["prohibited_claims"],
            {
                "verified_findings": [finding],
                "claim_verifications": [{"status": "PROHIBITED"}],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_allergen_mismatch_triggers_review(self, agent):
        finding = Finding(
            field="allergen_declaration",
            issue_type=IssueType.ALLERGEN_MISMATCH,
            severity=Severity.CRITICAL,
            description="Allergen mismatch",
            confidence=1.0,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["missing_declarations"],
            {
                "verified_findings": [finding],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_major_findings_yield_revise(self, agent):
        finding = Finding(
            field="storage_instructions",
            issue_type=IssueType.MISSING_DECLARATION,
            severity=Severity.MAJOR,
            description="Missing storage instructions",
            confidence=1.0,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["missing_declarations"],
            {
                "verified_findings": [finding],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        assert result["recommendation"] == Recommendation.REVISE_AND_RECHECK

    def test_never_clear_on_error(self, agent):
        """The system must NEVER return CLEAR_FOR_PRINT when there are errors."""
        for _ in range(10):
            result, _ = agent.run(
                SAMPLE_LABELS["compliant_biscuit"],
                {
                    "verified_findings": [],
                    "claim_verifications": [],
                    "has_errors": True,
                    "removed_count": 0,
                },
            )
            assert result["recommendation"] != Recommendation.CLEAR_FOR_PRINT

    def test_disclaimer_always_present(self, agent):
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"verified_findings": [], "claim_verifications": [], "has_errors": False, "removed_count": 0},
        )
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_high_critic_removals_triggers_review(self, agent):
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {
                "verified_findings": [],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 10,
            },
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_quality_warnings_block_clear_for_print_when_unconfirmed(self, agent):
        """Item 3: Quality gate must block CLEAR_FOR_PRINT when quality warnings exist and user has not confirmed."""
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {
                "verified_findings": [],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
                "image_quality_warnings": ["Photo appears blurry or out of focus. Please retake photo."],
                "user_confirmed": False,
                "mandatory_checks_passed": True,
            },
        )
        assert result["recommendation"] == Recommendation.REVISE_AND_RECHECK
        assert any("Quality gate" in r or "Blurry" in r for r in result["reasons"])

    def test_quality_warnings_cleared_after_user_confirmation(self, agent):
        """Item 3: When user visually verifies and confirms fields, CLEAR_FOR_PRINT can be granted."""
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {
                "verified_findings": [],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
                "image_quality_warnings": ["Photo appears blurry or out of focus. Please retake photo."],
                "user_confirmed": True,
                "mandatory_checks_passed": True,
            },
        )
        assert result["recommendation"] == Recommendation.CLEAR_FOR_PRINT
