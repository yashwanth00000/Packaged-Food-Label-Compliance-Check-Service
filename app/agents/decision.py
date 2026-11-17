"""
Decision Agent (Step 7)
Fully DETERMINISTIC rule-based policy. No LLM.
Reads thresholds from config/decision_policy.yaml.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.config import get_decision_policy, get_disclaimer, get_settings
from app.core.logging import get_logger
from app.models.enums import (
    ClaimStatus,
    IssueType,
    Recommendation,
    Severity,
)
from app.models.schemas import Finding, LabelSubmission

logger = get_logger(__name__)


class DecisionAgent(BaseAgent):
    """
    Step 7: Combine verified findings into a recommendation.
    FULLY DETERMINISTIC — uses YAML policy, never an LLM.
    Fail-safe: errors always lead to REGULATORY_REVIEW.
    """

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return self._evaluate(submission, context)
        except Exception as e:
            logger.error("decision_agent_error", error=str(e))
            # FAIL SAFE: errors → REGULATORY_REVIEW
            return {
                "recommendation": Recommendation.REGULATORY_REVIEW,
                "reasons": [f"Decision agent encountered an error: {e!s}. Defaulting to REGULATORY_REVIEW (fail-safe)."],
                "confidence": 0.0,
                "disclaimer": get_disclaimer(),
                "findings": context.get("verified_findings", []),
            }

    def _evaluate(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        policy = get_decision_policy()
        dp = policy.get("decision_policy", {})
        clear_rules = dp.get("clear_for_print", {})
        review_rules = dp.get("regulatory_review", {})

        verified_findings: list[Finding] = context.get("verified_findings", [])
        claim_verifications: list[dict] = context.get("claim_verifications", [])
        has_errors: bool = context.get("has_errors", False)
        removed_count: int = context.get("removed_count", 0)

        reasons: list[str] = []

        # A sample/demo corpus must never support a positive compliance recommendation.
        if get_settings().regulatory_require_authoritative_corpus:
            retrieved = context.get("retrieved_chunks", {}) or {}
            missing_source_groups = [
                key for key, group in retrieved.items()
                if not group or not any(not bool(chunk.get("is_sample", True)) for chunk in group)
            ]
            if not retrieved or missing_source_groups:
                detail = ", ".join(missing_source_groups[:6]) if missing_source_groups else "no regulatory queries"
                return self._result(
                    Recommendation.REGULATORY_REVIEW,
                    [f"Authoritative FSSAI evidence is missing for one or more required checks ({detail}). Sample/demo extracts cannot support a positive compliance recommendation."],
                    verified_findings,
                    0.0,
                )

        # --- Classify findings by severity ---
        critical = [f for f in verified_findings if f.severity == Severity.CRITICAL]
        major = [f for f in verified_findings if f.severity == Severity.MAJOR]
        minor = [f for f in verified_findings if f.severity == Severity.MINOR]

        # --- Missing mandatory fields ---
        missing_mandatory = [
            f for f in verified_findings
            if f.issue_type == IssueType.MISSING_DECLARATION
        ]

        # --- Prohibited claims ---
        prohibited_claims = [
            f for f in verified_findings
            if f.issue_type == IssueType.PROHIBITED_CLAIM
        ]

        # --- Unsupported claims ---
        unsupported_claims = [
            f for f in verified_findings
            if f.issue_type == IssueType.UNSUPPORTED_CLAIM
        ]

        # --- Claim verification statuses ---
        claim_statuses = [cv.get("status") for cv in claim_verifications]
        has_prohibited_claim = ClaimStatus.PROHIBITED in claim_statuses or len(prohibited_claims) > 0
        all_claims_supported = all(
            s in (ClaimStatus.SUPPORTED, "SUPPORTED") for s in claim_statuses
        ) if claim_statuses else True

        # --- Calculate overall confidence ---
        if verified_findings:
            weights = policy.get("severity_weights", {})
            weighted_conf = 0.0
            total_weight = 0.0
            for f in verified_findings:
                w = weights.get(f.severity.value, 0.5)
                weighted_conf += f.confidence * w
                total_weight += w
            overall_confidence = weighted_conf / total_weight if total_weight > 0 else 0.5
        else:
            overall_confidence = 0.95

        # --- Allergen mismatches ---
        allergen_mismatches = [
            f for f in verified_findings
            if f.issue_type == IssueType.ALLERGEN_MISMATCH
        ]

        # --- Security findings (injection attempts) ---
        security_findings = [
            f for f in verified_findings
            if f.field == "security"
        ]

        # =====================================================================
        # DECISION LOGIC (deterministic, fail-safe)
        # =====================================================================

        # --- REGULATORY_REVIEW triggers (any one is enough) ---
        if context.get("escalate_to_regulatory_review"):
            escalation_reasons = context.get("escalation_reasons", [])
            for er in escalation_reasons:
                reasons.append(f"Critic escalation: {er}")
            if not reasons:
                reasons.append("Critic escalated to REGULATORY_REVIEW due to discarded critical/major findings or agent disagreement.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        if has_errors:
            reasons.append("System error detected — fail-safe to REGULATORY_REVIEW.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        skipped_agents = context.get("skipped_agents", [])
        if skipped_agents:
            reasons.append(f"Pipeline integrity compromised: Required agents skipped: {', '.join(skipped_agents)}.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        if security_findings:
            reasons.append("Security concern: potential prompt injection detected.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        if has_prohibited_claim:
            reasons.append(f"Prohibited claim(s) detected: {len(prohibited_claims)} finding(s).")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        # Critical findings (excluding missing mandatory declarations which are fixable via REVISE_AND_RECHECK)
        critical_violations = [f for f in critical if f.issue_type != IssueType.MISSING_DECLARATION]
        if critical_violations and len(critical_violations) > 3:
            reasons.append(f"Multiple critical findings ({len(critical_violations)}) require regulatory review.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        if allergen_mismatches:
            reasons.append(f"Allergen declaration inconsistency: {len(allergen_mismatches)} mismatch(es) — food safety concern.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        # Veg/Non-Veg Mark Inconsistency (Critical statutory violation)
        veg_nonveg_inconsistencies = [
            f for f in verified_findings
            if f.field == "veg_nonveg_mark" and f.severity == Severity.CRITICAL
        ]
        if veg_nonveg_inconsistencies:
            reasons.append(f"Vegetarian/Non-vegetarian mark violation ({len(veg_nonveg_inconsistencies)} critical finding(s)) — misbranding concern.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        # Prohibited Additives / Colours
        prohibited_additives = [
            f for f in verified_findings
            if f.field == "additives" and f.issue_type == IssueType.REGULATORY_VIOLATION and f.severity == Severity.CRITICAL
        ]
        if prohibited_additives:
            reasons.append(f"Prohibited food additives/colours detected ({len(prohibited_additives)} critical finding(s)) — regulatory safety concern.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        threshold = review_rules.get("overall_confidence_below", 0.50)
        if overall_confidence < threshold:
            reasons.append(f"Overall confidence ({overall_confidence:.2f}) below threshold ({threshold}).")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        if removed_count > review_rules.get("critic_override_count_above", 3):
            reasons.append(f"Critic removed {removed_count} findings — high disagreement suggests review needed.")
            return self._result(Recommendation.REGULATORY_REVIEW, reasons, verified_findings, overall_confidence)

        # --- User Confirmation & Mandatory Affirmative Pass Checks ---
        quality_warnings = context.get("image_quality_warnings", [])
        has_quality_warnings = bool(quality_warnings) or context.get("has_image_quality_warnings", False)
        user_confirmed = context.get("user_confirmed", True)
        has_low_conf_unconfirmed = context.get("has_low_confidence_unconfirmed_fields", False) or bool(context.get("unconfirmed_low_confidence_fields"))
        # Affirmative pass requires mandatory agent ran and passed with 0 missing mandatory declarations
        mandatory_passed = context.get("mandatory_checks_passed", len(missing_mandatory) == 0 and "MandatoryDeclarationAgent" not in skipped_agents)

        # Quality gate: if image quality warnings exist, CLEAR_FOR_PRINT is strictly blocked unless user confirmed
        quality_gate_blocked = has_quality_warnings and not user_confirmed

        # --- CLEAR_FOR_PRINT conditions (ALL must be true) ---
        min_conf = clear_rules.get("min_overall_confidence", 0.80)
        is_clear = (
            len(critical) == 0
            and len(major) == 0
            and len(missing_mandatory) == 0
            and mandatory_passed is True
            and user_confirmed is True
            and not quality_gate_blocked
            and not has_low_conf_unconfirmed
            and all_claims_supported
            and overall_confidence >= min_conf
            and not has_errors
            and all(f.verified_by_critic for f in verified_findings)
        )

        if is_clear:
            if verified_findings:
                # Only minor/info findings remain
                reasons.append(f"No critical or major findings. {len(minor)} minor finding(s). All claims supported.")
            else:
                reasons.append("No compliance issues detected. All mandatory declarations present. All claims supported.")
            return self._result(Recommendation.CLEAR_FOR_PRINT, reasons, verified_findings, overall_confidence)

        # --- REVISE_AND_RECHECK (everything else) ---
        if has_quality_warnings and not user_confirmed:
            reasons.append(
                "Quality gate: Blurry, dark, or cropped image detected. "
                "CLEAR_FOR_PRINT is blocked until the operator confirms all affected fields or uploads a better photo."
            )
        elif not user_confirmed:
            reasons.append("Label extraction details have not been confirmed by the user against the physical label.")
        if has_low_conf_unconfirmed:
            reasons.append("One or more mandatory label fields have low extraction confidence (< 0.8) and were not verified.")
        if not mandatory_passed:
            reasons.append("Mandatory declaration check did not receive an affirmative PASS.")
        if missing_mandatory:
            reasons.append(f"{len(missing_mandatory)} missing mandatory declaration(s).")
        if unsupported_claims:
            reasons.append(f"{len(unsupported_claims)} unsupported claim(s).")
        if major:
            reasons.append(f"{len(major)} major finding(s) requiring attention.")
        if critical:
            reasons.append(f"{len(critical)} critical finding(s) requiring correction.")
        if not reasons:
            reasons.append("Some issues detected that need correction before approval.")

        return self._result(Recommendation.REVISE_AND_RECHECK, reasons, verified_findings, overall_confidence)

    def _result(
        self,
        recommendation: Recommendation,
        reasons: list[str],
        findings: list[Finding],
        confidence: float,
    ) -> dict[str, Any]:
        logger.info(
            "decision_made",
            recommendation=recommendation.value,
            reasons=reasons,
            confidence=confidence,
            finding_count=len(findings),
        )
        return {
            "recommendation": recommendation,
            "reasons": reasons,
            "confidence": confidence,
            "disclaimer": get_disclaimer(),
            "findings": findings,
        }
