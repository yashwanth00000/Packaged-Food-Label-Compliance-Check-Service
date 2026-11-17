"""
Critic / Verifier Agent (Step 6)
Independently re-checks findings. Discards findings without evidence.
Flags contradictions and lowers confidence for weak support.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger
from app.core.metrics import CRITIC_REMOVALS
from app.models.enums import Severity
from app.models.schemas import Finding, LabelSubmission

logger = get_logger(__name__)


class CriticAgent(BaseAgent):
    """
    Step 6: Verify each finding independently.
    Removes findings without evidence, flags contradictions,
    and adjusts confidence scores.
    """

    def __init__(
        self,
        retriever: Any = None,
        llm_service: Any = None,
    ):
        super().__init__(llm_service=llm_service)
        self.retriever = retriever

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        all_findings: list[dict] = context.get("all_findings", [])
        retriever = self.retriever or context.get("retriever")

        verified: list[Finding] = []
        removed_ids: list[str] = []
        discarded_findings: list[dict[str, Any]] = []
        escalation_reasons: list[str] = []
        escalate_to_review = False
        critic_disagreement = False

        for f_dict in all_findings:
            finding = Finding(**f_dict) if isinstance(f_dict, dict) else f_dict

            # --- Rule 1: Keep deterministic findings (confidence = 1.0) ---
            if finding.confidence >= 0.99:
                finding.verified_by_critic = True
                verified.append(finding)
                continue

            # --- Rule 2: Remove findings without any evidence ---
            has_evidence = bool(finding.evidence_quote) or bool(finding.regulation_name)
            if not has_evidence and finding.confidence < 0.7:
                removed_ids.append(finding.finding_id)
                discard_reason = "no_evidence_low_confidence"
                CRITIC_REMOVALS.inc()
                logger.info(
                    "critic_discarded_finding",
                    finding_id=finding.finding_id,
                    reason=discard_reason,
                    field=finding.field,
                    severity=finding.severity.value,
                    agent=finding.agent,
                )
                discarded_findings.append({
                    "finding_id": finding.finding_id,
                    "field": finding.field,
                    "severity": finding.severity,
                    "reason": discard_reason,
                    "agent": finding.agent,
                })
                # If Critic discards a CRITICAL or MAJOR finding, escalate!
                if finding.severity in (Severity.CRITICAL, Severity.MAJOR):
                    escalate_to_review = True
                    escalation_reasons.append(
                        f"Critic discarded {finding.severity.value} finding '{finding.field}' "
                        f"from {finding.agent or 'domain agent'} due to lack of evidence ({discard_reason})."
                    )
                continue

            # --- Rule 3: Verify chunk_id exists if provided ---
            if finding.chunk_id and retriever is not None:
                if not retriever.chunk_exists(finding.chunk_id):
                    # Chunk doesn't exist — lower confidence significantly
                    finding.confidence = max(0.2, finding.confidence - 0.3)
                    critic_disagreement = True
                    logger.warning(
                        "critic_chunk_not_found",
                        finding_id=finding.finding_id,
                        chunk_id=finding.chunk_id,
                    )
                    if finding.severity in (Severity.CRITICAL, Severity.MAJOR):
                        escalate_to_review = True
                        escalation_reasons.append(
                            f"Critic found non-existent chunk citation {finding.chunk_id} for "
                            f"{finding.severity.value} finding '{finding.field}'."
                        )

                # Validate citation: evidence_quote should be substring of chunk
                if finding.evidence_quote and finding.chunk_id:
                    if not retriever.is_valid_citation(finding.chunk_id, finding.evidence_quote):
                        finding.confidence = max(0.3, finding.confidence - 0.2)
                        critic_disagreement = True
                        logger.warning(
                            "critic_invalid_citation",
                            finding_id=finding.finding_id,
                            reason="evidence_quote_not_in_chunk",
                        )
                        if finding.severity in (Severity.CRITICAL, Severity.MAJOR):
                            escalate_to_review = True
                            escalation_reasons.append(
                                f"Critic found invalid citation quote for {finding.severity.value} finding '{finding.field}'."
                            )

            # --- Rule 4: Check for contradictions with other findings ---
            for other_dict in all_findings:
                other = Finding(**other_dict) if isinstance(other_dict, dict) else other_dict
                if other.finding_id == finding.finding_id:
                    continue
                if (other.field == finding.field and
                    other.agent != finding.agent and
                    other.confidence > finding.confidence + 0.3):
                    # Another agent is more confident about the same field
                    finding.confidence = max(0.2, finding.confidence - 0.15)
                    critic_disagreement = True
                    if finding.severity in (Severity.CRITICAL, Severity.MAJOR):
                        escalate_to_review = True
                        escalation_reasons.append(
                            f"Critic detected disagreement between {finding.agent} and {other.agent} "
                            f"on {finding.severity.value} field '{finding.field}'."
                        )

            # --- Rule 5: Remove very low confidence findings ---
            if finding.confidence < 0.15:
                removed_ids.append(finding.finding_id)
                discard_reason = f"very_low_confidence_{finding.confidence:.2f}"
                CRITIC_REMOVALS.inc()
                logger.info(
                    "critic_discarded_finding",
                    finding_id=finding.finding_id,
                    reason=discard_reason,
                    severity=finding.severity.value,
                    field=finding.field,
                    agent=finding.agent,
                )
                discarded_findings.append({
                    "finding_id": finding.finding_id,
                    "field": finding.field,
                    "severity": finding.severity,
                    "reason": discard_reason,
                    "agent": finding.agent,
                })
                if finding.severity in (Severity.CRITICAL, Severity.MAJOR):
                    escalate_to_review = True
                    escalation_reasons.append(
                        f"Critic discarded low-confidence {finding.severity.value} finding '{finding.field}'."
                    )
                continue

            finding.verified_by_critic = True
            verified.append(finding)

        logger.info(
            "critic_completed",
            total_input=len(all_findings),
            verified=len(verified),
            removed=len(removed_ids),
            escalate_to_review=escalate_to_review,
        )

        return {
            "findings": verified,
            "removed_finding_ids": removed_ids,
            "discarded_findings": discarded_findings,
            "escalate_to_regulatory_review": escalate_to_review,
            "escalation_reasons": escalation_reasons,
            "critic_disagreement": critic_disagreement,
            "confidence": 0.85,
            "verified_count": len(verified),
            "removed_count": len(removed_ids),
        }
