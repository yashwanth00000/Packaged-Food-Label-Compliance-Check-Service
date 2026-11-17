"""
Pipeline Orchestrator
Runs agents in the amended order: Intake → Retrieval → Mandatory/Claims/Additives → Critic → Decision
Stores full trace per check.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.additives import AdditivesAllergenAgent
from app.agents.claims import ClaimsVerificationAgent
from app.agents.critic import CriticAgent
from app.agents.decision import DecisionAgent
from app.agents.intake import IntakeAgent
from app.agents.mandatory import MandatoryDeclarationAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.core.config import get_disclaimer
from app.core.logging import get_logger
from app.core.metrics import (
    ACTIVE_CHECKS,
    CHECK_COUNT,
    FINDING_COUNT,
    RECOMMENDATION_DISTRIBUTION,
)
from app.models.enums import Recommendation
from app.models.schemas import (
    AgentTraceEntry,
    ClaimVerification,
    ComplianceReport,
    Finding,
    LabelSubmission,
)
from app.rag.retrieval import RAGRetriever
from app.services.llm_provider import LLMService

logger = get_logger(__name__)


class CompliancePipeline:
    """
    Orchestrates the compliance checking pipeline.
    Pipeline order (per user amendment):
    1. Intake/Normalizer → 2. Retrieval → 3. Mandatory → 4. Claims → 5. Additives → 6. Critic → 7. Decision
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        retriever: RAGRetriever | None = None,
    ):
        self.llm_service = llm_service
        self.retriever = retriever or RAGRetriever()

        # Initialise agents
        self.intake = IntakeAgent(llm_service=llm_service)
        self.retrieval = RetrievalAgent(retriever=self.retriever, llm_service=llm_service)
        self.mandatory = MandatoryDeclarationAgent(llm_service=llm_service)
        self.claims = ClaimsVerificationAgent(llm_service=llm_service)
        self.additives = AdditivesAllergenAgent(llm_service=llm_service)
        self.critic = CriticAgent(retriever=self.retriever, llm_service=llm_service)
        self.decision = DecisionAgent(llm_service=llm_service)

    def run(self, submission: LabelSubmission) -> ComplianceReport:
        """
        Execute the full compliance pipeline.
        Returns a ComplianceReport.
        On any error, defaults to REGULATORY_REVIEW (fail-safe).
        """
        ACTIVE_CHECKS.inc()
        start_time = time.time()
        agent_traces: list[AgentTraceEntry] = []
        context: dict[str, Any] = {}
        has_errors = False
        all_findings: list[Finding] = []
        claim_verifications: list[ClaimVerification] = []

        # Handle photo extraction user confirmation status
        user_confirmed = True
        if submission.extraction_id:
            user_confirmed = bool(submission.user_confirmed)
        elif submission.user_confirmed is not None:
            user_confirmed = bool(submission.user_confirmed)
        context["user_confirmed"] = user_confirmed
        context["extraction_id"] = submission.extraction_id

        try:
            # --- Step 1: Intake Agent ---
            try:
                result, trace = self.intake.run(submission, context)
                agent_traces.append(trace)
                all_findings.extend(result.get("findings", []))
                context["intake"] = result
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="IntakeAgent", status="error", error_message=str(e)
                ))
                logger.error("pipeline_intake_error", error=str(e))

            # --- Step 2: Retrieval Agent ---
            try:
                result, trace = self.retrieval.run(submission, context)
                agent_traces.append(trace)
                context["retrieved_chunks"] = result.get("retrieved_chunks", {})
                context["retrieval"] = result
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="RetrievalAgent", status="error", error_message=str(e)
                ))
                context["retrieved_chunks"] = {}
                logger.error("pipeline_retrieval_error", error=str(e))

            # --- Step 3: Mandatory Declaration Agent ---
            try:
                result, trace = self.mandatory.run(submission, context)
                agent_traces.append(trace)
                all_findings.extend(result.get("findings", []))
                context["mandatory_checks_passed"] = result.get("mandatory_checks_passed", False)
            except Exception as e:
                has_errors = True
                context["mandatory_checks_passed"] = False
                agent_traces.append(AgentTraceEntry(
                    agent_name="MandatoryDeclarationAgent", status="error", error_message=str(e)
                ))
                logger.error("pipeline_mandatory_error", error=str(e))

            # --- Step 4: Claims Verification Agent ---
            try:
                result, trace = self.claims.run(submission, context)
                agent_traces.append(trace)
                all_findings.extend(result.get("findings", []))
                claim_verifications = [
                    ClaimVerification(**cv) if isinstance(cv, dict) else cv
                    for cv in result.get("claim_verifications", [])
                ]
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="ClaimsVerificationAgent", status="error", error_message=str(e)
                ))
                logger.error("pipeline_claims_error", error=str(e))

            # --- Step 5: Additives & Allergen Agent ---
            try:
                result, trace = self.additives.run(submission, context)
                agent_traces.append(trace)
                all_findings.extend(result.get("findings", []))
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="AdditivesAllergenAgent", status="error", error_message=str(e)
                ))
                logger.error("pipeline_additives_error", error=str(e))

            # --- Step 6: Critic Agent ---
            critic_context = {
                **context,
                "all_findings": [f.model_dump() for f in all_findings],
            }
            removed_count = 0
            critic_escalate = False
            critic_escalation_reasons: list[str] = []
            try:
                result, trace = self.critic.run(submission, critic_context)
                agent_traces.append(trace)
                verified_findings = result.get("findings", [])
                removed_count = result.get("removed_count", 0)
                critic_escalate = result.get("escalate_to_regulatory_review", False)
                critic_escalation_reasons = result.get("escalation_reasons", [])
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="CriticAgent", status="error", error_message=str(e)
                ))
                # On critic failure, keep all findings (fail-safe)
                verified_findings = all_findings
                logger.error("pipeline_critic_error", error=str(e))

            # --- Step 7: Decision Agent ---
            executed_agent_names = {t.agent_name for t in agent_traces}
            required_agents = {
                "IntakeAgent",
                "RetrievalAgent",
                "MandatoryDeclarationAgent",
                "ClaimsVerificationAgent",
                "AdditivesAllergenAgent",
                "CriticAgent",
            }
            skipped_agents = list(required_agents - executed_agent_names)
            errored_agents = [t.agent_name for t in agent_traces if t.status == "error"]

            decision_context = {
                "verified_findings": verified_findings,
                "claim_verifications": [
                    cv.model_dump() if hasattr(cv, 'model_dump') else cv
                    for cv in claim_verifications
                ],
                "has_errors": has_errors or bool(errored_agents),
                "removed_count": removed_count,
                "escalate_to_regulatory_review": critic_escalate or bool(skipped_agents),
                "escalation_reasons": critic_escalation_reasons + (
                    [f"Required agents skipped: {', '.join(skipped_agents)}"] if skipped_agents else []
                ),
                "mandatory_checks_passed": context.get("mandatory_checks_passed", False),
                "user_confirmed": context.get("user_confirmed", True),
                "skipped_agents": skipped_agents,
                "errored_agents": errored_agents,
            }
            try:
                result, trace = self.decision.run(submission, decision_context)
                agent_traces.append(trace)
                recommendation = result.get("recommendation", Recommendation.REGULATORY_REVIEW)
                reasons = result.get("reasons", [])
                overall_confidence = result.get("confidence", 0.0)
            except Exception as e:
                has_errors = True
                agent_traces.append(AgentTraceEntry(
                    agent_name="DecisionAgent", status="error", error_message=str(e)
                ))
                recommendation = Recommendation.REGULATORY_REVIEW
                reasons = ["Decision agent failed — defaulting to REGULATORY_REVIEW (fail-safe)."]
                overall_confidence = 0.0
                logger.error("pipeline_decision_error", error=str(e))

        except Exception as e:
            # Catastrophic pipeline failure
            has_errors = True
            recommendation = Recommendation.REGULATORY_REVIEW
            reasons = [f"Pipeline failed: {e!s}. Defaulting to REGULATORY_REVIEW (fail-safe)."]
            verified_findings = all_findings
            overall_confidence = 0.0
            logger.error("pipeline_catastrophic_failure", error=str(e))

        # --- Build Report ---
        processing_time = time.time() - start_time

        # Record metrics
        ACTIVE_CHECKS.dec()
        status = "error" if has_errors else "success"
        CHECK_COUNT.labels(status=status).inc()
        RECOMMENDATION_DISTRIBUTION.labels(recommendation=recommendation.value).inc()
        for f in verified_findings:
            sev = f.severity.value if hasattr(f, 'severity') else "unknown"
            it = f.issue_type.value if hasattr(f, 'issue_type') else "unknown"
            FINDING_COUNT.labels(severity=sev, issue_type=it).inc()

        # Ensure all findings are Finding objects
        final_findings = []
        for f in verified_findings:
            if isinstance(f, dict):
                final_findings.append(Finding(**f))
            else:
                final_findings.append(f)

        report = ComplianceReport(
            submission=submission,
            recommendation=recommendation,
            findings=final_findings,
            claim_verifications=claim_verifications,
            reasons=reasons,
            disclaimer=get_disclaimer(),
            agent_trace=agent_traces,
            overall_confidence=overall_confidence,
            processing_time_seconds=round(processing_time, 3),
            has_errors=has_errors,
        )

        logger.info(
            "pipeline_completed",
            check_id=report.check_id,
            recommendation=recommendation.value,
            findings=len(final_findings),
            processing_time=processing_time,
            has_errors=has_errors,
        )

        return report
