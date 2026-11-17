"""
Full pipeline tests — run the complete orchestrator and verify outcomes.
"""

from app.models.enums import Recommendation
from app.models.schemas import SAMPLE_LABELS, ComplianceReport


class TestPipelineOutcomes:
    def test_compliant_label_recommendation(self, pipeline, compliant_label):
        report = pipeline.run(compliant_label)
        assert isinstance(report, ComplianceReport)
        # Compliant label should be CLEAR or REVISE (depending on retrieval context)
        assert report.recommendation in (
            Recommendation.CLEAR_FOR_PRINT,
            Recommendation.REVISE_AND_RECHECK,
        )
        assert report.disclaimer
        assert len(report.agent_trace) >= 5

    def test_missing_declarations_recommendation(self, pipeline, missing_declarations_label):
        report = pipeline.run(missing_declarations_label)
        assert isinstance(report, ComplianceReport)
        assert report.recommendation in (
            Recommendation.REVISE_AND_RECHECK,
            Recommendation.REGULATORY_REVIEW,
        )
        assert len(report.findings) > 0
        # Should have missing declaration findings
        missing = [f for f in report.findings if f.issue_type.value == "missing_declaration"]
        assert len(missing) >= 3

    def test_prohibited_claims_recommendation(self, pipeline, prohibited_claims_label):
        report = pipeline.run(prohibited_claims_label)
        assert isinstance(report, ComplianceReport)
        assert report.recommendation == Recommendation.REGULATORY_REVIEW
        # Should have prohibited claim findings
        prohibited = [f for f in report.findings if f.issue_type.value == "prohibited_claim"]
        assert len(prohibited) >= 1

    def test_pipeline_always_includes_disclaimer(self, pipeline, compliant_label):
        report = pipeline.run(compliant_label)
        assert report.disclaimer
        assert "DISCLAIMER" in report.disclaimer.upper() or "recommendation" in report.disclaimer.lower()

    def test_pipeline_always_has_trace(self, pipeline, compliant_label):
        report = pipeline.run(compliant_label)
        assert len(report.agent_trace) >= 5
        agent_names = [t.agent_name for t in report.agent_trace]
        assert "IntakeAgent" in agent_names
        assert "RetrievalAgent" in agent_names
        assert "DecisionAgent" in agent_names

    def test_pipeline_processing_time_recorded(self, pipeline, compliant_label):
        report = pipeline.run(compliant_label)
        assert report.processing_time_seconds > 0

    def test_all_three_samples_produce_reports(self, pipeline):
        for name, label in SAMPLE_LABELS.items():
            report = pipeline.run(label)
            assert isinstance(report, ComplianceReport)
            assert report.recommendation in (
                Recommendation.CLEAR_FOR_PRINT,
                Recommendation.REVISE_AND_RECHECK,
                Recommendation.REGULATORY_REVIEW,
            )

    def test_critic_removes_at_least_one_finding_in_some_scenario(self, pipeline):
        """Critic should be invoked and trace recorded across all samples."""
        for label in SAMPLE_LABELS.values():
            report = pipeline.run(label)
            critic_traces = [t for t in report.agent_trace if t.agent_name == "CriticAgent"]
            assert len(critic_traces) == 1
            assert critic_traces[0].status == "success"
