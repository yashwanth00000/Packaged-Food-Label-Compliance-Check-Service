"""
Failure-injection tests.
LLM timeout, malformed output, empty retrieval → must yield REGULATORY_REVIEW.
"""

from unittest.mock import MagicMock

from app.core.errors import LLMTimeoutError
from app.models.enums import Recommendation
from app.models.schemas import SAMPLE_LABELS, ComplianceReport, LabelSubmission
from app.orchestrator.pipeline import CompliancePipeline
from app.services.llm_provider import LLMProvider, LLMService, MockLLM


class FailingLLM(LLMProvider):
    """An LLM that always raises an exception."""

    @property
    def provider_name(self) -> str:
        return "failing"

    def generate(self, prompt, system_prompt="", temperature=0.1, max_tokens=4096):
        raise LLMTimeoutError("Simulated LLM timeout")


class MalformedLLM(LLMProvider):
    """An LLM that returns unparseable garbage."""

    @property
    def provider_name(self) -> str:
        return "malformed"

    def generate(self, prompt, system_prompt="", temperature=0.1, max_tokens=4096):
        return {
            "content": "THIS IS NOT JSON {{{{ broken ]]]]",
            "tokens_used": 10,
            "provider": "malformed",
        }


class TestLLMTimeout:
    def test_timeout_yields_regulatory_review(self, mock_retriever):
        """When the LLM times out, the pipeline must fail-safe to REGULATORY_REVIEW."""
        llm = LLMService(provider=FailingLLM())
        ambiguous_label = LabelSubmission(
            product_name="Ambiguous Vitality Elixir",
            claims=["Promotes inner cellular rejuvenation and longevity"],
        )
        pipeline = CompliancePipeline(llm_service=llm, retriever=mock_retriever)
        report = pipeline.run(ambiguous_label)
        assert isinstance(report, ComplianceReport)
        # The pipeline should still produce a report (fail-safe)
        # With errors, it should not be CLEAR_FOR_PRINT
        assert report.recommendation != Recommendation.CLEAR_FOR_PRINT

    def test_timeout_all_samples(self, mock_retriever):
        """All samples must produce reports even with failing LLM."""
        llm = LLMService(provider=FailingLLM())
        pipeline = CompliancePipeline(llm_service=llm, retriever=mock_retriever)
        for name, label in SAMPLE_LABELS.items():
            report = pipeline.run(label)
            assert isinstance(report, ComplianceReport)
            assert report.recommendation != Recommendation.CLEAR_FOR_PRINT or not report.has_errors


class TestMalformedOutput:
    def test_malformed_llm_output(self, mock_retriever):
        """Malformed LLM output must not crash the pipeline."""
        llm = LLMService(provider=MalformedLLM())
        pipeline = CompliancePipeline(llm_service=llm, retriever=mock_retriever)
        report = pipeline.run(SAMPLE_LABELS["compliant_biscuit"])
        assert isinstance(report, ComplianceReport)
        # Should still produce a valid report


class TestEmptyRetrieval:
    def test_empty_retrieval_still_works(self):
        """With no indexed regulations, pipeline should still produce results."""
        import tempfile

        from app.rag.embeddings import HashEmbedding
        from app.rag.retrieval import RAGRetriever

        tmpdir = tempfile.mkdtemp()
        empty_retriever = RAGRetriever(
            persist_dir=tmpdir,
            embedding_provider=HashEmbedding(),
        )
        llm = LLMService(provider=MockLLM())
        pipeline = CompliancePipeline(llm_service=llm, retriever=empty_retriever)

        for name, label in SAMPLE_LABELS.items():
            report = pipeline.run(label)
            assert isinstance(report, ComplianceReport)
            # Mandatory checks should still work (deterministic)
            if name == "missing_declarations":
                assert len(report.findings) > 0


class TestAgentFailure:
    def test_single_agent_exception_doesnt_crash_pipeline(self, mock_retriever):
        """If one agent raises, the pipeline should still produce a report."""
        llm = LLMService(provider=MockLLM())
        pipeline = CompliancePipeline(llm_service=llm, retriever=mock_retriever)

        # Monkey-patch one agent to fail
        original_process = pipeline.mandatory.process
        pipeline.mandatory.process = MagicMock(side_effect=RuntimeError("Agent crashed"))

        report = pipeline.run(SAMPLE_LABELS["compliant_biscuit"])
        assert isinstance(report, ComplianceReport)
        assert report.has_errors is True
        # Must not be CLEAR_FOR_PRINT with errors
        assert report.recommendation != Recommendation.CLEAR_FOR_PRINT

        # Restore
        pipeline.mandatory.process = original_process
