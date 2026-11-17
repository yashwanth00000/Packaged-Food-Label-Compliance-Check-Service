"""Production must not issue a positive recommendation from sample-only rules."""

from app.agents.decision import DecisionAgent
from app.core.config import get_settings
from app.models.enums import Recommendation
from app.models.schemas import SAMPLE_LABELS


def test_sample_only_or_missing_regulatory_sources_force_review(monkeypatch):
    monkeypatch.setenv("REGULATORY_REQUIRE_AUTHORITATIVE_CORPUS", "true")
    get_settings.cache_clear()
    try:
        context = {"retrieved_chunks": {"mandatory": [{"chunk_id": "sample-1", "is_sample": True}]}}
        result = DecisionAgent().process(SAMPLE_LABELS["compliant_biscuit"], context)
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW
        assert "sample/demo" in result["reasons"][0].lower()
    finally:
        get_settings.cache_clear()


def test_missing_retrieval_also_forces_review(monkeypatch):
    monkeypatch.setenv("REGULATORY_REQUIRE_AUTHORITATIVE_CORPUS", "true")
    get_settings.cache_clear()
    try:
        result = DecisionAgent().process(SAMPLE_LABELS["compliant_biscuit"], {})
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW
    finally:
        get_settings.cache_clear()
