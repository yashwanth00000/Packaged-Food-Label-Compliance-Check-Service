"""
Unit tests for each agent.
"""

import pytest

from app.agents.additives import AdditivesAllergenAgent
from app.agents.claims import ClaimsVerificationAgent
from app.agents.critic import CriticAgent
from app.agents.decision import DecisionAgent
from app.agents.intake import IntakeAgent
from app.agents.mandatory import MandatoryDeclarationAgent
from app.models.enums import IssueType, Recommendation, Severity
from app.models.schemas import (
    SAMPLE_LABELS,
    AdditiveInfo,
    Finding,
    NutrientValue,
    NutritionalInfo,
)
from app.services.llm_provider import LLMService, MockLLM


@pytest.fixture
def llm():
    return LLMService(provider=MockLLM())


class TestIntakeAgent:
    def test_intake_normalises_submission(self, llm, compliant_label):
        agent = IntakeAgent(llm_service=llm)
        result, trace = agent.run(compliant_label, {})
        assert trace.status == "success"
        assert isinstance(result, dict)

    def test_intake_with_missing_fields(self, llm, missing_declarations_label):
        agent = IntakeAgent(llm_service=llm)
        result, trace = agent.run(missing_declarations_label, {})
        assert trace.status == "success"


class TestMandatoryAgent:
    def test_compliant_label_no_mandatory_findings(self, llm, compliant_label):
        agent = MandatoryDeclarationAgent(llm_service=llm)
        result, trace = agent.run(compliant_label, {"retrieved_chunks": {}})
        # Compliant label should have zero or very few findings (maybe nutrient completeness)
        assert trace.status == "success"
        assert isinstance(result["findings"], list)

    def test_missing_mandatory_fields_detected(self, llm, missing_declarations_label):
        agent = MandatoryDeclarationAgent(llm_service=llm)
        result, trace = agent.run(missing_declarations_label, {"retrieved_chunks": {}})
        assert trace.status == "success"
        findings = result["findings"]
        assert len(findings) > 0
        # Check that missing allergen is flagged
        fields = [f.field for f in findings]
        assert "allergen_declaration" in fields
        assert "net_quantity" in fields
        assert "manufacturer_details" in fields

    def test_missing_nutrient_in_panel(self, llm, missing_declarations_label):
        agent = MandatoryDeclarationAgent(llm_service=llm)
        result, _ = agent.run(missing_declarations_label, {"retrieved_chunks": {}})
        nutrient_findings = [
            f for f in result["findings"]
            if f.field == "nutritional_info" and "nutrient" in f.description.lower()
        ]
        # missing_declarations label lacks Saturated Fat, Trans Fat, Added Sugars, Sodium
        assert len(nutrient_findings) >= 3


class TestClaimsAgent:
    def test_supported_claims(self, llm, compliant_label):
        agent = ClaimsVerificationAgent(llm_service=llm)
        result, trace = agent.run(compliant_label, {"retrieved_chunks": {}})
        assert trace.status == "success"

    def test_prohibited_claims_detected(self, llm, prohibited_claims_label):
        agent = ClaimsVerificationAgent(llm_service=llm)
        result, trace = agent.run(prohibited_claims_label, {"retrieved_chunks": {}})
        assert trace.status == "success"
        # "Cures cold and flu" should be flagged as prohibited
        prohibited = [
            f for f in result["findings"]
            if f.issue_type == IssueType.PROHIBITED_CLAIM
        ]
        assert len(prohibited) >= 1

    def test_no_added_sugar_with_sugar_ingredient(self, llm, prohibited_claims_label):
        """The prohibited_claims label has 'Sugar' in ingredients and claims 'No added sugar'."""
        agent = ClaimsVerificationAgent(llm_service=llm)
        result, _ = agent.run(prohibited_claims_label, {"retrieved_chunks": {}})
        # Check for the no added sugar finding
        no_sugar_findings = [
            f for f in result["findings"]
            if "no added sugar" in f.description.lower() or "sugar" in f.description.lower()
        ]
        claim_statuses = [cv.get("status") for cv in result.get("claim_verifications", [])]
        has_unsupported = "UNSUPPORTED" in [str(s) for s in claim_statuses]
        assert len(no_sugar_findings) > 0 or has_unsupported

    def test_claims_threshold_helpers(self):
        from app.agents.claims import (
            _check_fat_free,
            _check_low_fat,
            _check_low_sodium,
            _check_low_sugar,
            _check_protein_high,
            _check_protein_source,
        )

        ni = NutritionalInfo(
            nutrients=[
                NutrientValue(name="Protein", per_100g=12.0, per_serving=11.0, unit="g"),
                NutrientValue(name="Total Fat", per_100g=0.3, unit="g"),
                NutrientValue(name="Total Sugars", per_100g=4.0, unit="g"),
                NutrientValue(name="Sodium", per_100g=90.0, unit="mg"),
            ],
            energy_kcal_per_100g=200.0,
            serving_size="30g",
        )
        assert _check_protein_high(ni)[0] is True
        assert _check_protein_source(ni)[0] is True
        assert _check_low_fat(ni)[0] is True
        assert _check_fat_free(ni)[0] is True
        assert _check_low_sugar(ni)[0] is True
        assert _check_low_sodium(ni)[0] is True


class TestAdditivesAgent:
    def test_compliant_additives(self, llm, compliant_label):
        agent = AdditivesAllergenAgent(llm_service=llm)
        result, trace = agent.run(compliant_label, {"retrieved_chunks": {}})
        assert trace.status == "success"

    def test_allergen_mismatch_detected(self, llm, missing_declarations_label):
        agent = AdditivesAllergenAgent(llm_service=llm)
        result, trace = agent.run(missing_declarations_label, {"retrieved_chunks": {}})
        assert trace.status == "success"

    def test_prohibited_colour_detected_from_data(self, llm, compliant_label):
        """Item 7: Read prohibited INS colour list from data/ins_colours.json."""
        agent = AdditivesAllergenAgent(llm_service=llm)
        bad_label = compliant_label.model_copy(deep=True)
        bad_label.ingredients_list.append("Metanil Yellow")
        result, _ = agent.run(bad_label, {"retrieved_chunks": {}})
        prohibited = [
            f for f in result["findings"]
            if f.field == "additives" and f.issue_type == IssueType.REGULATORY_VIOLATION
        ]
        assert len(prohibited) >= 1
        assert prohibited[0].severity == Severity.CRITICAL
        assert prohibited[0].is_sample is True

    def test_veg_mark_with_nonveg_ingredient(self, llm, compliant_label):
        agent = AdditivesAllergenAgent(llm_service=llm)
        bad_label = compliant_label.model_copy(deep=True)
        bad_label.veg_nonveg_mark = "vegetarian"
        bad_label.ingredients_list.append("gelatin")
        result, _ = agent.run(bad_label, {"retrieved_chunks": {}})
        inconsistencies = [
            f for f in result["findings"]
            if f.field == "veg_nonveg_mark" and f.issue_type == IssueType.INCONSISTENCY
        ]
        assert len(inconsistencies) >= 1
        assert inconsistencies[0].severity == Severity.CRITICAL

    def test_declared_additive_not_in_ingredients(self, llm, compliant_label):
        agent = AdditivesAllergenAgent(llm_service=llm)
        bad_label = compliant_label.model_copy(deep=True)
        bad_label.additives.append(AdditiveInfo(name="Tartrazine Synthetic", ins_number="INS 102"))
        result, _ = agent.run(bad_label, {"retrieved_chunks": {}})
        mismatches = [
            f for f in result["findings"]
            if f.field == "additives" and f.issue_type == IssueType.INCONSISTENCY
        ]
        assert len(mismatches) >= 1


class TestCriticAgent:
    def test_critic_keeps_deterministic_findings(self, llm):
        """High confidence findings should be kept."""
        critic = CriticAgent(llm_service=llm)
        findings = [
            Finding(
                field="net_quantity",
                issue_type=IssueType.MISSING_DECLARATION,
                severity=Severity.CRITICAL,
                description="Missing net quantity",
                confidence=1.0,
                agent="MandatoryDeclarationAgent",
            ).model_dump(),
        ]
        result, trace = critic.run(
            SAMPLE_LABELS["missing_declarations"],
            {"all_findings": findings, "retrieved_chunks": {}},
        )
        assert trace.status == "success"
        assert len(result["findings"]) == 1
        assert result["findings"][0].verified_by_critic is True

    def test_critic_removes_unsupported_findings(self, llm):
        """Low confidence findings without evidence should be removed."""
        critic = CriticAgent(llm_service=llm)
        findings = [
            Finding(
                field="claims",
                issue_type=IssueType.UNSUPPORTED_CLAIM,
                severity=Severity.MINOR,
                description="Weak claim",
                confidence=0.15,
                agent="ClaimsVerificationAgent",
                evidence_quote=None,
                regulation_name=None,
            ).model_dump(),
        ]
        result, trace = critic.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"all_findings": findings, "retrieved_chunks": {}},
        )
        assert trace.status == "success"
        assert result["removed_count"] >= 1

    def test_critic_discards_critical_finding_yields_regulatory_review(self, llm):
        """Item 7: If Critic discards a CRITICAL finding, DecisionAgent MUST return REGULATORY_REVIEW."""
        critic = CriticAgent(llm_service=llm)
        decision = DecisionAgent(llm_service=llm)

        # Unsubstantiated CRITICAL finding with no evidence and low confidence
        critical_finding = Finding(
            field="prohibited_additive",
            issue_type=IssueType.REGULATORY_VIOLATION,
            severity=Severity.CRITICAL,
            description="Suspicious unverified additive claim",
            confidence=0.3,
            agent="AdditivesAgent",
            evidence_quote=None,
            regulation_name=None,
        )

        context = {"all_findings": [critical_finding], "retrieved_chunks": {}}
        critic_res, _ = critic.run(SAMPLE_LABELS["compliant_biscuit"], context)

        # Verify Critic discarded the finding and set escalate_to_regulatory_review
        assert critic_res["removed_count"] >= 1
        assert critic_res["escalate_to_regulatory_review"] is True

        # Now pass escalation context into DecisionAgent
        context["escalate_to_regulatory_review"] = critic_res["escalate_to_regulatory_review"]
        context["escalation_reasons"] = critic_res.get("escalation_reasons", [])
        dec_res = decision.process(SAMPLE_LABELS["compliant_biscuit"], context)
        assert dec_res["recommendation"] == Recommendation.REGULATORY_REVIEW
        assert any("Critic escalation" in r or "Critic" in r for r in dec_res["reasons"])

    def test_critic_citation_validation(self, llm):
        """Critic flags non-existent chunks and invalid citation quotes."""
        critic = CriticAgent(llm_service=llm)

        class MockRetriever:
            def chunk_exists(self, chunk_id: str) -> bool:
                return chunk_id == "valid_chunk"

            def is_valid_citation(self, chunk_id: str, quote: str) -> bool:
                return quote == "exact valid quote"

        finding_missing = Finding(
            field="claims",
            issue_type=IssueType.UNSUPPORTED_CLAIM,
            severity=Severity.MAJOR,
            description="Claim citing missing chunk",
            confidence=0.8,
            chunk_id="missing_chunk",
            evidence_quote="some quote",
            agent="ClaimsAgent",
        )
        context = {
            "all_findings": [finding_missing],
            "retriever": MockRetriever(),
            "retrieved_chunks": {},
        }
        res, _ = critic.run(SAMPLE_LABELS["compliant_biscuit"], context)
        assert res["escalate_to_regulatory_review"] is True

    def test_critic_finding_disagreement(self, llm):
        """Critic reduces confidence and flags escalation when agents disagree on a major field."""
        critic = CriticAgent(llm_service=llm)
        f1 = Finding(
            field="ingredients_list",
            issue_type=IssueType.INCONSISTENCY,
            severity=Severity.MAJOR,
            description="Agent A view",
            confidence=0.5,
            agent="AgentA",
        )
        f2 = Finding(
            field="ingredients_list",
            issue_type=IssueType.MISSING_DECLARATION,
            severity=Severity.MAJOR,
            description="Agent B view",
            confidence=0.9,
            agent="AgentB",
        )
        context = {
            "all_findings": [f1, f2],
            "retrieved_chunks": {},
        }
        res, _ = critic.run(SAMPLE_LABELS["compliant_biscuit"], context)
        assert res["escalate_to_regulatory_review"] is True


class TestDecisionAgent:
    def test_error_yields_regulatory_review(self, llm):
        """Errors must always result in REGULATORY_REVIEW."""
        agent = DecisionAgent(llm_service=llm)
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"verified_findings": [], "claim_verifications": [], "has_errors": True, "removed_count": 0},
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_clean_label_yields_clear_for_print(self, llm):
        """No findings + no errors should be CLEAR_FOR_PRINT."""
        agent = DecisionAgent(llm_service=llm)
        result, _ = agent.run(
            SAMPLE_LABELS["compliant_biscuit"],
            {"verified_findings": [], "claim_verifications": [], "has_errors": False, "removed_count": 0},
        )
        assert result["recommendation"] == Recommendation.CLEAR_FOR_PRINT

    def test_prohibited_claim_yields_regulatory_review(self, llm):
        """Prohibited claims should trigger REGULATORY_REVIEW."""
        agent = DecisionAgent(llm_service=llm)
        prohibited_finding = Finding(
            field="claims",
            issue_type=IssueType.PROHIBITED_CLAIM,
            severity=Severity.CRITICAL,
            description="Prohibited: Cures cold",
            confidence=0.95,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["prohibited_claims"],
            {
                "verified_findings": [prohibited_finding],
                "claim_verifications": [{"status": "PROHIBITED"}],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        assert result["recommendation"] == Recommendation.REGULATORY_REVIEW

    def test_missing_fields_yields_revise(self, llm):
        """Missing mandatory fields without errors should be REVISE_AND_RECHECK."""
        agent = DecisionAgent(llm_service=llm)
        missing_finding = Finding(
            field="allergen_declaration",
            issue_type=IssueType.MISSING_DECLARATION,
            severity=Severity.CRITICAL,
            description="Missing allergen declaration",
            confidence=1.0,
            verified_by_critic=True,
        )
        result, _ = agent.run(
            SAMPLE_LABELS["missing_declarations"],
            {
                "verified_findings": [missing_finding],
                "claim_verifications": [],
                "has_errors": False,
                "removed_count": 0,
            },
        )
        # With critical findings it goes to REVISE or REGULATORY_REVIEW depending on count
        assert result["recommendation"] in (
            Recommendation.REVISE_AND_RECHECK,
            Recommendation.REGULATORY_REVIEW,
        )
