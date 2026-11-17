"""
Pydantic v2 schemas for label submissions, findings, reports, and API I/O.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.models.enums import (
    CheckStatus,
    ClaimStatus,
    ExpertAction,
    IssueType,
    Recommendation,
    Severity,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Nutritional Information
# ---------------------------------------------------------------------------

class NutrientValue(BaseModel):
    """A single nutrient with per-100g and per-serving values."""
    name: str
    per_100g: float | None = None
    per_100ml: float | None = None
    per_serving: float | None = None
    unit: str = "g"


class NutritionalInfo(BaseModel):
    """Full nutrition panel."""
    serving_size: str | None = None
    energy_kcal_per_100g: float | None = None
    energy_kcal_per_100ml: float | None = None
    energy_kcal_per_serving: float | None = None
    nutrients: list[NutrientValue] = Field(default_factory=list)

    def get_nutrient(self, name: str) -> NutrientValue | None:
        """Look up a nutrient by name (case-insensitive)."""
        name_lower = name.lower()
        for n in self.nutrients:
            if n.name.lower() == name_lower:
                return n
        return None


class AdditiveInfo(BaseModel):
    """An additive declared on the label."""
    name: str
    ins_number: str | None = None
    purpose: str | None = None


# ---------------------------------------------------------------------------
# Label Submission (input)
# ---------------------------------------------------------------------------

class LabelSubmission(BaseModel):
    """Complete food label data submitted for compliance checking."""
    product_name: str = Field(..., min_length=1, max_length=500)
    product_category: str = Field(default="", max_length=200)
    ingredients_list: list[str] = Field(default_factory=list)
    allergen_declaration: str | None = None
    net_quantity: str | None = None
    nutritional_info: NutritionalInfo | None = None
    veg_nonveg_mark: str | None = None
    fssai_licence_number: str | None = None
    manufacturer_details: str | None = None
    batch_lot: str | None = None
    manufacturing_date: str | None = None
    expiry_date: str | None = None
    best_before: str | None = None
    storage_instructions: str | None = None
    country_of_origin: str | None = None
    additives: list[AdditiveInfo] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    free_text_content: str | None = None
    extraction_id: str | None = None
    user_confirmed: bool | None = None

    @field_validator("product_name")
    @classmethod
    def sanitize_product_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("ingredients_list", mode="before")
    @classmethod
    def parse_ingredients(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single compliance finding with evidence and citation."""
    finding_id: str = Field(default_factory=_uuid)
    field: str = ""
    issue_type: IssueType = IssueType.REGULATORY_VIOLATION
    severity: Severity = Severity.MAJOR
    description: str = ""
    evidence_quote: str | None = None
    regulation_name: str | None = None
    clause: str | None = None
    source_document: str | None = None
    chunk_id: str | None = None
    retrieval_score: float | None = None
    agent: str = ""
    verified_by_critic: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_fix: str | None = None
    is_sample: bool = False


class ClaimVerification(BaseModel):
    """Result of verifying a single claim."""
    claim_text: str
    status: ClaimStatus
    reason: str = ""
    evidence_quote: str | None = None
    regulation_name: str | None = None
    clause: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_sample: bool = False


# ---------------------------------------------------------------------------
# Agent Trace
# ---------------------------------------------------------------------------

class AgentTraceEntry(BaseModel):
    """Trace of a single agent's execution."""
    agent_name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    status: str = "success"
    findings_count: int = 0
    error_message: str | None = None
    llm_calls: int = 0
    tokens_used: int = 0
    retries: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compliance Report (output)
# ---------------------------------------------------------------------------

class ComplianceReport(BaseModel):
    """Full compliance check report."""
    check_id: str = Field(default_factory=_uuid)
    submission: LabelSubmission
    recommendation: Recommendation = Recommendation.REGULATORY_REVIEW
    findings: list[Finding] = Field(default_factory=list)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list)
    overall_confidence: float = 0.0
    processing_time_seconds: float = 0.0
    has_errors: bool = False
    is_sample: bool = False
    sample_notice: str | None = None


# ---------------------------------------------------------------------------
# Expert Decision
# ---------------------------------------------------------------------------

class ExpertDecisionRequest(BaseModel):
    """Expert's decision on a compliance check."""
    action: ExpertAction
    reason: str = Field(default="", max_length=2000)
    expert_name: str = Field(default="", max_length=200)


class ExpertDecisionResponse(BaseModel):
    """Stored expert decision with metadata."""
    decision_id: str
    check_id: str
    action: ExpertAction
    original_recommendation: Recommendation
    reason: str
    expert_name: str
    decided_at: datetime


# ---------------------------------------------------------------------------
# API Response Wrappers
# ---------------------------------------------------------------------------

class CheckResponse(BaseModel):
    """API response for a compliance check."""
    check_id: str
    status: CheckStatus
    recommendation: Recommendation | None = None
    report: ComplianceReport | None = None
    expert_decision: ExpertDecisionResponse | None = None
    disclaimer: str = ""
    message: str = ""
    is_sample: bool = False


class CheckListItem(BaseModel):
    """Summary item for the checks list."""
    check_id: str
    product_name: str
    status: CheckStatus
    recommendation: Recommendation | None = None
    submitted_at: datetime
    has_expert_decision: bool = False


class CheckListResponse(BaseModel):
    """Paginated list of checks."""
    items: list[CheckListItem]
    total: int
    page: int = 1
    per_page: int = 20


class RegulationSearchResult(BaseModel):
    """A regulation chunk returned from search."""
    chunk_id: str
    text: str
    regulation_name: str
    clause: str | None = None
    source_document: str
    score: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    database: str = "unknown"
    vector_store: str = "unknown"
    llm_provider: str = "unknown"


# ---------------------------------------------------------------------------
# Sample labels (for the "load sample" button)
# ---------------------------------------------------------------------------

SAMPLE_LABELS: dict[str, LabelSubmission] = {
    "compliant_biscuit": LabelSubmission(
        product_name="Wholesome Wheat Biscuits",
        product_category="Bakery - Biscuits",
        ingredients_list=[
            "Whole Wheat Flour (65%)", "Sugar", "Refined Palm Oil",
            "Milk Solids", "Raising Agents (INS 500(ii), INS 503(ii))",
            "Salt", "Emulsifier (INS 322(i))", "Dough Conditioner (INS 223)"
        ],
        allergen_declaration="Contains: Wheat, Milk, Sulphites. May contain traces of Nuts and Soy.",
        net_quantity="200 g",
        nutritional_info=NutritionalInfo(
            serving_size="30 g (3 biscuits)",
            energy_kcal_per_100g=480.0,
            energy_kcal_per_serving=144.0,
            nutrients=[
                NutrientValue(name="Protein", per_100g=8.5, per_serving=2.55, unit="g"),
                NutrientValue(name="Total Fat", per_100g=18.0, per_serving=5.4, unit="g"),
                NutrientValue(name="Saturated Fat", per_100g=8.0, per_serving=2.4, unit="g"),
                NutrientValue(name="Trans Fat", per_100g=0.0, per_serving=0.0, unit="g"),
                NutrientValue(name="Carbohydrate", per_100g=66.0, per_serving=19.8, unit="g"),
                NutrientValue(name="Total Sugars", per_100g=22.0, per_serving=6.6, unit="g"),
                NutrientValue(name="Added Sugars", per_100g=20.0, per_serving=6.0, unit="g"),
                NutrientValue(name="Dietary Fibre", per_100g=6.0, per_serving=1.8, unit="g"),
                NutrientValue(name="Sodium", per_100g=350.0, per_serving=105.0, unit="mg"),
            ],
        ),
        veg_nonveg_mark="veg",
        fssai_licence_number="10012345678901",
        manufacturer_details="XYZ Foods Pvt. Ltd., Plot 42, Industrial Area, Pune, Maharashtra 411001, India",
        batch_lot="B2024-0842",
        manufacturing_date="2024-08-15",
        expiry_date="2025-02-15",
        best_before="6 months from packaging",
        storage_instructions="Store in a cool, dry place away from direct sunlight.",
        country_of_origin="India",
        additives=[
            AdditiveInfo(name="Sodium Bicarbonate", ins_number="INS 500(ii)", purpose="Raising Agent"),
            AdditiveInfo(name="Ammonium Bicarbonate", ins_number="INS 503(ii)", purpose="Raising Agent"),
            AdditiveInfo(name="Soy Lecithin", ins_number="INS 322(i)", purpose="Emulsifier"),
            AdditiveInfo(name="Sodium Metabisulphite", ins_number="INS 223", purpose="Dough Conditioner"),
        ],
        claims=["Source of fibre", "No artificial colours"],
        free_text_content="Best enjoyed with tea or coffee.",
    ),
    "missing_declarations": LabelSubmission(
        product_name="Protein Power Bar",
        product_category="Health Supplements - Bars",
        ingredients_list=[
            "Whey Protein Concentrate", "Oats", "Honey", "Almonds",
            "Dark Chocolate Chips", "Soy Lecithin", "Natural Flavour"
        ],
        allergen_declaration=None,  # Missing!
        net_quantity=None,  # Missing!
        nutritional_info=NutritionalInfo(
            serving_size="50 g (1 bar)",
            energy_kcal_per_100g=420.0,
            energy_kcal_per_serving=210.0,
            nutrients=[
                NutrientValue(name="Protein", per_100g=25.0, per_serving=12.5, unit="g"),
                NutrientValue(name="Total Fat", per_100g=14.0, per_serving=7.0, unit="g"),
                NutrientValue(name="Carbohydrate", per_100g=45.0, per_serving=22.5, unit="g"),
                NutrientValue(name="Total Sugars", per_100g=18.0, per_serving=9.0, unit="g"),
            ],
        ),
        veg_nonveg_mark=None,  # Missing!
        fssai_licence_number="10098765432100",
        manufacturer_details=None,  # Missing!
        batch_lot="BX-2024-112",
        manufacturing_date="2024-07-01",
        expiry_date=None,  # Missing!
        storage_instructions=None,  # Missing!
        claims=["High protein", "No added sugar"],
        free_text_content="Fuel your workout!",
    ),
    "prohibited_claims": LabelSubmission(
        product_name="HerbalVita Immunity Juice",
        product_category="Beverages - Juice",
        ingredients_list=[
            "Water", "Apple Juice Concentrate", "Tulsi Extract",
            "Ashwagandha Extract", "Giloy Extract", "Sugar",
            "Citric Acid (INS 330)", "Permitted Natural Colour (INS 150a)",
            "Sodium Benzoate (INS 211)"
        ],
        allergen_declaration="No known allergens.",
        net_quantity="500 ml",
        nutritional_info=NutritionalInfo(
            serving_size="200 ml",
            energy_kcal_per_100ml=48.0,
            nutrients=[
                NutrientValue(name="Protein", per_100ml=0.2, unit="g"),
                NutrientValue(name="Total Fat", per_100ml=0.0, unit="g"),
                NutrientValue(name="Carbohydrate", per_100ml=11.5, unit="g"),
                NutrientValue(name="Total Sugars", per_100ml=10.0, unit="g"),
                NutrientValue(name="Added Sugars", per_100ml=5.0, unit="g"),
                NutrientValue(name="Sodium", per_100ml=15.0, unit="mg"),
            ],
        ),
        veg_nonveg_mark="veg",
        fssai_licence_number="20056789012345",
        manufacturer_details="HerbalVita Naturals Inc., 12 Green Park, Delhi 110016, India",
        batch_lot="HV-J-20240901",
        manufacturing_date="2024-09-01",
        expiry_date="2025-03-01",
        best_before="6 months from manufacturing",
        storage_instructions="Refrigerate after opening. Consume within 3 days.",
        country_of_origin="India",
        additives=[
            AdditiveInfo(name="Citric Acid", ins_number="INS 330", purpose="Acidity Regulator"),
            AdditiveInfo(name="Caramel Colour", ins_number="INS 150a", purpose="Colour"),
            AdditiveInfo(name="Sodium Benzoate", ins_number="INS 211", purpose="Preservative"),
        ],
        claims=[
            "Boosts immunity",
            "Cures cold and flu",
            "100% natural",
            "No added sugar",
            "Immunity booster",
        ],
        free_text_content="Clinically proven to prevent infections. Doctor recommended.",
    ),
}
