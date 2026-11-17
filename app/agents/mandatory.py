"""
Mandatory Declaration Agent (Step 3)
Checks required label elements using DETERMINISTIC Python rules.
LLM used only for ambiguous cases.
Grounded in chunks retrieved by the Retrieval Agent.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger
from app.models.enums import IssueType, Severity
from app.models.schemas import Finding, LabelSubmission

logger = get_logger(__name__)

# Mandatory fields as defined in FSSAI Labelling & Display Regulations 2020
MANDATORY_FIELDS = [
    {
        "field": "product_name",
        "attr": "product_name",
        "label": "Name of food",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(a), Clause 4",
        "suggested_fix": "Add the product name indicating the true nature of the food.",
    },
    {
        "field": "ingredients_list",
        "attr": "ingredients_list",
        "label": "List of ingredients",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(b), Clause 5",
        "suggested_fix": "Provide complete list of ingredients in descending order of composition.",
    },
    {
        "field": "allergen_declaration",
        "attr": "allergen_declaration",
        "label": "Allergen declaration",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 6",
        "suggested_fix": "Add allergen declaration starting with 'Contains:' followed by allergen names.",
    },
    {
        "field": "net_quantity",
        "attr": "net_quantity",
        "label": "Net quantity",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(f), Clause 7",
        "suggested_fix": "Declare net quantity in metric units (g, kg, ml, L).",
    },
    {
        "field": "nutritional_info",
        "attr": "nutritional_info",
        "label": "Nutritional information",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(c), Clause 11",
        "suggested_fix": "Add nutrition information panel with energy, protein, carbohydrate, fat, sugar, and sodium per 100g and per serving.",
    },
    {
        "field": "veg_nonveg_mark",
        "attr": "veg_nonveg_mark",
        "label": "Vegetarian / Non-vegetarian mark",
        "severity": Severity.MAJOR,
        "clause_hint": "Clause 8",
        "suggested_fix": "Add green circle (veg) or brown circle (non-veg) mark on principal display panel.",
    },
    {
        "field": "fssai_licence_number",
        "attr": "fssai_licence_number",
        "label": "FSSAI logo and licence number",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 9",
        "suggested_fix": "Display FSSAI logo and valid 14-digit licence number prominently.",
    },
    {
        "field": "manufacturer_details",
        "attr": "manufacturer_details",
        "label": "Manufacturer / packer name and address",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(e), Clause 12",
        "suggested_fix": "Add complete name and address of manufacturer or packer.",
    },
    {
        "field": "batch_lot",
        "attr": "batch_lot",
        "label": "Batch / lot number",
        "severity": Severity.MAJOR,
        "clause_hint": "Clause 3(g), Clause 15",
        "suggested_fix": "Add batch or lot number for traceability.",
    },
    {
        "field": "manufacturing_date",
        "attr": "manufacturing_date",
        "label": "Date of manufacture",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(h), Clause 10",
        "suggested_fix": "Add date of manufacture.",
    },
    {
        "field": "expiry_date",
        "attr": "expiry_date",
        "label": "Expiry date / best before date",
        "severity": Severity.CRITICAL,
        "clause_hint": "Clause 3(i), Clause 10",
        "check_alt": "best_before",  # Either expiry or best_before is acceptable
        "suggested_fix": "Add expiry date or best before date.",
    },
    {
        "field": "storage_instructions",
        "attr": "storage_instructions",
        "label": "Storage instructions",
        "severity": Severity.MAJOR,
        "clause_hint": "Clause 13",
        "suggested_fix": "Add storage instructions (e.g., 'Store in a cool, dry place').",
    },
]


def _is_empty(value: Any) -> bool:
    """Check if a field value is effectively empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _get_best_chunk(field: str, retrieved_chunks: dict) -> dict[str, Any] | None:
    """Get the best matching chunk for a field from retrieval context."""
    chunks = retrieved_chunks.get(field, [])
    if chunks:
        return chunks[0]  # Already sorted by relevance
    return None


class MandatoryDeclarationAgent(BaseAgent):
    """
    Step 3: Check mandatory label elements.
    Uses DETERMINISTIC Python rules for presence checks.
    Retrieves regulation citations from pre-fetched chunks.
    """

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        findings: list[Finding] = []
        retrieved_chunks = context.get("retrieved_chunks", {})

        for field_def in MANDATORY_FIELDS:
            field_name = field_def["field"]
            attr_name = field_def["attr"]
            value = getattr(submission, attr_name, None)

            # Special case: expiry_date can be substituted by best_before
            if field_name == "expiry_date":
                if _is_empty(value) and not _is_empty(submission.best_before):
                    continue  # best_before present, so this is OK

            if _is_empty(value):
                # Get citation from retrieved chunks
                chunk = _get_best_chunk(field_name, retrieved_chunks)
                evidence_quote = None
                regulation_name = None
                clause = field_def.get("clause_hint", "")
                source_doc = None
                chunk_id = None
                retrieval_score = None

                if chunk:
                    evidence_quote = chunk.get("text", "")[:200]
                    regulation_name = chunk.get("regulation_name", "")
                    clause = chunk.get("clause", "") or clause
                    source_doc = chunk.get("source_file", "")
                    chunk_id = chunk.get("chunk_id", "")
                    retrieval_score = chunk.get("score")

                findings.append(Finding(
                    field=field_name,
                    issue_type=IssueType.MISSING_DECLARATION,
                    severity=field_def["severity"],
                    description=f"Missing mandatory declaration: {field_def['label']}.",
                    evidence_quote=evidence_quote,
                    regulation_name=regulation_name,
                    clause=clause,
                    source_document=source_doc,
                    chunk_id=chunk_id,
                    retrieval_score=retrieval_score,
                    agent=self.name,
                    confidence=1.0,  # Deterministic check
                    suggested_fix=field_def.get("suggested_fix", ""),
                ))

        # --- FSSAI licence number format check (14 digits) ---
        if submission.fssai_licence_number:
            licence_str = str(submission.fssai_licence_number).strip()
            if not (licence_str.isdigit() and len(licence_str) == 14):
                chunk = _get_best_chunk("fssai_licence_number", retrieved_chunks)
                findings.append(Finding(
                    field="fssai_licence_number",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=Severity.CRITICAL,
                    description=f"Invalid FSSAI licence number '{licence_str}'. FSSAI licence must be exactly 14 digits.",
                    evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                    regulation_name=chunk.get("regulation_name", "") if chunk else None,
                    clause=chunk.get("clause", "") if chunk else "Clause 9",
                    source_document=chunk.get("source_file", "") if chunk else None,
                    chunk_id=chunk.get("chunk_id", "") if chunk else None,
                    retrieval_score=chunk.get("score") if chunk else None,
                    agent=self.name,
                    confidence=1.0,
                    suggested_fix="Display a valid 14-digit FSSAI licence number.",
                ))

        # --- Nutrition panel completeness check (deterministic) ---
        if submission.nutritional_info:
            ni = submission.nutritional_info
            if _is_empty(ni.serving_size):
                chunk = _get_best_chunk("nutritional_info", retrieved_chunks)
                findings.append(Finding(
                    field="nutritional_info",
                    issue_type=IssueType.MISSING_DECLARATION,
                    severity=Severity.MAJOR,
                    description="Missing serving size in nutrition information panel.",
                    evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                    regulation_name=chunk.get("regulation_name", "") if chunk else None,
                    clause=chunk.get("clause", "") if chunk else "Clause 11",
                    source_document=chunk.get("source_file", "") if chunk else None,
                    chunk_id=chunk.get("chunk_id", "") if chunk else None,
                    retrieval_score=chunk.get("score") if chunk else None,
                    agent=self.name,
                    confidence=1.0,
                    suggested_fix="Declare serving size (e.g. '30 g' or '1 bar (50 g)') in the nutrition panel.",
                ))

            required_nutrients = [
                "Protein", "Total Fat", "Saturated Fat", "Trans Fat",
                "Carbohydrate", "Total Sugars", "Added Sugars", "Sodium"
            ]
            for nutrient_name in required_nutrients:
                nutrient = ni.get_nutrient(nutrient_name)
                if nutrient is None:
                    chunk = _get_best_chunk("nutritional_info", retrieved_chunks)
                    findings.append(Finding(
                        field="nutritional_info",
                        issue_type=IssueType.MISSING_DECLARATION,
                        severity=Severity.MAJOR,
                        description=f"Missing required nutrient in nutrition panel: {nutrient_name}.",
                        evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                        regulation_name=chunk.get("regulation_name", "") if chunk else None,
                        clause=chunk.get("clause", "") if chunk else "Clause 11",
                        source_document=chunk.get("source_file", "") if chunk else None,
                        chunk_id=chunk.get("chunk_id", "") if chunk else None,
                        retrieval_score=chunk.get("score") if chunk else None,
                        agent=self.name,
                        confidence=1.0,
                        suggested_fix=f"Add {nutrient_name} (per 100g/ml and per serving) to the nutrition panel.",
                    ))

            energy_100g = getattr(ni, "energy_kcal_per_100g", None)
            energy_100ml = getattr(ni, "energy_kcal_per_100ml", None)
            if energy_100g is None and energy_100ml is None:
                findings.append(Finding(
                    field="nutritional_info",
                    issue_type=IssueType.MISSING_DECLARATION,
                    severity=Severity.MAJOR,
                    description="Missing energy (kcal) per 100g/100ml in nutrition panel.",
                    agent=self.name,
                    confidence=1.0,
                    suggested_fix="Add energy value in kcal per 100g or 100ml.",
                ))

        # --- Country of origin for imported food ---
        if submission.country_of_origin and submission.country_of_origin.lower() != "india":
            # If imported, check for importer details
            if not submission.manufacturer_details or "import" not in submission.manufacturer_details.lower():
                chunk = _get_best_chunk("country_of_origin", retrieved_chunks)
                findings.append(Finding(
                    field="country_of_origin",
                    issue_type=IssueType.MISSING_DECLARATION,
                    severity=Severity.MAJOR,
                    description="Imported food: importer name and address may be required in addition to manufacturer details.",
                    evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                    regulation_name=chunk.get("regulation_name", "") if chunk else None,
                    clause=chunk.get("clause", "") if chunk else "Clause 14",
                    source_document=chunk.get("source_file", "") if chunk else None,
                    chunk_id=chunk.get("chunk_id", "") if chunk else None,
                    retrieval_score=chunk.get("score") if chunk else None,
                    agent=self.name,
                    confidence=0.7,
                    suggested_fix="Add importer name and address for imported food.",
                ))

        return {
            "findings": findings,
            "confidence": 1.0,  # Deterministic checks
            "fields_checked": len(MANDATORY_FIELDS),
            "missing_count": len(findings),
            "mandatory_checks_passed": len(findings) == 0,
        }
