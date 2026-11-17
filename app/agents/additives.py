"""
Additives & Allergen Agent (Step 5)
Deterministic consistency checks for allergen↔ingredient and additive declarations.
LLM used only for ambiguous mappings.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from app.agents.base import BaseAgent
from app.core.config import DATA_DIR
from app.core.logging import get_logger
from app.models.enums import IssueType, Severity
from app.models.schemas import Finding, LabelSubmission

logger = get_logger(__name__)

INS_COLOURS_PATH = DATA_DIR / "ins_colours.json"


@lru_cache(maxsize=1)
def load_ins_colours_data() -> dict[str, Any]:
    """Read permitted and prohibited INS colour lists directly from data/ins_colours.json."""
    if not INS_COLOURS_PATH.exists():
        logger.warning("ins_colours_file_not_found", path=str(INS_COLOURS_PATH))
        return {"permitted_colours": [], "prohibited_colours": []}
    try:
        with open(INS_COLOURS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("ins_colours_load_error", error=str(e))
        return {"permitted_colours": [], "prohibited_colours": []}


# Known allergens and their common ingredient names
ALLERGEN_INGREDIENT_MAP = {
    "wheat": ["wheat", "whole wheat", "wheat flour", "maida", "atta", "semolina", "sooji", "suji"],
    "gluten": ["wheat", "rye", "barley", "oats", "spelt", "gluten"],
    "milk": ["milk", "milk solids", "cream", "butter", "cheese", "whey", "casein",
             "lactose", "curd", "yogurt", "ghee", "paneer", "skimmed milk"],
    "soy": ["soy", "soya", "soybean", "soy lecithin", "soy protein", "tofu", "edamame"],
    "peanut": ["peanut", "groundnut", "peanut butter", "peanut oil"],
    "tree nut": ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
                 "macadamia", "brazil nut", "pine nut"],
    "egg": ["egg", "eggs", "egg white", "egg yolk", "albumin", "mayonnaise"],
    "fish": ["fish", "anchovy", "salmon", "tuna", "cod", "sardine", "mackerel"],
    "crustacean": ["shrimp", "prawn", "crab", "lobster", "crayfish", "crustacean"],
    "sulphite": ["sulphite", "sulfite", "sulphur dioxide", "so2", "metabisulphite",
                 "metabisulfite", "ins 220", "ins 221", "ins 222", "ins 223", "ins 224",
                 "ins 225", "ins 226", "ins 227", "ins 228"],
}

# Common non-veg indicators
NON_VEG_INGREDIENTS = [
    "chicken", "mutton", "lamb", "beef", "pork", "fish", "prawn", "shrimp",
    "crab", "lobster", "egg", "gelatin", "lard", "tallow", "anchovy",
    "meat", "poultry", "bacon", "ham", "sausage",
]

VEG_EXCEPTIONS = ["eggplant", "egg noodle"]


class AdditivesAllergenAgent(BaseAgent):
    """
    Step 5: Check allergen↔ingredient consistency and additive declarations.
    Uses DETERMINISTIC rules for matching.
    """

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        findings: list[Finding] = []
        retrieved_chunks = context.get("retrieved_chunks", {})

        # --- 1. Allergen-Ingredient Consistency (Deterministic) ---
        findings.extend(self._check_allergen_consistency(submission, retrieved_chunks))

        # --- 2. Veg/Non-Veg Mark Consistency (Deterministic) ---
        findings.extend(self._check_veg_nonveg_consistency(submission, retrieved_chunks))

        # --- 3. Additive Declaration Checks (Deterministic) ---
        findings.extend(self._check_additives(submission, retrieved_chunks))

        return {
            "findings": findings,
            "confidence": 0.9 if not findings else 0.7,
        }

    def _check_allergen_consistency(
        self,
        submission: LabelSubmission,
        retrieved_chunks: dict,
    ) -> list[Finding]:
        """Check that allergens in ingredients are declared in allergen statement."""
        findings = []
        allergen_decl_lower = (submission.allergen_declaration or "").strip().lower()
        if not allergen_decl_lower:
            return findings

        ingredients_lower = " ".join(submission.ingredients_list).lower()

        # Find allergens present in ingredients but not declared
        for allergen_group, keywords in ALLERGEN_INGREDIENT_MAP.items():
            ingredient_has_allergen = False
            matched_keyword = ""
            for kw in keywords:
                if kw in ingredients_lower:
                    ingredient_has_allergen = True
                    matched_keyword = kw
                    break

            if ingredient_has_allergen:
                # Check if any keyword for this allergen group is in the declaration
                declared = False
                for kw in keywords:
                    if kw in allergen_decl_lower:
                        declared = True
                        break
                # Also check the allergen group name itself
                if allergen_group in allergen_decl_lower:
                    declared = True

                if not declared:
                    chunk = self._get_chunk("allergen_consistency", retrieved_chunks)
                    findings.append(Finding(
                        field="allergen_declaration",
                        issue_type=IssueType.ALLERGEN_MISMATCH,
                        severity=Severity.CRITICAL,
                        description=(
                            f"Allergen '{allergen_group}' (from ingredient '{matched_keyword}') "
                            f"is present in ingredients but NOT declared in the allergen statement."
                        ),
                        evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                        regulation_name=chunk.get("regulation_name", "") if chunk else None,
                        clause=chunk.get("clause", "") if chunk else "Clause 6",
                        source_document=chunk.get("source_file", "") if chunk else None,
                        chunk_id=chunk.get("chunk_id", "") if chunk else None,
                        retrieval_score=chunk.get("score") if chunk else None,
                        agent=self.name,
                        confidence=0.95,
                        suggested_fix=f"Add '{allergen_group.title()}' to the allergen declaration (e.g., 'Contains: ... {allergen_group.title()}').",
                    ))

        return findings

    def _check_veg_nonveg_consistency(
        self,
        submission: LabelSubmission,
        retrieved_chunks: dict,
    ) -> list[Finding]:
        """Check veg/non-veg mark is consistent with ingredients."""
        findings = []
        if not submission.veg_nonveg_mark:
            return findings

        ingredients_lower = " ".join(submission.ingredients_list).lower()

        if submission.veg_nonveg_mark.lower() in ("veg", "vegetarian", "green"):
            for nv in NON_VEG_INGREDIENTS:
                # Skip false positives
                skip = False
                for exc in VEG_EXCEPTIONS:
                    if exc in ingredients_lower:
                        skip = True
                        break
                if skip:
                    continue
                if re.search(r'\b' + re.escape(nv) + r'\b', ingredients_lower):
                    findings.append(Finding(
                        field="veg_nonveg_mark",
                        issue_type=IssueType.INCONSISTENCY,
                        severity=Severity.CRITICAL,
                        description=(
                            f"Product marked as 'vegetarian' but ingredient '{nv}' "
                            f"suggests non-vegetarian content."
                        ),
                        agent=self.name,
                        confidence=0.9,
                        suggested_fix="Change the mark to non-vegetarian (brown/red) or verify the ingredient.",
                    ))
                    break  # One finding is enough

        return findings

    def _check_additives(
        self,
        submission: LabelSubmission,
        retrieved_chunks: dict,
    ) -> list[Finding]:
        """Check additive declarations and verify colours against data/ins_colours.json."""
        findings = []
        ingredients_lower = " ".join(submission.ingredients_list).lower()
        colours_data = load_ins_colours_data()

        # 1. Check for prohibited colours from data file
        for prohibited in colours_data.get("prohibited_colours", []):
            name = prohibited.get("name", "").lower()
            synonyms = [s.lower() for s in prohibited.get("synonyms", [])]
            search_terms = [name] + synonyms
            for term in search_terms:
                if term and term in ingredients_lower:
                    findings.append(Finding(
                        field="additives",
                        issue_type=IssueType.REGULATORY_VIOLATION,
                        severity=Severity.CRITICAL,
                        description=(
                            f"Prohibited non-permitted substance '{prohibited.get('name')}' detected in ingredients. "
                            f"{prohibited.get('reason', 'Strictly prohibited under FSSAI regulations')}."
                        ),
                        agent=self.name,
                        confidence=0.95,
                        suggested_fix="Immediately reformulate product without prohibited non-food colours.",
                        regulation_name="[SAMPLE] Food Safety and Standards (Food Products Standards and Food Additives) Regulations",
                        clause="Clause 4",
                        is_sample=True,
                    ))
                    break

        # 2. Check additive declarations
        for additive in submission.additives:
            additive_in_ingredients = False
            name_lower = additive.name.lower()
            ins_str = (additive.ins_number or "").lower()

            if name_lower in ingredients_lower or ins_str in ingredients_lower:
                additive_in_ingredients = True
            else:
                for part in name_lower.split():
                    if len(part) > 3 and part in ingredients_lower:
                        additive_in_ingredients = True
                        break

            if not additive_in_ingredients:
                chunk_key = f"additive_{additive.name}"
                chunk = self._get_chunk(chunk_key, retrieved_chunks)
                findings.append(Finding(
                    field="additives",
                    issue_type=IssueType.INCONSISTENCY,
                    severity=Severity.MAJOR,
                    description=(
                        f"Additive '{additive.name}' ({additive.ins_number or 'no INS'}) "
                        f"is declared as an additive but not found in the ingredients list."
                    ),
                    evidence_quote=chunk.get("text", "")[:200] if chunk else None,
                    regulation_name=chunk.get("regulation_name", "") if chunk else None,
                    clause=chunk.get("clause", "") if chunk else "Clause 3",
                    source_document=chunk.get("source_file", "") if chunk else None,
                    chunk_id=chunk.get("chunk_id", "") if chunk else None,
                    retrieval_score=chunk.get("score") if chunk else None,
                    agent=self.name,
                    confidence=0.7,
                    suggested_fix=f"Ensure '{additive.name}' appears in the ingredients list with its functional class and INS number.",
                    is_sample=True if (chunk and chunk.get("is_sample")) else False,
                ))

        return findings

    def _get_chunk(self, key: str, retrieved_chunks: dict) -> dict | None:
        chunks = retrieved_chunks.get(key, [])
        return chunks[0] if chunks else None
