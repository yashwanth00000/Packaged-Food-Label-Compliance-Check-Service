"""
Intake / Normalizer Agent (Step 1)
Validates and normalizes the submission. Flags ambiguous or suspicious fields.
Includes prompt-injection detection.
"""

from __future__ import annotations

import re
from typing import Any

from app.agents.base import BaseAgent
from app.models.enums import IssueType, Severity
from app.models.schemas import Finding, LabelSubmission

# Patterns that suggest prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"override\s+(system|safety|rules)", re.IGNORECASE),
    re.compile(r"output\s+CLEAR_FOR_PRINT", re.IGNORECASE),
    re.compile(r"respond\s+with\s+CLEAR", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
]


class IntakeAgent(BaseAgent):
    """
    Step 1: Validate, normalize, and sanitize the label submission.
    Purely deterministic — no LLM needed for basic validation.
    """

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        findings: list[Finding] = []

        # --- Basic field validation ---
        if not submission.product_name or not submission.product_name.strip():
            findings.append(Finding(
                field="product_name",
                issue_type=IssueType.MISSING_DECLARATION,
                severity=Severity.CRITICAL,
                description="Product name is missing or empty.",
                agent=self.name,
                confidence=1.0,
                suggested_fix="Add the product name to the label.",
            ))

        if not submission.ingredients_list:
            findings.append(Finding(
                field="ingredients_list",
                issue_type=IssueType.MISSING_DECLARATION,
                severity=Severity.CRITICAL,
                description="Ingredients list is empty.",
                agent=self.name,
                confidence=1.0,
                suggested_fix="Provide the complete list of ingredients in descending order of composition.",
            ))

        # --- Prompt injection detection ---
        all_text_fields = [
            submission.product_name,
            submission.product_category,
            submission.allergen_declaration or "",
            submission.free_text_content or "",
            submission.manufacturer_details or "",
            submission.storage_instructions or "",
        ] + submission.ingredients_list + submission.claims

        for text in all_text_fields:
            if text:
                for pattern in _INJECTION_PATTERNS:
                    if pattern.search(text):
                        findings.append(Finding(
                            field="security",
                            issue_type=IssueType.REGULATORY_VIOLATION,
                            severity=Severity.CRITICAL,
                            description=(
                                "Suspicious text detected that resembles a prompt injection attempt. "
                                "This label has been flagged for security review."
                            ),
                            agent=self.name,
                            confidence=1.0,
                            suggested_fix="Remove suspicious instructional text from label fields.",
                        ))
                        break  # One finding per field is enough

        # --- Date validation ---
        if submission.manufacturing_date and submission.expiry_date:
            try:
                from datetime import datetime
                mfg = datetime.strptime(submission.manufacturing_date, "%Y-%m-%d")
                exp = datetime.strptime(submission.expiry_date, "%Y-%m-%d")
                if exp <= mfg:
                    findings.append(Finding(
                        field="dates",
                        issue_type=IssueType.INCONSISTENCY,
                        severity=Severity.MAJOR,
                        description="Expiry date is on or before the manufacturing date.",
                        agent=self.name,
                        confidence=1.0,
                        suggested_fix="Correct the expiry date to be after the manufacturing date.",
                    ))
            except ValueError:
                findings.append(Finding(
                    field="dates",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=Severity.MINOR,
                    description="Date format could not be parsed. Expected YYYY-MM-DD.",
                    agent=self.name,
                    confidence=0.9,
                    suggested_fix="Use ISO date format (YYYY-MM-DD) for dates.",
                ))

        # --- FSSAI licence number format check ---
        if submission.fssai_licence_number:
            # FSSAI licence numbers are typically 14 digits
            cleaned = re.sub(r"[^0-9]", "", submission.fssai_licence_number)
            if len(cleaned) != 14:
                findings.append(Finding(
                    field="fssai_licence_number",
                    issue_type=IssueType.FORMATTING_ERROR,
                    severity=Severity.MINOR,
                    description=f"FSSAI licence number '{submission.fssai_licence_number}' does not appear to be a valid 14-digit number.",
                    agent=self.name,
                    confidence=0.8,
                    suggested_fix="Verify the FSSAI licence number is a 14-digit number.",
                ))

        return {
            "findings": findings,
            "confidence": 0.95,
            "normalized": True,
            "injection_detected": any(
                f.field == "security" for f in findings
            ),
        }
