"""Enumerations used across the application."""

from __future__ import annotations

from enum import Enum


class Recommendation(str, Enum):
    """Possible compliance check outcomes."""
    CLEAR_FOR_PRINT = "CLEAR_FOR_PRINT"
    REVISE_AND_RECHECK = "REVISE_AND_RECHECK"
    REGULATORY_REVIEW = "REGULATORY_REVIEW"


class Severity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ClaimStatus(str, Enum):
    """Status of a label claim after verification."""
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    PROHIBITED = "PROHIBITED"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"


class IssueType(str, Enum):
    """Categories of compliance issues."""
    MISSING_DECLARATION = "missing_declaration"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    PROHIBITED_CLAIM = "prohibited_claim"
    INCONSISTENCY = "inconsistency"
    ALLERGEN_MISMATCH = "allergen_mismatch"
    ADDITIVE_ISSUE = "additive_issue"
    FORMATTING_ERROR = "formatting_error"
    AMBIGUOUS_FIELD = "ambiguous_field"
    REGULATORY_VIOLATION = "regulatory_violation"


class CheckStatus(str, Enum):
    """Status of a compliance check submission."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExpertAction(str, Enum):
    """Actions an expert can take on a recommendation."""
    ACCEPT = "accept"
    OVERRIDE = "override"


class VegNonVeg(str, Enum):
    """Vegetarian / Non-Vegetarian classification."""
    VEG = "veg"
    NON_VEG = "non-veg"
