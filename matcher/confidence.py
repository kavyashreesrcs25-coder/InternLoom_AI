"""
InternLoom AI - Confidence Calculator
Determines parse confidence and scoring reliability for each candidate.

Confidence levels:
  High   — Clean digital parse, all critical fields present
  Medium — Partial parse or OCR parse, most fields present
  Low    — Failed parse, too few fields, or OCR with poor quality

Confidence affects how scores are interpreted in the ranking UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger
from parser.extractor import CandidateData
from parser.pdf_parser import ParseResult
from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONFIDENCE_MAP,
    PARSE_STATUS_CLEAN,
    PARSE_STATUS_PARTIAL,
    PARSE_STATUS_OCR,
    PARSE_STATUS_FAILED,
    OCR_WORD_THRESHOLD,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class ConfidenceReport:
    """Parse confidence details for one candidate."""
    overall: str                       # "High" | "Medium" | "Low"
    parse_status: str                  # From ParseResult
    word_count: int
    fields_present: list[str]
    fields_missing: list[str]
    completeness_pct: float            # 0–100
    penalties: list[str]               # Descriptions of confidence deductions
    notes: str = ""


# ─────────────────────────────────────────────
# FIELD WEIGHTS FOR COMPLETENESS
# ─────────────────────────────────────────────

# (field_name, attribute_path, weight)
FIELD_DEFINITIONS: list[tuple[str, str, int]] = [
    ("Name",            "name",              15),
    ("Email",           "email",             10),
    ("Phone",           "phone",              5),
    ("College",         "college",            8),
    ("Degree",          "degree",             8),
    ("Skills",          "skills",            20),
    ("Graduation Year", "graduation_year",    5),
    ("CGPA/Grade",      "normalized_cgpa",   10),
    ("Projects",        "projects",          10),
    ("Internships",     "internships",        5),
    ("Certifications",  "certifications",     4),
]

_TOTAL_WEIGHT = sum(w for _, _, w in FIELD_DEFINITIONS)


def _field_present(candidate: CandidateData, attr: str) -> bool:
    """Return True if the candidate attribute has a non-empty value."""
    val = getattr(candidate, attr, None)
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (int, float)):
        return val > 0
    return bool(val)


# ─────────────────────────────────────────────
# CONFIDENCE CALCULATOR
# ─────────────────────────────────────────────

class ConfidenceCalculator:
    """
    Calculates parse confidence for a candidate.

    Inputs:
      - ParseResult   (word count, strategy, parse_status)
      - CandidateData (extracted fields)

    Output:
      - ConfidenceReport
    """

    def calculate(
        self,
        parse_result: ParseResult,
        candidate: CandidateData,
    ) -> ConfidenceReport:
        """
        Compute overall confidence and field completeness.

        Args:
            parse_result: Output of the PDF parser.
            candidate:    Extracted candidate data.

        Returns:
            ConfidenceReport with detailed breakdown.
        """
        penalties: list[str] = []
        fields_present: list[str] = []
        fields_missing: list[str] = []
        weighted_present = 0

        # ── Field completeness ─────────────────────────────────────────────
        for field_name, attr, weight in FIELD_DEFINITIONS:
            if _field_present(candidate, attr):
                fields_present.append(field_name)
                weighted_present += weight
            else:
                fields_missing.append(field_name)

        completeness_pct = round((weighted_present / _TOTAL_WEIGHT) * 100, 1)

        # ── Base confidence from parse status ─────────────────────────────
        base_confidence = CONFIDENCE_MAP.get(
            parse_result.parse_status, CONFIDENCE_LOW
        )

        # ── Penalty rules ─────────────────────────────────────────────────
        # Word count too low
        if parse_result.word_count < OCR_WORD_THRESHOLD:
            penalties.append(
                f"Low word count ({parse_result.word_count} words < {OCR_WORD_THRESHOLD} threshold)"
            )

        # Failed parse
        if parse_result.parse_status == PARSE_STATUS_FAILED:
            penalties.append("PDF parse failed — no text extracted")
            return ConfidenceReport(
                overall=CONFIDENCE_LOW,
                parse_status=parse_result.parse_status,
                word_count=parse_result.word_count,
                fields_present=[],
                fields_missing=[f for f, _, _ in FIELD_DEFINITIONS],
                completeness_pct=0.0,
                penalties=penalties,
                notes="Cannot assess candidate — resume could not be parsed.",
            )

        # OCR parse
        if parse_result.parse_status == PARSE_STATUS_OCR:
            penalties.append("Text extracted via OCR — may contain recognition errors")

        # Missing critical fields
        if "Name" in fields_missing:
            penalties.append("Candidate name could not be extracted")
        if "Email" in fields_missing:
            penalties.append("Email address not found")
        if "Skills" in fields_missing:
            penalties.append("No skills detected in resume")

        # Low completeness
        if completeness_pct < 40:
            penalties.append(f"Low field completeness ({completeness_pct}%)")
        elif completeness_pct < 60:
            penalties.append(f"Moderate field completeness ({completeness_pct}%)")

        # ── Adjust confidence based on completeness and penalties ──────────
        final_confidence = _adjust_confidence(
            base_confidence, completeness_pct, len(penalties)
        )

        notes = _build_notes(parse_result, completeness_pct, final_confidence)

        return ConfidenceReport(
            overall=final_confidence,
            parse_status=parse_result.parse_status,
            word_count=parse_result.word_count,
            fields_present=fields_present,
            fields_missing=fields_missing,
            completeness_pct=completeness_pct,
            penalties=penalties,
            notes=notes,
        )


def _adjust_confidence(
    base: str, completeness: float, penalty_count: int
) -> str:
    """
    Downgrade base confidence based on completeness and penalty count.

    Rules:
      High   → if completeness < 50% or penalties > 2 → Medium
      Medium → if completeness < 30% or penalties > 3 → Low
    """
    if base == CONFIDENCE_HIGH:
        if completeness < 50 or penalty_count > 2:
            return CONFIDENCE_MEDIUM
    if base == CONFIDENCE_MEDIUM:
        if completeness < 30 or penalty_count > 3:
            return CONFIDENCE_LOW
    return base


def _build_notes(
    parse_result: ParseResult,
    completeness: float,
    confidence: str,
) -> str:
    """Build a human-readable notes string for the UI."""
    parts = [
        f"Parsed via {parse_result.strategy_used} ({parse_result.word_count} words).",
        f"Field completeness: {completeness}%.",
        f"Overall confidence: {confidence}.",
    ]
    if parse_result.parse_status == PARSE_STATUS_OCR:
        parts.append("OCR used — verify extracted data manually for accuracy.")
    if parse_result.parse_status == PARSE_STATUS_PARTIAL:
        parts.append("Partial extraction — some sections may be incomplete.")
    return " ".join(parts)


# ─────────────────────────────────────────────
# STANDALONE HELPERS
# ─────────────────────────────────────────────

def confidence_from_status(parse_status: str) -> str:
    """Quick confidence lookup by parse status string."""
    return CONFIDENCE_MAP.get(parse_status, CONFIDENCE_LOW)


def score_confidence_multiplier(confidence: str) -> float:
    """
    Return a score multiplier based on confidence.
    Used to slightly discount low-confidence candidate scores in the UI.
    (Does NOT affect the raw score — only visual display weight.)
    """
    return {
        CONFIDENCE_HIGH: 1.0,
        CONFIDENCE_MEDIUM: 0.95,
        CONFIDENCE_LOW: 0.85,
    }.get(confidence, 0.85)
