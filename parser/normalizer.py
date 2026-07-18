"""
InternLoom AI - Data Normalizer
Normalises grade scales, candidate data fields, and JD text.

Grade normalisation strategy:
  - 10-point CGPA  → used as-is
  - 4-point GPA    → multiply by 2.5  (gives 10-pt equivalent)
  - Percentage     → divide by 9.5    (approximate 10-pt equivalent)
  - Unknown scale  → flag as low confidence, best-guess attempted

JD normalisation:
  - Extracts role, required skills, preferred skills,
    min CGPA, preferred degree, responsibilities, slots
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger
from utils.helper import (
    clean_text,
    normalize_whitespace,
    deduplicate_list,
    extract_lines,
    safe_float,
)
from parser.extractor import CandidateData, _extract_skills_from_text
from config import (
    CGPA_10_MAX,
    GPA_4_MAX,
    GPA_4_MULTIPLIER,
    PERCENTAGE_DIVISOR,
    JD_REQUIRED_KEYWORDS,
    JD_PREFERRED_KEYWORDS,
    JD_RESPONSIBILITY_KEYWORDS,
    CONFIDENCE_LOW,
    KNOWN_SKILLS,
    ALIAS_TO_CANONICAL,
    CGPA_PATTERN,
    PERCENTAGE_PATTERN,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# GRADE NORMALISATION
# ─────────────────────────────────────────────

def normalize_grade(candidate: CandidateData) -> CandidateData:
    """
    Normalise all grade values to a 10-point CGPA equivalent and
    attach to candidate.normalized_cgpa.

    Decision tree:
      1. If raw_cgpa present and scale == "10" → use directly
      2. If raw_gpa present and scale == "4"   → multiply by 2.5
      3. If raw_percentage present             → divide by 9.5
      4. If scale == "unknown":
           - value <= 4.0                      → assume 4-pt GPA
           - 4.0 < value <= 10.0               → assume 10-pt CGPA
           - value > 10.0 and <= 100           → assume percentage
           - else                              → set None, keep low confidence

    Clamps result to [0, 10].
    """
    normalized: Optional[float] = None
    scale_used = candidate.grade_scale

    try:
        if candidate.raw_cgpa is not None and scale_used in ("10", "unknown"):
            val = candidate.raw_cgpa
            if val <= CGPA_10_MAX:
                normalized = val
                candidate.grade_scale = "10"

        elif candidate.raw_gpa is not None and scale_used in ("4", "unknown"):
            val = candidate.raw_gpa
            if val <= GPA_4_MAX:
                normalized = val * GPA_4_MULTIPLIER
                candidate.grade_scale = "4"

        elif candidate.raw_percentage is not None:
            val = candidate.raw_percentage
            if 0 < val <= 100:
                normalized = val / PERCENTAGE_DIVISOR
                candidate.grade_scale = "percentage"

        # Unknown scale — try to infer
        elif scale_used == "unknown":
            val = (
                candidate.raw_cgpa
                or candidate.raw_gpa
                or candidate.raw_percentage
            )
            if val is not None:
                if val <= 4.0:
                    normalized = val * GPA_4_MULTIPLIER
                    candidate.grade_scale = "4 (inferred)"
                elif val <= 10.0:
                    normalized = val
                    candidate.grade_scale = "10 (inferred)"
                elif val <= 100.0:
                    normalized = val / PERCENTAGE_DIVISOR
                    candidate.grade_scale = "percentage (inferred)"

        # Clamp to valid range
        if normalized is not None:
            normalized = max(0.0, min(CGPA_10_MAX, normalized))
            normalized = round(normalized, 2)

    except Exception as exc:
        log.warning("Grade normalisation error for %s: %s", candidate.filename, exc)
        normalized = None

    candidate.normalized_cgpa = normalized
    return candidate


# ─────────────────────────────────────────────
# CANDIDATE NORMALISATION
# ─────────────────────────────────────────────

def normalize_candidate(candidate: CandidateData) -> CandidateData:
    """
    Run all normalisation steps on a CandidateData object:
      - Grade normalisation
      - Skill deduplication and canonical form
      - Text field cleanup

    Args:
        candidate: Raw extracted CandidateData.

    Returns:
        Normalised CandidateData (mutated in place and returned).
    """
    # Grades
    candidate = normalize_grade(candidate)

    # Skills: deduplicate, resolve aliases, sort
    candidate.skills = _normalize_skills(candidate.skills)

    # Text cleanup
    for attr in ("name", "email", "college", "degree", "branch"):
        val = getattr(candidate, attr, "")
        if val:
            setattr(candidate, attr, clean_text(normalize_whitespace(val)).strip())

    # Name: title-case if all-caps
    if candidate.name and candidate.name.isupper():
        candidate.name = candidate.name.title()

    # College: remove trailing punctuation
    if candidate.college:
        candidate.college = candidate.college.strip(" ,;|.-")

    # List fields: strip each item
    for attr in ("projects", "internships", "work_experience", "certifications"):
        items = getattr(candidate, attr, [])
        cleaned = [normalize_whitespace(item) for item in items if item.strip()]
        setattr(candidate, attr, cleaned)

    log.debug("Normalised candidate: %s | CGPA: %s", candidate.name, candidate.normalized_cgpa)
    return candidate


def _normalize_skills(skills: list[str]) -> list[str]:
    """
    Deduplicate and canonicalise skill names.
    Aliases are resolved to their canonical form.
    """
    canonical_set: set[str] = set()
    result: list[str] = []

    for skill in skills:
        skill_stripped = skill.strip()
        skill_lower = skill_stripped.lower()

        # Resolve alias
        if skill_lower in ALIAS_TO_CANONICAL:
            skill_stripped = ALIAS_TO_CANONICAL[skill_lower]

        if skill_stripped.lower() not in canonical_set:
            canonical_set.add(skill_stripped.lower())
            result.append(skill_stripped)

    return sorted(result)


# ─────────────────────────────────────────────
# JOB DESCRIPTION DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class JobDescription:
    """Parsed and structured job description."""
    raw_text: str = ""
    role: str = ""
    source_name: str = ""           # filename or "Pasted JD"
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_cgpa: Optional[float] = None
    preferred_degree: str = ""
    responsibilities: list[str] = field(default_factory=list)
    num_slots: Optional[int] = None
    experience_years: Optional[int] = None

    def all_skills(self) -> list[str]:
        """Combined list of required + preferred skills (deduped)."""
        return deduplicate_list(self.required_skills + self.preferred_skills)

    def display_name(self) -> str:
        """Human-readable name: role if available, else source_name."""
        if self.role and self.role != "Not Specified":
            return self.role
        return self.source_name or "Unnamed JD"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "source_name": self.source_name,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "min_cgpa": self.min_cgpa,
            "preferred_degree": self.preferred_degree,
            "responsibilities": self.responsibilities,
            "num_slots": self.num_slots,
            "experience_years": self.experience_years,
        }


# ─────────────────────────────────────────────
# JD PARSER
# ─────────────────────────────────────────────

_ROLE_PATTERNS = [
    r"(?:position|role|title|job title|opening)[:\s]+([^\n]+)",
    r"(?:hiring|recruiting)\s+(?:for\s+)?([^\n]+)",
    r"^([A-Z][^\n]{5,50})$",  # First capitalised short line as role
]

_SLOTS_PATTERN = re.compile(
    r"(?:openings?|positions?|vacancies|slots?|seats?)[:\s]*(\d+)",
    re.IGNORECASE,
)

_EXP_PATTERN = re.compile(
    r"(\d+)(?:\+)?\s*(?:to\s*\d+)?\s*years?\s*(?:of\s*)?(?:experience|exp)",
    re.IGNORECASE,
)


def _extract_jd_role(text: str) -> str:
    """Extract job role / title from JD text."""
    for pattern in _ROLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            role = match.group(1).strip(" :-")
            if 3 < len(role) < 80:
                return role
    # Fallback: first non-empty line
    for line in extract_lines(text):
        if 3 < len(line) < 80 and not re.search(r"[:\|@]", line):
            return line
    return "Not Specified"


def _extract_jd_skills_from_block(block_text: str) -> list[str]:
    """Extract skills from a JD section block."""
    skills: list[str] = []
    block_lower = block_text.lower()

    for skill in KNOWN_SKILLS:
        pattern = r"(?<![a-zA-Z0-9\.\+\#])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9\.\+\#])"
        if re.search(pattern, block_lower):
            skills.append(skill)

    for alias, canonical in ALIAS_TO_CANONICAL.items():
        pattern = r"(?<![a-zA-Z0-9\.\+\#])" + re.escape(alias.lower()) + r"(?![a-zA-Z0-9\.\+\#])"
        if re.search(pattern, block_lower):
            if canonical not in skills:
                skills.append(canonical)

    return deduplicate_list(skills)


def _split_jd_blocks(text: str) -> dict[str, str]:
    """
    Split JD text into blocks: required, preferred, responsibilities, other.
    """
    blocks = {
        "required": [],
        "preferred": [],
        "responsibilities": [],
        "other": [],
    }
    current = "other"
    lines = text.splitlines()

    for line in lines:
        line_lower = line.lower().strip()

        if any(kw in line_lower for kw in JD_REQUIRED_KEYWORDS):
            current = "required"
        elif any(kw in line_lower for kw in JD_PREFERRED_KEYWORDS):
            current = "preferred"
        elif any(kw in line_lower for kw in JD_RESPONSIBILITY_KEYWORDS):
            current = "responsibilities"

        blocks[current].append(line)

    return {k: "\n".join(v) for k, v in blocks.items()}


def parse_job_description(text: str) -> JobDescription:
    """
    Parse raw JD text into a structured JobDescription object.

    Args:
        text: Raw job description text.

    Returns:
        JobDescription with all fields populated.
    """
    if not text or not text.strip():
        return JobDescription()

    text = clean_text(text)
    jd = JobDescription(raw_text=text)

    # Role
    jd.role = _extract_jd_role(text)

    # Block-split
    blocks = _split_jd_blocks(text)

    # Required skills — from required block, or full text if block is empty
    req_block = blocks["required"] or text
    jd.required_skills = _extract_jd_skills_from_block(req_block)

    # Preferred skills — from preferred block
    pref_block = blocks["preferred"]
    if pref_block:
        jd.preferred_skills = _extract_jd_skills_from_block(pref_block)
        # Ensure no overlap with required
        req_set = {s.lower() for s in jd.required_skills}
        jd.preferred_skills = [s for s in jd.preferred_skills if s.lower() not in req_set]

    # Responsibilities
    resp_block = blocks["responsibilities"]
    if resp_block:
        items = [
            line.strip().lstrip("-•*▪►✓ ")
            for line in resp_block.splitlines()
            if line.strip() and len(line.strip()) > 10
        ]
        jd.responsibilities = items[:15]

    # Min CGPA
    cgpa_matches = re.findall(CGPA_PATTERN, text.lower(), re.IGNORECASE)
    for val_str, _ in cgpa_matches:
        val = safe_float(val_str)
        if 0 < val <= 10:
            jd.min_cgpa = val
            break

    # Percentage fallback
    if jd.min_cgpa is None:
        pct_matches = re.findall(PERCENTAGE_PATTERN, text.lower())
        for m in pct_matches:
            val = safe_float(m)
            if 40 <= val <= 100:
                jd.min_cgpa = round(val / PERCENTAGE_DIVISOR, 2)
                break

    # Preferred degree
    degree_match = re.search(
        r"\b(b\.?tech|m\.?tech|bca|mca|b\.?sc|m\.?sc|bba|mba|b\.?e|m\.?e|phd)\b",
        text, re.IGNORECASE
    )
    if degree_match:
        jd.preferred_degree = degree_match.group(0).upper()

    # Number of openings
    slots_match = _SLOTS_PATTERN.search(text)
    if slots_match:
        jd.num_slots = int(slots_match.group(1))

    # Experience years
    exp_match = _EXP_PATTERN.search(text)
    if exp_match:
        jd.experience_years = int(exp_match.group(1))

    # If required_skills still empty after block split — scan full text
    if not jd.required_skills:
        jd.required_skills = _extract_skills_from_text(text)

    log.info(
        "JD parsed: role=%r, req_skills=%d, pref_skills=%d, min_cgpa=%s",
        jd.role, len(jd.required_skills), len(jd.preferred_skills), jd.min_cgpa,
    )
    return jd


# ─────────────────────────────────────────────
# MULTI-FORMAT JD FILE PARSER
# ─────────────────────────────────────────────

def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a DOCX file."""
    try:
        import docx
        import io as _io
        doc = docx.Document(_io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        log.warning("python-docx not installed — cannot parse DOCX files.")
        return ""
    except Exception as exc:
        log.error("DOCX parse error: %s", exc)
        return ""


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using the existing PDFParser."""
    try:
        from parser.pdf_parser import PDFParser
        # For JDs we never want OCR — they're always text-layer PDFs.
        # Use pdfplumber/pymupdf only; accept partial text even if < 50 words.
        parser = PDFParser()
        result = parser.parse(file_bytes, "jd.pdf", skip_ocr=True)
        return result.full_text or ""
    except Exception as exc:
        log.error("PDF JD parse error: %s", exc)
        return ""


def parse_jd_file(
    file_bytes: bytes,
    filename: str,
) -> Optional[JobDescription]:
    """
    Parse a JD from raw file bytes supporting PDF, DOCX, and TXT formats.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect format and set source_name).

    Returns:
        Populated JobDescription, or None on complete failure.
    """
    fname_lower = filename.lower()

    if fname_lower.endswith(".pdf"):
        text = _extract_text_from_pdf_bytes(file_bytes)
    elif fname_lower.endswith(".docx"):
        text = _extract_text_from_docx(file_bytes)
    elif fname_lower.endswith((".txt", ".md", ".text")):
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    else:
        # Try UTF-8 as a best guess
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            log.error("Cannot decode JD file: %s", filename)
            return None

    if not text.strip():
        log.warning("Empty text extracted from JD file: %s", filename)
        return None

    jd = parse_job_description(text)
    jd.source_name = filename

    # If role extraction gave "Not Specified", use the filename stem
    if jd.role == "Not Specified":
        stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
        if stem:
            jd.role = stem

    log.info(
        "Parsed JD [%s]: role=%r, req=%d, pref=%d",
        filename, jd.role, len(jd.required_skills), len(jd.preferred_skills),
    )
    return jd


def parse_multiple_jds(
    jd_files: list[tuple[str, bytes]],
) -> tuple[list[JobDescription], list[tuple[str, str]]]:
    """
    Parse multiple JD files in sequence.

    Args:
        jd_files: List of (filename, file_bytes) tuples.

    Returns:
        (list_of_JobDescription, list_of_(filename, error_message))
    """
    parsed:  list[JobDescription]      = []
    errors:  list[tuple[str, str]]     = []

    for filename, file_bytes in jd_files:
        try:
            jd = parse_jd_file(file_bytes, filename)
            if jd is not None:
                parsed.append(jd)
            else:
                errors.append((filename, "No text could be extracted from this file."))
        except Exception as exc:
            log.error("Failed to parse JD %s: %s", filename, exc)
            errors.append((filename, str(exc)))

    log.info(
        "Multi-JD parse complete: %d parsed, %d errors",
        len(parsed), len(errors),
    )
    return parsed, errors
