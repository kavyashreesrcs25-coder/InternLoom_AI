"""
InternLoom AI - Structured Field Extractor
Extracts structured candidate information from raw resume text.

Fields extracted:
  name, email, phone, college, degree, branch, graduation_year,
  cgpa, percentage, gpa, skills, projects, internship,
  work_experience, certifications, github, linkedin, portfolio

Strategy:
  - Regex for deterministic fields (email, phone, URLs, grades)
  - spaCy NER for name detection
  - Section-aware parsing for skills, projects, experience, certifications
  - Full-text skill scan (skills appear anywhere, not just skill section)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger
from utils.helper import (
    clean_text,
    normalize_whitespace,
    extract_lines,
    deduplicate_list,
    detect_section_header,
    is_valid_email,
    is_valid_phone,
    extract_year,
)
from config import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    GITHUB_PATTERN,
    LINKEDIN_PATTERN,
    PORTFOLIO_PATTERN,
    CGPA_PATTERN,
    PERCENTAGE_PATTERN,
    GRAD_YEAR_PATTERN,
    SECTION_HEADERS,
    KNOWN_SKILLS,
    ALIAS_TO_CANONICAL,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# CANDIDATE DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class CandidateData:
    """All structured fields extracted from one resume."""
    filename: str = ""
    name: str = ""
    email: str = ""
    phone: str = ""
    college: str = ""
    degree: str = ""
    branch: str = ""
    graduation_year: Optional[int] = None
    raw_cgpa: Optional[float] = None
    raw_gpa: Optional[float] = None
    raw_percentage: Optional[float] = None
    grade_scale: str = ""          # "10", "4", "percentage", "unknown"
    normalized_cgpa: Optional[float] = None   # Always on 10-pt scale
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    internships: list[str] = field(default_factory=list)
    work_experience: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    github: str = ""
    linkedin: str = ""
    portfolio: str = ""
    raw_sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "college": self.college,
            "degree": self.degree,
            "branch": self.branch,
            "graduation_year": self.graduation_year,
            "raw_cgpa": self.raw_cgpa,
            "raw_gpa": self.raw_gpa,
            "raw_percentage": self.raw_percentage,
            "grade_scale": self.grade_scale,
            "normalized_cgpa": self.normalized_cgpa,
            "skills": self.skills,
            "projects": self.projects,
            "internships": self.internships,
            "work_experience": self.work_experience,
            "certifications": self.certifications,
            "github": self.github,
            "linkedin": self.linkedin,
            "portfolio": self.portfolio,
        }


# ─────────────────────────────────────────────
# SPACY LOADER (lazy, singleton)
# ─────────────────────────────────────────────

_nlp = None

def _get_nlp():
    """Lazy-load spaCy English model."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            log.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            _nlp = None
    except ImportError:
        log.warning("spaCy not installed — name extraction will use heuristics only.")
        _nlp = None
    return _nlp


# ─────────────────────────────────────────────
# REGEX EXTRACTORS
# ─────────────────────────────────────────────

def _extract_email(text: str) -> str:
    matches = re.findall(EMAIL_PATTERN, text)
    for m in matches:
        if is_valid_email(m):
            return m.lower()
    return ""


def _extract_phone(text: str) -> str:
    matches = re.findall(PHONE_PATTERN, text)
    for m in matches:
        digits = re.sub(r"\D", "", m)
        if is_valid_phone(m) and len(digits) >= 10:
            return m.strip()
    return ""


def _extract_github(text: str) -> str:
    match = re.search(GITHUB_PATTERN, text, re.IGNORECASE)
    return match.group(0) if match else ""


def _extract_linkedin(text: str) -> str:
    match = re.search(LINKEDIN_PATTERN, text, re.IGNORECASE)
    return match.group(0) if match else ""


def _extract_portfolio(text: str) -> str:
    # Remove github and linkedin matches first
    sanitised = re.sub(GITHUB_PATTERN, "", text, flags=re.IGNORECASE)
    sanitised = re.sub(LINKEDIN_PATTERN, "", sanitised, flags=re.IGNORECASE)
    match = re.search(PORTFOLIO_PATTERN, sanitised, re.IGNORECASE)
    return match.group(0) if match else ""


def _extract_graduation_year(text: str) -> Optional[int]:
    """Find graduation year heuristically (latest year in education section)."""
    years = [int(y) for y in re.findall(GRAD_YEAR_PATTERN, text)]
    return max(years) if years else None


def _extract_grade(text: str) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Extract CGPA / GPA / Percentage and detect scale.

    Returns:
        (raw_cgpa, raw_gpa, raw_percentage, scale)
        scale: "10" | "4" | "percentage" | "unknown"
    """
    raw_cgpa: Optional[float] = None
    raw_gpa: Optional[float] = None
    raw_percentage: Optional[float] = None
    scale = "unknown"

    text_lower = text.lower()

    # ── CGPA patterns ──────────────────────────────────────────────────────
    cgpa_matches = re.findall(CGPA_PATTERN, text_lower, re.IGNORECASE)
    for value_str, max_str in cgpa_matches:
        try:
            value = float(value_str)
            max_val = float(max_str) if max_str else None

            if max_val and abs(max_val - 10.0) < 0.1:
                raw_cgpa = value
                scale = "10"
                break
            elif max_val and abs(max_val - 4.0) < 0.1:
                raw_gpa = value
                scale = "4"
                break
            elif value <= 4.0:
                raw_gpa = value
                scale = "4"
                break
            elif value <= 10.0:
                raw_cgpa = value
                scale = "10"
                break
        except ValueError:
            continue

    # ── Percentage ────────────────────────────────────────────────────────
    pct_matches = re.findall(PERCENTAGE_PATTERN, text_lower)
    for m in pct_matches:
        try:
            val = float(m)
            if 30.0 <= val <= 100.0:
                raw_percentage = val
                if scale == "unknown":
                    scale = "percentage"
                break
        except ValueError:
            continue

    return raw_cgpa, raw_gpa, raw_percentage, scale


# ─────────────────────────────────────────────
# NAME EXTRACTION
# ─────────────────────────────────────────────

_NAME_BLACKLIST = {
    "resume", "curriculum vitae", "cv", "profile", "contact",
    "objective", "summary", "skills", "education", "experience",
    "projects", "certifications", "references", "page",
    "phone", "email", "address", "linkedin", "github",
}

_DEGREE_KEYWORDS = {
    "b.tech", "m.tech", "btech", "mtech", "b.e", "m.e", "be", "me",
    "bca", "mca", "bsc", "msc", "b.sc", "m.sc", "bba", "mba",
    "b.com", "m.com", "phd", "ph.d", "bachelor", "master", "engineering",
}


def _extract_name_spacy(text: str) -> str:
    """Use spaCy NER to find PERSON entities near the top of the document."""
    nlp = _get_nlp()
    if nlp is None:
        return ""

    # Only analyse first 500 chars (name is usually at the very top)
    top_text = text[:500]
    try:
        doc = nlp(top_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if (
                    2 <= len(name.split()) <= 4
                    and name.lower() not in _NAME_BLACKLIST
                    and not any(kw in name.lower() for kw in _DEGREE_KEYWORDS)
                ):
                    return name
    except Exception as exc:
        log.debug("spaCy NER error: %s", exc)
    return ""


def _extract_name_heuristic(text: str) -> str:
    """
    Heuristic fallback: scan first few non-empty lines for a
    capitalised 2–3 word name that doesn't look like a section header.
    """
    lines = extract_lines(text)
    for line in lines[:10]:
        # Skip lines with common resume keywords
        line_lower = line.lower()
        if any(kw in line_lower for kw in _NAME_BLACKLIST):
            continue
        if any(kw in line_lower for kw in _DEGREE_KEYWORDS):
            continue
        # Skip lines with @, http, digits
        if re.search(r"[@|http|www|\d]", line):
            continue
        # Must be 2–4 words, each capitalised
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            # No short connector-only words that aren't names
            if not all(len(w) <= 2 for w in words):
                return line.strip()
    return ""


def _extract_name(text: str) -> str:
    name = _extract_name_spacy(text)
    if not name:
        name = _extract_name_heuristic(text)
    return name


# ─────────────────────────────────────────────
# EDUCATION EXTRACTION
# ─────────────────────────────────────────────

_DEGREE_PATTERNS = [
    (r"\b(b\.?tech|bachelor\s+of\s+technology)\b", "B.Tech"),
    (r"\b(m\.?tech|master\s+of\s+technology)\b", "M.Tech"),
    (r"\b(b\.?e\.?|bachelor\s+of\s+engineering)\b", "B.E"),
    (r"\b(m\.?e\.?|master\s+of\s+engineering)\b", "M.E"),
    (r"\b(bca|bachelor\s+of\s+computer\s+applications)\b", "BCA"),
    (r"\b(mca|master\s+of\s+computer\s+applications)\b", "MCA"),
    (r"\b(b\.?sc|bachelor\s+of\s+science)\b", "B.Sc"),
    (r"\b(m\.?sc|master\s+of\s+science)\b", "M.Sc"),
    (r"\b(bba|bachelor\s+of\s+business\s+administration)\b", "BBA"),
    (r"\b(mba|master\s+of\s+business\s+administration)\b", "MBA"),
    (r"\b(b\.?com)\b", "B.Com"),
    (r"\b(ph\.?d|doctor\s+of\s+philosophy)\b", "PhD"),
]

_BRANCH_PATTERNS = [
    r"computer\s+science(?:\s+(?:and\s+)?engineering)?",
    r"information\s+technology",
    r"electronics?\s+(?:and\s+)?(?:communication|electrical)",
    r"mechanical\s+engineering",
    r"civil\s+engineering",
    r"electrical\s+engineering",
    r"data\s+science",
    r"artificial\s+intelligence",
    r"machine\s+learning",
    r"cyber\s+security",
    r"software\s+engineering",
]

_UNIVERSITY_KEYWORDS = [
    "university", "institute", "college", "iit", "nit", "bits",
    "iiit", "school of", "faculty of", "polytechnic",
]


def _extract_education(text: str) -> tuple[str, str, str]:
    """
    Extract degree, branch, and college from text.

    Returns:
        (degree, branch, college)
    """
    text_lower = text.lower()

    # Degree
    degree = ""
    for pattern, label in _DEGREE_PATTERNS:
        if re.search(pattern, text_lower):
            degree = label
            break

    # Branch
    branch = ""
    for pattern in _BRANCH_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            branch = match.group(0).title()
            break

    # College — find line containing university keyword
    college = ""
    for line in extract_lines(text):
        line_lower = line.lower()
        if any(kw in line_lower for kw in _UNIVERSITY_KEYWORDS):
            # Clean up: remove parenthetical years, extra spaces
            cleaned = re.sub(r"\(.*?\)", "", line).strip()
            if 5 < len(cleaned) < 100:
                college = cleaned
                break

    return degree, branch, college


# ─────────────────────────────────────────────
# SECTION SPLITTER
# ─────────────────────────────────────────────

def _split_into_sections(text: str) -> dict[str, str]:
    """
    Split resume text into labelled sections using SECTION_HEADERS.

    Returns:
        Dict mapping section name → section text content.
    """
    sections: dict[str, str] = {"preamble": ""}
    current_section = "preamble"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        detected = detect_section_header(stripped, SECTION_HEADERS)
        if detected:
            # Save previous section
            sections[current_section] = "\n".join(current_lines).strip()
            current_section = detected
            current_lines = []
        else:
            current_lines.append(stripped)

    # Save last section
    sections[current_section] = "\n".join(current_lines).strip()
    return sections


# ─────────────────────────────────────────────
# SKILL EXTRACTION
# ─────────────────────────────────────────────

def _extract_skills_from_text(text: str) -> list[str]:
    """
    Extract skills from ANY section of the text using:
      1. Exact case-insensitive match against KNOWN_SKILLS
      2. Alias resolution via ALIAS_TO_CANONICAL

    Skills are detected in free text, not just bullet-point lists.

    Args:
        text: Full resume text or any section text.

    Returns:
        Deduplicated list of canonical skill names.
    """
    found: set[str] = set()
    text_lower = text.lower()

    for skill in KNOWN_SKILLS:
        # Use word-boundary aware matching
        pattern = r"(?<![a-zA-Z0-9\.\+\#])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9\.\+\#])"
        if re.search(pattern, text_lower):
            found.add(skill)

    # Check aliases
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        pattern = r"(?<![a-zA-Z0-9\.\+\#])" + re.escape(alias.lower()) + r"(?![a-zA-Z0-9\.\+\#])"
        if re.search(pattern, text_lower):
            found.add(canonical)

    # Also scan comma/bullet separated skill lists
    skill_list_pattern = re.compile(
        r"(?:^|\n)\s*(?:[-•*▪►✓]|\d+\.?)\s*(.+?)(?:\n|$)", re.MULTILINE
    )
    for match in skill_list_pattern.finditer(text):
        token = match.group(1).strip()
        # Split on separators
        for part in re.split(r"[,|/;]", token):
            part = part.strip()
            if not part:
                continue
            part_lower = part.lower()
            # Check canonical match
            for skill in KNOWN_SKILLS:
                if part_lower == skill.lower():
                    found.add(skill)
                    break
            # Check alias
            if part_lower in ALIAS_TO_CANONICAL:
                found.add(ALIAS_TO_CANONICAL[part_lower])

    return sorted(found)


# ─────────────────────────────────────────────
# EXPERIENCE / PROJECTS / CERTIFICATIONS
# ─────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*[-•*▪►✓]\s+", re.MULTILINE)
_NUMBERING_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def _extract_items_from_section(section_text: str, max_items: int = 10) -> list[str]:
    """
    Extract bullet-pointed or numbered items from a section.
    Falls back to splitting by double-newline if no bullets found.
    """
    if not section_text:
        return []

    items: list[str] = []

    # Try bullet points
    bullets = _BULLET_RE.split(section_text)
    if len(bullets) > 1:
        for b in bullets[1:]:  # Skip text before first bullet
            b = normalize_whitespace(b)
            if b and len(b) > 5:
                items.append(b)
    else:
        # Try numbered
        numbered = _NUMBERING_RE.split(section_text)
        if len(numbered) > 1:
            for n in numbered[1:]:
                n = normalize_whitespace(n)
                if n and len(n) > 5:
                    items.append(n)
        else:
            # Fall back to paragraph splits
            paras = [p.strip() for p in re.split(r"\n{2,}", section_text) if p.strip()]
            items = [p for p in paras if len(p) > 10]

    return deduplicate_list(items[:max_items])


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class ResumeExtractor:
    """
    Extracts all structured fields from raw resume text.

    Usage:
        extractor = ResumeExtractor()
        candidate = extractor.extract("raw text", "resume.pdf")
    """

    def extract(self, text: str, filename: str = "") -> CandidateData:
        """
        Run full extraction pipeline on resume text.

        Args:
            text: Raw (cleaned) resume text.
            filename: Source filename for reference.

        Returns:
            CandidateData with all extracted fields populated.
        """
        if not text or not text.strip():
            log.warning("Empty text provided for extraction: %s", filename)
            return CandidateData(filename=filename)

        text = clean_text(text)
        candidate = CandidateData(filename=filename)

        # ── Split into sections ────────────────────────────────────────────
        sections = _split_into_sections(text)
        candidate.raw_sections = {k: v for k, v in sections.items() if v}

        preamble = sections.get("preamble", "")
        full_text_for_regex = text  # Use full text for regex patterns

        # ── Contact info ──────────────────────────────────────────────────
        candidate.email = _extract_email(full_text_for_regex)
        candidate.phone = _extract_phone(full_text_for_regex)
        candidate.github = _extract_github(full_text_for_regex)
        candidate.linkedin = _extract_linkedin(full_text_for_regex)
        candidate.portfolio = _extract_portfolio(full_text_for_regex)

        # ── Name ──────────────────────────────────────────────────────────
        candidate.name = _extract_name(preamble or text[:600])

        # ── Education ─────────────────────────────────────────────────────
        edu_text = sections.get("education", "") or text
        candidate.degree, candidate.branch, candidate.college = _extract_education(edu_text)
        candidate.graduation_year = _extract_graduation_year(edu_text)

        # ── Grades ────────────────────────────────────────────────────────
        (
            candidate.raw_cgpa,
            candidate.raw_gpa,
            candidate.raw_percentage,
            candidate.grade_scale,
        ) = _extract_grade(text)

        # ── Skills — scan entire resume ────────────────────────────────────
        candidate.skills = _extract_skills_from_text(text)

        # ── Projects ─────────────────────────────────────────────────────
        candidate.projects = _extract_items_from_section(
            sections.get("projects", "")
        )

        # ── Internships ───────────────────────────────────────────────────
        candidate.internships = _extract_items_from_section(
            sections.get("internship", "")
        )

        # ── Work Experience ───────────────────────────────────────────────
        candidate.work_experience = _extract_items_from_section(
            sections.get("experience", "")
        )

        # ── Certifications ────────────────────────────────────────────────
        candidate.certifications = _extract_items_from_section(
            sections.get("certifications", "")
        )

        log.debug(
            "Extracted [%s]: name=%r, email=%r, skills=%d, skills_list=%s",
            filename, candidate.name, candidate.email,
            len(candidate.skills), candidate.skills[:5],
        )

        return candidate
