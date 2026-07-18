"""
InternLoom AI - General-purpose Helper Utilities
Reusable functions shared across parser, matcher, and UI modules.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Optional
import hashlib
import io

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise unicode, collapse whitespace, strip control characters.

    Args:
        text: Raw string to clean.

    Returns:
        Cleaned string.
    """
    if not text:
        return ""

    # Normalise unicode (NFKD → ASCII-compatible where possible)
    text = unicodedata.normalize("NFKD", text)
    # Encode to ascii bytes ignoring non-ascii, then decode back
    text = text.encode("ascii", "ignore").decode("ascii")
    # Replace non-printable control characters except newlines/tabs
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Return word count of a string."""
    return len(text.split()) if text.strip() else 0


def extract_lines(text: str) -> list[str]:
    """Split text into non-empty stripped lines."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


# ─────────────────────────────────────────────
# STRING MATCHING
# ─────────────────────────────────────────────

def normalize_skill(skill: str) -> str:
    """Lowercase, strip, collapse whitespace for skill comparison."""
    return re.sub(r"\s+", " ", skill.strip().lower())


def deduplicate_list(items: list[str]) -> list[str]:
    """
    Remove duplicates from a list while preserving insertion order.
    Case-insensitive deduplication.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def flatten_list(nested: list[Any]) -> list[Any]:
    """Recursively flatten a nested list."""
    result: list[Any] = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


# ─────────────────────────────────────────────
# SAFE VALUE EXTRACTION
# ─────────────────────────────────────────────

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely; return default on failure."""
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely; return default on failure."""
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a float between lo and hi."""
    return max(lo, min(hi, value))


def first_non_empty(*values: Any) -> Any:
    """Return first truthy value from args, or None."""
    for v in values:
        if v:
            return v
    return None


# ─────────────────────────────────────────────
# FILE UTILITIES
# ─────────────────────────────────────────────

def get_file_hash(content: bytes) -> str:
    """Return MD5 hex digest of file bytes — used for dedup."""
    return hashlib.md5(content).hexdigest()


def safe_filename(name: str) -> str:
    """Sanitise a string to be safe as a filename component."""
    name = re.sub(r"[^\w\s\-.]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def bytes_to_io(data: bytes) -> io.BytesIO:
    """Wrap bytes in a BytesIO object."""
    return io.BytesIO(data)


# ─────────────────────────────────────────────
# DATAFRAME UTILITIES
# ─────────────────────────────────────────────

def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert a list of dicts to a DataFrame, handling empty input."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def sort_by_score(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Sort DataFrame descending by score column."""
    if score_col in df.columns:
        return df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return df


def add_rank_column(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Add a 1-based Rank column sorted by score descending."""
    df = sort_by_score(df, score_col)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def status_badge(status: str) -> str:
    """Return an emoji badge for a shortlist status string."""
    mapping = {
        "Shortlisted": "✅",
        "Reserve": "🟡",
        "Rejected": "❌",
    }
    return mapping.get(status, "⬜")


def confidence_badge(confidence: str) -> str:
    """Return an emoji badge for confidence level."""
    mapping = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴",
    }
    return mapping.get(confidence, "⬜")


def score_color(score: float) -> str:
    """Return a hex color string based on score bracket."""
    if score >= 70:
        return "#26de81"   # green
    if score >= 50:
        return "#F7B731"   # amber
    return "#FF6584"       # red


def format_skills_list(skills: list[str], max_display: int = 15) -> str:
    """
    Format a skills list as a comma-separated string,
    truncating with a count if too long.
    """
    if not skills:
        return "—"
    if len(skills) <= max_display:
        return ", ".join(skills)
    shown = ", ".join(skills[:max_display])
    return f"{shown} (+{len(skills) - max_display} more)"


# ─────────────────────────────────────────────
# CANDIDATE DATA HELPERS
# ─────────────────────────────────────────────

def build_candidate_summary(candidate: dict) -> str:
    """
    Build a short human-readable summary string from a candidate dict.
    Used in debug logs and tooltips.
    """
    name = candidate.get("name", "Unknown")
    email = candidate.get("email", "—")
    score = candidate.get("score", 0)
    status = candidate.get("shortlist_status", "—")
    skills = candidate.get("matched_skills", [])
    return (
        f"{name} | {email} | Score: {score:.1f} | "
        f"Status: {status} | Skills: {len(skills)}"
    )


def merge_candidate_dicts(base: dict, update: dict) -> dict:
    """
    Merge two candidate dicts. Lists are combined (deduped),
    scalars from `update` overwrite `base`.
    """
    result = dict(base)
    for key, val in update.items():
        if key in result and isinstance(result[key], list) and isinstance(val, list):
            result[key] = deduplicate_list(result[key] + val)
        elif val:  # Only overwrite with non-empty values
            result[key] = val
    return result


# ─────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────

def detect_section_header(line: str, section_map: dict[str, list[str]]) -> Optional[str]:
    """
    Check if a line is a section header.

    Args:
        line: Stripped text line from resume.
        section_map: Dict mapping section names to keyword lists.

    Returns:
        Section name if matched, else None.
    """
    line_lower = line.lower().strip(" :-•|")
    for section, keywords in section_map.items():
        for kw in keywords:
            if line_lower == kw or line_lower.startswith(kw):
                return section
    return None


def extract_year(text: str) -> Optional[int]:
    """Extract the first 4-digit year (2000-2035) from text."""
    match = re.search(r"\b(20[0-3]\d)\b", text)
    if match:
        return int(match.group(1))
    return None


def is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email))


def is_valid_phone(phone: str) -> bool:
    """Check phone has at least 10 digits."""
    digits = re.sub(r"\D", "", phone)
    return 10 <= len(digits) <= 13
