"""
InternLoom AI - Skill Matcher
Three-tier skill matching engine:

  Tier 1 — Exact match (case-insensitive)
  Tier 2 — Synonym / alias match via ALIAS_TO_CANONICAL + RapidFuzz
  Tier 3 — Semantic similarity via Sentence-Transformers cosine distance

Each matched skill carries a match_type and a score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from utils.logger import get_logger
from utils.helper import normalize_skill, deduplicate_list
from config import (
    ALIAS_TO_CANONICAL,
    SKILL_ALIASES,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_SIMILARITY_THRESHOLD,
    FUZZY_MATCH_THRESHOLD,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class SkillMatch:
    """Result of matching one JD skill against a candidate's skill set."""
    jd_skill: str                  # Skill required in JD
    matched_skill: Optional[str]   # Candidate skill that matched (or None)
    match_type: str                # "exact" | "alias" | "fuzzy" | "semantic" | "none"
    score: float                   # Match strength [0.0 – 1.0]

    @property
    def matched(self) -> bool:
        return self.match_type != "none"


@dataclass
class MatchResult:
    """Aggregated matching results between a candidate and a JD."""
    matched_skills: list[str]
    missing_skills: list[str]
    skill_matches: list[SkillMatch]
    match_score: float             # 0.0 – 1.0 (fraction of JD skills matched)
    semantic_boost: float          # Additional score from semantic near-matches


# ─────────────────────────────────────────────
# SENTENCE TRANSFORMER — lazy singleton
# ─────────────────────────────────────────────

_embedding_model = None
_embedding_cache: dict[str, list[float]] = {}


def _get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence transformer: %s", SEMANTIC_MODEL_NAME)
        _embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        log.info("Sentence transformer loaded.")
    except ImportError:
        log.warning("sentence-transformers not installed — semantic matching disabled.")
    except Exception as exc:
        log.error("Failed to load sentence transformer: %s", exc)
    return _embedding_model


def _get_embedding(text: str) -> Optional[list[float]]:
    """Return (cached) embedding vector for a text string."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, convert_to_numpy=True).tolist()
        _embedding_cache[text] = vec
        return vec
    except Exception as exc:
        log.debug("Embedding error for %r: %s", text, exc)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        import numpy as np
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# TIER 1 — EXACT MATCH
# ─────────────────────────────────────────────

def _exact_match(jd_skill: str, candidate_skills_lower: set[str]) -> Optional[str]:
    """
    Case-insensitive exact match.

    Returns:
        The original-cased candidate skill if matched, else None.
    """
    return jd_skill if normalize_skill(jd_skill) in candidate_skills_lower else None


# ─────────────────────────────────────────────
# TIER 2 — ALIAS / SYNONYM MATCH
# ─────────────────────────────────────────────

def _build_alias_index(candidate_skills: list[str]) -> dict[str, str]:
    """
    Build a dict: normalised_alias → original_candidate_skill
    for every alias of every candidate skill.
    """
    index: dict[str, str] = {}
    for skill in candidate_skills:
        skill_norm = normalize_skill(skill)
        index[skill_norm] = skill  # self-reference

        # Canonical → aliases
        canonical = ALIAS_TO_CANONICAL.get(skill_norm, skill)
        canonical_norm = normalize_skill(canonical)
        index[canonical_norm] = skill

        # If this skill is a canonical, add all its aliases
        for canon, aliases in SKILL_ALIASES.items():
            if normalize_skill(canon) == skill_norm:
                for alias in aliases:
                    index[normalize_skill(alias)] = skill

    return index


def _alias_match(
    jd_skill: str,
    alias_index: dict[str, str],
) -> Optional[str]:
    """
    Check if a JD skill matches a candidate skill via alias table.

    Returns:
        Matched candidate skill string, or None.
    """
    jd_norm = normalize_skill(jd_skill)

    # Direct alias lookup
    if jd_norm in alias_index:
        return alias_index[jd_norm]

    # Resolve JD skill's own canonical and check that
    canonical = ALIAS_TO_CANONICAL.get(jd_norm)
    if canonical:
        canonical_norm = normalize_skill(canonical)
        if canonical_norm in alias_index:
            return alias_index[canonical_norm]

    return None


# ─────────────────────────────────────────────
# TIER 2b — FUZZY MATCH (RapidFuzz)
# ─────────────────────────────────────────────

def _fuzzy_match(
    jd_skill: str,
    candidate_skills: list[str],
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """
    Token-sort ratio fuzzy match using RapidFuzz.

    Returns:
        (matched_skill, score_0_to_1) or None.
    """
    try:
        from rapidfuzz import fuzz, process

        jd_norm = normalize_skill(jd_skill)
        candidates_norm = [normalize_skill(s) for s in candidate_skills]

        result = process.extractOne(
            jd_norm,
            candidates_norm,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result:
            best_match_norm, score, idx = result
            return candidate_skills[idx], score / 100.0

    except ImportError:
        log.debug("RapidFuzz not installed — fuzzy matching skipped.")
    except Exception as exc:
        log.debug("Fuzzy match error: %s", exc)

    return None


# ─────────────────────────────────────────────
# TIER 3 — SEMANTIC MATCH
# ─────────────────────────────────────────────

def _semantic_match(
    jd_skill: str,
    candidate_skills: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """
    Find the most semantically similar candidate skill using cosine similarity
    of sentence-transformer embeddings.

    Returns:
        (best_matched_skill, similarity_score) or None.
    """
    jd_vec = _get_embedding(jd_skill)
    if jd_vec is None or not candidate_skills:
        return None

    best_skill: Optional[str] = None
    best_score = 0.0

    for skill in candidate_skills:
        skill_vec = _get_embedding(skill)
        if skill_vec is None:
            continue
        sim = _cosine_similarity(jd_vec, skill_vec)
        if sim > best_score:
            best_score = sim
            best_skill = skill

    if best_score >= threshold and best_skill:
        return best_skill, best_score

    return None


# ─────────────────────────────────────────────
# MAIN SKILL MATCHER
# ─────────────────────────────────────────────

class SkillMatcher:
    """
    Three-tier skill matching engine.

    For each skill required/preferred in the JD, runs:
      1. Exact match
      2. Alias/synonym match
      3. RapidFuzz token-sort match
      4. Semantic similarity match (if ST model available)

    Returns a MatchResult with all matched/missing skills and scores.
    """

    def __init__(
        self,
        use_semantic: bool = True,
        fuzzy_threshold: int = FUZZY_MATCH_THRESHOLD,
        semantic_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    ) -> None:
        self.use_semantic = use_semantic
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold

        # Pre-warm semantic model in background
        if use_semantic:
            _get_embedding_model()

    def match(
        self,
        jd_skills: list[str],
        candidate_skills: list[str],
    ) -> MatchResult:
        """
        Match JD skills against candidate skills using all tiers.

        Args:
            jd_skills: Skills required/preferred in the JD.
            candidate_skills: Skills found in the candidate's resume.

        Returns:
            MatchResult with match details and aggregate score.
        """
        if not jd_skills:
            return MatchResult(
                matched_skills=[],
                missing_skills=[],
                skill_matches=[],
                match_score=0.0,
                semantic_boost=0.0,
            )

        # Build lookup structures
        candidate_skills_clean = deduplicate_list(candidate_skills)
        candidate_lower_set = {normalize_skill(s) for s in candidate_skills_clean}
        alias_index = _build_alias_index(candidate_skills_clean)

        skill_matches: list[SkillMatch] = []
        matched_set: set[str] = set()
        semantic_scores: list[float] = []

        for jd_skill in jd_skills:
            sm = self._match_single(
                jd_skill=jd_skill,
                candidate_skills=candidate_skills_clean,
                candidate_lower_set=candidate_lower_set,
                alias_index=alias_index,
            )
            skill_matches.append(sm)
            if sm.matched:
                matched_set.add(jd_skill)
                if sm.match_type == "semantic":
                    semantic_scores.append(sm.score)

        matched_skills = [sm.jd_skill for sm in skill_matches if sm.matched]
        missing_skills = [sm.jd_skill for sm in skill_matches if not sm.matched]

        match_score = len(matched_skills) / len(jd_skills) if jd_skills else 0.0
        semantic_boost = (
            (sum(semantic_scores) / len(semantic_scores)) * 0.1
            if semantic_scores else 0.0
        )

        return MatchResult(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            skill_matches=skill_matches,
            match_score=round(match_score, 4),
            semantic_boost=round(semantic_boost, 4),
        )

    def _match_single(
        self,
        jd_skill: str,
        candidate_skills: list[str],
        candidate_lower_set: set[str],
        alias_index: dict[str, str],
    ) -> SkillMatch:
        """Run all tiers for a single JD skill."""

        # Tier 1 — Exact
        exact = _exact_match(jd_skill, candidate_lower_set)
        if exact:
            return SkillMatch(
                jd_skill=jd_skill,
                matched_skill=exact,
                match_type="exact",
                score=1.0,
            )

        # Tier 2 — Alias
        alias = _alias_match(jd_skill, alias_index)
        if alias:
            return SkillMatch(
                jd_skill=jd_skill,
                matched_skill=alias,
                match_type="alias",
                score=0.95,
            )

        # Tier 2b — Fuzzy
        fuzzy = _fuzzy_match(jd_skill, candidate_skills, self.fuzzy_threshold)
        if fuzzy:
            matched_skill, score = fuzzy
            return SkillMatch(
                jd_skill=jd_skill,
                matched_skill=matched_skill,
                match_type="fuzzy",
                score=score,
            )

        # Tier 3 — Semantic
        if self.use_semantic:
            semantic = _semantic_match(jd_skill, candidate_skills, self.semantic_threshold)
            if semantic:
                matched_skill, score = semantic
                return SkillMatch(
                    jd_skill=jd_skill,
                    matched_skill=matched_skill,
                    match_type="semantic",
                    score=score,
                )

        return SkillMatch(
            jd_skill=jd_skill,
            matched_skill=None,
            match_type="none",
            score=0.0,
        )

    def match_required_and_preferred(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        candidate_skills: list[str],
    ) -> tuple[MatchResult, MatchResult]:
        """
        Separately match required and preferred skills.

        Returns:
            (required_match_result, preferred_match_result)
        """
        required_match = self.match(required_skills, candidate_skills)
        preferred_match = self.match(preferred_skills, candidate_skills)
        return required_match, preferred_match
