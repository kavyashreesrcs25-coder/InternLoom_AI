"""
InternLoom AI - Weighted Scoring Engine
Computes a 0–100 score for each candidate against a job description.

Default weight allocation (sums to 100):
  Required Skills  = 40
  Preferred Skills = 20
  Projects         = 15
  CGPA             = 10
  Internship       = 10
  Certifications   =  5

Weights auto-adjust when a component is unavailable (e.g. no CGPA
extracted), redistributing that weight proportionally among the rest.

Score breakdown is returned as a dict so the UI can render
per-component bars and explanation reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger
from utils.helper import clamp, safe_float
from parser.extractor import CandidateData
from parser.normalizer import JobDescription
from matcher.skill_matcher import SkillMatcher, MatchResult
from config import SCORING_WEIGHTS, CGPA_10_MAX

log = get_logger(__name__)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ComponentScore:
    """Score for a single scoring component."""
    name: str
    raw_score: float       # 0.0 – 1.0 (fraction earned within component)
    weight: float          # Actual weight used after adjustment
    weighted_score: float  # raw_score × weight (contributes to total /100)
    detail: str            # Human-readable explanation


@dataclass
class ScoreBreakdown:
    """Complete scoring result for one candidate."""
    total_score: float                         # 0 – 100
    components: list[ComponentScore]
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    reasons: list[str]                         # 3 top explanation sentences
    weights_used: dict[str, float]
    required_match_result: Optional[MatchResult] = None
    preferred_match_result: Optional[MatchResult] = None


# ─────────────────────────────────────────────
# WEIGHT ADJUSTER
# ─────────────────────────────────────────────

def _adjust_weights(
    base_weights: dict[str, float],
    available: dict[str, bool],
) -> dict[str, float]:
    """
    Redistribute weights for unavailable components proportionally.

    If a component has no data (available[key]=False), its weight is
    distributed among all available components in proportion to their
    original weights.

    Args:
        base_weights: Original weight mapping.
        available: Bool per key — whether that component has data.

    Returns:
        New weights dict that sums to 100.
    """
    unavailable_weight = sum(
        w for k, w in base_weights.items() if not available.get(k, True)
    )
    if unavailable_weight == 0:
        return dict(base_weights)

    available_total = sum(
        w for k, w in base_weights.items() if available.get(k, True)
    )
    if available_total == 0:
        # All unavailable — equal distribution across all
        per = 100.0 / len(base_weights)
        return {k: per for k in base_weights}

    adjusted: dict[str, float] = {}
    for key, weight in base_weights.items():
        if not available.get(key, True):
            adjusted[key] = 0.0
        else:
            # Scale up proportionally
            adjusted[key] = weight + (weight / available_total) * unavailable_weight

    # Normalise to exactly 100
    total = sum(adjusted.values())
    if total > 0:
        factor = 100.0 / total
        adjusted = {k: round(v * factor, 4) for k, v in adjusted.items()}

    return adjusted


# ─────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────

def _score_required_skills(
    match_result: MatchResult, weight: float
) -> ComponentScore:
    raw = match_result.match_score + match_result.semantic_boost
    raw = clamp(raw, 0.0, 1.0)
    n_matched = len(match_result.matched_skills)
    n_total = n_matched + len(match_result.missing_skills)
    detail = f"Matched {n_matched}/{n_total} required skills"
    return ComponentScore(
        name="Required Skills",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=detail,
    )


def _score_preferred_skills(
    match_result: MatchResult, weight: float
) -> ComponentScore:
    raw = match_result.match_score + match_result.semantic_boost
    raw = clamp(raw, 0.0, 1.0)
    n_matched = len(match_result.matched_skills)
    n_total = n_matched + len(match_result.missing_skills)
    detail = f"Matched {n_matched}/{n_total} preferred skills"
    return ComponentScore(
        name="Preferred Skills",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=detail,
    )


def _score_projects(candidate: CandidateData, weight: float) -> ComponentScore:
    """
    Score based on number of projects present.
    0 projects = 0, 1 = 0.4, 2 = 0.7, 3+ = 1.0
    """
    n = len(candidate.projects) + len(candidate.internships)
    if n == 0:
        raw = 0.0
        detail = "No projects found in resume"
    elif n == 1:
        raw = 0.4
        detail = "1 project listed"
    elif n == 2:
        raw = 0.7
        detail = f"{n} projects listed"
    else:
        raw = 1.0
        detail = f"{n} projects listed"
    return ComponentScore(
        name="Projects",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=detail,
    )


def _score_cgpa(
    candidate: CandidateData,
    jd: JobDescription,
    weight: float,
) -> ComponentScore:
    """
    Score based on normalised CGPA (10-point scale).
    If JD specifies min_cgpa, uses that as the floor reference.
    Score = (cgpa / 10) with bonus for exceeding JD minimum.
    """
    cgpa = candidate.normalized_cgpa
    if cgpa is None:
        return ComponentScore(
            name="CGPA",
            raw_score=0.0,
            weight=weight,
            weighted_score=0.0,
            detail="CGPA not found in resume",
        )

    # Base score: linear 0–10 → 0–1
    raw = clamp(cgpa / CGPA_10_MAX, 0.0, 1.0)

    detail_parts = [f"CGPA: {cgpa:.2f}/10"]

    # Adjust for JD minimum
    if jd.min_cgpa:
        if cgpa < jd.min_cgpa:
            penalty = (jd.min_cgpa - cgpa) / jd.min_cgpa * 0.3
            raw = clamp(raw - penalty, 0.0, 1.0)
            detail_parts.append(f"(below JD minimum {jd.min_cgpa:.1f})")
        else:
            detail_parts.append(f"(meets JD minimum {jd.min_cgpa:.1f})")

    return ComponentScore(
        name="CGPA",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=" ".join(detail_parts),
    )


def _score_internship(candidate: CandidateData, weight: float) -> ComponentScore:
    """
    Score based on internship / work experience presence.
    Internships: 0=0, 1=0.6, 2+=1.0
    Work experience also counts.
    """
    total = len(candidate.internships) + len(candidate.work_experience)
    if total == 0:
        raw = 0.0
        detail = "No internship or work experience found"
    elif total == 1:
        raw = 0.6
        detail = "1 internship/work experience listed"
    else:
        raw = 1.0
        detail = f"{total} internships/experiences listed"
    return ComponentScore(
        name="Internship",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=detail,
    )


def _score_certifications(candidate: CandidateData, weight: float) -> ComponentScore:
    """0 certs = 0, 1 = 0.5, 2+ = 1.0"""
    n = len(candidate.certifications)
    if n == 0:
        raw = 0.0
        detail = "No certifications found"
    elif n == 1:
        raw = 0.5
        detail = "1 certification listed"
    else:
        raw = 1.0
        detail = f"{n} certifications listed"
    return ComponentScore(
        name="Certifications",
        raw_score=raw,
        weight=weight,
        weighted_score=round(raw * weight, 2),
        detail=detail,
    )


# ─────────────────────────────────────────────
# REASON GENERATOR
# ─────────────────────────────────────────────

def _generate_reasons(
    breakdown: "ScoreBreakdown",
    candidate: CandidateData,
    jd: JobDescription,
) -> list[str]:
    """
    Generate exactly 3 human-readable explanation sentences for the score.
    Used in the UI cards and reports.
    """
    reasons: list[str] = []

    # Reason 1: Skills
    n_req = len(breakdown.matched_required)
    n_req_total = n_req + len(breakdown.missing_required)
    n_pref = len(breakdown.matched_preferred)

    if n_req_total > 0:
        pct = int((n_req / n_req_total) * 100)
        if pct >= 80:
            reasons.append(
                f"Strong skill alignment — matched {n_req}/{n_req_total} "
                f"required skills ({pct}%) including "
                f"{', '.join(breakdown.matched_required[:3])}."
            )
        elif pct >= 50:
            missing_str = (
                ", ".join(breakdown.missing_required[:3]) or "none listed"
            )
            reasons.append(
                f"Partial skill match — met {n_req}/{n_req_total} required skills. "
                f"Missing: {missing_str}."
            )
        else:
            missing_str = ", ".join(breakdown.missing_required[:4]) or "most required"
            reasons.append(
                f"Weak skill alignment — only {n_req}/{n_req_total} required skills matched. "
                f"Key gaps: {missing_str}."
            )
    else:
        reasons.append("No required skills specified in JD — skill match not assessed.")

    # Reason 2: Academic / Experience
    cgpa_str = f"{candidate.normalized_cgpa:.2f}" if candidate.normalized_cgpa else "N/A"
    exp_count = len(candidate.internships) + len(candidate.work_experience)
    proj_count = len(candidate.projects)

    academic_parts = [f"CGPA: {cgpa_str}/10"]
    if exp_count > 0:
        academic_parts.append(f"{exp_count} internship(s)/experience(s)")
    if proj_count > 0:
        academic_parts.append(f"{proj_count} project(s)")
    reasons.append("Academic & experience profile — " + ", ".join(academic_parts) + ".")

    # Reason 3: Overall assessment
    score = breakdown.total_score
    if score >= 75:
        reasons.append(
            f"Overall strong candidate with {score:.1f}/100 — "
            f"recommended for interview shortlisting."
        )
    elif score >= 55:
        reasons.append(
            f"Moderate fit with {score:.1f}/100 — "
            f"consider for reserve pool pending stronger candidates."
        )
    else:
        reasons.append(
            f"Below threshold with {score:.1f}/100 — "
            f"significant skill or experience gaps relative to JD requirements."
        )

    return reasons


# ─────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────

class CandidateScorer:
    """
    Computes a weighted 0–100 score for a candidate against a JD.

    Usage:
        scorer = CandidateScorer()
        breakdown = scorer.score(candidate, jd)
    """

    def __init__(self, use_semantic: bool = True) -> None:
        self.skill_matcher = SkillMatcher(use_semantic=use_semantic)
        self.base_weights = dict(SCORING_WEIGHTS)

    def score(
        self,
        candidate: CandidateData,
        jd: JobDescription,
    ) -> ScoreBreakdown:
        """
        Score a single candidate against the job description.

        Args:
            candidate: Normalised CandidateData.
            jd:        Parsed JobDescription.

        Returns:
            ScoreBreakdown with total score and per-component details.
        """
        # ── Skill matching ─────────────────────────────────────────────────
        req_match, pref_match = self.skill_matcher.match_required_and_preferred(
            required_skills=jd.required_skills,
            preferred_skills=jd.preferred_skills,
            candidate_skills=candidate.skills,
        )

        # ── Determine which components have data ───────────────────────────
        available = {
            "required_skills": len(jd.required_skills) > 0,
            "preferred_skills": len(jd.preferred_skills) > 0,
            "projects":         True,   # Always assessed (0 if none found)
            "cgpa":             candidate.normalized_cgpa is not None,
            "internship":       True,   # Always assessed
            "certifications":   True,   # Always assessed
        }

        weights = _adjust_weights(self.base_weights, available)

        # ── Score each component ───────────────────────────────────────────
        components: list[ComponentScore] = []

        if available["required_skills"]:
            components.append(_score_required_skills(req_match, weights["required_skills"]))
        else:
            components.append(ComponentScore(
                name="Required Skills", raw_score=0.0, weight=0.0,
                weighted_score=0.0, detail="No required skills in JD"
            ))

        if available["preferred_skills"]:
            components.append(_score_preferred_skills(pref_match, weights["preferred_skills"]))
        else:
            components.append(ComponentScore(
                name="Preferred Skills", raw_score=0.0, weight=0.0,
                weighted_score=0.0, detail="No preferred skills in JD"
            ))

        components.append(_score_projects(candidate, weights["projects"]))
        components.append(_score_cgpa(candidate, jd, weights["cgpa"]))
        components.append(_score_internship(candidate, weights["internship"]))
        components.append(_score_certifications(candidate, weights["certifications"]))

        # ── Total score ────────────────────────────────────────────────────
        total = clamp(sum(c.weighted_score for c in components), 0.0, 100.0)
        total = round(total, 2)

        breakdown = ScoreBreakdown(
            total_score=total,
            components=components,
            matched_required=req_match.matched_skills,
            missing_required=req_match.missing_skills,
            matched_preferred=pref_match.matched_skills,
            missing_preferred=pref_match.missing_skills,
            reasons=[],   # filled below
            weights_used=weights,
            required_match_result=req_match,
            preferred_match_result=pref_match,
        )

        # ── Explanation reasons ────────────────────────────────────────────
        breakdown.reasons = _generate_reasons(breakdown, candidate, jd)

        log.debug(
            "Scored [%s]: %.1f/100 | req=%d/%d | pref=%d/%d",
            candidate.filename,
            total,
            len(req_match.matched_skills),
            len(jd.required_skills),
            len(pref_match.matched_skills),
            len(jd.preferred_skills),
        )

        return breakdown
