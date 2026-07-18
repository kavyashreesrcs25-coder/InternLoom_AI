"""
InternLoom AI - Multi-JD Matching Engine
Scores every resume against every uploaded JD, selects the best match
per candidate, builds job-wise rankings, and generates all CSV exports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import pandas as pd

from utils.logger import get_logger, parse_logger
from utils.helper import records_to_df, df_to_csv_bytes, clamp, status_badge, confidence_badge
from parser.pdf_parser import PDFParser
from parser.extractor import ResumeExtractor, CandidateData
from parser.normalizer import normalize_candidate, JobDescription
from matcher.scorer import CandidateScorer, ScoreBreakdown
from matcher.confidence import ConfidenceCalculator, ConfidenceReport
from matcher.ranking import CandidateResult, classify_candidate
from config import (
    OUTPUT_DIR, SHORTLIST_THRESHOLD, RESERVE_THRESHOLD,
    KNOWN_SKILLS, SCORING_WEIGHTS,
)

log = get_logger(__name__)

# ── Bonus role-recommendation pool (drawn from KNOWN_SKILLS taxonomy) ──────
ROLE_SKILL_MAP: dict[str, list[str]] = {
    "Frontend Developer":    ["React", "JavaScript", "TypeScript", "HTML", "CSS", "Vue.js", "Next.js"],
    "Backend Developer":     ["Python", "Node.js", "FastAPI", "Django", "Flask", "PostgreSQL", "REST API"],
    "Full Stack Developer":  ["React", "Node.js", "MongoDB", "JavaScript", "REST API", "Docker"],
    "Data Scientist":        ["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "TensorFlow"],
    "ML Engineer":           ["TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "Python", "Docker"],
    "AI Engineer":           ["LLM", "LangChain", "RAG", "Prompt Engineering", "Python", "Vector Database"],
    "DevOps Engineer":       ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Linux", "Jenkins"],
    "Cloud Engineer":        ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform"],
    "Mobile Developer":      ["Flutter", "React Native", "Android", "iOS", "Dart", "Firebase"],
    "Data Engineer":         ["Python", "Spark", "Hadoop", "Airflow", "PostgreSQL", "Kafka"],
    "Cybersecurity Analyst": ["Cybersecurity", "Penetration Testing", "OWASP", "Linux", "Python"],
    "Blockchain Developer":  ["Solidity", "Blockchain", "Web3", "Ethereum", "Smart Contracts"],
    "NLP Engineer":          ["NLP", "Python", "BERT", "Transformers", "Hugging Face", "LLM"],
}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class JDScore:
    """Score of one candidate against one JD."""
    jd_role:           str
    jd_source:         str
    score:             float
    shortlist_status:  str
    matched_required:  list[str]
    missing_required:  list[str]
    matched_preferred: list[str]
    reasons:           list[str]
    breakdown:         Optional[ScoreBreakdown] = None


@dataclass
class RoleRecommendation:
    """Bonus AI recommendation for low-scoring candidates."""
    role:       str
    match_pct:  float
    reason:     str


@dataclass
class MultiJDCandidateResult:
    """Full multi-JD result for one candidate."""
    filename:          str
    candidate:         CandidateData
    parse_result:      object          # ParseResult
    confidence_report: ConfidenceReport
    jd_scores:         list[JDScore]   = field(default_factory=list)
    best_jd_role:      str             = ""
    best_jd_source:    str             = ""
    best_score:        float           = 0.0
    best_shortlist:    str             = "Rejected"
    recommendations:   list[RoleRecommendation] = field(default_factory=list)
    processing_time_s: float           = 0.0

    @property
    def name(self) -> str:
        return self.candidate.name or Path(self.filename).stem

    @property
    def confidence(self) -> str:
        return self.confidence_report.overall

    def score_for_jd(self, jd_role: str) -> float:
        for s in self.jd_scores:
            if s.jd_role == jd_role:
                return s.score
        return 0.0

    def to_flat_dict(self) -> dict:
        c  = self.candidate
        cr = self.confidence_report
        return {
            "filename":            self.filename,
            "name":                self.name,
            "email":               c.email,
            "phone":               c.phone,
            "college":             c.college,
            "degree":              c.degree,
            "normalized_cgpa":     c.normalized_cgpa,
            "skills_count":        len(c.skills),
            "all_skills":          "; ".join(c.skills),
            "best_jd":             self.best_jd_role,
            "best_score":          self.best_score,
            "shortlist_status":    self.best_shortlist,
            "confidence":          self.confidence,
            "parse_status":        self.parse_result.parse_status,
            "matched_required":    "; ".join(
                next((s.matched_required for s in self.jd_scores
                      if s.jd_role == self.best_jd_role), [])
            ),
            "missing_required":    "; ".join(
                next((s.missing_required for s in self.jd_scores
                      if s.jd_role == self.best_jd_role), [])
            ),
            "reason_1": self._best_reasons(0),
            "reason_2": self._best_reasons(1),
            "reason_3": self._best_reasons(2),
        }

    def _best_reasons(self, idx: int) -> str:
        for s in self.jd_scores:
            if s.jd_role == self.best_jd_role:
                return s.reasons[idx] if idx < len(s.reasons) else ""
        return ""


# ─────────────────────────────────────────────
# ROLE RECOMMENDATION (Bonus AI Feature)
# ─────────────────────────────────────────────

def _recommend_roles(candidate: CandidateData, top_n: int = 3) -> list[RoleRecommendation]:
    """
    For candidates scoring below threshold on ALL JDs, recommend the
    top_n most suitable roles based on skill overlap with ROLE_SKILL_MAP.
    """
    cand_skills_lower = {s.lower() for s in candidate.skills}
    scores: list[tuple[str, float, str]] = []

    for role, role_skills in ROLE_SKILL_MAP.items():
        role_lower = [s.lower() for s in role_skills]
        matched = [s for s in role_lower if s in cand_skills_lower]
        pct = round((len(matched) / len(role_skills)) * 100, 1) if role_skills else 0.0
        if pct > 0:
            matched_names = [s for s in role_skills if s.lower() in cand_skills_lower]
            reason = (
                f"Matched {len(matched)}/{len(role_skills)} key skills: "
                f"{', '.join(matched_names[:4])}."
            )
            scores.append((role, pct, reason))

    scores.sort(key=lambda x: -x[1])
    return [
        RoleRecommendation(role=r, match_pct=p, reason=rsn)
        for r, p, rsn in scores[:top_n]
    ]


# ─────────────────────────────────────────────
# MULTI-JD RANKING ENGINE
# ─────────────────────────────────────────────

class MultiJDRankingEngine:
    """
    Scores every resume against every JD, picks each candidate's
    best-matching role, builds job-wise ranked lists, and exports CSVs.

    Usage:
        engine  = MultiJDRankingEngine()
        results = engine.process(pdf_files, jd_list, allow_multi_match=False)
        reports = engine.generate_reports(results, jd_list)
    """

    def __init__(self, use_semantic: bool = True) -> None:
        self.pdf_parser     = PDFParser()
        self.extractor      = ResumeExtractor()
        self.scorer         = CandidateScorer(use_semantic=use_semantic)
        self.conf_calc      = ConfidenceCalculator()

    # ── Parse one resume ──────────────────────────────────────────────────
    def _parse_resume(
        self, pdf_bytes: bytes, filename: str
    ) -> tuple[object, CandidateData]:
        parse_result = self.pdf_parser.parse(pdf_bytes, filename)
        candidate    = self.extractor.extract(parse_result.full_text, filename)
        candidate    = normalize_candidate(candidate)
        return parse_result, candidate

    # ── Score one candidate against all JDs ───────────────────────────────
    def _score_against_all(
        self, candidate: CandidateData, jds: list[JobDescription]
    ) -> list[JDScore]:
        scores: list[JDScore] = []
        for jd in jds:
            try:
                sb     = self.scorer.score(candidate, jd)
                status = classify_candidate(sb.total_score)
                scores.append(JDScore(
                    jd_role           = jd.display_name(),
                    jd_source         = jd.source_name,
                    score             = sb.total_score,
                    shortlist_status  = status,
                    matched_required  = sb.matched_required,
                    missing_required  = sb.missing_required,
                    matched_preferred = sb.matched_preferred,
                    reasons           = sb.reasons,
                    breakdown         = sb,
                ))
            except Exception as exc:
                log.error("Scoring error [%s vs %s]: %s", candidate.filename, jd.role, exc)
        return scores

    # ── Process all resumes ───────────────────────────────────────────────
    def process(
        self,
        pdf_files: list[tuple[str, bytes]],
        jds: list[JobDescription],
        allow_multi_match: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[MultiJDCandidateResult]:
        """
        Full pipeline: parse every resume → score vs every JD → pick best.

        Args:
            pdf_files:         List of (filename, bytes).
            jds:               List of parsed JobDescription objects.
            allow_multi_match: If False, each candidate is assigned to
                               only one JD (highest scoring). If True,
                               candidates appear in every JD ranking
                               they qualify for.
            progress_callback: callable(current, total, filename).

        Returns:
            List of MultiJDCandidateResult sorted by best_score desc.
        """
        results: list[MultiJDCandidateResult] = []
        total = len(pdf_files)

        for idx, (filename, pdf_bytes) in enumerate(pdf_files):
            t0 = time.perf_counter()
            try:
                parse_result, candidate = self._parse_resume(pdf_bytes, filename)
                conf_report = self.conf_calc.calculate(parse_result, candidate)
                jd_scores   = self._score_against_all(candidate, jds)

                # Best match
                best = max(jd_scores, key=lambda s: s.score) if jd_scores else None
                best_role   = best.jd_role if best else "No JD"
                best_source = best.jd_source if best else ""
                best_score  = best.score if best else 0.0
                best_status = best.shortlist_status if best else "Rejected"

                # Bonus recommendations if below threshold on ALL JDs
                recs: list[RoleRecommendation] = []
                if all(s.score < SHORTLIST_THRESHOLD for s in jd_scores):
                    recs = _recommend_roles(candidate)

                results.append(MultiJDCandidateResult(
                    filename          = filename,
                    candidate         = candidate,
                    parse_result      = parse_result,
                    confidence_report = conf_report,
                    jd_scores         = jd_scores,
                    best_jd_role      = best_role,
                    best_jd_source    = best_source,
                    best_score        = best_score,
                    best_shortlist    = best_status,
                    recommendations   = recs,
                    processing_time_s = time.perf_counter() - t0,
                ))
                log.info(
                    "[%s] best=%s %.1f%%  conf=%s",
                    filename, best_role, best_score, conf_report.overall,
                )
            except Exception as exc:
                log.error("Pipeline error [%s]: %s", filename, exc)
                from parser.pdf_parser import ParseResult as PR
                from matcher.confidence import ConfidenceReport as CR
                failed_parse = PR(
                    filename=filename, full_text="",
                    parse_status="Failed", strategy_used="none", error=str(exc),
                )
                results.append(MultiJDCandidateResult(
                    filename          = filename,
                    candidate         = CandidateData(filename=filename),
                    parse_result      = failed_parse,
                    confidence_report = CR(
                        overall="Low", parse_status="Failed", word_count=0,
                        fields_present=[], fields_missing=[], completeness_pct=0.0,
                        penalties=[str(exc)],
                    ),
                    best_shortlist = "Rejected",
                ))

            if progress_callback:
                try: progress_callback(idx + 1, total, filename)
                except Exception: pass

        results.sort(key=lambda r: r.best_score, reverse=True)
        log.info(
            "Multi-JD complete: %d candidates, %d JDs",
            len(results), len(jds),
        )
        return results

    # ── Job-wise rankings ─────────────────────────────────────────────────
    def build_jobwise_rankings(
        self,
        results: list[MultiJDCandidateResult],
        jds: list[JobDescription],
        allow_multi_match: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Build a ranked list for every JD.

        Returns:
            { jd_display_name: [row_dict, ...] } sorted by score desc.
        """
        # Track which candidates are already assigned (single-match mode)
        assigned: set[str] = set()
        rankings: dict[str, list[dict]] = {}

        for jd in jds:
            role_name = jd.display_name()
            rows: list[dict] = []

            # Collect scores for this JD, sorted desc
            scored = sorted(
                results,
                key=lambda r: r.score_for_jd(role_name),
                reverse=True,
            )
            rank = 1
            for r in scored:
                score = r.score_for_jd(role_name)
                if score == 0.0:
                    continue
                if not allow_multi_match:
                    if r.filename in assigned and role_name != r.best_jd_role:
                        continue
                rows.append({
                    "rank":             rank,
                    "name":             r.name,
                    "email":            r.candidate.email or "—",
                    "score":            round(score, 1),
                    "shortlist_status": classify_candidate(score),
                    "confidence":       r.confidence,
                    "cgpa":             r.candidate.normalized_cgpa,
                    "matched_required": "; ".join(
                        next((s.matched_required for s in r.jd_scores
                              if s.jd_role == role_name), [])
                    ),
                    "missing_required": "; ".join(
                        next((s.missing_required for s in r.jd_scores
                              if s.jd_role == role_name), [])
                    ),
                    "filename": r.filename,
                })
                rank += 1

            if not allow_multi_match:
                for row in rows:
                    assigned.add(row["filename"])
            rankings[role_name] = rows

        return rankings

    # ── Stats ─────────────────────────────────────────────────────────────
    @staticmethod
    def get_summary_stats(
        results: list[MultiJDCandidateResult],
        jds: list[JobDescription],
    ) -> dict:
        if not results:
            return {
                "total": 0, "shortlisted": 0, "reserve": 0,
                "rejected": 0, "avg_score": 0.0, "top_score": 0.0,
                "parse_success_rate": 0.0, "jd_count": len(jds),
            }
        scores  = [r.best_score for r in results]
        failed  = sum(1 for r in results if r.parse_result.parse_status == "Failed")
        return {
            "total":             len(results),
            "shortlisted":       sum(1 for r in results if r.best_shortlist == "Shortlisted"),
            "reserve":           sum(1 for r in results if r.best_shortlist == "Reserve"),
            "rejected":          sum(1 for r in results if r.best_shortlist == "Rejected"),
            "avg_score":         round(sum(scores) / len(scores), 1),
            "top_score":         round(max(scores), 1),
            "parse_success_rate": round((len(results) - failed) / len(results) * 100, 1),
            "jd_count":          len(jds),
        }

    # ── Report generation ─────────────────────────────────────────────────
    def generate_reports(
        self,
        results: list[MultiJDCandidateResult],
        jds: list[JobDescription],
        allow_multi_match: bool = False,
        save_to_disk: bool = True,
    ) -> dict[str, pd.DataFrame]:
        reports: dict[str, pd.DataFrame] = {}

        # candidate_best_match.csv
        best_rows = [r.to_flat_dict() for r in results]
        reports["candidate_best_match"] = records_to_df(best_rows)

        # job_wise_ranking.csv
        rankings = self.build_jobwise_rankings(results, jds, allow_multi_match)
        jw_rows: list[dict] = []
        for role, rows in rankings.items():
            for row in rows:
                jw_rows.append({"job_role": role, **row})
        reports["job_wise_ranking"] = records_to_df(jw_rows)

        # shortlisted.csv / reserve.csv / rejected.csv
        for status in ("Shortlisted", "Reserve", "Rejected"):
            key = status.lower()
            subset = [r.to_flat_dict() for r in results if r.best_shortlist == status]
            reports[key] = records_to_df(subset)

        # parse_quality_report.csv
        parse_rows = []
        for r in results:
            cr = r.confidence_report
            parse_rows.append({
                "filename":        r.filename,
                "parse_status":    r.parse_result.parse_status,
                "strategy_used":   r.parse_result.strategy_used,
                "word_count":      r.parse_result.word_count,
                "confidence":      cr.overall,
                "completeness":    cr.completeness_pct,
                "fields_missing":  "; ".join(cr.fields_missing),
                "penalties":       "; ".join(cr.penalties),
            })
        reports["parse_quality"] = records_to_df(parse_rows)

        # analytics.csv
        analytics_rows: list[dict] = []
        skill_freq: dict[str, int] = {}
        for r in results:
            for sk in r.candidate.skills:
                skill_freq[sk] = skill_freq.get(sk, 0) + 1
        for sk, cnt in sorted(skill_freq.items(), key=lambda x: -x[1]):
            analytics_rows.append({"metric": "skill_frequency", "label": sk, "value": cnt})
        for role, rows in rankings.items():
            analytics_rows.append({"metric": "candidates_per_job", "label": role, "value": len(rows)})
            avg = round(sum(row["score"] for row in rows) / len(rows), 1) if rows else 0.0
            analytics_rows.append({"metric": "avg_score_per_job", "label": role, "value": avg})
        for status in ("Shortlisted", "Reserve", "Rejected"):
            analytics_rows.append({
                "metric": "shortlist_status", "label": status,
                "value": sum(1 for r in results if r.best_shortlist == status),
            })
        reports["analytics"] = records_to_df(analytics_rows)

        if save_to_disk:
            file_map = {
                "candidate_best_match": "candidate_best_match.csv",
                "job_wise_ranking":     "job_wise_ranking.csv",
                "shortlisted":          "shortlisted.csv",
                "reserve":              "reserve.csv",
                "rejected":             "rejected.csv",
                "parse_quality":        "parse_quality_report.csv",
                "analytics":            "analytics.csv",
            }
            for key, fname in file_map.items():
                df = reports.get(key)
                if df is not None and not df.empty:
                    try:
                        df.to_csv(OUTPUT_DIR / fname, index=False)
                        log.info("Saved: %s", fname)
                    except OSError as exc:
                        log.error("Could not save %s: %s", fname, exc)

        return reports
