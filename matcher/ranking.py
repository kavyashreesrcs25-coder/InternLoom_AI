"""
InternLoom AI - Candidate Ranking Engine
Orchestrates the full pipeline for multiple resumes against one JD:
  parse → extract → normalize → score → rank → classify → export

Outputs:
  - Ranked list of CandidateResult objects
  - sample_output.csv
  - ranking.csv
  - parse_quality_report.csv
  - analytics.csv
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import get_logger, parse_logger
from utils.helper import (
    records_to_df,
    df_to_csv_bytes,
    add_rank_column,
    status_badge,
    confidence_badge,
    format_skills_list,
)
from parser.pdf_parser import PDFParser, ParseResult
from parser.extractor import ResumeExtractor, CandidateData
from parser.normalizer import normalize_candidate, JobDescription
from matcher.scorer import CandidateScorer, ScoreBreakdown
from matcher.confidence import ConfidenceCalculator, ConfidenceReport
from config import (
    SHORTLIST_THRESHOLD,
    RESERVE_THRESHOLD,
    OUTPUT_DIR,
    SAMPLE_OUTPUT_CSV,
    RANKING_CSV,
    PARSE_QUALITY_CSV,
    ANALYTICS_CSV,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# DATA CLASS — single candidate result
# ─────────────────────────────────────────────

@dataclass
class CandidateResult:
    """All data for one processed candidate."""
    # Source
    filename: str
    # Candidate info
    candidate: CandidateData
    # Parse metadata
    parse_result: ParseResult
    # Scores
    score_breakdown: ScoreBreakdown
    # Confidence
    confidence_report: ConfidenceReport
    # Derived
    shortlist_status: str = ""          # Shortlisted | Reserve | Rejected
    rank: int = 0
    processing_time_s: float = 0.0

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.candidate.name or Path(self.filename).stem

    @property
    def score(self) -> float:
        return self.score_breakdown.total_score

    @property
    def confidence(self) -> str:
        return self.confidence_report.overall

    @property
    def matched_skills(self) -> list[str]:
        return self.score_breakdown.matched_required + self.score_breakdown.matched_preferred

    @property
    def missing_skills(self) -> list[str]:
        return self.score_breakdown.missing_required

    def to_flat_dict(self) -> dict:
        """Flat dict for CSV export / DataFrame row."""
        c = self.candidate
        sb = self.score_breakdown
        cr = self.confidence_report
        return {
            "rank":                 self.rank,
            "filename":             self.filename,
            "name":                 self.name,
            "email":                c.email,
            "phone":                c.phone,
            "college":              c.college,
            "degree":               c.degree,
            "branch":               c.branch,
            "graduation_year":      c.graduation_year,
            "normalized_cgpa":      c.normalized_cgpa,
            "grade_scale":          c.grade_scale,
            "skills_count":         len(c.skills),
            "all_skills":           "; ".join(c.skills),
            "matched_required":     "; ".join(sb.matched_required),
            "missing_required":     "; ".join(sb.missing_required),
            "matched_preferred":    "; ".join(sb.matched_preferred),
            "projects_count":       len(c.projects),
            "internships_count":    len(c.internships),
            "certifications_count": len(c.certifications),
            "github":               c.github,
            "linkedin":             c.linkedin,
            "score":                self.score,
            "shortlist_status":     self.shortlist_status,
            "confidence":           self.confidence,
            "parse_status":         self.parse_result.parse_status,
            "strategy_used":        self.parse_result.strategy_used,
            "word_count":           self.parse_result.word_count,
            "reason_1":             sb.reasons[0] if len(sb.reasons) > 0 else "",
            "reason_2":             sb.reasons[1] if len(sb.reasons) > 1 else "",
            "reason_3":             sb.reasons[2] if len(sb.reasons) > 2 else "",
            "processing_time_s":    round(self.processing_time_s, 2),
        }


# ─────────────────────────────────────────────
# SHORTLISTING
# ─────────────────────────────────────────────

def classify_candidate(score: float) -> str:
    """Assign shortlist status based on score thresholds."""
    if score >= SHORTLIST_THRESHOLD:
        return "Shortlisted"
    elif score >= RESERVE_THRESHOLD:
        return "Reserve"
    else:
        return "Rejected"


# ─────────────────────────────────────────────
# RANKING ENGINE
# ─────────────────────────────────────────────

class RankingEngine:
    """
    Orchestrates multi-resume processing and ranking.

    Usage:
        engine = RankingEngine()
        results = engine.process(pdf_files, jd)
    """

    def __init__(self, use_semantic: bool = True) -> None:
        self.pdf_parser = PDFParser()
        self.extractor = ResumeExtractor()
        self.scorer = CandidateScorer(use_semantic=use_semantic)
        self.confidence_calc = ConfidenceCalculator()

    def process_single(
        self,
        pdf_bytes: bytes,
        filename: str,
        jd: JobDescription,
    ) -> CandidateResult:
        """
        Full pipeline for one PDF resume.

        Args:
            pdf_bytes: Raw PDF bytes.
            filename:  Display name.
            jd:        Parsed job description.

        Returns:
            CandidateResult with all data populated.
        """
        t_start = time.perf_counter()
        log.info("Processing: %s", filename)

        # 1. Parse PDF
        parse_result = self.pdf_parser.parse(pdf_bytes, filename)

        # 2. Extract structured fields
        candidate = self.extractor.extract(parse_result.full_text, filename)

        # 3. Normalise (grades, skills)
        candidate = normalize_candidate(candidate)

        # 4. Score against JD
        score_breakdown = self.scorer.score(candidate, jd)

        # 5. Confidence assessment
        confidence_report = self.confidence_calc.calculate(parse_result, candidate)

        # 6. Shortlist classification
        shortlist_status = classify_candidate(score_breakdown.total_score)

        elapsed = time.perf_counter() - t_start

        result = CandidateResult(
            filename=filename,
            candidate=candidate,
            parse_result=parse_result,
            score_breakdown=score_breakdown,
            confidence_report=confidence_report,
            shortlist_status=shortlist_status,
            processing_time_s=elapsed,
        )

        log.info(
            "Done [%s]: score=%.1f | status=%s | confidence=%s | time=%.2fs",
            filename, result.score, shortlist_status,
            confidence_report.overall, elapsed,
        )
        return result

    def process(
        self,
        pdf_files: list[tuple[str, bytes]],
        jd: JobDescription,
        progress_callback=None,
    ) -> list[CandidateResult]:
        """
        Process multiple PDF resumes against a single JD.

        Args:
            pdf_files:         List of (filename, bytes) tuples.
            jd:                Parsed JobDescription.
            progress_callback: Optional callable(current, total, filename)
                               called after each resume is processed.

        Returns:
            Ranked list of CandidateResult (highest score first).
        """
        results: list[CandidateResult] = []
        total = len(pdf_files)

        for idx, (filename, pdf_bytes) in enumerate(pdf_files):
            try:
                result = self.process_single(pdf_bytes, filename, jd)
                results.append(result)
            except Exception as exc:
                log.error("Unhandled error processing %s: %s", filename, exc)
                parse_logger.record(filename, "Failed", str(exc), "ERROR")
                # Create a failed placeholder result
                from parser.pdf_parser import ParseResult as PR
                from matcher.scorer import ScoreBreakdown as SB
                from matcher.confidence import ConfidenceReport as CR
                failed_parse = PR(
                    filename=filename, full_text="",
                    parse_status="Failed", strategy_used="none",
                    error=str(exc),
                )
                failed_candidate = CandidateData(filename=filename)
                failed_score = SB(
                    total_score=0.0, components=[], matched_required=[],
                    missing_required=[], matched_preferred=[],
                    missing_preferred=[], reasons=["Processing failed."],
                    weights_used={},
                )
                failed_confidence = CR(
                    overall="Low", parse_status="Failed", word_count=0,
                    fields_present=[], fields_missing=[], completeness_pct=0.0,
                    penalties=[f"Error: {exc}"],
                )
                results.append(CandidateResult(
                    filename=filename,
                    candidate=failed_candidate,
                    parse_result=failed_parse,
                    score_breakdown=failed_score,
                    confidence_report=failed_confidence,
                    shortlist_status="Rejected",
                ))

            if progress_callback:
                try:
                    progress_callback(idx + 1, total, filename)
                except Exception:
                    pass

        # Rank by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        for rank_idx, result in enumerate(results, start=1):
            result.rank = rank_idx

        log.info(
            "Ranking complete: %d candidates | shortlisted=%d | reserve=%d | rejected=%d",
            len(results),
            sum(1 for r in results if r.shortlist_status == "Shortlisted"),
            sum(1 for r in results if r.shortlist_status == "Reserve"),
            sum(1 for r in results if r.shortlist_status == "Rejected"),
        )
        return results

    # ─────────────────────────────────────────────────────────────────────
    # REPORT GENERATION
    # ─────────────────────────────────────────────────────────────────────

    def generate_reports(
        self,
        results: list[CandidateResult],
        jd: JobDescription,
        save_to_disk: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate all four report DataFrames and optionally save to disk.

        Returns:
            Dict with keys: "sample_output", "ranking",
                            "parse_quality", "analytics"
        """
        reports: dict[str, pd.DataFrame] = {}

        # ── sample_output.csv ─────────────────────────────────────────────
        flat_records = [r.to_flat_dict() for r in results]
        sample_df = records_to_df(flat_records)
        reports["sample_output"] = sample_df

        # ── ranking.csv ───────────────────────────────────────────────────
        ranking_cols = [
            "rank", "name", "email", "score", "shortlist_status",
            "confidence", "matched_required", "missing_required",
            "normalized_cgpa", "skills_count", "projects_count",
            "internships_count", "reason_1",
        ]
        ranking_df = sample_df[[c for c in ranking_cols if c in sample_df.columns]].copy()
        reports["ranking"] = ranking_df

        # ── parse_quality_report.csv ──────────────────────────────────────
        parse_records = []
        for r in results:
            cr = r.confidence_report
            parse_records.append({
                "filename":         r.filename,
                "parse_status":     r.parse_result.parse_status,
                "strategy_used":    r.parse_result.strategy_used,
                "word_count":       r.parse_result.word_count,
                "confidence":       cr.overall,
                "completeness_pct": cr.completeness_pct,
                "fields_present":   "; ".join(cr.fields_present),
                "fields_missing":   "; ".join(cr.fields_missing),
                "penalties":        "; ".join(cr.penalties),
                "notes":            cr.notes,
            })
        parse_df = records_to_df(parse_records)
        reports["parse_quality"] = parse_df

        # ── analytics.csv ─────────────────────────────────────────────────
        analytics_records = self._build_analytics(results, jd)
        analytics_df = records_to_df(analytics_records)
        reports["analytics"] = analytics_df

        # ── Save to disk ───────────────────────────────────────────────────
        if save_to_disk:
            self._save_reports(reports)

        return reports

    def _build_analytics(
        self,
        results: list[CandidateResult],
        jd: JobDescription,
    ) -> list[dict]:
        """Build analytics summary records."""
        records = []
        all_skills: dict[str, int] = {}

        for r in results:
            for skill in r.candidate.skills:
                all_skills[skill] = all_skills.get(skill, 0) + 1

        # Top skills frequency
        for skill, count in sorted(all_skills.items(), key=lambda x: -x[1]):
            records.append({
                "metric": "skill_frequency",
                "label": skill,
                "value": count,
            })

        # Score distribution buckets
        buckets = {"90-100": 0, "80-89": 0, "70-79": 0,
                   "60-69": 0, "50-59": 0, "<50": 0}
        for r in results:
            s = r.score
            if s >= 90:     buckets["90-100"] += 1
            elif s >= 80:   buckets["80-89"] += 1
            elif s >= 70:   buckets["70-79"] += 1
            elif s >= 60:   buckets["60-69"] += 1
            elif s >= 50:   buckets["50-59"] += 1
            else:           buckets["<50"] += 1

        for bucket, count in buckets.items():
            records.append({
                "metric": "score_distribution",
                "label": bucket,
                "value": count,
            })

        # Status summary
        for status in ("Shortlisted", "Reserve", "Rejected"):
            records.append({
                "metric": "shortlist_status",
                "label": status,
                "value": sum(1 for r in results if r.shortlist_status == status),
            })

        # Parse quality
        for status in ("Clean", "Partial", "OCR", "Failed"):
            records.append({
                "metric": "parse_quality",
                "label": status,
                "value": sum(
                    1 for r in results
                    if r.parse_result.parse_status == status
                ),
            })

        return records

    def _save_reports(self, reports: dict[str, pd.DataFrame]) -> None:
        """Write report DataFrames to CSV in the outputs directory."""
        name_map = {
            "sample_output": SAMPLE_OUTPUT_CSV,
            "ranking": RANKING_CSV,
            "parse_quality": PARSE_QUALITY_CSV,
            "analytics": ANALYTICS_CSV,
        }
        for key, df in reports.items():
            if df.empty:
                continue
            path = OUTPUT_DIR / name_map.get(key, f"{key}.csv")
            try:
                df.to_csv(path, index=False)
                log.info("Report saved: %s", path)
            except OSError as exc:
                log.error("Could not save report %s: %s", path, exc)

    # ─────────────────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def results_to_display_df(results: list[CandidateResult]) -> pd.DataFrame:
        """
        Build a display-friendly DataFrame for the Streamlit ranking table.
        """
        rows = []
        for r in results:
            rows.append({
                "Rank":        r.rank,
                "Name":        r.name,
                "Score":       f"{r.score:.1f}",
                "Status":      f"{status_badge(r.shortlist_status)} {r.shortlist_status}",
                "Confidence":  f"{confidence_badge(r.confidence)} {r.confidence}",
                "CGPA":        f"{r.candidate.normalized_cgpa:.2f}" if r.candidate.normalized_cgpa else "N/A",
                "Skills":      format_skills_list(r.matched_skills, max_display=5),
                "Missing":     format_skills_list(r.missing_skills, max_display=3),
                "Parse":       r.parse_result.parse_status,
                "Email":       r.candidate.email or "—",
            })
        return pd.DataFrame(rows)

    @staticmethod
    def get_summary_stats(results: list[CandidateResult]) -> dict:
        """Return summary statistics dict for dashboard KPI cards."""
        if not results:
            return {
                "total": 0, "shortlisted": 0, "reserve": 0,
                "rejected": 0, "avg_score": 0.0,
                "top_score": 0.0, "parse_success_rate": 0.0,
            }
        scores = [r.score for r in results]
        failed = sum(1 for r in results if r.parse_result.parse_status == "Failed")
        return {
            "total":             len(results),
            "shortlisted":       sum(1 for r in results if r.shortlist_status == "Shortlisted"),
            "reserve":           sum(1 for r in results if r.shortlist_status == "Reserve"),
            "rejected":          sum(1 for r in results if r.shortlist_status == "Rejected"),
            "avg_score":         round(sum(scores) / len(scores), 1),
            "top_score":         round(max(scores), 1),
            "parse_success_rate": round((len(results) - failed) / len(results) * 100, 1),
        }
