"""
InternLoom AI - Multi-Strategy PDF Parser
Attempts multiple extraction strategies in order of quality:
  1. PyMuPDF  (fast, handles most digital PDFs)
  2. pdfplumber (better for tables and complex layouts)
  3. pdfminer.six (deep text stream analysis)
  4. OCR fallback (for scanned / image-only PDFs)

For two-column and Canva-style resumes, PyMuPDF block-sorting
is applied to reconstruct reading order correctly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.logger import get_logger, parse_logger
from utils.helper import clean_text, count_words
from config import (
    OCR_WORD_THRESHOLD,
    MIN_TEXT_LENGTH,
    MAX_PDF_PAGES,
    PARSE_STATUS_CLEAN,
    PARSE_STATUS_PARTIAL,
    PARSE_STATUS_OCR,
    PARSE_STATUS_FAILED,
)

log = get_logger(__name__)


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class PageText:
    """Text content extracted from a single PDF page."""
    page_number: int
    raw_text: str
    word_count: int
    strategy: str          # "pymupdf" | "pdfplumber" | "pdfminer" | "ocr"


@dataclass
class ParseResult:
    """Full parse result for one PDF file."""
    filename: str
    full_text: str
    pages: list[PageText] = field(default_factory=list)
    word_count: int = 0
    parse_status: str = PARSE_STATUS_FAILED
    strategy_used: str = "none"
    error: Optional[str] = None
    page_count: int = 0


# ─────────────────────────────────────────────
# STRATEGY 1 — PyMuPDF
# ─────────────────────────────────────────────

def _extract_pymupdf(pdf_bytes: bytes, filename: str) -> Optional[ParseResult]:
    """
    Extract text using PyMuPDF (fitz).

    Uses block-level extraction and sorts blocks by (y0, x0) so that
    two-column layouts are read left-to-right, top-to-bottom.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[PageText] = []
        all_text_parts: list[str] = []

        for page_num in range(min(len(doc), MAX_PDF_PAGES)):
            page = doc[page_num]

            # Get text blocks with position: (x0, y0, x1, y1, text, ...)
            blocks = page.get_text("blocks")

            # Sort blocks: primarily by vertical band (rounded), then horizontal
            # This handles two-column layouts correctly.
            blocks_sorted = sorted(
                blocks,
                key=lambda b: (round(b[1] / 20) * 20, b[0])  # (band_y, x0)
            )

            page_text = "\n".join(
                b[4].strip() for b in blocks_sorted
                if b[4].strip() and len(b) > 4
            )
            page_text = clean_text(page_text)
            wc = count_words(page_text)

            pages.append(PageText(
                page_number=page_num + 1,
                raw_text=page_text,
                word_count=wc,
                strategy="pymupdf",
            ))
            if page_text:
                all_text_parts.append(page_text)

        doc.close()

        full_text = "\n\n".join(all_text_parts)
        total_words = count_words(full_text)

        if total_words < MIN_TEXT_LENGTH:
            return None  # Not enough text — try next strategy

        return ParseResult(
            filename=filename,
            full_text=full_text,
            pages=pages,
            word_count=total_words,
            parse_status=PARSE_STATUS_CLEAN if total_words >= OCR_WORD_THRESHOLD else PARSE_STATUS_PARTIAL,
            strategy_used="pymupdf",
            page_count=len(pages),
        )

    except ImportError:
        log.warning("PyMuPDF not installed — skipping strategy.")
        return None
    except Exception as exc:
        log.debug("PyMuPDF failed for %s: %s", filename, exc)
        return None


# ─────────────────────────────────────────────
# STRATEGY 2 — pdfplumber
# ─────────────────────────────────────────────

def _extract_pdfplumber(pdf_bytes: bytes, filename: str) -> Optional[ParseResult]:
    """
    Extract text using pdfplumber.

    pdfplumber excels at tables and structured layouts.
    It extracts table cells separately and appends them to body text.
    """
    try:
        import pdfplumber

        pages: list[PageText] = []
        all_text_parts: list[str] = []

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages[:MAX_PDF_PAGES]):
                parts: list[str] = []

                # Extract regular text
                body = page.extract_text(x_tolerance=3, y_tolerance=3)
                if body:
                    parts.append(body.strip())

                # Extract table content
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = " | ".join(
                            cell.strip() for cell in row if cell and cell.strip()
                        )
                        if row_text:
                            parts.append(row_text)

                page_text = clean_text("\n".join(parts))
                wc = count_words(page_text)

                pages.append(PageText(
                    page_number=page_num + 1,
                    raw_text=page_text,
                    word_count=wc,
                    strategy="pdfplumber",
                ))
                if page_text:
                    all_text_parts.append(page_text)

        full_text = "\n\n".join(all_text_parts)
        total_words = count_words(full_text)

        if total_words < MIN_TEXT_LENGTH:
            return None

        return ParseResult(
            filename=filename,
            full_text=full_text,
            pages=pages,
            word_count=total_words,
            parse_status=PARSE_STATUS_CLEAN if total_words >= OCR_WORD_THRESHOLD else PARSE_STATUS_PARTIAL,
            strategy_used="pdfplumber",
            page_count=len(pages),
        )

    except ImportError:
        log.warning("pdfplumber not installed — skipping strategy.")
        return None
    except Exception as exc:
        log.debug("pdfplumber failed for %s: %s", filename, exc)
        return None


# ─────────────────────────────────────────────
# STRATEGY 3 — pdfminer.six
# ─────────────────────────────────────────────

def _extract_pdfminer(pdf_bytes: bytes, filename: str) -> Optional[ParseResult]:
    """
    Extract text using pdfminer.six.

    pdfminer performs deep character-stream analysis and handles
    unusual font encodings / embedded text layers that other parsers miss.
    """
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output_buf = io.StringIO()
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
        )
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            output_buf,
            laparams=laparams,
            output_type="text",
            codec="utf-8",
        )

        raw = output_buf.getvalue()
        full_text = clean_text(raw)
        total_words = count_words(full_text)

        if total_words < MIN_TEXT_LENGTH:
            return None

        pages = [PageText(
            page_number=1,
            raw_text=full_text,
            word_count=total_words,
            strategy="pdfminer",
        )]

        return ParseResult(
            filename=filename,
            full_text=full_text,
            pages=pages,
            word_count=total_words,
            parse_status=PARSE_STATUS_CLEAN if total_words >= OCR_WORD_THRESHOLD else PARSE_STATUS_PARTIAL,
            strategy_used="pdfminer",
            page_count=1,
        )

    except ImportError:
        log.warning("pdfminer.six not installed — skipping strategy.")
        return None
    except Exception as exc:
        log.debug("pdfminer failed for %s: %s", filename, exc)
        return None


# ─────────────────────────────────────────────
# MAIN PARSER — Orchestrates all strategies
# ─────────────────────────────────────────────

class PDFParser:
    """
    Orchestrates multi-strategy PDF text extraction.

    Tries strategies in quality order and falls back to OCR
    when extracted text is below the word threshold.
    """

    def __init__(self) -> None:
        # Import OCR here to avoid circular imports (ocr.py imports this)
        self._ocr_extractor = None

    def _get_ocr(self):
        """Lazy-load OCR extractor to avoid startup cost."""
        if self._ocr_extractor is None:
            try:
                from parser.ocr import OCRExtractor
                self._ocr_extractor = OCRExtractor()
            except Exception as exc:
                log.warning("OCR extractor unavailable: %s", exc)
        return self._ocr_extractor

    def parse(self, pdf_bytes: bytes, filename: str = "resume.pdf",
              skip_ocr: bool = False) -> ParseResult:
        """
        Parse a PDF and return a ParseResult.

        Strategy order:
          1. PyMuPDF  (fastest)
          2. pdfplumber (tables / structured)
          3. pdfminer.six (deep encoding)
          4. EasyOCR  (scanned / image PDFs) — skipped if skip_ocr=True

        OCR is triggered if word_count < OCR_WORD_THRESHOLD after
        digital extraction, or if all digital strategies fail.

        Args:
            pdf_bytes: Raw bytes of the PDF file.
            filename:  Display name for logging / reporting.
            skip_ocr:  If True, never attempt OCR (use for JD PDFs).

        Returns:
            ParseResult with extracted text and metadata.
        """
        log.info("Parsing: %s (%d bytes)", filename, len(pdf_bytes))

        # ── Try digital extraction strategies ─────────────────────────────
        result: Optional[ParseResult] = None

        for strategy_fn in (
            _extract_pymupdf,
            _extract_pdfplumber,
            _extract_pdfminer,
        ):
            candidate = strategy_fn(pdf_bytes, filename)
            if candidate and candidate.word_count > (result.word_count if result else 0):
                result = candidate
                if result.word_count >= OCR_WORD_THRESHOLD * 2:
                    break

        # ── OCR fallback (skipped for JDs) ────────────────────────────────
        if not skip_ocr and (result is None or result.word_count < OCR_WORD_THRESHOLD):
            log.info(
                "%s: word count %d < threshold %d — attempting OCR.",
                filename,
                result.word_count if result else 0,
                OCR_WORD_THRESHOLD,
            )
            ocr_engine = self._get_ocr()
            if ocr_engine:
                try:
                    ocr_result = ocr_engine.extract_from_pdf_bytes(pdf_bytes, filename)
                    if ocr_result and ocr_result.word_count > (result.word_count if result else 0):
                        result = ocr_result
                except Exception as exc:
                    log.warning("OCR failed for %s: %s", filename, exc)
                    parse_logger.record(filename, PARSE_STATUS_OCR, f"OCR error: {exc}", "WARNING")

        # ── Final result assembly ──────────────────────────────────────────
        if result is None or result.word_count < 5:
            log.error("All strategies failed for %s", filename)
            parse_logger.record(filename, PARSE_STATUS_FAILED, "No text extracted by any strategy.", "ERROR")
            return ParseResult(
                filename=filename,
                full_text="",
                parse_status=PARSE_STATUS_FAILED,
                strategy_used="none",
                error="All extraction strategies failed.",
            )

        log.info(
            "%s parsed: %d words via %s [%s]",
            filename, result.word_count, result.strategy_used, result.parse_status
        )
        parse_logger.record(
            filename,
            result.parse_status,
            f"Extracted {result.word_count} words via {result.strategy_used}",
            "INFO",
        )
        return result

    def parse_file(self, path: Path) -> ParseResult:
        """Parse from a file path instead of bytes."""
        try:
            pdf_bytes = path.read_bytes()
            return self.parse(pdf_bytes, path.name)
        except OSError as exc:
            log.error("Cannot read file %s: %s", path, exc)
            return ParseResult(
                filename=path.name,
                full_text="",
                parse_status=PARSE_STATUS_FAILED,
                strategy_used="none",
                error=str(exc),
            )
