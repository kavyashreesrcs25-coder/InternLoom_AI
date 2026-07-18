"""
InternLoom AI - OCR Extractor
Uses EasyOCR + OpenCV image preprocessing to extract text from
scanned PDFs and image-only resumes.

Pipeline per page:
  1. Render PDF page to image (PyMuPDF high-DPI raster)
  2. Preprocess with OpenCV: grayscale → denoise → threshold → deskew
  3. Run EasyOCR on the cleaned image
  4. Assemble text preserving approximate reading order
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from parser.pdf_parser import ParseResult, PageText
from utils.logger import get_logger
from utils.helper import clean_text, count_words
from config import (
    MAX_PDF_PAGES,
    PARSE_STATUS_OCR,
    PARSE_STATUS_FAILED,
    MIN_TEXT_LENGTH,
)

log = get_logger(__name__)

# Suppress EasyOCR verbose output
os.environ.setdefault("EASYOCR_LOG_LEVEL", "ERROR")


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def _preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """
    Prepare a raw RGB image array for OCR:
      1. Convert to grayscale
      2. Gaussian denoise
      3. Adaptive threshold (CLAHE contrast enhancement)
      4. Slight dilation to bold thin characters
      5. Deskew (rotate to correct tilt)

    Args:
        img_array: HxWx3 uint8 numpy array (RGB).

    Returns:
        HxW uint8 numpy array (processed grayscale).
    """
    try:
        import cv2

        # Grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Denoise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # Deskew
        skewed = _deskew(binary)

        return skewed

    except ImportError:
        log.warning("OpenCV not available — using raw image for OCR.")
        try:
            import cv2
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        except Exception:
            return img_array
    except Exception as exc:
        log.debug("Image preprocessing error: %s", exc)
        return img_array


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in a binary image using minimum-area rectangle on
    the largest contour blob of foreground pixels.

    Returns the original image unchanged if rotation is < 0.5°.
    """
    try:
        import cv2

        # Invert for contour detection (text = white on black)
        inv = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inv > 0))
        if coords.shape[0] < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        # Normalise angle to [-45, 45]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.5:
            return image  # Skip trivial rotation

        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    except Exception:
        return image


# ─────────────────────────────────────────────
# PDF → IMAGES
# ─────────────────────────────────────────────

def _pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[np.ndarray]:
    """
    Render each page of a PDF to a numpy RGB image array.

    Args:
        pdf_bytes: Raw PDF bytes.
        dpi: Render resolution (higher = better OCR, slower).

    Returns:
        List of numpy arrays, one per page.
    """
    images: list[np.ndarray] = []
    try:
        import fitz  # PyMuPDF
        from PIL import Image

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(min(len(doc), MAX_PDF_PAGES)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.array(img))

        doc.close()

    except ImportError:
        log.warning("PyMuPDF or Pillow not available for PDF→image conversion.")
    except Exception as exc:
        log.error("PDF→image conversion failed: %s", exc)

    return images


# ─────────────────────────────────────────────
# OCR TEXT ASSEMBLY
# ─────────────────────────────────────────────

def _sort_ocr_results(results: list) -> list:
    """
    Sort EasyOCR results by reading order (top-to-bottom, left-to-right).
    Each result is (bbox, text, confidence).
    bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    """
    def bbox_top_y(result):
        bbox = result[0]
        y_coords = [pt[1] for pt in bbox]
        return min(y_coords)

    def bbox_left_x(result):
        bbox = result[0]
        x_coords = [pt[0] for pt in bbox]
        return min(x_coords)

    # Group into horizontal bands (±20px), then sort left-to-right within band
    if not results:
        return results

    results_sorted = sorted(results, key=lambda r: (
        round(bbox_top_y(r) / 20) * 20,
        bbox_left_x(r)
    ))
    return results_sorted


def _results_to_text(results: list, min_confidence: float = 0.3) -> str:
    """
    Convert sorted EasyOCR results to a single text string.

    Args:
        results: List of (bbox, text, confidence) from EasyOCR.
        min_confidence: Filter out low-confidence tokens.

    Returns:
        Assembled text string.
    """
    lines: list[str] = []
    current_line_y: Optional[float] = None
    current_line_tokens: list[str] = []
    LINE_GAP = 15  # px gap between lines

    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        text = text.strip()
        if not text:
            continue

        y = min(pt[1] for pt in bbox)

        if current_line_y is None or abs(y - current_line_y) > LINE_GAP:
            if current_line_tokens:
                lines.append(" ".join(current_line_tokens))
            current_line_tokens = [text]
            current_line_y = y
        else:
            current_line_tokens.append(text)

    if current_line_tokens:
        lines.append(" ".join(current_line_tokens))

    return "\n".join(lines)


# ─────────────────────────────────────────────
# OCR EXTRACTOR CLASS
# ─────────────────────────────────────────────

class OCRExtractor:
    """
    EasyOCR-backed text extractor for scanned / image PDFs.

    The EasyOCR reader is initialised once and reused for performance.
    Supports English (and optionally other languages via config).
    """

    def __init__(self, languages: list[str] = None, gpu: bool = False) -> None:
        self._reader = None
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._init_reader()

    def _init_reader(self) -> None:
        """Initialise EasyOCR reader (downloads models on first run)."""
        try:
            import easyocr
            # Clean up any leftover temp.zip from a previous interrupted download
            import pathlib
            temp_zip = pathlib.Path.home() / ".EasyOCR" / "model" / "temp.zip"
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                    log.info("Removed stale EasyOCR temp.zip before init.")
                except OSError:
                    pass  # locked by another process — ignore
            log.info("Initialising EasyOCR reader (languages=%s, gpu=%s) …", self._languages, self._gpu)
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            log.info("EasyOCR reader ready.")
        except ImportError:
            log.warning("EasyOCR not installed — OCR extraction unavailable.")
        except Exception as exc:
            log.error("EasyOCR initialisation failed: %s", exc)
            self._reader = None   # Ensure reader is None so callers skip OCR gracefully

    def _ocr_image(self, image: np.ndarray) -> str:
        """Run OCR on a single preprocessed image array."""
        if self._reader is None:
            return ""
        try:
            results = self._reader.readtext(image, detail=1, paragraph=False)
            results_sorted = _sort_ocr_results(results)
            return _results_to_text(results_sorted)
        except Exception as exc:
            log.warning("OCR read error: %s", exc)
            return ""

    def extract_from_pdf_bytes(
        self, pdf_bytes: bytes, filename: str = "resume.pdf"
    ) -> Optional[ParseResult]:
        """
        Extract text from a PDF using OCR.

        Args:
            pdf_bytes: Raw PDF bytes.
            filename:  Display name.

        Returns:
            ParseResult with OCR-extracted text, or None on failure.
        """
        if self._reader is None:
            log.error("OCR reader not available.")
            return None

        log.info("OCR extraction started for: %s", filename)

        images = _pdf_bytes_to_images(pdf_bytes)
        if not images:
            log.error("Could not render PDF pages to images for OCR: %s", filename)
            return None

        pages: list[PageText] = []
        all_text_parts: list[str] = []

        for idx, raw_image in enumerate(images):
            try:
                processed = _preprocess_image(raw_image)
                page_text = self._ocr_image(processed)
                page_text = clean_text(page_text)
                wc = count_words(page_text)

                pages.append(PageText(
                    page_number=idx + 1,
                    raw_text=page_text,
                    word_count=wc,
                    strategy="ocr",
                ))
                if page_text:
                    all_text_parts.append(page_text)
                log.debug("OCR page %d: %d words", idx + 1, wc)

            except Exception as exc:
                log.warning("OCR failed on page %d of %s: %s", idx + 1, filename, exc)

        full_text = "\n\n".join(all_text_parts)
        total_words = count_words(full_text)

        if total_words < MIN_TEXT_LENGTH:
            log.warning("OCR produced insufficient text for %s: %d words", filename, total_words)
            return None

        return ParseResult(
            filename=filename,
            full_text=full_text,
            pages=pages,
            word_count=total_words,
            parse_status=PARSE_STATUS_OCR,
            strategy_used="ocr",
            page_count=len(pages),
        )

    def extract_from_image_bytes(self, image_bytes: bytes) -> str:
        """
        Run OCR directly on raw image bytes (PNG/JPEG).
        Convenience method for non-PDF image inputs.

        Returns:
            Extracted text string.
        """
        if self._reader is None:
            return ""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)
            processed = _preprocess_image(arr)
            return clean_text(self._ocr_image(processed))
        except Exception as exc:
            log.error("Image OCR failed: %s", exc)
            return ""
