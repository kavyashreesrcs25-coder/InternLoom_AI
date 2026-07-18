"""
InternLoom AI - Google Drive Import Utility (Fixed v2)
Reliable public-folder import using Drive v3 API + gdown download backend.

Download strategy (by file_id only — never by filename):
  1. List files via Drive v3 JSON API (public endpoint).
  2. Download each PDF via gdown (handles confirm tokens automatically).
  3. Fallback to requests-based download on gdown failure.
  4. Verify every file: MIME = application/pdf AND magic bytes = %PDF-.
  5. Retry failed downloads once.
  6. Save to data/temp_resumes/.
"""
from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger(__name__)

_BASE_DIR = Path(__file__).parent.parent
TEMP_RESUME_DIR = _BASE_DIR / "data" / "temp_resumes"
TEMP_RESUME_DIR.mkdir(parents=True, exist_ok=True)

_DRIVE_API_FILES = "https://www.googleapis.com/drive/v3/files"
_DRIVE_DOWNLOAD  = "https://drive.google.com/uc"
_CHUNK_SIZE      = 512 * 1024


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class DriveFile:
    file_id:    str
    name:       str
    mime_type:  str
    size_bytes: int = 0

    @property
    def is_pdf(self) -> bool:
        return (
            self.mime_type == "application/pdf"
            or self.name.lower().endswith(".pdf")
        )


@dataclass
class DriveImportResult:
    folder_id:   str
    folder_url:  str
    total_found: int = 0
    total_pdfs:  int = 0
    downloaded:  list[tuple[str, bytes]] = field(default_factory=list)
    skipped:     list[tuple[str, str]]   = field(default_factory=list)
    failed:      list[tuple[str, str]]   = field(default_factory=list)
    error:       Optional[str]           = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.downloaded) > 0


# ─────────────────────────────────────────────
# URL PARSING
# ─────────────────────────────────────────────

_FOLDER_PATTERNS = [
    r"drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_\-]+)",
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_\-]+)",
    r"drive\.google\.com/folderview\?id=([a-zA-Z0-9_\-]+)",
    r"^([a-zA-Z0-9_\-]{25,})$",
]


def extract_folder_id(url: str) -> Optional[str]:
    url = url.strip()
    for pat in _FOLDER_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def is_valid_drive_url(url: str) -> bool:
    return extract_folder_id(url) is not None


# ─────────────────────────────────────────────
# STEP 1 — LIST FILES
# ─────────────────────────────────────────────

def list_drive_folder(folder_id: str) -> tuple[list[DriveFile], Optional[str]]:
    """List all PDFs in a public Google Drive folder via v3 API."""
    try:
        import requests
    except ImportError:
        return [], "requests not installed. Run: pip install requests"

    api_key     = os.environ.get("GOOGLE_API_KEY", "")
    all_files:  list[DriveFile] = []
    page_token: Optional[str]   = None

    while True:
        params: dict = {
            "q":         f"'{folder_id}' in parents and trashed=false",
            "fields":    "nextPageToken,files(id,name,mimeType,size)",
            "pageSize":  1000,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if api_key:
            params["key"] = api_key
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(_DRIVE_API_FILES, params=params, timeout=20)
        except Exception as exc:
            return [], f"Network error contacting Drive API: {exc}"

        if resp.status_code == 403:
            return [], (
                "This Google Drive folder is not publicly accessible.\n"
                "Please enable 'Anyone with the link can view' in the "
                "folder sharing settings and try again."
            )
        if resp.status_code == 404:
            return [], "Folder not found (404). Verify the URL is correct."
        if resp.status_code != 200:
            return [], (
                f"Drive API returned HTTP {resp.status_code}. "
                "Ensure the folder is publicly shared."
            )

        data = resp.json()
        if "error" in data:
            return [], f"Drive API error: {data['error'].get('message', str(data['error']))}"

        for f in data.get("files", []):
            all_files.append(DriveFile(
                file_id    = f["id"],
                name       = f.get("name", "unknown"),
                mime_type  = f.get("mimeType", ""),
                size_bytes = int(f.get("size", 0) or 0),
            ))

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    log.info("Drive API: %d total files in folder %s", len(all_files), folder_id)

    if not all_files:
        return [], (
            "No files found in this folder.\n"
            "• Share the folder as 'Anyone with the link → Viewer'\n"
            "• The folder must not be empty\n"
            "• Paste a folder URL, not a single-file URL"
        )

    pdfs = [f for f in all_files if f.is_pdf]
    if not pdfs:
        return [], (
            f"Found {len(all_files)} file(s) but none are PDFs. "
            "Upload PDF resumes to the folder and try again."
        )

    return pdfs, None


# ─────────────────────────────────────────────
# STEP 2 — DOWNLOAD BY FILE ID
# ─────────────────────────────────────────────

def _download_gdown(file_id: str, dest: Path) -> tuple[bool, str]:
    """Primary downloader: gdown handles confirm tokens natively."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        out = gdown.download(url, str(dest), quiet=True, fuzzy=False)
        if out is None:
            return False, "gdown returned None — file may require sign-in."
        return True, ""
    except ImportError:
        return False, "gdown not installed. Run: pip install gdown"
    except Exception as exc:
        return False, f"gdown: {exc}"


def _download_requests(file_id: str, dest: Path) -> tuple[bool, str]:
    """Fallback downloader: requests + manual confirm-token handling."""
    try:
        import requests
        session = requests.Session()
        headers = {"User-Agent": "Mozilla/5.0"}

        resp = session.get(
            _DRIVE_DOWNLOAD,
            params={"id": file_id, "export": "download"},
            headers=headers, stream=True, timeout=60,
        )

        if resp.status_code in (403, 404):
            return False, f"HTTP {resp.status_code} — file not accessible."
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"

        # Grab confirm token from cookies
        confirm = next(
            (v for k, v in resp.cookies.items() if k.startswith("download_warning")),
            None,
        )
        if confirm:
            resp = session.get(
                _DRIVE_DOWNLOAD,
                params={"id": file_id, "export": "download", "confirm": confirm},
                headers=headers, stream=True, timeout=60,
            )

        # Detect HTML error page
        ct = resp.headers.get("Content-Type", "")
        if "text/html" in ct:
            preview = next(resp.iter_content(2048), b"")
            if b"<html" in preview[:200].lower():
                return False, "Drive returned HTML instead of PDF — check sharing."

        max_bytes = 50 * 1024 * 1024
        written   = 0
        with dest.open("wb") as fh:
            # If we already read a preview chunk, write it first
            if 'preview' in locals() and preview:
                fh.write(preview)
                written += len(preview)
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    written += len(chunk)
                    if written > max_bytes:
                        dest.unlink(missing_ok=True)
                        return False, "File exceeds 50 MB limit."
                    fh.write(chunk)
        return True, ""
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return False, f"requests fallback: {exc}"


def _verify_pdf(path: Path) -> tuple[bool, str]:
    """Confirm file exists, is non-empty, and starts with %PDF-."""
    if not path.exists():
        return False, "File missing after download."
    if path.stat().st_size == 0:
        return False, "Downloaded file is 0 bytes."
    with path.open("rb") as fh:
        magic = fh.read(5)
    if not magic.startswith(b"%PDF-"):
        return False, f"Not a valid PDF (magic={magic!r}). May be HTML or redirect."
    return True, ""


def download_drive_file(
    drive_file: DriveFile,
    dest_dir: Path = TEMP_RESUME_DIR,
    retry: bool = True,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Download one Drive file by its file_id, verify it, return bytes.
    Never uses the filename to construct a URL.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", drive_file.name)
    dest = dest_dir / safe_name

    # Primary: gdown
    ok, err = _download_gdown(drive_file.file_id, dest)

    # Retry once with requests fallback
    if not ok:
        log.warning("Primary download failed [%s]: %s", drive_file.name, err)
        if retry:
            log.info("Retry with requests fallback: %s", drive_file.name)
            ok, err = _download_requests(drive_file.file_id, dest)

    if not ok:
        return None, err

    valid, reason = _verify_pdf(dest)
    if not valid:
        dest.unlink(missing_ok=True)
        return None, f"Verification failed: {reason}"

    try:
        data = dest.read_bytes()
        log.info("✓ %s  (%d KB)", drive_file.name, len(data) // 1024)
        return data, None
    except OSError as exc:
        return None, f"Read error after download: {exc}"


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def import_from_drive(
    folder_url: str,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    max_files: int = 200,
    delay: float = 0.4,
) -> DriveImportResult:
    """
    Full Google Drive import pipeline:
      1. Extract folder ID  2. List PDFs  3. Download each  4. Return bytes

    progress_callback(current, total, filename, status)
      status: "listing" | "ok" | "failed" | "skipped"
    """
    folder_id = extract_folder_id(folder_url)
    if not folder_id:
        return DriveImportResult(
            folder_id="", folder_url=folder_url,
            error=(
                "Could not extract a Google Drive folder ID.\n"
                "Expected: https://drive.google.com/drive/folders/<ID>"
            ),
        )

    result = DriveImportResult(folder_id=folder_id, folder_url=folder_url)

    if progress_callback:
        try: progress_callback(0, 1, "Listing folder…", "listing")
        except Exception: pass

    pdfs, list_err = list_drive_folder(folder_id)
    if list_err:
        result.error = list_err
        return result

    if len(pdfs) > max_files:
        log.warning("Capping at %d files (folder has %d)", max_files, len(pdfs))
        pdfs = pdfs[:max_files]

    result.total_found = len(pdfs)
    result.total_pdfs  = len(pdfs)
    total = len(pdfs)

    for idx, df in enumerate(pdfs):
        # Skip non-PDF MIME types (double safety)
        if df.mime_type and df.mime_type != "application/pdf" and not df.name.lower().endswith(".pdf"):
            result.skipped.append((df.name, f"Non-PDF MIME type: {df.mime_type}"))
            if progress_callback:
                try: progress_callback(idx + 1, total, df.name, "skipped")
                except Exception: pass
            continue

        pdf_bytes, err = download_drive_file(df)

        if err:
            result.failed.append((df.name, err))
            log.error("Failed [%s]: %s", df.name, err)
            if progress_callback:
                try: progress_callback(idx + 1, total, df.name, "failed")
                except Exception: pass
        else:
            result.downloaded.append((df.name, pdf_bytes))
            if progress_callback:
                try: progress_callback(idx + 1, total, df.name, "ok")
                except Exception: pass

        if delay > 0 and idx < total - 1:
            time.sleep(delay)

    log.info(
        "Drive import done: %d downloaded, %d skipped, %d failed",
        len(result.downloaded), len(result.skipped), len(result.failed),
    )

    if not result.downloaded:
        result.error = (
            f"All {total} download(s) failed.\n"
            "Check that each file in the folder is publicly shared."
        )

    return result
