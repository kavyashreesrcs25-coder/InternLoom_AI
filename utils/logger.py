"""
InternLoom AI - Centralized Logging Utility
Provides structured, leveled logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


LOG_DIR = Path(__file__).parent.parent / "data" / "outputs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with console + optional file output.

    Args:
        name: Logger name (usually __name__ of the calling module).
        level: Logging level (default DEBUG).
        log_to_file: Whether to write logs to a rotating file.
        log_filename: Override log file name (default: internloom_<date>.log).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ───────────────────────────────────────────────────────
    if log_to_file:
        if log_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            log_filename = f"internloom_{date_str}.log"

        log_path = LOG_DIR / log_filename
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Could not create log file %s: %s", log_path, exc)

    # Prevent propagation to root logger
    logger.propagate = False
    return logger


class ParseLogger:
    """
    Specialised logger for tracking per-resume parse events.
    Accumulates warnings/errors for the parse quality report.
    """

    def __init__(self) -> None:
        self._log = get_logger("ParseLogger")
        self._events: list[dict] = []

    def record(
        self,
        filename: str,
        status: str,
        message: str,
        level: str = "INFO",
    ) -> None:
        """Record a parse event for later reporting."""
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "filename": filename,
            "status": status,
            "message": message,
            "level": level,
        }
        self._events.append(entry)

        log_fn = getattr(self._log, level.lower(), self._log.info)
        log_fn("[%s] %s — %s", filename, status, message)

    def get_events(self) -> list[dict]:
        """Return all recorded parse events."""
        return list(self._events)

    def clear(self) -> None:
        """Reset recorded events (e.g., between sessions)."""
        self._events.clear()


# ── Module-level default logger ────────────────────────────────────────────
log = get_logger("internloom")
parse_logger = ParseLogger()
