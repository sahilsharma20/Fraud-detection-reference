"""Structured, configured-once logging used across the whole codebase.

WHY a module like this (and why a fresher gets it wrong):
    Freshers sprinkle ``print()`` everywhere. print() has no levels, no
    timestamps, no module context, can't be silenced in production, and can't be
    shipped to a log aggregator. We configure the root logger ONCE here and every
    module just does ``from src.logger import get_logger; log = get_logger(__name__)``.
    Calling ``logging.basicConfig`` repeatedly (the common mistake) is a no-op
    after the first call and silently drops your handlers — so we guard against
    double-configuration explicitly.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False
_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Configure the root logger exactly once.

    Adds a console handler (stdout — important so logs are captured by Docker /
    Render / CI) and a rotating file handler. Idempotent: safe to call from
    multiple entrypoints (training script, API, tests) without duplicating logs.

    Args:
        log_dir: Directory for the rotating log file. Created if missing.
        level: Minimum log level for the root logger.

    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Force UTF-8 on stdout so currency/unicode (₹) in log messages don't crash
    # the Windows console (cp1252) handler with a UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # drop any handlers a library installed before us

    # Console handler FIRST — this is the one that matters in containers (Docker /
    # Render capture stdout). It must always be present.
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File logging is BEST-EFFORT. In a read-only / non-root container the working
    # dir often isn't writable (e.g. Render runs us as a non-root user in /app).
    # Logging must NEVER crash the app — if we can't open a log file, fall back to
    # console-only (which the platform already captures).
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        # UTC date stamp keeps log filenames stable regardless of server timezone.
        logfile = log_path / f"run_{datetime.now(UTC):%Y%m%d}.log"
        # Rotate at 5 MB, keep 3 backups — bounded disk usage on long-running services.
        file_handler = RotatingFileHandler(
            logfile, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("File logging disabled (%s) — logging to stdout only.", exc)

    # Quiet down chatty third-party loggers so our signal isn't buried.
    for noisy in ("matplotlib", "shap", "git", "urllib3", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Args:
        name: Pass ``__name__`` from the calling module.

    Returns:
        A ``logging.Logger`` namespaced to the calling module.

    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
