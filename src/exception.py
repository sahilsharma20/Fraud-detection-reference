"""Custom exception that captures *where* an error happened, not just *what*.

WHY (fresher pitfall):
    A bare ``raise ValueError("bad data")`` tells you nothing about which file
    and line blew up once it's buried under three layers of pipeline calls.
    Wrapping the original error and attaching file:line gives you a one-glance
    diagnosis in the logs — the kind of thing you only add after you've debugged
    a production incident at 2am.
"""

from __future__ import annotations

import sys
import traceback
from types import TracebackType


def _format_detail(error: BaseException, exc_tb: TracebackType | None) -> str:
    """Build a 'file:line - message' string from an exception's traceback."""
    if exc_tb is None:
        return f"{type(error).__name__}: {error}"
    # Walk to the LAST frame in the traceback — that's where the error actually
    # originated, not where we happened to catch it.
    last = traceback.extract_tb(exc_tb)[-1]
    return (
        f"{type(error).__name__} in [{last.filename}] "
        f"at line [{last.lineno}] in [{last.name}]: {error}"
    )


class FraudDetectionError(Exception):
    """Domain exception carrying rich location context.

    Usage:
        try:
            risky()
        except Exception as exc:                 # noqa: BLE001 - re-raise wrapped
            raise FraudDetectionError(exc) from exc
    """

    def __init__(self, error: BaseException | str):
        if isinstance(error, BaseException):
            _, _, exc_tb = sys.exc_info()
            message = _format_detail(error, exc_tb)
        else:
            message = str(error)
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # noqa: D105
        return self.message
