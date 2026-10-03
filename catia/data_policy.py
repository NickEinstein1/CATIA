"""
Production data policy for CATIA.

Live-first by default. Mock data is opt-in for tests/demos only.
Silent mock substitution is forbidden in live mode unless explicitly allowed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class DataUnavailableError(RuntimeError):
    """Raised when required live data cannot be obtained."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def live_mode_default() -> bool:
    """
    Default for analysis runs: True = use live APIs (not mock).

    Override with ``CATIA_USE_MOCK_DATA=1`` to force mock as the process default
    (tests/CI only). Explicit ``use_mock_data=`` on call sites still wins.
    """
    if _env_bool("CATIA_USE_MOCK_DATA", False):
        return False  # mock enabled → not live
    return True


def use_mock_by_default() -> bool:
    return not live_mode_default()


def allow_mock_fallback() -> bool:
    """
    When live fetch fails, may we synthesize mock?

    Default **False** in production. Set ``CATIA_ALLOW_MOCK_FALLBACK=1`` only for
    degraded demos — reports must still flag ``data_mode=degraded_mock``.
    """
    return _env_bool("CATIA_ALLOW_MOCK_FALLBACK", False)


def require_live_feeds() -> bool:
    """Dashboard/API live Earth must surface an error if every feed fails."""
    return _env_bool("CATIA_LIVE_REQUIRE_SOURCE", True)


def provenance_blob(
    *,
    mode: str,
    sources: Optional[Dict[str, Any]] = None,
    notes: Optional[list] = None,
) -> Dict[str, Any]:
    return {
        "data_mode": mode,  # live | mock | degraded_mock
        "sources": sources or {},
        "notes": notes or [],
        "policy": {
            "live_default": live_mode_default(),
            "allow_mock_fallback": allow_mock_fallback(),
        },
    }
