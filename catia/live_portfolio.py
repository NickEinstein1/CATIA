"""
Map live catastrophe feed events onto portfolio one-storm shocks.

Uses event lat/lon + inferred peril; intensity is parsed when available
(USGS magnitude) or filled with peril defaults / score-scaled heuristics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from catia.live_intel import infer_catia_peril

# Vulnerability curve units: hurricane mph, flood depth-ish, EQ magnitude, wildfire index
DEFAULT_INTENSITY: Dict[str, float] = {
    "hurricane": 120.0,
    "flood": 6.0,
    "wildfire": 55.0,
    "earthquake": 6.5,
    "drought": 40.0,
}

DEFAULT_RADIUS_KM: Dict[str, float] = {
    "hurricane": 300.0,
    "flood": 80.0,
    "wildfire": 40.0,
    "earthquake": 120.0,
    "drought": 200.0,
}


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _parse_magnitude(event: Dict[str, Any]) -> Optional[float]:
    for key in ("severity_value", "magnitude", "mag"):
        m = _safe_float(event.get(key))
        if m is not None:
            return m
    label = str(event.get("severity_label") or "")
    m = re.search(r"M\s*([0-9]+(?:\.[0-9]+)?)", label, re.I)
    if m:
        return float(m.group(1))
    return None


def _intensity_for_event(event: Dict[str, Any], peril: str) -> Dict[str, Any]:
    """Return intensity and provenance note."""
    if peril == "earthquake":
        mag = _parse_magnitude(event)
        if mag is not None:
            return {"intensity": mag, "intensity_source": "event_magnitude"}
    sev = _safe_float(event.get("severity_value"))
    if sev is not None and peril != "earthquake":
        # Clamp into a plausible band for the peril curve
        base = DEFAULT_INTENSITY.get(peril, 50.0)
        # Treat unknown numeric severity as a 0–1 or 0–100 score-ish
        if sev <= 1.0:
            intensity = base * (0.6 + 0.6 * sev)
        elif sev <= 10.0 and peril == "flood":
            intensity = sev
        else:
            intensity = min(base * 1.3, max(base * 0.5, sev))
        return {"intensity": round(float(intensity), 2), "intensity_source": "severity_value"}

    score = _safe_float(event.get("catia_score"))
    base = DEFAULT_INTENSITY.get(peril, 80.0)
    if score is not None:
        # score 0–100 → 0.7×–1.25× default
        factor = 0.7 + 0.55 * max(0.0, min(1.0, score / 100.0))
        return {
            "intensity": round(base * factor, 2),
            "intensity_source": "score_scaled_default",
        }
    return {"intensity": base, "intensity_source": "peril_default"}


def live_event_to_one_storm(
    event: Dict[str, Any],
    *,
    radius_km: Optional[float] = None,
    intensity: Optional[float] = None,
    peril: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a PortfolioOneStormSpec-compatible dict from a live feed event.
    """
    lat = _safe_float(event.get("lat"))
    lon = _safe_float(event.get("lon"))
    if lat is None or lon is None:
        raise ValueError("Live event requires lat and lon")

    peril_id = peril or event.get("catia_peril") or infer_catia_peril(event)
    if not peril_id:
        raise ValueError(
            f"Cannot map live event category {event.get('category')!r} to a CATIA peril"
        )
    peril_id = str(peril_id)
    if peril_id == "drought":
        # VulnerabilitySet may not include drought; fall back to wildfire-ish heat proxy
        peril_id = "wildfire"

    if intensity is not None:
        inten = float(intensity)
        src = "user_override"
    else:
        meta = _intensity_for_event(event, peril_id)
        inten = float(meta["intensity"])
        src = str(meta["intensity_source"])

    r = float(radius_km) if radius_km is not None else float(
        DEFAULT_RADIUS_KM.get(peril_id, 150.0)
    )

    return {
        "peril": peril_id,
        "intensity": inten,
        "mode": "radius",
        "center_lat": lat,
        "center_lon": lon,
        "radius_km": r,
        "live_event_id": event.get("id"),
        "live_event_title": event.get("title"),
        "live_event_source": event.get("source"),
        "intensity_source": src,
        "note": (
            "Mapped from live feed event — radius footprint + single intensity; "
            "not a catalog windfield or shake map."
        ),
    }


def find_live_event(
    events: List[Dict[str, Any]],
    event_id: str,
) -> Dict[str, Any]:
    eid = str(event_id)
    for ev in events:
        if str(ev.get("id")) == eid:
            return ev
    raise ValueError(f"Live event not found: {event_id}")
