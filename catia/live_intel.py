"""
CATIA Intelligence layer for live catastrophe feeds.

Turns raw live events into actionable items by:
- mapping events to CATIA perils
- scoring relevance (severity, recency, proximity to focal region)
- producing compact "top alerts" for the dashboard
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from catia.geo_regions import REGION_CENTROIDS


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def infer_catia_peril(event: Dict[str, Any]) -> Optional[str]:
    """
    Map a live event category to a CATIA peril id when possible.
    Returns one of: hurricane, flood, wildfire, earthquake, drought, or None.
    """
    cat = str(event.get("category") or "").lower()
    label = str(event.get("category_label") or "").lower()
    blob = f"{cat} {label}"

    if "earthquake" in blob or cat == "earthquake":
        return "earthquake"
    if "wildfire" in blob or "wildfires" in blob:
        return "wildfire"
    if "flood" in blob or "floods" in blob:
        return "flood"
    if "drought" in blob:
        return "drought"
    # EONET has “Severe Storms”; treat as hurricane-ish for CATIA perils.
    if "storm" in blob or "cyclone" in blob or "hurricane" in blob or "typhoon" in blob:
        return "hurricane"
    return None


def _parse_usgs_magnitude(severity_label: str) -> Optional[float]:
    # expected like "M 4.3"
    s = (severity_label or "").strip().upper()
    if not s.startswith("M"):
        return None
    parts = s.replace("M", "").strip().split()
    if not parts:
        return None
    return _safe_float(parts[0])


def _parse_iso_to_epoch_seconds(iso: str) -> Optional[float]:
    if not iso:
        return None
    # Current format in feeds: "YYYY-mm-dd HH:MM UTC"
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def score_event(
    event: Dict[str, Any],
    *,
    focal_region: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Score an event into [0, 100]. Returns (score, components).

    Components:
    - severity: based on magnitude when available (earthquakes), else baseline
    - recency: decays over ~48h when time is known
    - proximity: 1.0 at 0km, ~0.0 at >=3000km (only when focal_region is known)
    """
    sev = 0.25  # baseline when unknown
    cat = str(event.get("category") or "").lower()
    if cat == "earthquake":
        mag = _parse_usgs_magnitude(str(event.get("severity_label") or ""))
        if mag is not None:
            sev = max(0.0, min(1.0, (mag - 2.5) / 5.5))  # 2.5→0, 8.0→1

    rec = 0.35  # default when time is missing
    t = _parse_iso_to_epoch_seconds(str(event.get("time_iso") or ""))
    if t is not None:
        age_h = max(0.0, (datetime.now(timezone.utc).timestamp() - t) / 3600.0)
        rec = math.exp(-age_h / 24.0)  # ~0.37 at 24h, ~0.14 at 48h

    prox = 0.5  # neutral when no focal region
    if focal_region and focal_region in REGION_CENTROIDS:
        fl, fn = REGION_CENTROIDS[focal_region]
        lat = _safe_float(event.get("lat"))
        lon = _safe_float(event.get("lon"))
        if lat is not None and lon is not None:
            d_km = _haversine_km(fl, fn, lat, lon)
            prox = max(0.0, min(1.0, 1.0 - (d_km / 3000.0)))

    # weights tuned for usefulness
    score = 100.0 * (0.45 * sev + 0.30 * rec + 0.25 * prox)
    return score, {"severity": sev, "recency": rec, "proximity": prox}


@dataclass
class EnrichedLiveEvent:
    raw: Dict[str, Any]
    catia_peril: Optional[str]
    score: float
    score_components: Dict[str, float]


def enrich_and_rank_events(
    events: List[Dict[str, Any]],
    *,
    focal_region: Optional[str] = None,
    peril_filter: Optional[str] = None,
    limit: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Enrich live events with:
    - catia_peril
    - score + components
    Returns sorted list (highest score first).

    ``peril_filter``:
    - ``None`` or ``"all"``: all events
    - ``"__unmapped__"``: only events with no CATIA peril mapping
    - otherwise: exact peril id (e.g. ``"earthquake"``)
    """
    enriched: List[Dict[str, Any]] = []
    pf = (peril_filter or "").strip()
    for ev in events[:limit]:
        peril = infer_catia_peril(ev)
        pk = peril or ""
        if pf == "__unmapped__":
            if pk:
                continue
        elif pf and pf != "all":
            if pk != pf:
                continue
        score, comps = score_event(ev, focal_region=focal_region)
        out = dict(ev)
        out["catia_peril"] = peril or ""
        out["catia_score"] = float(score)
        out["catia_score_components"] = comps
        enriched.append(out)
    enriched.sort(key=lambda e: float(e.get("catia_score") or 0.0), reverse=True)
    return enriched

