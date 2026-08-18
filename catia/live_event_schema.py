"""
Normalized live-event fields: geometry, provenance, and confidence.

Shared by dashboard maps, REST API, and agents consuming the intelligence layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

PARSER_VERSION = "catia_live_v3"

# Baseline trust by upstream feed (observational, not modeled loss).
_SOURCE_RELIABILITY: Dict[str, float] = {
    "usgs": 0.92,
    "eonet": 0.78,
    "gdacs": 0.86,
}

_GEOMETRY_CONFIDENCE: Dict[str, float] = {
    "point": 0.82,
    "linestring": 0.88,
    "polygon": 0.92,
    "multipoint": 0.85,
    "geometrycollection": 0.87,
    "unknown": 0.65,
}


def point_geometry(lon: float, lat: float) -> Dict[str, Any]:
    return {"type": "Point", "coordinates": [float(lon), float(lat)]}


def geometry_kind(geom: Optional[Dict[str, Any]]) -> str:
    if not geom or not isinstance(geom, dict):
        return "unknown"
    gtype = str(geom.get("type") or "unknown").lower()
    if gtype == "multipoint":
        return "multipoint"
    if gtype in ("linestring", "multilinestring"):
        return "linestring"
    if gtype in ("polygon", "multipolygon"):
        return "polygon"
    if gtype == "geometrycollection":
        return "geometrycollection"
    if gtype == "point":
        return "point"
    return "unknown"


def _severity_is_known(event: Dict[str, Any]) -> bool:
    label = str(event.get("severity_label") or "").strip()
    if not label:
        return False
    if event.get("severity_value") is not None:
        return True
    return label.upper().startswith("M") or any(c.isdigit() for c in label)


def _time_is_known(event: Dict[str, Any]) -> bool:
    return bool(str(event.get("time_iso") or "").strip())


def compute_confidence(event: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Heuristic confidence in [0, 1] with factor breakdown."""
    feed = str((event.get("provenance") or {}).get("feed") or event.get("source") or "").lower()
    feed_key = "eonet" if "eonet" in feed else feed.split()[0].lower() if feed else "unknown"
    if feed_key not in _SOURCE_RELIABILITY:
        if "usgs" in feed_key:
            feed_key = "usgs"
        elif "gdacs" in feed_key:
            feed_key = "gdacs"
        elif "eonet" in feed_key:
            feed_key = "eonet"
        else:
            feed_key = "eonet"

    src_rel = _SOURCE_RELIABILITY.get(feed_key, 0.7)
    gkind = str(event.get("geometry_kind") or geometry_kind(event.get("geometry")))
    geom_c = _GEOMETRY_CONFIDENCE.get(gkind, _GEOMETRY_CONFIDENCE["unknown"])
    sev_c = 0.9 if _severity_is_known(event) else 0.55
    time_c = 0.88 if _time_is_known(event) else 0.5

    score = 0.35 * src_rel + 0.30 * geom_c + 0.20 * sev_c + 0.15 * time_c
    score = max(0.0, min(1.0, score))
    return score, {
        "source_reliability": src_rel,
        "geometry": geom_c,
        "severity_known": sev_c,
        "time_known": time_c,
    }


def attach_provenance(
    event: Dict[str, Any],
    *,
    feed: str,
    source_event_id: str,
    source_url: str = "",
    observed_at: str = "",
    updated_at: str = "",
) -> Dict[str, Any]:
    out = dict(event)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out["provenance"] = {
        "feed": feed,
        "source": str(out.get("source") or feed.upper()),
        "source_event_id": source_event_id,
        "source_url": source_url or str(out.get("url") or ""),
        "parser_version": PARSER_VERSION,
        "ingested_at": now,
        "observed_at": observed_at or str(out.get("time_iso") or ""),
        "updated_at": updated_at or observed_at or str(out.get("time_iso") or ""),
    }
    return out


def finalize_event(
    event: Dict[str, Any],
    *,
    geometry: Optional[Dict[str, Any]] = None,
    geometry_collection: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach geometry metadata, provenance (if missing), and confidence."""
    out = dict(event)
    if geometry is not None:
        out["geometry"] = geometry
    if geometry_collection:
        out["geometry_collection"] = geometry_collection
    g = out.get("geometry")
    if isinstance(g, dict):
        out["geometry_kind"] = geometry_kind(g)
    else:
        out["geometry_kind"] = "unknown"
    if "provenance" not in out:
        out = attach_provenance(
            out,
            feed=str(out.get("source") or "unknown").lower(),
            source_event_id=str(out.get("id") or ""),
            source_url=str(out.get("url") or ""),
        )
    conf, factors = compute_confidence(out)
    out["confidence"] = conf
    out["confidence_factors"] = factors
    return out
