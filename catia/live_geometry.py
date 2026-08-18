"""
GeoJSON helpers for geometry-aware live events (tracks / polygons).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catia.live_catastrophe_feeds import category_color
from catia.geo_hazards import PERIL_VIS_COLORS


def _event_stroke_color(event: Dict[str, Any]) -> str:
    peril = event.get("catia_peril")
    if isinstance(peril, str) and peril in PERIL_VIS_COLORS:
        return PERIL_VIS_COLORS[peril]
    return category_color(str(event.get("category") or ""))


def eonet_entries_to_geometries(geom: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert EONET geometry entries to GeoJSON geometry dicts."""
    out: List[Dict[str, Any]] = []
    for g in geom or []:
        gtype = g.get("type")
        coords = g.get("coordinates")
        if gtype and coords is not None:
            out.append({"type": gtype, "coordinates": coords})
    return out


def eonet_primary_geometry(geom: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the latest EONET geometry entry as primary footprint."""
    if not geom:
        return None
    best: Optional[Dict[str, Any]] = None
    best_ts = ""
    for g in geom:
        if g.get("type") and g.get("coordinates") is not None:
            d = str(g.get("date") or "")
            if best is None or d >= best_ts:
                best = {"type": g["type"], "coordinates": g["coordinates"]}
                best_ts = d
    return best


def events_to_feature_collection(
    events: List[Dict[str, Any]],
    *,
    include_points: bool = True,
    min_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Build a GeoJSON FeatureCollection for map overlays.

    Includes polygon/line footprints and optional point fallbacks.
    """
    features: List[Dict[str, Any]] = []
    for ev in events:
        try:
            conf = float(ev.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_confidence:
            continue

        color = _event_stroke_color(ev)
        props = {
            "id": ev.get("id"),
            "title": str(ev.get("title") or "")[:120],
            "catia_score": ev.get("catia_score"),
            "catia_peril": ev.get("catia_peril"),
            "confidence": conf,
            "stroke": color,
            "fill": color,
            "fillOpacity": 0.18,
            "weight": 2,
        }

        geom = ev.get("geometry")
        gkind = str(ev.get("geometry_kind") or "")
        if isinstance(geom, dict) and gkind not in ("point", "unknown"):
            features.append({"type": "Feature", "geometry": geom, "properties": props})
            continue

        for extra in ev.get("geometry_collection") or []:
            if isinstance(extra, dict) and extra.get("type") not in (None, "Point"):
                features.append({"type": "Feature", "geometry": extra, "properties": props})

        if include_points and ev.get("lat") is not None and ev.get("lon") is not None:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(ev["lon"]), float(ev["lat"])],
                    },
                    "properties": {**props, "role": "centroid"},
                }
            )

    return {"type": "FeatureCollection", "features": features}
