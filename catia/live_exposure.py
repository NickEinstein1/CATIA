"""
Indicative exposure-region overlap for live events (situational overlay only).

Uses coarse world-region polygons shipped with CATIA — not modeled portfolio exposure.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TIER_SCORE = {"high": 1.0, "medium": 0.65, "low": 0.35}


def _ray_cast_point_in_ring(lon: float, lat: float, ring: List[List[float]]) -> bool:
    """Even-odd rule for a single ring (GeoJSON: [lon, lat])."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def _point_in_geojson_polygon(lon: float, lat: float, coords: List[Any]) -> bool:
    if not coords:
        return False
    outer = coords[0]
    if not _ray_cast_point_in_ring(lon, lat, outer):
        return False
    for hole in coords[1:]:
        if _ray_cast_point_in_ring(lon, lat, hole):
            return False
    return True


def _point_in_geometry(lon: float, lat: float, geom: Dict[str, Any]) -> bool:
    gtype = str(geom.get("type") or "")
    coords = geom.get("coordinates")
    if gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
        return abs(float(coords[0]) - lon) < 1e-6 and abs(float(coords[1]) - lat) < 1e-6
    if gtype == "Polygon":
        return _point_in_geojson_polygon(lon, lat, coords)
    if gtype == "MultiPolygon" and isinstance(coords, list):
        return any(_point_in_geojson_polygon(lon, lat, poly) for poly in coords)
    return False


@lru_cache(maxsize=1)
def _load_indicative_regions() -> Tuple[List[Dict[str, Any]], Path]:
    path = Path(__file__).resolve().parent / "data" / "indicative_exposure_regions.geojson"
    if not path.is_file():
        return [], path
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("features") or []), path


def indicative_exposure_geojson() -> Dict[str, Any]:
    features, path = _load_indicative_regions()
    return {
        "type": "FeatureCollection",
        "description": "Indicative world regions — not modeled CATIA exposure.",
        "source_path": str(path),
        "features": features,
    }


def exposure_overlap_for_point(lon: float, lat: float) -> Dict[str, Any]:
    """Return matched indicative regions and a 0–1 overlap score."""
    features, _ = _load_indicative_regions()
    matched: List[str] = []
    tiers: List[str] = []
    for feat in features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        if _point_in_geometry(lon, lat, geom):
            name = str(props.get("name") or "region")
            matched.append(name)
            tier = str(props.get("tier_hint") or "medium").lower()
            tiers.append(tier)

    if not matched:
        return {"regions": [], "tier_hints": [], "overlap_score": 0.0}

    tier_scores = [_TIER_SCORE.get(t, 0.5) for t in tiers]
    overlap = max(tier_scores)
    return {
        "regions": matched,
        "tier_hints": tiers,
        "overlap_score": round(overlap, 3),
    }


def attach_exposure_overlap(event: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(event)
    try:
        lon = float(event["lon"])
        lat = float(event["lat"])
    except (KeyError, TypeError, ValueError):
        out["exposure_overlap"] = {"regions": [], "tier_hints": [], "overlap_score": 0.0}
        return out
    out["exposure_overlap"] = exposure_overlap_for_point(lon, lat)
    return out
