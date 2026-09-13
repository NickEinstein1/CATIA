"""
Hazard footprint intensity fields for portfolio one-storm / live-hit shocks.

Replaces flat “same intensity everywhere in the radius” with distance-decay
stubs: windfield (TC), shake (EQ), flood bowl, wildfire ring — plus optional
GeoJSON track / polygon masks from live feeds.

Indicative research geometry only — not NHC windfields or USGS ShakeMaps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from catia.site_geo import haversine_km

FootprintKind = Literal[
    "auto",
    "uniform",
    "windfield",
    "shake",
    "flood_bowl",
    "wildfire",
]

PERIL_DEFAULT_FOOTPRINT: Dict[str, FootprintKind] = {
    "hurricane": "windfield",
    "earthquake": "shake",
    "flood": "flood_bowl",
    "wildfire": "wildfire",
    "drought": "uniform",
}


def resolve_footprint_kind(peril: str, footprint: Optional[str] = None) -> str:
    kind = (footprint or "auto").lower().strip()
    if kind in ("", "auto"):
        return PERIL_DEFAULT_FOOTPRINT.get(str(peril), "uniform")
    if kind not in ("uniform", "windfield", "shake", "flood_bowl", "wildfire"):
        raise ValueError(
            f"Unknown footprint {footprint!r}; use auto|uniform|windfield|shake|flood_bowl|wildfire"
        )
    return kind


def _point_in_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon for a single exterior ring (lon/lat)."""
    if len(ring) < 3:
        return False
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_geojson_polygon(lon: float, lat: float, geom: Dict[str, Any]) -> bool:
    gtype = str(geom.get("type") or "")
    coords = geom.get("coordinates")
    if not coords:
        return False
    if gtype == "Polygon":
        # exterior ring only for stub
        return _point_in_ring(lon, lat, coords[0])
    if gtype == "MultiPolygon":
        for poly in coords:
            if poly and _point_in_ring(lon, lat, poly[0]):
                return True
    return False


def _segment_distance_km(
    lat: float,
    lon: float,
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
) -> float:
    """Approximate distance from point to great-circle segment (sampled)."""
    # Coarse sample along segment — good enough for indicative footprints
    best = min(
        haversine_km(lat, lon, a_lat, a_lon),
        haversine_km(lat, lon, b_lat, b_lon),
    )
    for t in (0.25, 0.5, 0.75):
        plat = a_lat + t * (b_lat - a_lat)
        plon = a_lon + t * (b_lon - a_lon)
        best = min(best, haversine_km(lat, lon, plat, plon))
    return best


def distance_to_linestring_km(
    lat: float,
    lon: float,
    coords: Sequence[Sequence[float]],
) -> float:
    """Min distance (km) from point to LineString coordinates [[lon,lat], ...]."""
    if not coords:
        return float("inf")
    if len(coords) == 1:
        return haversine_km(lat, lon, float(coords[0][1]), float(coords[0][0]))
    best = float("inf")
    for i in range(len(coords) - 1):
        a_lon, a_lat = float(coords[i][0]), float(coords[i][1])
        b_lon, b_lat = float(coords[i + 1][0]), float(coords[i + 1][1])
        best = min(best, _segment_distance_km(lat, lon, a_lat, a_lon, b_lat, b_lon))
    return best


def extract_track_coords(geometry: Optional[Dict[str, Any]]) -> Optional[List[List[float]]]:
    if not geometry or not isinstance(geometry, dict):
        return None
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "LineString" and isinstance(coords, list) and coords:
        return [[float(c[0]), float(c[1])] for c in coords if len(c) >= 2]
    if gtype == "MultiLineString" and isinstance(coords, list) and coords:
        # flatten first line for stub
        line = coords[0]
        return [[float(c[0]), float(c[1])] for c in line if len(c) >= 2]
    return None


def extract_polygon(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not geometry or not isinstance(geometry, dict):
        return None
    gtype = str(geometry.get("type") or "")
    if gtype in ("Polygon", "MultiPolygon"):
        return geometry
    return None


def decay_intensity(
    peak: float,
    distance_km: float,
    radius_km: float,
    kind: str,
) -> float:
    """
    Local intensity at distance from event axis/center.

    Returns 0 when outside the effective radius.
    """
    r = max(float(radius_km), 1e-6)
    d = max(0.0, float(distance_km))
    p = float(peak)
    if d > r:
        return 0.0
    x = d / r  # 0 at center, 1 at edge

    if kind == "uniform":
        return p
    if kind == "windfield":
        # Soft Holland-like: peak near RMW (~0.2 R), then exponential decay
        rmw = 0.2
        if x <= rmw:
            # rise from 0.85 peak at eye toward peak at RMW
            return p * (0.85 + 0.15 * (x / rmw))
        # beyond RMW: exp decay to ~0.15 peak at R
        t = (x - rmw) / (1.0 - rmw)
        return p * math.exp(-2.2 * t)
    if kind == "shake":
        # Magnitude / intensity attenuates with log distance
        # I(d) = peak * (1 / (1 + d/d0)^1.4) clipped to radius
        d0 = max(8.0, 0.15 * r)
        return p / ((1.0 + d / d0) ** 1.4)
    if kind == "flood_bowl":
        # Depth falls roughly linearly from center
        return p * max(0.0, 1.0 - x) ** 1.1
    if kind == "wildfire":
        # Hot core then sharp falloff
        return p * math.exp(-3.0 * x)
    return p


def local_intensity_for_point(
    lat: float,
    lon: float,
    *,
    center_lat: float,
    center_lon: float,
    peak_intensity: float,
    radius_km: float,
    footprint: str,
    track_coords: Optional[Sequence[Sequence[float]]] = None,
    polygon: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate footprint at one location.

    Returns dict with intensity, distance_km, inside, reference ('center'|'track'|'polygon').
    """
    kind = footprint
    # Geometry mask: polygon requires containment (plus optional buffer = radius from centroid)
    if polygon is not None:
        inside_poly = point_in_geojson_polygon(lon, lat, polygon)
        d_center = haversine_km(lat, lon, center_lat, center_lon)
        if not inside_poly:
            return {
                "intensity": 0.0,
                "distance_km": round(d_center, 2),
                "inside": False,
                "reference": "polygon",
            }
        inten = decay_intensity(peak_intensity, d_center, radius_km, kind)
        return {
            "intensity": round(inten, 4),
            "distance_km": round(d_center, 2),
            "inside": inten > 0,
            "reference": "polygon",
        }

    if track_coords:
        d = distance_to_linestring_km(lat, lon, track_coords)
        inten = decay_intensity(peak_intensity, d, radius_km, kind)
        return {
            "intensity": round(inten, 4),
            "distance_km": round(d, 2),
            "inside": inten > 0 and d <= radius_km,
            "reference": "track",
        }

    d = haversine_km(lat, lon, center_lat, center_lon)
    inten = decay_intensity(peak_intensity, d, radius_km, kind)
    return {
        "intensity": round(inten, 4),
        "distance_km": round(d, 2),
        "inside": inten > 0 and d <= radius_km,
        "reference": "center",
    }


def footprint_summary(
    *,
    kind: str,
    peril: str,
    peak_intensity: float,
    radius_km: float,
    has_track: bool,
    has_polygon: bool,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "peril": peril,
        "peak_intensity": peak_intensity,
        "radius_km": radius_km,
        "geometry": (
            "polygon" if has_polygon else "track" if has_track else "radial"
        ),
        "note": (
            f"Indicative {kind} intensity field — not a catalog windfield, "
            "ShakeMap, or flood inundation product."
        ),
    }
