"""
Site geography helpers for property / land viability assessment.

Resolve coordinates (and optional address) to the nearest CATIA named region.
Indicative only — not asset-level underwriting geography.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Optional, Tuple

import requests

from catia.geo_regions import REGION_CENTROIDS
from catia.live_catastrophe_feeds import _session
from catia.live_exposure import exposure_overlap_for_point

logger = logging.getLogger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def nearest_region(lat: float, lon: float) -> Dict[str, Any]:
    """Return nearest CATIA region id, centroid, and distance in km."""
    best_id = ""
    best_dist = float("inf")
    best_centroid: Tuple[float, float] = (0.0, 0.0)
    for rid, (rlat, rlon) in REGION_CENTROIDS.items():
        d = haversine_km(lat, lon, rlat, rlon)
        if d < best_dist:
            best_dist = d
            best_id = rid
            best_centroid = (rlat, rlon)
    return {
        "region_id": best_id,
        "region_label": best_id.replace("_", " "),
        "centroid_lat": best_centroid[0],
        "centroid_lon": best_centroid[1],
        "distance_km": round(best_dist, 1),
    }


def geocode_address(address: str, *, timeout: int = 12) -> Optional[Dict[str, Any]]:
    """
    Optional OpenStreetMap Nominatim geocode (respect usage policy).

    Disabled unless ``CATIA_SITE_GEOCODE=1``. Returns lat/lon/display_name or None.
    """
    if os.environ.get("CATIA_SITE_GEOCODE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    q = (address or "").strip()
    if not q:
        return None
    url = os.environ.get(
        "CATIA_NOMINATIM_URL",
        "https://nominatim.openstreetmap.org/search",
    )
    try:
        sess = _session()
        r = sess.get(
            url,
            params={"q": q, "format": "json", "limit": 1},
            headers={"User-Agent": "CATIA-site-viability/1.0 (research)"},
            timeout=timeout,
        )
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        hit = rows[0]
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "display_name": str(hit.get("display_name") or q)[:240],
            "source": "nominatim",
        }
    except Exception as e:
        logger.warning("Geocode failed for %r: %s", q[:80], e)
        return None


def resolve_site_location(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve a site to coordinates + nearest CATIA region + indicative exposure.

    Prefer explicit lat/lon; fall back to geocode when enabled.
    """
    resolved_address = (address or "").strip() or None
    src = "coordinates"

    if lat is None or lon is None:
        if not resolved_address:
            raise ValueError("Provide lat/lon or address")
        geo = geocode_address(resolved_address)
        if geo is None:
            raise ValueError(
                "Could not resolve address. Pass lat/lon, or set CATIA_SITE_GEOCODE=1 "
                "for Nominatim geocoding."
            )
        lat = float(geo["lat"])
        lon = float(geo["lon"])
        resolved_address = str(geo.get("display_name") or resolved_address)
        src = "geocode"

    lat_f = float(lat)
    lon_f = float(lon)
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        raise ValueError("lat must be in [-90, 90] and lon in [-180, 180]")

    nearest = nearest_region(lat_f, lon_f)
    overlap = exposure_overlap_for_point(lon_f, lat_f)

    return {
        "lat": lat_f,
        "lon": lon_f,
        "address": resolved_address,
        "resolution_source": src,
        "nearest_region": nearest,
        "exposure_overlap": overlap,
    }
