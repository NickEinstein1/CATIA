"""
Real DEM and floodplain providers for site viability assessment.

Sources (public, no API keys by default):

- **USGS EPQS** — CONUS / US territories elevation from 3DEP (meters).
- **Open-Meteo Elevation** — global DEM fallback.
- **FEMA NFHL** — US flood hazard zone identify (SFHA / zone codes).

Respect each provider's terms and rate limits. Results are cached on disk.
When providers fail or ``CATIA_SITE_TOPO=0``, callers fall back to heuristics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catia.live_catastrophe_feeds import _session

logger = logging.getLogger(__name__)

USGS_EPQS_URL = os.environ.get(
    "CATIA_USGS_EPQS_URL",
    "https://epqs.nationalmap.gov/v1/json",
)
OPEN_METEO_ELEV_URL = os.environ.get(
    "CATIA_OPEN_METEO_ELEV_URL",
    "https://api.open-meteo.com/v1/elevation",
)
FEMA_NFHL_MAPSERVER_URL = os.environ.get(
    "CATIA_FEMA_NFHL_URL",
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer",
)
# Flood Hazard Zones layer (S_Fld_Haz_Ar) — confirmed on public NFHL MapServer
FEMA_FLOOD_ZONE_LAYER = int(os.environ.get("CATIA_FEMA_FLOOD_LAYER", "28"))

_CACHE_TTL_SEC = float(os.environ.get("CATIA_SITE_TOPO_CACHE_SEC", str(7 * 24 * 3600)))
_OFFSET_DEG = 0.0009  # ~100 m at mid latitudes for slope neighborhood


def topo_providers_enabled() -> bool:
    return os.environ.get("CATIA_SITE_TOPO", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _in_conus_ish(lat: float, lon: float) -> bool:
    """Rough CONUS + nearby coastal shelf for USGS/FEMA eligibility."""
    return 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.0


def _in_us_territories_or_pr(lat: float, lon: float) -> bool:
    if 17.5 <= lat <= 18.6 and -67.5 <= lon <= -65.0:  # Puerto Rico
        return True
    if 18.0 <= lat <= 22.5 and -161.0 <= lon <= -154.0:  # Hawaii
        return True
    if 51.0 <= lat <= 72.0 and -180.0 <= lon <= -129.0:  # Alaska (partial)
        return True
    return False


def _fema_eligible(lat: float, lon: float) -> bool:
    return _in_conus_ish(lat, lon) or (17.5 <= lat <= 18.6 and -67.5 <= lon <= -65.0)


def _cache_dir() -> Path:
    base = os.environ.get("CATIA_CACHE_DIR")
    if base:
        root = Path(base)
    else:
        from catia.config import DATA_CONFIG

        root = Path(DATA_CONFIG.get("cache_dir", "data/cache"))
    path = root / "site_topo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(kind: str, lat: float, lon: float) -> str:
    raw = f"{kind}:{lat:.5f}:{lon:.5f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _cache_get(kind: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    path = _cache_dir() / f"{_cache_key(kind, lat, lon)}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        if time.time() - float(blob.get("_ts", 0)) > _CACHE_TTL_SEC:
            return None
        data = blob.get("data")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cache_set(kind: str, lat: float, lon: float, data: Dict[str, Any]) -> None:
    path = _cache_dir() / f"{_cache_key(kind, lat, lon)}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "data": data}, f)
    except Exception as e:
        logger.debug("Topo cache write skipped: %s", e)


def fetch_usgs_elevation(lat: float, lon: float, *, timeout: int = 12) -> Optional[Dict[str, Any]]:
    """USGS Elevation Point Query Service (3DEP)."""
    cached = _cache_get("usgs_epqs", lat, lon)
    if cached is not None:
        return cached
    try:
        sess = _session()
        r = sess.get(
            USGS_EPQS_URL,
            params={
                "x": lon,
                "y": lat,
                "wkid": 4326,
                "units": "Meters",
                "includeDate": "false",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("value")
        if raw is None or str(raw).lower() in ("", "null", "nan", "-1000000"):
            return None
        elev = float(raw)
        if elev < -500:  # EPQS sentinel for no data
            return None
        out = {
            "elevation_m": round(elev, 2),
            "source": "usgs_epqs",
            "resolution_m": data.get("resolution"),
            "provider_url": USGS_EPQS_URL,
        }
        _cache_set("usgs_epqs", lat, lon, out)
        return out
    except Exception as e:
        logger.warning("USGS EPQS failed for (%.4f, %.4f): %s", lat, lon, e)
        return None


def fetch_open_meteo_elevation(lat: float, lon: float, *, timeout: int = 12) -> Optional[Dict[str, Any]]:
    """Open-Meteo global elevation API."""
    cached = _cache_get("open_meteo", lat, lon)
    if cached is not None:
        return cached
    try:
        sess = _session()
        r = sess.get(
            OPEN_METEO_ELEV_URL,
            params={"latitude": lat, "longitude": lon},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        elevs = data.get("elevation") or []
        if not elevs:
            return None
        elev = float(elevs[0])
        out = {
            "elevation_m": round(elev, 2),
            "source": "open_meteo",
            "provider_url": OPEN_METEO_ELEV_URL,
        }
        _cache_set("open_meteo", lat, lon, out)
        return out
    except Exception as e:
        logger.warning("Open-Meteo elevation failed for (%.4f, %.4f): %s", lat, lon, e)
        return None


def fetch_elevation(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Prefer USGS for US points, else Open-Meteo."""
    if _in_conus_ish(lat, lon) or _in_us_territories_or_pr(lat, lon):
        hit = fetch_usgs_elevation(lat, lon)
        if hit is not None:
            return hit
    return fetch_open_meteo_elevation(lat, lon)


def _meters_per_deg(lat: float) -> Tuple[float, float]:
    """Approximate meters per degree longitude/latitude at ``lat``."""
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat))
    return max(1.0, m_lon), m_lat


def estimate_slope_percent(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Estimate local slope % from a 5-point DEM neighborhood (~100 m offsets).
    """
    center = fetch_elevation(lat, lon)
    if center is None:
        return None
    z0 = float(center["elevation_m"])
    samples: List[Tuple[str, float, float, float]] = [("center", lat, lon, z0)]
    offsets = [
        ("n", lat + _OFFSET_DEG, lon),
        ("s", lat - _OFFSET_DEG, lon),
        ("e", lat, lon + _OFFSET_DEG),
        ("w", lat, lon - _OFFSET_DEG),
    ]
    for name, la, lo in offsets:
        hit = fetch_elevation(la, lo)
        if hit is None:
            continue
        samples.append((name, la, lo, float(hit["elevation_m"])))

    if len(samples) < 3:
        return {
            "slope_percent": 0.0,
            "slope_class": "unknown",
            "samples": len(samples),
            "elevation_source": center.get("source"),
        }

    m_lon, m_lat = _meters_per_deg(lat)
    max_slope = 0.0
    for name, la, lo, z in samples[1:]:
        dy = (la - lat) * m_lat
        dx = (lo - lon) * m_lon
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            continue
        slope = abs(z - z0) / dist * 100.0
        max_slope = max(max_slope, slope)

    if max_slope < 2.0:
        slope_class = "flat_to_gentle"
    elif max_slope < 8.0:
        slope_class = "gentle"
    elif max_slope < 20.0:
        slope_class = "moderate"
    else:
        slope_class = "steep_or_varied"

    return {
        "slope_percent": round(max_slope, 2),
        "slope_class": slope_class,
        "samples": len(samples),
        "elevation_source": center.get("source"),
        "center_elevation_m": z0,
    }


def _classify_fema_zone(fld_zone: str, zone_subty: str = "", sfha: str = "") -> Dict[str, Any]:
    z = (fld_zone or "").strip().upper()
    sub = (zone_subty or "").strip().upper()
    sfha_tf = str(sfha).strip().upper() in ("T", "TRUE", "YES", "1")

    high_zones = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE", "VO"}
    moderate_zones = {"X", "B", "C"}

    if z in high_zones or z.startswith("A") or z.startswith("V"):
        risk = "sfha_high"
        floodplain_hint = "elevated_flood_sensitivity"
        in_sfha = True
    elif z == "X" and ("0.2" in sub or "SHADED" in sub or "500" in sub):
        risk = "moderate_0_2_pct"
        floodplain_hint = "moderate_flood_sensitivity"
        in_sfha = False
    elif z in moderate_zones:
        risk = "low_to_moderate"
        floodplain_hint = "lower_flood_sensitivity"
        in_sfha = sfha_tf
    elif z == "D":
        risk = "undetermined"
        floodplain_hint = "moderate_flood_sensitivity"
        in_sfha = False
    elif not z:
        risk = "unknown"
        floodplain_hint = "moderate_flood_sensitivity"
        in_sfha = sfha_tf
    else:
        risk = "other"
        floodplain_hint = "moderate_flood_sensitivity"
        in_sfha = sfha_tf or z in high_zones

    return {
        "fld_zone": z or None,
        "zone_subtype": zone_subty or None,
        "sfha": in_sfha,
        "flood_risk_class": risk,
        "floodplain_hint": floodplain_hint,
    }


def fetch_fema_flood_zone(lat: float, lon: float, *, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """FEMA National Flood Hazard Layer query at a point (US only)."""
    if not _fema_eligible(lat, lon):
        return None
    cached = _cache_get("fema_nfhl", lat, lon)
    if cached is not None:
        return cached
    query_url = f"{FEMA_NFHL_MAPSERVER_URL.rstrip('/')}/{FEMA_FLOOD_ZONE_LAYER}/query"
    try:
        sess = _session()
        r = sess.get(
            query_url,
            params={
                "where": "1=1",
                "geometry": f"{lon},{lat}",
                "geometryType": "esriGeometryPoint",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,V_DATUM,DEPTH,LEN_UNIT",
                "returnGeometry": "false",
                "f": "json",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            logger.warning("FEMA NFHL error payload: %s", data.get("error"))
            return None
        features = data.get("features") or []
        best: Optional[Dict[str, Any]] = None
        for feat in features:
            attrs = feat.get("attributes") or {}
            fld = attrs.get("FLD_ZONE") or attrs.get("fld_zone")
            sub = attrs.get("ZONE_SUBTY") or attrs.get("zone_subty") or ""
            sfha = attrs.get("SFHA_TF") or attrs.get("sfha_tf") or ""
            if not fld:
                continue
            classified = _classify_fema_zone(str(fld), str(sub), str(sfha))
            bfe = attrs.get("STATIC_BFE")
            try:
                bfe_f = (
                    float(bfe)
                    if bfe is not None and str(bfe) not in ("", "-9999", "Null")
                    else None
                )
            except (TypeError, ValueError):
                bfe_f = None
            best = {
                **classified,
                "source": "fema_nfhl",
                "base_flood_elevation": bfe_f,
                "vertical_datum": attrs.get("V_DATUM"),
                "depth": attrs.get("DEPTH"),
                "provider_url": query_url,
                "layer": FEMA_FLOOD_ZONE_LAYER,
            }
            if classified["sfha"] or classified["flood_risk_class"] == "sfha_high":
                break
        if best is None:
            best = {
                "fld_zone": None,
                "zone_subtype": None,
                "sfha": False,
                "flood_risk_class": "no_zone_returned",
                "floodplain_hint": "lower_flood_sensitivity",
                "source": "fema_nfhl",
                "provider_url": query_url,
                "layer": FEMA_FLOOD_ZONE_LAYER,
                "note": "NFHL layer query returned no intersecting flood-zone features.",
            }
        _cache_set("fema_nfhl", lat, lon, best)
        return best
    except Exception as e:
        logger.warning("FEMA NFHL query failed for (%.4f, %.4f): %s", lat, lon, e)
        return None


def fetch_site_terrain(lat: float, lon: float) -> Dict[str, Any]:
    """
    Compose DEM + slope + floodplain for a site.

    Returns a dict with ``ok`` False when real providers are disabled or empty.
    """
    if not topo_providers_enabled():
        return {"ok": False, "reason": "CATIA_SITE_TOPO disabled"}

    elev = fetch_elevation(lat, lon)
    slope = estimate_slope_percent(lat, lon) if elev else None
    flood = fetch_fema_flood_zone(lat, lon)

    if elev is None and flood is None:
        return {"ok": False, "reason": "providers_unavailable"}

    return {
        "ok": True,
        "elevation": elev,
        "slope": slope,
        "floodplain": flood,
        "providers_used": [
            x
            for x in (
                (elev or {}).get("source"),
                (flood or {}).get("source"),
            )
            if x
        ],
    }
