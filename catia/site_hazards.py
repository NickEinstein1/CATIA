"""
Hazard and topography profile for a site (actuarial screening).

Combines CATIA PERIL_CONFIG regional membership, real DEM/floodplain providers
(USGS EPQS, Open-Meteo, FEMA NFHL) when available, and heuristic fallbacks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catia.config import PERIL_CONFIG, SITE_VIABILITY_CONFIG
from catia.site_topo import fetch_site_terrain, topo_providers_enabled


def _coastal_hint(lat: float, lon: float) -> bool:
    """Coarse coastal proximity using longitude bands near major basins."""
    if 15.0 <= lat <= 45.0 and -100.0 <= lon <= -60.0:
        return abs(lon + 80) < 25 or abs(lat - 28) < 8
    if -10.0 <= lat <= 45.0 and 100.0 <= lon <= 150.0:
        return True
    if 30.0 <= lat <= 46.0 and -10.0 <= lon <= 40.0:
        return True
    return False


def _elevation_band(elev: float, coastal: bool) -> str:
    if elev < 50 or (coastal and elev < 80):
        return "lowland_coastal" if coastal or elev < 30 else "lowland_inland"
    if elev < 250:
        return "lowland_inland"
    if elev < 800:
        return "upland"
    return "highland"


def topography_stub(lat: float, lon: float, region_id: str) -> Dict[str, Any]:
    """Heuristic topography when DEM/floodplain providers are unavailable."""
    coastal = _coastal_hint(lat, lon)
    elev = max(0.0, abs(lat) * 12.0 + (0.0 if coastal else 180.0))
    if coastal:
        elev = min(elev, 80.0)
    if "Midwest" in region_id or "Southwest" in region_id:
        elev = max(elev, 200.0)
    if region_id in ("Chile", "Japan", "Turkey", "Indonesia"):
        elev = max(elev, 350.0)

    elev_band = _elevation_band(elev, coastal)
    if elev < 50:
        slope_class = "flat_to_gentle"
        floodplain_hint = "elevated_flood_sensitivity"
    elif elev < 250:
        slope_class = "gentle"
        floodplain_hint = "moderate_flood_sensitivity"
    elif elev < 800:
        slope_class = "moderate"
        floodplain_hint = "lower_flood_sensitivity"
    else:
        slope_class = "steep_or_varied"
        floodplain_hint = "terrain_driven_hazards"

    wildfire_terrain = elev_band in ("upland", "highland") or "West_Coast" in region_id
    seismic_terrain = region_id in ("US_West_Coast", "Japan", "Turkey", "Chile", "Indonesia")

    return {
        "elevation_m": round(elev, 1),
        "elevation_m_estimate": round(elev, 0),
        "elevation_band": elev_band,
        "slope_percent": None,
        "slope_class": slope_class,
        "coastal_proximity_hint": coastal,
        "floodplain_hint": floodplain_hint,
        "flood_zone": None,
        "sfha": None,
        "flood_risk_class": None,
        "wildfire_terrain_hint": wildfire_terrain,
        "seismic_terrain_hint": seismic_terrain,
        "data_quality": "heuristic_stub",
        "providers": [],
        "notes": (
            "Topography is a heuristic stub (DEM/floodplain providers unavailable or disabled). "
            "Enable CATIA_SITE_TOPO=1 for USGS / Open-Meteo / FEMA layers."
        ),
    }


def topography_from_providers(
    lat: float,
    lon: float,
    region_id: str,
    terrain: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge real DEM + FEMA floodplain into the topography payload."""
    coastal = _coastal_hint(lat, lon)
    elev_info = terrain.get("elevation") or {}
    slope_info = terrain.get("slope") or {}
    flood = terrain.get("floodplain") or {}

    elev = elev_info.get("elevation_m")
    if elev is None and slope_info.get("center_elevation_m") is not None:
        elev = slope_info["center_elevation_m"]
    if elev is None:
        # Provider pack without elevation — fall back partially
        stub = topography_stub(lat, lon, region_id)
        stub["floodplain_hint"] = flood.get("floodplain_hint") or stub["floodplain_hint"]
        stub["flood_zone"] = flood.get("fld_zone")
        stub["sfha"] = flood.get("sfha")
        stub["flood_risk_class"] = flood.get("flood_risk_class")
        stub["data_quality"] = "partial_providers"
        stub["providers"] = list(terrain.get("providers_used") or [])
        stub["floodplain_detail"] = flood or None
        return stub

    elev_f = float(elev)
    elev_band = _elevation_band(elev_f, coastal)
    slope_class = slope_info.get("slope_class") or "unknown"
    slope_pct = slope_info.get("slope_percent")

    floodplain_hint = flood.get("floodplain_hint")
    if not floodplain_hint:
        if elev_f < 50:
            floodplain_hint = "elevated_flood_sensitivity"
        elif elev_f < 250:
            floodplain_hint = "moderate_flood_sensitivity"
        else:
            floodplain_hint = "lower_flood_sensitivity"

    wildfire_terrain = (
        elev_band in ("upland", "highland")
        or "West_Coast" in region_id
        or (isinstance(slope_pct, (int, float)) and float(slope_pct) >= 15)
    )
    seismic_terrain = region_id in ("US_West_Coast", "Japan", "Turkey", "Chile", "Indonesia")

    providers = list(terrain.get("providers_used") or [])
    quality = "dem_floodplain" if flood.get("source") and elev_info.get("source") else (
        "dem_only" if elev_info.get("source") else "floodplain_only"
    )

    notes_parts = [
        f"Elevation from {elev_info.get('source', 'DEM')}",
    ]
    if slope_pct is not None:
        notes_parts.append(f"local slope ≈ {slope_pct}%")
    if flood.get("source"):
        z = flood.get("fld_zone") or "n/a"
        notes_parts.append(f"FEMA zone {z}" + (" (SFHA)" if flood.get("sfha") else ""))
    notes_parts.append("Screening layers — not a survey or LOMA determination.")

    return {
        "elevation_m": round(elev_f, 2),
        "elevation_m_estimate": round(elev_f, 0),
        "elevation_band": elev_band,
        "elevation_source": elev_info.get("source"),
        "elevation_resolution_m": elev_info.get("resolution_m"),
        "slope_percent": slope_pct,
        "slope_class": slope_class,
        "slope_samples": slope_info.get("samples"),
        "coastal_proximity_hint": coastal,
        "floodplain_hint": floodplain_hint,
        "flood_zone": flood.get("fld_zone"),
        "flood_zone_subtype": flood.get("zone_subtype"),
        "sfha": flood.get("sfha"),
        "flood_risk_class": flood.get("flood_risk_class"),
        "floodplain_detail": flood or None,
        "wildfire_terrain_hint": wildfire_terrain,
        "seismic_terrain_hint": seismic_terrain,
        "data_quality": quality,
        "providers": providers,
        "notes": "; ".join(notes_parts),
    }


def resolve_topography(lat: float, lon: float, region_id: str) -> Dict[str, Any]:
    """Prefer real DEM/floodplain; fall back to heuristic stub."""
    if topo_providers_enabled():
        terrain = fetch_site_terrain(lat, lon)
        if terrain.get("ok"):
            return topography_from_providers(lat, lon, region_id, terrain)
    return topography_stub(lat, lon, region_id)


def applicable_perils(region_id: str) -> List[str]:
    out: List[str] = []
    for pid, cfg in PERIL_CONFIG.items():
        regions = cfg.get("regions") or []
        if region_id in regions:
            out.append(pid)
    return out


def _flood_topo_boost(topo: Dict[str, Any]) -> float:
    """Stronger flood weights when SFHA / high FEMA zones are present."""
    risk = str(topo.get("flood_risk_class") or "")
    if topo.get("sfha") or risk == "sfha_high":
        return 1.45
    if risk == "moderate_0_2_pct":
        return 1.2
    hint = str(topo.get("floodplain_hint") or "")
    if hint.startswith("elevated"):
        return 1.25
    if hint.startswith("moderate"):
        return 1.1
    return 1.0


def build_hazard_assessment(
    *,
    lat: float,
    lon: float,
    region_id: str,
    property_type: str,
    exposure_overlap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-peril hazard rows + topography for a resolved site."""
    topo = resolve_topography(lat, lon, region_id)
    perils = applicable_perils(region_id)
    modifiers = SITE_VIABILITY_CONFIG["property_type_modifiers"].get(
        property_type, SITE_VIABILITY_CONFIG["property_type_modifiers"]["buy_building"]
    )
    tier = "medium"
    if exposure_overlap:
        hints = exposure_overlap.get("tier_hints") or []
        if hints:
            order = {"low": 0, "medium": 1, "high": 2}
            tier = max(hints, key=lambda t: order.get(str(t).lower(), 1))

    hazards: List[Dict[str, Any]] = []
    for pid in perils or list(PERIL_CONFIG.keys())[:1]:
        cfg = PERIL_CONFIG.get(pid) or {}
        freq = float(cfg.get("frequency_base") or 0.3)
        sev = cfg.get("severity_params") or {}
        mu = float(sev.get("mu") or 14)
        sev_rel = max(0.15, min(1.0, (mu - 11.0) / 6.0))
        weight = float(modifiers.get(pid, 1.0))
        topo_boost = 1.0
        if pid == "flood":
            topo_boost = _flood_topo_boost(topo)
        elif pid == "wildfire" and topo.get("wildfire_terrain_hint"):
            topo_boost = 1.15
            slope_pct = topo.get("slope_percent")
            if isinstance(slope_pct, (int, float)) and float(slope_pct) >= 15:
                topo_boost = 1.25
        elif pid == "earthquake" and topo.get("seismic_terrain_hint"):
            topo_boost = 1.2
        elif pid == "hurricane" and topo.get("coastal_proximity_hint"):
            topo_boost = 1.2
            if float(topo.get("elevation_m") or 999) < 10:
                topo_boost = 1.3

        notes = str(cfg.get("description") or "")
        if pid == "flood" and topo.get("flood_zone"):
            notes = f"FEMA {topo.get('flood_zone')}" + (
                " SFHA" if topo.get("sfha") else ""
            ) + f" · {notes}"

        score = 100.0 * min(
            1.0, (0.45 * min(1.0, freq / 1.5) + 0.35 * sev_rel) * weight * topo_boost
        )
        hazards.append(
            {
                "peril": pid,
                "name": cfg.get("name", pid),
                "applicable": pid in perils,
                "frequency_base": freq,
                "relative_severity": round(sev_rel, 3),
                "property_modifier": weight,
                "topography_factor": round(topo_boost, 3),
                "hazard_score": round(score, 1),
                "notes": notes,
            }
        )

    hazards.sort(key=lambda h: float(h["hazard_score"]), reverse=True)
    return {
        "topography": topo,
        "exposure_tier_hint": str(tier).lower(),
        "hazards": hazards,
        "applicable_peril_ids": perils,
    }
