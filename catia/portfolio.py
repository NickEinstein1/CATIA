"""
Portfolio ingest and location resolution for accumulation analysis.

Accepts CSV or GeoJSON of many sites (lat/lon/TIV), maps each to the nearest
CATIA region, and optionally attaches FEMA flood zones.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from catia.site_geo import nearest_region
from catia.site_hazards import applicable_perils

logger = logging.getLogger(__name__)

# Hard caps to bound CPU / outbound FEMA amplification on public API hosts.
MAX_PORTFOLIO_LOCATIONS = 2000
MAX_FLOOD_ZONE_LEN = 32
MAX_FEMA_LOOKUPS = 100

DISCLAIMER = (
    "Indicative portfolio accumulation for research and actuarial analytics. "
    "Not binding underwriting, treaty pricing, or regulatory capital. "
    "Regional intensity assumptions apply — not site-specific hazard footprints."
)


@dataclass
class PortfolioLocation:
    id: str
    lat: float
    lon: float
    tiv: float
    construction_type: Optional[str] = None
    occupancy: Optional[str] = None
    property_type: Optional[str] = None
    region_id: str = ""
    region_label: str = ""
    region_distance_km: float = 0.0
    flood_zone: Optional[str] = None
    sfha: Optional[bool] = None
    flood_risk_class: Optional[str] = None
    applicable_perils: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _alias_map(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common column aliases to canonical keys."""
    lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    out: Dict[str, Any] = {}

    def pick(*names: str) -> Any:
        for n in names:
            if n in lower and lower[n] not in (None, ""):
                return lower[n]
        return None

    out["lat"] = pick("lat", "latitude", "y")
    out["lon"] = pick("lon", "longitude", "lng", "long", "x")
    out["tiv"] = pick("tiv", "value", "sum_insured", "suminsured", "insured_value", "exposure")
    out["id"] = pick("id", "location_id", "loc_id", "name", "policy_id")
    out["construction_type"] = pick("construction_type", "construction", "const")
    out["occupancy"] = pick("occupancy", "occ", "occupancy_type")
    out["property_type"] = pick("property_type", "intent", "occupancy_class")
    out["flood_zone"] = pick("flood_zone", "fld_zone", "fema_zone", "zone")
    out["region"] = pick("region", "region_id", "catia_region")
    return out


def _coerce_float(v: Any, field: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid {field}: {v!r}") from e


def validate_raw_location(row: Dict[str, Any], *, index: int = 0) -> Dict[str, Any]:
    n = _alias_map(row)
    if n["lat"] is None or n["lon"] is None:
        raise ValueError(f"Row {index}: lat and lon are required")
    if n["tiv"] is None:
        raise ValueError(f"Row {index}: tiv (or value/sum_insured) is required")
    lat = _coerce_float(n["lat"], "lat")
    lon = _coerce_float(n["lon"], "lon")
    tiv = _coerce_float(n["tiv"], "tiv")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"Row {index}: lat/lon out of range")
    if tiv <= 0:
        raise ValueError(f"Row {index}: tiv must be > 0")
    loc_id = str(n["id"] or f"loc-{index + 1}")
    return {
        "id": loc_id,
        "lat": lat,
        "lon": lon,
        "tiv": tiv,
        "construction_type": str(n["construction_type"]) if n["construction_type"] else None,
        "occupancy": str(n["occupancy"]) if n["occupancy"] else None,
        "property_type": str(n["property_type"]) if n["property_type"] else None,
        "flood_zone": (
            str(n["flood_zone"]).upper()[:MAX_FLOOD_ZONE_LEN] if n["flood_zone"] else None
        ),
        "region": str(n["region"]) if n["region"] else None,
    }


def parse_portfolio_csv(text: str) -> List[Dict[str, Any]]:
    """Parse CSV text into raw location dicts (not yet region-resolved)."""
    if len(text) > 2_000_000:
        raise ValueError("CSV text exceeds 2MB limit")
    f = io.StringIO(text.strip())
    reader = csv.DictReader(f)
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(reader):
        if not any(str(v or "").strip() for v in row.values()):
            continue
        rows.append(validate_raw_location(row, index=i))
    if not rows:
        raise ValueError("CSV contained no location rows")
    if len(rows) > MAX_PORTFOLIO_LOCATIONS:
        raise ValueError(
            f"CSV exceeds max locations ({MAX_PORTFOLIO_LOCATIONS}); got {len(rows)}"
        )
    return rows


def parse_portfolio_geojson(data: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse GeoJSON FeatureCollection of Point features."""
    payload = json.loads(data) if isinstance(data, str) else data
    if not isinstance(payload, dict):
        raise ValueError("GeoJSON must be an object")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection requires features[]")
    rows: List[Dict[str, Any]] = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        props = dict(feat.get("properties") or {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates") or []
            if len(coords) >= 2:
                props.setdefault("lon", coords[0])
                props.setdefault("lat", coords[1])
        if feat.get("id") is not None and "id" not in props:
            props["id"] = feat["id"]
        rows.append(validate_raw_location(props, index=i))
    if not rows:
        raise ValueError("GeoJSON contained no Point features with tiv")
    if len(rows) > MAX_PORTFOLIO_LOCATIONS:
        raise ValueError(
            f"GeoJSON exceeds max locations ({MAX_PORTFOLIO_LOCATIONS}); got {len(rows)}"
        )
    return rows


def parse_portfolio_payload(
    *,
    locations: Optional[List[Dict[str, Any]]] = None,
    csv_text: Optional[str] = None,
    geojson: Optional[Union[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if locations:
        if len(locations) > MAX_PORTFOLIO_LOCATIONS:
            raise ValueError(
                f"locations[] exceeds max ({MAX_PORTFOLIO_LOCATIONS}); got {len(locations)}"
            )
        return [validate_raw_location(r, index=i) for i, r in enumerate(locations)]
    if csv_text and csv_text.strip():
        return parse_portfolio_csv(csv_text)
    if geojson is not None:
        return parse_portfolio_geojson(geojson)
    raise ValueError("Provide locations[], csv_text, or geojson")


def resolve_portfolio_locations(
    raw_rows: List[Dict[str, Any]],
    *,
    include_fema: bool = False,
) -> List[PortfolioLocation]:
    """Map each raw location to nearest CATIA region (+ optional FEMA zone)."""
    from catia.site_topo import fetch_fema_flood_zone

    out: List[PortfolioLocation] = []
    fema_calls = 0
    for row in raw_rows:
        lat, lon = float(row["lat"]), float(row["lon"])
        if row.get("region"):
            region_id = str(row["region"])
            nr = nearest_region(lat, lon)
            # Prefer explicit region; still compute distance to its centroid if known
            from catia.geo_regions import REGION_CENTROIDS

            if region_id in REGION_CENTROIDS:
                clat, clon = REGION_CENTROIDS[region_id]
                from catia.site_geo import haversine_km

                dist = haversine_km(lat, lon, clat, clon)
                label = region_id.replace("_", " ")
            else:
                dist = float(nr["distance_km"])
                label = region_id.replace("_", " ")
                # Fall back to nearest if unknown region string
                if region_id not in REGION_CENTROIDS:
                    region_id = nr["region_id"]
                    label = nr["region_label"]
                    dist = float(nr["distance_km"])
        else:
            nr = nearest_region(lat, lon)
            region_id = nr["region_id"]
            label = nr["region_label"]
            dist = float(nr["distance_km"])

        flood_zone = row.get("flood_zone")
        sfha = None
        flood_risk = None
        if flood_zone:
            from catia.site_topo import _classify_fema_zone

            classified = _classify_fema_zone(str(flood_zone), "", "")
            sfha = classified["sfha"]
            flood_risk = classified["flood_risk_class"]
        elif include_fema and fema_calls < MAX_FEMA_LOOKUPS:
            fema_calls += 1
            try:
                fema = fetch_fema_flood_zone(lat, lon)
                if fema:
                    flood_zone = fema.get("fld_zone")
                    sfha = fema.get("sfha")
                    flood_risk = fema.get("flood_risk_class")
            except Exception as e:
                logger.debug("FEMA skip for %s: %s", row.get("id"), e)

        out.append(
            PortfolioLocation(
                id=str(row["id"]),
                lat=lat,
                lon=lon,
                tiv=float(row["tiv"]),
                construction_type=row.get("construction_type"),
                occupancy=row.get("occupancy"),
                property_type=row.get("property_type"),
                region_id=region_id,
                region_label=label,
                region_distance_km=round(dist, 1),
                flood_zone=flood_zone,
                sfha=sfha,
                flood_risk_class=flood_risk,
                applicable_perils=applicable_perils(region_id),
            )
        )
    return out


def portfolio_summary(locations: List[PortfolioLocation]) -> Dict[str, Any]:
    regions: Dict[str, int] = {}
    zones: Dict[str, int] = {}
    total_tiv = 0.0
    for loc in locations:
        total_tiv += loc.tiv
        regions[loc.region_id] = regions.get(loc.region_id, 0) + 1
        z = loc.flood_zone or "unknown"
        zones[z] = zones.get(z, 0) + 1
    return {
        "location_count": len(locations),
        "total_tiv": round(total_tiv, 2),
        "region_counts": regions,
        "flood_zone_counts": zones,
        "region_count": len(regions),
    }
