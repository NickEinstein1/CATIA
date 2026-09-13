"""
Portfolio accumulation: AAL/VaR rollups and one-storm book shock.

Uses ExposureStore + run_exposure_based_simulation for Monte Carlo book metrics,
sliced stores for region / flood-zone tables, and VulnerabilitySet.damage_ratio
for a deterministic “one storm hits the book” scenario.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from catia.config import DEFAULT_PERILS, SITE_VIABILITY_CONFIG
from catia.exposure import ExposureStore
from catia.financial_impact import (
    FinancialImpactSimulator,
    MultiPerilSimulator,
    run_exposure_based_simulation,
)
from catia.hazard_footprints import (
    extract_polygon,
    extract_track_coords,
    footprint_summary,
    local_intensity_for_point,
    resolve_footprint_kind,
)
from catia.live_portfolio import live_event_to_one_storm
from catia.portfolio import (
    DISCLAIMER,
    PortfolioLocation,
    parse_portfolio_payload,
    portfolio_summary,
    resolve_portfolio_locations,
)
from catia.reinsurance import apply_layers_to_loss, apply_layers_to_losses, parse_layers
from catia.site_geo import haversine_km
from catia.vulnerability import VulnerabilitySet

GroupBy = Literal["region", "flood_zone"]

# Bound O(N) Monte Carlo slice fan-out when attacker sets unique flood_zone per row.
MAX_SLICE_BUCKETS = 50
MAX_LOCATIONS_IN_RESPONSE = 500
MAX_CSV_CHARS = 2_000_000
MAX_NUM_ITERATIONS = 5000


def _f(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _metrics_brief(metrics: Dict[str, Any]) -> Dict[str, Any]:
    desc = (metrics or {}).get("descriptive_stats") or {}
    risk = (metrics or {}).get("risk_metrics") or {}
    return {
        "mean": _f(desc.get("mean")),
        "median": _f(desc.get("median")),
        "std": _f(desc.get("std")),
        "var_95": _f(risk.get("var")),
        "tvar_95": _f(risk.get("tvar")),
        "return_periods": {
            str(k): _f(v) for k, v in ((metrics or {}).get("return_periods") or {}).items()
        },
    }


def _locations_to_store(locations: List[PortfolioLocation]) -> ExposureStore:
    store = ExposureStore()
    for loc in locations:
        store.add_record(
            region=loc.region_id,
            tiv=loc.tiv,
            line_of_business=loc.property_type,
            construction_type=loc.construction_type,
            occupancy=loc.occupancy,
        )
    return store


def _run_book_simulation(
    locations: List[PortfolioLocation],
    *,
    perils: List[str],
    num_iterations: int,
    scenario_id: Optional[str] = None,
    reinsurance_layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not locations:
        return {"aggregate": {}, "by_peril": {}, "contributions": []}
    store = _locations_to_store(locations)
    vuln = VulnerabilitySet()
    results = run_exposure_based_simulation(
        store,
        vuln,
        perils=perils,
        num_iterations=num_iterations,
        scenario_id=scenario_id,
    )
    contrib_df = MultiPerilSimulator(perils, use_correlation=False).get_peril_contribution(results)
    contributions = []
    if contrib_df is not None and hasattr(contrib_df, "to_dict"):
        contributions = contrib_df.to_dict(orient="records")
    by_peril = {}
    for p, blob in (results.get("by_peril") or {}).items():
        by_peril[p] = {
            "name": blob.get("name", p),
            **_metrics_brief(blob.get("metrics") or {}),
        }
    out: Dict[str, Any] = {
        "aggregate": _metrics_brief((results.get("aggregate") or {}).get("metrics") or {}),
        "by_peril": by_peril,
        "contributions": contributions,
        "iterations": num_iterations,
        "scenario_id": scenario_id or "baseline",
    }
    layers = parse_layers(reinsurance_layers)
    if layers:
        gross = (results.get("aggregate") or {}).get("losses")
        if gross is not None:
            applied = apply_layers_to_losses(gross, layers)
            metric_sim = FinancialImpactSimulator(1.0, {"mu": 15, "sigma": 2})
            out["aggregate_net"] = _metrics_brief(
                metric_sim.calculate_aggregate_metrics(applied["net_losses"])
            )
            out["reinsurance"] = {
                "mean_gross": applied["mean_gross"],
                "mean_recovered": applied["mean_recovered"],
                "mean_net": applied["mean_net"],
                "layers": applied["layer_means"],
            }
    return out


def _group_key(loc: PortfolioLocation, group_by: GroupBy) -> str:
    if group_by == "flood_zone":
        return str(loc.flood_zone or "unknown")
    return loc.region_id or "unknown"


def _run_sliced(
    locations: List[PortfolioLocation],
    *,
    group_by: GroupBy,
    perils: List[str],
    num_iterations: int,
    scenario_id: Optional[str] = None,
    max_buckets: int = MAX_SLICE_BUCKETS,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[PortfolioLocation]] = {}
    for loc in locations:
        buckets.setdefault(_group_key(loc, group_by), []).append(loc)
    ranked = sorted(buckets.items(), key=lambda kv: -sum(l.tiv for l in kv[1]))
    if len(ranked) > max_buckets:
        head = ranked[: max_buckets - 1]
        rest_locs: List[PortfolioLocation] = []
        for _, locs in ranked[max_buckets - 1 :]:
            rest_locs.extend(locs)
        ranked = head + [("other", rest_locs)]
    rows: List[Dict[str, Any]] = []
    for key, locs in ranked:
        sim = _run_book_simulation(
            locs,
            perils=perils,
            num_iterations=num_iterations,
            scenario_id=scenario_id,
        )
        rows.append(
            {
                "group": key,
                "group_by": group_by,
                "location_count": len(locs),
                "total_tiv": round(sum(l.tiv for l in locs), 2),
                "aggregate": sim["aggregate"],
                "by_peril": sim["by_peril"],
            }
        )
    return rows


def one_storm_scenario(
    locations: List[PortfolioLocation],
    *,
    peril: str,
    intensity: float,
    mode: Literal["region", "radius"] = "region",
    region_id: Optional[str] = None,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    radius_km: float = 250.0,
    footprint: str = "auto",
    geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic shock with optional intensity footprint decay.

    Modes:
    - region: all locations in a CATIA region (uniform peak unless footprint decays
      from region centroid — still requires center when footprint != uniform)
    - radius: locations within radius_km of center / track / polygon

    Footprints (``auto`` picks by peril): uniform, windfield, shake, flood_bowl, wildfire.
    Optional GeoJSON ``geometry`` (LineString track or Polygon) from live feeds.
    """
    vuln = VulnerabilitySet()
    peak = float(intensity)
    kind = resolve_footprint_kind(peril, footprint)
    track = extract_track_coords(geometry)
    polygon = extract_polygon(geometry)

    # Region mode without coords: use uniform on whole region (legacy behavior)
    if mode == "region":
        if not region_id:
            raise ValueError("region_id is required when mode='region'")
        candidates = [loc for loc in locations if loc.region_id == region_id]
        if kind == "uniform" or (center_lat is None or center_lon is None):
            damage_ratio = max(0.0, min(1.0, float(vuln.damage_ratio(peril, peak))))
            by_location: List[Dict[str, Any]] = []
            by_region: Dict[str, Dict[str, float]] = {}
            total_loss = 0.0
            tiv_hit = 0.0
            for loc in candidates:
                loss = loc.tiv * damage_ratio
                total_loss += loss
                tiv_hit += loc.tiv
                by_location.append(
                    {
                        "id": loc.id,
                        "lat": loc.lat,
                        "lon": loc.lon,
                        "region_id": loc.region_id,
                        "flood_zone": loc.flood_zone,
                        "tiv": loc.tiv,
                        "intensity": peak,
                        "damage_ratio": round(damage_ratio, 4),
                        "loss": round(loss, 2),
                    }
                )
                bucket = by_region.setdefault(
                    loc.region_id, {"tiv": 0.0, "loss": 0.0, "location_count": 0}
                )
                bucket["tiv"] += loc.tiv
                bucket["loss"] += loss
                bucket["location_count"] += 1
            by_location.sort(key=lambda r: float(r["loss"]), reverse=True)
            for b in by_region.values():
                b["tiv"] = round(b["tiv"], 2)
                b["loss"] = round(b["loss"], 2)
            return {
                "peril": peril,
                "intensity": peak,
                "damage_ratio": round(damage_ratio, 4),
                "mode": mode,
                "region_id": region_id,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_km": None,
                "footprint": footprint_summary(
                    kind="uniform",
                    peril=peril,
                    peak_intensity=peak,
                    radius_km=0.0,
                    has_track=False,
                    has_polygon=False,
                ),
                "hit_count": len(candidates),
                "tiv_hit": round(tiv_hit, 2),
                "total_loss": round(total_loss, 2),
                "loss_ratio_of_book": round(
                    total_loss / sum(l.tiv for l in locations) if locations else 0.0, 4
                ),
                "by_region": by_region,
                "top_locations": by_location[:25],
                "note": (
                    "Region-wide uniform intensity — enable radius + footprint for "
                    "distance decay."
                ),
            }
        # Fall through: treat region candidates with radial footprint from center
        hit_pool = candidates
        use_radius_gate = False
        if candidates and center_lat is not None and center_lon is not None:
            max_d = max(
                haversine_km(loc.lat, loc.lon, float(center_lat), float(center_lon))
                for loc in candidates
            )
            r_eff = max(float(radius_km), max_d * 1.05, 1.0)
        else:
            r_eff = float(radius_km)
    else:
        if center_lat is None or center_lon is None:
            raise ValueError("center_lat and center_lon are required when mode='radius'")
        hit_pool = locations
        use_radius_gate = True
        r_eff = float(radius_km)

    r = r_eff
    by_location = []
    by_region = {}
    total_loss = 0.0
    tiv_hit = 0.0
    hit_count = 0
    for loc in hit_pool:
        field = local_intensity_for_point(
            loc.lat,
            loc.lon,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            peak_intensity=peak,
            radius_km=r,
            footprint=kind,
            track_coords=track,
            polygon=polygon,
        )
        if use_radius_gate and not field["inside"]:
            continue
        local_i = float(field["intensity"])
        if local_i <= 0:
            continue
        dr = max(0.0, min(1.0, float(vuln.damage_ratio(peril, local_i))))
        loss = loc.tiv * dr
        total_loss += loss
        tiv_hit += loc.tiv
        hit_count += 1
        by_location.append(
            {
                "id": loc.id,
                "lat": loc.lat,
                "lon": loc.lon,
                "region_id": loc.region_id,
                "flood_zone": loc.flood_zone,
                "tiv": loc.tiv,
                "distance_km": field["distance_km"],
                "intensity": local_i,
                "damage_ratio": round(dr, 4),
                "loss": round(loss, 2),
            }
        )
        bucket = by_region.setdefault(
            loc.region_id, {"tiv": 0.0, "loss": 0.0, "location_count": 0}
        )
        bucket["tiv"] += loc.tiv
        bucket["loss"] += loss
        bucket["location_count"] += 1

    by_location.sort(key=lambda row: float(row["loss"]), reverse=True)
    for b in by_region.values():
        b["tiv"] = round(b["tiv"], 2)
        b["loss"] = round(b["loss"], 2)

    peak_dr = max(0.0, min(1.0, float(vuln.damage_ratio(peril, peak))))
    return {
        "peril": peril,
        "intensity": peak,
        "damage_ratio": round(peak_dr, 4),
        "mode": mode,
        "region_id": region_id,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": float(radius_km) if mode == "radius" else round(r, 2),
        "footprint": footprint_summary(
            kind=kind,
            peril=peril,
            peak_intensity=peak,
            radius_km=r,
            has_track=bool(track),
            has_polygon=bool(polygon),
        ),
        "hit_count": hit_count,
        "tiv_hit": round(tiv_hit, 2),
        "total_loss": round(total_loss, 2),
        "loss_ratio_of_book": round(
            total_loss / sum(l.tiv for l in locations) if locations else 0.0, 4
        ),
        "by_region": by_region,
        "top_locations": by_location[:25],
        "note": (
            f"Deterministic {kind} footprint on filtered locations — "
            "not a probabilistic catalog match or official hazard product."
        ),
    }


def analyze_portfolio(
    *,
    locations: Optional[List[Dict[str, Any]]] = None,
    csv_text: Optional[str] = None,
    geojson: Optional[Any] = None,
    perils: Optional[List[str]] = None,
    num_iterations: Optional[int] = None,
    scenario_id: Optional[str] = None,
    include_fema: bool = False,
    group_by: Optional[List[str]] = None,
    run_simulation: bool = True,
    one_storm: Optional[Dict[str, Any]] = None,
    live_event: Optional[Dict[str, Any]] = None,
    reinsurance_layers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    End-to-end portfolio accumulation payload for API and dashboard.

    Optional ``live_event`` maps a feed event onto a radius one-storm shock.
    Optional ``reinsurance_layers`` produce net-of-XL book metrics and storm recoveries.
    """
    raw = parse_portfolio_payload(locations=locations, csv_text=csv_text, geojson=geojson)
    resolved = resolve_portfolio_locations(raw, include_fema=include_fema)
    summary = portfolio_summary(resolved)

    peril_list = list(perils or DEFAULT_PERILS)
    n_iter = int(
        num_iterations
        or SITE_VIABILITY_CONFIG.get("indicative_iterations", 2000)
    )
    n_iter = max(100, min(n_iter, MAX_NUM_ITERATIONS))
    # Cap iterations for large books in interactive UI
    if len(resolved) > 200:
        n_iter = min(n_iter, 1500)

    layers = parse_layers(reinsurance_layers)
    layer_dicts = [L.to_dict() for L in layers]

    storm_spec = one_storm
    if live_event and not storm_spec:
        storm_spec = live_event_to_one_storm(live_event)
    elif live_event and storm_spec:
        # Fill missing radius fields from live event
        mapped = live_event_to_one_storm(live_event)
        merged = {**mapped, **{k: v for k, v in storm_spec.items() if v is not None}}
        storm_spec = merged

    groups = group_by or ["region", "flood_zone"]
    book_sim = None
    by_region = None
    by_flood_zone = None
    if run_simulation and resolved:
        book_sim = _run_book_simulation(
            resolved,
            perils=peril_list,
            num_iterations=n_iter,
            scenario_id=scenario_id,
            reinsurance_layers=layer_dicts or None,
        )
        if "region" in groups:
            by_region = _run_sliced(
                resolved,
                group_by="region",
                perils=peril_list,
                num_iterations=min(n_iter, 1500),
                scenario_id=scenario_id,
            )
        if "flood_zone" in groups:
            by_flood_zone = _run_sliced(
                resolved,
                group_by="flood_zone",
                perils=peril_list,
                num_iterations=min(n_iter, 1500),
                scenario_id=scenario_id,
            )

    storm = None
    if storm_spec:
        storm = one_storm_scenario(
            resolved,
            peril=str(storm_spec.get("peril") or "hurricane"),
            intensity=float(storm_spec.get("intensity") or 120),
            mode=str(storm_spec.get("mode") or "region"),  # type: ignore[arg-type]
            region_id=storm_spec.get("region_id"),
            center_lat=storm_spec.get("center_lat"),
            center_lon=storm_spec.get("center_lon"),
            radius_km=float(storm_spec.get("radius_km") or 250),
            footprint=str(storm_spec.get("footprint") or "auto"),
            geometry=storm_spec.get("geometry"),
        )
        for key in (
            "live_event_id",
            "live_event_title",
            "live_event_source",
            "intensity_source",
        ):
            if storm_spec.get(key) is not None:
                storm[key] = storm_spec.get(key)
        if storm_spec.get("note"):
            storm["note"] = storm_spec["note"]
        if layers:
            storm["reinsurance"] = apply_layers_to_loss(float(storm["total_loss"]), layers)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    loc_dicts = [loc.to_dict() for loc in resolved]
    truncated = False
    if len(loc_dicts) > MAX_LOCATIONS_IN_RESPONSE:
        loc_dicts = loc_dicts[:MAX_LOCATIONS_IN_RESPONSE]
        truncated = True
    return {
        "analyzed_at": now,
        "summary": summary,
        "locations": loc_dicts,
        "locations_truncated": truncated,
        "perils": peril_list,
        "book": book_sim,
        "by_region": by_region,
        "by_flood_zone": by_flood_zone,
        "one_storm": storm,
        "reinsurance_layers": layer_dicts,
        "include_fema": include_fema,
        "disclaimer": DISCLAIMER,
    }
