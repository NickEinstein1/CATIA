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
from catia.financial_impact import MultiPerilSimulator, run_exposure_based_simulation
from catia.portfolio import (
    DISCLAIMER,
    PortfolioLocation,
    parse_portfolio_payload,
    portfolio_summary,
    resolve_portfolio_locations,
)
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
    return {
        "aggregate": _metrics_brief((results.get("aggregate") or {}).get("metrics") or {}),
        "by_peril": by_peril,
        "contributions": contributions,
        "iterations": num_iterations,
        "scenario_id": scenario_id or "baseline",
    }


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
) -> Dict[str, Any]:
    """
    Deterministic shock: apply one peril intensity to all locations in a region
    or within a radius — “what if this one storm hits the book.”
    """
    vuln = VulnerabilitySet()
    damage_ratio = float(vuln.damage_ratio(peril, float(intensity)))
    damage_ratio = max(0.0, min(1.0, damage_ratio))

    hit: List[PortfolioLocation] = []
    if mode == "region":
        if not region_id:
            raise ValueError("region_id is required when mode='region'")
        hit = [loc for loc in locations if loc.region_id == region_id]
    else:
        if center_lat is None or center_lon is None:
            raise ValueError("center_lat and center_lon are required when mode='radius'")
        r = float(radius_km)
        hit = [
            loc
            for loc in locations
            if haversine_km(float(center_lat), float(center_lon), loc.lat, loc.lon) <= r
        ]

    by_location: List[Dict[str, Any]] = []
    by_region: Dict[str, Dict[str, float]] = {}
    total_loss = 0.0
    tiv_hit = 0.0
    for loc in hit:
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
        "intensity": float(intensity),
        "damage_ratio": round(damage_ratio, 4),
        "mode": mode,
        "region_id": region_id,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": float(radius_km) if mode == "radius" else None,
        "hit_count": len(hit),
        "tiv_hit": round(tiv_hit, 2),
        "total_loss": round(total_loss, 2),
        "loss_ratio_of_book": round(
            total_loss / sum(l.tiv for l in locations) if locations else 0.0, 4
        ),
        "by_region": by_region,
        "top_locations": by_location[:25],
        "note": (
            "Deterministic single-intensity shock on filtered locations — "
            "not a probabilistic footprint or event catalog match."
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
) -> Dict[str, Any]:
    """
    End-to-end portfolio accumulation payload for API and dashboard.
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
            # Only meaningful when zones known; still run for unknown bucket
            by_flood_zone = _run_sliced(
                resolved,
                group_by="flood_zone",
                perils=peril_list,
                num_iterations=min(n_iter, 1500),
                scenario_id=scenario_id,
            )

    storm = None
    if one_storm:
        storm = one_storm_scenario(
            resolved,
            peril=str(one_storm.get("peril") or "hurricane"),
            intensity=float(one_storm.get("intensity") or 120),
            mode=str(one_storm.get("mode") or "region"),  # type: ignore[arg-type]
            region_id=one_storm.get("region_id"),
            center_lat=one_storm.get("center_lat"),
            center_lon=one_storm.get("center_lon"),
            radius_km=float(one_storm.get("radius_km") or 250),
        )

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
        "include_fema": include_fema,
        "disclaimer": DISCLAIMER,
    }
