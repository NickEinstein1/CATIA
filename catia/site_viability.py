"""
Site viability assessment for buy-land / build / buy-building decisions.

Combines nearest CATIA region, heuristic topography, peril applicability, and
optional indicative exposure simulation into an insurance-oriented screening report.

Indicative / research use only — not binding underwriting or reinsurance advice.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from catia.config import SITE_VIABILITY_CONFIG
from catia.site_geo import resolve_site_location
from catia.site_hazards import build_hazard_assessment

DISCLAIMER = (
    "Indicative CATIA site screening for research and actuarial analytics. "
    "DEM elevation (USGS EPQS / Open-Meteo) and FEMA NFHL flood zones are screening layers — "
    "not a survey, LOMA, flood certificate, binding quote, or reinsurance treaty advice. "
    "Validate with licensed hazard data and underwriting judgment before decisions."
)


def _score_to_category(score: float) -> str:
    thresholds = SITE_VIABILITY_CONFIG["score_thresholds"]
    if score >= float(thresholds["severe"]):
        return "severe"
    if score >= float(thresholds["high"]):
        return "high"
    if score >= float(thresholds["moderate"]):
        return "moderate"
    return "low"


def _viability_from_category(category: str, property_type: str, top_hazards: List[str]) -> Dict[str, Any]:
    table = SITE_VIABILITY_CONFIG["insurance_viability"]
    row = dict(table.get(category) or table["moderate"])
    guidance = list(SITE_VIABILITY_CONFIG["property_guidance"].get(property_type) or [])
    peril_notes = []
    for p in top_hazards[:3]:
        peril_notes.append(SITE_VIABILITY_CONFIG["peril_guidance"].get(p, f"Review {p} exposure carefully."))
    return {
        "status": row["status"],
        "label": row["label"],
        "narrative": row["narrative"],
        "reinsurance_notes": row["reinsurance_notes"],
        "property_guidance": guidance,
        "peril_guidance": peril_notes,
        "suggested_actions": row.get("suggested_actions") or [],
    }


def _composite_score(hazards: List[Dict[str, Any]], exposure_tier: str) -> Dict[str, Any]:
    if not hazards:
        return {"score": 35.0, "components": {"hazard": 0.35, "concentration": 0.0, "exposure_tier": 0.5}}
    top = sorted(hazards, key=lambda h: float(h["hazard_score"]), reverse=True)
    # Concentration: more high-scoring perils → higher portfolio risk
    scores = [float(h["hazard_score"]) for h in top]
    primary = scores[0] / 100.0
    secondary = (sum(scores[1:3]) / max(1, len(scores[1:3])) / 100.0) if len(scores) > 1 else 0.0
    conc = min(1.0, 0.35 * len([s for s in scores if s >= 55]) / 3.0 + 0.2 * secondary)
    tier_map = {"low": 0.35, "medium": 0.55, "high": 0.8}
    tier_c = tier_map.get(str(exposure_tier).lower(), 0.55)
    score = 100.0 * (0.55 * primary + 0.25 * conc + 0.20 * tier_c)
    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "components": {
            "primary_hazard": round(primary, 3),
            "multi_peril_concentration": round(conc, 3),
            "exposure_tier": round(tier_c, 3),
        },
    }


def _optional_simulation(
    *,
    region_id: str,
    tiv: float,
    perils: List[str],
    property_type: str,
    construction_type: Optional[str],
    occupancy: Optional[str],
) -> Optional[Dict[str, Any]]:
    if tiv <= 0 or not perils:
        return None
    try:
        from catia.exposure import ExposureStore
        from catia.financial_impact import run_exposure_based_simulation
        from catia.vulnerability import VulnerabilitySet

        store = ExposureStore()
        store.add_record(
            region=region_id,
            tiv=float(tiv),
            line_of_business=property_type,
            construction_type=construction_type,
            occupancy=occupancy,
        )
        n = int(SITE_VIABILITY_CONFIG.get("indicative_iterations", 2000))
        result = run_exposure_based_simulation(
            store,
            VulnerabilitySet(),
            perils=perils,
            num_iterations=n,
        )
        agg = (result or {}).get("aggregate") or {}
        metrics = agg.get("metrics") or {}
        desc = metrics.get("descriptive_stats") or {}
        risk = metrics.get("risk_metrics") or {}
        def _f(x: Any) -> Optional[float]:
            try:
                return float(x) if x is not None else None
            except (TypeError, ValueError):
                return None

        rps = metrics.get("return_periods") or {}
        return {
            "iterations": n,
            "perils": perils,
            "tiv": float(tiv),
            "mean_annual_loss": _f(desc.get("mean")),
            "var_95": _f(risk.get("var")),
            "tvar_95": _f(risk.get("tvar")),
            "return_periods": {str(k): _f(v) for k, v in rps.items()},
            "note": "Indicative Monte Carlo on regional intensity assumptions — not site-specific hazard.",
        }
    except Exception as e:
        return {"error": str(e), "note": "Indicative simulation unavailable"}


def assess_site_viability(
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    address: Optional[str] = None,
    property_type: str = "buy_building",
    construction_type: Optional[str] = None,
    occupancy: Optional[str] = None,
    tiv: Optional[float] = None,
    include_simulation: bool = False,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full site viability payload for API, dashboard, and agents.

    ``property_type``: ``buy_land`` | ``build`` | ``buy_building``.
    """
    pt = (property_type or "buy_building").strip().lower()
    if pt not in SITE_VIABILITY_CONFIG["property_type_modifiers"]:
        pt = "buy_building"

    location = resolve_site_location(lat=lat, lon=lon, address=address)
    region_id = location["nearest_region"]["region_id"]
    hazard_pack = build_hazard_assessment(
        lat=location["lat"],
        lon=location["lon"],
        region_id=region_id,
        property_type=pt,
        exposure_overlap=location.get("exposure_overlap"),
    )
    composite = _composite_score(hazard_pack["hazards"], hazard_pack["exposure_tier_hint"])
    category = _score_to_category(composite["score"])
    top_perils = [h["peril"] for h in hazard_pack["hazards"][:4]]
    insurance = _viability_from_category(category, pt, top_perils)

    default_tiv = float(
        SITE_VIABILITY_CONFIG["default_tiv"].get(pt, 500_000)
    )
    sim_tiv = float(tiv) if tiv is not None and float(tiv) > 0 else default_tiv
    sim: Optional[Dict[str, Any]] = None
    if include_simulation:
        sim = _optional_simulation(
            region_id=region_id,
            tiv=sim_tiv,
            perils=hazard_pack["applicable_peril_ids"] or top_perils[:2],
            property_type=pt,
            construction_type=construction_type,
            occupancy=occupancy,
        )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return {
        "assessed_at": now,
        "property_type": pt,
        "construction_type": construction_type,
        "occupancy": occupancy,
        "scenario_id": scenario_id or "baseline",
        "location": location,
        "risk_score": composite["score"],
        "risk_category": category,
        "score_components": composite["components"],
        "hazard_assessment": hazard_pack["hazards"],
        "topography": hazard_pack["topography"],
        "insurance_viability": insurance,
        "indicative_simulation": sim,
        "disclaimer": DISCLAIMER,
    }
