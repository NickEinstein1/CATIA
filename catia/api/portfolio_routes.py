"""
REST endpoints for portfolio accumulation analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from catia.api.schemas import (
    PortfolioAccumulateRequest,
    PortfolioAccumulateResponse,
    PortfolioExportRequest,
    PortfolioLiveHitRequest,
    PortfolioOneStormRequest,
)
from catia.live_portfolio import find_live_event, live_event_to_one_storm
from catia.live_service import fetch_live_events_base
from catia.portfolio import (
    parse_portfolio_payload,
    portfolio_summary,
    resolve_portfolio_locations,
)
from catia.portfolio_accumulation import analyze_portfolio, one_storm_scenario
from catia.portfolio_export import build_portfolio_export_zip
from catia.reinsurance import apply_layers_to_loss

logger = logging.getLogger(__name__)
portfolio_router = APIRouter(prefix="/portfolio", tags=["Portfolio Accumulation"])


def _locations_payload(body: Any) -> Dict[str, Any]:
    locs = None
    if getattr(body, "locations", None):
        locs = [loc.model_dump() for loc in body.locations]
    return {
        "locations": locs,
        "csv_text": getattr(body, "csv_text", None),
        "geojson": getattr(body, "geojson", None),
    }


def _layers_payload(body: Any) -> Optional[List[Dict[str, Any]]]:
    layers = getattr(body, "reinsurance_layers", None)
    if not layers:
        return None
    return [L.model_dump() if hasattr(L, "model_dump") else dict(L) for L in layers]


@portfolio_router.post("/resolve")
async def portfolio_resolve(body: PortfolioAccumulateRequest) -> Dict[str, Any]:
    """Parse and resolve locations to CATIA regions (no Monte Carlo)."""
    try:
        raw = parse_portfolio_payload(**_locations_payload(body))
        resolved = resolve_portfolio_locations(raw, include_fema=body.include_fema)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {
        "summary": portfolio_summary(resolved),
        "locations": [r.to_dict() for r in resolved],
        "include_fema": body.include_fema,
    }


@portfolio_router.post("/accumulate", response_model=PortfolioAccumulateResponse)
async def portfolio_accumulate(body: PortfolioAccumulateRequest):
    """
    Roll up indicative AAL/VaR by peril, region, and flood zone.

    Optional ``one_storm`` / ``live_event`` and ``reinsurance_layers``.
    """
    try:
        storm = body.one_storm.model_dump() if body.one_storm else None
        result = analyze_portfolio(
            **_locations_payload(body),
            perils=[p.value for p in body.perils],
            num_iterations=body.num_iterations,
            scenario_id=body.scenario_id,
            include_fema=body.include_fema,
            group_by=body.group_by,
            run_simulation=body.run_simulation,
            one_storm=storm,
            live_event=body.live_event,
            reinsurance_layers=_layers_payload(body),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("portfolio accumulate failed")
        raise HTTPException(status_code=500, detail="Portfolio accumulation failed") from e
    return PortfolioAccumulateResponse(**result)


@portfolio_router.post("/one-storm")
async def portfolio_one_storm(body: PortfolioOneStormRequest) -> Dict[str, Any]:
    """What if this one storm hits the book (region or radius filter)."""
    try:
        raw = parse_portfolio_payload(**_locations_payload(body))
        resolved = resolve_portfolio_locations(raw, include_fema=body.include_fema)
        if body.live_event:
            spec = live_event_to_one_storm(body.live_event)
            peril = spec["peril"]
            intensity = float(spec["intensity"])
            mode = "radius"
            region_id = None
            center_lat = spec["center_lat"]
            center_lon = spec["center_lon"]
            radius_km = float(spec["radius_km"])
            live_meta = {
                k: spec.get(k)
                for k in (
                    "live_event_id",
                    "live_event_title",
                    "live_event_source",
                    "intensity_source",
                    "note",
                )
            }
        else:
            peril = body.peril.value
            intensity = body.intensity
            mode = body.mode
            region_id = body.region_id
            center_lat = body.center_lat
            center_lon = body.center_lon
            radius_km = body.radius_km
            live_meta = {}
        storm = one_storm_scenario(
            resolved,
            peril=peril,
            intensity=intensity,
            mode=mode,  # type: ignore[arg-type]
            region_id=region_id,
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
        )
        storm.update({k: v for k, v in live_meta.items() if v is not None})
        layers = _layers_payload(body)
        if layers:
            storm["reinsurance"] = apply_layers_to_loss(float(storm["total_loss"]), layers)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("portfolio one-storm failed")
        raise HTTPException(status_code=500, detail="One-storm scenario failed") from e
    return {
        "summary": portfolio_summary(resolved),
        "one_storm": storm,
        "reinsurance_layers": layers,
    }


@portfolio_router.post("/live-hit", response_model=PortfolioAccumulateResponse)
async def portfolio_live_hit(body: PortfolioLiveHitRequest):
    """Live feed event hits the book (radius shock + optional XL layers)."""
    try:
        event = body.event
        if event is None:
            if not body.event_id:
                raise ValueError("Provide event or event_id")
            store = fetch_live_events_base(force=False)
            events = (store or {}).get("events") or []
            event = find_live_event(events, body.event_id)
        storm = live_event_to_one_storm(
            event,
            radius_km=body.radius_km,
            intensity=body.intensity,
        )
        result = analyze_portfolio(
            **_locations_payload(body),
            include_fema=body.include_fema,
            run_simulation=body.run_simulation,
            num_iterations=body.num_iterations,
            one_storm=storm,
            reinsurance_layers=_layers_payload(body),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("portfolio live-hit failed")
        raise HTTPException(status_code=500, detail="Live-hit scenario failed") from e
    return PortfolioAccumulateResponse(**result)


@portfolio_router.post("/export")
async def portfolio_export(body: PortfolioExportRequest):
    """ZIP underwriting pack: JSON + CSV tables + HTML summary."""
    try:
        storm = body.one_storm.model_dump() if body.one_storm else None
        result = analyze_portfolio(
            **_locations_payload(body),
            perils=[p.value for p in body.perils],
            num_iterations=body.num_iterations,
            scenario_id=body.scenario_id,
            include_fema=body.include_fema,
            group_by=body.group_by,
            run_simulation=body.run_simulation,
            one_storm=storm,
            live_event=body.live_event,
            reinsurance_layers=_layers_payload(body),
        )
        data, filename = build_portfolio_export_zip(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("portfolio export failed")
        raise HTTPException(status_code=500, detail="Portfolio export failed") from e
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
