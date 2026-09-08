"""
REST endpoints for portfolio accumulation analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from catia.api.schemas import (
    PortfolioAccumulateRequest,
    PortfolioAccumulateResponse,
    PortfolioOneStormRequest,
)
from catia.portfolio import (
    parse_portfolio_payload,
    portfolio_summary,
    resolve_portfolio_locations,
)
from catia.portfolio_accumulation import analyze_portfolio, one_storm_scenario

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

    Optional ``one_storm`` applies a deterministic intensity shock to part of the book.
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
        storm = one_storm_scenario(
            resolved,
            peril=body.peril.value,
            intensity=body.intensity,
            mode=body.mode,  # type: ignore[arg-type]
            region_id=body.region_id,
            center_lat=body.center_lat,
            center_lon=body.center_lon,
            radius_km=body.radius_km,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("portfolio one-storm failed")
        raise HTTPException(status_code=500, detail="One-storm scenario failed") from e
    return {
        "summary": portfolio_summary(resolved),
        "one_storm": storm,
    }
