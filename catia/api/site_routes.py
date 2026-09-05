"""
REST endpoints for site / property viability assessment.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from catia.api.schemas import SiteAssessRequest, SiteAssessResponse
from catia.site_geo import nearest_region
from catia.site_viability import assess_site_viability

site_router = APIRouter(prefix="/site", tags=["Site Viability"])


@site_router.get("/regions/nearest")
async def site_nearest_region(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Nearest CATIA named region for a coordinate (screening geography)."""
    return nearest_region(lat, lon)


@site_router.post("/assess", response_model=SiteAssessResponse)
async def site_assess(body: SiteAssessRequest):
    """
    Assess buy-land / build / buy-building viability from topography + regional risk.

    Indicative actuarial screening — not binding underwriting.
    """
    if body.lat is None and body.lon is None and not (body.address or "").strip():
        raise HTTPException(
            status_code=422,
            detail="Provide lat and lon, or an address (with CATIA_SITE_GEOCODE=1).",
        )
    if (body.lat is None) ^ (body.lon is None):
        raise HTTPException(status_code=422, detail="Provide both lat and lon together.")
    try:
        result = assess_site_viability(
            lat=body.lat,
            lon=body.lon,
            address=body.address,
            property_type=body.property_type.value,
            construction_type=body.construction_type,
            occupancy=body.occupancy,
            tiv=body.tiv,
            include_simulation=body.include_simulation,
            scenario_id=body.scenario_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return SiteAssessResponse(**result)
