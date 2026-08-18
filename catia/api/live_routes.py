"""
REST endpoints for live catastrophe intelligence (API-first layer).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from catia.api.schemas import LiveEventsResponse, LiveGeoJsonResponse, LiveHealthResponse
from catia.live_geometry import events_to_feature_collection
from catia.live_exposure import indicative_exposure_geojson
from catia.live_service import fetch_enriched_live_events, live_health_payload

live_router = APIRouter(prefix="/live", tags=["Live Intelligence"])


@live_router.get("/health", response_model=LiveHealthResponse)
async def live_feed_health(force: bool = Query(False, description="Bypass cache for a fresh probe")):
    """Upstream feed health, latency, and cache status."""
    return LiveHealthResponse(**live_health_payload(force=force))


@live_router.get("/events", response_model=LiveEventsResponse)
async def live_events_raw(
    force: bool = Query(False, description="Bypass caches"),
    limit: int = Query(500, ge=1, le=2000),
):
    """Normalized merged feed events (geometry + provenance + confidence, pre-scoring)."""
    from catia.live_catastrophe_feeds import fetch_all_live_events
    from catia.live_exposure import attach_exposure_overlap

    feed = fetch_all_live_events(use_cache=not force, force=force)
    from catia.live_exposure import attach_exposure_overlap

    events = [attach_exposure_overlap(e) for e in feed.events][:limit]
    return LiveEventsResponse(
        fetched_at_iso=feed.fetched_at_iso,
        offline_mode=feed.offline_mode,
        cache_hit=feed.cache_hit,
        cache_backend=feed.cache_backend,
        sources_ok=feed.sources_ok,
        latency_ms=feed.latency_ms,
        http_status=feed.http_status,
        errors=feed.errors,
        count=len(events),
        events=events,
    )


@live_router.get("/events/enriched", response_model=LiveEventsResponse)
async def live_events_enriched(
    focal_region: Optional[str] = Query(None, description="Region id for proximity scoring"),
    peril: Optional[str] = Query(None, description="CATIA peril filter"),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    limit: int = Query(500, ge=1, le=2000),
    force: bool = Query(False, description="Bypass caches"),
):
    """Full intelligence layer: CATIA scores, confidence, and indicative exposure overlap."""
    payload = fetch_enriched_live_events(
        focal_region=focal_region,
        peril_filter=peril,
        min_score=min_score,
        limit=limit,
        force=force,
    )
    return LiveEventsResponse(
        fetched_at_iso=payload["fetched_at_iso"],
        offline_mode=payload["offline_mode"],
        cache_hit=payload["cache_hit"],
        cache_backend=payload["cache_backend"],
        sources_ok=payload["sources_ok"],
        latency_ms=payload["latency_ms"],
        http_status=payload["http_status"],
        errors=payload["errors"],
        count=payload["count_filtered"],
        geometry_summary=payload.get("geometry_summary"),
        disclaimer=payload.get("disclaimer"),
        events=payload["events"],
    )


@live_router.get("/geojson", response_model=LiveGeoJsonResponse)
async def live_events_geojson(
    focal_region: Optional[str] = Query(None),
    peril: Optional[str] = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=100.0),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(500, ge=1, le=2000),
    force: bool = Query(False),
    include_points: bool = Query(True),
):
    """GeoJSON FeatureCollection for map overlays (tracks / polygons + centroids)."""
    payload = fetch_enriched_live_events(
        focal_region=focal_region,
        peril_filter=peril,
        min_score=min_score,
        limit=limit,
        force=force,
    )
    fc = events_to_feature_collection(
        payload["events"],
        include_points=include_points,
        min_confidence=min_confidence,
    )
    return LiveGeoJsonResponse(
        fetched_at_iso=payload["fetched_at_iso"],
        feature_count=len(fc.get("features") or []),
        geojson=fc,
    )


@live_router.get("/exposure/regions")
async def live_exposure_regions() -> Dict[str, Any]:
    """Indicative world exposure regions (GeoJSON) — not modeled portfolio data."""
    return indicative_exposure_geojson()
