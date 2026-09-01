"""
API-first live intelligence service — single entry for dashboard, REST, and agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catia.live_catastrophe_feeds import LiveFeedResult, fetch_all_live_events
from catia.live_event_schema import finalize_event
from catia.live_exposure import attach_exposure_overlap, indicative_exposure_geojson
from catia.live_intel import enrich_and_rank_events


def _post_process_event(event: Dict[str, Any]) -> Dict[str, Any]:
    ev = finalize_event(event)
    return attach_exposure_overlap(ev)


def fetch_live_events_base(
    *,
    force: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Fetch and post-process events without scoring filters (for dashboard store caching)."""
    feed: LiveFeedResult = fetch_all_live_events(use_cache=use_cache, force=force)
    processed = [_post_process_event(e) for e in feed.events]
    return {
        "events": processed,
        "fetched_at_iso": feed.fetched_at_iso,
        "offline_mode": feed.offline_mode,
        "cache_hit": feed.cache_hit,
        "cache_backend": feed.cache_backend,
        "sources_ok": feed.sources_ok,
        "latency_ms": feed.latency_ms,
        "http_status": feed.http_status,
        "errors": feed.errors,
        "count_raw": len(feed.events),
    }


def enrich_stored_live_events(
    store: Dict[str, Any],
    *,
    focal_region: Optional[str] = None,
    peril_filter: Optional[str] = None,
    min_score: float = 0.0,
    limit: int = 500,
) -> Dict[str, Any]:
    """Apply CATIA scoring and filters to cached post-processed events (no network)."""
    processed = list(store.get("events") or [])
    enriched = enrich_and_rank_events(
        processed,
        focal_region=focal_region,
        peril_filter=peril_filter,
        limit=limit,
    )
    if min_score > 0:
        enriched = [e for e in enriched if float(e.get("catia_score") or 0.0) >= min_score]
    return {**store, "events": enriched, "count_filtered": len(enriched)}


def fetch_enriched_live_events(
    *,
    focal_region: Optional[str] = None,
    peril_filter: Optional[str] = None,
    min_score: float = 0.0,
    limit: int = 500,
    force: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fetch feeds, enrich with CATIA intelligence, exposure overlap, and confidence.

    Returns a JSON-serializable payload for API consumers and agents.
    """
    feed: LiveFeedResult = fetch_all_live_events(use_cache=use_cache, force=force)
    base = fetch_live_events_base(force=force, use_cache=use_cache)
    enriched = enrich_stored_live_events(
        base,
        focal_region=focal_region,
        peril_filter=peril_filter,
        min_score=min_score,
        limit=limit,
    )
    events = enriched["events"]

    with_geom = sum(1 for e in events if e.get("geometry_kind") not in (None, "", "unknown"))
    with_footprint = sum(
        1 for e in events if str(e.get("geometry_kind") or "") in ("polygon", "linestring")
    )

    return {
        "fetched_at_iso": feed.fetched_at_iso,
        "offline_mode": feed.offline_mode,
        "cache_hit": feed.cache_hit,
        "cache_backend": feed.cache_backend,
        "sources_ok": feed.sources_ok,
        "latency_ms": feed.latency_ms,
        "http_status": feed.http_status,
        "errors": feed.errors,
        "count_raw": base["count_raw"],
        "count_filtered": enriched["count_filtered"],
        "geometry_summary": {
            "with_geometry": with_geom,
            "with_footprint": with_footprint,
        },
        "events": events,
        "disclaimer": (
            "Observational live intelligence only — heuristic CATIA scores and indicative "
            "exposure overlap are not modeled loss or underwriting advice."
        ),
    }


def live_health_payload(*, force: bool = False) -> Dict[str, Any]:
    feed = fetch_all_live_events(use_cache=not force, force=force)
    return {
        "fetched_at_iso": feed.fetched_at_iso,
        "offline_mode": feed.offline_mode,
        "cache_hit": feed.cache_hit,
        "cache_backend": feed.cache_backend,
        "sources_ok": feed.sources_ok,
        "latency_ms": feed.latency_ms,
        "http_status": feed.http_status,
        "errors": feed.errors,
        "event_count": len(feed.events),
    }
