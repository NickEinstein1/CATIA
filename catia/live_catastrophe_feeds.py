"""
Near–real-time environmental / catastrophe event feeds from public APIs.

Sources (no API keys; respect each provider's terms and rate limits):

- **USGS** Earthquake Hazards Program — global earthquakes (GeoJSON feeds).
- **NASA EONET** — open natural events (wildfires, severe storms, volcanoes, etc.).

Data is normalized for the dashboard; timestamps and positions are as reported by the source.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

USER_AGENT = "CATIA-dashboard/1.0 (research; +https://github.com/)"

# Defaults — override with env if needed
USGS_FEED_URL = os.environ.get(
    "CATIA_USGS_GEOJSON_URL",
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson",
)
EONET_EVENTS_URL = os.environ.get(
    "CATIA_EONET_EVENTS_URL",
    "https://eonet.gsfc.nasa.gov/api/v3/events",
)

# In-memory cache: avoid hammering APIs when the dashboard callback runs often.
_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": None}
_DEFAULT_TTL_SEC = float(os.environ.get("CATIA_LIVE_FEED_CACHE_SEC", "90"))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _iso_from_ms(ms: Optional[float]) -> str:
    if ms is None:
        return ""
    try:
        dt = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return ""


def _collect_lon_lat_pairs(coords: Any, acc: List[Tuple[float, float]]) -> None:
    """Flatten GeoJSON-like nested coordinate lists to (lon, lat) pairs."""
    if isinstance(coords, (int, float)):
        return
    if isinstance(coords, list) and len(coords) >= 2:
        a, b = coords[0], coords[1]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            acc.append((float(a), float(b)))
            return
        for c in coords:
            _collect_lon_lat_pairs(c, acc)


def _centroid_from_eonet_geometry(geom: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """Mean position from EONET geometry (points, lines, polygons)."""
    if not geom:
        return None
    pairs: List[Tuple[float, float]] = []
    for g in geom:
        coords = g.get("coordinates")
        if coords is not None:
            _collect_lon_lat_pairs(coords, pairs)
    if not pairs:
        return None
    lon_m = sum(p[0] for p in pairs) / len(pairs)
    lat_m = sum(p[1] for p in pairs) / len(pairs)
    return lat_m, lon_m


def fetch_usgs_earthquakes(session: requests.Session, timeout: int = 15) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    r = session.get(USGS_FEED_URL, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    for feat in data.get("features") or []:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        mag = props.get("mag")
        place = props.get("place") or props.get("title") or "Earthquake"
        tid = str(props.get("id") or feat.get("id") or f"usgs-{lat:.2f}-{lon:.2f}")
        out.append(
            {
                "id": f"usgs:{tid}",
                "lat": lat,
                "lon": lon,
                "title": place,
                "category": "earthquake",
                "category_label": "Earthquake",
                "time_iso": _iso_from_ms(props.get("time")),
                "severity_label": f"M {mag:.1f}" if isinstance(mag, (int, float)) else "",
                "source": "USGS",
                "url": props.get("url") or "https://earthquake.usgs.gov/",
            }
        )
    return out


def fetch_eonet_events(session: requests.Session, timeout: int = 15) -> List[Dict[str, Any]]:
    params = {"status": "open", "days": 14, "limit": 120}
    r = session.get(EONET_EVENTS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    out: List[Dict[str, Any]] = []
    for ev in data.get("events") or []:
        geom = ev.get("geometry") or []
        ll = _centroid_from_eonet_geometry(geom)
        if ll is None:
            continue
        lat, lon = ll
        cats = ev.get("categories") or [{}]
        cat_title = (cats[0] or {}).get("title") or "Natural event"
        cat_id = str((cats[0] or {}).get("id") or "eonet").lower()
        slug = cat_title.lower().replace(" ", "_")[:40]
        eid = str(ev.get("id") or ev.get("title") or "eonet")
        link = ev.get("link") or "https://eonet.gsfc.nasa.gov/"
        title = str(ev.get("title") or cat_title)
        out.append(
            {
                "id": f"eonet:{eid}",
                "lat": lat,
                "lon": lon,
                "title": title[:200],
                "category": slug,
                "category_label": cat_title,
                "time_iso": "",
                "severity_label": "",
                "source": "NASA EONET",
                "url": link,
            }
        )
    return out


@dataclass
class LiveFeedResult:
    """Normalized live feed for the dashboard."""

    events: List[Dict[str, Any]]
    errors: List[str]
    fetched_at_iso: str
    sources_ok: Dict[str, bool]


def fetch_all_live_events(
    *,
    use_cache: bool = True,
    force: bool = False,
    ttl_sec: Optional[float] = None,
    timeout: int = 15,
) -> LiveFeedResult:
    """
    Fetch USGS + EONET, merge and return. Uses a short TTL cache by default.
    Pass ``force=True`` to bypass cache (e.g. when user opens the Live Earth tab).
    """
    if force:
        _CACHE["payload"] = None
    ttl = ttl_sec if ttl_sec is not None else _DEFAULT_TTL_SEC
    now = time.monotonic()
    if use_cache and _CACHE["payload"] is not None and (now - float(_CACHE["ts"])) < ttl:
        return _CACHE["payload"]  # type: ignore[return-value]

    errors: List[str] = []
    events: List[Dict[str, Any]] = []
    sources_ok: Dict[str, bool] = {"usgs": False, "eonet": False}
    sess = _session()

    try:
        events.extend(fetch_usgs_earthquakes(sess, timeout=timeout))
        sources_ok["usgs"] = True
    except Exception as e:
        logger.warning("USGS live feed failed: %s", e)
        errors.append(f"USGS: {e}")

    try:
        events.extend(fetch_eonet_events(sess, timeout=timeout))
        sources_ok["eonet"] = True
    except Exception as e:
        logger.warning("EONET live feed failed: %s", e)
        errors.append(f"EONET: {e}")

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result = LiveFeedResult(
        events=events,
        errors=errors,
        fetched_at_iso=fetched_at,
        sources_ok=sources_ok,
    )
    _CACHE["ts"] = now
    _CACHE["payload"] = result
    return result


def category_color(category: str) -> str:
    """Stable color for map markers by normalized category slug."""
    key = (category or "other").lower()
    palette = {
        "earthquake": "#f97316",
        "wildfires": "#ef4444",
        "wildfire": "#ef4444",
        "severe_storms": "#eab308",
        "severe_storms_(meteorological)": "#eab308",
        "volcanoes": "#a855f7",
        "volcanic_activity": "#a855f7",
        "floods": "#3b82f6",
        "landslides": "#78716c",
        "drought": "#ca8a04",
        "dust_and_haze": "#64748b",
        "dust_&_haze": "#64748b",
        "sea_and_lake_ice": "#06b6d4",
        "water_color": "#0ea5e9",
        "manmade": "#f43f5e",
        "snow": "#e2e8f0",
    }
    return palette.get(key, "#22d3ee")
