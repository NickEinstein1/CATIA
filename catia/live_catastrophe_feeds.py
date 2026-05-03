"""
Near–real-time environmental / catastrophe event feeds from public APIs.

Sources (no API keys; respect each provider's terms and rate limits):

- **USGS** Earthquake Hazards Program — global earthquakes (GeoJSON feeds).
- **NASA EONET** — open natural events (wildfires, severe storms, volcanoes, etc.).
- **GDACS** — JRC / UN-style event list (GeoJSON API).

Data is normalized for the dashboard; timestamps and positions are as reported by the source.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
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
GDACS_EVENTLIST_URL = os.environ.get(
    "CATIA_GDACS_URL",
    "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?limit=80",
)

_CACHE_KEY_PERSIST = "catia_live_feed_v2"

# In-memory cache: avoid hammering APIs when the dashboard callback runs often.
_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": None}
_DEFAULT_TTL_SEC = float(os.environ.get("CATIA_LIVE_FEED_CACHE_SEC", "90"))


def _get_timed(
    session: requests.Session, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 15
) -> Tuple[Any, float, int]:
    """HTTP GET returning (response, latency_ms, status_code)."""
    t0 = time.perf_counter()
    r = session.get(url, params=params, timeout=timeout)
    ms = (time.perf_counter() - t0) * 1000.0
    return r, ms, r.status_code


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


def _latest_eonet_geometry_time_iso(geom: List[Dict[str, Any]]) -> str:
    best: Optional[datetime] = None
    for g in geom or []:
        d = g.get("date")
        if not d:
            continue
        try:
            txt = str(d).strip()
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            if best is None or dt > best:
                best = dt
        except Exception:
            continue
    return best.strftime("%Y-%m-%d %H:%M UTC") if best else ""


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


def _parse_usgs_geojson(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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


def fetch_usgs_earthquakes(session: requests.Session, timeout: int = 15) -> List[Dict[str, Any]]:
    r = session.get(USGS_FEED_URL, timeout=timeout)
    r.raise_for_status()
    return _parse_usgs_geojson(r.json())


def fetch_eonet_events(session: requests.Session, timeout: int = 15) -> List[Dict[str, Any]]:
    params = {"status": "open", "days": 14, "limit": 120}
    r = session.get(EONET_EVENTS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return _parse_eonet_json(r.json())


GDACS_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    "EQ": ("earthquake", "Earthquake (GDACS)"),
    "TC": ("hurricane", "Tropical cyclone (GDACS)"),
    "FL": ("floods", "Flood (GDACS)"),
    "DR": ("drought", "Drought (GDACS)"),
    "VO": ("volcanoes", "Volcano (GDACS)"),
    "WF": ("wildfire", "Wildfire (GDACS)"),
}


def _parse_gdacs_geojson(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for feat in data.get("features") or []:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        p = feat.get("properties") or {}
        et = str(p.get("eventtype") or "?").upper()
        slug, label = GDACS_TYPE_MAP.get(et, (et.lower(), f"GDACS ({et})"))
        eid = p.get("eventid", "")
        epid = p.get("episodeid", "")
        title = str(p.get("name") or p.get("eventname") or "GDACS event")
        urls = p.get("url") or {}
        link = urls.get("details") if isinstance(urls, dict) else None
        if not link:
            link = "https://www.gdacs.org/"
        alert = str(p.get("alertlevel") or "")
        sev = str(p.get("severitydata", {}).get("severitytext", "")) if isinstance(p.get("severitydata"), dict) else ""
        sev_label = " · ".join(x for x in (alert, sev) if x)[:120]
        t_raw = p.get("fromdate") or p.get("datemodified") or ""
        t_iso = ""
        if t_raw:
            try:
                txt = str(t_raw).replace("Z", "+00:00")
                if "+" not in txt and "T" in txt:
                    txt += "+00:00"
                dt = datetime.fromisoformat(txt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                t_iso = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                t_iso = str(t_raw)[:32]
        out.append(
            {
                "id": f"gdacs:{et}:{eid}:{epid}",
                "lat": lat,
                "lon": lon,
                "title": title[:200],
                "category": slug,
                "category_label": label,
                "time_iso": t_iso,
                "severity_label": sev_label,
                "source": "GDACS",
                "url": str(link),
            }
        )
    return out


def fetch_gdacs_events(session: requests.Session, timeout: int = 15) -> List[Dict[str, Any]]:
    r = session.get(GDACS_EVENTLIST_URL, timeout=timeout)
    r.raise_for_status()
    return _parse_gdacs_geojson(r.json())


@dataclass
class LiveFeedResult:
    """Normalized live feed for the dashboard."""

    events: List[Dict[str, Any]]
    errors: List[str]
    fetched_at_iso: str
    sources_ok: Dict[str, bool]
    latency_ms: Dict[str, float] = field(default_factory=dict)
    http_status: Dict[str, Optional[int]] = field(default_factory=dict)
    cache_hit: bool = False
    cache_backend: str = "memory"
    offline_mode: bool = False

    def to_cache_blob(self) -> Dict[str, Any]:
        return {
            "events": self.events,
            "errors": self.errors,
            "fetched_at_iso": self.fetched_at_iso,
            "sources_ok": self.sources_ok,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
            "offline_mode": self.offline_mode,
        }

    @staticmethod
    def from_cache_blob(d: Dict[str, Any], *, cache_backend: str) -> "LiveFeedResult":
        return LiveFeedResult(
            events=list(d.get("events") or []),
            errors=list(d.get("errors") or []),
            fetched_at_iso=str(d.get("fetched_at_iso") or ""),
            sources_ok=dict(d.get("sources_ok") or {}),
            latency_ms=dict(d.get("latency_ms") or {}),
            http_status={k: v for k, v in dict(d.get("http_status") or {}).items()},
            cache_hit=True,
            cache_backend=cache_backend,
            offline_mode=bool(d.get("offline_mode", False)),
        )


def fetch_all_live_events(
    *,
    use_cache: bool = True,
    force: bool = False,
    ttl_sec: Optional[float] = None,
    timeout: int = 15,
) -> LiveFeedResult:
    """
    Fetch USGS + EONET + GDACS (optional), merge and return.

    Caching: in-process TTL, optional disk (``CATIA_LIVE_DISK_CACHE``) or Redis
    (``CATIA_REDIS_URL``). Set ``CATIA_LIVE_OFFLINE=1`` for demos without network.

    Pass ``force=True`` to bypass caches for a fresh pull.
    """
    if force:
        _CACHE["payload"] = None
    ttl = ttl_sec if ttl_sec is not None else _DEFAULT_TTL_SEC
    now = time.monotonic()

    if os.environ.get("CATIA_LIVE_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return LiveFeedResult(
            events=[],
            errors=["Offline mode (CATIA_LIVE_OFFLINE) — no network fetch."],
            fetched_at_iso=fetched_at,
            sources_ok={"usgs": False, "eonet": False, "gdacs": False},
            offline_mode=True,
        )

    if use_cache and _CACHE["payload"] is not None and (now - float(_CACHE["ts"])) < ttl:
        return _CACHE["payload"]  # type: ignore[return-value]

    if use_cache and not force:
        try:
            from catia.live_cache import cache_get

            blob, backend = cache_get(_CACHE_KEY_PERSIST, ttl)
            if blob is not None:
                result = LiveFeedResult.from_cache_blob(blob, cache_backend=backend)
                _CACHE["ts"] = now
                _CACHE["payload"] = result
                return result
        except Exception as e:
            logger.debug("Persistent cache read skipped: %s", e)

    errors: List[str] = []
    events: List[Dict[str, Any]] = []
    sources_ok: Dict[str, bool] = {"usgs": False, "eonet": False, "gdacs": False}
    latency_ms: Dict[str, float] = {}
    http_status: Dict[str, Optional[int]] = {}
    sess = _session()

    try:
        r, ms, code = _get_timed(sess, USGS_FEED_URL, timeout=timeout)
        latency_ms["usgs"] = ms
        http_status["usgs"] = code
        r.raise_for_status()
        events.extend(_parse_usgs_geojson(r.json()))
        sources_ok["usgs"] = True
    except Exception as e:
        logger.warning("USGS live feed failed: %s", e)
        errors.append(f"USGS: {e}")

    try:
        r, ms, code = _get_timed(
            sess,
            EONET_EVENTS_URL,
            params={"status": "open", "days": 14, "limit": 120},
            timeout=timeout,
        )
        latency_ms["eonet"] = ms
        http_status["eonet"] = code
        r.raise_for_status()
        events.extend(_parse_eonet_json(r.json()))
        sources_ok["eonet"] = True
    except Exception as e:
        logger.warning("EONET live feed failed: %s", e)
        errors.append(f"EONET: {e}")

    if os.environ.get("CATIA_LIVE_FETCH_GDACS", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            r, ms, code = _get_timed(sess, GDACS_EVENTLIST_URL, timeout=timeout)
            latency_ms["gdacs"] = ms
            http_status["gdacs"] = code
            r.raise_for_status()
            events.extend(_parse_gdacs_geojson(r.json()))
            sources_ok["gdacs"] = True
        except Exception as e:
            logger.warning("GDACS live feed failed: %s", e)
            errors.append(f"GDACS: {e}")

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result = LiveFeedResult(
        events=events,
        errors=errors,
        fetched_at_iso=fetched_at,
        sources_ok=sources_ok,
        latency_ms=latency_ms,
        http_status=http_status,
    )
    _CACHE["ts"] = now
    _CACHE["payload"] = result

    if use_cache:
        try:
            from catia.live_cache import cache_set

            cache_set(_CACHE_KEY_PERSIST, result.to_cache_blob())
        except Exception as e:
            logger.debug("Persistent cache write skipped: %s", e)

    return result


def _parse_eonet_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse EONET API JSON (same as ``fetch_eonet_events`` body)."""
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
        t_iso = _latest_eonet_geometry_time_iso(geom)
        out.append(
            {
                "id": f"eonet:{eid}",
                "lat": lat,
                "lon": lon,
                "title": title[:200],
                "category": slug,
                "category_label": cat_title,
                "time_iso": t_iso,
                "severity_label": "",
                "source": "NASA EONET",
                "url": link,
            }
        )
    return out


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
        "hurricane": "#22d3ee",
        "volcano": "#a855f7",
        "volcanoes": "#a855f7",
    }
    return palette.get(key, "#22d3ee")
