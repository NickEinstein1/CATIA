"""
Disk / Redis cache for live feed payloads (JSON-serializable dicts).

Use when ``CATIA_LIVE_DISK_CACHE`` is truthy or ``CATIA_REDIS_URL`` is set.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "live_feed_cache.json"


def _cache_path() -> Path:
    base = os.environ.get("CATIA_CACHE_DIR") or os.environ.get("CATIA_CACHE_PATH")
    if base:
        return Path(base) / _CACHE_FILENAME
    from catia.config import DATA_CONFIG

    return Path(DATA_CONFIG.get("cache_dir", "data/cache")) / _CACHE_FILENAME


def _redis_client():
    url = os.environ.get("CATIA_REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis  # type: ignore
    except ImportError:
        logger.warning("CATIA_REDIS_URL set but redis package not installed; use pip install redis")
        return None
    return redis.from_url(url, decode_responses=True)


def cache_get(key: str, ttl_sec: float) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Return (payload, backend) where backend is ``redis``, ``disk``, or ``miss``.
    """
    r = _redis_client()
    if r is not None:
        try:
            raw = r.get(key)
            if raw:
                blob = json.loads(raw)
                if time.time() - float(blob.get("_ts", 0)) < ttl_sec:
                    return blob.get("data"), "redis"
        except Exception as e:
            logger.debug("Redis cache read failed: %s", e)
        return None, "miss"

    if os.environ.get("CATIA_LIVE_DISK_CACHE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None, "miss"

    path = _cache_path()
    if not path.is_file():
        return None, "miss"
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("key") != key:
            return None, "miss"
        if time.time() - float(blob.get("_ts", 0)) >= ttl_sec:
            return None, "miss"
        data = blob.get("data")
        if isinstance(data, dict):
            return data, "disk"
    except Exception as e:
        logger.debug("Disk cache read failed: %s", e)
    return None, "miss"


def cache_set(key: str, data: Dict[str, Any]) -> str:
    """Persist payload; returns backend used (``redis``, ``disk``, or ``none``)."""
    r = _redis_client()
    if r is not None:
        try:
            blob = json.dumps({"_ts": time.time(), "data": data})
            r.setex(key, 86400, blob)
            return "redis"
        except Exception as e:
            logger.warning("Redis cache write failed: %s", e)

    if os.environ.get("CATIA_LIVE_DISK_CACHE", "1").strip().lower() in ("0", "false", "no", "off"):
        return "none"

    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"key": key, "_ts": time.time(), "data": data}, f, indent=0)
        return "disk"
    except Exception as e:
        logger.warning("Disk cache write failed: %s", e)
    return "none"
