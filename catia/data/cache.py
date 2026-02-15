"""
File-based cache for data connectors.
Cache key = hash of (source, params). TTL supported.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _cache_key(source: str, params: Dict[str, Any]) -> str:
    """Stable hash for cache key."""
    payload = json.dumps({"source": source, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class FileDataCache:
    """
    File-based cache for DataFrame and dict payloads.
    Uses pickle for speed; optional TTL in seconds.
    """

    def __init__(self, cache_dir: str, ttl_seconds: Optional[int] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds  # None = no expiry

    def get(self, source: str, params: Dict[str, Any]) -> Optional[Any]:
        """Return cached value if present and not expired."""
        key = _cache_key(source, params)
        path = self.cache_dir / f"{key}.pkl"
        meta_path = self.cache_dir / f"{key}.meta.json"
        if not path.exists():
            return None
        if self.ttl_seconds and meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if time.time() - meta.get("created_at", 0) > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
            except Exception:
                pass
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", key, e)
            path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def set(self, source: str, params: Dict[str, Any], value: Any) -> None:
        """Store value in cache."""
        key = _cache_key(source, params)
        path = self.cache_dir / f"{key}.pkl"
        meta_path = self.cache_dir / f"{key}.meta.json"
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
            with open(meta_path, "w") as f:
                json.dump({"source": source, "params": params, "created_at": time.time()}, f)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key, e)
