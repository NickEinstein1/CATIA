"""
User-configurable rules for highlighting high-priority live events in the dashboard.

Rules are JSON-serializable; optional file ``outputs/live_alert_rules.json`` (override with
``CATIA_LIVE_ALERT_RULES_PATH``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import math

from catia.config import OUTPUT_CONFIG
from catia.geo_regions import REGION_CENTROIDS

logger = logging.getLogger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))

DEFAULT_RULES: List[Dict[str, Any]] = [
    {"id": "high_score_anywhere", "min_score": 75.0, "label": "CATIA score ≥ 75"},
    {"id": "eq_focus_gulf", "min_score": 55.0, "perils": ["earthquake"], "region": "US_Gulf_Coast", "radius_km": 2500},
]


def rules_path() -> Path:
    p = os.environ.get("CATIA_LIVE_ALERT_RULES_PATH", "").strip()
    if p:
        return Path(p)
    base = Path(OUTPUT_CONFIG.get("output_dir", "outputs"))
    return base / "live_alert_rules.json"


def load_rules() -> List[Dict[str, Any]]:
    path = rules_path()
    if not path.is_file():
        return list(DEFAULT_RULES)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data["rules"]
    except Exception as e:
        logger.warning("Could not load live alert rules from %s: %s", path, e)
    return list(DEFAULT_RULES)


@dataclass
class LiveAlertHit:
    rule_id: str
    label: str
    event_title: str
    score: float


def evaluate_live_rules(events: List[Dict[str, Any]], rules: Optional[List[Dict[str, Any]]] = None) -> List[LiveAlertHit]:
    """Return rule hits for enriched events (need lat, lon, catia_score, catia_peril)."""
    rl = rules if rules is not None else load_rules()
    hits: List[LiveAlertHit] = []
    for rule in rl:
        rid = str(rule.get("id", "rule"))
        label = str(rule.get("label", rid))
        min_sc = float(rule.get("min_score", 0))
        perils = rule.get("perils")
        region = rule.get("region")
        radius_km = float(rule.get("radius_km", 3000))

        fl: Optional[float] = None
        fn: Optional[float] = None
        if region and region in REGION_CENTROIDS:
            fl, fn = REGION_CENTROIDS[region]

        for ev in events:
            try:
                sc = float(ev.get("catia_score") or 0)
            except (TypeError, ValueError):
                continue
            if sc < min_sc:
                continue
            pk = str(ev.get("catia_peril") or "")
            if perils and pk not in perils:
                continue
            if fl is not None and fn is not None:
                try:
                    lat = float(ev.get("lat"))
                    lon = float(ev.get("lon"))
                except (TypeError, ValueError):
                    continue
                if _haversine_km(fl, fn, lat, lon) > radius_km:
                    continue
            hits.append(
                LiveAlertHit(
                    rule_id=rid,
                    label=label,
                    event_title=str(ev.get("title", ""))[:120],
                    score=sc,
                )
            )
    return hits
