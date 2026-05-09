"""
Approximate geographic centroids for CATIA named regions.

Used for dashboards, proximity scoring, and visualization. Not for underwriting.

Kept dependency-free (no Plotly) so lightweight callers can import safely.
"""

from __future__ import annotations

from typing import Dict, Tuple

# (lat, lon) — aligned with PERIL_CONFIG region ids
REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "US_Gulf_Coast": (28.5, -90.0),
    "US_East_Coast": (38.0, -75.0),
    "US_West_Coast": (37.5, -121.0),
    "US_Midwest": (41.5, -93.0),
    "US_Southwest": (34.0, -112.0),
    "Caribbean": (18.0, -66.0),
    "Southeast_Asia": (5.0, 110.0),
    "South_Asia": (22.0, 79.0),
    "Europe": (48.0, 10.0),
    "Mediterranean": (38.0, 18.0),
    "Australia": (-25.0, 134.0),
    "South_America": (-15.0, -60.0),
    "Africa": (0.0, 20.0),
    "Japan": (36.0, 138.0),
    "Turkey": (39.0, 35.0),
    "Chile": (-30.0, -71.0),
    "Indonesia": (-2.0, 118.0),
}
