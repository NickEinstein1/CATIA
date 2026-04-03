"""
OpenStreetMap integration for CATIA (dashboard maps).

Uses standard **raster map tiles** from the OpenStreetMap community tile server
for a 2D slippy map. See https://www.openstreetmap.org/copyright for license
and attribution.

Notes:

- Tiles are for **visualization** only; follow OSMF tile usage policy for
  production (cache, no bulk scraping from the app server — browser fetches are
  typical for dashboards).
- **Vector / building data** from OSM (Overpass API, geofabrik extracts) can be
  layered later for exposure-grade workflows; this module focuses on the basemap
  + CATIA hazard markers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catia.config import PERIL_CONFIG
from catia.geo_hazards import (
    PERIL_VIS_COLORS,
    REGION_CENTROIDS,
    aggregate_region_incidents,
)

# OSM standard raster tile URL pattern (browser requests tiles).
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Required attribution for OSM tiles.
OSM_ATTRIBUTION_HTML = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)


def build_osm_leaflet_map(
    report: Optional[Dict[str, Any]] = None,
    focal_region: Optional[str] = None,
    *,
    height: str = "460px",
    zoom: Optional[int] = None,
) -> Any:
    """
    Build a dash-leaflet Map with OSM tiles and hazard markers from ``report``.

    Returns a ``dash_leaflet.Map`` instance, or ``None`` if ``dash_leaflet`` is
    not installed.
    """
    try:
        import dash_leaflet as dl
    except ImportError:
        return None

    incidents = aggregate_region_incidents(report)
    center_lat, center_lon = 15.0, 0.0
    map_zoom = zoom if zoom is not None else 2
    if focal_region and focal_region in REGION_CENTROIDS:
        center_lat, center_lon = REGION_CENTROIDS[focal_region]
        map_zoom = zoom if zoom is not None else 4

    layers: List[Any] = [
        dl.TileLayer(url=OSM_TILE_URL, attribution=OSM_ATTRIBUTION_HTML),
    ]

    for i in incidents:
        lat, lon = i["lat"], i["lon"]
        dom = i["dominant_peril"]
        color = PERIL_VIS_COLORS.get(dom, "#94a3b8")
        r = max(8, min(36, int(8 + 28 * (float(i["intensity_norm"]) ** 0.5))))
        pname = PERIL_CONFIG.get(dom, {}).get("name", dom)
        rid = i["region_id"].replace("_", " ")
        popup_text = f"{rid} · {pname} · index {100 * float(i['intensity_norm']):.0f}%"
        layers.append(
            dl.CircleMarker(
                center=[lat, lon],
                radius=r,
                pathOptions=dict(
                    color=color,
                    fillColor=color,
                    fillOpacity=0.78,
                    weight=2,
                ),
                children=dl.Popup(popup_text),
            )
        )

    if focal_region and focal_region in REGION_CENTROIDS:
        flat, flon = REGION_CENTROIDS[focal_region]
        layers.append(
            dl.CircleMarker(
                center=[flat, flon],
                radius=22,
                pathOptions=dict(
                    color="#ec4899",
                    fillColor="#ec4899",
                    fillOpacity=0.18,
                    weight=3,
                ),
                children=dl.Popup("Analysis focal region"),
            )
        )

    return dl.Map(
        center=[center_lat, center_lon],
        zoom=map_zoom,
        style={"height": height, "width": "100%", "borderRadius": "12px"},
        children=layers,
    )
