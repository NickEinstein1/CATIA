"""
OpenStreetMap integration for CATIA (dashboard maps).

Default basemap: **CARTO Dark Matter** raster tiles (aligned with Deck.gl), with
OpenStreetMap + CARTO attribution. Override via ``CATIA_OSM_TILE_URL``.
See https://www.openstreetmap.org/copyright and CARTO attribution requirements.

Notes:

- Tiles are for **visualization** only; follow each provider's tile usage policy
  for production (cache, no bulk scraping from the app server — browser fetches
  are typical for dashboards).
- **Vector / building data** from OSM (Overpass API, geofabrik extracts) can be
  layered later for exposure-grade workflows; this module focuses on the basemap
  + CATIA hazard markers.
"""

from __future__ import annotations

import os
from html import escape
from typing import Any, Dict, List, Optional

from catia.config import PERIL_CONFIG
from catia.geo_hazards import PERIL_VIS_COLORS, aggregate_region_incidents
from catia.geo_regions import REGION_CENTROIDS
from catia.live_catastrophe_feeds import category_color
from catia.live_geometry import events_to_feature_collection

# Imported at module load: Dash rejects component libraries first imported
# inside a callback (ImportedInsideCallbackError).
try:
    import dash_leaflet as dl
except ImportError:  # optional dependency
    dl = None

# Dark basemap aligned with Deck.gl ``CARTO_DARK_MATTER`` (browser requests tiles).
# Override with ``CATIA_OSM_TILE_URL`` if needed.
OSM_TILE_URL = os.environ.get(
    "CATIA_OSM_TILE_URL",
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
)

# Attribution for CARTO dark tiles (must retain OSM credit).
OSM_ATTRIBUTION_HTML = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    ' · &copy; <a href="https://carto.com/attributions">CARTO</a>'
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
    if dl is None:
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


def build_osm_live_catastrophe_map(
    events: List[Dict[str, Any]],
    *,
    height: str = "520px",
    zoom: int = 2,
    cluster_markers: Optional[bool] = None,
) -> Any:
    """
    Leaflet map with OSM tiles and markers for live feed events (USGS, EONET, etc.).

    When ``cluster_markers`` is True (default from env ``CATIA_LIVE_MAP_CLUSTER``),
    markers are wrapped in ``MarkerClusterGroup`` for dense views. Falls back to a
    flat layer if clustering is unavailable.
    """
    if dl is None:
        return None

    if cluster_markers is None:
        cluster_markers = os.environ.get("CATIA_LIVE_MAP_CLUSTER", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    markers: List[Any] = []
    for ev in events:
        lat = ev.get("lat")
        lon = ev.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        sc_raw = ev.get("catia_score")
        try:
            sc = float(sc_raw) if sc_raw is not None else 45.0
        except (TypeError, ValueError):
            sc = 45.0
        sc = max(0.0, min(100.0, sc))
        r = int(max(5, min(22, 5 + 17 * (sc / 100.0) ** 0.55)))
        cat = str(ev.get("category") or "other")
        peril = ev.get("catia_peril")
        if isinstance(peril, str) and peril in PERIL_VIS_COLORS:
            color = PERIL_VIS_COLORS[peril]
        else:
            color = category_color(cat)
        sev = ev.get("severity_label") or ""
        src = ev.get("source") or ""
        title = str(ev.get("title") or "Event")[:180]
        label = ev.get("category_label") or cat
        tiso = ev.get("time_iso") or ""
        parts = [escape(title), escape(f"{label}" + (f" · {sev}" if sev else ""))]
        if tiso:
            parts.append(escape(tiso))
        parts.append(escape(f"CATIA score: {sc:.0f}"))
        conf = ev.get("confidence")
        if conf is not None:
            try:
                parts.append(escape(f"Confidence: {float(conf):.0%}"))
            except (TypeError, ValueError):
                pass
        exp = ev.get("exposure_overlap") or {}
        regions = exp.get("regions") or []
        if regions:
            parts.append(escape(f"Indicative exposure: {', '.join(str(r) for r in regions[:3])}"))
        gkind = ev.get("geometry_kind")
        if gkind and gkind not in ("point", "unknown"):
            parts.append(escape(f"Footprint: {gkind}"))
        if src:
            parts.append(escape(f"Source: {src}"))
        url = ev.get("url")
        if url:
            parts.append(escape(str(url)))
        popup_text = "\n".join(parts)
        markers.append(
            dl.CircleMarker(
                center=[lat_f, lon_f],
                radius=r,
                pathOptions=dict(
                    color=color,
                    fillColor=color,
                    fillOpacity=0.85,
                    weight=2,
                ),
                children=dl.Popup(popup_text),
            )
        )

    tile = dl.TileLayer(url=OSM_TILE_URL, attribution=OSM_ATTRIBUTION_HTML)
    footprint_fc = events_to_feature_collection(events, include_points=False)
    footprint_layer: Optional[Any] = None
    if footprint_fc.get("features"):
        footprint_layer = dl.GeoJSON(
            data=footprint_fc,
            options={
                "style": {
                    "color": "#22d3ee",
                    "weight": 2,
                    "fillColor": "#22d3ee",
                    "fillOpacity": 0.12,
                }
            },
        )

    layers: List[Any]
    if markers and cluster_markers:
        try:
            clustered = dl.MarkerClusterGroup(id="catia-live-marker-cluster", children=markers)
            layers = [tile]
            if footprint_layer is not None:
                layers.append(footprint_layer)
            layers.append(clustered)
        except Exception:
            layers = [tile]
            if footprint_layer is not None:
                layers.append(footprint_layer)
            layers.extend(markers)
    else:
        layers = [tile]
        if footprint_layer is not None:
            layers.append(footprint_layer)
        layers.extend(markers)

    return dl.Map(
        center=[20.0, 0.0],
        zoom=zoom,
        style={"height": height, "width": "100%", "borderRadius": "12px"},
        children=layers,
    )
