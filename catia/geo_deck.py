"""
Deck.gl + MapLibre GL map for high-density live catastrophe points.

Uses ``deckgl-dash`` (WebGL ScatterplotLayer over a vector MapLibre basemap).
Falls back gracefully when the optional dependency is not installed.

Basemap styles are third-party (e.g. CARTO); follow each provider's terms.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catia.geo_hazards import PERIL_VIS_COLORS
from catia.live_catastrophe_feeds import category_color

# Imported at module load: Dash rejects component libraries first imported
# inside a callback (ImportedInsideCallbackError).
try:
    from deckgl_dash import DeckGL
    from deckgl_dash.layers import GeoJsonLayer, ScatterplotLayer
    from deckgl_dash.maplibre import MapLibreConfig, MapLibreStyle
except ImportError:  # optional dependency
    DeckGL = None
    GeoJsonLayer = None
    ScatterplotLayer = None
    MapLibreConfig = None
    MapLibreStyle = None


def _hex_to_rgba(hex_color: str, alpha: int = 210) -> Tuple[int, int, int, int]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 6:
        return (
            int(h[0:2], 16),
            int(h[2:4], 16),
            int(h[4:6], 16),
            alpha,
        )
    return (34, 211, 238, alpha)


def _event_rgba(event: Dict[str, Any]) -> Tuple[int, int, int, int]:
    peril = event.get("catia_peril")
    if isinstance(peril, str) and peril in PERIL_VIS_COLORS:
        return _hex_to_rgba(PERIL_VIS_COLORS[peril])
    cat = str(event.get("category") or "other")
    return _hex_to_rgba(category_color(cat))


def _radius_meters(score: float) -> float:
    s = max(0.0, min(100.0, score))
    # Geographic radius (meters); scales gently with CATIA score for zoomed-out views
    return 8_000.0 + 95_000.0 * (s / 100.0) ** 0.6


def _resolve_maplibre_style():
    if MapLibreStyle is None:
        return None
    name = os.environ.get("CATIA_DECK_MAP_STYLE", "CARTO_DARK_MATTER").strip().upper()
    return getattr(MapLibreStyle, name, MapLibreStyle.CARTO_DARK_MATTER)


def build_live_deck_earth_map(
    events: List[Dict[str, Any]],
    *,
    height: str = "480px",
    map_component_id: str = "catia-live-deck-map",
) -> Optional[Any]:
    """
    Build a Dash ``DeckGL`` MapLibre + ScatterplotLayer for live feed events.

    Returns ``None`` if ``deckgl-dash`` is not installed or initialization fails.
    """
    if DeckGL is None:
        return None

    style_token = _resolve_maplibre_style()
    if style_token is None:
        return None

    rows: List[Dict[str, Any]] = []
    for e in events:
        try:
            lon = float(e["lon"])
            lat = float(e["lat"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            sc = float(e.get("catia_score") if e.get("catia_score") is not None else 45.0)
        except (TypeError, ValueError):
            sc = 45.0
        rgba = list(_event_rgba(e))
        rows.append(
            {
                "coordinates": [lon, lat],
                "radius": _radius_meters(sc),
                "fillColor": rgba,
            }
        )

    if not rows:
        rows.append(
            {
                "coordinates": [0.0, 20.0],
                "radius": 8000.0,
                "fillColor": [100, 116, 139, 40],
            }
        )

    scatter_layer = ScatterplotLayer(
        id="catia-live-scatter",
        data=rows,
        get_position="@@=coordinates",
        get_radius="@@=radius",
        get_fill_color="@@=fillColor",
        radius_units="meters",
        pickable=True,
        stroked=True,
        get_line_color=[15, 23, 42, 180],
        line_width_min_pixels=1.0,
        opacity=0.92,
    )

    layers: List[Any] = []
    overlay_on = os.environ.get("CATIA_EXPOSURE_OVERLAY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    exp_path = Path(__file__).resolve().parent / "data" / "indicative_exposure_regions.geojson"
    if overlay_on and exp_path.is_file():
        try:
            with open(exp_path, encoding="utf-8") as ef:
                exp_geo: Dict[str, Any] = json.load(ef)
            layers.append(
                GeoJsonLayer(
                    id="catia-exposure-indicative",
                    data=exp_geo,
                    pickable=False,
                    stroked=False,
                    filled=True,
                    get_fill_color=[251, 191, 36, 32],
                    opacity=0.85,
                )
            )
        except Exception:
            pass
    layers.append(scatter_layer)

    # Camera — centroid of rendered points (same as Scatterplot data)
    lons = [r["coordinates"][0] for r in rows]
    lats = [r["coordinates"][1] for r in rows]
    lon0 = sum(lons) / len(lons)
    lat0 = sum(lats) / len(lats)
    ivs: Dict[str, Any] = {
        "longitude": lon0,
        "latitude": lat0,
        "zoom": 2.0,
        "pitch": 0,
        "bearing": 0,
    }

    ml = MapLibreConfig(style=style_token).to_dict()

    try:
        return DeckGL(
            id=map_component_id,
            layers=layers,
            initial_view_state=ivs,
            maplibre_config=ml,
            style={"width": "100%", "height": height, "minHeight": height},
            controller=True,
            tooltip=True,
        )
    except Exception:
        return None
