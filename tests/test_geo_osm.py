"""Tests for OpenStreetMap / Leaflet map builder."""

from catia.geo_osm import OSM_TILE_URL, build_osm_leaflet_map


def test_osm_tile_url_is_dark_basemap():
    # Default aligns with Deck.gl CARTO Dark Matter; OSM daylight tiles are opt-in via env.
    assert "cartocdn.com" in OSM_TILE_URL or "openstreetmap.org" in OSM_TILE_URL


def test_build_osm_leaflet_map_returns_map():
    m = build_osm_leaflet_map(None, None)
    assert m is not None
    # dash_leaflet.Map
    assert hasattr(m, "zoom")
