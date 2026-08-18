"""Tests for live event schema, geometry, exposure, and REST API."""

from __future__ import annotations

import pytest

from catia.live_catastrophe_feeds import _parse_eonet_json, _parse_gdacs_geojson, _parse_usgs_geojson
from catia.live_event_schema import geometry_kind
from catia.live_exposure import attach_exposure_overlap, exposure_overlap_for_point
from catia.live_geometry import events_to_feature_collection


def test_usgs_event_has_geometry_and_provenance():
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "abc",
                    "mag": 4.2,
                    "place": "Testville",
                    "time": 1_700_000_000_000,
                    "url": "https://earthquake.usgs.gov/",
                },
                "geometry": {"type": "Point", "coordinates": [-100.5, 35.2]},
            }
        ],
    }
    rows = _parse_usgs_geojson(data)
    r = rows[0]
    assert r["geometry"]["type"] == "Point"
    assert r["geometry_kind"] == "point"
    assert r["provenance"]["feed"] == "usgs"
    assert 0.0 < r["confidence"] <= 1.0


def test_eonet_preserves_polygon_geometry():
    data = {
        "events": [
            {
                "id": "EONET-POLY",
                "title": "Wildfire footprint",
                "link": "https://eonet.gsfc.nasa.gov/",
                "categories": [{"id": "wildfires", "title": "Wildfires"}],
                "geometry": [
                    {
                        "date": "2024-01-15T12:00:00Z",
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [12.0, 56.0],
                                [13.0, 56.0],
                                [13.0, 57.0],
                                [12.0, 57.0],
                                [12.0, 56.0],
                            ]
                        ],
                    }
                ],
            }
        ]
    }
    rows = _parse_eonet_json(data)
    r = rows[0]
    assert r["geometry"]["type"] == "Polygon"
    assert r["geometry_kind"] == "polygon"
    assert r["confidence_factors"]["geometry"] >= 0.9


def test_events_to_feature_collection_includes_footprint():
    events = [
        {
            "id": "x1",
            "lat": 40.0,
            "lon": -74.0,
            "title": "Test",
            "category": "wildfire",
            "confidence": 0.8,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-75.0, 39.0],
                        [-73.0, 39.0],
                        [-73.0, 41.0],
                        [-75.0, 41.0],
                        [-75.0, 39.0],
                    ]
                ],
            },
            "geometry_kind": "polygon",
        }
    ]
    fc = events_to_feature_collection(events, include_points=False)
    assert len(fc["features"]) == 1
    assert fc["features"][0]["geometry"]["type"] == "Polygon"


def test_exposure_overlap_for_known_region():
    # North America indicative bbox includes New York area
    overlap = exposure_overlap_for_point(-74.0, 40.7)
    assert isinstance(overlap["overlap_score"], float)
    ev = attach_exposure_overlap({"lat": 40.7, "lon": -74.0, "id": "t"})
    assert "exposure_overlap" in ev


def test_geometry_kind_helper():
    assert geometry_kind({"type": "LineString", "coordinates": []}) == "linestring"


@pytest.fixture
def api_client(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CATIA_LIVE_OFFLINE", "1")
    from catia.api.app import app

    return TestClient(app)


def test_live_health_offline(api_client):
    r = api_client.get("/api/v1/live/health")
    assert r.status_code == 200
    body = r.json()
    assert body["offline_mode"] is True
    assert "sources_ok" in body


def test_live_events_enriched_offline(api_client):
    r = api_client.get("/api/v1/live/events/enriched")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert "disclaimer" in body


def test_live_geojson_offline(api_client):
    r = api_client.get("/api/v1/live/geojson")
    assert r.status_code == 200
    body = r.json()
    assert body["geojson"]["type"] == "FeatureCollection"


def test_live_exposure_regions(api_client):
    r = api_client.get("/api/v1/live/exposure/regions")
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert len(body.get("features") or []) > 0
