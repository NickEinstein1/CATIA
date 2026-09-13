"""Tests for hazard footprint intensity fields."""

from __future__ import annotations

from catia.hazard_footprints import (
    decay_intensity,
    distance_to_linestring_km,
    local_intensity_for_point,
    point_in_geojson_polygon,
    resolve_footprint_kind,
)
from catia.live_portfolio import live_event_to_one_storm
from catia.portfolio import resolve_portfolio_locations
from catia.portfolio_accumulation import analyze_portfolio, one_storm_scenario


SAMPLE_CSV = """id,lat,lon,tiv
near,29.95,-90.07,5000000
mid,30.5,-90.5,4000000
far,35.0,-95.0,3000000
"""


def test_resolve_footprint_auto_by_peril():
    assert resolve_footprint_kind("hurricane", "auto") == "windfield"
    assert resolve_footprint_kind("earthquake", "auto") == "shake"
    assert resolve_footprint_kind("flood", None) == "flood_bowl"


def test_decay_windfield_falls_with_distance():
    peak = 140.0
    r = 300.0
    near = decay_intensity(peak, 20.0, r, "windfield")
    far = decay_intensity(peak, 250.0, r, "windfield")
    assert near > far
    assert decay_intensity(peak, 400.0, r, "windfield") == 0.0


def test_decay_shake_and_uniform():
    assert decay_intensity(7.0, 10.0, 120.0, "shake") < 7.0
    assert decay_intensity(7.0, 10.0, 120.0, "uniform") == 7.0


def test_point_in_polygon_and_track_distance():
    poly = {
        "type": "Polygon",
        "coordinates": [
            [
                [-91.0, 29.0],
                [-89.0, 29.0],
                [-89.0, 31.0],
                [-91.0, 31.0],
                [-91.0, 29.0],
            ]
        ],
    }
    assert point_in_geojson_polygon(-90.07, 29.95, poly)
    assert not point_in_geojson_polygon(-95.0, 35.0, poly)

    track = [[-90.5, 29.5], [-90.0, 30.0], [-89.5, 30.5]]
    d = distance_to_linestring_km(30.0, -90.0, track)
    assert d < 50.0


def test_local_intensity_track_vs_center():
    track = [[-90.2, 29.8], [-89.9, 30.1]]
    field = local_intensity_for_point(
        30.0,
        -90.0,
        center_lat=29.95,
        center_lon=-90.07,
        peak_intensity=120.0,
        radius_km=200.0,
        footprint="windfield",
        track_coords=track,
    )
    assert field["reference"] == "track"
    assert field["inside"]
    assert field["intensity"] > 0


def test_one_storm_windfield_near_loses_more_than_far():
    locs = resolve_portfolio_locations(
        [
            {"id": "near", "lat": 29.95, "lon": -90.07, "tiv": 5_000_000},
            {"id": "far", "lat": 32.5, "lon": -93.0, "tiv": 5_000_000},
        ],
        include_fema=False,
    )
    storm = one_storm_scenario(
        locs,
        peril="hurricane",
        intensity=140,
        mode="radius",
        center_lat=29.95,
        center_lon=-90.07,
        radius_km=400,
        footprint="windfield",
    )
    assert storm["footprint"]["kind"] == "windfield"
    assert storm["hit_count"] >= 1
    by_id = {r["id"]: r for r in storm["top_locations"]}
    if "near" in by_id and "far" in by_id:
        assert by_id["near"]["loss"] > by_id["far"]["loss"]
        assert by_id["near"]["intensity"] > by_id["far"]["intensity"]


def test_one_storm_uniform_equals_flat_damage():
    locs = resolve_portfolio_locations(
        [{"id": "a", "lat": 29.95, "lon": -90.07, "tiv": 1_000_000}],
        include_fema=False,
    )
    storm = one_storm_scenario(
        locs,
        peril="hurricane",
        intensity=120,
        mode="radius",
        center_lat=29.95,
        center_lon=-90.07,
        radius_km=50,
        footprint="uniform",
    )
    assert storm["hit_count"] == 1
    assert storm["top_locations"][0]["intensity"] == 120


def test_live_event_carries_geometry_and_auto_footprint():
    event = {
        "id": "storm-track",
        "lat": 29.5,
        "lon": -90.0,
        "title": "Gulf TC",
        "category": "severeStorms",
        "source": "EONET",
        "geometry": {
            "type": "LineString",
            "coordinates": [[-91.0, 28.5], [-90.0, 29.5], [-89.0, 30.5]],
        },
    }
    spec = live_event_to_one_storm(event, footprint="auto")
    assert spec["footprint"] == "auto"
    assert spec["geometry"]["type"] == "LineString"
    result = analyze_portfolio(
        csv_text=SAMPLE_CSV,
        num_iterations=100,
        run_simulation=False,
        one_storm=spec,
    )
    assert result["one_storm"]["footprint"]["kind"] == "windfield"
    assert result["one_storm"]["footprint"]["geometry"] == "track"


def test_api_one_storm_with_footprint():
    from fastapi.testclient import TestClient

    from catia.api.app import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/portfolio/one-storm",
        json={
            "csv_text": SAMPLE_CSV,
            "peril": "hurricane",
            "intensity": 130,
            "mode": "radius",
            "center_lat": 29.95,
            "center_lon": -90.07,
            "radius_km": 350,
            "footprint": "windfield",
        },
    )
    assert resp.status_code == 200, resp.text
    storm = resp.json()["one_storm"]
    assert storm["footprint"]["kind"] == "windfield"
    assert storm["hit_count"] >= 1
