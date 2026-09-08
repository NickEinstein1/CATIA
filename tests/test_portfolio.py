"""Tests for portfolio ingest and accumulation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from catia.portfolio import (
    parse_portfolio_csv,
    parse_portfolio_geojson,
    portfolio_summary,
    resolve_portfolio_locations,
)
from catia.portfolio_accumulation import analyze_portfolio, one_storm_scenario


SAMPLE_CSV = """id,lat,lon,tiv,flood_zone
nola-1,29.95,-90.07,5000000,AE
miami-1,25.76,-80.19,8000000,X
sf-1,37.77,-122.42,12000000,X
"""

SAMPLE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"id": "gulf-a", "tiv": 2_000_000, "flood_zone": "VE"},
            "geometry": {"type": "Point", "coordinates": [-90.07, 29.95]},
        },
        {
            "type": "Feature",
            "properties": {"id": "gulf-b", "tiv": 3_000_000},
            "geometry": {"type": "Point", "coordinates": [-89.9, 30.1]},
        },
    ],
}


def test_parse_portfolio_csv():
    rows = parse_portfolio_csv(SAMPLE_CSV)
    assert len(rows) == 3
    assert rows[0]["id"] == "nola-1"
    assert rows[0]["tiv"] == 5_000_000
    assert rows[0]["flood_zone"] == "AE"


def test_parse_portfolio_geojson():
    rows = parse_portfolio_geojson(SAMPLE_GEOJSON)
    assert len(rows) == 2
    assert rows[0]["lat"] == 29.95
    assert rows[0]["lon"] == -90.07
    assert rows[0]["tiv"] == 2_000_000


def test_resolve_portfolio_locations():
    raw = parse_portfolio_csv(SAMPLE_CSV)
    locs = resolve_portfolio_locations(raw, include_fema=False)
    assert len(locs) == 3
    assert locs[0].region_id == "US_Gulf_Coast"
    assert locs[1].region_id == "US_Southeast" or locs[1].region_id.startswith("US_")
    summary = portfolio_summary(locs)
    assert summary["location_count"] == 3
    assert summary["total_tiv"] == 25_000_000


def test_one_storm_by_region():
    locs = resolve_portfolio_locations(parse_portfolio_csv(SAMPLE_CSV), include_fema=False)
    storm = one_storm_scenario(
        locs,
        peril="hurricane",
        intensity=140,
        mode="region",
        region_id="US_Gulf_Coast",
    )
    assert storm["hit_count"] >= 1
    assert storm["total_loss"] > 0
    assert 0 < storm["damage_ratio"] <= 1


def test_one_storm_by_radius():
    locs = resolve_portfolio_locations(parse_portfolio_csv(SAMPLE_CSV), include_fema=False)
    storm = one_storm_scenario(
        locs,
        peril="hurricane",
        intensity=120,
        mode="radius",
        center_lat=29.95,
        center_lon=-90.07,
        radius_km=50,
    )
    assert storm["hit_count"] >= 1
    assert storm["mode"] == "radius"


def test_analyze_portfolio_csv_small():
    result = analyze_portfolio(
        csv_text=SAMPLE_CSV,
        num_iterations=200,
        include_fema=False,
        run_simulation=True,
        one_storm={
            "peril": "hurricane",
            "intensity": 130,
            "mode": "region",
            "region_id": "US_Gulf_Coast",
        },
    )
    assert result["summary"]["location_count"] == 3
    assert result["book"] is not None
    assert result["book"]["aggregate"].get("mean") is not None
    assert result["book"]["by_peril"]
    assert result["by_region"]
    assert result["one_storm"]["total_loss"] > 0
    assert "disclaimer" in result


def test_analyze_portfolio_geojson_no_sim():
    result = analyze_portfolio(
        geojson=SAMPLE_GEOJSON,
        include_fema=False,
        run_simulation=False,
    )
    assert result["summary"]["location_count"] == 2
    assert result["book"] is None
    assert result["locations"][0]["region_id"] == "US_Gulf_Coast"


def test_portfolio_api_accumulate():
    from catia.api.app import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/portfolio/accumulate",
        json={
            "csv_text": SAMPLE_CSV,
            "num_iterations": 200,
            "include_fema": False,
            "run_simulation": True,
            "one_storm": {
                "peril": "hurricane",
                "intensity": 130,
                "mode": "region",
                "region_id": "US_Gulf_Coast",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["summary"]["location_count"] == 3
    assert data["book"]["aggregate"]["mean"] is not None


def test_portfolio_api_one_storm():
    from catia.api.app import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/portfolio/one-storm",
        json={
            "geojson": SAMPLE_GEOJSON,
            "peril": "flood",
            "intensity": 8,
            "mode": "radius",
            "center_lat": 29.95,
            "center_lon": -90.07,
            "radius_km": 100,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["one_storm"]["hit_count"] >= 1


def test_portfolio_csv_requires_tiv():
    with pytest.raises(ValueError, match="tiv"):
        parse_portfolio_csv("lat,lon\n29.9,-90.0\n")


def test_portfolio_rejects_oversized_book():
    from catia.portfolio import MAX_PORTFOLIO_LOCATIONS

    rows = [
        {"lat": 29.95, "lon": -90.07, "tiv": 1_000_000, "id": f"x{i}"}
        for i in range(MAX_PORTFOLIO_LOCATIONS + 1)
    ]
    with pytest.raises(ValueError, match="max"):
        from catia.portfolio import parse_portfolio_payload

        parse_portfolio_payload(locations=rows)


def test_slice_buckets_capped():
    from catia.portfolio import PortfolioLocation
    from catia.portfolio_accumulation import _run_sliced

    locs = [
        PortfolioLocation(
            id=f"z{i}",
            lat=29.95,
            lon=-90.07,
            tiv=1000.0,
            region_id="US_Gulf_Coast",
            flood_zone=f"Z{i}",
        )
        for i in range(80)
    ]
    rows = _run_sliced(
        locs,
        group_by="flood_zone",
        perils=["hurricane"],
        num_iterations=50,
        max_buckets=10,
    )
    assert len(rows) <= 10
    assert any(r["group"] == "other" for r in rows)
