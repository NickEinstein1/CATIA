"""Tests for site viability assessment (property / land screening)."""

from __future__ import annotations

import pytest

from catia.site_geo import nearest_region, resolve_site_location
from catia.site_hazards import build_hazard_assessment, topography_stub
from catia.site_viability import assess_site_viability


def test_nearest_region_gulf():
    hit = nearest_region(29.95, -90.07)
    assert hit["region_id"] == "US_Gulf_Coast"
    assert hit["distance_km"] < 250


def test_nearest_region_japan():
    hit = nearest_region(35.68, 139.69)
    assert hit["region_id"] == "Japan"


def test_resolve_site_requires_coords_or_address():
    with pytest.raises(ValueError):
        resolve_site_location()


def test_topography_stub_coastal_gulf():
    topo = topography_stub(29.95, -90.07, "US_Gulf_Coast")
    assert topo["data_quality"] == "heuristic_stub"
    assert topo["floodplain_hint"]


def test_hazard_assessment_buy_land_vs_build():
    land = build_hazard_assessment(
        lat=29.95,
        lon=-90.07,
        region_id="US_Gulf_Coast",
        property_type="buy_land",
    )
    build = build_hazard_assessment(
        lat=29.95,
        lon=-90.07,
        region_id="US_Gulf_Coast",
        property_type="build",
    )
    assert land["hazards"]
    assert "hurricane" in land["applicable_peril_ids"] or "flood" in land["applicable_peril_ids"]
    # Build typically amplifies structural perils vs vacant land
    land_h = {h["peril"]: h["hazard_score"] for h in land["hazards"]}
    build_h = {h["peril"]: h["hazard_score"] for h in build["hazards"]}
    if "hurricane" in land_h and "hurricane" in build_h:
        assert build_h["hurricane"] >= land_h["hurricane"]


def test_assess_site_viability_new_orleans_building(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "0")
    result = assess_site_viability(
        lat=29.95,
        lon=-90.07,
        property_type="buy_building",
        include_simulation=False,
    )
    assert result["location"]["nearest_region"]["region_id"] == "US_Gulf_Coast"
    assert result["risk_category"] in ("low", "moderate", "high", "severe")
    assert 0 <= result["risk_score"] <= 100
    assert result["insurance_viability"]["status"]
    assert result["topography"]["elevation_band"]
    assert "disclaimer" in result


def test_assess_site_viability_west_coast_build(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "0")
    result = assess_site_viability(
        lat=37.77,
        lon=-122.42,
        property_type="build",
        include_simulation=False,
    )
    assert result["location"]["nearest_region"]["region_id"] == "US_West_Coast"
    perils = {h["peril"] for h in result["hazard_assessment"]}
    assert "earthquake" in perils or "wildfire" in perils


def test_assess_with_mocked_dem_floodplain(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "1")

    def _fake_terrain(lat, lon):
        return {
            "ok": True,
            "elevation": {"elevation_m": 1.8, "source": "usgs_epqs"},
            "slope": {
                "slope_percent": 0.8,
                "slope_class": "flat_to_gentle",
                "samples": 5,
                "center_elevation_m": 1.8,
            },
            "floodplain": {
                "fld_zone": "AE",
                "sfha": True,
                "flood_risk_class": "sfha_high",
                "floodplain_hint": "elevated_flood_sensitivity",
                "source": "fema_nfhl",
            },
            "providers_used": ["usgs_epqs", "fema_nfhl"],
        }

    monkeypatch.setattr("catia.site_topo.fetch_site_terrain", _fake_terrain)
    monkeypatch.setattr("catia.site_hazards.fetch_site_terrain", _fake_terrain)
    # resolve_topography imports fetch_site_terrain from site_topo at call time via site_hazards
    monkeypatch.setattr(
        "catia.site_hazards.fetch_site_terrain",
        _fake_terrain,
    )
    result = assess_site_viability(
        lat=29.95,
        lon=-90.07,
        property_type="buy_land",
        include_simulation=False,
    )
    assert result["topography"]["data_quality"] in ("dem_floodplain", "dem_only", "partial_providers")
    assert result["topography"]["flood_zone"] == "AE"
    assert result["topography"]["sfha"] is True
    flood = next(h for h in result["hazard_assessment"] if h["peril"] == "flood")
    assert flood["topography_factor"] >= 1.4


def test_site_api_assess(monkeypatch):
    monkeypatch.setenv("CATIA_SITE_TOPO", "0")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from catia.api.app import app

    client = TestClient(app)
    r = client.post(
        "/api/v1/site/assess",
        json={
            "lat": 29.95,
            "lon": -90.07,
            "property_type": "buy_land",
            "include_simulation": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["risk_category"]
    assert body["location"]["nearest_region"]["region_id"] == "US_Gulf_Coast"


def test_site_api_nearest():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from catia.api.app import app

    client = TestClient(app)
    r = client.get("/api/v1/site/regions/nearest", params={"lat": 35.68, "lon": 139.69})
    assert r.status_code == 200
    assert r.json()["region_id"] == "Japan"
