"""Production data policy and live-first acquisition."""

from __future__ import annotations

import pytest

from catia.data_acquisition import DataAcquisition, fetch_all_data
from catia.data_policy import DataUnavailableError, use_mock_by_default
from catia.run_spec import RunSpec, merge_cli_run_spec


def test_live_is_default_policy(monkeypatch):
    monkeypatch.delenv("CATIA_USE_MOCK_DATA", raising=False)
    assert use_mock_by_default() is False


def test_mock_env_forces_mock_default(monkeypatch):
    monkeypatch.setenv("CATIA_USE_MOCK_DATA", "1")
    assert use_mock_by_default() is True


def test_run_spec_defaults_to_live():
    assert RunSpec().use_mock_data is False


def test_merge_cli_mock_data_flag():
    s = merge_cli_run_spec(mock_data=True)
    assert s.use_mock_data is True
    s2 = merge_cli_run_spec(no_mock_data=True)
    assert s2.use_mock_data is False


def test_explicit_mock_still_works():
    data = fetch_all_data("US_Gulf_Coast", use_mock=True, perils=["hurricane"])
    assert data["provenance"]["data_mode"] == "mock"
    assert len(data["climate"]) > 0
    assert data["provenance"]["sources"]["climate"] == "mock"


def test_live_mode_raises_without_fallback(monkeypatch):
    monkeypatch.setenv("CATIA_ALLOW_MOCK_FALLBACK", "0")

    class Boom(DataAcquisition):
        def fetch_climate_data(self, region, start_date, end_date):
            self.use_mock_data = False
            return self._fail_or_mock(
                "climate",
                lambda: self._generate_mock_climate_data(region, start_date, end_date),
            )

    da = Boom(use_mock_data=False)
    with pytest.raises(DataUnavailableError):
        da.fetch_climate_data("US_Gulf_Coast", "2020-01-01", "2020-01-10")


@pytest.mark.integration
def test_open_meteo_live_climate():
    """Requires network — skip when CATIA_SKIP_LIVE_NET=1 or DNS/network unavailable."""
    import os

    from catia.data_policy import DataUnavailableError

    if os.environ.get("CATIA_SKIP_LIVE_NET", "").strip().lower() in ("1", "true", "yes"):
        pytest.skip("CATIA_SKIP_LIVE_NET set")
    da = DataAcquisition(use_mock_data=False)
    try:
        df = da.fetch_climate_data("US_Gulf_Coast", "2022-01-01", "2022-03-31")
    except DataUnavailableError as e:
        pytest.skip(f"live climate unreachable: {e}")
    assert len(df) > 30
    assert da.provenance()["sources"]["climate"] in ("open-meteo", "noaa", "live_climate")
    assert "temperature" in df.columns
