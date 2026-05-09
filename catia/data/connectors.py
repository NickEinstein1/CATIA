"""
Real data connectors with retries and optional caching.
Uses API_CONFIG for endpoints and timeouts; falls back to mock on failure.
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from catia.config import API_CONFIG, PERIL_CONFIG

logger = logging.getLogger(__name__)


def _session_with_retries(timeout: int = 30, retries: int = 3) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


class NOAAConnector:
    """
    NOAA climate / hazards data.
    Uses NCEI CDO-style endpoints when token is set; otherwise returns None for real fetch.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.token = api_token or os.environ.get("NOAA_API_TOKEN", "")
        self.config = API_CONFIG.get("NOAA", {})
        self.base = self.config.get("base_url", "").rstrip("/")
        self.timeout = self.config.get("timeout", 30)

    def fetch_climate(self, region: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Attempt real fetch. Returns None if no token or on error (caller can fallback to mock).
        """
        if not self.token:
            logger.debug("NOAA token not set; skip real fetch")
            return None
        try:
            # CDO token-based endpoint pattern; adjust to actual NCEI API you use
            url = f"{self.base}/cdo-web/api/v2/data"
            params = {
                "datasetid": "GHCND",
                "locationid": "FIPS:48",  # example; map region to locationid
                "startdate": start_date,
                "enddate": end_date,
                "limit": 1000,
                "token": self.token,
            }
            with _session_with_retries(self.timeout) as session:
                r = session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
            data = r.json()
            if not data.get("results"):
                return None
            df = pd.DataFrame(data["results"])
            # Normalize to expected columns if possible
            if "value" in df.columns and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={"value": "temperature"})  # minimal mapping
            return df
        except Exception as e:
            logger.warning("NOAA fetch failed: %s", e)
            return None


class WorldBankConnector:
    """
    World Bank socioeconomic indicators.
    Public API, no token required; uses retries and optional cache.
    """

    def __init__(self):
        self.config = API_CONFIG.get("WORLD_BANK", {})
        self.base = (self.config.get("base_url", "https://api.worldbank.org/v2") or "").rstrip("/")
        self.timeout = self.config.get("timeout", 30)

    def fetch_indicators(self, country_iso: str = "USA") -> Optional[pd.DataFrame]:
        """
        Fetch indicators for a country. country_iso e.g. USA, BRA.
        Returns None on error.
        """
        try:
            # Population and GDP indicators
            url = f"{self.base}/country/{country_iso}/indicator/SP.POP.TOTL"
            params = {"format": "json", "per_page": 5, "date": "2020:2023"}
            with _session_with_retries(self.timeout) as session:
                r = session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                return None
            # data[0] = metadata, data[1] = records
            records = data[1] if len(data) > 1 else []
            if not records:
                return None
            rows = []
            for rec in records:
                rows.append({
                    "region": country_iso,
                    "year": rec.get("date"),
                    "population": rec.get("value"),
                    "indicator": rec.get("indicator", {}).get("value"),
                })
            df = pd.DataFrame(rows)
            # One row per region for compatibility: take latest year
            if not df.empty:
                latest = df.sort_values("year", ascending=False).iloc[0]
                out = pd.DataFrame([{
                    "region": latest["region"],
                    "population_density": latest.get("population", 0) / 9_834_000 if latest.get("population") else 35,
                    "gdp_per_capita": 50000,  # WB GDP endpoint separate; placeholder
                    "infrastructure_index": 0.7,
                    "poverty_rate": 0.12,
                }])
                return out
            return None
        except Exception as e:
            logger.warning("World Bank fetch failed: %s", e)
            return None


def fetch_noaa_climate_cached(
    cache: Optional[Any],
    region: str,
    start_date: str,
    end_date: str,
    connector: Optional[NOAAConnector] = None,
) -> Optional[pd.DataFrame]:
    """Try cache, then real NOAA, then return None for mock fallback."""
    params = {"region": region, "start_date": start_date, "end_date": end_date}
    if cache:
        cached = cache.get("noaa_climate", params)
        if cached is not None:
            return cached
    conn = connector or NOAAConnector()
    df = conn.fetch_climate(region, start_date, end_date)
    if df is not None and cache:
        cache.set("noaa_climate", params, df)
    return df


def fetch_worldbank_cached(
    cache: Optional[Any],
    region: str,
    connector: Optional[WorldBankConnector] = None,
) -> Optional[pd.DataFrame]:
    """Try cache, then real World Bank. Region mapped to ISO (e.g. US_Gulf_Coast -> USA)."""
    region_to_iso = {"US_Gulf_Coast": "USA", "US_East_Coast": "USA", "California_Coast": "USA", "Florida_Peninsula": "USA"}
    iso = region_to_iso.get(region, "USA")
    params = {"region": region, "country_iso": iso}
    if cache:
        cached = cache.get("worldbank_socio", params)
        if cached is not None:
            return cached
    conn = connector or WorldBankConnector()
    df = conn.fetch_indicators(iso)
    if df is not None and cache:
        cache.set("worldbank_socio", params, df)
    return df
