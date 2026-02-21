"""
Exposure schema and store for CATIA.

Defines exposure (locations/regions): total insured value (TIV), line of business,
construction/occupancy. Enables loss = f(exposure, hazard, vulnerability).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Default columns for exposure data
EXPOSURE_REQUIRED_COLUMNS = ["region", "tiv"]
EXPOSURE_OPTIONAL_COLUMNS = ["line_of_business", "construction_type", "occupancy", "peril"]


def _normalize_region(r: Any) -> str:
    """Normalize region to string."""
    return str(r).strip() if r is not None else ""


def validate_exposure_row(row: Dict[str, Any]) -> None:
    """Raise ValueError if required fields are missing or invalid."""
    region = _normalize_region(row.get("region"))
    if not region:
        raise ValueError("Exposure row missing 'region'")
    tiv = row.get("tiv")
    if tiv is None:
        raise ValueError("Exposure row missing 'tiv'")
    try:
        tiv_f = float(tiv)
    except (TypeError, ValueError):
        raise ValueError("Exposure 'tiv' must be numeric")
    if tiv_f <= 0:
        raise ValueError("Exposure 'tiv' must be positive")


class ExposureStore:
    """
    In-memory exposure store: regions and total insured values (TIV).

    Optionally supports line_of_business, construction_type, occupancy, peril.
    Load from DataFrame, CSV, or JSON; query by region or peril.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def add_record(
        self,
        region: str,
        tiv: float,
        line_of_business: Optional[str] = None,
        construction_type: Optional[str] = None,
        occupancy: Optional[str] = None,
        peril: Optional[str] = None,
    ) -> None:
        """Add a single exposure record. Validates required fields."""
        row = {
            "region": _normalize_region(region),
            "tiv": float(tiv),
            "line_of_business": line_of_business,
            "construction_type": construction_type,
            "occupancy": occupancy,
            "peril": peril,
        }
        validate_exposure_row(row)
        self._records.append(row)

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load exposure from a DataFrame.

        Required columns: region, tiv.
        Optional: line_of_business, construction_type, occupancy, peril.
        """
        for col in EXPOSURE_REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Exposure DataFrame missing required column: {col}")
        self._records.clear()
        for _, r in df.iterrows():
            row = {"region": _normalize_region(r["region"]), "tiv": float(r["tiv"])}
            for c in EXPOSURE_OPTIONAL_COLUMNS:
                if c in df.columns and pd.notna(r.get(c)):
                    row[c] = str(r[c]).strip()
                else:
                    row[c] = None
            validate_exposure_row(row)
            self._records.append(row)
        logger.info("ExposureStore loaded %s records from DataFrame", len(self._records))

    def load_from_csv(self, path: Union[str, Path]) -> None:
        """Load exposure from a CSV file (required columns: region, tiv)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Exposure file not found: {path}")
        df = pd.read_csv(path)
        self.load_from_dataframe(df)

    def load_from_json(self, path: Union[str, Path]) -> None:
        """Load exposure from a JSON file (array of objects or path to key with array)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Exposure file not found: {path}")
        import json
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("exposure", data.get("records", []))
            if not isinstance(records, list):
                raise ValueError("JSON must contain an array or key 'exposure'/'records' with array")
        else:
            raise ValueError("JSON must be an array or object with exposure/records array")
        self._records.clear()
        for row in records:
            r = dict(row)
            validate_exposure_row(r)
            self._records.append({
                "region": _normalize_region(r["region"]),
                "tiv": float(r["tiv"]),
                "line_of_business": r.get("line_of_business"),
                "construction_type": r.get("construction_type"),
                "occupancy": r.get("occupancy"),
                "peril": r.get("peril"),
            })
        logger.info("ExposureStore loaded %s records from JSON", len(self._records))

    def get_total_tiv(self, region: Optional[str] = None, peril: Optional[str] = None) -> float:
        """Total TIV across all records, optionally filtered by region and/or peril."""
        total = 0.0
        for r in self._records:
            if region is not None and r["region"] != region:
                continue
            if peril is not None and r.get("peril") is not None and r["peril"] != peril:
                continue
            total += r["tiv"]
        return total

    def get_by_region(self, region: str) -> List[Dict[str, Any]]:
        """All records for a given region."""
        return [r for r in self._records if r["region"] == region]

    def to_dataframe(self) -> pd.DataFrame:
        """Export records as a DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=EXPOSURE_REQUIRED_COLUMNS + EXPOSURE_OPTIONAL_COLUMNS)
        return pd.DataFrame(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[Dict[str, Any]]:
        """Return a copy of all records."""
        return [dict(r) for r in self._records]
