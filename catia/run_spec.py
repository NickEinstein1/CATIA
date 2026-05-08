"""
Declarative run configuration for CATIA analyses.

Load a JSON or YAML file (see ``examples/runs/``) or construct a :class:`RunSpec`
in code. Pass to :func:`catia.pipeline.run_catia_analysis` as keyword arguments
via :meth:`RunSpec.to_kwargs` or use :func:`catia.pipeline.run_from_spec`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from catia.climate_scenarios import get_scenario_info
from catia.config import DEFAULT_PERILS, PERIL_CONFIG

KNOWN_ARTIFACTS = frozenset(
    {"report", "assumption_register", "compliance", "dashboard", "enhancements"}
)


class RunSpec(BaseModel):
    """
    Parameters for a single CATIA end-to-end analysis run.

    ``artifacts`` selects which optional outputs to write. ``None`` means all
    (default behavior). The main ``catia_report.json`` is gated by ``report``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    region: str = "US_Gulf_Coast"
    perils: List[str] = Field(default_factory=lambda: list(DEFAULT_PERILS))
    use_mock_data: bool = True
    scenario_id: Optional[str] = None
    monte_carlo_iterations: Optional[int] = Field(
        default=None,
        gt=0,
        description="Override SIMULATION_CONFIG monte_carlo_iterations for this run",
    )
    random_seed: Optional[int] = None
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory for outputs; defaults to OUTPUT_CONFIG['output_dir']",
    )
    artifacts: Optional[List[str]] = Field(
        default=None,
        description="Subset of outputs to write; None writes everything",
    )

    @field_validator("perils")
    @classmethod
    def _perils_known(cls, v: List[str]) -> List[str]:
        bad = [p for p in v if p not in PERIL_CONFIG]
        if bad:
            raise ValueError(f"Unknown perils: {bad}. Valid: {list(PERIL_CONFIG.keys())}")
        return v

    @field_validator("scenario_id")
    @classmethod
    def _scenario_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        get_scenario_info(v)
        return v

    @field_validator("artifacts")
    @classmethod
    def _artifacts_subset(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        bad = [a for a in v if a not in KNOWN_ARTIFACTS]
        if bad:
            raise ValueError(
                f"Unknown artifacts: {bad}. Valid: {sorted(KNOWN_ARTIFACTS)}"
            )
        return v

    def to_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for :func:`catia.pipeline.run_catia_analysis`."""
        return {
            "region": self.region,
            "perils": list(self.perils),
            "use_mock_data": self.use_mock_data,
            "scenario_id": self.scenario_id,
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
            "artifacts": list(self.artifacts) if self.artifacts is not None else None,
        }


def load_run_spec(path: str | Path) -> RunSpec:
    """Load a run spec from ``.json``, ``.yaml``, or ``.yml``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Run spec not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported run spec format: {suffix} (use .json, .yaml, .yml)")
    if not isinstance(data, dict):
        raise ValueError("Run spec file must contain a JSON/YAML object")
    return RunSpec.model_validate(data)
