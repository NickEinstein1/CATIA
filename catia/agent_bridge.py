"""
Bridge facades from the interactive agent to CATIA internals.

The repository does not ship modules literally named ``RiskAnalysis`` or
``ActuarialScience``; these classes are **stable agent-facing entry points** that
delegate to:

- **RiskAnalysis** → :mod:`catia.data_acquisition`, :mod:`catia.risk_prediction`
- **ActuarialScience** → :mod:`catia.financial_impact` (Monte Carlo / multi-peril)

Use them from the REPL or other tools instead of importing pipeline internals
directly when you want a clear separation of “ML risk” vs “actuarial simulation”.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from catia.config import DEFAULT_PERILS
from catia.data_acquisition import fetch_all_data
from catia.financial_impact import run_multi_peril_analysis
from catia.risk_prediction import train_risk_model


@dataclass
class RiskAnalysisResult:
    """Outcome of a risk-analysis (data + model) step."""

    region: str
    perils: List[str]
    use_mock_data: bool
    data: Dict[str, Any]
    model_summary: Dict[str, Any]


class RiskAnalysis:
    """
    Agent-facing **risk / ML** workflow: ingest data and train the risk model.

    Wraps ``fetch_all_data`` and ``train_risk_model``.
    """

    def acquire(
        self,
        region: str,
        *,
        use_mock_data: bool = True,
        perils: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        perils = perils or list(DEFAULT_PERILS)
        return fetch_all_data(region, use_mock=use_mock_data, perils=perils)

    def train(self, data_bundle: Dict[str, Any]) -> Dict[str, Any]:
        predictor = train_risk_model(
            data_bundle["climate"],
            data_bundle["socioeconomic"],
            data_bundle["historical_events"],
        )
        return {
            "probability_model": type(predictor.probability_model).__name__
            if getattr(predictor, "probability_model", None) is not None
            else None,
            "severity_model": type(predictor.severity_model).__name__
            if getattr(predictor, "severity_model", None) is not None
            else None,
            "feature_names": getattr(predictor, "feature_names", None) or [],
            "train_status": "ok",
        }

    def run(
        self,
        region: str,
        *,
        use_mock_data: bool = True,
        perils: Optional[List[str]] = None,
    ) -> RiskAnalysisResult:
        """Fetch data and train; return structured result for the agent UI."""
        perils = perils or list(DEFAULT_PERILS)
        data = self.acquire(region, use_mock_data=use_mock_data, perils=perils)
        summary = self.train(data)
        return RiskAnalysisResult(
            region=region,
            perils=perils,
            use_mock_data=use_mock_data,
            data=data,
            model_summary=summary,
        )


@dataclass
class ActuarialResult:
    """Outcome of an actuarial / simulation step."""

    perils: List[str]
    aggregate_metrics: Dict[str, Any]
    raw: Dict[str, Any]


class ActuarialScience:
    """
    Agent-facing **actuarial** workflow: Monte Carlo / multi-peril simulation.

    Wraps :func:`catia.financial_impact.run_multi_peril_analysis`.
    """

    def multi_peril(
        self,
        perils: Optional[List[str]] = None,
        *,
        include_uncertainty: bool = True,
        include_correlation: bool = True,
        scenario_id: Optional[str] = None,
        num_iterations: Optional[int] = None,
        n_bootstrap: int = 200,
    ) -> ActuarialResult:
        perils = perils or list(DEFAULT_PERILS)
        raw = run_multi_peril_analysis(
            perils,
            include_uncertainty=include_uncertainty,
            include_correlation=include_correlation,
            scenario_id=scenario_id,
            num_iterations=num_iterations,
            n_bootstrap=n_bootstrap,
        )
        return ActuarialResult(
            perils=perils,
            aggregate_metrics=raw["aggregate_metrics"],
            raw=raw,
        )
