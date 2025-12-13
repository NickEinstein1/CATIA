"""
Pydantic schemas for CATIA API request/response models.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class PerilType(str, Enum):
    """Supported peril types."""
    HURRICANE = "hurricane"
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    EARTHQUAKE = "earthquake"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for running a CATIA analysis."""
    region: str = Field(default="US_Gulf_Coast", description="Geographic region for analysis")
    perils: List[PerilType] = Field(
        default=[PerilType.HURRICANE, PerilType.FLOOD, PerilType.WILDFIRE, PerilType.EARTHQUAKE],
        description="List of perils to analyze"
    )
    use_mock_data: bool = Field(default=True, description="Use mock data (True) or real APIs (False)")


class SimulationRequest(BaseModel):
    """Request model for financial simulation."""
    perils: List[PerilType] = Field(
        default=[PerilType.HURRICANE],
        description="Perils to simulate"
    )
    num_iterations: Optional[int] = Field(default=None, description="Monte Carlo iterations (uses config default if None)")


class MitigationRequest(BaseModel):
    """Request model for mitigation optimization."""
    baseline_loss: float = Field(..., description="Baseline annual loss in USD", gt=0)
    budget: Optional[float] = Field(default=None, description="Budget constraint (uses config default if None)")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PerilInfo(BaseModel):
    """Information about a peril type."""
    id: str
    name: str
    frequency_base: float
    severity_mu: float
    severity_sigma: float
    climate_drivers: List[str]
    seasonality: List[int]
    magnitude_scale: str
    regions: List[str]


class PerilListResponse(BaseModel):
    """Response listing all available perils."""
    perils: List[PerilInfo]
    count: int


class RiskMetrics(BaseModel):
    """Risk metrics from simulation."""
    mean: float
    median: float
    std: float
    var_95: float
    tvar_95: float


class ReturnPeriods(BaseModel):
    """Return period losses."""
    year_10: float
    year_25: float
    year_50: float
    year_100: float
    year_250: float
    year_500: float
    year_1000: float


class PerilContribution(BaseModel):
    """Per-peril contribution to total loss."""
    peril: str
    peril_name: str
    mean_loss: float
    contribution_pct: float
    var_95: float
    tvar_95: float


class SimulationResponse(BaseModel):
    """Response from financial simulation."""
    perils_analyzed: List[str]
    num_iterations: int
    aggregate_metrics: RiskMetrics
    return_periods: ReturnPeriods
    peril_contributions: List[PerilContribution]


class MitigationStrategy(BaseModel):
    """A mitigation strategy recommendation."""
    name: str
    cost: float
    risk_reduction: float
    effectiveness: float
    benefit_cost_ratio: float
    npv: float


class MitigationResponse(BaseModel):
    """Response from mitigation optimization."""
    baseline_loss: float
    mitigated_loss: float
    total_risk_reduction: float
    total_cost: float
    strategies: List[MitigationStrategy]
    priority_order: List[str]


class AnalysisResponse(BaseModel):
    """Full analysis response."""
    region: str
    perils_analyzed: List[str]
    timestamp: str
    risk_metrics: RiskMetrics
    return_periods: ReturnPeriods
    peril_contributions: List[PerilContribution]
    mitigation_summary: Dict[str, Any]
    data_summary: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str

