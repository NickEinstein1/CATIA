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
    DROUGHT = "drought"


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


class ExposureRecord(BaseModel):
    """Single exposure record: region and total insured value (TIV)."""
    region: str = Field(..., description="Geographic region or location id")
    tiv: float = Field(..., gt=0, description="Total insured value (USD)")
    line_of_business: Optional[str] = Field(default=None, description="e.g. residential, commercial")
    construction_type: Optional[str] = Field(default=None, description="Construction class")
    occupancy: Optional[str] = Field(default=None, description="Occupancy type")
    peril: Optional[str] = Field(default=None, description="Peril filter if exposure is peril-specific")


class SimulationRequest(BaseModel):
    """Request model for financial simulation."""
    perils: List[PerilType] = Field(
        default=[PerilType.HURRICANE],
        description="Perils to simulate"
    )
    num_iterations: Optional[int] = Field(default=None, description="Monte Carlo iterations (uses config default if None)")
    exposure: Optional[List[ExposureRecord]] = Field(
        default=None,
        description="Optional exposure records for loss = exposure × vulnerability; if provided, simulation uses exposure-based loss",
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description="Optional climate scenario (e.g. baseline, RCP4.5_mid, SSP2_2050, high_stress)",
    )


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
    """Health check response (liveness)."""
    status: str
    version: str
    timestamp: str


class ErrorDetail(BaseModel):
    """Single validation or error detail."""
    loc: Optional[List[str]] = None
    msg: str
    type: Optional[str] = None


class ErrorResponse(BaseModel):
    """Structured error response for all API errors."""
    success: bool = False
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable message")
    detail: Optional[Any] = Field(default=None, description="Extra detail (e.g. validation errors)")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracing")
    timestamp: str = Field(..., description="ISO timestamp")
    path: Optional[str] = Field(default=None, description="Request path")


class ReadinessCheck(BaseModel):
    """Single readiness check result."""
    name: str
    status: str  # "ok" | "degraded" | "error"
    message: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Readiness check response (dependencies, disk, config)."""
    ready: bool
    version: str
    timestamp: str
    checks: List[ReadinessCheck]


# ============================================================================
# ASYNC JOB SCHEMAS (Phase C)
# ============================================================================

class JobSubmitResponse(BaseModel):
    """Response after submitting an async analysis job."""
    job_id: str
    status: str = "pending"
    message: str = "Job submitted. Poll GET /api/v1/analysis/jobs/{job_id} for status."
    created_at: str


class JobStatusResponse(BaseModel):
    """Job status (pending | running | completed | failed)."""
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    """Full result when job is completed (same shape as AnalysisResponse)."""
    job_id: str
    status: str = "completed"
    result: Optional[AnalysisResponse] = None
    error: Optional[str] = None


# ============================================================================
# STRESS SCENARIOS (Solvency-II-style)
# ============================================================================

class StressScenarioRequest(BaseModel):
    """Request for predefined stress scenarios."""
    baseline_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional baseline (mean, var_95, tvar_95, return_periods). If omitted, a quick simulation is run."
    )
    scenario_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of scenario keys (e.g. solvency_frequency_shock). If omitted, all predefined scenarios are applied."
    )
    perils: Optional[List[PerilType]] = Field(
        default=None,
        description="Used only when baseline_metrics is omitted, to run quick simulation."
    )


class StressedMetrics(BaseModel):
    """Risk metrics under one stress scenario."""
    name: str
    description: str
    mean: float
    var_95: float
    tvar_95: float
    return_periods: Dict[str, float]


class StressScenarioResponse(BaseModel):
    """Response with baseline and stressed metrics."""
    baseline: Dict[str, Any]
    scenarios: Dict[str, StressedMetrics]


# ============================================================================
# LIVE INTELLIGENCE (API-first)
# ============================================================================

class LiveProvenance(BaseModel):
    """Source lineage for a live event."""
    feed: str
    source: str
    source_event_id: str
    source_url: str = ""
    parser_version: str = ""
    ingested_at: str = ""
    observed_at: str = ""
    updated_at: str = ""


class LiveExposureOverlap(BaseModel):
    """Indicative exposure-region overlap (not modeled loss)."""
    regions: List[str] = Field(default_factory=list)
    tier_hints: List[str] = Field(default_factory=list)
    overlap_score: float = 0.0


class LiveEvent(BaseModel):
    """Normalized live catastrophe event with intelligence fields."""
    id: str
    lat: float
    lon: float
    title: str
    category: str
    category_label: str = ""
    time_iso: str = ""
    severity_label: str = ""
    source: str
    url: str = ""
    geometry: Optional[Dict[str, Any]] = None
    geometry_kind: Optional[str] = None
    geometry_collection: Optional[List[Dict[str, Any]]] = None
    provenance: Optional[LiveProvenance] = None
    confidence: Optional[float] = None
    confidence_factors: Optional[Dict[str, float]] = None
    exposure_overlap: Optional[LiveExposureOverlap] = None
    catia_peril: Optional[str] = None
    catia_score: Optional[float] = None
    model_config = {"extra": "allow"}


class LiveEventsResponse(BaseModel):
    """Live events payload for agents and external systems."""
    fetched_at_iso: str
    offline_mode: bool = False
    cache_hit: bool = False
    cache_backend: str = "memory"
    sources_ok: Dict[str, bool] = Field(default_factory=dict)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    http_status: Dict[str, Optional[int]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    count: int = 0
    geometry_summary: Optional[Dict[str, int]] = None
    disclaimer: Optional[str] = None
    events: List[LiveEvent] = Field(default_factory=list)


class LiveHealthResponse(BaseModel):
    """Live feed health strip."""
    fetched_at_iso: str
    offline_mode: bool = False
    cache_hit: bool = False
    cache_backend: str = "memory"
    sources_ok: Dict[str, bool] = Field(default_factory=dict)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    http_status: Dict[str, Optional[int]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    event_count: int = 0


class LiveGeoJsonResponse(BaseModel):
    """GeoJSON overlay for live event footprints."""
    fetched_at_iso: str
    feature_count: int
    geojson: Dict[str, Any]


# ============================================================================
# SITE VIABILITY (property / land screening)
# ============================================================================

class PropertyType(str, Enum):
    """Decision context for site assessment."""
    BUY_LAND = "buy_land"
    BUILD = "build"
    BUY_BUILDING = "buy_building"


class SiteAssessRequest(BaseModel):
    """Assess viability of buying land, building, or buying a property at a site."""
    lat: Optional[float] = Field(default=None, description="Latitude (−90…90)")
    lon: Optional[float] = Field(default=None, description="Longitude (−180…180)")
    address: Optional[str] = Field(
        default=None,
        description="Optional address (requires CATIA_SITE_GEOCODE=1 for Nominatim)",
    )
    property_type: PropertyType = Field(
        default=PropertyType.BUY_BUILDING,
        description="buy_land | build | buy_building",
    )
    construction_type: Optional[str] = Field(default=None)
    occupancy: Optional[str] = Field(default=None)
    tiv: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional total insured / development value (USD) for indicative simulation",
    )
    include_simulation: bool = Field(
        default=False,
        description="Run indicative exposure×vulnerability Monte Carlo",
    )
    scenario_id: Optional[str] = Field(default=None, description="Climate scenario id")


class SiteAssessResponse(BaseModel):
    """Site viability screening response."""
    assessed_at: str
    property_type: str
    construction_type: Optional[str] = None
    occupancy: Optional[str] = None
    scenario_id: str = "baseline"
    location: Dict[str, Any]
    risk_score: float
    risk_category: str
    score_components: Dict[str, float] = Field(default_factory=dict)
    hazard_assessment: List[Dict[str, Any]] = Field(default_factory=list)
    topography: Dict[str, Any] = Field(default_factory=dict)
    insurance_viability: Dict[str, Any] = Field(default_factory=dict)
    indicative_simulation: Optional[Dict[str, Any]] = None
    disclaimer: str
    model_config = {"extra": "allow"}

