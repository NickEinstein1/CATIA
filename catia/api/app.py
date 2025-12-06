"""
CATIA FastAPI Application

Main entry point for the REST API server.
Run with: uvicorn catia.api.app:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from catia import __version__
from catia.config import LOGGING_CONFIG
from catia.api.routes import (
    router,
    perils_router,
    analysis_router,
    simulation_router,
    mitigation_router
)

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("CATIA API starting up...")
    logger.info(f"Version: {__version__}")
    yield
    logger.info("CATIA API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="CATIA - Catastrophe AI System",
    description="""
## Climate Risk Modeling REST API

CATIA provides comprehensive catastrophe modeling capabilities including:

* **Multi-Peril Analysis** - Hurricane, Flood, Wildfire, Earthquake
* **Monte Carlo Simulation** - Financial impact modeling with 10,000+ iterations
* **Risk Metrics** - VaR, TVaR, Return Periods, Loss Exceedance Curves
* **Mitigation Optimization** - Cost-benefit analysis of risk reduction strategies

### Quick Start

1. **List Perils**: `GET /api/v1/perils/`
2. **Run Simulation**: `POST /api/v1/simulation/run`
3. **Full Analysis**: `POST /api/v1/analysis/run`
4. **Get Mitigation**: `POST /api/v1/mitigation/optimize`
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(router, prefix="/api/v1")
app.include_router(perils_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(mitigation_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """API root - redirect to docs."""
    return {
        "message": "Welcome to CATIA - Catastrophe AI System",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "catia.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

