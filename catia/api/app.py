"""
CATIA FastAPI Application

Main entry point for the REST API server.
Run with: uvicorn catia.api.app:app --reload
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from catia import __version__
from catia.config import LOGGING_CONFIG
from catia.api.routes import (
    router,
    perils_router,
    analysis_router,
    simulation_router,
    mitigation_router
)
from catia.api.middleware import RequestIDMiddleware
from catia.api.schemas import ErrorResponse

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
4. **Async Job**: `POST /api/v1/analysis/jobs` then `GET /api/v1/analysis/jobs/{id}` and `GET /api/v1/analysis/jobs/{id}/result`
5. **Get Mitigation**: `POST /api/v1/mitigation/optimize`
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Request ID first (so it's available in exception handlers)
app.add_middleware(RequestIDMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(
    request: Request,
    error: str,
    message: str,
    status_code: int,
    detail: Any = None,
) -> JSONResponse:
    """Build a structured error response with request ID."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            success=False,
            error=error,
            message=message,
            detail=detail,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            path=request.url.path,
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return structured error for HTTP exceptions (4xx/5xx)."""
    return _error_response(
        request,
        error="http_error",
        message=exc.detail if isinstance(exc.detail, str) else "Request failed",
        status_code=exc.status_code,
        detail=exc.detail if not isinstance(exc.detail, str) else None,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured error for validation failures (422)."""
    return _error_response(
        request,
        error="validation_error",
        message="Request validation failed",
        status_code=422,
        detail=exc.errors(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Return structured error for unhandled exceptions (500)."""
    logger.exception("Unhandled exception: %s", exc)
    return _error_response(
        request,
        error="internal_error",
        message=str(exc) or "An unexpected error occurred",
        status_code=500,
    )

# Register routers
app.include_router(router, prefix="/api/v1")
app.include_router(perils_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(mitigation_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """API root with links to docs and health."""
    return {
        "message": "Welcome to CATIA - Catastrophe AI System",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
        "ready": "/api/v1/ready",
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

