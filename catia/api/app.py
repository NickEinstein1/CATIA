"""
CATIA FastAPI Application

Main entry point for the REST API server.
Run with: uvicorn catia.api.app:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, List, Tuple

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
from catia.api.live_routes import live_router
from catia.api.site_routes import site_router
from catia.api.middleware import RequestIDMiddleware, RateLimitMiddleware
from catia.api.schemas import ErrorResponse

# Configure logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

_DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8050",
    "http://localhost:8050",
]


def _cors_allow_origins_and_credentials() -> Tuple[List[str], bool]:
    """
    CORS policy for the API.

    - Set ``CATIA_CORS_ORIGINS`` to a comma-separated list of allowed browser origins.
    - Or set ``CATIA_CORS_ALLOW_ANY=1`` for ``*`` with ``allow_credentials=False``.
    """
    raw = os.environ.get("CATIA_CORS_ORIGINS", "").strip()
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        if not origins:
            return list(_DEFAULT_CORS_ORIGINS), True
        return origins, True
    if os.environ.get("CATIA_CORS_ALLOW_ANY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return ["*"], False
    return list(_DEFAULT_CORS_ORIGINS), True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("CATIA API starting up...")
    logger.info("Version: %s", __version__)
    o, _ = _cors_allow_origins_and_credentials()
    if o == ["*"]:
        logger.info("CORS: allow any origin (CATIA_CORS_ALLOW_ANY), credentials disabled")
    else:
        logger.info(
            "CORS: %d allowed origin(s) (set CATIA_CORS_ORIGINS to override)", len(o)
        )
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
# Rate limit expensive endpoints (order: last added = first executed after request)
app.add_middleware(RateLimitMiddleware)

_cors_origins, _cors_credentials = _cors_allow_origins_and_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
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
app.include_router(live_router, prefix="/api/v1")
app.include_router(site_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """API root with links to docs and health."""
    return {
        "message": "Welcome to CATIA - Catastrophe AI System",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
        "ready": "/api/v1/ready",
        "site_assess": "/api/v1/site/assess",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "catia.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )

