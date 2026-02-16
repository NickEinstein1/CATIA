"""
API middleware: request ID, rate limiting, and structured logging for tracing.
"""

import logging
import time
import uuid
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limit: (path_prefix, method) -> (max_requests, window_seconds)
RATE_LIMIT_RULES = [
    ("/api/v1/analysis/run", "POST"),
    ("/api/v1/analysis/jobs", "POST"),
    ("/api/v1/simulation/run", "POST"),
]
RATE_LIMIT_MAX_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

# In-memory store: client_key -> list of request timestamps (for limited paths)
_rate_limit_store: dict = defaultdict(list)


def _client_key(request: Request) -> str:
    """Client identifier for rate limiting (IP)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _is_rate_limited_path(request: Request) -> bool:
    path = request.url.path
    method = request.method.upper()
    for prefix, rule_method in RATE_LIMIT_RULES:
        if path.startswith(prefix) or path == prefix.rstrip("/"):
            if rule_method == method:
                return True
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-IP rate limit for expensive endpoints (sliding window)."""

    async def dispatch(self, request: Request, call_next):
        if not _is_rate_limited_path(request):
            return await call_next(request)

        key = _client_key(request)
        now = time.monotonic()
        window = RATE_LIMIT_WINDOW_SECONDS
        max_req = RATE_LIMIT_MAX_REQUESTS

        # Sliding window: drop timestamps outside [now - window, now]
        if key in _rate_limit_store:
            _rate_limit_store[key] = [t for t in _rate_limit_store[key] if now - t < window]
        else:
            _rate_limit_store[key] = []

        if len(_rate_limit_store[key]) >= max_req:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: max {max_req} requests per {window}s",
                    "retry_after_seconds": max(1, int(window - (now - min(_rate_limit_store[key])))),
                },
                headers={"Retry-After": str(max(1, int(window)))},
            )

        _rate_limit_store[key].append(now)
        response = await call_next(request)
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add a unique request_id to each request and response headers."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
