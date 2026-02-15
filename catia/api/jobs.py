"""
In-memory job store for async CATIA analysis.
Submit long-running analyses and poll for status/result.
"""

import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory store: job_id -> {status, created_at, completed_at?, result?, error?}
_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def create_job() -> str:
    """Create a new job and return its id."""
    job_id = str(uuid.uuid4())
    with _lock:
        _store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def set_job_running(job_id: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "running"


def set_job_result(job_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "completed"
            _store[job_id]["completed_at"] = datetime.now().isoformat()
            _store[job_id]["result"] = result
            _store[job_id]["error"] = None


def set_job_error(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "failed"
            _store[job_id]["completed_at"] = datetime.now().isoformat()
            _store[job_id]["result"] = None
            _store[job_id]["error"] = error


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _store.get(job_id)


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Return result only if status is completed."""
    job = get_job(job_id)
    if job and job.get("status") == "completed" and job.get("result") is not None:
        return job["result"]
    return None
