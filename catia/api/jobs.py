"""
Persistent job store for async CATIA analysis.
Jobs are saved to disk so they survive process restarts.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache: job_id -> {status, created_at, completed_at?, result?, error?}
_store: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _jobs_dir() -> str:
    """Directory for job files (from config)."""
    try:
        from catia.config import OUTPUT_CONFIG
        return OUTPUT_CONFIG.get("jobs_dir", "outputs/jobs")
    except Exception:
        return "outputs/jobs"


def _path(job_id: str) -> str:
    """Path to job metadata/state file."""
    d = _jobs_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{job_id}.json")


def _load(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job state from disk."""
    p = _path(job_id)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load job %s: %s", job_id, e)
        return None


def _save(job_id: str, data: Dict[str, Any]) -> None:
    """Persist job state to disk. Result is stored inline (omit very large blobs if needed)."""
    p = _path(job_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    # Make JSON-serializable
    out = {
        "job_id": data.get("job_id"),
        "status": data.get("status"),
        "created_at": data.get("created_at"),
        "completed_at": data.get("completed_at"),
        "error": data.get("error"),
        "result": data.get("result"),
    }
    try:
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Failed to save job %s: %s", job_id, e)


def create_job() -> str:
    """Create a new job and return its id. State is persisted to disk."""
    job_id = str(uuid.uuid4())
    data = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }
    with _lock:
        _store[job_id] = data.copy()
    _save(job_id, data)
    return job_id


def set_job_running(job_id: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "running"
            _save(job_id, _store[job_id])
        else:
            data = _load(job_id)
            if data:
                data["status"] = "running"
                _store[job_id] = data
                _save(job_id, data)


def set_job_result(job_id: str, result: Dict[str, Any]) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "completed"
            _store[job_id]["completed_at"] = datetime.now().isoformat()
            _store[job_id]["result"] = result
            _store[job_id]["error"] = None
            _save(job_id, _store[job_id])
        else:
            data = _load(job_id) or {
                "job_id": job_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
            data["status"] = "completed"
            data["completed_at"] = datetime.now().isoformat()
            data["result"] = result
            data["error"] = None
            _store[job_id] = data
            _save(job_id, data)


def set_job_error(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "failed"
            _store[job_id]["completed_at"] = datetime.now().isoformat()
            _store[job_id]["result"] = None
            _store[job_id]["error"] = error
            _save(job_id, _store[job_id])
        else:
            data = _load(job_id) or {
                "job_id": job_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "completed_at": None,
                "result": None,
                "error": None,
            }
            data["status"] = "failed"
            data["completed_at"] = datetime.now().isoformat()
            data["result"] = None
            data["error"] = error
            _store[job_id] = data
            _save(job_id, data)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return job state; load from disk if not in memory (e.g. after restart)."""
    with _lock:
        if job_id in _store:
            return _store[job_id].copy()
        data = _load(job_id)
        if data:
            _store[job_id] = data
            return data.copy()
    return None


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Return result only if status is completed."""
    job = get_job(job_id)
    if job and job.get("status") == "completed" and job.get("result") is not None:
        return job["result"]
    return None
