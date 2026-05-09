"""Tests for persistent job store."""

import os
import tempfile
import pytest

# Use a temp dir for job storage so we don't pollute outputs/jobs
@pytest.fixture
def jobs_dir(tmp_path):
    import catia.api.jobs as jobs_mod
    import catia.config
    old = catia.config.OUTPUT_CONFIG.get("jobs_dir")
    catia.config.OUTPUT_CONFIG["jobs_dir"] = str(tmp_path)
    yield str(tmp_path)
    catia.config.OUTPUT_CONFIG["jobs_dir"] = old
    # Clear in-memory store so next test starts clean
    jobs_mod._store.clear()


def test_create_job(jobs_dir):
    from catia.api.jobs import create_job, get_job
    job_id = create_job()
    assert job_id
    job = get_job(job_id)
    assert job["status"] == "pending"
    assert job["job_id"] == job_id
    assert os.path.isfile(os.path.join(jobs_dir, f"{job_id}.json"))


def test_job_persistence_after_restart(jobs_dir):
    """Simulate restart: after set_job_result, clear in-memory store and get_job from disk."""
    from catia.api import jobs as jobs_mod
    from catia.api.jobs import create_job, set_job_result, get_job, get_job_result
    job_id = create_job()
    result = {"mean": 1e6, "var_95": 2e6}
    set_job_result(job_id, result)
    # Simulate process restart: in-memory store is empty
    jobs_mod._store.clear()
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["result"] == result
    assert get_job_result(job_id) == result


def test_job_error_persistence(jobs_dir):
    from catia.api import jobs as jobs_mod
    from catia.api.jobs import create_job, set_job_error, get_job
    job_id = create_job()
    set_job_error(job_id, "Simulated failure")
    jobs_mod._store.clear()
    job = get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "Simulated failure"
    assert job["result"] is None
