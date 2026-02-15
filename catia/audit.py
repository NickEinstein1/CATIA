"""
Audit and reproducibility support for CATIA.

Provides run IDs and config snapshots so every analysis can be reproduced
and traced back to exact code version, config, and (optionally) data.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict

from catia import __version__
from catia.config import (
    SIMULATION_CONFIG,
    RISK_METRICS,
    ML_CONFIG,
    PERIL_CONFIG,
    get_config,
)

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    """Generate a unique run identifier (UUID4 + short hash)."""
    u = str(uuid.uuid4())
    short = hashlib.sha256(u.encode()).hexdigest()[:8]
    return f"catia-{datetime.now().strftime('%Y%m%d')}-{short}"


def get_config_snapshot(
    *,
    include_perils: bool = True,
    include_ml: bool = True,
    include_simulation: bool = True,
    include_risk_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Build a snapshot of current config for reproducibility.

    Excludes sensitive or volatile keys. Use this in reports and logs.
    """
    snapshot = {
        "catia_version": __version__,
        "snapshot_at": datetime.now().isoformat(),
    }
    if include_simulation:
        snapshot["simulation"] = dict(SIMULATION_CONFIG)
        # Ensure seed is present for reproducibility
        if "random_seed" not in snapshot["simulation"]:
            snapshot["simulation"]["random_seed"] = 42
    if include_risk_metrics:
        snapshot["risk_metrics"] = dict(RISK_METRICS)
    if include_ml:
        snapshot["ml"] = {
            "model_type": ML_CONFIG.get("model_type"),
            "train_test_split": ML_CONFIG.get("train_test_split"),
            "random_state": ML_CONFIG.get("hyperparameters", {}).get("random_state"),
        }
    if include_perils:
        snapshot["perils"] = {
            p: {
                "frequency_base": c.get("frequency_base"),
                "severity_params": c.get("severity_params"),
            }
            for p, c in PERIL_CONFIG.items()
        }
    return snapshot


def get_config_snapshot_hash(snapshot: Dict[str, Any]) -> str:
    """Stable hash of config snapshot for change detection."""
    canonical = json.dumps(snapshot, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def create_audit_metadata(
    run_id: str,
    region: str,
    perils: list,
    use_mock_data: bool,
) -> Dict[str, Any]:
    """
    Create audit metadata to attach to reports and logs.

    Includes run_id, config snapshot, and config hash.
    """
    snapshot = get_config_snapshot()
    config_hash = get_config_snapshot_hash(snapshot)
    return {
        "run_id": run_id,
        "region": region,
        "perils": perils,
        "use_mock_data": use_mock_data,
        "config_snapshot": snapshot,
        "config_hash": config_hash,
        "timestamp": datetime.now().isoformat(),
    }
