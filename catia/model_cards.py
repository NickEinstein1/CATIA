"""
Model cards: per-model summary (intent, data, metrics, limitations).

Provides transparency and auditability for risk models. Generate on train/register
and attach to the model registry.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from catia import __version__

logger = logging.getLogger(__name__)


def build_model_card(
    model_type: str,
    version_id: str,
    *,
    intent: Optional[str] = None,
    training_data_summary: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    limitations: Optional[List[str]] = None,
    training_date: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a model card dict (schema aligned with responsible ML practice).

    Use for documentation and compliance; can be written alongside the model.
    """
    card = {
        "schema_version": "1.0",
        "model_card_version": "1.0",
        "catia_version": __version__,
        "model_type": model_type,
        "version_id": version_id,
        "created_at": training_date or datetime.now().isoformat(),
        "intent": intent or "Catastrophe risk prediction: event probability and severity from climate and socioeconomic features.",
        "training_data_summary": training_data_summary or {
            "description": "Climate aggregates, socioeconomic features, historical event labels.",
            "source": "Data acquisition module (NOAA/ECMWF/World Bank or mock).",
            "notes": "Customize per run if using real data.",
        },
        "metrics": metrics or {},
        "limitations": limitations or [
            "Trained on historical and/or synthetic data; may not capture unprecedented events.",
            "Performance depends on feature quality and region coverage.",
            "Not a substitute for actuarial review or regulatory approval.",
        ],
        "hyperparameters": hyperparameters or {},
    }
    return card


def write_model_card(
    path: str | Path,
    model_type: str,
    version_id: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build model card and write to JSON. Returns the card dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    card = build_model_card(model_type, version_id, **kwargs)
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card written to %s", path)
    return card


def get_model_card_path(model_path: str, registry_dir: Optional[str] = None) -> Path:
    """Suggested path for model card: same dir as model, name model_card_<version>.json."""
    base = Path(model_path).parent
    stem = Path(model_path).stem
    return base / f"model_card_{stem}.json"
