"""
Model registry: versioned storage of risk model paths and metadata.
Minimal implementation: JSON-backed list of versions with path, timestamp, metrics.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from catia.config import ML_CONFIG

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry of trained model versions. Each entry: version_id, path, created_at, metadata.
    """

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or ML_CONFIG.get("registry_path", "models/registry.json"))
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    self._entries = json.load(f)
                if not isinstance(self._entries, list):
                    self._entries = []
            except Exception as e:
                logger.warning("Registry load failed: %s", e)
                self._entries = []

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def register(
        self,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        version_id: Optional[str] = None,
    ) -> str:
        """
        Register a model. Returns version_id (e.g. v_20260212_143022 or provided).
        """
        version_id = version_id or f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        entry = {
            "version_id": version_id,
            "path": os.path.abspath(model_path),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._entries.append(entry)
        self._save()
        logger.info("Registered model %s at %s", version_id, entry["path"])
        return version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """Return all entries, newest last."""
        return list(self._entries)

    def get(self, version_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get entry by version_id; if None, return latest."""
        if not self._entries:
            return None
        if version_id:
            for e in self._entries:
                if e.get("version_id") == version_id:
                    return e
            return None
        return self._entries[-1]

    def get_path(self, version_id: Optional[str] = None) -> Optional[str]:
        """Return model path for version, or latest."""
        entry = self.get(version_id)
        return entry.get("path") if entry else None

    def load_latest_path(self) -> Optional[str]:
        """Convenience: path of latest registered model."""
        return self.get_path(None)


def get_registry(registry_path: Optional[str] = None) -> ModelRegistry:
    """Return a ModelRegistry instance using config or provided path."""
    return ModelRegistry(registry_path or ML_CONFIG.get("registry_path"))
