"""
Structured logging configuration for CATIA.
When CATIA_STRUCTURED_LOGS=1, logs are emitted as JSON for log aggregation.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "run_id"):
            log_obj["run_id"] = record.run_id
        return json.dumps(log_obj)


def setup_structured_logging(level: str = "INFO") -> None:
    """
    Configure structured JSON logging if CATIA_STRUCTURED_LOGS env var is set.
    Otherwise uses standard format.
    """
    use_structured = os.environ.get("CATIA_STRUCTURED_LOGS", "").lower() in ("1", "true", "yes")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    if use_structured:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    root.addHandler(handler)
