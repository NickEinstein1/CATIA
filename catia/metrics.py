"""
Simple metrics registry for CATIA (counters, histograms).
Optional Prometheus export format.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from catia.config import LOGGING_CONFIG


@dataclass
class Counter:
    """Counter metric."""
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class Histogram:
    """Histogram metric."""
    buckets: List[float] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def observe(self, value: float) -> None:
        self.buckets.append(value)


class MetricsRegistry:
    """
    In-memory metrics registry. Thread-safe for read-only; writes should be serialized.
    """

    def __init__(self, enabled: bool = None):
        self.enabled = enabled if enabled is not None else os.environ.get("CATIA_METRICS", "").lower() in ("1", "true", "yes")
        self._counters: Dict[str, Counter] = defaultdict(lambda: Counter())
        self._histograms: Dict[str, Histogram] = defaultdict(lambda: Histogram())

    def counter(self, name: str, labels: Dict[str, str] = None) -> Counter:
        """Get or create a counter."""
        if not self.enabled:
            return Counter()
        key = f"{name}:{sorted((labels or {}).items())}"
        if key not in self._counters:
            self._counters[key] = Counter(labels=labels or {})
        return self._counters[key]

    def histogram(self, name: str, labels: Dict[str, str] = None) -> Histogram:
        """Get or create a histogram."""
        if not self.enabled:
            return Histogram()
        key = f"{name}:{sorted((labels or {}).items())}"
        if key not in self._histograms:
            self._histograms[key] = Histogram(labels=labels or {})
        return self._histograms[key]

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, counter in self._counters.items():
            base_name = name.split(":")[0]
            label_str = ",".join(f'{k}="{v}"' for k, v in counter.labels.items())
            if label_str:
                label_str = "{" + label_str + "}"
            lines.append(f"catia_{base_name}{label_str} {counter.value}")
        for name, hist in self._histograms.items():
            base_name = name.split(":")[0]
            label_str = ",".join(f'{k}="{v}"' for k, v in hist.labels.items())
            if label_str:
                label_str = "{" + label_str + "}"
            if hist.buckets:
                count = len(hist.buckets)
                total = sum(hist.buckets)
                lines.append(f"catia_{base_name}_count{label_str} {count}")
                lines.append(f"catia_{base_name}_sum{label_str} {total}")
                lines.append(f"catia_{base_name}_avg{label_str} {total / count if count else 0}")
        return "\n".join(lines)


# Global registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the global metrics registry."""
    return _registry


def record_simulation_duration(duration_seconds: float, perils: List[str] = None) -> None:
    """Record simulation duration metric."""
    reg = get_registry()
    reg.histogram("simulation_duration_seconds", labels={"perils": ",".join(perils or [])}).observe(duration_seconds)
    reg.counter("simulation_count", labels={"perils": ",".join(perils or [])}).inc()


def record_analysis_run(region: str, perils: List[str] = None) -> None:
    """Record an analysis run."""
    reg = get_registry()
    reg.counter("analysis_runs_total", labels={"region": region, "perils": ",".join(perils or [])}).inc()
