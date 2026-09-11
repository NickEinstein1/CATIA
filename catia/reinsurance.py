"""
Simple excess-of-loss (XL) reinsurance layers for portfolio metrics.

Each layer recovers: share × min(limit, max(0, gross − attachment)).
Layers apply independently on the same gross loss (non-overlapping tower optional
by choosing non-overlapping attachments). Net = gross − total recovered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

LayerLike = Union[Dict[str, Any], "ReinsuranceLayer"]


@dataclass
class ReinsuranceLayer:
    attachment: float
    limit: float
    share: float = 1.0
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.attachment = float(self.attachment)
        self.limit = float(self.limit)
        self.share = float(self.share)
        if self.attachment < 0:
            raise ValueError("attachment must be >= 0")
        if self.limit <= 0:
            raise ValueError("limit must be > 0")
        if not (0.0 < self.share <= 1.0):
            raise ValueError("share must be in (0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_layers(raw: Optional[Sequence[LayerLike]]) -> List[ReinsuranceLayer]:
    if not raw:
        return []
    out: List[ReinsuranceLayer] = []
    for i, item in enumerate(raw):
        if isinstance(item, ReinsuranceLayer):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Layer {i}: expected object")
        out.append(
            ReinsuranceLayer(
                attachment=float(item.get("attachment") or 0),
                limit=float(item.get("limit") or 0),
                share=float(item.get("share") if item.get("share") is not None else 1.0),
                name=str(item["name"]) if item.get("name") else f"XL-{i + 1}",
            )
        )
    if len(out) > 10:
        raise ValueError("At most 10 reinsurance layers allowed")
    return out


def layer_recovery(gross: float, layer: ReinsuranceLayer) -> float:
    g = float(gross)
    excess = max(0.0, g - layer.attachment)
    return layer.share * min(layer.limit, excess)


def apply_layers_to_loss(
    gross: float,
    layers: Sequence[LayerLike],
) -> Dict[str, Any]:
    """Apply XL layers to a single gross loss (e.g. one-storm total)."""
    parsed = parse_layers(layers)
    g = float(gross)
    by_layer: List[Dict[str, Any]] = []
    recovered = 0.0
    for layer in parsed:
        rec = layer_recovery(g, layer)
        recovered += rec
        by_layer.append({**layer.to_dict(), "recovered": round(rec, 2)})
    net = max(0.0, g - recovered)
    return {
        "gross_loss": round(g, 2),
        "recovered": round(recovered, 2),
        "net_loss": round(net, 2),
        "ceded_ratio": round(recovered / g, 4) if g > 0 else 0.0,
        "layers": by_layer,
        "note": (
            "Indicative XL recoveries on modeled gross — not binding treaty settlement."
        ),
    }


def apply_layers_to_losses(
    gross_losses: np.ndarray,
    layers: Sequence[LayerLike],
) -> Dict[str, Any]:
    """Vectorized XL application for Monte Carlo aggregate samples."""
    parsed = parse_layers(layers)
    g = np.asarray(gross_losses, dtype=float)
    recovered = np.zeros_like(g)
    layer_means: List[Dict[str, Any]] = []
    for layer in parsed:
        excess = np.maximum(0.0, g - layer.attachment)
        rec = layer.share * np.minimum(layer.limit, excess)
        recovered += rec
        layer_means.append(
            {
                **layer.to_dict(),
                "mean_recovered": float(np.mean(rec)),
            }
        )
    net = np.maximum(0.0, g - recovered)
    return {
        "gross_losses": g,
        "net_losses": net,
        "recovered_losses": recovered,
        "layer_means": layer_means,
        "mean_gross": float(np.mean(g)),
        "mean_recovered": float(np.mean(recovered)),
        "mean_net": float(np.mean(net)),
    }
