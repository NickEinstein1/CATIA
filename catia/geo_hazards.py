"""
Geographic centroids for CATIA dashboard globe overlays.

Maps logical regions (PERIL_CONFIG labels) to approximate lat/lon for visualization.
Not for underwriting — display only.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go

from catia.config import PERIL_CONFIG
from catia.geo_regions import REGION_CENTROIDS

PERIL_VIS_COLORS: Dict[str, str] = {
    "hurricane": "#22d3ee",
    "flood": "#38bdf8",
    "wildfire": "#fb923c",
    "earthquake": "#c084fc",
    "drought": "#facc15",
}


def _resolve_peril_key(row: Dict[str, Any]) -> Optional[str]:
    pid = row.get("peril")
    if isinstance(pid, str) and pid in PERIL_CONFIG:
        return pid
    name = row.get("peril_name") or ""
    for k, cfg in PERIL_CONFIG.items():
        if cfg.get("name") == name:
            return k
    return None


def aggregate_region_incidents(report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build incident points for the globe: one row per region with lat, lon, intensity, peril, label.
    """
    region_loss: Dict[str, float] = defaultdict(float)
    region_peril_parts: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    if report and report.get("multi_peril_contributions"):
        for row in report["multi_peril_contributions"]:
            pid = _resolve_peril_key(row)
            if not pid:
                continue
            loss = float(row.get("mean_loss", 0))
            regs = PERIL_CONFIG[pid].get("regions") or []
            if not regs:
                continue
            share = loss / len(regs)
            for r in regs:
                region_loss[r] += share
                region_peril_parts[r].append((pid, share))
    else:
        for pid, cfg in PERIL_CONFIG.items():
            w = float(cfg.get("frequency_base", 0.5))
            regs = cfg.get("regions") or []
            if not regs:
                continue
            sh = w / len(regs)
            for r in regs:
                region_loss[r] += sh
                region_peril_parts[r].append((pid, sh))

    out: List[Dict[str, Any]] = []
    mx = max(region_loss.values()) if region_loss else 1.0

    for rid, val in sorted(region_loss.items(), key=lambda x: -x[1]):
        if rid not in REGION_CENTROIDS:
            continue
        lat, lon = REGION_CENTROIDS[rid]
        dom = max(region_peril_parts[rid], key=lambda x: x[1])[0]
        out.append({
            "region_id": rid,
            "lat": lat,
            "lon": lon,
            "intensity": val,
            "intensity_norm": val / mx if mx else 0,
            "dominant_peril": dom,
            "hover": (
                f"<b>{rid.replace('_', ' ')}</b><br>"
                f"Relative index: {100 * val / mx:.1f}%<br>"
                f"Dominant peril: {PERIL_CONFIG.get(dom, {}).get('name', dom)}<br>"
                f"Model value: ${val:,.0f}"
            ),
        })
    return out


def fig_global_hazard_globe(
    report: Optional[Dict[str, Any]] = None,
    *,
    focal_region: Optional[str] = None,
    rotation_lon: Optional[float] = None,
) -> go.Figure:
    """
    Orthographic (globe) view with incident markers sized by modeled risk exposure.
    """
    incidents = aggregate_region_incidents(report)
    traces: List[Any] = []

    if incidents:
        lats = [i["lat"] for i in incidents]
        lons = [i["lon"] for i in incidents]
        texts = [i["hover"] for i in incidents]
        colors = [PERIL_VIS_COLORS.get(i["dominant_peril"], "#94a3b8") for i in incidents]
        sizes = [12 + 38 * (i["intensity_norm"] ** 0.5) for i in incidents]

        traces.append(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                text=texts,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.92,
                    line=dict(width=1.5, color="rgba(255,255,255,0.85)"),
                ),
                name="Hazard exposure",
            )
        )

    if focal_region and focal_region in REGION_CENTROIDS:
        flat, flon = REGION_CENTROIDS[focal_region]
        traces.append(
            go.Scattergeo(
                lon=[flon],
                lat=[flat],
                text=[f"<b>Focal region</b><br>{focal_region.replace('_', ' ')}"],
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    size=24,
                    color="rgba(236,72,153,0.4)",
                    line=dict(width=3, color="#f472b6"),
                ),
                name="Analysis focal",
            )
        )

    rot_lon = rotation_lon
    if rot_lon is None and focal_region and focal_region in REGION_CENTROIDS:
        rot_lon = REGION_CENTROIDS[focal_region][1]
    elif rot_lon is None:
        rot_lon = -40

    fig = go.Figure(data=traces)
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=rot_lon, lat=15),
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0c1222",
        showcountries=True,
        countrycolor="#334155",
        coastlinecolor="#475569",
        coastlinewidth=0.6,
        bgcolor="rgba(0,0,0,0)",
        showlakes=True,
        lakecolor="#0c1222",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10,12,24,0)",
        plot_bgcolor="rgba(10,12,24,0)",
        margin=dict(l=0, r=0, t=48, b=0),
        height=720,
        title=dict(
            text="Global climate & catastrophe exposure overlay",
            font=dict(size=18, color="#e2e8f0", family="'Segoe UI', 'IBM Plex Sans', sans-serif"),
            x=0.5,
        ),
        font=dict(color="#cbd5e1"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.78)",
            bordercolor="#334155",
            borderwidth=1,
            x=0.02,
            y=0.98,
        ),
    )

    if not incidents:
        fig.add_annotation(
            text="No region data — check PERIL_CONFIG regions.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )

    return fig
