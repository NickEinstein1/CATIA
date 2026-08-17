"""
CATIA system dashboard — interactive front-end with global hazard globe.

Run:  catia --dashboard
Or:   python -m catia.dashboard

Uses Dash + Plotly orthographic globe. Reads outputs/catia_report.json for loss-weighted overlays.

Live Earth env (optional): ``CATIA_LIVE_REFRESH_MS``, ``CATIA_LIVE_GLOBE_MAX_POINTS`` (globe cap),
``CATIA_LIVE_MAP_CLUSTER`` (0 = disable Leaflet marker clustering),
``CATIA_DECK_MAP_STYLE`` (e.g. ``CARTO_DARK_MATTER``, ``OPENFREEMAP_LIBERTY`` — see ``deckgl-dash`` MapLibre styles),
``CATIA_PUBLIC_DASH_URL`` (optional base for “copy share link”), ``CATIA_EXPOSURE_OVERLAY`` (Deck.gl indicative regions).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from catia import __version__
from catia.config import CLIMATE_SCENARIOS, OUTPUT_CONFIG, PERIL_CONFIG
from catia.geo_hazards import PERIL_VIS_COLORS, fig_global_hazard_globe
from catia.geo_deck import build_live_deck_earth_map
from catia.geo_osm import build_osm_leaflet_map, build_osm_live_catastrophe_map
from catia.geo_regions import REGION_CENTROIDS
from catia.live_alert_rules import evaluate_live_rules, load_rules
from catia.live_catastrophe_feeds import LiveFeedResult, category_color, fetch_all_live_events
from catia.live_compliance import attribution_footer
from catia.live_intel import enrich_and_rank_events

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Plotly toolbar: keep useful export, drop lasso (not helpful on geo)
_PLOTLY_UI_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "catia_chart"},
}

_GLOBE_MAX_POINTS = int(os.environ.get("CATIA_LIVE_GLOBE_MAX_POINTS", "800"))


def _report_path(output_dir: Optional[str] = None) -> Path:
    base = Path(output_dir or OUTPUT_CONFIG.get("output_dir", "outputs"))
    return base / "catia_report.json"


def _register_path(output_dir: Optional[str] = None) -> Path:
    base = Path(output_dir or OUTPUT_CONFIG.get("output_dir", "outputs"))
    return base / "assumption_register.json"


def load_report(output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = _report_path(output_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load catia_report.json: %s", e)
        return None


def load_assumption_register(output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = _register_path(output_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load assumption_register.json: %s", e)
        return None


def _list_output_files(output_dir: Path) -> List[str]:
    if not output_dir.is_dir():
        return []
    return sorted(
        f.name for f in output_dir.iterdir()
        if f.is_file() and (f.suffix in {".html", ".json"} or f.name.endswith(".log"))
    )


def _style_dark_chart(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#cbd5e1"),
        title_font=dict(color="#e2e8f0"),
    )


def _live_marker_color(event: Dict[str, Any]) -> str:
    peril = event.get("catia_peril") or ""
    if isinstance(peril, str) and peril and peril in PERIL_VIS_COLORS:
        return PERIL_VIS_COLORS[peril]
    return category_color(str(event.get("category") or ""))


def _live_marker_size(event: Dict[str, Any]) -> float:
    try:
        sc = float(event.get("catia_score") if event.get("catia_score") is not None else 45.0)
    except (TypeError, ValueError):
        sc = 45.0
    sc = max(0.0, min(100.0, sc))
    return max(7.0, min(24.0, 6.0 + 18.0 * (sc / 100.0) ** 0.55))


def fig_live_catastrophe_globe(events: List[Dict[str, Any]]) -> go.Figure:
    """Orthographic globe with live USGS / EONET / GDACS points."""
    if not events:
        fig = go.Figure()
        fig.update_layout(
            title="Live events (no data — check network or API status)",
            height=520,
            annotations=[
                dict(
                    text="Open feeds failed or returned no points.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#94a3b8"),
                )
            ],
        )
        _style_dark_chart(fig)
        return fig
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=[float(e["lon"]) for e in events],
                lat=[float(e["lat"]) for e in events],
                mode="markers",
                marker=dict(
                    size=[_live_marker_size(e) for e in events],
                    color=[_live_marker_color(e) for e in events],
                    line=dict(width=1, color="rgba(15,23,42,0.9)"),
                    opacity=0.92,
                ),
                text=[
                    f"{str(e.get('title', ''))[:70]}<br>"
                    f"{e.get('category_label', '')} · {e.get('source', '')} {e.get('severity_label', '')}<br>"
                    f"CATIA score: {float(e.get('catia_score') or 0):.0f}"
                    for e in events
                ],
                hoverinfo="text",
            )
        ],
        layout=dict(
            title="Near–real-time events (USGS · NASA EONET · GDACS when enabled)",
            height=560,
            margin=dict(l=8, r=8, t=56, b=8),
            geo=dict(
                projection=dict(type="orthographic", rotation=dict(lon=0, lat=15, roll=0)),
                showland=True,
                landcolor="#334155",
                oceancolor="#0f172a",
                showocean=True,
                bgcolor="rgba(15,23,42,0.6)",
            ),
        ),
    )
    _style_dark_chart(fig)
    return fig


def _score_badge(score: float) -> html.Span:
    s = float(score)
    cls = "catia-score-pill catia-score-pill--high" if s >= 70.0 else "catia-score-pill"
    return html.Span(f"{s:.0f}", className=cls)


_LIVE_FEED_LABELS = (
    ("usgs", "USGS"),
    ("eonet", "EONET"),
    ("gdacs", "GDACS"),
)


def _live_kpi_strip(
    feed_fetched_at: str,
    n_events: int,
    top_score: float,
    focal: Optional[str],
    sources_ok: Dict[str, bool],
    *,
    raw_from_feeds: Optional[int] = None,
    filter_hint: Optional[str] = None,
) -> html.Div:
    def _feed_dot(ok: bool) -> str:
        return "catia-feed-dot catia-feed-dot--ok" if ok else "catia-feed-dot catia-feed-dot--bad"

    meta_parts = [f"Last refresh · {feed_fetched_at}"]
    if raw_from_feeds is not None:
        meta_parts.append(f"{n_events} shown after filters · {raw_from_feeds} raw ingested")

    tail: List[Any] = [
        html.P(" · ".join(meta_parts), className="catia-kpi-strip__meta"),
    ]
    if filter_hint:
        tail.append(html.P(filter_hint, className="catia-kpi-strip__filters"))

    return html.Div(
        className="catia-kpi-strip",
        children=[
            html.Div(
                className="catia-kpi-grid",
                children=[
                    html.Div(
                        className="catia-kpi-card",
                        children=[
                            html.Div("Live events", className="catia-kpi-card__label"),
                            html.Div(str(n_events), className="catia-kpi-card__value"),
                        ],
                    ),
                    html.Div(
                        className="catia-kpi-card",
                        children=[
                            html.Div("Top CATIA score", className="catia-kpi-card__label"),
                            html.Div(
                                f"{top_score:.0f}" if n_events else "—",
                                className="catia-kpi-card__value",
                            ),
                        ],
                    ),
                    html.Div(
                        className="catia-kpi-card",
                        children=[
                            html.Div("Focal region", className="catia-kpi-card__label"),
                            html.Div(
                                (focal or "—").replace("_", " "),
                                className="catia-kpi-card__value catia-kpi-card__value--sm",
                            ),
                        ],
                    ),
                    html.Div(
                        className="catia-kpi-card",
                        children=[
                            html.Div("Feeds", className="catia-kpi-card__label"),
                            html.Div(
                                className="catia-kpi-card__feeds",
                                children=[
                                    html.Span(
                                        children=[
                                            html.Span(
                                                className=_feed_dot(
                                                    bool(sources_ok.get(key, False))
                                                )
                                            ),
                                            f" {label}",
                                        ]
                                    )
                                    for key, label in _LIVE_FEED_LABELS
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            *tail,
        ],
    )


def _live_feed_health_strip(feed: LiveFeedResult) -> html.Div:
    pills: List[Any] = []
    for key, label in _LIVE_FEED_LABELS:
        ms = feed.latency_ms.get(key)
        code = feed.http_status.get(key)
        ok = bool(feed.sources_ok.get(key, False))
        lat_s = f"{float(ms):.0f} ms" if isinstance(ms, (int, float)) else "—"
        st_s = str(code) if code is not None else "—"
        cls = "catia-health-pill catia-health-pill--ok" if ok else "catia-health-pill catia-health-pill--bad"
        pills.append(html.Span(className=cls, children=f"{label}: {lat_s} · HTTP {st_s}"))
    cache = "hit" if feed.cache_hit else "miss"
    cb = feed.cache_backend or "memory"
    off = " · offline" if feed.offline_mode else ""
    return html.Div(
        className="catia-live-health",
        children=[
            html.Span("Feed health · ", className="catia-live-health__lead"),
            *pills,
            html.Span(
                className="catia-live-health__cache",
                children=f" · Cache: {cache} ({cb}){off}",
            ),
        ],
    )


def _live_alert_hits_banner(events: List[Dict[str, Any]]) -> Optional[html.Div]:
    hits = evaluate_live_rules(events, load_rules())
    if not hits:
        return None
    lines = [f"{h.label}: {h.event_title} (score {h.score:.0f})" for h in hits[:8]]
    return html.Div(
        className="catia-flash catia-flash--info",
        children=[
            html.P("Live alert rules matched:", className="catia-flash__title"),
            html.Ul([html.Li(t) for t in lines], className="catia-flash__list"),
        ],
    )


def _live_peril_filter_options() -> List[Dict[str, str]]:
    opts: List[Dict[str, str]] = [{"label": "All mapped perils", "value": "all"}]
    for pid, cfg in PERIL_CONFIG.items():
        opts.append({"label": str(cfg.get("name", pid)), "value": pid})
    opts.append({"label": "Unmapped only", "value": "__unmapped__"})
    return opts


def _focal_region_dropdown_options() -> List[Dict[str, str]]:
    opts: List[Dict[str, str]] = [{"label": "Use latest run focal region", "value": ""}]
    for rid in sorted(REGION_CENTROIDS.keys()):
        opts.append({"label": rid.replace("_", " "), "value": rid})
    return opts


def _category_chips(pairs: List[Tuple[str, int]]) -> html.Div:
    if not pairs:
        return html.Div(
            className="catia-chip-row",
            children=[html.Span("No breakdown", className="catia-chip catia-chip--muted")],
        )
    return html.Div(
        className="catia-chip-row",
        children=[
            html.Span(f"{k}: {v}", className="catia-chip") for k, v in pairs[:14]
        ],
    )


def _live_feed_legend() -> html.Div:
    samples = [
        ("earthquake", "Earthquake (USGS)"),
        ("wildfires", "Wildfires (EONET)"),
        ("severe_storms", "Severe storms (EONET)"),
        ("volcanoes", "Volcanoes (EONET)"),
        ("floods", "Floods (EONET)"),
        ("hurricane", "GDACS / tropical cyclone–class events"),
    ]
    items = []
    for slug, label in samples:
        c = category_color(slug)
        items.append(
            html.Div(
                className="catia-legend-item",
                children=[
                    html.Span(className="catia-legend-dot", style={"backgroundColor": c, "color": c}),
                    html.Span(label),
                ],
            )
        )
    return html.Div(className="catia-legend", children=items)


def fig_return_periods(report: Dict[str, Any]) -> Optional[go.Figure]:
    rm = report.get("risk_metrics", {}) or {}
    rp = rm.get("return_periods") or {}
    if not rp:
        return None
    xs, ys = [], []
    for key in sorted(rp.keys(), key=lambda k: int(k.split("_")[0]) if k.split("_")[0].isdigit() else 0):
        if "_year" in key:
            xs.append(int(key.split("_")[0]))
            ys.append(float(rp[key]) / 1e6)
    if not xs:
        return None
    fig = go.Figure(
        data=[go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(width=2, color="#22d3ee"))],
        layout=dict(
            title="Return periods (loss $M)",
            xaxis_title="Return period (years)",
            yaxis_title="Loss ($ millions)",
            height=400,
        ),
    )
    fig.update_xaxes(type="log")
    _style_dark_chart(fig)
    return fig


def fig_peril_contributions(report: Dict[str, Any]) -> Optional[go.Figure]:
    rows = report.get("multi_peril_contributions") or []
    if not rows:
        return None
    names = [r.get("peril_name", r.get("peril", "?")) for r in rows]
    colors = []
    for r in rows:
        pk = r.get("peril")
        if isinstance(pk, str) and pk in PERIL_VIS_COLORS:
            colors.append(PERIL_VIS_COLORS[pk])
        else:
            colors.append("#38bdf8")
    vals = [float(r.get("mean_loss", 0)) for r in rows]
    fig = go.Figure(
        data=[go.Bar(x=names, y=[v / 1e6 for v in vals], marker_color=colors)],
        layout=dict(
            title="Mean loss by peril ($M)",
            height=400,
        ),
    )
    _style_dark_chart(fig)
    return fig


def fig_mitigation(report: Dict[str, Any]) -> Optional[go.Figure]:
    strat = report.get("mitigation_strategies") or []
    if not strat:
        return None
    names = [s.get("Strategy", "?") for s in strat]
    costs = [float(s.get("Cost", 0)) / 1e6 for s in strat]
    reductions = [float(s.get("Risk_Reduction", 0)) * 100 for s in strat]
    fig = go.Figure(
        data=[
            go.Bar(name="Cost ($M)", x=names, y=costs, marker_color="#34d399"),
            go.Bar(
                name="Risk reduction (%)",
                x=names,
                y=reductions,
                marker_color="#a855f7",
                yaxis="y2",
            ),
        ],
        layout=dict(
            title="Mitigation strategies",
            height=420,
            yaxis=dict(title="Cost ($M)", color="#94a3b8"),
            yaxis2=dict(title="Risk reduction (%)", overlaying="y", side="right", color="#94a3b8"),
        ),
    )
    _style_dark_chart(fig)
    return fig


def _peril_legend_row() -> html.Div:
    items = []
    for pid, hex_c in PERIL_VIS_COLORS.items():
        name = PERIL_CONFIG.get(pid, {}).get("name", pid)
        items.append(
            html.Div(
                className="catia-legend-item",
                children=[
                    html.Span(className="catia-legend-dot", style={"backgroundColor": hex_c, "color": hex_c}),
                    html.Span(f"{name}"),
                ],
            )
        )
    return html.Div(className="catia-legend", children=items)


def _weather_command_header(output_dir: Path) -> html.Header:
    """Atmospheric-operations masthead shared by every dashboard view."""
    utc_now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return html.Header(
        className="wx-command-header",
        children=[
            html.Div(
                className="wx-command-header__copy",
                children=[
                    html.Div(
                        className="wx-brand-row",
                        children=[
                            html.Div(
                                className="wx-brand-mark",
                                **{"aria-hidden": "true"},
                                children=[
                                    html.Span(className="wx-brand-mark__orbit"),
                                    html.Span(className="wx-brand-mark__core"),
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.P(
                                        "CATIA / ATMOSPHERIC INTELLIGENCE",
                                        className="wx-eyebrow",
                                    ),
                                    html.H1(
                                        [
                                            "Global Hazard",
                                            html.Br(),
                                            html.Span("Command System"),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.P(
                        "A live decision surface for catastrophe signals, modeled risk, "
                        "and regional exposure—ranked by the CATIA intelligence layer.",
                        className="wx-command-header__lede",
                    ),
                    html.Div(
                        className="wx-network-strip",
                        children=[
                            html.Span(
                                [
                                    html.I(className="wx-status-dot wx-status-dot--live"),
                                    "LIVE OBSERVATION NETWORK",
                                ],
                                className="wx-network-strip__live",
                            ),
                            html.Span("USGS", className="wx-source-token"),
                            html.Span("NASA EONET", className="wx-source-token"),
                            html.Span("GDACS", className="wx-source-token"),
                            html.Span("BUILD 2.5.0", className="wx-source-token"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="wx-radar-console",
                **{"aria-label": "CATIA system status"},
                children=[
                    html.Div(
                        className="wx-radar",
                        **{"aria-hidden": "true"},
                        children=[
                            html.Span(className="wx-radar__ring wx-radar__ring--one"),
                            html.Span(className="wx-radar__ring wx-radar__ring--two"),
                            html.Span(className="wx-radar__ring wx-radar__ring--three"),
                            html.Span(className="wx-radar__crosshair"),
                            html.Span(className="wx-radar__sweep"),
                            html.Span(className="wx-radar__ping wx-radar__ping--one"),
                            html.Span(className="wx-radar__ping wx-radar__ping--two"),
                            html.Span(className="wx-radar__ping wx-radar__ping--three"),
                        ],
                    ),
                    html.Div(
                        className="wx-console-meta",
                        children=[
                            html.Div(
                                [
                                    html.Span("SYSTEM", className="wx-console-meta__label"),
                                    html.Strong("NOMINAL"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("UTC", className="wx-console-meta__label"),
                                    html.Strong(utc_now),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("BUILD", className="wx-console-meta__label"),
                                    html.Strong("2.5.0"),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("DATA", className="wx-console-meta__label"),
                                    html.Strong(output_dir.name.upper()),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_dash_app(
    output_dir: Optional[str] = None,
    *,
    api_base_url: str = "http://127.0.0.1:8000",
) -> Dash:
    """Build Dash application instance."""
    out = Path(output_dir or OUTPUT_CONFIG.get("output_dir", "outputs"))
    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    app = Dash(
        __name__,
        title="CATIA · Global command · 2.5.0",
        suppress_callback_exceptions=True,
        assets_folder=str(_ASSETS_DIR),
    )
    app._catia_output_dir = str(out)  # type: ignore[attr-defined]
    app._catia_api_base = api_base_url  # type: ignore[attr-defined]

    app.layout = html.Div(
        className="catia-future-root",
        children=[
            dcc.Location(id="url", refresh=False),
            html.Div(className="wx-ambient-grid", **{"aria-hidden": "true"}),
            _weather_command_header(out),
            html.Div(
                className="wx-section-label",
                children=[
                    html.Span("COMMAND MODULES"),
                    html.Span("Select an intelligence surface", className="wx-section-label__hint"),
                ],
            ),
            dcc.Tabs(
                id="dash-tabs",
                value="tab-globe",
                className="catia-tabs",
                colors={
                    "border": "rgba(51,65,85,0.8)",
                    "primary": "#22d3ee",
                    "background": "rgba(15,23,42,0.5)",
                },
                children=[
                    dcc.Tab(label="01  Global", value="tab-globe"),
                    dcc.Tab(label="02  Live Earth", value="tab-live"),
                    dcc.Tab(label="03  Overview", value="tab-overview"),
                    dcc.Tab(label="04  Latest Run", value="tab-run"),
                    dcc.Tab(label="05  Analytics", value="tab-charts"),
                    dcc.Tab(label="06  Scenarios", value="tab-perils"),
                    dcc.Tab(label="07  Assumptions", value="tab-assumptions"),
                    dcc.Tab(label="08  System", value="tab-api"),
                ],
            ),
            html.Div(
                id="live-toolbar",
                className="catia-live-toolbar",
                style={"display": "none"},
                children=[
                    html.Div(
                        className="catia-live-toolbar__header",
                        children=[
                            html.Span("Live Earth filters", className="catia-live-toolbar__title"),
                            html.Span(
                                "These controls filter what you see on this tab only — "
                                "they do not change upstream feeds.",
                                className="catia-live-toolbar__hint",
                            ),
                        ],
                    ),
                    html.Div(
                        className="catia-live-toolbar__grid",
                        children=[
                            html.Div(
                                className="catia-live-toolbar__field",
                                children=[
                                    html.Label("CATIA peril", className="catia-live-toolbar__label"),
                                    dcc.Dropdown(
                                        id="live-filter-peril",
                                        options=_live_peril_filter_options(),
                                        value="all",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        className="catia-dash-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-live-toolbar__field catia-live-toolbar__field--wide",
                                children=[
                                    html.Label(
                                        "Minimum CATIA score",
                                        className="catia-live-toolbar__label",
                                    ),
                                    dcc.Slider(
                                        id="live-filter-min-score",
                                        min=0,
                                        max=90,
                                        step=5,
                                        value=0,
                                        marks={0: "0", 45: "45", 90: "90"},
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                        persistence=True,
                                        persistence_type="session",
                                        className="catia-dash-slider",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-live-toolbar__field",
                                children=[
                                    html.Label(
                                        "Proximity focal region",
                                        className="catia-live-toolbar__label",
                                    ),
                                    dcc.Dropdown(
                                        id="live-filter-region",
                                        options=_focal_region_dropdown_options(),
                                        value="",
                                        clearable=False,
                                        persistence=True,
                                        persistence_type="session",
                                        className="catia-dash-dropdown",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="catia-live-toolbar__row2",
                        children=[
                            html.Div(
                                className="catia-live-toolbar__field catia-live-toolbar__field--wide",
                                children=[
                                    html.Label(
                                        "Compare to modeled view",
                                        className="catia-live-toolbar__label",
                                    ),
                                    dcc.RadioItems(
                                        id="live-compare-mode",
                                        options=[
                                            {
                                                "label": "Live feeds only",
                                                "value": "live_only",
                                            },
                                            {
                                                "label": "Split: modeled globe + live",
                                                "value": "split",
                                            },
                                        ],
                                        value="live_only",
                                        persistence=True,
                                        persistence_type="session",
                                        className="catia-live-toolbar__radios",
                                        inputClassName="catia-live-toolbar__radio",
                                        labelClassName="catia-live-toolbar__radio-label",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-live-toolbar__actions",
                                children=[
                                    html.Button(
                                        "Export preset",
                                        id="live-preset-export-btn",
                                        type="button",
                                        className="catia-btn catia-btn--ghost",
                                    ),
                                    dcc.Upload(
                                        id="live-preset-upload",
                                        children=html.Span("Import preset"),
                                        className="catia-upload-zone",
                                        multiple=False,
                                    ),
                                    dcc.Clipboard(
                                        id="live-share-clipboard",
                                        title="Copy shareable link",
                                        className="catia-clipboard",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Download(id="live-preset-download"),
            dcc.Loading(
                id="tab-loading",
                type="circle",
                color="#22d3ee",
                className="catia-page-loading",
                children=html.Div(id="tab-content", style={"marginTop": "16px"}),
            ),
            dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
            dcc.Interval(
                id="live-feed-interval",
                interval=int(os.environ.get("CATIA_LIVE_REFRESH_MS", "180000")),
                n_intervals=0,
            ),
        ],
    )

    @app.callback(
        Output("live-toolbar", "style"),
        Input("dash-tabs", "value"),
    )
    def toggle_live_toolbar(active: str):
        if active == "tab-live":
            return {"display": "block", "marginTop": "12px", "marginBottom": "4px"}
        return {"display": "none"}

    @app.callback(
        Output("tab-content", "children"),
        Input("dash-tabs", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("live-feed-interval", "n_intervals"),
        Input("live-filter-peril", "value"),
        Input("live-filter-min-score", "value"),
        Input("live-filter-region", "value"),
        Input("live-compare-mode", "value"),
    )
    def render_tab(
        active: str,
        _n: int,
        _live_n: int,
        peril_sel: Optional[str],
        min_score: Optional[Any],
        region_sel: Optional[str],
        compare_mode: Optional[str],
    ):
        triggered = ""
        if callback_context.triggered:
            triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        force_live = active == "tab-live" and triggered == "dash-tabs"

        report = load_report(str(out))
        reg = load_assumption_register(str(out))
        files = _list_output_files(out)
        focal = (report or {}).get("metadata", {}).get("region")

        if active == "tab-globe":
            globe = fig_global_hazard_globe(report, focal_region=focal)
            caption = (
                "Markers scale with modeled mean loss by peril (latest run). "
                "Without a report, footprint reflects peril frequencies. "
                "Pink ring: analysis focal region."
                if report
                else "Demo footprint from peril configuration — run "
                "`python main.py` or `catia` to weight markers by simulation output."
            )
            osm_map = build_osm_leaflet_map(report, focal)
            osm_section: Any
            if osm_map is not None:
                osm_section = html.Div(
                    className="catia-panel",
                    style={"padding": "12px"},
                    children=[
                        html.H3(
                            "OpenStreetMap — 2D base map",
                            style={"marginTop": 0, "marginBottom": "8px"},
                        ),
                        html.P(
                            "Same hazard markers on community map tiles (pan/zoom). "
                            "Data © OpenStreetMap contributors.",
                            style={"color": "#94a3b8", "fontSize": "0.88rem", "marginBottom": "12px"},
                        ),
                        osm_map,
                    ],
                )
            else:
                osm_section = html.Div(
                    className="catia-panel",
                    children=[
                        html.H3("OpenStreetMap map"),
                        html.P(
                            [
                                "Install ",
                                html.Code("dash-leaflet"),
                                " for the 2D OSM map (",
                                html.Code("pip install dash-leaflet"),
                                ").",
                            ],
                            style={"color": "#94a3b8"},
                        ),
                    ],
                )

            return html.Div(
                [
                    html.Div(
                        className="catia-panel",
                        style={"padding": "12px 12px 4px"},
                        children=[
                            html.H3(
                                "Orthographic globe",
                                style={"marginTop": 0, "marginBottom": "8px"},
                            ),
                            dcc.Graph(
                                figure=globe,
                                style={"minHeight": "640px"},
                                config=_PLOTLY_UI_CONFIG,
                            ),
                        ],
                    ),
                    html.P(caption, className="globe-caption"),
                    _peril_legend_row(),
                    osm_section,
                ]
            )

        if active == "tab-live":
            feed = fetch_all_live_events(force=force_live)
            raw_n = len(feed.events)
            pf = (peril_sel or "all").strip()
            try:
                min_sc = float(min_score) if min_score is not None else 0.0
            except (TypeError, ValueError):
                min_sc = 0.0
            min_sc = max(0.0, min(90.0, min_sc))
            region_override = (region_sel or "").strip() or None
            focal_run = (report or {}).get("metadata", {}).get("region")
            focal_eff = region_override if region_override else focal_run

            peril_arg = None if pf == "all" else pf
            events = enrich_and_rank_events(
                feed.events,
                focal_region=focal_eff,
                peril_filter=peril_arg,
            )
            events = [e for e in events if float(e.get("catia_score") or 0.0) >= min_sc]

            top_sc = max((float(e.get("catia_score") or 0.0) for e in events), default=0.0)

            filter_hint = (
                f"Peril filter: {pf} · min score ≥ {min_sc:.0f} · focal for proximity scoring: "
                f"{(focal_eff or '—').replace('_', ' ')}"
            )
            compare_raw = (compare_mode or "live_only").strip()

            sr_status = html.Div(
                f"Live Earth updated: {len(events)} events match filters.",
                className="catia-sr-only",
                role="status",
                **{"aria-live": "polite", "aria-atomic": "true"},
            )

            err_banner: Optional[html.Div] = None
            if feed.errors:
                err_banner = html.Div(
                    className="catia-flash catia-flash--warn",
                    children=[
                        html.P(
                            "Some feeds failed (partial data). " + " · ".join(feed.errors),
                            className="catia-flash__text",
                        ),
                    ],
                )

            globe_plot = events[:_GLOBE_MAX_POINTS]
            globe_live = fig_live_catastrophe_globe(globe_plot)
            live_map = build_osm_live_catastrophe_map(events)
            map_section: Any
            if live_map is not None:
                map_section = html.Div(
                    className="catia-panel catia-panel--tight",
                    style={"padding": "12px"},
                    children=[
                        html.Div(
                            className="catia-section-head",
                            children=[
                                html.H3("Surface map", className="catia-section-head__title"),
                                html.P(
                                    "Pan / zoom · circle size ∝ CATIA score · clustered when many points",
                                    className="catia-section-head__sub",
                                ),
                            ],
                        ),
                        live_map,
                    ],
                )
            else:
                map_section = html.Div(
                    className="catia-panel",
                    children=[
                        html.P(
                            [
                                "Install ",
                                html.Code("dash-leaflet"),
                                " for the 2D map (",
                                html.Code("pip install dash-leaflet"),
                                ").",
                            ],
                            style={"color": "#94a3b8"},
                        ),
                    ],
                )

            counts: Dict[str, int] = {}
            for e in events:
                k = str(e.get("category_label") or e.get("category") or "?")
                counts[k] = counts.get(k, 0) + 1
            top_counts = sorted(counts.items(), key=lambda x: -x[1])[:12]
            type_breakdown = html.Div(
                className="catia-panel",
                style={"padding": "12px"},
                children=[
                    html.Div(
                        className="catia-section-head",
                        children=[
                            html.H3("By event type", className="catia-section-head__title"),
                            html.P(
                                "Distribution for the filtered points currently shown on the maps.",
                                className="catia-section-head__sub",
                            ),
                        ],
                    ),
                    _category_chips(top_counts),
                    html.P(
                        [
                            "Attribution: ",
                            html.A(
                                "USGS",
                                href="https://earthquake.usgs.gov/earthquakes/feed/",
                                target="_blank",
                                rel="noopener noreferrer",
                                className="catia-link",
                            ),
                            " · ",
                            html.A(
                                "NASA EONET",
                                href="https://eonet.gsfc.nasa.gov/",
                                target="_blank",
                                rel="noopener noreferrer",
                                className="catia-link",
                            ),
                            " · ",
                            html.A(
                                "GDACS",
                                href="https://www.gdacs.org/",
                                target="_blank",
                                rel="noopener noreferrer",
                                className="catia-link",
                            ),
                            " · Auto-refresh: ",
                            html.Code("CATIA_LIVE_REFRESH_MS"),
                        ],
                        className="catia-footnote",
                    ),
                ],
            )

            kpi = _live_kpi_strip(
                feed.fetched_at_iso,
                len(events),
                top_sc,
                focal_eff,
                feed.sources_ok,
                raw_from_feeds=raw_n,
                filter_hint=filter_hint,
            )

            top_alert_rows = [
                html.Tr(
                    [
                        html.Th("Score"),
                        html.Th("Peril"),
                        html.Th("Where / what"),
                        html.Th("Type"),
                        html.Th("Source"),
                        html.Th("When / detail"),
                    ],
                    className="catia-table__headrow",
                )
            ]
            for e in events[:12]:
                top_alert_rows.append(
                    html.Tr(
                        className="catia-table__row",
                        children=[
                            html.Td(
                                _score_badge(float(e.get("catia_score") or 0.0)),
                                className="catia-table__num",
                            ),
                            html.Td(str(e.get("catia_peril") or "—")),
                            html.Td(str(e.get("title", ""))[:90]),
                            html.Td(str(e.get("category_label", ""))[:40]),
                            html.Td(str(e.get("source", ""))),
                            html.Td(
                                " ".join(
                                    x
                                    for x in (e.get("time_iso"), e.get("severity_label"))
                                    if x
                                )
                                or "—"
                            ),
                        ],
                    )
                )

            table_rows = [
                html.Tr(
                    [
                        html.Th("Score"),
                        html.Th("CATIA peril"),
                        html.Th("Where / what"),
                        html.Th("Type"),
                        html.Th("Source"),
                        html.Th("When / detail"),
                    ],
                    className="catia-table__headrow",
                )
            ]
            for e in events[:40]:
                table_rows.append(
                    html.Tr(
                        className="catia-table__row",
                        children=[
                            html.Td(
                                _score_badge(float(e.get("catia_score") or 0.0)),
                                className="catia-table__num",
                            ),
                            html.Td(str(e.get("catia_peril") or "—")),
                            html.Td(str(e.get("title", ""))[:80]),
                            html.Td(str(e.get("category_label", ""))[:40]),
                            html.Td(str(e.get("source", ""))),
                            html.Td(
                                " ".join(
                                    x
                                    for x in (e.get("time_iso"), e.get("severity_label"))
                                    if x
                                )
                                or "—"
                            ),
                        ],
                    )
                )

            globe_live_panel = html.Div(
                className="catia-panel catia-panel--tight",
                style={"padding": "12px 12px 4px"},
                children=[
                    html.Div(
                        className="catia-section-head",
                        children=[
                            html.H3(
                                "Live globe",
                                className="catia-section-head__title",
                            ),
                            html.P(
                                "Orthographic · marker size ∝ CATIA score"
                                + (
                                    f" · showing top {_GLOBE_MAX_POINTS} by score "
                                    f"({len(events)} match filters)"
                                    if len(events) > _GLOBE_MAX_POINTS
                                    else ""
                                ),
                                className="catia-section-head__sub",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        figure=globe_live,
                        style={"minHeight": "520px"},
                        config=_PLOTLY_UI_CONFIG,
                    ),
                ],
            )

            if compare_raw == "split":
                if report:
                    globe_modeled = fig_global_hazard_globe(report, focal_region=focal_eff)
                else:
                    globe_modeled = go.Figure()
                    globe_modeled.update_layout(
                        title="Modeled globe (no catia_report.json)",
                        height=520,
                        annotations=[
                            dict(
                                text="Run an analysis to compare live feeds with loss-weighted peril markers.",
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                y=0.5,
                                showarrow=False,
                                font=dict(size=14, color="#94a3b8"),
                            )
                        ],
                    )
                    _style_dark_chart(globe_modeled)
                maps_row = html.Div(
                    className="catia-split-grid catia-split-grid--compare",
                    children=[
                        html.Div(
                            className="catia-split-grid__col",
                            children=[
                                html.Div(
                                    className="catia-panel catia-panel--tight",
                                    style={"padding": "12px 12px 4px"},
                                    children=[
                                        html.Div(
                                            className="catia-section-head",
                                            children=[
                                                html.H3(
                                                    "Modeled CATIA globe",
                                                    className="catia-section-head__title",
                                                ),
                                                html.P(
                                                    "Latest run: loss-weighted markers (not live feeds).",
                                                    className="catia-section-head__sub",
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(
                                            figure=globe_modeled,
                                            style={"minHeight": "520px"},
                                            config=_PLOTLY_UI_CONFIG,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(className="catia-split-grid__col", children=[globe_live_panel]),
                        html.Div(className="catia-split-grid__col", children=[map_section]),
                    ],
                )
            else:
                maps_row = html.Div(
                    className="catia-split-grid",
                    children=[
                        html.Div(
                            className="catia-split-grid__col",
                            children=[globe_live_panel],
                        ),
                        html.Div(className="catia-split-grid__col", children=[map_section]),
                    ],
                )

            deck_gl = build_live_deck_earth_map(events)
            deck_section: Any
            if deck_gl is not None:
                deck_section = html.Div(
                    className="catia-panel catia-panel--deck",
                    style={"padding": "12px"},
                    children=[
                        html.Div(
                            className="catia-section-head",
                            children=[
                                html.H3(
                                    "Deck.gl + MapLibre (GPU)",
                                    className="catia-section-head__title",
                                ),
                                html.P(
                                    "WebGL ScatterplotLayer on a vector MapLibre basemap — ideal for dense feeds.",
                                    className="catia-section-head__sub",
                                ),
                            ],
                        ),
                        deck_gl,
                        html.P(
                            [
                                "Indicative exposure regions (Deck layer under points) respect ",
                                html.Code("CATIA_EXPOSURE_OVERLAY"),
                                ". Style: ",
                                html.Code("CATIA_DECK_MAP_STYLE"),
                                " (e.g. ",
                                html.Code("CARTO_DARK_MATTER"),
                                ", ",
                                html.Code("OPENFREEMAP_LIBERTY"),
                                "). Follow basemap provider terms.",
                            ],
                            className="catia-footnote",
                        ),
                    ],
                )
            else:
                deck_section = html.Div(
                    className="catia-panel",
                    style={"padding": "12px"},
                    children=[
                        html.H3(
                            "Deck.gl + MapLibre (GPU)",
                            style={"marginTop": 0},
                        ),
                        html.P(
                            [
                                "Install ",
                                html.Code("deckgl-dash"),
                                " for the GPU map (",
                                html.Code("pip install deckgl-dash"),
                                ").",
                            ],
                            style={"color": "#94a3b8"},
                        ),
                    ],
                )

            live_blocks: List[Any] = [sr_status]
            if err_banner is not None:
                live_blocks.append(err_banner)
            live_blocks.append(kpi)
            live_blocks.append(_live_feed_health_strip(feed))
            _hits_banner = _live_alert_hits_banner(events)
            if _hits_banner is not None:
                live_blocks.append(_hits_banner)
            live_blocks.append(type_breakdown)
            live_blocks.extend(
                [
                    html.Div(
                        className="catia-panel",
                        style={"padding": "12px", "overflowX": "auto"},
                        children=[
                            html.Div(
                                className="catia-section-head",
                                children=[
                                    html.H3(
                                        "Top alerts",
                                        className="catia-section-head__title",
                                    ),
                                    html.P(
                                        "Ranked by CATIA score (severity + recency + proximity).",
                                        className="catia-section-head__sub",
                                    ),
                                ],
                            ),
                            html.Table(
                                [
                                    html.Caption(
                                        "Top alerts ranked by CATIA score.",
                                        className="catia-table-caption",
                                    ),
                                    html.Thead(top_alert_rows[0]),
                                    html.Tbody(top_alert_rows[1:]),
                                ],
                                className="catia-table",
                            ),
                        ],
                    ),
                    maps_row,
                    deck_section,
                    html.P(
                        "Globe / Leaflet / Deck layers combine USGS (typically M≥2.5, last 24h), NASA EONET, "
                        "and GDACS when enabled. Observational activity only — not CATIA modeled loss.",
                        className="globe-caption",
                    ),
                    _live_feed_legend(),
                    html.Div(
                        className="catia-panel",
                        style={"padding": "12px", "overflowX": "auto"},
                        children=[
                            html.Div(
                                className="catia-section-head",
                                children=[
                                    html.H3(
                                        f"All rows (up to {min(40, len(events))} of {len(events)})",
                                        className="catia-section-head__title",
                                    ),
                                    html.P(
                                        "Scroll horizontally on small screens.",
                                        className="catia-section-head__sub",
                                    ),
                                ],
                            ),
                            html.Table(
                                [
                                    html.Caption(
                                        "Full filtered list (truncated for readability).",
                                        className="catia-table-caption",
                                    ),
                                    html.Thead(table_rows[0]),
                                    html.Tbody(table_rows[1:]),
                                ],
                                className="catia-table",
                            ),
                        ],
                    ),
                    attribution_footer(compact=True),
                ]
            )
            return html.Div(live_blocks)

        if active == "tab-overview":
            return html.Div(
                [
                    html.Div(className="catia-panel", children=[
                        html.H3("Quick start"),
                        html.Ul([
                            html.Li([html.Code("catia"), " — full analysis"]),
                            html.Li([html.Code("catia --api --port 8000"), " — REST API"]),
                            html.Li([html.Code("catia --dashboard"), " — this command center"]),
                            html.Li(
                                "Open Global view for modeled exposure, or Live Earth for filtered "
                                "real-time feeds."
                            ),
                        ]),
                    ]),
                    html.Div(className="catia-panel", children=[
                        html.H3("Output files"),
                        html.P(f"{len(files)} file(s):"),
                        html.Ul([html.Li(f) for f in files[:40]] or [html.Li("(empty — run analysis first)")]),
                    ]),
                ]
            )

        if active == "tab-run":
            if not report:
                return html.Div(className="catia-panel", children=[
                    html.P(["No ", html.Code("catia_report.json"), " found."]),
                    html.P(["Run: ", html.Code("python main.py"), " or ", html.Code("catia -r US_Gulf_Coast")]),
                ])
            meta = report.get("metadata", {})
            rm = report.get("risk_metrics", {}) or {}
            desc = rm.get("descriptive_stats", {}) or {}
            risk = rm.get("risk_metrics", {}) or {}
            return html.Div(
                [
                    html.Div(className="catia-panel", children=[
                        html.H3("Run metadata"),
                        html.Table([
                            html.Tr([html.Th("Field"), html.Th("Value")]),
                            html.Tr([html.Td("Run ID"), html.Td(meta.get("run_id", "—"))]),
                            html.Tr([html.Td("Region"), html.Td(meta.get("region", "—"))]),
                            html.Tr([html.Td("Timestamp"), html.Td(meta.get("timestamp", "—"))]),
                            html.Tr([html.Td("Perils"), html.Td(", ".join(meta.get("perils_analyzed", [])))]),
                            html.Tr([html.Td("Mock data"), html.Td(str(meta.get("use_mock_data", "—")))]),
                        ]),
                    ]),
                    html.Div(className="catia-panel", children=[
                        html.H3("Aggregate risk metrics"),
                        html.Table([
                            html.Tr([html.Th("Metric"), html.Th("Value")]),
                            html.Tr([html.Td("Mean annual loss"), html.Td(f"${desc.get('mean', 0):,.0f}")]),
                            html.Tr([html.Td("VaR (95%)"), html.Td(f"${risk.get('var', 0):,.0f}")]),
                            html.Tr([html.Td("TVaR (95%)"), html.Td(f"${risk.get('tvar', 0):,.0f}")]),
                        ]),
                    ]),
                    html.Div(className="catia-panel", children=[
                        html.H3("Mitigation summary"),
                        html.Pre(json.dumps(report.get("mitigation_summary", {}), indent=2)),
                    ]),
                ]
            )

        if active == "tab-charts":
            if not report:
                return html.P("No report — run analysis first.", style={"color": "#94a3b8"})
            charts = []
            for f in [
                fig_return_periods(report),
                fig_peril_contributions(report),
                fig_mitigation(report),
            ]:
                if f is not None:
                    charts.append(
                        html.Div(
                            className="catia-panel",
                            style={"padding": "8px"},
                            children=[dcc.Graph(figure=f, config=_PLOTLY_UI_CONFIG)],
                        )
                    )
            if not charts:
                return html.P("No chart data in report.", style={"color": "#94a3b8"})
            return html.Div(charts)

        if active == "tab-perils":
            rows = []
            for pid, cfg in PERIL_CONFIG.items():
                rows.append(html.Tr([
                    html.Td(pid),
                    html.Td(cfg.get("name", "")),
                    html.Td(str(cfg.get("frequency_base", ""))),
                    html.Td(str(cfg.get("severity_params", ""))[:80] + "…"),
                ]))
            scen_items = []
            for sid, s in CLIMATE_SCENARIOS.items():
                scen_items.append(html.Li(f"{sid}: {s.get('description', '')}"))
            return html.Div(
                className="catia-panel",
                children=[
                    html.H3("Configured perils"),
                    html.Table(
                        [html.Tr([html.Th("Id"), html.Th("Name"), html.Th("Frequency"), html.Th("Severity")])] + rows,
                    ),
                    html.H3("Climate scenarios", style={"marginTop": "24px"}),
                    html.Ul(scen_items),
                ],
            )

        if active == "tab-assumptions":
            if not reg:
                return html.Div(className="catia-panel", children=[
                    html.P(["No ", html.Code("assumption_register.json"), " — run a full analysis."]),
                ])
            blob = json.dumps(reg, indent=2)
            return html.Div(
                className="catia-panel",
                children=[
                    html.H3("Assumption register"),
                    html.Pre(
                        blob[:20000] + ("…\n(truncated)" if len(blob) > 20000 else ""),
                        style={"maxHeight": "600px", "overflow": "auto", "fontSize": "12px"},
                    ),
                ],
            )

        api = getattr(app, "_catia_api_base", api_base_url)
        return html.Div(
            className="catia-panel",
            children=[
                html.H3("REST API"),
                html.P([
                    "OpenAPI: ",
                    html.A(
                        f"{api}/docs",
                        href=f"{api}/docs",
                        target="_blank",
                        style={"color": "#22d3ee"},
                    ),
                ]),
                html.Pre("catia --api --port 8000"),
                html.H3("Static HTML charts"),
                html.P([
                    "Open ",
                    html.Code(str(out / "loss_exceedance_curve.html")),
                    " locally after a run.",
                ]),
            ],
        )

    @app.callback(
        Output("live-filter-peril", "value"),
        Output("live-filter-min-score", "value"),
        Output("live-filter-region", "value"),
        Input("url", "search"),
        Input("dash-tabs", "value"),
        State("live-filter-peril", "value"),
        State("live-filter-min-score", "value"),
        State("live-filter-region", "value"),
        prevent_initial_call=False,
    )
    def hydrate_live_from_url(
        search: Optional[str],
        tab: str,
        cur_p: Optional[str],
        cur_m: Optional[Any],
        cur_r: Optional[str],
    ):
        if tab != "tab-live":
            raise PreventUpdate
        q = urllib.parse.parse_qs((search or "").lstrip("?"))
        if not any(k in q for k in ("lp", "lm", "lr")):
            raise PreventUpdate
        out_p = cur_p
        out_m = cur_m
        out_r = cur_r
        if q.get("lp") and q["lp"][0]:
            out_p = q["lp"][0]
        if q.get("lm") and q["lm"][0] != "":
            try:
                raw_m = float(q["lm"][0])
                out_m = round(max(0.0, min(90.0, raw_m)) / 5.0) * 5.0
            except ValueError:
                pass
        if "lr" in q:
            out_r = q["lr"][0] if q["lr"] else ""
        return out_p, out_m, out_r

    @app.callback(
        Output("live-filter-peril", "value", allow_duplicate=True),
        Output("live-filter-min-score", "value", allow_duplicate=True),
        Output("live-filter-region", "value", allow_duplicate=True),
        Input("live-preset-upload", "contents"),
        State("live-preset-upload", "filename"),
        prevent_initial_call=True,
    )
    def import_live_preset(contents: Optional[str], _filename: Optional[str]):
        if not contents:
            raise PreventUpdate
        try:
            meta = str(contents).split(",", 1)[1]
            raw = base64.b64decode(meta)
            data = json.loads(raw.decode("utf-8"))
            peril = str(data.get("peril") or "all")
            min_sc = float(data.get("min_score", 0))
            region = str(data.get("region") or "")
            min_sc = round(max(0.0, min(90.0, min_sc)) / 5.0) * 5.0
            return peril, min_sc, region
        except Exception:
            raise PreventUpdate

    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("live-filter-peril", "value"),
        Input("live-filter-min-score", "value"),
        Input("live-filter-region", "value"),
        State("dash-tabs", "value"),
        State("url", "search"),
        prevent_initial_call=True,
    )
    def push_live_filters_to_url(
        peril: Optional[str],
        min_sc: Optional[Any],
        region: Optional[str],
        tab: str,
        cur_search: Optional[str],
    ):
        if tab != "tab-live":
            raise PreventUpdate
        params: Dict[str, str] = {}
        if peril and peril != "all":
            params["lp"] = peril
        try:
            msv = float(min_sc) if min_sc is not None else 0.0
        except (TypeError, ValueError):
            msv = 0.0
        if msv > 0:
            params["lm"] = str(int(msv)) if float(msv).is_integer() else str(msv)
        if region:
            params["lr"] = region
        new_search = f"?{urllib.parse.urlencode(params)}" if params else ""
        cur = cur_search or ""
        if new_search == cur:
            raise PreventUpdate
        return new_search

    @app.callback(
        Output("live-preset-download", "data"),
        Input("live-preset-export-btn", "n_clicks"),
        State("live-filter-peril", "value"),
        State("live-filter-min-score", "value"),
        State("live-filter-region", "value"),
        prevent_initial_call=True,
    )
    def export_live_preset(
        n_clicks: Optional[int],
        peril: Optional[str],
        min_sc: Optional[Any],
        region: Optional[str],
    ):
        if not n_clicks:
            raise PreventUpdate
        try:
            ms = float(min_sc) if min_sc is not None else 0.0
        except (TypeError, ValueError):
            ms = 0.0
        payload = {
            "version": 1,
            "peril": peril or "all",
            "min_score": ms,
            "region": region or "",
        }
        body = json.dumps(payload, indent=2)
        return {"content": body, "filename": "catia_live_preset.json"}

    @app.callback(
        Output("live-share-clipboard", "content"),
        Input("live-filter-peril", "value"),
        Input("live-filter-min-score", "value"),
        Input("live-filter-region", "value"),
        State("url", "pathname"),
    )
    def update_live_share_clipboard(
        peril: Optional[str],
        min_sc: Optional[Any],
        region: Optional[str],
        pathname: Optional[str],
    ):
        params: Dict[str, str] = {}
        if peril and peril != "all":
            params["lp"] = peril
        try:
            msv = float(min_sc) if min_sc is not None else 0.0
        except (TypeError, ValueError):
            msv = 0.0
        if msv > 0:
            params["lm"] = str(int(msv)) if float(msv).is_integer() else str(msv)
        if region:
            params["lr"] = region
        qs = urllib.parse.urlencode(params)
        path = pathname or "/"
        suffix = f"{path}?{qs}" if qs else path
        base = os.environ.get("CATIA_PUBLIC_DASH_URL", "").rstrip("/")
        if base:
            return f"{base}{suffix}"
        return suffix

    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8050,
    output_dir: Optional[str] = None,
    api_base_url: str = "http://127.0.0.1:8000",
    debug: bool = False,
) -> None:
    """Run the Dash development server."""
    app = create_dash_app(output_dir=output_dir, api_base_url=api_base_url)
    logger.info("CATIA dashboard at http://%s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard()
