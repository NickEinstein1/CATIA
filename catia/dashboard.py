"""
CATIA system dashboard — interactive front-end with global hazard globe.

Run:  catia --dashboard
Or:   python -m catia.dashboard

Uses Dash + Plotly orthographic globe. Reads outputs/catia_report.json for loss-weighted overlays.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html

from catia import __version__
from catia.config import CLIMATE_SCENARIOS, OUTPUT_CONFIG, PERIL_CONFIG
from catia.geo_hazards import PERIL_VIS_COLORS, fig_global_hazard_globe
from catia.geo_osm import build_osm_leaflet_map, build_osm_live_catastrophe_map
from catia.live_catastrophe_feeds import category_color, fetch_all_live_events

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


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


def fig_live_catastrophe_globe(events: List[Dict[str, Any]]) -> go.Figure:
    """Orthographic globe with live USGS / EONET points."""
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
                    size=11,
                    color=[category_color(str(e.get("category") or "")) for e in events],
                    line=dict(width=1, color="#0f172a"),
                ),
                text=[
                    f"{str(e.get('title', ''))[:70]}<br>"
                    f"{e.get('category_label', '')} · {e.get('source', '')} {e.get('severity_label', '')}"
                    for e in events
                ],
                hoverinfo="text",
            )
        ],
        layout=dict(
            title="Near–real-time events (USGS earthquakes + NASA EONET)",
            height=560,
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


def _live_feed_legend() -> html.Div:
    samples = [
        ("earthquake", "Earthquake (USGS)"),
        ("wildfires", "Wildfires (EONET)"),
        ("severe_storms", "Severe storms (EONET)"),
        ("volcanoes", "Volcanoes (EONET)"),
        ("floods", "Floods (EONET)"),
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
        title="CATIA · Global command",
        suppress_callback_exceptions=True,
        assets_folder=str(_ASSETS_DIR),
    )
    app._catia_output_dir = str(out)  # type: ignore[attr-defined]
    app._catia_api_base = api_base_url  # type: ignore[attr-defined]

    app.layout = html.Div(
        className="catia-future-root",
        children=[
            html.Div(
                className="catia-hero",
                children=[
                    html.H1("CATIA"),
                    html.P("Catastrophe AI · Global hazard intelligence", className="sub"),
                    html.Div(
                        className="badge",
                        children=f"BUILD {__version__} · OUTPUT {out.name}/",
                    ),
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
                    dcc.Tab(label="Global view", value="tab-globe"),
                    dcc.Tab(label="Live Earth", value="tab-live"),
                    dcc.Tab(label="Overview", value="tab-overview"),
                    dcc.Tab(label="Latest run", value="tab-run"),
                    dcc.Tab(label="Charts", value="tab-charts"),
                    dcc.Tab(label="Perils & scenarios", value="tab-perils"),
                    dcc.Tab(label="Assumptions", value="tab-assumptions"),
                    dcc.Tab(label="API & files", value="tab-api"),
                ],
                style={"marginTop": "20px"},
            ),
            html.Div(id="tab-content", style={"marginTop": "16px"}),
            dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
            dcc.Interval(
                id="live-feed-interval",
                interval=int(os.environ.get("CATIA_LIVE_REFRESH_MS", "180000")),
                n_intervals=0,
            ),
        ],
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("dash-tabs", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("live-feed-interval", "n_intervals"),
    )
    def render_tab(active: str, _n: int, _live_n: int):
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
                            dcc.Graph(figure=globe, style={"minHeight": "640px"}),
                        ],
                    ),
                    html.P(caption, className="globe-caption"),
                    _peril_legend_row(),
                    osm_section,
                ]
            )

        if active == "tab-live":
            feed = fetch_all_live_events(force=force_live)
            events = feed.events
            err_banner: Optional[html.Div] = None
            if feed.errors:
                err_banner = html.Div(
                    className="catia-panel",
                    style={
                        "padding": "10px",
                        "borderLeft": "3px solid #f97316",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.P(
                            "Some feeds failed (partial data). " + " · ".join(feed.errors),
                            style={"color": "#fdba74", "margin": 0},
                        ),
                    ],
                )
            globe_live = fig_live_catastrophe_globe(events)
            live_map = build_osm_live_catastrophe_map(events)
            map_section: Any
            if live_map is not None:
                map_section = html.Div(
                    className="catia-panel",
                    style={"padding": "12px"},
                    children=[
                        html.H3(
                            "OpenStreetMap — same events (pan/zoom)",
                            style={"marginTop": 0, "marginBottom": "8px"},
                        ),
                        html.P(
                            "Data © OpenStreetMap contributors. Event data © USGS / NASA.",
                            style={"color": "#94a3b8", "fontSize": "0.88rem", "marginBottom": "12px"},
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
            # Small summary counts by coarse category
            counts: Dict[str, int] = {}
            for e in events:
                k = str(e.get("category_label") or e.get("category") or "?")
                counts[k] = counts.get(k, 0) + 1
            top_counts = sorted(counts.items(), key=lambda x: -x[1])[:12]
            summary = html.Div(
                className="catia-panel",
                style={"padding": "12px"},
                children=[
                    html.H3("Snapshot", style={"marginTop": 0}),
                    html.P(
                        f"Last updated: {feed.fetched_at_iso} · {len(events)} point(s) on map.",
                        style={"color": "#94a3b8"},
                    ),
                    html.Ul([html.Li(f"{k}: {v}") for k, v in top_counts] or [html.Li("—")]),
                    html.P(
                        [
                            "Sources: ",
                            html.A(
                                "USGS earthquake feeds",
                                href="https://earthquake.usgs.gov/earthquakes/feed/",
                                target="_blank",
                                rel="noopener noreferrer",
                                style={"color": "#22d3ee"},
                            ),
                            " · ",
                            html.A(
                                "NASA EONET",
                                href="https://eonet.gsfc.nasa.gov/",
                                target="_blank",
                                rel="noopener noreferrer",
                                style={"color": "#22d3ee"},
                            ),
                            ". Polling interval can be set with ",
                            html.Code("CATIA_LIVE_REFRESH_MS"),
                            " (ms).",
                        ],
                        style={"fontSize": "0.88rem", "color": "#94a3b8", "marginTop": "8px"},
                    ),
                ],
            )
            table_rows = [
                html.Tr([html.Th("Where / what"), html.Th("Type"), html.Th("Source"), html.Th("When / detail")])
            ]
            for e in events[:40]:
                table_rows.append(
                    html.Tr([
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
                    ])
                )
            live_blocks: List[Any] = [summary]
            if err_banner is not None:
                live_blocks.insert(0, err_banner)
            live_blocks.extend(
                [
                    html.Div(
                        className="catia-panel",
                        style={"padding": "12px 12px 4px"},
                        children=[
                            html.H3("Orthographic globe", style={"marginTop": 0, "marginBottom": "8px"}),
                            dcc.Graph(figure=globe_live, style={"minHeight": "580px"}),
                        ],
                    ),
                    html.P(
                        "Markers combine USGS (typically M≥2.5, last 24h) and NASA EONET open events. "
                        "This is observational activity, not CATIA modeled loss.",
                        className="globe-caption",
                    ),
                    _live_feed_legend(),
                    map_section,
                    html.Div(
                        className="catia-panel",
                        style={"padding": "12px", "overflowX": "auto"},
                        children=[
                            html.H3("Recent rows (up to 40)", style={"marginTop": 0}),
                            html.Table(table_rows),
                        ],
                    ),
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
                            html.Li("Open Global view for the live globe after each run."),
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
                        html.Div(className="catia-panel", style={"padding": "8px"}, children=[dcc.Graph(figure=f)])
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
