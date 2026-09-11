"""
Dashboard UI for portfolio accumulation analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import html

from catia.geo_osm import build_osm_live_catastrophe_map


def _money(v: Any) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _metrics_row(label: str, metrics: Optional[Dict[str, Any]]) -> html.Tr:
    m = metrics or {}
    return html.Tr(
        className="catia-table__row",
        children=[
            html.Td(label),
            html.Td(_money(m.get("mean"))),
            html.Td(_money(m.get("var_95"))),
            html.Td(_money(m.get("tvar_95"))),
        ],
    )


def build_portfolio_panel(result: Optional[Dict[str, Any]]) -> html.Div:
    """Render portfolio accumulation results (or empty state)."""
    if not result:
        return html.Div(
            className="catia-panel",
            children=[
                html.H3("Portfolio accumulation", className="catia-section-head__title"),
                html.P(
                    "Upload a CSV or GeoJSON of locations (lat, lon, tiv), then run accumulation. "
                    "CATIA maps each site to a region, rolls up indicative AAL/VaR by peril and zone, "
                    "and can apply a single-storm shock to the book.",
                    className="catia-section-head__sub",
                ),
                html.P(
                    "CSV headers: lat, lon, tiv [, id, construction_type, occupancy, flood_zone, region]",
                    className="catia-footnote",
                ),
            ],
        )

    summary = result.get("summary") or {}
    book = result.get("book") or {}
    locations = result.get("locations") or []
    by_region = result.get("by_region") or []
    by_zone = result.get("by_flood_zone") or []
    storm = result.get("one_storm")

    map_events = []
    for loc in locations[:400]:
        map_events.append(
            {
                "lat": loc.get("lat"),
                "lon": loc.get("lon"),
                "title": f"{loc.get('id')} · TIV {_money(loc.get('tiv'))}",
                "category": "portfolio",
                "category_label": loc.get("region_id") or "location",
                "source": "Portfolio",
                "catia_score": min(100.0, float(loc.get("tiv") or 0) / 50_000.0),
                "severity_label": loc.get("flood_zone") or "",
                "confidence": 0.8,
            }
        )
    site_map = build_osm_live_catastrophe_map(
        map_events, height="380px", zoom=3, cluster_markers=True
    )
    map_block: Any
    if site_map is not None:
        map_block = html.Div(
            className="catia-panel catia-panel--tight",
            style={"padding": "12px"},
            children=[site_map],
        )
    else:
        map_block = html.Div(
            className="catia-panel",
            children=[html.P("Install dash-leaflet for the portfolio map.", style={"color": "#94a3b8"})],
        )

    peril_rows: List[Any] = [
        html.Tr(
            [html.Th("Peril"), html.Th("AAL (mean)"), html.Th("VaR 95%"), html.Th("TVaR 95%")],
            className="catia-table__headrow",
        )
    ]
    for peril, metrics in (book.get("by_peril") or {}).items():
        peril_rows.append(_metrics_row(str(metrics.get("name") or peril), metrics))
    if book.get("aggregate"):
        peril_rows.append(_metrics_row("Aggregate", book.get("aggregate")))

    region_rows: List[Any] = [
        html.Tr(
            [
                html.Th("Region"),
                html.Th("Locs"),
                html.Th("TIV"),
                html.Th("AAL"),
                html.Th("VaR 95%"),
            ],
            className="catia-table__headrow",
        )
    ]
    for row in by_region:
        agg = row.get("aggregate") or {}
        region_rows.append(
            html.Tr(
                className="catia-table__row",
                children=[
                    html.Td(str(row.get("group"))),
                    html.Td(str(row.get("location_count"))),
                    html.Td(_money(row.get("total_tiv"))),
                    html.Td(_money(agg.get("mean"))),
                    html.Td(_money(agg.get("var_95"))),
                ],
            )
        )

    zone_rows: List[Any] = [
        html.Tr(
            [
                html.Th("Flood zone"),
                html.Th("Locs"),
                html.Th("TIV"),
                html.Th("AAL"),
                html.Th("VaR 95%"),
            ],
            className="catia-table__headrow",
        )
    ]
    for row in by_zone:
        agg = row.get("aggregate") or {}
        zone_rows.append(
            html.Tr(
                className="catia-table__row",
                children=[
                    html.Td(str(row.get("group"))),
                    html.Td(str(row.get("location_count"))),
                    html.Td(_money(row.get("total_tiv"))),
                    html.Td(_money(agg.get("mean"))),
                    html.Td(_money(agg.get("var_95"))),
                ],
            )
        )

    storm_block: Any = html.Div()
    if isinstance(storm, dict):
        rein = storm.get("reinsurance") or {}
        live_title = storm.get("live_event_title")
        storm_block = html.Div(
            className="catia-panel",
            children=[
                html.H3(
                    "Live event / one storm hits the book" if live_title else "One storm hits the book",
                    style={"marginTop": 0},
                ),
                html.P(
                    (
                        f"{live_title} · "
                        if live_title
                        else ""
                    )
                    + f"{storm.get('peril')} @ intensity {storm.get('intensity')} · "
                    f"damage ratio {storm.get('damage_ratio')} · mode {storm.get('mode')}"
                    + (
                        f" · region {storm.get('region_id')}"
                        if storm.get("region_id")
                        else f" · radius {storm.get('radius_km')} km"
                    )
                    + (
                        f" · intensity via {storm.get('intensity_source')}"
                        if storm.get("intensity_source")
                        else ""
                    )
                ),
                html.Div(
                    className="catia-kpi-grid",
                    children=[
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("Locations hit", className="catia-kpi-card__label"),
                                html.Div(str(storm.get("hit_count")), className="catia-kpi-card__value"),
                            ],
                        ),
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("TIV hit", className="catia-kpi-card__label"),
                                html.Div(_money(storm.get("tiv_hit")), className="catia-kpi-card__value catia-kpi-card__value--sm"),
                            ],
                        ),
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("Gross loss", className="catia-kpi-card__label"),
                                html.Div(
                                    _money(storm.get("total_loss")),
                                    className="catia-kpi-card__value catia-kpi-card__value--sm",
                                ),
                            ],
                        ),
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("Net after XL", className="catia-kpi-card__label"),
                                html.Div(
                                    _money(rein.get("net_loss") if rein else storm.get("total_loss")),
                                    className="catia-kpi-card__value catia-kpi-card__value--sm",
                                ),
                            ],
                        ),
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("Recovered", className="catia-kpi-card__label"),
                                html.Div(
                                    _money(rein.get("recovered") or 0),
                                    className="catia-kpi-card__value catia-kpi-card__value--sm",
                                ),
                            ],
                        ),
                        html.Div(
                            className="catia-kpi-card",
                            children=[
                                html.Div("% of book TIV", className="catia-kpi-card__label"),
                                html.Div(
                                    f"{100 * float(storm.get('loss_ratio_of_book') or 0):.1f}%",
                                    className="catia-kpi-card__value catia-kpi-card__value--sm",
                                ),
                            ],
                        ),
                    ],
                ),
                html.P(str(storm.get("note") or ""), className="catia-footnote"),
            ],
        )

    rein_block: Any = html.Div()
    layers = result.get("reinsurance_layers") or []
    book_net = book.get("aggregate_net") or {}
    book_rein = book.get("reinsurance") or {}
    if layers or book_net:
        rein_block = html.Div(
            className="catia-panel",
            children=[
                html.H3("Reinsurance (XL)", style={"marginTop": 0}),
                html.P(
                    f"Mean recovered {_money(book_rein.get('mean_recovered'))} · "
                    f"net AAL {_money(book_net.get('mean'))} · "
                    f"net VaR95 {_money(book_net.get('var_95'))}"
                    if book_net
                    else "Layers applied to one-storm gross only (simulation off or empty book metrics).",
                    className="catia-section-head__sub",
                ),
                html.Ul(
                    [
                        html.Li(
                            f"{L.get('name') or 'XL'}: attach {_money(L.get('attachment'))} · "
                            f"limit {_money(L.get('limit'))} · share {L.get('share')}"
                        )
                        for L in layers
                    ]
                    or [html.Li("No layers")]
                ),
            ],
        )

    return html.Div(
        children=[
            html.Div(
                className="catia-kpi-strip",
                children=[
                    html.Div(
                        className="catia-kpi-grid",
                        children=[
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Locations", className="catia-kpi-card__label"),
                                    html.Div(
                                        str(summary.get("location_count") or 0),
                                        className="catia-kpi-card__value",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Book TIV", className="catia-kpi-card__label"),
                                    html.Div(
                                        _money(summary.get("total_tiv")),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Regions", className="catia-kpi-card__label"),
                                    html.Div(
                                        str(summary.get("region_count") or 0),
                                        className="catia-kpi-card__value",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Book AAL (gross)", className="catia-kpi-card__label"),
                                    html.Div(
                                        _money((book.get("aggregate") or {}).get("mean")),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Book AAL (net)", className="catia-kpi-card__label"),
                                    html.Div(
                                        _money((book.get("aggregate_net") or book.get("aggregate") or {}).get("mean")),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.P(
                        f"Analyzed {result.get('analyzed_at')} · perils: {', '.join(result.get('perils') or [])}",
                        className="catia-kpi-strip__meta",
                    ),
                ],
            ),
            map_block,
            rein_block,
            storm_block,
            html.Div(
                className="catia-panel",
                style={"padding": "12px", "overflowX": "auto"},
                children=[
                    html.H3("AAL / VaR by peril", style={"marginTop": 0}),
                    html.Table(
                        [html.Thead(peril_rows[0]), html.Tbody(peril_rows[1:])],
                        className="catia-table",
                    ),
                ],
            ),
            html.Div(
                className="catia-split-grid",
                children=[
                    html.Div(
                        className="catia-split-grid__col",
                        children=[
                            html.Div(
                                className="catia-panel",
                                style={"padding": "12px", "overflowX": "auto"},
                                children=[
                                    html.H3("By region", style={"marginTop": 0}),
                                    html.Table(
                                        [html.Thead(region_rows[0]), html.Tbody(region_rows[1:] or [])],
                                        className="catia-table",
                                    ),
                                ],
                            )
                        ],
                    ),
                    html.Div(
                        className="catia-split-grid__col",
                        children=[
                            html.Div(
                                className="catia-panel",
                                style={"padding": "12px", "overflowX": "auto"},
                                children=[
                                    html.H3("By flood zone", style={"marginTop": 0}),
                                    html.Table(
                                        [html.Thead(zone_rows[0]), html.Tbody(zone_rows[1:] or [])],
                                        className="catia-table",
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            html.P(str(result.get("disclaimer") or ""), className="globe-caption"),
        ]
    )
