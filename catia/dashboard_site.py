"""
Dashboard UI for site / property viability assessment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html

from catia.geo_osm import build_osm_live_catastrophe_map


def _category_class(category: str) -> str:
    c = (category or "moderate").lower()
    return f"catia-risk-badge catia-risk-badge--{c}"


def build_site_assessment_panel(result: Optional[Dict[str, Any]]) -> html.Div:
    """Render assessment results (or empty state)."""
    if not result:
        return html.Div(
            className="catia-panel",
            children=[
                html.H3("Site viability", className="catia-section-head__title"),
                html.P(
                    "Enter coordinates (or address) and property intent, then run assessment. "
                    "CATIA maps the site to regional peril load, topography stubs, and "
                    "insurance-oriented screening guidance.",
                    className="catia-section-head__sub",
                ),
            ],
        )

    loc = result.get("location") or {}
    nearest = loc.get("nearest_region") or {}
    insurance = result.get("insurance_viability") or {}
    topo = result.get("topography") or {}
    hazards = result.get("hazard_assessment") or []
    category = str(result.get("risk_category") or "moderate")
    score = float(result.get("risk_score") or 0.0)

    hazard_rows: List[Any] = [
        html.Tr(
            [
                html.Th("Peril"),
                html.Th("Score"),
                html.Th("Freq"),
                html.Th("Topo ×"),
                html.Th("Notes"),
            ],
            className="catia-table__headrow",
        )
    ]
    for h in hazards[:8]:
        hazard_rows.append(
            html.Tr(
                className="catia-table__row",
                children=[
                    html.Td(str(h.get("name") or h.get("peril"))),
                    html.Td(f"{float(h.get('hazard_score') or 0):.0f}"),
                    html.Td(f"{float(h.get('frequency_base') or 0):.2f}"),
                    html.Td(f"{float(h.get('topography_factor') or 1):.2f}"),
                    html.Td(str(h.get("notes") or "")[:80]),
                ],
            )
        )

    guidance_items = []
    for g in insurance.get("property_guidance") or []:
        guidance_items.append(html.Li(g))
    for g in insurance.get("peril_guidance") or []:
        guidance_items.append(html.Li(g))
    for g in insurance.get("suggested_actions") or []:
        guidance_items.append(html.Li(g))

    pin_events = [
        {
            "lat": loc.get("lat"),
            "lon": loc.get("lon"),
            "title": f"Site · {result.get('property_type')}",
            "category": "site",
            "category_label": "Assessed site",
            "source": "CATIA Site",
            "catia_score": score,
            "severity_label": category,
            "confidence": 0.85,
            "geometry": {
                "type": "Point",
                "coordinates": [float(loc.get("lon") or 0), float(loc.get("lat") or 0)],
            },
            "geometry_kind": "point",
        }
    ]
    site_map = build_osm_live_catastrophe_map(pin_events, height="360px", zoom=5, cluster_markers=False)
    map_block: Any
    if site_map is not None:
        map_block = html.Div(className="catia-panel catia-panel--tight", style={"padding": "12px"}, children=[site_map])
    else:
        map_block = html.Div(
            className="catia-panel",
            children=[html.P("Install dash-leaflet to see the site map.", style={"color": "#94a3b8"})],
        )

    sim = result.get("indicative_simulation")
    sim_block: Any = html.Div()
    if isinstance(sim, dict) and not sim.get("error"):
        sim_block = html.Div(
            className="catia-panel",
            children=[
                html.H3("Indicative simulation", style={"marginTop": 0}),
                html.P(
                    f"TIV ${float(sim.get('tiv') or 0):,.0f} · "
                    f"iterations {sim.get('iterations')} · "
                    f"mean ${float(sim.get('mean_annual_loss') or 0):,.0f} · "
                    f"VaR95 ${float(sim.get('var_95') or 0):,.0f}",
                ),
                html.P(str(sim.get("note") or ""), className="catia-footnote"),
            ],
        )
    elif isinstance(sim, dict) and sim.get("error"):
        sim_block = html.Div(
            className="catia-flash catia-flash--warn",
            children=[html.P(f"Simulation skipped: {sim.get('error')}", className="catia-flash__text")],
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
                                    html.Div("Risk score", className="catia-kpi-card__label"),
                                    html.Div(f"{score:.0f}", className="catia-kpi-card__value"),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Category", className="catia-kpi-card__label"),
                                    html.Div(
                                        html.Span(
                                            category.upper(),
                                            className=_category_class(category),
                                        ),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Nearest region", className="catia-kpi-card__label"),
                                    html.Div(
                                        str(nearest.get("region_label") or "—"),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="catia-kpi-card",
                                children=[
                                    html.Div("Insurance screen", className="catia-kpi-card__label"),
                                    html.Div(
                                        str(insurance.get("label") or insurance.get("status") or "—"),
                                        className="catia-kpi-card__value catia-kpi-card__value--sm",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.P(
                        f"{loc.get('lat'):.4f}, {loc.get('lon'):.4f} · "
                        f"{nearest.get('distance_km', '—')} km from region centroid · "
                        f"intent: {result.get('property_type')}",
                        className="catia-kpi-strip__meta",
                    ),
                ],
            ),
            html.Div(
                className="catia-split-grid",
                children=[
                    html.Div(className="catia-split-grid__col", children=[map_block]),
                    html.Div(
                        className="catia-split-grid__col",
                        children=[
                            html.Div(
                                className="catia-panel",
                                children=[
                                    html.H3("Insurance / reinsurance screening", style={"marginTop": 0}),
                                    html.P(str(insurance.get("narrative") or "")),
                                    html.P(
                                        str(insurance.get("reinsurance_notes") or ""),
                                        className="catia-section-head__sub",
                                    ),
                                    html.Ul(guidance_items or [html.Li("No guidance generated.")]),
                                ],
                            ),
                            html.Div(
                                className="catia-panel",
                                children=[
                                    html.H3("Topography / floodplain", style={"marginTop": 0}),
                                    html.Table(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td("Data quality"),
                                                    html.Td(str(topo.get("data_quality"))),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Providers"),
                                                    html.Td(
                                                        ", ".join(topo.get("providers") or [])
                                                        or "heuristic"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Elevation"),
                                                    html.Td(
                                                        f"{topo.get('elevation_m', topo.get('elevation_m_estimate'))} m"
                                                        + (
                                                            f" ({topo.get('elevation_source')})"
                                                            if topo.get("elevation_source")
                                                            else ""
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Elevation band"),
                                                    html.Td(str(topo.get("elevation_band"))),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Slope"),
                                                    html.Td(
                                                        (
                                                            f"{topo.get('slope_percent')}% · "
                                                            if topo.get("slope_percent") is not None
                                                            else ""
                                                        )
                                                        + str(topo.get("slope_class"))
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("FEMA flood zone"),
                                                    html.Td(
                                                        str(topo.get("flood_zone") or "—")
                                                        + (
                                                            " (SFHA)"
                                                            if topo.get("sfha")
                                                            else ""
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Base flood elev."),
                                                    html.Td(
                                                        (
                                                            f"{topo.get('floodplain_detail', {}).get('base_flood_elevation')} "
                                                            f"{topo.get('floodplain_detail', {}).get('vertical_datum') or ''}".strip()
                                                            if isinstance(topo.get("floodplain_detail"), dict)
                                                            and topo.get("floodplain_detail", {}).get(
                                                                "base_flood_elevation"
                                                            )
                                                            is not None
                                                            else "—"
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Flood risk class"),
                                                    html.Td(
                                                        str(
                                                            topo.get("flood_risk_class")
                                                            or topo.get("floodplain_hint")
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td("Coastal hint"),
                                                    html.Td(str(topo.get("coastal_proximity_hint"))),
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.P(str(topo.get("notes") or ""), className="catia-footnote"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="catia-panel",
                style={"padding": "12px", "overflowX": "auto"},
                children=[
                    html.H3("Hazard assessment by peril", style={"marginTop": 0}),
                    html.Table(
                        [html.Thead(hazard_rows[0]), html.Tbody(hazard_rows[1:])],
                        className="catia-table",
                    ),
                ],
            ),
            sim_block,
            html.P(str(result.get("disclaimer") or ""), className="globe-caption"),
        ]
    )
