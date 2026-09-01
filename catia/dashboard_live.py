"""
Live Earth tab UI — rendered from cached feed store (no network in filter path).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from catia.geo_hazards import fig_global_hazard_globe
from catia.geo_deck import build_live_deck_earth_map
from catia.geo_osm import build_osm_live_catastrophe_map
from catia.live_catastrophe_feeds import LiveFeedResult
from catia.live_compliance import attribution_footer
from catia.live_service import enrich_stored_live_events


def build_live_earth_content(
    *,
    store: Dict[str, Any],
    report: Optional[Dict[str, Any]],
    peril_sel: Optional[str],
    min_score: Optional[Any],
    region_sel: Optional[str],
    compare_mode: Optional[str],
) -> html.Div:
    """Render Live Earth from a cached feed payload; scoring filters only (fast)."""
    from catia.dashboard import (
        _GLOBE_MAX_POINTS,
        _PLOTLY_UI_CONFIG,
        _category_chips,
        _live_alert_hits_banner,
        _live_feed_health_strip,
        _live_feed_legend,
        _live_kpi_strip,
        _score_badge,
        _style_dark_chart,
        fig_live_catastrophe_globe,
    )

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

    live_payload = enrich_stored_live_events(
        store,
        focal_region=focal_eff,
        peril_filter=peril_arg,
        min_score=min_sc,
    )
    events = live_payload["events"]
    raw_n = int(live_payload.get("count_raw") or 0)
    feed = LiveFeedResult(
        events=events,
        errors=list(live_payload.get("errors") or []),
        fetched_at_iso=str(live_payload.get("fetched_at_iso") or ""),
        sources_ok=dict(live_payload.get("sources_ok") or {}),
        latency_ms=dict(live_payload.get("latency_ms") or {}),
        http_status=dict(live_payload.get("http_status") or {}),
        cache_hit=bool(live_payload.get("cache_hit")),
        cache_backend=str(live_payload.get("cache_backend") or "memory"),
        offline_mode=bool(live_payload.get("offline_mode")),
    )

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
    if live_map is not None:
        map_section: Any = html.Div(
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

    top_alert_rows: List[Any] = [
        html.Tr(
            [
                html.Th("Score"),
                html.Th("Peril"),
                html.Th("Where / what"),
                html.Th("Type"),
                html.Th("Source"),
                html.Th("When / detail"),
                html.Th("Conf / exposure"),
            ],
            className="catia-table__headrow",
        )
    ]
    for e in events[:12]:
        exp = e.get("exposure_overlap") or {}
        exp_txt = ", ".join(exp.get("regions") or [])[:40] or "—"
        conf_txt = f"{float(e.get('confidence') or 0):.0%}" if e.get("confidence") is not None else "—"
        top_alert_rows.append(
            html.Tr(
                className="catia-table__row",
                children=[
                    html.Td(_score_badge(float(e.get("catia_score") or 0.0)), className="catia-table__num"),
                    html.Td(str(e.get("catia_peril") or "—")),
                    html.Td(str(e.get("title", ""))[:90]),
                    html.Td(str(e.get("category_label", ""))[:40]),
                    html.Td(str(e.get("source", ""))),
                    html.Td(
                        " ".join(x for x in (e.get("time_iso"), e.get("severity_label")) if x) or "—"
                    ),
                    html.Td(f"{conf_txt} · {exp_txt}"),
                ],
            )
        )

    table_rows: List[Any] = [
        html.Tr(
            [
                html.Th("Score"),
                html.Th("CATIA peril"),
                html.Th("Where / what"),
                html.Th("Type"),
                html.Th("Source"),
                html.Th("When / detail"),
                html.Th("Conf / exposure"),
            ],
            className="catia-table__headrow",
        )
    ]
    for e in events[:40]:
        exp = e.get("exposure_overlap") or {}
        exp_txt = ", ".join(exp.get("regions") or [])[:40] or "—"
        conf_txt = f"{float(e.get('confidence') or 0):.0%}" if e.get("confidence") is not None else "—"
        table_rows.append(
            html.Tr(
                className="catia-table__row",
                children=[
                    html.Td(_score_badge(float(e.get("catia_score") or 0.0)), className="catia-table__num"),
                    html.Td(str(e.get("catia_peril") or "—")),
                    html.Td(str(e.get("title", ""))[:80]),
                    html.Td(str(e.get("category_label", ""))[:40]),
                    html.Td(str(e.get("source", ""))),
                    html.Td(
                        " ".join(x for x in (e.get("time_iso"), e.get("severity_label")) if x) or "—"
                    ),
                    html.Td(f"{conf_txt} · {exp_txt}"),
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
                    html.H3("Live globe", className="catia-section-head__title"),
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
            dcc.Graph(figure=globe_live, style={"minHeight": "520px"}, config=_PLOTLY_UI_CONFIG),
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
                html.Div(className="catia-split-grid__col", children=[globe_live_panel]),
                html.Div(className="catia-split-grid__col", children=[map_section]),
            ],
        )

    deck_gl = build_live_deck_earth_map(events)
    if deck_gl is not None:
        deck_section: Any = html.Div(
            className="catia-panel catia-panel--deck",
            style={"padding": "12px"},
            children=[
                html.Div(
                    className="catia-section-head",
                    children=[
                        html.H3("Deck.gl + MapLibre (GPU)", className="catia-section-head__title"),
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
                        ". Follow basemap provider terms.",
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
                html.H3("Deck.gl + MapLibre (GPU)", style={"marginTop": 0}),
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
    live_blocks.extend(
        [
            kpi,
            _live_feed_health_strip(feed),
            type_breakdown,
            html.Div(
                className="catia-panel",
                style={"padding": "12px", "overflowX": "auto"},
                children=[
                    html.Div(
                        className="catia-section-head",
                        children=[
                            html.H3("Top alerts", className="catia-section-head__title"),
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
                "Globe / Leaflet / Deck layers combine USGS, NASA EONET, and GDACS when enabled. "
                "Observational activity only — not CATIA modeled loss.",
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
    hits_banner = _live_alert_hits_banner(events)
    if hits_banner is not None:
        live_blocks.insert(3, hits_banner)

    return html.Div(live_blocks)
