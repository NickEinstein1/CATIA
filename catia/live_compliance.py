"""
Attribution, disclaimers, and compliance copy for live / map surfaces.

Observational data and third-party basemaps are not CATIA model output.
"""

from __future__ import annotations

from typing import Any, List

from dash import html


def observational_disclaimer_short() -> str:
    return (
        "Live feeds show third-party observational data for situational awareness only. "
        "Not for underwriting, emergency response dispatch, or regulatory filing. "
        "CATIA modeled loss appears only after you run an analysis and open Global view."
    )


def attribution_footer(*, compact: bool = False) -> html.Div:
    """
    Required-style attribution block for maps and live data (links + disclaimer).
    """
    links: List[Any] = [
        html.A("USGS Earthquake Hazards", href="https://earthquake.usgs.gov/", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("NASA EONET", href="https://eonet.gsfc.nasa.gov/", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("GDACS", href="https://www.gdacs.org/", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("OpenStreetMap", href="https://www.openstreetmap.org/copyright", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("CARTO basemaps", href="https://carto.com/legal/", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("MapLibre", href="https://maplibre.org/", target="_blank", rel="noopener noreferrer", className="catia-link"),
        " · ",
        html.A("deck.gl", href="https://deck.gl/", target="_blank", rel="noopener noreferrer", className="catia-link"),
    ]
    body = [
        html.P(links, className="catia-compliance__links"),
        html.P(observational_disclaimer_short(), className="catia-compliance__disclaimer"),
    ]
    if not compact:
        body.insert(
            0,
            html.P(
                "Respect each provider’s terms, rate limits, and attribution requirements.",
                className="catia-compliance__lead",
            ),
        )
    return html.Div(className="catia-compliance", children=body)
