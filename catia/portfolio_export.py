"""
Portfolio underwriting export pack: JSON + CSV tables + HTML summary as a ZIP.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _money(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return ""


def _csv_table(headers: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def _summary_csv(result: Dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    book = result.get("book") or {}
    agg = book.get("aggregate") or {}
    net = book.get("aggregate_net") or {}
    rows = [
        ["location_count", summary.get("location_count")],
        ["total_tiv", summary.get("total_tiv")],
        ["region_count", summary.get("region_count")],
        ["book_aal_gross", agg.get("mean")],
        ["book_var95_gross", agg.get("var_95")],
        ["book_tvar95_gross", agg.get("tvar_95")],
        ["book_aal_net", net.get("mean")],
        ["book_var95_net", net.get("var_95")],
        ["analyzed_at", result.get("analyzed_at")],
    ]
    return _csv_table(["metric", "value"], rows)


def _peril_csv(result: Dict[str, Any]) -> str:
    book = result.get("book") or {}
    rows = []
    for peril, m in (book.get("by_peril") or {}).items():
        rows.append(
            [
                peril,
                m.get("name") or peril,
                m.get("mean"),
                m.get("var_95"),
                m.get("tvar_95"),
            ]
        )
    return _csv_table(["peril", "name", "aal", "var_95", "tvar_95"], rows)


def _region_csv(result: Dict[str, Any]) -> str:
    rows = []
    for row in result.get("by_region") or []:
        agg = row.get("aggregate") or {}
        rows.append(
            [
                row.get("group"),
                row.get("location_count"),
                row.get("total_tiv"),
                agg.get("mean"),
                agg.get("var_95"),
            ]
        )
    return _csv_table(
        ["region", "location_count", "total_tiv", "aal", "var_95"], rows
    )


def _storm_csv(result: Dict[str, Any]) -> str:
    storm = result.get("one_storm") or {}
    if not storm:
        return _csv_table(["metric", "value"], [["one_storm", "none"]])
    rein = storm.get("reinsurance") or {}
    rows = [
        ["peril", storm.get("peril")],
        ["intensity", storm.get("intensity")],
        ["mode", storm.get("mode")],
        ["hit_count", storm.get("hit_count")],
        ["tiv_hit", storm.get("tiv_hit")],
        ["gross_loss", storm.get("total_loss")],
        ["net_loss", rein.get("net_loss")],
        ["recovered", rein.get("recovered")],
        ["live_event_id", storm.get("live_event_id")],
        ["live_event_title", storm.get("live_event_title")],
    ]
    return _csv_table(["metric", "value"], rows)


def _locations_csv(result: Dict[str, Any]) -> str:
    rows = []
    for loc in result.get("locations") or []:
        rows.append(
            [
                loc.get("id"),
                loc.get("lat"),
                loc.get("lon"),
                loc.get("tiv"),
                loc.get("region_id"),
                loc.get("flood_zone"),
            ]
        )
    return _csv_table(
        ["id", "lat", "lon", "tiv", "region_id", "flood_zone"], rows
    )


def _html_report(result: Dict[str, Any]) -> str:
    summary = result.get("summary") or {}
    book = result.get("book") or {}
    agg = book.get("aggregate") or {}
    net = book.get("aggregate_net") or {}
    storm = result.get("one_storm") or {}
    rein = (storm or {}).get("reinsurance") or {}
    layers = result.get("reinsurance_layers") or []

    def esc(x: Any) -> str:
        return (
            str(x)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    layer_rows = "".join(
        f"<tr><td>{esc(L.get('name'))}</td><td>{esc(L.get('attachment'))}</td>"
        f"<td>{esc(L.get('limit'))}</td><td>{esc(L.get('share'))}</td></tr>"
        for L in layers
    )
    storm_html = ""
    if storm:
        storm_html = f"""
        <h2>One-storm / live hit</h2>
        <p>{esc(storm.get('live_event_title') or storm.get('peril'))}

           intensity {esc(storm.get('intensity'))} · hits {esc(storm.get('hit_count'))}</p>
        <ul>
          <li>Gross loss: ${_money(storm.get('total_loss'))}</li>
          <li>Recovered: ${_money(rein.get('recovered'))}</li>
          <li>Net loss: ${_money(rein.get('net_loss') or storm.get('total_loss'))}</li>
        </ul>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>CATIA Portfolio Pack</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #1a1a1a; }}
    h1 {{ font-size: 1.6rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    .note {{ color: #555; font-size: 0.9rem; max-width: 40rem; }}
  </style>
</head>
<body>
  <h1>CATIA portfolio export pack</h1>
  <p>Analyzed {esc(result.get('analyzed_at'))}</p>
  <h2>Book summary</h2>
  <ul>
    <li>Locations: {esc(summary.get('location_count'))}</li>
    <li>TIV: ${_money(summary.get('total_tiv'))}</li>
    <li>Gross AAL: ${_money(agg.get('mean'))} · VaR95 ${_money(agg.get('var_95'))}</li>
    <li>Net AAL: ${_money(net.get('mean'))} · VaR95 ${_money(net.get('var_95'))}</li>
  </ul>
  <h2>Reinsurance layers</h2>
  <table>
    <tr><th>Name</th><th>Attachment</th><th>Limit</th><th>Share</th></tr>
    {layer_rows or '<tr><td colspan="4">None</td></tr>'}
  </table>
  {storm_html}
  <p class="note">{esc(result.get('disclaimer') or '')}</p>
</body>
</html>
"""


def build_portfolio_export_files(result: Dict[str, Any]) -> Dict[str, str]:
    """Return filename → text content for the pack."""
    return {
        "portfolio_report.json": json.dumps(result, indent=2, default=str),
        "summary.csv": _summary_csv(result),
        "by_peril.csv": _peril_csv(result),
        "by_region.csv": _region_csv(result),
        "one_storm.csv": _storm_csv(result),
        "locations.csv": _locations_csv(result),
        "portfolio_pack.html": _html_report(result),
    }


def build_portfolio_export_zip(result: Dict[str, Any]) -> Tuple[bytes, str]:
    """Build in-memory ZIP bytes and suggested filename."""
    files = build_portfolio_export_files(result)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"catia_portfolio_pack_{stamp}.zip"
    return buf.getvalue(), filename
