"""
Compliance report template for CATIA.
Aligns with CAS Catastrophe Modeling Guidelines, SOA Risk Management, NAIC Model Act.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from catia import __version__

logger = logging.getLogger(__name__)


def generate_compliance_report(
    audit_metadata: Dict[str, Any],
    results: Dict[str, Any],
    output_path: str = None,
) -> str:
    """
    Generate an HTML compliance report with full audit trail and framework references.

    Args:
        audit_metadata: From create_audit_metadata()
        results: Full analysis results (risk_metrics, mitigation_summary, etc.)
        output_path: If set, write HTML to this path.

    Returns:
        HTML string of the report.
    """
    run_id = audit_metadata.get("run_id", "N/A")
    region = audit_metadata.get("region", "N/A")
    perils = audit_metadata.get("perils", [])
    config_hash = audit_metadata.get("config_hash", "N/A")
    ts = audit_metadata.get("timestamp", datetime.now().isoformat())

    risk = results.get("risk_metrics", {})
    desc = risk.get("descriptive_stats", {})
    risk_metrics = risk.get("risk_metrics", {})
    rp = risk.get("return_periods", {})
    mitigation = results.get("mitigation_summary", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CATIA Compliance Report – {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ border-bottom: 2px solid #333; }}
    h2 {{ color: #444; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .meta {{ background: #f9f9f9; padding: 1rem; border-radius: 6px; margin: 1rem 0; }}
    .framework {{ margin: 1rem 0; padding: 0.75rem; background: #e8f4fc; border-radius: 6px; }}
    footer {{ margin-top: 2rem; font-size: 0.9rem; color: #666; }}
  </style>
</head>
<body>
  <h1>CATIA Compliance Report</h1>
  <p><strong>Run ID:</strong> {run_id} &nbsp;|&nbsp; <strong>Generated:</strong> {ts}</p>

  <h2>1. Audit trail &amp; reproducibility</h2>
  <div class="meta">
    <table>
      <tr><th>Item</th><th>Value</th></tr>
      <tr><td>Run ID</td><td><code>{run_id}</code></td></tr>
      <tr><td>Region</td><td>{region}</td></tr>
      <tr><td>Perils</td><td>{", ".join(perils)}</td></tr>
      <tr><td>Config hash</td><td><code>{config_hash}</code></td></tr>
      <tr><td>CATIA version</td><td>{__version__}</td></tr>
      <tr><td>Timestamp</td><td>{ts}</td></tr>
    </table>
  </div>

  <h2>2. Key risk metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Mean annual loss</td><td>${desc.get('mean', 0):,.0f}</td></tr>
    <tr><td>Median annual loss</td><td>${desc.get('median', 0):,.0f}</td></tr>
    <tr><td>VaR (95%)</td><td>${risk_metrics.get('var', 0):,.0f}</td></tr>
    <tr><td>TVaR (95%)</td><td>${risk_metrics.get('tvar', 0):,.0f}</td></tr>
    <tr><td>100-year return period loss</td><td>${rp.get('100_year', 0):,.0f}</td></tr>
    <tr><td>250-year return period loss</td><td>${rp.get('250_year', 0):,.0f}</td></tr>
  </table>

  <h2>3. Mitigation summary</h2>
  <table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Baseline loss</td><td>${mitigation.get('baseline_loss', 0):,.0f}</td></tr>
    <tr><td>Mitigated loss</td><td>${mitigation.get('mitigated_loss', 0):,.0f}</td></tr>
    <tr><td>Total risk reduction</td><td>{mitigation.get('total_risk_reduction', 0):.2%}</td></tr>
  </table>

  <h2>4. Framework alignment</h2>
  <div class="framework">
    <p><strong>CAS Catastrophe Modeling Guidelines</strong><br>
    This report documents model assumptions (config snapshot in audit), risk metrics (VaR, TVaR, return periods),
    and a reproducible run ID and config hash for auditability.</p>
  </div>
  <div class="framework">
    <p><strong>SOA Risk Management Framework</strong><br>
    Catastrophe risk is quantified via frequency-severity simulation; tail risk via EVT where enabled;
    mitigation strategies and cost-benefit are reported.</p>
  </div>
  <div class="framework">
    <p><strong>NAIC Model Act (insurance applications)</strong><br>
    Outputs support loss modeling and capital adequacy; assumptions and model version are documented
    for regulatory review.</p>
  </div>

  <h2>5. Assumptions</h2>
  <p>Configuration snapshot (perils, simulation parameters, random seed) is stored in the audit metadata
  for this run. Same run_id and config hash allow exact reproduction of results.</p>

  <footer>
    CATIA v{__version__} – Catastrophe AI System for Climate Risk Modeling. Report generated at {ts}.
  </footer>
</body>
</html>
"""
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info("Compliance report written: %s", output_path)
    return html
