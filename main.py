"""
CATIA root entry point for ``python main.py``.

The full workflow lives in :mod:`catia.pipeline`; this module configures logging
when executed as a script and re-exports ``run_catia_analysis`` for callers that
have the repo root on ``sys.path``.
"""

import logging
import sys

from catia.config import LOGGING_CONFIG
from catia.pipeline import run_catia_analysis

try:
    from catia.logging_config import setup_structured_logging

    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False
    setup_structured_logging = None

if _OBSERVABILITY_AVAILABLE and setup_structured_logging:
    setup_structured_logging(LOGGING_CONFIG["level"])
else:
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["log_file"]),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)

__all__ = ["run_catia_analysis"]

if __name__ == "__main__":
    try:
        results = run_catia_analysis(
            region="US_Gulf_Coast",
            use_mock_data=True,
            perils=["hurricane", "flood", "wildfire", "earthquake"],
        )

        print("\n" + "=" * 80)
        print("KEY METRICS SUMMARY (Multi-Peril)")
        print("=" * 80)
        print(f"Perils Analyzed: {', '.join(results['metadata']['perils_analyzed'])}")
        print(
            f"\nAggregate Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}"
        )
        print(
            f"Median Annual Loss: ${results['risk_metrics']['descriptive_stats']['median']:,.0f}"
        )
        print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
        print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")

        print("\nLoss by Peril:")
        for contrib in results.get("multi_peril_contributions", []):
            print(
                f"  {contrib['peril_name']}: ${contrib['mean_loss']:,.0f} "
                f"({contrib['contribution_pct']:.1f}%)"
            )

        print(
            f"\n100-Year Loss: ${results['risk_metrics']['return_periods']['100_year']:,.0f}"
        )
        print(
            f"500-Year Loss: ${results['risk_metrics']['return_periods']['500_year']:,.0f}"
        )
        print(
            f"\nMitigation Potential: {results['mitigation_summary']['total_risk_reduction']:.2%}"
        )
        print(
            f"Mitigated Loss: ${results['mitigation_summary']['mitigated_loss']:,.0f}"
        )
        print("=" * 80)

        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
