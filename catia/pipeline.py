"""
End-to-end CATIA analysis pipeline.

Stable import path for ``run_catia_analysis`` (also re-exported from ``main`` for
repo-root scripts). Prefer::

    from catia.pipeline import run_catia_analysis, run_from_spec
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from catia import __version__ as _CATIA_VERSION
from catia.audit import create_audit_metadata, generate_run_id
from catia.config import DEFAULT_PERILS, OUTPUT_CONFIG, SIMULATION_CONFIG
from catia.data_acquisition import fetch_all_data
from catia.export import ReportExporter
from catia.financial_impact import (
    FinancialImpactSimulator,
    run_multi_peril_analysis,
)
from catia.mitigation import generate_mitigation_recommendations
from catia.risk_alerts import RiskAlertSystem
from catia.risk_prediction import train_risk_model
from catia.run_spec import RunSpec
from catia.scenario_analysis import ScenarioAnalyzer
from catia.sensitivity_analysis import QuickSensitivityAnalysis
from catia.transparency import build_pipeline_manifest, log_pipeline_manifest
from catia.visualization import create_dashboard

try:
    from catia.metrics import record_analysis_run

    _metrics = record_analysis_run
except ImportError:
    _metrics = None

logger = logging.getLogger(__name__)


def _artifact_wanted(artifacts: Optional[List[str]], name: str) -> bool:
    if artifacts is None:
        return True
    return name in artifacts


def run_catia_analysis(
    region: str = "US_Gulf_Coast",
    use_mock_data: bool = True,
    perils: Optional[list] = None,
    *,
    scenario_id: Optional[str] = None,
    monte_carlo_iterations: Optional[int] = None,
    random_seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    artifacts: Optional[List[str]] = None,
    explain: bool = False,
) -> Dict[str, Any]:
    """
    Run complete CATIA analysis workflow.

    Args:
        region: Geographic region for analysis
        use_mock_data: Use mock data if True
        perils: List of peril types (uses DEFAULT_PERILS if None)
        scenario_id: Optional climate scenario id (see ``CLIMATE_SCENARIOS`` in config)
        monte_carlo_iterations: Override global simulation iteration count for this run
        random_seed: Override ``SIMULATION_CONFIG['random_seed']`` for this run
        output_dir: Directory for written artifacts (defaults to OUTPUT_CONFIG)
        artifacts: Optional subset of outputs to write; known keys:
            ``report``, ``assumption_register``, ``compliance``, ``dashboard``, ``enhancements``.
            ``None`` writes everything (default).
        explain: If True, log a pre-run transparency manifest (steps, data, parameters).

    Returns:
        Dictionary with all analysis results
    """
    perils = perils or list(DEFAULT_PERILS)
    out_dir = output_dir or OUTPUT_CONFIG.get("output_dir", "outputs")

    _restore_sim: Dict[str, Any] = {}
    if monte_carlo_iterations is not None:
        _restore_sim["monte_carlo_iterations"] = SIMULATION_CONFIG["monte_carlo_iterations"]
        SIMULATION_CONFIG["monte_carlo_iterations"] = int(monte_carlo_iterations)
    if random_seed is not None:
        _restore_sim["random_seed"] = SIMULATION_CONFIG["random_seed"]
        SIMULATION_CONFIG["random_seed"] = int(random_seed)

    try:
        return _run_catia_analysis_body(
            region=region,
            use_mock_data=use_mock_data,
            perils=perils,
            scenario_id=scenario_id,
            out_dir=out_dir,
            artifacts=artifacts,
            explain=explain,
        )
    finally:
        for key, val in _restore_sim.items():
            SIMULATION_CONFIG[key] = val


def _run_catia_analysis_body(
    *,
    region: str,
    use_mock_data: bool,
    perils: List[str],
    scenario_id: Optional[str],
    out_dir: str,
    artifacts: Optional[List[str]],
    explain: bool,
) -> Dict[str, Any]:
    mc_iter_applied = int(SIMULATION_CONFIG["monte_carlo_iterations"])
    run_id = generate_run_id()

    manifest = build_pipeline_manifest(
        region=region,
        use_mock_data=use_mock_data,
        perils=perils,
        scenario_id=scenario_id,
        output_dir=out_dir,
        artifacts=artifacts,
        monte_carlo_iterations=mc_iter_applied,
        random_seed=SIMULATION_CONFIG.get("random_seed"),
        severity_distribution=str(SIMULATION_CONFIG.get("severity_distribution", "Lognormal")),
        catia_version=_CATIA_VERSION,
    )
    effective_explain = explain or (
        os.environ.get("CATIA_EXPLAIN", "").lower() in ("1", "true", "yes")
    )
    if effective_explain:
        log_pipeline_manifest(manifest, logger)

    audit = create_audit_metadata(run_id, region, perils, use_mock_data)
    if _metrics:
        _metrics(region, perils)

    logger.info("=" * 80)
    logger.info("CATIA: Catastrophe AI System for Climate Risk Modeling")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Analysis Region: %s", region)
    logger.info("Perils: %s", ", ".join(perils))
    if scenario_id:
        logger.info("Climate scenario: %s", scenario_id)
    logger.info("Timestamp: %s", datetime.now().isoformat())
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 80)

    # STEP 1
    logger.info("\n[STEP 1] DATA ACQUISITION (Multi-Peril)")
    logger.info("-" * 80)
    try:
        data = fetch_all_data(region, use_mock=use_mock_data, perils=perils)
        logger.info("✓ Climate data: %s records", len(data["climate"]))
        logger.info("✓ Socioeconomic data: %s records", len(data["socioeconomic"]))
        logger.info("✓ Historical events: %s records", len(data["historical_events"]))
        logger.info("✓ Perils analyzed: %s", ", ".join(data["perils_analyzed"]))
        for peril, events in data.get("events_by_peril", {}).items():
            logger.info("    - %s: %s events", peril, len(events))
    except Exception as e:
        logger.error("✗ Data acquisition failed: %s", e)
        raise

    # STEP 2
    logger.info("\n[STEP 2] RISK PREDICTION MODEL")
    logger.info("-" * 80)
    try:
        predictor = train_risk_model(
            data["climate"], data["socioeconomic"], data["historical_events"]
        )
        logger.info("✓ Risk prediction model trained and saved")

        if os.environ.get("CATIA_USE_SHAP", "").lower() in ("1", "true", "yes"):
            try:
                from catia.explainability import SHAP_AVAILABLE, RiskExplainer

                if SHAP_AVAILABLE:
                    X, _, _ = predictor.prepare_features(
                        data["climate"],
                        data["socioeconomic"],
                        data["historical_events"],
                    )
                    X_scaled = predictor.scaler.transform(X)
                    X_arr = np.asarray(X_scaled)
                    explainer = RiskExplainer(
                        predictor.probability_model,
                        feature_names=predictor.feature_names,
                        background_samples=min(100, max(10, len(X_arr) // 5)),
                    )
                    explainer.fit(X_arr)
                    importance = explainer.get_global_importance(X_arr)
                    os.makedirs(out_dir, exist_ok=True)
                    fi_path = os.path.join(out_dir, "feature_importance.json")
                    with open(fi_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "feature_names": importance.feature_names,
                                "importance_scores": importance.importance_scores.tolist(),
                                "ranking": [(n, float(v)) for n, v in importance.ranking],
                            },
                            f,
                            indent=2,
                        )
                    logger.info("✓ Feature importance written: %s", fi_path)
                else:
                    logger.debug("SHAP not available; feature importance skipped")
            except Exception as e:
                logger.debug("SHAP step skipped: %s", e)

        if os.environ.get("CATIA_USE_ENSEMBLE", "").lower() in ("1", "true", "yes"):
            try:
                from sklearn.preprocessing import StandardScaler

                from catia.ensemble import (
                    RobustVotingClassifier,
                    RobustVotingRegressor,
                    get_base_classifiers,
                    get_base_regressors,
                )

                X, y_prob, y_sev = predictor.prepare_features(
                    data["climate"],
                    data["socioeconomic"],
                    data["historical_events"],
                )
                X_scaled = StandardScaler().fit_transform(X)
                clf = RobustVotingClassifier(
                    estimators=get_base_classifiers(), voting="soft", auto_weight=True
                )
                reg = RobustVotingRegressor(
                    estimators=get_base_regressors(), auto_weight=True
                )
                clf.fit(X_scaled, y_prob)
                reg.fit(X_scaled, y_sev)
                logger.info("✓ Ensemble (voting) risk model trained")
            except Exception as e:
                logger.debug("Ensemble step skipped: %s", e)
    except Exception as e:
        logger.error("✗ Risk prediction failed: %s", e)
        raise

    # STEP 3
    logger.info("\n[STEP 3] FINANCIAL IMPACT SIMULATION (Multi-Peril)")
    logger.info("-" * 80)
    try:
        multi_peril_results = run_multi_peril_analysis(
            perils,
            include_uncertainty=True,
            n_bootstrap=200,
            scenario_id=scenario_id,
            num_iterations=None,
        )

        logger.info(
            "✓ Monte Carlo simulations: %s iterations", mc_iter_applied
        )
        logger.info("✓ Perils simulated: %s", ", ".join(perils))

        for contrib in multi_peril_results["contributions"]:
            logger.info(
                "    - %s: Mean=$%s (%.1f%%)",
                contrib["peril_name"],
                f"{contrib['mean_loss']:,.0f}",
                contrib["contribution_pct"],
            )

        agg_metrics = multi_peril_results["aggregate_metrics"]
        logger.info(
            "✓ Aggregate Mean Annual Loss: $%s",
            f"{agg_metrics['descriptive_stats']['mean']:,.0f}",
        )
        logger.info(
            "✓ Aggregate VaR (95%%): $%s",
            f"{agg_metrics['risk_metrics']['var']:,.0f}",
        )
        logger.info(
            "✓ Aggregate TVaR (95%%): $%s",
            f"{agg_metrics['risk_metrics']['tvar']:,.0f}",
        )

        aggregate_losses = multi_peril_results["results"]["aggregate"]["losses"]
        loss_levels = np.percentile(aggregate_losses, np.linspace(0, 99.9, 100))
        exceedance_probs = 1 - np.linspace(0, 0.999, 100)

        financial_results = {
            "metrics": agg_metrics,
            "simulation_results": {"all_losses": aggregate_losses},
            "loss_exceedance_curve": {
                "loss_levels": loss_levels,
                "exceedance_probabilities": exceedance_probs,
            },
            "multi_peril": multi_peril_results,
        }
    except Exception as e:
        logger.error("✗ Financial impact simulation failed: %s", e)
        raise

    # STEP 4
    logger.info("\n[STEP 4] MITIGATION RECOMMENDATIONS")
    logger.info("-" * 80)
    try:
        baseline_loss = financial_results["metrics"]["descriptive_stats"]["mean"]
        mitigation_results = generate_mitigation_recommendations(baseline_loss)

        logger.info(
            "✓ Baseline loss: $%s",
            f"{mitigation_results['summary']['baseline_loss']:,.0f}",
        )
        logger.info(
            "✓ Mitigated loss: $%s",
            f"{mitigation_results['summary']['mitigated_loss']:,.0f}",
        )
        logger.info(
            "✓ Risk reduction: %s",
            f"{mitigation_results['summary']['total_risk_reduction']:.2%}",
        )
        logger.info(
            "✓ Priority strategies: %s",
            ", ".join(mitigation_results["priority_order"][:3]),
        )
    except Exception as e:
        logger.error("✗ Mitigation recommendations failed: %s", e)
        raise

    # STEP 5
    logger.info("\n[STEP 5] VISUALIZATION & REPORTING")
    logger.info("-" * 80)
    try:
        import pandas as pd

        cba_df = pd.DataFrame(mitigation_results["strategies"])
        if _artifact_wanted(artifacts, "dashboard"):
            dashboard_dir = create_dashboard(
                financial_results, data["climate"], cba_df, output_dir=out_dir
            )
            logger.info("✓ Dashboard created: %s", dashboard_dir)
            logger.info("  - loss_exceedance_curve.html")
            logger.info("  - risk_distribution.html")
            logger.info("  - return_period_curve.html")
            logger.info("  - mitigation_comparison.html")
        else:
            logger.info("✓ Dashboard skipped (artifacts filter)")
    except Exception as e:
        logger.error("✗ Visualization failed: %s", e)
        raise

    # COMPILE
    logger.info("\n[STEP 6] RESULTS COMPILATION")
    logger.info("-" * 80)

    results: Dict[str, Any] = {
        "metadata": {
            "run_id": run_id,
            "region": region,
            "timestamp": datetime.now().isoformat(),
            "use_mock_data": use_mock_data,
            "perils_analyzed": perils,
            "scenario_id": scenario_id,
            "monte_carlo_iterations": mc_iter_applied,
            "random_seed": SIMULATION_CONFIG.get("random_seed"),
            "output_dir": out_dir,
            "artifacts": artifacts,
            "transparency": manifest,
        },
        "audit": audit,
        "data_summary": {
            "climate_records": len(data["climate"]),
            "socioeconomic_records": len(data["socioeconomic"]),
            "historical_events": len(data["historical_events"]),
            "events_by_peril": {
                p: len(e) for p, e in data.get("events_by_peril", {}).items()
            },
        },
        "risk_metrics": financial_results["metrics"],
        "multi_peril_contributions": financial_results.get("multi_peril", {}).get(
            "contributions", []
        ),
        "mitigation_summary": mitigation_results["summary"],
        "mitigation_strategies": mitigation_results["strategies"],
        "priority_order": mitigation_results["priority_order"],
    }

    os.makedirs(out_dir, exist_ok=True)
    if _artifact_wanted(artifacts, "report"):
        output_file = os.path.join(out_dir, "catia_report.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("✓ Report saved: %s", output_file)

    if _artifact_wanted(artifacts, "assumption_register"):
        try:
            from catia.assumption_register import write_assumption_register

            reg_path = os.path.join(out_dir, "assumption_register.json")
            write_assumption_register(reg_path)
            logger.info("✓ Assumption register saved: %s", reg_path)
        except Exception as e:
            logger.debug("Assumption register skipped: %s", e)

    if _artifact_wanted(artifacts, "compliance"):
        try:
            from catia.compliance import generate_compliance_report

            compliance_path = os.path.join(out_dir, "compliance_report.html")
            generate_compliance_report(
                results["audit"], results, output_path=compliance_path
            )
            logger.info("✓ Compliance report: %s", compliance_path)
        except Exception as e:
            logger.debug("Compliance report skipped: %s", e)

    if _artifact_wanted(artifacts, "enhancements"):
        logger.info("\n[PHASE 1] QUICK WINS ENHANCEMENTS")
        logger.info("-" * 80)
        try:
            event_frequency = len(data["historical_events"]) / max(
                data["historical_events"]["year"].max()
                - data["historical_events"]["year"].min(),
                1,
            )
            severity_params = {"mu": 15, "sigma": 2}
            simulator = FinancialImpactSimulator(event_frequency, severity_params)

            logger.info("\n[ENHANCEMENT 1] Sensitivity Analysis")
            try:
                analyzer = QuickSensitivityAnalysis(simulator)
                sensitivity_results = analyzer.analyze(
                    {
                        "event_frequency": [0.3, 0.4, 0.5, 0.6, 0.7],
                        "severity_mu": [14, 15, 16, 17, 18],
                    }
                )
                analyzer.plot_tornado(sensitivity_results).write_html(
                    os.path.join(out_dir, "sensitivity_tornado.html")
                )
                analyzer.plot_sensitivity_heatmap(sensitivity_results).write_html(
                    os.path.join(out_dir, "sensitivity_heatmap.html")
                )
                logger.info("✓ Sensitivity analysis complete")
                logger.info(analyzer.generate_summary(sensitivity_results))
            except Exception as e:
                logger.error("✗ Sensitivity analysis failed: %s", e)

            logger.info("\n[ENHANCEMENT 2] Scenario Analysis")
            try:
                scenario_analyzer = ScenarioAnalyzer(simulator)
                scenario_results = scenario_analyzer.run_scenarios()
                scenario_analyzer.plot_scenarios(scenario_results).write_html(
                    os.path.join(out_dir, "scenarios.html")
                )
                scenario_analyzer.plot_return_periods(scenario_results).write_html(
                    os.path.join(out_dir, "return_periods.html")
                )
                logger.info("✓ Scenario analysis complete")
                logger.info(scenario_analyzer.generate_summary(scenario_results))
            except Exception as e:
                logger.error("✗ Scenario analysis failed: %s", e)

            logger.info("\n[ENHANCEMENT 3] Risk Alerts")
            try:
                alert_system = RiskAlertSystem(
                    {
                        "var_max": 100,
                        "mean_loss_max": 50,
                        "loss_ratio_max": 1.1,
                        "tvar_max": 150,
                    }
                )
                alert_metrics = {
                    "var_95": results["risk_metrics"]["risk_metrics"]["var"],
                    "tvar_95": results["risk_metrics"]["risk_metrics"]["tvar"],
                    "mean_loss": results["risk_metrics"]["descriptive_stats"]["mean"],
                    "loss_ratio": 1.0,
                }
                alert_system.check_alerts(alert_metrics)
                logger.info(alert_system.format_alerts())
            except Exception as e:
                logger.error("✗ Risk alerts failed: %s", e)

            logger.info("\n[ENHANCEMENT 4] Export Results")
            try:
                exporter = ReportExporter(results["risk_metrics"], output_dir=out_dir)
                export_paths = exporter.export_all()
                logger.info("✓ JSON export: %s", export_paths["json"])
                logger.info("✓ CSV export: %s", export_paths["csv"])
                logger.info("✓ HTML export: %s", export_paths["html"])
            except Exception as e:
                logger.error("✗ Export failed: %s", e)

            logger.info("\n✓ Phase 1 Enhancements Complete")
        except Exception as e:
            logger.error("✗ Phase 1 enhancements failed: %s", e)
    else:
        logger.info("✓ Phase 1 enhancements skipped (artifacts filter)")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("Region: %s", region)
    logger.info("Perils Analyzed: %s", ", ".join(perils))
    logger.info(
        "Mean Annual Loss: $%s",
        f"{results['risk_metrics']['descriptive_stats']['mean']:,.0f}",
    )
    logger.info(
        "VaR (95%%): $%s",
        f"{results['risk_metrics']['risk_metrics']['var']:,.0f}",
    )
    logger.info(
        "Risk Reduction Potential: %s",
        f"{results['mitigation_summary']['total_risk_reduction']:.2%}",
    )
    logger.info("Output Directory: %s", out_dir)
    logger.info("=" * 80)

    return results


def run_from_spec(spec: RunSpec) -> Dict[str, Any]:
    """Execute ``run_catia_analysis`` using a validated :class:`RunSpec`."""
    return run_catia_analysis(**spec.to_kwargs())
