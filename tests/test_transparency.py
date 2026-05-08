"""Transparency manifest (open-source disclosure)."""

from catia.transparency import build_pipeline_manifest


def test_manifest_contains_pipeline_and_limits():
    m = build_pipeline_manifest(
        region="US_Gulf_Coast",
        use_mock_data=True,
        perils=["hurricane"],
        scenario_id="baseline",
        output_dir="outputs",
        artifacts=None,
        monte_carlo_iterations=10_000,
        random_seed=42,
        severity_distribution="Lognormal",
        catia_version="9.9.9-test",
    )
    assert m["catia_version"] == "9.9.9-test"
    assert "pipeline_steps_plain" in m
    assert len(m["pipeline_steps_plain"]) >= 4
    assert "key_modules" in m
