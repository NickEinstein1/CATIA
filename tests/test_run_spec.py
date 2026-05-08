"""RunSpec loading and CLI merge validation."""

import pytest
from pydantic import ValidationError

from catia.run_spec import RunSpec, merge_cli_run_spec


def test_merge_cli_run_spec_revalidates_scenario():
    """CLI overrides must pass RunSpec validators (scenario_id known to registry)."""
    with pytest.raises(ValidationError):
        merge_cli_run_spec(scenario_id="totally_invalid_xyz")


def test_merge_cli_run_spec_valid_scenario():
    s = merge_cli_run_spec(scenario_id="baseline")
    assert s.scenario_id == "baseline"


def test_merge_cli_run_spec_empty_string_scenario_becomes_none():
    # Validator maps "" to None
    s = merge_cli_run_spec(scenario_id="")
    assert s.scenario_id is None


def test_runspec_iterations_must_be_positive():
    with pytest.raises(ValidationError):
        RunSpec(monte_carlo_iterations=0)
