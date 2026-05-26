from __future__ import annotations

from core.harness.evolve import (
    EvolutionRecord,
    EvolutionResult,
    TargetSuggestionSet,
    ValidationResult,
)


def test_validation_result_passed() -> None:
    result = ValidationResult(passed=True)

    assert result.passed is True
    assert result.errors == []


def test_validation_result_failed() -> None:
    result = ValidationResult(passed=False, errors=["e1"])

    assert result.passed is False
    assert result.errors == ["e1"]


def test_validation_result_default_errors() -> None:
    result = ValidationResult(passed=True)

    assert result.errors == []


def test_target_suggestion_set_construction() -> None:
    suggestion_set = TargetSuggestionSet(target="t", kind="k")

    assert suggestion_set.target == "t"
    assert suggestion_set.kind == "k"


def test_target_suggestion_set_default_lists() -> None:
    suggestion_set = TargetSuggestionSet(target="t", kind="k")

    assert suggestion_set.failure_patterns == []
    assert suggestion_set.success_anchors == []
    assert suggestion_set.suggestions == []


def test_evolution_record_accepted() -> None:
    record = EvolutionRecord(
        target_file="f",
        original_content="o",
        evolved_content="n",
        reason="r",
        status="accepted",
    )

    assert record.status == "accepted"


def test_evolution_record_rejected() -> None:
    record = EvolutionRecord(
        target_file="f",
        original_content="o",
        evolved_content="n",
        reason="r",
        status="rejected",
    )

    assert record.status == "rejected"


def test_evolution_result_successful() -> None:
    result = EvolutionResult(skills_version="v3", prompts_version=None)

    assert result.skills_version == "v3"
    assert result.prompts_version is None


def test_evolution_result_all_rejected() -> None:
    result = EvolutionResult(skills_version=None, prompts_version=None)

    assert result.skills_version is None
    assert result.prompts_version is None
