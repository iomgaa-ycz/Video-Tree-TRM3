from __future__ import annotations

from core.harness.evolve import (
    EvolutionRecord,
    EvolutionResult,
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


def test_evolution_record_new_fields() -> None:
    """EvolutionRecord 应包含 target_type, source_version, result_version 字段。"""
    record = EvolutionRecord(
        target_file="temporal-reasoning.md",
        target_type="skill",
        original_content="old",
        evolved_content="new",
        reason="改进搜索步骤",
        status="accepted",
        suggestions=[
            {"section": "Step 1", "problem": "p", "change": "c", "related_cases": []}
        ],
        source_version="v1",
        result_version="v2",
    )
    assert record.target_type == "skill"
    assert record.source_version == "v1"
    assert record.result_version == "v2"
    assert isinstance(record.suggestions[0], dict)


def test_evolution_record_rejected_no_result_version() -> None:
    """rejected 的记录 result_version 应为 None。"""
    record = EvolutionRecord(
        target_file="system.md",
        target_type="system",
        original_content="old",
        evolved_content="old",
        reason="验证失败",
        status="rejected",
        source_version="v1",
        result_version=None,
        validation_errors=["frontmatter 被改"],
    )
    assert record.status == "rejected"
    assert record.result_version is None
    assert len(record.validation_errors) == 1


def test_evolution_record_default_fields() -> None:
    """默认字段应为空列表或 None。"""
    record = EvolutionRecord(
        target_file="f",
        target_type="skill",
        original_content="o",
        evolved_content="n",
        reason="r",
        status="accepted",
        source_version="v1",
    )
    assert record.result_version is None
    assert record.suggestions == []
    assert record.attempts == []
    assert record.validation_errors == []


def test_evolution_result_counts() -> None:
    """EvolutionResult 应正确统计 accepted/rejected/skipped 数量。"""
    result = EvolutionResult(
        skills_version="v2",
        prompts_version=None,
        records=[],
        accepted_count=3,
        rejected_count=1,
        skipped_count=5,
    )
    assert result.accepted_count == 3
    assert result.skipped_count == 5
    assert result.prompts_version is None


def test_evolution_result_empty() -> None:
    result = EvolutionResult(skills_version=None, prompts_version=None)
    assert result.records == []
    assert result.accepted_count == 0
