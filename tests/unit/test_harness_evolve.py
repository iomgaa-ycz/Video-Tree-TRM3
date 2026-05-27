from __future__ import annotations

from core.harness.evolve import (
    EvolutionRecord,
    EvolutionResult,
    ValidationResult,
    validate_skill,
    validate_system,
    validate_tool,
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


# ---------------------------------------------------------------------------
# validate_skill
# ---------------------------------------------------------------------------

_SKILL_FM = "---\nname: temporal-reasoning\ndescription: 时间推理类问题\ntask_type: Temporal Reasoning\n---\n"


def test_validate_skill_pass() -> None:
    """合法的 skill 改写应通过所有检查。"""
    original = _SKILL_FM + "## 适用场景\n旧内容\n"
    evolved = _SKILL_FM + "## 适用场景\n新内容，更好的描述\n"
    result = validate_skill(original, evolved)
    assert result.passed
    assert result.errors == []


def test_validate_skill_frontmatter_changed() -> None:
    """frontmatter 中 task_type 被改时应失败。"""
    original = _SKILL_FM + "内容"
    evolved = original.replace("Temporal Reasoning", "Time Reasoning")
    result = validate_skill(original, evolved)
    assert not result.passed
    assert any("task_type" in e for e in result.errors)


def test_validate_skill_too_long() -> None:
    """改写后长度超过原文 2 倍应失败。"""
    original = _SKILL_FM + "short"
    evolved = _SKILL_FM + "x" * 10000
    result = validate_skill(original, evolved)
    assert not result.passed
    assert any("长度" in e for e in result.errors)


def test_validate_skill_too_short() -> None:
    """改写后长度低于原文 0.3 倍应失败。"""
    original = _SKILL_FM + "content " * 100
    evolved = _SKILL_FM + "hi"
    result = validate_skill(original, evolved)
    assert not result.passed
    assert any("长度" in e for e in result.errors)


def test_validate_skill_no_frontmatter() -> None:
    """改写后缺少 frontmatter 应失败。"""
    original = _SKILL_FM + "content"
    evolved = "没有 frontmatter 的内容"
    result = validate_skill(original, evolved)
    assert not result.passed


def test_validate_skill_unclosed_code_block() -> None:
    """未闭合的代码块应失败。"""
    original = _SKILL_FM + "content"
    evolved = _SKILL_FM + "```json\n{}\n"
    result = validate_skill(original, evolved)
    assert not result.passed
    assert any("代码块" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_system
# ---------------------------------------------------------------------------


def test_validate_system_frozen_section_removed() -> None:
    """system.md 的冻结区被移除时应失败。"""
    original = "## 角色\n描述\n\n## 能力边界\n事实\n\n## 决策原则\n策略"
    evolved = "## 角色\n新描述\n\n## 决策原则\n新策略"
    result = validate_system(original, evolved)
    assert not result.passed
    assert any("能力边界" in e for e in result.errors)


def test_validate_system_frozen_section_modified() -> None:
    """system.md 的冻结区被修改时应失败。"""
    original = "## 角色\n描述\n\n## 能力边界\n事实A\n\n## 决策原则\n策略"
    evolved = "## 角色\n新描述\n\n## 能力边界\n事实B\n\n## 决策原则\n新策略"
    result = validate_system(original, evolved)
    assert not result.passed
    assert any("能力边界" in e and "修改" in e for e in result.errors)


def test_validate_system_pass() -> None:
    """只修改非冻结区应通过。"""
    original = "## 角色\n描述\n\n## 能力边界\n事实\n\n## 决策原则\n策略"
    evolved = "## 角色\n新描述\n\n## 能力边界\n事实\n\n## 决策原则\n新策略"
    result = validate_system(original, evolved)
    assert result.passed


# ---------------------------------------------------------------------------
# validate_tool
# ---------------------------------------------------------------------------


def test_validate_tool_format_removed() -> None:
    """tool prompt 的输出格式被移除时应失败。"""
    orig_extract = "## 工作原则\n规则\n\n## 输出格式\n格式A"
    evol_extract = "## 工作原则\n新规则"
    result = validate_tool(orig_extract, evol_extract, "v", "v")
    assert not result.passed
    assert any("输出格式" in e for e in result.errors)


def test_validate_tool_pass() -> None:
    """只修改工作原则应通过。"""
    orig_extract = "## 工作原则\n规则\n\n## 输出格式\n格式"
    evol_extract = "## 工作原则\n新规则\n\n## 输出格式\n格式"
    result = validate_tool(orig_extract, evol_extract, "v", "v")
    assert result.passed
