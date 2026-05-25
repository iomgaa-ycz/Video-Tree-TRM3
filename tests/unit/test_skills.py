"""SkillRegistry 与 frontmatter 解析单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.search.skills import SkillRegistry, parse_frontmatter, strip_frontmatter

_SKILL_WITH_FM = """---
name: counting-problem
description: 计数类问题
task_type: Counting Problem
always: false
---

## 适用场景

问题要求统计某类事物的数量。
"""

_SKILL_WITHOUT_FM = """## 没有 frontmatter 的内容

直接正文。
"""

_SKILL_WITH_INCOMPLETE_FM = """---
name: broken-skill
description: 不完整的 frontmatter

## 正文
"""


class TestStripFrontmatter:
    """测试 frontmatter 去除行为。"""

    def test_with_frontmatter(self) -> None:
        assert (
            strip_frontmatter(_SKILL_WITH_FM)
            == "\n## 适用场景\n\n问题要求统计某类事物的数量。\n"
        )

    def test_without_frontmatter(self) -> None:
        assert strip_frontmatter(_SKILL_WITHOUT_FM) == _SKILL_WITHOUT_FM

    def test_incomplete_frontmatter(self) -> None:
        assert strip_frontmatter(_SKILL_WITH_INCOMPLETE_FM) == _SKILL_WITH_INCOMPLETE_FM


class TestParseFrontmatter:
    """测试 frontmatter 字段提取。"""

    def test_extracts_full_fields(self) -> None:
        assert parse_frontmatter(_SKILL_WITH_FM) == {
            "name": "counting-problem",
            "description": "计数类问题",
            "task_type": "Counting Problem",
            "always": "false",
        }

    def test_missing_frontmatter_returns_empty_dict(self) -> None:
        assert parse_frontmatter(_SKILL_WITHOUT_FM) == {}


class TestSkillRegistry:
    """测试技能注册表读取行为。"""

    def test_read_registered_skill(self, tmp_path: Path) -> None:
        skill_path = tmp_path / "counting-problem.md"
        skill_path.write_text(_SKILL_WITH_FM, encoding="utf-8")

        registry = SkillRegistry()
        registry.set_paths({"counting-problem": skill_path})

        assert (
            registry.read("counting-problem")
            == "\n## 适用场景\n\n问题要求统计某类事物的数量。\n"
        )

    def test_read_unregistered_skill_raises(self) -> None:
        registry = SkillRegistry()
        registry.set_paths({})

        with pytest.raises(KeyError):
            registry.read("missing-skill")

    def test_read_from_empty_registry_raises(self) -> None:
        registry = SkillRegistry()

        with pytest.raises(KeyError):
            registry.read("missing-skill")
