"""SkillRegistry 与 frontmatter 解析单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.search.skills import (
    SkillRegistry,
    discover_skills,
    parse_frontmatter,
    strip_frontmatter,
)

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


_ALWAYS_SKILL = """---
name: general-strategy
description: 通用搜索策略
always: true
---

通用策略正文内容。
"""

_TASK_SKILL = """---
name: counting-problem
description: 计数类问题搜索策略
task_type: Counting Problem
---

计数题专用策略。
"""

_TASK_SKILL_2 = """---
name: ocr-problems
description: OCR 文字识别策略
task_type: OCR Problems
---

OCR 专用策略。
"""


class TestDiscoverSkills:
    """测试技能目录扫描与分类。"""

    def test_empty_dir(self, tmp_path: Path) -> None:
        always_text, task_map, catalog, _registry = discover_skills(tmp_path)
        assert always_text == ""
        assert task_map == {}
        assert catalog == ""

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        always_text, task_map, catalog, _registry = discover_skills(
            tmp_path / "nonexistent"
        )
        assert always_text == ""
        assert task_map == {}

    def test_mixed_skills(self, tmp_path: Path) -> None:
        (tmp_path / "always.md").write_text(_ALWAYS_SKILL)
        (tmp_path / "counting.md").write_text(_TASK_SKILL)
        (tmp_path / "ocr.md").write_text(_TASK_SKILL_2)

        always_text, task_map, catalog, registry = discover_skills(tmp_path)

        assert "通用策略正文内容" in always_text
        assert "Counting Problem" in task_map
        assert "计数题专用策略" in task_map["Counting Problem"]
        assert "OCR Problems" in task_map
        assert "counting-problem" in catalog
        assert "ocr-problems" in catalog
        assert registry.read("counting-problem").strip() == "计数题专用策略。"

    def test_always_skill_not_in_catalog(self, tmp_path: Path) -> None:
        """always 技能不应出现在 catalog 和 registry 中。"""
        (tmp_path / "always.md").write_text(_ALWAYS_SKILL)
        _, _, catalog, registry = discover_skills(tmp_path)
        assert "general-strategy" not in catalog
        with pytest.raises(KeyError):
            registry.read("general-strategy")
