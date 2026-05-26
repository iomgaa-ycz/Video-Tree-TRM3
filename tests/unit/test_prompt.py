"""PromptManager 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.search.prompt import PromptManager


_FAKE_SYSTEM = """\
## 角色

你是一个测试用 Agent。

## 搜索循环

content 输出纯 JSON。
"""


@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    """创建包含 system.md 和一个额外 prompt 文件的临时 prompts 目录。"""
    (tmp_path / "system.md").write_text(_FAKE_SYSTEM, encoding="utf-8")
    (tmp_path / "diagnose_span.md").write_text("span prompt 内容", encoding="utf-8")
    return tmp_path


class TestPromptManagerInit:
    """构造函数行为。"""

    def test_caches_system_base(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        assert pm._system_base == _FAKE_SYSTEM

    def test_missing_system_md_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromptManager(tmp_path)


class TestBuildInferencePrompt:
    """inference system prompt 组装。"""

    def test_none_mode_base_and_tools_only(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="none",
            task_type="Counting Problem",
            always_skills_text="",
            task_skill_map={},
            catalog_text="",
        )
        assert "测试用 Agent" in result
        assert "view_node" in result
        assert "通用搜索策略" not in result
        assert "当前题型搜索策略" not in result

    def test_auto_mode_injects_task_skill(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="auto",
            task_type="Counting Problem",
            always_skills_text="",
            task_skill_map={"Counting Problem": "逐项计数策略正文"},
            catalog_text="",
        )
        assert "当前题型搜索策略" in result
        assert "逐项计数策略正文" in result

    def test_auto_mode_no_matching_skill(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="auto",
            task_type="Unknown Type",
            always_skills_text="",
            task_skill_map={"Counting Problem": "策略"},
            catalog_text="",
        )
        assert "当前题型搜索策略" not in result

    def test_manual_mode_injects_catalog(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="manual",
            task_type="Counting Problem",
            always_skills_text="",
            task_skill_map={},
            catalog_text="- counting-problem: 计数类问题",
        )
        assert "可用搜索策略" in result
        assert "read_skill" in result
        assert "counting-problem" in result

    def test_always_skills_prepended(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="none",
            task_type="Counting Problem",
            always_skills_text="通用策略正文",
            task_skill_map={},
            catalog_text="",
        )
        assert "通用搜索策略" in result
        assert "通用策略正文" in result

    def test_manual_mode_includes_read_skill_in_tools(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="manual",
            task_type="any",
            always_skills_text="",
            task_skill_map={},
            catalog_text="目录",
        )
        assert "read_skill" in result

    def test_non_manual_mode_excludes_read_skill(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.build_inference_prompt(
            skill_mode="auto",
            task_type="any",
            always_skills_text="",
            task_skill_map={},
            catalog_text="",
        )
        tool_section_start = result.find("## 可用工具")
        tool_section = result[tool_section_start:]
        assert "read_skill" not in tool_section


class TestFormatUserPrompt:
    """user prompt 格式化。"""

    def test_basic_format(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        qa = {
            "question": "视频中出现了什么动物？",
            "options": ["A. 猫", "B. 狗", "C. 鸟", "D. 鱼"],
        }
        result = pm.format_user_prompt(qa, ["L1_000", "L1_001"])
        assert "视频中出现了什么动物？" in result
        assert "A. 猫" in result
        assert "L1_000, L1_001" in result

    def test_with_task_type(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        qa = {"question": "问题", "options": ["A. 选项"]}
        result = pm.format_user_prompt(qa, ["L1_000"], task_type="Counting Problem")
        assert "**题型**: Counting Problem" in result

    def test_without_task_type(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        qa = {"question": "问题", "options": ["A. 选项"]}
        result = pm.format_user_prompt(qa, ["L1_000"])
        assert "题型" not in result


class TestLoad:
    """通用 prompt 文件加载。"""

    def test_load_existing_file(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        result = pm.load("diagnose_span.md")
        assert result == "span prompt 内容"

    def test_load_missing_file_raises(self, prompts_dir: Path) -> None:
        pm = PromptManager(prompts_dir)
        with pytest.raises(FileNotFoundError):
            pm.load("nonexistent.md")
