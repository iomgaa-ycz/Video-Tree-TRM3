"""tools 单元测试 — parse_action、dispatch、get_tool_descriptions。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.tree.tools import dispatch, get_tool_descriptions, parse_action


class TestParseAction:
    """解析模型输出的 JSON content。"""

    def test_valid_json(self) -> None:
        content = json.dumps(
            {
                "reflect": {"learned": "something"},
                "plan": {"goal": "查看节点"},
                "action": {
                    "tool": "view_node",
                    "args": {"node_id": "L1_000", "question": "问题"},
                },
            }
        )
        tool, args = parse_action(content)
        assert tool == "view_node"
        assert args["node_id"] == "L1_000"

    def test_submit_answer(self) -> None:
        content = json.dumps(
            {
                "reflect": {},
                "plan": {},
                "action": {
                    "tool": "submit_answer",
                    "args": {"answer": "B", "evidence": "证据", "reasoning": "推理"},
                },
            }
        )
        tool, args = parse_action(content)
        assert tool == "submit_answer"
        assert args["answer"] == "B"

    def test_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            parse_action("这不是JSON")

    def test_missing_action(self) -> None:
        with pytest.raises(ValueError, match="action"):
            parse_action(json.dumps({"reflect": {}, "plan": {}}))

    def test_missing_tool_in_action(self) -> None:
        content = json.dumps(
            {
                "reflect": {},
                "plan": {},
                "action": {"args": {"x": 1}},
            }
        )
        with pytest.raises(ValueError, match="tool"):
            parse_action(content)

    def test_missing_args_in_action(self) -> None:
        content = json.dumps(
            {
                "reflect": {},
                "plan": {},
                "action": {"tool": "view_node"},
            }
        )
        with pytest.raises(ValueError, match="args"):
            parse_action(content)


class TestDispatch:
    """按工具名分发执行。"""

    def test_submit_answer(self) -> None:
        result = dispatch(
            "submit_answer",
            {"answer": "C", "evidence": "证据", "reasoning": "推理"},
            env=MagicMock(),
        )
        assert "[ok]" in result
        assert "C" in result

    def test_view_node(self) -> None:
        env = MagicMock()
        env.view_node.return_value = "节点信息"
        result = dispatch(
            "view_node", {"node_id": "L1_000", "question": "问题"}, env=env
        )
        assert result == "节点信息"
        env.view_node.assert_called_once_with("L1_000", "问题")

    def test_search_similar(self) -> None:
        env = MagicMock()
        env.search_similar.return_value = "搜索结果"
        result = dispatch(
            "search_similar",
            {"query": "公园", "question": "问题", "k": 3},
            env=env,
        )
        assert result == "搜索结果"
        env.search_similar.assert_called_once_with("公园", "问题", 3)

    def test_search_similar_default_k(self) -> None:
        env = MagicMock()
        env.search_similar.return_value = "搜索结果"
        dispatch("search_similar", {"query": "公园", "question": "问题"}, env=env)
        env.search_similar.assert_called_once_with("公园", "问题", 5)

    def test_observe_frame(self) -> None:
        env = MagicMock()
        env.resolve_frame_paths.return_value = [Path("/fake/frame.jpg")]
        env.get_subtitle.return_value = ""

        with patch(
            "core.tree.vision.observe_frame",
            return_value="[视觉观察] 结果",
        ):
            result = dispatch(
                "observe_frame",
                {"node_ids": ["L3_000"], "question": "问题"},
                env=env,
                vl_client=MagicMock(),
                prompts_dir=Path("/fake/prompts"),
            )
        assert "结果" in result

    def test_observe_frame_with_subtitle(self) -> None:
        env = MagicMock()
        env.resolve_frame_paths.return_value = [Path("/fake/frame.jpg")]
        env.get_subtitle.return_value = "这是字幕"

        with patch(
            "core.tree.vision.observe_frame",
            return_value="[视觉观察] 结果",
        ):
            result = dispatch(
                "observe_frame",
                {"node_ids": ["L3_000"], "question": "问题"},
                env=env,
                vl_client=MagicMock(),
                prompts_dir=Path("/fake/prompts"),
            )
        assert "字幕上下文" in result
        assert "这是字幕" in result

    def test_observe_frame_empty_question(self) -> None:
        result = dispatch(
            "observe_frame",
            {"node_ids": ["L3_000"], "question": "  "},
            env=MagicMock(),
        )
        assert "工具执行错误" in result

    def test_read_skill(self) -> None:
        skills = MagicMock()
        skills.read.return_value = "策略正文"
        result = dispatch(
            "read_skill",
            {"name": "counting-problem"},
            env=MagicMock(),
            skills=skills,
        )
        assert result == "策略正文"

    def test_read_skill_disabled(self) -> None:
        result = dispatch(
            "read_skill",
            {"name": "counting-problem"},
            env=MagicMock(),
            skills=None,
        )
        assert "skills 未启用" in result

    def test_unknown_tool(self) -> None:
        result = dispatch("nonexistent", {}, env=MagicMock())
        assert "未知工具" in result

    def test_tool_error_caught(self) -> None:
        env = MagicMock()
        env.view_node.side_effect = KeyError("节点不存在: X")
        result = dispatch("view_node", {"node_id": "X", "question": "q"}, env=env)
        assert "工具执行错误" in result


class TestGetToolDescriptions:
    """工具描述文本。"""

    def test_base_tools_included(self) -> None:
        desc = get_tool_descriptions(include_read_skill=False)
        assert "view_node" in desc
        assert "search_similar" in desc
        assert "submit_answer" in desc
        assert "observe_frame" in desc
        assert "read_skill" not in desc

    def test_with_read_skill(self) -> None:
        desc = get_tool_descriptions(include_read_skill=True)
        assert "read_skill" in desc
