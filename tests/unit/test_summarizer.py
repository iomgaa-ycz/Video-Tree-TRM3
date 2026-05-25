"""summarizer 单元测试 — 用 mock LLMClient 验证两轮摘要逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


from core.tree.summarizer import (
    summarize_children,
    summarize_node,
    summarize_nodes_batch,
)


def _make_mock_client(*responses: str) -> MagicMock:
    """创建返回固定序列响应的 mock LLMClient。"""
    client = MagicMock()
    side_effects = []
    for text in responses:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = text
        side_effects.append(resp)
    client.chat.side_effect = side_effects
    return client


class TestSummarizeNode:
    """单节点两轮摘要。"""

    def test_both_rounds_succeed(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "view_node_extract.md").write_text(
            "提取 prompt", encoding="utf-8"
        )
        (prompt_dir / "view_node_verify.md").write_text("验证 prompt", encoding="utf-8")

        client = _make_mock_client("提取结果", "验证结果")
        result = summarize_node(client, "节点原始文本", "问题", prompt_dir)
        assert "[内容摘要]" in result
        assert "提取结果" in result
        assert "[核实]" in result
        assert "验证结果" in result

    def test_extract_fails(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "view_node_extract.md").write_text("p", encoding="utf-8")

        client = MagicMock()
        client.chat.side_effect = Exception("API 超时")
        result = summarize_node(client, "文本", "问题", prompt_dir)
        assert "[摘要错误]" in result

    def test_verify_fails_returns_extract_only(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "view_node_extract.md").write_text("p", encoding="utf-8")
        (prompt_dir / "view_node_verify.md").write_text("p", encoding="utf-8")

        extract_resp = MagicMock()
        extract_resp.choices = [MagicMock()]
        extract_resp.choices[0].message.content = "提取OK"
        client = MagicMock()
        client.chat.side_effect = [extract_resp, Exception("验证失败")]
        result = summarize_node(client, "文本", "问题", prompt_dir)
        assert "提取OK" in result
        assert "跳过" in result


class TestSummarizeChildren:
    """子节点相关性标注。"""

    def test_annotates_children(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "view_node_children_extract.md").write_text("p", encoding="utf-8")
        (prompt_dir / "view_node_children_verify.md").write_text("p", encoding="utf-8")

        children = [
            {"id": "L2_000", "time_range": [0, 30], "summary": "开场"},
            {"id": "L2_001", "time_range": [30, 60], "summary": "主体"},
        ]
        client = _make_mock_client("★L2_001 高相关", "核实通过")
        result = summarize_children(client, children, "问题", prompt_dir)
        assert "L2_001" in result


class TestSummarizeNodesBatch:
    """并发批量摘要。"""

    def test_batch_returns_ordered_results(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "search_similar_extract.md").write_text("p", encoding="utf-8")
        (prompt_dir / "search_similar_verify.md").write_text("p", encoding="utf-8")

        def fake_chat(messages: Any, **kwargs: Any) -> MagicMock:
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "摘要"
            return resp

        client = MagicMock()
        client.chat.side_effect = fake_chat

        items = [
            ("node_a", "文本A", "extra_a"),
            ("node_b", "文本B", "extra_b"),
        ]
        results = summarize_nodes_batch(client, items, "问题", prompt_dir)
        assert len(results) == 2
        assert results[0][0] == "node_a"
        assert results[1][0] == "node_b"
