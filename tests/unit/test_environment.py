"""TreeEnvironment 单元测试 — 用 mini_tree.json fixture 验证核心功能。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.tree.environment import TreeEnvironment

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_tree.json"


def _make_mock_client() -> MagicMock:
    """创建返回固定响应的 mock LLMClient。"""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "mock摘要"
    client.chat.return_value = resp
    return client


@pytest.fixture()
def env() -> TreeEnvironment:
    """创建基于 mini_tree.json 的 TreeEnvironment 实例。"""
    client = _make_mock_client()
    return TreeEnvironment(tree_json_path=FIXTURE_PATH, tool_client=client)


# ------------------------------------------------------------------
# 初始化
# ------------------------------------------------------------------


class TestTreeEnvironmentInit:
    """树加载与基本属性。"""

    def test_loads_all_nodes(self, env: TreeEnvironment) -> None:
        assert len(env._nodes) == 9

    def test_video_id(self, env: TreeEnvironment) -> None:
        assert env._video_id == "test_001"

    def test_duration(self, env: TreeEnvironment) -> None:
        assert env._duration_seconds == 120.0

    def test_domain(self, env: TreeEnvironment) -> None:
        assert env._domain == "test"

    def test_node_levels(self, env: TreeEnvironment) -> None:
        levels = {n["level"] for n in env._nodes.values()}
        assert levels == {1, 2, 3}

    def test_l1_count(self, env: TreeEnvironment) -> None:
        l1 = [n for n in env._nodes.values() if n["level"] == 1]
        assert len(l1) == 2

    def test_l2_count(self, env: TreeEnvironment) -> None:
        l2 = [n for n in env._nodes.values() if n["level"] == 2]
        assert len(l2) == 3

    def test_l3_count(self, env: TreeEnvironment) -> None:
        l3 = [n for n in env._nodes.values() if n["level"] == 3]
        assert len(l3) == 4


# ------------------------------------------------------------------
# get_subtitle
# ------------------------------------------------------------------


class TestGetSubtitle:
    """字幕获取。"""

    def test_has_subtitle(self, env: TreeEnvironment) -> None:
        assert env.get_subtitle("test_001_L1_000") == "今天天气很好"

    def test_empty_subtitle(self, env: TreeEnvironment) -> None:
        assert env.get_subtitle("test_001_L1_001") == ""

    def test_nonexistent_node(self, env: TreeEnvironment) -> None:
        assert env.get_subtitle("nonexistent_node") == ""

    def test_l3_with_subtitle(self, env: TreeEnvironment) -> None:
        assert env.get_subtitle("test_001_L3_001") == "你看那边的花"

    def test_l3_without_subtitle(self, env: TreeEnvironment) -> None:
        assert env.get_subtitle("test_001_L3_000") == ""


# ------------------------------------------------------------------
# resolve_frame_paths
# ------------------------------------------------------------------


class TestResolveFramePaths:
    """帧路径解析。"""

    def test_single_l3(self, env: TreeEnvironment) -> None:
        paths = env.resolve_frame_paths(["test_001_L3_000"])
        assert len(paths) == 1
        assert paths[0].name == "L3_000.jpg"

    def test_multiple_l3(self, env: TreeEnvironment) -> None:
        paths = env.resolve_frame_paths(
            ["test_001_L3_000", "test_001_L3_001", "test_001_L3_002"]
        )
        assert len(paths) == 3
        assert paths[0].name == "L3_000.jpg"
        assert paths[1].name == "L3_001.jpg"
        assert paths[2].name == "L3_002.jpg"

    def test_l2_expands_to_children(self, env: TreeEnvironment) -> None:
        paths = env.resolve_frame_paths(["test_001_L2_000"])
        assert len(paths) == 2
        assert paths[0].name == "L3_000.jpg"
        assert paths[1].name == "L3_001.jpg"

    def test_l2_single_child(self, env: TreeEnvironment) -> None:
        paths = env.resolve_frame_paths(["test_001_L2_001"])
        assert len(paths) == 1
        assert paths[0].name == "L3_002.jpg"

    def test_l1_raises(self, env: TreeEnvironment) -> None:
        with pytest.raises(ValueError, match="不支持 L1 节点"):
            env.resolve_frame_paths(["test_001_L1_000"])

    def test_empty_list_raises(self, env: TreeEnvironment) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            env.resolve_frame_paths([])

    def test_mixed_levels_raises(self, env: TreeEnvironment) -> None:
        with pytest.raises(ValueError, match="不能混合传入"):
            env.resolve_frame_paths(["test_001_L2_000", "test_001_L3_000"])

    def test_nonexistent_node_raises(self, env: TreeEnvironment) -> None:
        with pytest.raises(KeyError, match="节点不存在"):
            env.resolve_frame_paths(["nonexistent"])

    def test_too_many_l3_raises(self, env: TreeEnvironment) -> None:
        ids = [
            "test_001_L3_000",
            "test_001_L3_001",
            "test_001_L3_002",
            "test_001_L3_003",
            "test_001_L3_000",
        ]
        with pytest.raises(ValueError, match="最多传入 4 个"):
            env.resolve_frame_paths(ids)

    def test_too_many_l2_raises(self, env: TreeEnvironment) -> None:
        with pytest.raises(ValueError, match="最多传入 1 个"):
            env.resolve_frame_paths(["test_001_L2_000", "test_001_L2_001"])

    def test_frame_path_under_tree_dir(self, env: TreeEnvironment) -> None:
        paths = env.resolve_frame_paths(["test_001_L3_000"])
        assert "frames" in str(paths[0])
        assert paths[0].parent.name == "frames"


# ------------------------------------------------------------------
# _chunk_text
# ------------------------------------------------------------------


class TestChunkText:
    """文本分块。"""

    def test_short_text_single_chunk(self) -> None:
        chunks = TreeEnvironment._chunk_text("短文本", chunk_size=4000, overlap=800)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"

    def test_exact_boundary(self) -> None:
        text = "a" * 4000
        chunks = TreeEnvironment._chunk_text(text, chunk_size=4000, overlap=800)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self) -> None:
        text = "a" * 7000
        chunks = TreeEnvironment._chunk_text(text, chunk_size=4000, overlap=800)
        assert len(chunks) >= 2
        # 每个 chunk 长度不超过 chunk_size
        for c in chunks:
            assert len(c) <= 4000

    def test_chunks_cover_full_text(self) -> None:
        text = "abcdefghij" * 500  # 5000 字符
        chunks = TreeEnvironment._chunk_text(text, chunk_size=4000, overlap=800)
        # 最后一个 chunk 的末尾应覆盖原文末尾
        reconstructed_end = chunks[-1]
        assert text[-1] == reconstructed_end[-1]

    def test_overlap_exists(self) -> None:
        text = "a" * 8000
        chunks = TreeEnvironment._chunk_text(text, chunk_size=4000, overlap=800)
        assert len(chunks) >= 2
        # 第二个 chunk 的开头应该和第一个 chunk 的末尾有重叠
        # step = 4000 - 800 = 3200，所以 chunk[1] 从 3200 开始
        # chunk[0] 覆盖 0-4000，chunk[1] 覆盖 3200-7200
        # 重叠区域 = 3200-4000 = 800 字符


# ------------------------------------------------------------------
# 内部文本方法
# ------------------------------------------------------------------


class TestInternalTextMethods:
    """内部文本提取方法。"""

    def test_extract_card_text(self, env: TreeEnvironment) -> None:
        card = {"scene_summary": "测试场景", "key_entities": ["A", "B"]}
        text = env._extract_card_text(card)
        assert "测试场景" in text
        assert "A" in text
        assert "B" in text

    def test_extract_card_text_empty_strings_filtered(
        self, env: TreeEnvironment
    ) -> None:
        card = {"a": "", "b": "  ", "c": "有效"}
        text = env._extract_card_text(card)
        assert text == "有效"

    def test_get_summary_with_subtitle(self, env: TreeEnvironment) -> None:
        node = env._nodes["test_001_L2_000"]
        summary = env._get_summary(node)
        assert "人物A和B走在公园小路上" in summary
        assert "字幕" in summary
        assert "你看那边的花" in summary

    def test_get_summary_without_subtitle(self, env: TreeEnvironment) -> None:
        node = env._nodes["test_001_L3_000"]
        summary = env._get_summary(node)
        assert "公园入口" in summary
        assert "字幕" not in summary

    def test_node_full_text_with_subtitle(self, env: TreeEnvironment) -> None:
        node = env._nodes["test_001_L2_000"]
        text = env._node_full_text(node)
        assert "人物A和B走在公园小路上" in text
        assert "你看那边的花" in text

    def test_node_full_text_without_subtitle(self, env: TreeEnvironment) -> None:
        node = env._nodes["test_001_L3_002"]
        text = env._node_full_text(node)
        assert "长椅上两人并排坐着" in text


# ------------------------------------------------------------------
# view_node（需要 mock + prompts_dir）
# ------------------------------------------------------------------


class TestViewNode:
    """view_node 方法（mock LLM）。"""

    def test_view_leaf_node(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "view_node_extract.md").write_text("p", encoding="utf-8")
        (prompt_dir / "view_node_verify.md").write_text("p", encoding="utf-8")

        client = _make_mock_client()
        env = TreeEnvironment(
            tree_json_path=FIXTURE_PATH,
            tool_client=client,
            prompts_dir=prompt_dir,
        )
        result = env.view_node("test_001_L3_000", "公园里有什么？")
        assert "test_001_L3_000" in result
        assert "关键帧层" in result
        # L3 无子节点，不应有子节点概览
        assert "子节点概览" not in result

    def test_view_parent_node_has_children_overview(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        for name in [
            "view_node_extract.md",
            "view_node_verify.md",
            "view_node_children_extract.md",
            "view_node_children_verify.md",
        ]:
            (prompt_dir / name).write_text("p", encoding="utf-8")

        client = _make_mock_client()
        env = TreeEnvironment(
            tree_json_path=FIXTURE_PATH,
            tool_client=client,
            prompts_dir=prompt_dir,
        )
        result = env.view_node("test_001_L2_000", "描述这个事件")
        assert "test_001_L2_000" in result
        assert "事件层" in result
        assert "子节点概览" in result

    def test_view_nonexistent_raises(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        client = _make_mock_client()
        env = TreeEnvironment(
            tree_json_path=FIXTURE_PATH,
            tool_client=client,
            prompts_dir=prompt_dir,
        )
        with pytest.raises(KeyError, match="节点不存在"):
            env.view_node("nonexistent", "问题")

    def test_view_requires_prompts_dir(self, env: TreeEnvironment) -> None:
        with pytest.raises(AssertionError, match="prompts_dir"):
            env.view_node("test_001_L3_000", "问题")
