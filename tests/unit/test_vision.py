"""vision 单元测试 — 用 mock LLMClient 验证两轮 VL 逻辑。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from core.tree.vision import observe_frame


def _make_mock_vl_client(*responses: str) -> MagicMock:
    """创建返回固定序列响应的 mock VL LLMClient。"""
    client = MagicMock()
    side_effects = []
    for text in responses:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = text
        side_effects.append(resp)
    client.chat.side_effect = side_effects
    return client


class TestObserveFrame:
    """两轮 VL 帧观察。"""

    def test_both_rounds_succeed(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "observe_frame_extract.md").write_text("提取", encoding="utf-8")
        (prompt_dir / "observe_frame_verify.md").write_text("验证", encoding="utf-8")

        frame = tmp_path / "frame.jpg"
        frame.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        )

        client = _make_mock_vl_client("画面中有两个人", "核实：确认两人")
        result = observe_frame(client, [frame], "画面中有几个人？", prompt_dir)
        assert "[视觉观察]" in result
        assert "两个人" in result
        assert "[验证]" in result

    def test_frame_not_found(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        client = MagicMock()
        result = observe_frame(
            client, [tmp_path / "nonexistent.jpg"], "问题", prompt_dir
        )
        assert "[VL错误]" in result

    def test_extract_fails(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "observe_frame_extract.md").write_text("p", encoding="utf-8")

        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")

        client = MagicMock()
        client.chat.side_effect = Exception("VL API 超时")
        result = observe_frame(client, [frame], "问题", prompt_dir)
        assert "[VL错误]" in result

    def test_verify_fails_returns_extract_only(self, tmp_path: Path) -> None:
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "observe_frame_extract.md").write_text("p", encoding="utf-8")
        (prompt_dir / "observe_frame_verify.md").write_text("p", encoding="utf-8")

        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")

        extract_resp = MagicMock()
        extract_resp.choices = [MagicMock()]
        extract_resp.choices[0].message.content = "视觉证据"
        client = MagicMock()
        client.chat.side_effect = [extract_resp, Exception("验证失败")]
        result = observe_frame(client, [frame], "问题", prompt_dir)
        assert "视觉证据" in result
        assert "跳过" in result
