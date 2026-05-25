"""LLMClient 单元测试。"""

from __future__ import annotations

from core.llm_client import detect_provider, _build_thinking_body


class TestDetectProvider:
    """从 base_url 推断提供商。"""

    def test_deepseek(self) -> None:
        assert detect_provider("https://api.deepseek.com/v1") == "deepseek"

    def test_qwen_dashscope(self) -> None:
        assert (
            detect_provider("https://dashscope.aliyuncs.com/compatible-mode/v1")
            == "qwen"
        )

    def test_qwen_dashscope_intl(self) -> None:
        assert (
            detect_provider("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
            == "qwen"
        )

    def test_unknown_provider(self) -> None:
        assert detect_provider("https://api.groq.com/openai/v1") == "unknown"


class TestBuildThinkingBody:
    """根据提供商生成 extra_body。"""

    def test_deepseek_enabled(self) -> None:
        assert _build_thinking_body("deepseek", True) == {
            "thinking": {"type": "enabled"}
        }

    def test_deepseek_disabled(self) -> None:
        assert _build_thinking_body("deepseek", False) == {
            "thinking": {"type": "disabled"}
        }

    def test_qwen_enabled(self) -> None:
        assert _build_thinking_body("qwen", True) == {"enable_thinking": True}

    def test_qwen_disabled(self) -> None:
        assert _build_thinking_body("qwen", False) == {"enable_thinking": False}

    def test_unknown_returns_empty(self) -> None:
        assert _build_thinking_body("unknown", True) == {}
