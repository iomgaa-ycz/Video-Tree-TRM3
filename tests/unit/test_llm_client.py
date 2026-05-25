"""LLMClient 单元测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import LLMClient, _build_thinking_body, detect_provider


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


class TestLLMClientFromEnv:
    """from_env 工厂方法。"""

    def test_creates_client_from_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TEST_LLM_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("TEST_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("TEST_LLM_API_KEY", "sk-test")

        client = LLMClient.from_env("TEST_LLM", thinking=True)

        assert client.model == "deepseek-v4-pro"
        assert client.provider == "deepseek"
        assert client.thinking is True

    def test_qwen_provider_detected(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("VL_MODEL", "qwen3.6-plus")
        monkeypatch.setenv(
            "VL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        monkeypatch.setenv("VL_API_KEY", "sk-test")

        client = LLMClient.from_env("VL")

        assert client.provider == "qwen"
        assert client.thinking is False

    def test_missing_env_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            LLMClient.from_env("NONEXISTENT_PREFIX")


class TestLLMClientChat:
    """chat() 方法。"""

    def test_chat_injects_thinking_body(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MOCK_LLM_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("MOCK_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("MOCK_LLM_API_KEY", "sk-test")

        client = LLMClient.from_env("MOCK_LLM", thinking=False)

        mock_response = MagicMock()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(messages=[{"role": "user", "content": "hello"}])
            assert mock_create.call_args.kwargs["extra_body"] == {
                "thinking": {"type": "disabled"}
            }

    def test_chat_per_call_thinking_override(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MOCK_LLM_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("MOCK_LLM_BASE_URL", "https://api.deepseek.com/v1")
        monkeypatch.setenv("MOCK_LLM_API_KEY", "sk-test")

        client = LLMClient.from_env("MOCK_LLM", thinking=False)

        mock_response = MagicMock()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(messages=[{"role": "user", "content": "hello"}], thinking=True)
            assert mock_create.call_args.kwargs["extra_body"] == {
                "thinking": {"type": "enabled"}
            }

    def test_chat_extra_body_merges(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MOCK_LLM_MODEL", "qwen3.6-plus")
        monkeypatch.setenv(
            "MOCK_LLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        monkeypatch.setenv("MOCK_LLM_API_KEY", "sk-test")

        client = LLMClient.from_env("MOCK_LLM", thinking=True)

        mock_response = MagicMock()
        with patch.object(
            client._client.chat.completions, "create", return_value=mock_response
        ) as mock_create:
            client.chat(
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"enable_search": False},
            )
            assert mock_create.call_args.kwargs["extra_body"] == {
                "enable_thinking": True,
                "enable_search": False,
            }
