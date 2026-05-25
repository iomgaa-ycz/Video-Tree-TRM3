"""ASRClient 单元测试。"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.asr_client import ASRClient


class TestASRClientFromEnv:
    """from_env 工厂方法。"""

    def test_creates_client_from_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TEST_ASR_MODEL", "whisper-large-v3")
        monkeypatch.setenv("TEST_ASR_BASE_URL", "https://api.groq.com/openai/v1")
        monkeypatch.setenv("TEST_ASR_API_KEY", "gsk-test")

        client = ASRClient.from_env("TEST_ASR")

        assert client.model == "whisper-large-v3"

    def test_missing_env_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            ASRClient.from_env("NONEXISTENT_PREFIX")


class TestASRClientTranscribe:
    """transcribe() 方法。"""

    def test_transcribe_calls_audio_api(self, monkeypatch: Any, tmp_path: Any) -> None:
        monkeypatch.setenv("MOCK_ASR_MODEL", "whisper-large-v3")
        monkeypatch.setenv("MOCK_ASR_BASE_URL", "https://api.groq.com/openai/v1")
        monkeypatch.setenv("MOCK_ASR_API_KEY", "gsk-test")

        client = ASRClient.from_env("MOCK_ASR")

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        with patch.object(
            client._client.audio.transcriptions,
            "create",
            return_value=mock_response,
        ) as mock_create:
            with audio_file.open("rb") as fh:
                result = client.transcribe(file=fh)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "whisper-large-v3"
            assert call_kwargs["response_format"] == "verbose_json"
            assert call_kwargs["timestamp_granularities"] == ["segment"]
            assert result is mock_response

    def test_transcribe_passes_language(self, monkeypatch: Any, tmp_path: Any) -> None:
        monkeypatch.setenv("MOCK_ASR_MODEL", "whisper-large-v3")
        monkeypatch.setenv("MOCK_ASR_BASE_URL", "https://api.groq.com/openai/v1")
        monkeypatch.setenv("MOCK_ASR_API_KEY", "gsk-test")

        client = ASRClient.from_env("MOCK_ASR")

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        mock_response = MagicMock()
        with patch.object(
            client._client.audio.transcriptions,
            "create",
            return_value=mock_response,
        ) as mock_create:
            with audio_file.open("rb") as fh:
                client.transcribe(file=fh, language="zh")

            assert mock_create.call_args.kwargs["language"] == "zh"
