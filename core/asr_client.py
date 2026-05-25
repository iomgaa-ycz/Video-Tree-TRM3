"""ASR 客户端 — OpenAI 兼容音频转录，带重试。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, BinaryIO

from dotenv import load_dotenv
from httpx import Timeout
from loguru import logger
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# 模块级加载 .env，确保 from_env() 工厂方法可直接读取环境变量
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError)
    ),
    reraise=True,
)
def _transcribe_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    """带指数退避重试的音频转录调用。

    参数:
        client: OpenAI 兼容客户端。
        **kwargs: 透传给 client.audio.transcriptions.create()。

    返回:
        Transcription 响应对象。
    """
    return client.audio.transcriptions.create(**kwargs)


class ASRClient:
    """OpenAI 兼容 ASR 客户端。

    参数:
        client: OpenAI 实例。
        model: ASR 模型名称。
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        """模型名称。"""
        return self._model

    @classmethod
    def from_env(cls, prefix: str) -> ASRClient:
        """从环境变量创建 ASRClient。

        读取 {prefix}_MODEL / {prefix}_BASE_URL / {prefix}_API_KEY。

        参数:
            prefix: 环境变量前缀，如 "ASR"。

        返回:
            ASRClient 实例。

        异常:
            KeyError: 环境变量缺失时抛出。
        """
        model = os.environ[f"{prefix}_MODEL"]
        base_url = os.environ[f"{prefix}_BASE_URL"]
        api_key = os.environ[f"{prefix}_API_KEY"]
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=Timeout(connect=10, read=300, write=300, pool=60),
        )
        logger.debug("ASRClient 就绪: prefix={}, model={}", prefix, model)
        return cls(client, model)

    def transcribe(
        self,
        file: BinaryIO,
        *,
        language: str | None = None,
    ) -> Any:
        """转录音频文件。

        参数:
            file: 已打开的音频文件句柄（rb 模式）。
            language: 转录语言代码，如 "zh"、"en"，None 时自动检测。

        返回:
            Transcription 响应对象（verbose_json 格式，含 segment 时间戳）。
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "file": file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language is not None:
            kwargs["language"] = language
        return _transcribe_with_retry(self._client, **kwargs)
