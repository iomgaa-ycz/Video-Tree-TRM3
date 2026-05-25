"""LLM 客户端 — OpenAI 兼容，内置 thinking 适配与重试。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def detect_provider(base_url: str) -> str:
    """从 base_url 推断提供商。

    参数:
        base_url: OpenAI 兼容 API 的 base URL。

    返回:
        提供商标识 — "deepseek" / "qwen" / "unknown"。
    """
    url = base_url.lower()
    if "deepseek.com" in url:
        return "deepseek"
    if "dashscope" in url:
        return "qwen"
    return "unknown"


def _build_thinking_body(provider: str, thinking: bool) -> dict[str, Any]:
    """根据提供商构建 thinking 相关的 extra_body。

    参数:
        provider: 提供商标识。
        thinking: 是否启用 thinking。

    返回:
        extra_body 字典，unknown 提供商返回空字典。
    """
    if provider == "deepseek":
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}
    if provider == "qwen":
        return {"enable_thinking": thinking}
    return {}


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, RateLimitError)
    ),
    reraise=True,
)
def _chat_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    """带指数退避重试的 chat completions 调用。

    参数:
        client: OpenAI 兼容客户端。
        **kwargs: 透传给 client.chat.completions.create()。

    返回:
        ChatCompletion 响应对象。
    """
    return client.chat.completions.create(**kwargs)


class LLMClient:
    """OpenAI 兼容 LLM 客户端，内置 thinking 适配。

    参数:
        client: OpenAI 实例。
        model: 模型名称。
        provider: 提供商标识。
        thinking: 默认 thinking 开关。
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        provider: str,
        thinking: bool = False,
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider
        self._thinking = thinking

    @property
    def model(self) -> str:
        """模型名称。"""
        return self._model

    @property
    def provider(self) -> str:
        """提供商标识。"""
        return self._provider

    @property
    def thinking(self) -> bool:
        """默认 thinking 开关。"""
        return self._thinking

    @classmethod
    def from_env(cls, prefix: str, thinking: bool = False) -> LLMClient:
        """从环境变量创建 LLMClient。

        读取 {prefix}_MODEL / {prefix}_BASE_URL / {prefix}_API_KEY。

        参数:
            prefix: 环境变量前缀，如 "SEARCH_LLM"。
            thinking: 默认 thinking 开关。

        返回:
            LLMClient 实例。

        异常:
            KeyError: 环境变量缺失时抛出。
        """
        model = os.environ[f"{prefix}_MODEL"]
        base_url = os.environ[f"{prefix}_BASE_URL"]
        api_key = os.environ[f"{prefix}_API_KEY"]
        provider = detect_provider(base_url)
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=Timeout(connect=10, read=180, write=180, pool=60),
        )
        logger.debug(
            "LLMClient 就绪: prefix={}, model={}, provider={}", prefix, model, provider
        )
        return cls(client, model, provider, thinking)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """发送 chat 请求，自动注入 thinking extra_body。

        参数:
            messages: OpenAI 格式的消息列表。
            thinking: 覆盖实例默认 thinking 开关，None 时用默认值。
            extra_body: 额外的 body 字段，与自动生成的合并（调用方优先）。

        返回:
            ChatCompletion 响应对象。
        """
        effective_thinking = thinking if thinking is not None else self._thinking
        body = _build_thinking_body(self._provider, effective_thinking)
        if extra_body:
            body.update(extra_body)
        return _chat_with_retry(
            self._client,
            model=self._model,
            messages=messages,
            extra_body=body,
        )
