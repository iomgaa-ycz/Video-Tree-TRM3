"""LLM 客户端 — OpenAI 兼容，内置 thinking 适配。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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
