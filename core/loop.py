"""Agent Loop 引擎的核心数据结构，支持 Thinking+JSON 推理模式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class Step:
    """记录单步 Agent 决策：思考 → 结构化 JSON（reflect/plan/action）→ 工具调度 → 工具输出。"""

    thought: str
    reflect: dict[str, Any]
    plan: dict[str, Any]
    tool_call: dict[str, Any]
    tool_output: str
    raw_content: str


@dataclass
class LoopResult:
    """Agent Loop 完整运行的输出结果汇总。"""

    result: dict[str, Any] | None = None
    steps: list[Step] = field(default_factory=list)
    steps_used: int = 0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    stop_reason: str = "finished"


class LoopHook:
    """Agent Loop 生命周期钩子基类，默认方法均为空操作。"""

    def before_step(self, iteration: int, messages: list[dict]) -> None:
        return None

    def after_tool(self, iteration: int, step: Step) -> None:
        return None

    def after_step(self, iteration: int, messages: list[dict]) -> None:
        return None

    def on_finish(self, result: LoopResult) -> None:
        return None


class CompositeHook(LoopHook):
    """组合多个 LoopHook，异常隔离确保单个钩子失败不影响其他钩子。"""

    def __init__(self, hooks: list[LoopHook]) -> None:
        self._hooks = hooks

    def _safe_call(self, method: str, *args: Any, **kwargs: Any) -> None:
        for hook in self._hooks:
            try:
                getattr(hook, method)(*args, **kwargs)
            except Exception:
                logger.exception("LoopHook 调用失败: {}", method)

    def before_step(self, iteration: int, messages: list[dict]) -> None:
        self._safe_call("before_step", iteration, messages)

    def after_tool(self, iteration: int, step: Step) -> None:
        self._safe_call("after_tool", iteration, step)

    def after_step(self, iteration: int, messages: list[dict]) -> None:
        self._safe_call("after_step", iteration, messages)

    def on_finish(self, result: LoopResult) -> None:
        self._safe_call("on_finish", result)
