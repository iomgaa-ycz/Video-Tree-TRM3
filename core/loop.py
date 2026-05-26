"""Agent Loop 引擎 — Thinking+JSON 推理循环，pluggy 驱动 hook。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pluggy

hookspec = pluggy.HookspecMarker("agent_loop")
hookimpl = pluggy.HookimplMarker("agent_loop")


class AgentLoopSpec:
    """AgentLoop 生命周期扩展点。

    每个 hookimpl 可选择观察（返回 None）或变换（返回值）。
    """

    @hookspec
    def before_step(self, iteration: int, messages: list[dict]) -> None:
        """LLM 调用前触发。"""

    @hookspec
    def after_tool(self, iteration: int, step: Step) -> str | None:
        """工具执行后触发。返回非 None 字符串时，拼接到反馈消息。"""

    @hookspec
    def after_step(self, iteration: int, messages: list[dict]) -> None:
        """一轮结束后触发。"""

    @hookspec
    def on_finish(self, result: LoopResult) -> None:
        """循环终止后触发。"""


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
