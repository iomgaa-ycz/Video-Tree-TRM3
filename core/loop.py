"""Agent Loop 引擎 — Thinking+JSON 推理循环，pluggy 驱动 hook。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pluggy
from json_repair import repair_json
from loguru import logger

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


class AgentLoop:
    """Thinking+JSON 推理循环引擎。

    类比 nn.Module：接收 prompt + 工具函数，返回 LoopResult。
    不感知视频树、QA、数据库等领域概念。

    参数:
        client: LLMClient 实例（duck typing，需提供 chat 方法）。
        max_steps: 最大有效步数（每次成功工具调用计一步）。
        max_retries: JSON 解析连续失败的最大容忍次数。
    """

    def __init__(
        self,
        client: Any,
        *,
        max_steps: int = 15,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._max_steps = max_steps
        self._max_retries = max_retries

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_fn: Callable[[str, dict[str, Any]], str],
        plugins: list[object] | None = None,
    ) -> LoopResult:
        """执行 Thinking+JSON 推理循环。

        参数:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            tool_fn: 工具执行函数，(name, args) → str，抛 ValueError 表示无效调用。
            plugins: pluggy 插件列表。

        返回:
            LoopResult 实例。
        """
        pm = self._create_plugin_manager(plugins)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        steps: list[Step] = []
        token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        step_count = 0
        retry_count = 0
        iteration = 0

        while step_count < self._max_steps:
            pm.hook.before_step(iteration=iteration, messages=messages)

            # Phase 1: LLM 调用
            try:
                response = self._call_llm(messages, token_usage)
            except Exception as e:
                logger.error("LLM API 调用失败: {}", e)
                result = LoopResult(
                    steps=steps,
                    steps_used=step_count,
                    token_usage=token_usage,
                    stop_reason="error",
                )
                pm.hook.on_finish(result=result)
                return result

            # Phase 2: 解析响应
            parsed = self._parse_response(response)
            if parsed is None:
                retry_count += 1
                logger.warning(
                    "响应解析失败 (retry {}/{})", retry_count, self._max_retries
                )
                raw = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你的输出不是合法 JSON。请严格输出 JSON 格式："
                            '{"reflect": {...}, "plan": {...}, '
                            '"action": {"tool": "...", "args": {...}}}'
                        ),
                    }
                )
                if retry_count >= self._max_retries:
                    result = LoopResult(
                        steps=steps,
                        steps_used=step_count,
                        token_usage=token_usage,
                        stop_reason="parse_error",
                    )
                    pm.hook.after_step(iteration=iteration, messages=messages)
                    pm.hook.on_finish(result=result)
                    return result
                pm.hook.after_step(iteration=iteration, messages=messages)
                iteration += 1
                continue

            thought, reflect, plan, raw_content, action = parsed
            retry_count = 0
            messages.append({"role": "assistant", "content": raw_content})

            # Phase 3: 执行工具
            tool_name: str = action["tool"]
            tool_args: dict[str, Any] = action["args"]
            output, is_valid = self._execute_tool(tool_fn, tool_name, tool_args)
            if not is_valid:
                messages.append(
                    {
                        "role": "user",
                        "content": f"[工具调用无效: {tool_name}] {output}",
                    }
                )
                pm.hook.after_step(iteration=iteration, messages=messages)
                iteration += 1
                continue

            step_count += 1
            step = Step(
                thought=thought,
                reflect=reflect,
                plan=plan,
                tool_call={"tool": tool_name, "args": tool_args},
                tool_output=output,
                raw_content=raw_content,
            )
            steps.append(step)

            # Phase 4: Hook + 反馈
            hints = pm.hook.after_tool(iteration=iteration, step=step)
            feedback = self._build_feedback(tool_name, output, hints)
            messages.append(feedback)
            pm.hook.after_step(iteration=iteration, messages=messages)

            # Phase 5: 终止检查
            if tool_name == "submit_answer":
                result = LoopResult(
                    result=tool_args,
                    steps=steps,
                    steps_used=step_count,
                    token_usage=token_usage,
                    stop_reason="finished",
                )
                pm.hook.on_finish(result=result)
                return result

            iteration += 1

        result = LoopResult(
            steps=steps,
            steps_used=step_count,
            token_usage=token_usage,
            stop_reason="budget_exceeded",
        )
        pm.hook.on_finish(result=result)
        return result

    def _create_plugin_manager(
        self, plugins: list[object] | None
    ) -> pluggy.PluginManager:
        """创建并注册 plugins 的 PluginManager。"""
        pm = pluggy.PluginManager("agent_loop")
        pm.add_hookspecs(AgentLoopSpec)
        for plugin in plugins or []:
            pm.register(plugin)
        return pm

    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        token_usage: dict[str, int],
    ) -> Any:
        """调用 LLMClient 并累加 token 使用量。

        参数:
            messages: 消息历史。
            token_usage: 可变字典，就地累加。

        返回:
            ChatCompletion 响应对象。
        """
        response = self._client.chat(messages)
        usage = response.usage
        if usage is not None:
            token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            token_usage["completion_tokens"] += (
                getattr(usage, "completion_tokens", 0) or 0
            )
        return response

    def _parse_response(
        self, response: Any
    ) -> tuple[str, dict, dict, str, dict] | None:
        """从 LLM 响应中提取 thought、reflect、plan、raw_content、action。

        参数:
            response: ChatCompletion 响应对象。

        返回:
            解析成功返回 (thought, reflect, plan, raw_content, action)；
            解析失败返回 None。
        """
        msg = response.choices[0].message
        content = msg.content or ""
        thought = getattr(msg, "reasoning_content", "") or ""

        if not content.strip():
            return None

        repaired = repair_json(content)
        try:
            data = json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict) or "action" not in data:
            return None

        action = data["action"]
        if not isinstance(action, dict) or "tool" not in action or "args" not in action:
            return None

        reflect = data.get("reflect", {})
        plan = data.get("plan", {})
        return thought, reflect, plan, content, action

    def _execute_tool(
        self,
        tool_fn: Callable[[str, dict[str, Any]], str],
        name: str,
        args: dict[str, Any],
    ) -> tuple[str, bool]:
        """执行工具调用。

        参数:
            tool_fn: 工具执行函数。
            name: 工具名称。
            args: 工具参数。

        返回:
            (output, is_valid) — ValueError 时 is_valid=False 且不计步数。
        """
        try:
            output = tool_fn(name, args)
            return output, True
        except ValueError as e:
            return f"工具调用失败: {e}", False

    def _build_feedback(
        self,
        tool_name: str,
        tool_output: str,
        hints: list[str | None],
    ) -> dict[str, Any]:
        """组装工具结果反馈消息。

        参数:
            tool_name: 工具名称。
            tool_output: 工具原始输出。
            hints: hook 返回的 hint 列表（含 None）。

        返回:
            user role 消息字典。
        """
        parts = [f"[工具执行结果: {tool_name}]", tool_output]
        for hint in hints:
            if hint is not None:
                parts.append(hint)
        return {"role": "user", "content": "\n".join(parts)}
