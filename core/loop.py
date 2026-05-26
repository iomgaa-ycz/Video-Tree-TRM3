"""Agent Loop 引擎 — Thinking+JSON 推理循环，pluggy 驱动 hook。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pluggy
from pluggy import _manager
from pluggy._callers import HookCallError, _raise_wrapfail, run_old_style_hookwrapper
from typing import cast


def _multicall_in_registration_order(
    hook_name: str,
    hook_impls: list[Any],
    caller_kwargs: dict[str, object],
    firstresult: bool,
) -> object | list[object]:
    """按插件注册顺序执行 hook，实现与项目测试约定一致的行为。"""

    results: list[object] = []
    exception: BaseException | None = None
    teardowns: list[Any] = []
    try:
        for hook_impl in hook_impls:
            try:
                args = [caller_kwargs[argname] for argname in hook_impl.argnames]
            except KeyError as exc:
                for argname in hook_impl.argnames:
                    if argname not in caller_kwargs:
                        raise HookCallError(
                            f"hook call must provide argument {argname!r}"
                        ) from exc
                raise

            if hook_impl.hookwrapper:
                function_gen = run_old_style_hookwrapper(hook_impl, hook_name, args)
                next(function_gen)
                teardowns.append(function_gen)
                continue

            if hook_impl.wrapper:
                try:
                    res = hook_impl.function(*args)
                    function_gen = cast(object, res)
                    next(function_gen)
                    teardowns.append(function_gen)
                except StopIteration:
                    _raise_wrapfail(function_gen, "did not yield")
                continue

            res = hook_impl.function(*args)
            results.append(res)
            if firstresult and res is not None:
                break
    except BaseException as exc:
        exception = exc
    finally:
        result = results[0] if firstresult else results
        for teardown in reversed(teardowns):
            try:
                if exception is not None:
                    try:
                        teardown.throw(exception)
                    except RuntimeError as runtime_error:
                        if (
                            isinstance(exception, StopIteration)
                            and runtime_error.__cause__ is exception
                        ):
                            teardown.close()
                            continue
                        raise
                else:
                    teardown.send(result)
                teardown.close()
            except StopIteration as stop_iteration:
                result = stop_iteration.value
                exception = None
                continue
            except BaseException as exc:
                exception = exc
                continue
            _raise_wrapfail(teardown, "has second yield")

    if exception is not None:
        raise exception
    return result


_manager._multicall = _multicall_in_registration_order

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
