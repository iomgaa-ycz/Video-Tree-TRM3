"""core.loop 模块的单元测试。"""

from __future__ import annotations

from core.loop import CompositeHook, LoopHook, LoopResult, Step


def test_step_construction_with_all_fields() -> None:
    """验证 Step 可使用完整字段正确构造。"""

    step = Step(
        thought="先分析任务",
        reflect={"status": "ok"},
        plan={"next": "调用工具"},
        tool_call={"tool": "search", "args": {"query": "tree"}},
        tool_output="工具输出内容",
        raw_content="原始模型输出",
    )

    assert step.thought == "先分析任务"
    assert step.reflect == {"status": "ok"}
    assert step.plan == {"next": "调用工具"}
    assert step.tool_call == {"tool": "search", "args": {"query": "tree"}}
    assert step.tool_output == "工具输出内容"
    assert step.raw_content == "原始模型输出"


def test_step_construction_with_empty_reflect_and_plan() -> None:
    """验证 Step 支持空的 reflect 与 plan 字典。"""

    step = Step(
        thought="继续执行",
        reflect={},
        plan={},
        tool_call={"tool": "submit_answer", "args": {}},
        tool_output="",
        raw_content="{}",
    )

    assert step.reflect == {}
    assert step.plan == {}


def test_loop_result_defaults() -> None:
    """验证 LoopResult 的默认字段值。"""

    result = LoopResult()

    assert result.result is None
    assert result.steps == []
    assert result.steps_used == 0
    assert result.token_usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert result.stop_reason == "finished"


def test_loop_result_with_budget_exceeded_stop_reason() -> None:
    """验证 LoopResult 可设置预算超限停止原因。"""

    result = LoopResult(stop_reason="budget_exceeded")

    assert result.stop_reason == "budget_exceeded"


def test_loop_result_with_custom_result() -> None:
    """验证 LoopResult 可携带自定义结果字典。"""

    result = LoopResult(result={"answer": "A", "score": 1})

    assert result.result == {"answer": "A", "score": 1}


def test_loop_hook_methods_are_noops() -> None:
    """验证 LoopHook 基类方法均为空操作并返回 None。"""

    hook = LoopHook()
    step = Step(
        thought="思考",
        reflect={},
        plan={},
        tool_call={},
        tool_output="输出",
        raw_content="原始内容",
    )
    result = LoopResult()
    messages = [{"role": "user", "content": "hello"}]

    assert hook.before_step(1, messages) is None
    assert hook.after_tool(1, step) is None
    assert hook.after_step(1, messages) is None
    assert hook.on_finish(result) is None


def test_composite_hook_calls_all_hooks_in_order() -> None:
    """验证 CompositeHook 会按顺序调用全部钩子。"""

    calls: list[tuple[str, str, int]] = []

    class RecordingHook(LoopHook):
        def __init__(self, name: str) -> None:
            self.name = name

        def before_step(self, iteration: int, messages: list[dict]) -> None:
            calls.append((self.name, "before_step", iteration))

        def after_tool(self, iteration: int, step: Step) -> None:
            calls.append((self.name, "after_tool", iteration))

        def after_step(self, iteration: int, messages: list[dict]) -> None:
            calls.append((self.name, "after_step", iteration))

        def on_finish(self, result: LoopResult) -> None:
            calls.append((self.name, "on_finish", result.steps_used))

    composite = CompositeHook([RecordingHook("first"), RecordingHook("second")])
    step = Step(
        thought="思考",
        reflect={},
        plan={},
        tool_call={},
        tool_output="输出",
        raw_content="原始内容",
    )
    result = LoopResult(steps_used=7)
    messages = [{"role": "assistant", "content": "test"}]

    composite.before_step(3, messages)
    composite.after_tool(3, step)
    composite.after_step(3, messages)
    composite.on_finish(result)

    assert calls == [
        ("first", "before_step", 3),
        ("second", "before_step", 3),
        ("first", "after_tool", 3),
        ("second", "after_tool", 3),
        ("first", "after_step", 3),
        ("second", "after_step", 3),
        ("first", "on_finish", 7),
        ("second", "on_finish", 7),
    ]


def test_composite_hook_isolates_hook_exceptions() -> None:
    """验证 CompositeHook 中单个钩子异常不会阻断后续钩子。"""

    calls: list[str] = []

    class FailingHook(LoopHook):
        def before_step(self, iteration: int, messages: list[dict]) -> None:
            calls.append("failing")
            raise RuntimeError("boom")

    class RecordingHook(LoopHook):
        def before_step(self, iteration: int, messages: list[dict]) -> None:
            calls.append("recording")

    composite = CompositeHook([FailingHook(), RecordingHook()])

    composite.before_step(1, [])

    assert calls == ["failing", "recording"]
