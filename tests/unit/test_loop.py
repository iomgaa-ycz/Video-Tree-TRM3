"""core.loop 模块的单元测试。"""

from __future__ import annotations

import pluggy

from core.loop import AgentLoopSpec, LoopResult, Step, hookimpl


class TestHookSpec:
    """hookspec 注册与调用。"""

    def test_plugin_manager_registers_hookspecs(self) -> None:
        """验证 AgentLoopSpec 可被 PluginManager 正确注册。"""
        pm = pluggy.PluginManager("agent_loop")
        pm.add_hookspecs(AgentLoopSpec)
        assert pm.hook.before_step is not None
        assert pm.hook.after_tool is not None
        assert pm.hook.after_step is not None
        assert pm.hook.on_finish is not None

    def test_hookimpl_observation_called(self) -> None:
        """验证观察型 hookimpl（返回 None）被正确调用。"""
        calls: list[str] = []

        class ObserverPlugin:
            @hookimpl
            def before_step(self, iteration: int, messages: list[dict]) -> None:
                calls.append(f"before_{iteration}")

            @hookimpl
            def on_finish(self, result: LoopResult) -> None:
                calls.append("finish")

        pm = pluggy.PluginManager("agent_loop")
        pm.add_hookspecs(AgentLoopSpec)
        pm.register(ObserverPlugin())

        pm.hook.before_step(iteration=0, messages=[])
        pm.hook.on_finish(result=LoopResult())

        assert calls == ["before_0", "finish"]

    def test_hookimpl_transform_returns_value(self) -> None:
        """验证变换型 hookimpl 的返回值被收集。"""

        class TransformPlugin:
            @hookimpl
            def after_tool(self, iteration: int, step: Step) -> str:
                return "hint text"

        pm = pluggy.PluginManager("agent_loop")
        pm.add_hookspecs(AgentLoopSpec)
        pm.register(TransformPlugin())

        step = Step(
            thought="t",
            reflect={},
            plan={},
            tool_call={"tool": "x", "args": {}},
            tool_output="out",
            raw_content="{}",
        )
        results = pm.hook.after_tool(iteration=0, step=step)
        assert results == ["hint text"]

    def test_multiple_plugins_all_called(self) -> None:
        """验证多个插件都被调用，返回值被收集（pluggy LIFO 顺序）。"""
        calls: list[str] = []

        class PluginA:
            @hookimpl
            def after_tool(self, iteration: int, step: Step) -> None:
                calls.append("A")

        class PluginB:
            @hookimpl
            def after_tool(self, iteration: int, step: Step) -> str:
                calls.append("B")
                return "hint_B"

        pm = pluggy.PluginManager("agent_loop")
        pm.add_hookspecs(AgentLoopSpec)
        pm.register(PluginA())
        pm.register(PluginB())

        step = Step(
            thought="t",
            reflect={},
            plan={},
            tool_call={"tool": "x", "args": {}},
            tool_output="out",
            raw_content="{}",
        )
        results = pm.hook.after_tool(iteration=0, step=step)
        assert set(calls) == {"A", "B"}
        assert "hint_B" in results


class TestStepDataStructure:
    """Step 数据结构测试。"""

    def test_step_construction_with_all_fields(self) -> None:
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

    def test_step_construction_with_empty_reflect_and_plan(self) -> None:
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


class TestLoopResultDataStructure:
    """LoopResult 数据结构测试。"""

    def test_loop_result_defaults(self) -> None:
        """验证 LoopResult 的默认字段值。"""
        result = LoopResult()
        assert result.result is None
        assert result.steps == []
        assert result.steps_used == 0
        assert result.token_usage == {"prompt_tokens": 0, "completion_tokens": 0}
        assert result.stop_reason == "finished"

    def test_loop_result_with_budget_exceeded_stop_reason(self) -> None:
        """验证 LoopResult 可设置预算超限停止原因。"""
        result = LoopResult(stop_reason="budget_exceeded")
        assert result.stop_reason == "budget_exceeded"

    def test_loop_result_with_custom_result(self) -> None:
        """验证 LoopResult 可携带自定义结果字典。"""
        result = LoopResult(result={"answer": "A", "score": 1})
        assert result.result == {"answer": "A", "score": 1}
