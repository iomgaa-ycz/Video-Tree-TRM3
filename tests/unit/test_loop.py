"""core.loop 模块的单元测试。"""

from __future__ import annotations

import json
from typing import Any

import pluggy
import pytest

from core.loop import AgentLoop, AgentLoopSpec, LoopResult, Step, hookimpl


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


# ── Mock 辅助 ──


class _MockMessage:
    """模拟 OpenAI Message 对象。"""

    def __init__(self, content: str, reasoning_content: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _MockChoice:
    def __init__(self, message: _MockMessage) -> None:
        self.message = message


class _MockUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _MockResponse:
    """模拟 ChatCompletion 响应对象。"""

    def __init__(
        self,
        content: str,
        reasoning_content: str | None = None,
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
    ) -> None:
        self.choices = [_MockChoice(_MockMessage(content, reasoning_content))]
        self.usage = _MockUsage(prompt_tokens, completion_tokens)


class _MockLLMClient:
    """按预设顺序返回响应的 LLMClient 替身。"""

    def __init__(self, responses: list[_MockResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> _MockResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


def _make_json(
    tool: str = "view_node",
    args: dict | None = None,
    reflect: dict | None = None,
    plan: dict | None = None,
) -> str:
    """构造标准 Agent JSON 输出。"""
    return json.dumps(
        {
            "reflect": reflect or {},
            "plan": plan or {"next": "explore"},
            "action": {
                "tool": tool,
                "args": args or {"node_id": "L1_001", "question": "q"},
            },
        },
        ensure_ascii=False,
    )


# ── _parse_response 测试 ──


class TestParseResponse:
    """_parse_response 子方法。"""

    def _make_loop(self) -> AgentLoop:
        return AgentLoop(_MockLLMClient([]), max_steps=5, max_retries=3)

    def test_valid_json_with_all_fields(self) -> None:
        """完整合法 JSON → 正确解析五元组。"""
        loop = self._make_loop()
        content = _make_json(
            tool="search_similar",
            args={"query": "cat", "question": "q"},
            reflect={"status": "exploring"},
            plan={"next": "search"},
        )
        resp = _MockResponse(content, reasoning_content="先搜索相关节点")
        result = loop._parse_response(resp)

        assert result is not None
        thought, reflect, plan, raw_content, action = result
        assert thought == "先搜索相关节点"
        assert reflect == {"status": "exploring"}
        assert plan == {"next": "search"}
        assert action == {
            "tool": "search_similar",
            "args": {"query": "cat", "question": "q"},
        }
        assert raw_content == content

    def test_json_repair_fixes_trailing_comma(self) -> None:
        """json_repair 修复尾逗号后正确解析。"""
        loop = self._make_loop()
        broken = (
            '{"reflect": {}, "plan": {}, '
            '"action": {"tool": "view_node", "args": {"node_id": "L1_001",}},}'
        )
        resp = _MockResponse(broken)
        result = loop._parse_response(resp)

        assert result is not None
        _, _, _, _, action = result
        assert action["tool"] == "view_node"

    def test_missing_action_returns_none(self) -> None:
        """缺少 action 字段 → 返回 None。"""
        loop = self._make_loop()
        resp = _MockResponse('{"reflect": {}, "plan": {}}')
        assert loop._parse_response(resp) is None

    def test_non_json_returns_none(self) -> None:
        """完全非 JSON → 返回 None。"""
        loop = self._make_loop()
        resp = _MockResponse("I think we should look at node L1_001")
        assert loop._parse_response(resp) is None

    def test_action_missing_tool_returns_none(self) -> None:
        """action 缺少 tool 字段 → 返回 None。"""
        loop = self._make_loop()
        resp = _MockResponse('{"reflect": {}, "plan": {}, "action": {"args": {}}}')
        assert loop._parse_response(resp) is None

    def test_no_reasoning_content_defaults_empty(self) -> None:
        """无 reasoning_content 属性 → thought 为空字符串。"""
        loop = self._make_loop()
        resp = _MockResponse(_make_json())
        delattr(resp.choices[0].message, "reasoning_content")
        result = loop._parse_response(resp)

        assert result is not None
        assert result[0] == ""

    def test_empty_content_returns_none(self) -> None:
        """空 content → 返回 None。"""
        loop = self._make_loop()
        resp = _MockResponse("")
        assert loop._parse_response(resp) is None

    def test_reflect_plan_default_to_empty_dict(self) -> None:
        """JSON 中省略 reflect/plan → 默认空字典。"""
        loop = self._make_loop()
        resp = _MockResponse('{"action": {"tool": "view_node", "args": {}}}')
        result = loop._parse_response(resp)

        assert result is not None
        _, reflect, plan, _, _ = result
        assert reflect == {}
        assert plan == {}


# ── _execute_tool 测试 ──


class TestExecuteTool:
    """_execute_tool 子方法。"""

    def _make_loop(self) -> AgentLoop:
        return AgentLoop(_MockLLMClient([]), max_steps=5, max_retries=3)

    def test_valid_tool_call_returns_output_and_true(self) -> None:
        """正常工具调用 → (output, True)。"""
        loop = self._make_loop()
        output, is_valid = loop._execute_tool(
            lambda n, a: f"结果: {n}", "view_node", {"node_id": "L1_001"}
        )
        assert output == "结果: view_node"
        assert is_valid is True

    def test_value_error_returns_error_and_false(self) -> None:
        """ValueError → (error_msg, False)。"""
        loop = self._make_loop()

        def bad_fn(name: str, args: dict) -> str:
            raise ValueError("node_id 不存在")

        output, is_valid = loop._execute_tool(bad_fn, "view_node", {"node_id": "bad"})
        assert "node_id 不存在" in output
        assert is_valid is False

    def test_non_value_error_propagates(self) -> None:
        """非 ValueError 异常不捕获，向上传播。"""
        loop = self._make_loop()

        def crash_fn(name: str, args: dict) -> str:
            raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            loop._execute_tool(crash_fn, "view_node", {})


# ── _build_feedback 测试 ──


class TestBuildFeedback:
    """_build_feedback 子方法。"""

    def _make_loop(self) -> AgentLoop:
        return AgentLoop(_MockLLMClient([]), max_steps=5, max_retries=3)

    def test_basic_feedback_message(self) -> None:
        """基础反馈消息格式正确。"""
        loop = self._make_loop()
        msg = loop._build_feedback("view_node", "节点内容", [])

        assert msg["role"] == "user"
        assert "[工具执行结果: view_node]" in msg["content"]
        assert "节点内容" in msg["content"]

    def test_hints_appended(self) -> None:
        """hook 返回的 hint 被拼接到消息末尾。"""
        loop = self._make_loop()
        msg = loop._build_feedback("view_node", "节点内容", ["⏱ 已用 3/15 步"])

        assert "⏱ 已用 3/15 步" in msg["content"]

    def test_no_hints_no_extra_content(self) -> None:
        """无 hint 时消息只包含工具结果。"""
        loop = self._make_loop()
        msg = loop._build_feedback("search_similar", "搜索结果", [])

        assert "[工具执行结果: search_similar]" in msg["content"]
        assert "⏱" not in msg["content"]

    def test_multiple_hints_all_appended(self) -> None:
        """多个 hint 全部拼接。"""
        loop = self._make_loop()
        msg = loop._build_feedback("view_node", "out", ["hint_A", "hint_B"])

        assert "hint_A" in msg["content"]
        assert "hint_B" in msg["content"]


# ── _call_llm 测试 ──


class TestCallLlm:
    """_call_llm 子方法。"""

    def test_accumulates_token_usage(self) -> None:
        """token 使用量正确累加。"""
        resp = _MockResponse(_make_json(), prompt_tokens=100, completion_tokens=50)
        client = _MockLLMClient([resp])
        loop = AgentLoop(client, max_steps=5)
        token_usage = {"prompt_tokens": 10, "completion_tokens": 5}

        loop._call_llm([{"role": "user", "content": "hi"}], token_usage)

        assert token_usage == {"prompt_tokens": 110, "completion_tokens": 55}

    def test_returns_response_object(self) -> None:
        """返回原始响应对象。"""
        resp = _MockResponse(_make_json())
        client = _MockLLMClient([resp])
        loop = AgentLoop(client, max_steps=5)

        result = loop._call_llm([], {"prompt_tokens": 0, "completion_tokens": 0})
        assert result is resp

    def test_none_usage_no_error(self) -> None:
        """response.usage 为 None 时不报错。"""
        resp = _MockResponse(_make_json())
        resp.usage = None
        client = _MockLLMClient([resp])
        loop = AgentLoop(client, max_steps=5)
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        loop._call_llm([], token_usage)
        assert token_usage == {"prompt_tokens": 0, "completion_tokens": 0}


# ── AgentLoop.run() 集成测试 ──


class TestAgentLoopRun:
    """AgentLoop.run() 集成测试。"""

    def test_finish_on_submit_answer(self) -> None:
        """调用 submit_answer → stop_reason="finished"。"""
        content = _make_json(
            tool="submit_answer",
            args={"answer": "B", "evidence": "e", "reasoning": "r"},
        )
        client = _MockLLMClient([_MockResponse(content)])
        loop = AgentLoop(client, max_steps=15)

        result = loop.run(
            system_prompt="system",
            user_prompt="question",
            tool_fn=lambda name, args: f"[ok] {name}",
        )

        assert result.stop_reason == "finished"
        assert result.result == {"answer": "B", "evidence": "e", "reasoning": "r"}
        assert result.steps_used == 1
        assert len(result.steps) == 1
        assert result.steps[0].tool_call["tool"] == "submit_answer"

    def test_tool_then_submit(self) -> None:
        """先调用工具，再提交答案 → 2 步完成。"""
        resp1 = _MockResponse(_make_json())
        resp2 = _MockResponse(_make_json(tool="submit_answer", args={"answer": "A"}))
        client = _MockLLMClient([resp1, resp2])
        loop = AgentLoop(client, max_steps=15)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: f"output_{name}",
        )

        assert result.stop_reason == "finished"
        assert result.steps_used == 2
        assert result.steps[0].tool_output == "output_view_node"
        assert result.steps[1].tool_call["tool"] == "submit_answer"

    def test_budget_exceeded(self) -> None:
        """步数耗尽 → stop_reason="budget_exceeded"。"""
        responses = [_MockResponse(_make_json()) for _ in range(5)]
        client = _MockLLMClient(responses)
        loop = AgentLoop(client, max_steps=3)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "output",
        )

        assert result.stop_reason == "budget_exceeded"
        assert result.steps_used == 3

    def test_parse_error_after_max_retries(self) -> None:
        """连续 JSON 解析失败 → stop_reason="parse_error"。"""
        responses = [_MockResponse("not json") for _ in range(5)]
        client = _MockLLMClient(responses)
        loop = AgentLoop(client, max_steps=15, max_retries=3)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "output",
        )

        assert result.stop_reason == "parse_error"
        assert result.steps_used == 0

    def test_api_error_returns_error_result(self) -> None:
        """LLM API 异常 → stop_reason="error"。"""

        class ErrorClient:
            def chat(self, messages, **kw):
                raise ConnectionError("network down")

        loop = AgentLoop(ErrorClient(), max_steps=15)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "output",
        )

        assert result.stop_reason == "error"

    def test_invalid_tool_no_step_count(self) -> None:
        """ValueError 工具调用不计步数，模型可重试。"""
        call_count = 0

        def tool_fn(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("bad node_id")
            return "ok"

        resp1 = _MockResponse(_make_json())
        resp2 = _MockResponse(_make_json(tool="submit_answer", args={"answer": "A"}))
        client = _MockLLMClient([resp1, resp2])
        loop = AgentLoop(client, max_steps=15)

        result = loop.run(system_prompt="s", user_prompt="q", tool_fn=tool_fn)

        assert result.stop_reason == "finished"
        assert result.steps_used == 1
        assert call_count == 2

    def test_hooks_invoked(self) -> None:
        """验证 hook 在循环各阶段被调用。"""
        events: list[str] = []

        class TrackingPlugin:
            @hookimpl
            def before_step(self, iteration: int, messages: list[dict]) -> None:
                events.append(f"before_{iteration}")

            @hookimpl
            def after_tool(self, iteration: int, step: Step) -> None:
                events.append(f"after_tool_{iteration}")

            @hookimpl
            def after_step(self, iteration: int, messages: list[dict]) -> None:
                events.append(f"after_step_{iteration}")

            @hookimpl
            def on_finish(self, result: LoopResult) -> None:
                events.append("finish")

        content = _make_json(tool="submit_answer", args={"answer": "A"})
        client = _MockLLMClient([_MockResponse(content)])
        loop = AgentLoop(client, max_steps=15)

        loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "ok",
            plugins=[TrackingPlugin()],
        )

        assert "before_0" in events
        assert "after_tool_0" in events
        assert "after_step_0" in events
        assert "finish" in events

    def test_after_tool_hint_injected_into_feedback(self) -> None:
        """after_tool 返回的 hint 被拼入反馈消息。"""
        captured_messages: list[list[dict]] = []

        class HintPlugin:
            @hookimpl
            def after_tool(self, iteration: int, step: Step) -> str:
                return "⏱ 已用 1/15 步"

        class CapturePlugin:
            @hookimpl
            def after_step(self, iteration: int, messages: list[dict]) -> None:
                captured_messages.append([m.copy() for m in messages])

        resp1 = _MockResponse(_make_json())
        resp2 = _MockResponse(_make_json(tool="submit_answer", args={"answer": "A"}))
        client = _MockLLMClient([resp1, resp2])
        loop = AgentLoop(client, max_steps=15)

        loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "output",
            plugins=[HintPlugin(), CapturePlugin()],
        )

        first_round = captured_messages[0]
        feedback = first_round[-1]
        assert "⏱ 已用 1/15 步" in feedback["content"]

    def test_token_usage_accumulated(self) -> None:
        """token 使用量跨多轮累加。"""
        resp1 = _MockResponse(_make_json(), prompt_tokens=100, completion_tokens=50)
        resp2 = _MockResponse(
            _make_json(tool="submit_answer", args={"answer": "A"}),
            prompt_tokens=200,
            completion_tokens=80,
        )
        client = _MockLLMClient([resp1, resp2])
        loop = AgentLoop(client, max_steps=15)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "ok",
        )

        assert result.token_usage == {"prompt_tokens": 300, "completion_tokens": 130}

    def test_messages_structure(self) -> None:
        """验证消息历史格式：system → user → assistant → user(feedback)。"""
        captured: list[list[dict]] = []

        class CapturePlugin:
            @hookimpl
            def after_step(self, iteration: int, messages: list[dict]) -> None:
                captured.append([m.copy() for m in messages])

        content = _make_json(tool="submit_answer", args={"answer": "A"})
        client = _MockLLMClient([_MockResponse(content)])
        loop = AgentLoop(client, max_steps=15)

        loop.run(
            system_prompt="sys",
            user_prompt="usr",
            tool_fn=lambda name, args: "ok",
            plugins=[CapturePlugin()],
        )

        msgs = captured[0]
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "usr"}
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert "[工具执行结果: submit_answer]" in msgs[3]["content"]

    def test_step_records_thought_from_reasoning_content(self) -> None:
        """Step.thought 来自 reasoning_content。"""
        content = _make_json(tool="submit_answer", args={"answer": "C"})
        client = _MockLLMClient(
            [_MockResponse(content, reasoning_content="深度推理过程")]
        )
        loop = AgentLoop(client, max_steps=15)

        result = loop.run(
            system_prompt="s",
            user_prompt="q",
            tool_fn=lambda name, args: "ok",
        )

        assert result.steps[0].thought == "深度推理过程"
