from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.harness.inference import InferenceResult, TracePlugin, _run_single_question
from core.harness.log import HarnessLog
from core.harness.question_gen import GeneratedQuestion
from core.loop import LoopResult, Step


def _make_step(tool_name: str, node_id: str = "") -> Step:
    args = {"node_id": node_id} if node_id else {}
    return Step(
        thought="t",
        reflect={},
        plan={},
        tool_call={"tool": tool_name, "args": args},
        tool_output="out",
        raw_content="raw",
    )


def test_full_construction() -> None:
    result = InferenceResult(
        run_id="run-001",
        accuracy=0.875,
        total=8,
        correct=7,
        per_task_type={
            "classification": {"accuracy": 1.0, "total": 4},
            "reasoning": {"accuracy": 0.75, "total": 4},
        },
        steps_mean=3.5,
        token_usage={"prompt_tokens": 120, "completion_tokens": 45},
        stop_reason_counts={"finished": 6, "max_steps": 2},
    )

    assert result.run_id == "run-001"
    assert result.accuracy == 0.875
    assert result.total == 8
    assert result.correct == 7
    assert result.per_task_type == {
        "classification": {"accuracy": 1.0, "total": 4},
        "reasoning": {"accuracy": 0.75, "total": 4},
    }
    assert result.steps_mean == 3.5
    assert result.token_usage == {"prompt_tokens": 120, "completion_tokens": 45}
    assert result.stop_reason_counts == {"finished": 6, "max_steps": 2}


def test_frozen_immutability() -> None:
    result = InferenceResult(
        run_id="run-002",
        accuracy=1.0,
        total=5,
        correct=5,
        per_task_type={"qa": {"accuracy": 1.0, "total": 5}},
        steps_mean=2.0,
        token_usage={"prompt_tokens": 80, "completion_tokens": 20},
        stop_reason_counts={"finished": 5},
    )

    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.run_id = "run-003"


class TestTracePlugin(unittest.TestCase):
    def test_after_tool_writes_trace(self) -> None:
        log = HarnessLog(":memory:", run_id="run-001", git_sha="test-sha")
        log.create_table(
            "traces",
            {
                "video_id": "TEXT",
                "question_id": "TEXT",
                "step": "INTEGER",
                "tool_name": "TEXT",
                "tool_args": "TEXT",
                "tool_output": "TEXT",
                "thought": "TEXT",
            },
        )
        plugin = TracePlugin(log, "vid1", "q1")
        step = _make_step("view_node", "seg_L1_001")

        plugin.after_tool(iteration=0, step=step)

        rows = log.query("SELECT * FROM traces")
        assert len(rows) == 1
        assert rows[0]["video_id"] == "vid1"
        assert rows[0]["tool_name"] == "view_node"

    def test_on_finish_writes_validation_flags(self) -> None:
        log = HarnessLog(":memory:", run_id="run-001", git_sha="test-sha")
        log.create_table(
            "validation_flags",
            {
                "video_id": "TEXT",
                "question_id": "TEXT",
                "l1_visits": "INTEGER",
                "l2_visits": "INTEGER",
                "l3_visits": "INTEGER",
            },
        )
        plugin = TracePlugin(log, "vid1", "q1")
        steps = [
            _make_step("view_node", "seg_L1_001"),
            _make_step("view_node", "seg_L2_002"),
            _make_step("view_node", "seg_L3_003"),
            _make_step("search_similar"),
        ]
        loop_result = LoopResult(
            result={"answer": "A"},
            stop_reason="finished",
            steps=steps,
            token_usage={},
        )

        plugin.on_finish(result=loop_result)

        rows = log.query("SELECT * FROM validation_flags")
        assert len(rows) == 1
        assert rows[0]["l1_visits"] == 1
        assert rows[0]["l2_visits"] == 1
        assert rows[0]["l3_visits"] == 1


@patch("core.harness.inference.AgentLoop")
@patch("core.harness.inference.LLMClient")
def test_run_single_question_finished(
    mock_llm_client: MagicMock,
    mock_agent_loop: MagicMock,
) -> None:
    mock_llm_client.from_env.return_value = MagicMock()
    mock_loop_instance = mock_agent_loop.return_value
    mock_loop_instance.run.return_value = SimpleNamespace(
        answer="B",
        stop_reason="finished",
        steps=[],
        token_usage={},
    )
    mock_pm = MagicMock()
    mock_pm.build_inference_prompt.return_value = "system"
    mock_pm.format_user_prompt.return_value = "user"
    env = MagicMock()
    env._nodes = {}
    log = HarnessLog(":memory:", run_id="run-001", git_sha="test-sha")
    log.create_table(
        "predictions",
        {
            "video_id": "TEXT",
            "question_id": "TEXT",
            "prediction": "TEXT",
            "answer": "TEXT",
            "correct": "INTEGER",
            "stop_reason": "TEXT",
            "steps": "TEXT",
            "token_usage": "TEXT",
            "steps_count": "INTEGER",
        },
    )
    qa = GeneratedQuestion(
        video_id="v1",
        question_id="q1",
        question="Q?",
        options=["A", "B", "C", "D"],
        answer="B",
        task_type="mc",
    )

    result = _run_single_question(
        qa,
        env,
        vl_client=MagicMock(),
        prompt_mgr=mock_pm,
        skill_registry=MagicMock(),
        log=log,
        max_steps=10,
        skill_mode="auto",
        always_skills_text="",
        task_skill_map={},
        catalog_text="",
        prompts_dir=Path("/tmp"),
    )

    assert result["prediction"] == "B"
    rows = log.query("SELECT * FROM predictions")
    assert len(rows) == 1
    assert rows[0]["prediction"] == "B"


@patch("core.harness.inference.AgentLoop")
@patch("core.harness.inference.LLMClient")
def test_run_single_question_error(
    mock_llm_client: MagicMock,
    mock_agent_loop: MagicMock,
) -> None:
    mock_llm_client.from_env.return_value = MagicMock()
    mock_loop_instance = mock_agent_loop.return_value
    mock_loop_instance.run.side_effect = RuntimeError("fail")
    mock_pm = MagicMock()
    mock_pm.build_inference_prompt.return_value = "system"
    mock_pm.format_user_prompt.return_value = "user"
    env = MagicMock()
    env._nodes = {}
    log = HarnessLog(":memory:", run_id="run-001", git_sha="test-sha")
    log.create_table(
        "predictions",
        {
            "video_id": "TEXT",
            "question_id": "TEXT",
            "prediction": "TEXT",
            "answer": "TEXT",
            "correct": "INTEGER",
            "stop_reason": "TEXT",
            "steps": "TEXT",
            "token_usage": "TEXT",
            "steps_count": "INTEGER",
        },
    )
    qa = GeneratedQuestion(
        video_id="v1",
        question_id="q1",
        question="Q?",
        options=["A", "B", "C", "D"],
        answer="B",
        task_type="mc",
    )

    result = _run_single_question(
        qa,
        env,
        vl_client=MagicMock(),
        prompt_mgr=mock_pm,
        skill_registry=MagicMock(),
        log=log,
        max_steps=10,
        skill_mode="auto",
        always_skills_text="",
        task_skill_map={},
        catalog_text="",
        prompts_dir=Path("/tmp"),
    )

    assert result["stop_reason"] == "error"
    rows = log.query("SELECT * FROM predictions")
    assert len(rows) == 1
    assert rows[0]["stop_reason"] == "error"
