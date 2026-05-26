from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.harness.inference import (
    InferenceResult,
    TracePlugin,
    _run_single_question,
    run_inference,
)
from core.harness.log import HarnessLog
from core.harness.question_gen import GeneratedQuestion
from core.loop import LoopResult, Step
from core.workspace import init_workspace


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


_PREDICTIONS_SCHEMA = {
    "video_id": "TEXT",
    "question_id": "TEXT",
    "task_type": "TEXT",
    "prediction": "TEXT",
    "answer": "TEXT",
    "evidence": "TEXT",
    "reasoning": "TEXT",
    "steps_used": "INTEGER",
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
    "stop_reason": "TEXT",
    "steps_json": "JSON",
}

_TRACES_SCHEMA = {
    "video_id": "TEXT",
    "question_id": "TEXT",
    "step": "INTEGER",
    "tool_name": "TEXT",
    "tool_args": "JSON",
    "tool_output": "TEXT",
    "thought": "TEXT",
}

_VALIDATION_FLAGS_SCHEMA = {
    "video_id": "TEXT",
    "question_id": "TEXT",
    "has_l3_visit": "INTEGER",
    "l1_count": "INTEGER",
    "l2_count": "INTEGER",
    "l3_count": "INTEGER",
}


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
    assert result.steps_mean == 3.5


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


class TestTracePlugin:
    def test_after_tool_writes_trace(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        log = HarnessLog(db_path, "run-trace", git_sha="abc")
        log.create_table("traces", _TRACES_SCHEMA)

        plugin = TracePlugin(log, "vid-1", "q-1")
        step = _make_step("view_node", "vid-1_L1_000")
        plugin.after_tool(iteration=0, step=step)

        rows = log.query("SELECT * FROM traces WHERE question_id = ?", ("q-1",))
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "view_node"
        assert rows[0]["video_id"] == "vid-1"
        assert rows[0]["thought"] == "t"
        log.close()

    def test_on_finish_writes_validation_flags(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        log = HarnessLog(db_path, "run-flags", git_sha="abc")
        log.create_table("validation_flags", _VALIDATION_FLAGS_SCHEMA)

        plugin = TracePlugin(log, "vid-1", "q-1")
        result = LoopResult(
            result={"answer": "A"},
            steps=[
                _make_step("view_node", "vid-1_L1_000"),
                _make_step("view_node", "vid-1_L1_000_L2_001"),
                _make_step("view_node", "vid-1_L1_000_L2_001_L3_002"),
                _make_step("search_similar"),
            ],
            steps_used=4,
            stop_reason="finished",
        )
        plugin.on_finish(result=result)

        rows = log.query(
            "SELECT * FROM validation_flags WHERE question_id = ?", ("q-1",)
        )
        assert len(rows) == 1
        assert rows[0]["has_l3_visit"] == 1
        assert rows[0]["l1_count"] == 1
        assert rows[0]["l2_count"] == 1
        assert rows[0]["l3_count"] == 1
        log.close()


@patch("core.harness.inference.AgentLoop")
@patch("core.harness.inference.LLMClient")
def test_run_single_question_finished(
    mock_llm_client: MagicMock,
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """正常完成时，prediction 写入 log 且返回正确结构。"""
    mock_llm_client.from_env.return_value = MagicMock()
    mock_loop_instance = mock_agent_loop.return_value
    mock_loop_instance.run.return_value = LoopResult(
        result={"answer": "B", "evidence": "saw 2 cats", "reasoning": "counted"},
        steps=[_make_step("view_node", "v1_L1_000"), _make_step("submit_answer")],
        steps_used=2,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        stop_reason="finished",
    )

    db_path = str(tmp_path / "test.db")
    log = HarnessLog(db_path, "run-sq", git_sha="abc")
    log.create_table("predictions", _PREDICTIONS_SCHEMA)
    log.create_table("traces", _TRACES_SCHEMA)
    log.create_table("validation_flags", _VALIDATION_FLAGS_SCHEMA)

    mock_pm = MagicMock()
    mock_pm.build_inference_prompt.return_value = "system"
    mock_pm.format_user_prompt.return_value = "user"
    env = MagicMock()
    env._nodes = {"v1_L1_000": {"level": 1, "children_ids": []}}

    qa = GeneratedQuestion(
        question_id="q-1",
        video_id="v1",
        task_type="Counting",
        question="How many?",
        options=["A. 1", "B. 2"],
        answer="B",
    )

    result = _run_single_question(
        qa=qa,
        env=env,
        vl_client=MagicMock(),
        prompt_mgr=mock_pm,
        skill_registry=MagicMock(),
        log=log,
        max_steps=15,
        skill_mode="auto",
        always_skills_text="",
        task_skill_map={},
        catalog_text="",
        prompts_dir=tmp_path,
    )

    assert result["prediction"] == "B"
    assert result["stop_reason"] == "finished"
    rows = log.query("SELECT * FROM predictions WHERE question_id = ?", ("q-1",))
    assert len(rows) == 1
    assert rows[0]["prediction"] == "B"
    log.close()


@patch("core.harness.inference.AgentLoop")
@patch("core.harness.inference.LLMClient")
def test_run_single_question_error(
    mock_llm_client: MagicMock,
    mock_agent_loop: MagicMock,
    tmp_path: Path,
) -> None:
    """异常时写入 stop_reason=error 的兜底记录。"""
    mock_llm_client.from_env.return_value = MagicMock()
    mock_agent_loop.return_value.run.side_effect = RuntimeError("LLM down")

    db_path = str(tmp_path / "err.db")
    log = HarnessLog(db_path, "run-err", git_sha="abc")
    log.create_table("predictions", _PREDICTIONS_SCHEMA)
    log.create_table("traces", _TRACES_SCHEMA)
    log.create_table("validation_flags", _VALIDATION_FLAGS_SCHEMA)

    mock_pm = MagicMock()
    mock_pm.build_inference_prompt.return_value = "sys"
    mock_pm.format_user_prompt.return_value = "usr"
    env = MagicMock()
    env._nodes = {}

    qa = GeneratedQuestion(
        question_id="q-1",
        video_id="v1",
        task_type="Test",
        question="Q?",
        options=["A. a", "B. b"],
        answer="A",
    )

    result = _run_single_question(
        qa=qa,
        env=env,
        vl_client=MagicMock(),
        prompt_mgr=mock_pm,
        skill_registry=MagicMock(),
        log=log,
        max_steps=15,
        skill_mode="none",
        always_skills_text="",
        task_skill_map={},
        catalog_text="",
        prompts_dir=tmp_path,
    )

    assert result["stop_reason"] == "error"
    rows = log.query("SELECT * FROM predictions WHERE question_id = ?", ("q-1",))
    assert len(rows) == 1
    assert rows[0]["stop_reason"] == "error"
    log.close()


# ---------------------------------------------------------------------------
# run_inference 测试
# ---------------------------------------------------------------------------


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """创建最小 store + workspace 用于测试。"""
    store = tmp_path / "store"
    store.mkdir()
    (store / "videos").mkdir()
    vid_dir = store / "videos" / "vid-1"
    vid_dir.mkdir()
    tree_data = {
        "video_id": "vid-1",
        "videoID": "vid-1",
        "duration_category": "short",
        "duration_seconds": 60.0,
        "domain": "test",
        "built_at": "2026-01-01",
        "build_stats": {},
        "failures": [],
        "nodes": {
            "vid-1_L1_000": {
                "node_id": "vid-1_L1_000",
                "level": 1,
                "time_range": [0.0, 60.0],
                "parent_id": None,
                "children_ids": [],
                "card": {"scene_summary": "A test scene"},
                "subtitle": "",
            }
        },
    }
    (vid_dir / "tree.json").write_text(json.dumps(tree_data))

    skills_dir = store / "skills" / "v1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "meta.json").write_text(
        '{"version":"v1","source":"manual","created_at":"2026-01-01"}'
    )

    prompts_dir = store / "prompts" / "v1"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "system.md").write_text("You are a test agent.")
    (prompts_dir / "meta.json").write_text(
        '{"version":"v1","source":"manual","created_at":"2026-01-01"}'
    )

    questions_dir = store / "questions" / "benchmarks" / "test"
    questions_dir.mkdir(parents=True)

    ws = tmp_path / "ws"
    init_workspace(ws, store, "benchmarks/test", "v1", "v1")

    return ws, store


@patch("core.harness.inference.TreeEnvironment")
@patch("core.harness.inference.AgentLoop")
@patch("core.harness.inference.LLMClient")
def test_run_inference_basic(
    mock_llm: MagicMock,
    mock_loop_cls: MagicMock,
    mock_env_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """基础编排：单题推理，验证 InferenceResult 结构。"""
    ws, _ = _setup_workspace(tmp_path)

    mock_llm.from_env.return_value = MagicMock()
    mock_loop_cls.return_value.run.return_value = LoopResult(
        result={"answer": "A", "evidence": "saw it", "reasoning": "obvious"},
        steps=[_make_step("submit_answer")],
        steps_used=1,
        token_usage={"prompt_tokens": 50, "completion_tokens": 20},
        stop_reason="finished",
    )
    mock_env_instance = MagicMock()
    mock_env_instance._nodes = {"vid-1_L1_000": {"level": 1, "children_ids": []}}
    mock_env_cls.return_value = mock_env_instance

    questions = [
        GeneratedQuestion(
            question_id="q-1",
            video_id="vid-1",
            task_type="Test",
            question="What?",
            options=["A. Yes", "B. No"],
            answer="A",
        )
    ]

    result = run_inference(
        workspace_dir=ws,
        questions=questions,
        concurrency=1,
        max_steps=15,
        skill_mode="none",
    )

    assert result.run_id is not None
    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0
    assert result.stop_reason_counts == {"finished": 1}


def test_run_inference_empty_questions(tmp_path: Path) -> None:
    """空题目列表应直接返回零结果。"""
    ws, _ = _setup_workspace(tmp_path)

    result = run_inference(
        workspace_dir=ws,
        questions=[],
        concurrency=1,
        max_steps=15,
        skill_mode="none",
    )

    assert result.total == 0
    assert result.accuracy == 0.0
    assert result.correct == 0
