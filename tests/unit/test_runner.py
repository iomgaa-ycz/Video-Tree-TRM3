# tests/unit/test_runner.py
"""Runner 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.harness.config import RunConfig
from core.harness.runner import Runner


@pytest.fixture()
def workspace_env(tmp_path: Path) -> dict[str, Path]:
    """构建最小可用的 store + workspace 结构。"""
    store_dir = tmp_path / "store"
    (store_dir / "videos").mkdir(parents=True)
    (store_dir / "questions" / "benchmarks" / "Video-MME").mkdir(parents=True)
    (store_dir / "skills" / "v1").mkdir(parents=True)
    (store_dir / "prompts" / "v1").mkdir(parents=True)

    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    (ws_dir / "analyses").mkdir()
    (ws_dir / "runs").mkdir()

    import os

    store_rel = os.path.relpath(store_dir.resolve(), ws_dir.resolve())
    manifest = {
        "name": "ws",
        "created_at": "2026-01-01T00:00:00+00:00",
        "store": store_rel,
        "current": {
            "videos": "videos",
            "questions": "questions/benchmarks/Video-MME",
            "skills": "skills/v1",
            "prompts": "prompts/v1",
        },
        "history": [],
    }
    (ws_dir / "manifest.json").write_text(json.dumps(manifest))
    return {"store_dir": store_dir, "workspace_dir": ws_dir}


def _make_config(workspace_env: dict[str, Path], **overrides) -> RunConfig:
    """用 workspace_env 构造 RunConfig。"""
    defaults = {
        "workspace_dir": workspace_env["workspace_dir"],
        "store_dir": workspace_env["store_dir"],
        "mode": "infer",
        "run_id": "",
        "concurrency": 2,
        "max_steps": 5,
        "skill_mode": "auto",
        "n_samples": 0,
        "questions": "benchmarks/Video-MME",
        "skills_version": "v1",
        "prompts_version": "v1",
        "epochs": 1,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestRunnerInit:
    """Runner 初始化测试。"""

    def test_init_resolves_paths(self, workspace_env: dict[str, Path]) -> None:
        """Runner 初始化应成功解析 workspace paths。"""
        config = _make_config(workspace_env)
        runner = Runner(config)
        assert runner._paths.workspace_dir == workspace_env["workspace_dir"].resolve()

    def test_init_missing_workspace_raises(self, tmp_path: Path) -> None:
        """workspace_dir 不存在应报错。"""
        config = RunConfig(
            workspace_dir=tmp_path / "nonexistent",
            store_dir=tmp_path / "store",
            mode="infer",
            run_id="",
            concurrency=2,
            max_steps=5,
            skill_mode="auto",
            n_samples=0,
            questions="benchmarks/Video-MME",
            skills_version="v1",
            prompts_version="v1",
            epochs=1,
        )
        with pytest.raises(FileNotFoundError):
            Runner(config)


class TestRunnerInfer:
    """Runner.infer() 测试。"""

    def test_infer_calls_run_inference(self, workspace_env: dict[str, Path]) -> None:
        """infer() 应调用 run_inference 并返回其结果。"""
        config = _make_config(workspace_env, n_samples=0)
        runner = Runner(config)

        from core.harness.inference import InferenceResult

        mock_result = InferenceResult(
            run_id="test_run",
            accuracy=0.5,
            total=2,
            correct=1,
            per_task_type={},
            steps_mean=3.0,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            stop_reason_counts={"submit": 2},
        )

        with patch(
            "core.harness.runner.run_inference", return_value=mock_result
        ) as mock_fn:
            with patch("core.harness.runner.load_benchmark", return_value=[]):
                result = runner.infer()

        assert result is mock_result
        mock_fn.assert_called_once()

    def test_infer_with_n_samples(self, workspace_env: dict[str, Path]) -> None:
        """n_samples > 0 时应截取前 N 道题。"""
        config = _make_config(workspace_env, n_samples=2)
        runner = Runner(config)

        from core.harness.question_gen import GeneratedQuestion

        fake_questions = [
            GeneratedQuestion(
                question_id=f"q{i}",
                video_id=f"v{i}",
                task_type="perception",
                question="?",
                options=["A", "B"],
                answer="A",
            )
            for i in range(5)
        ]

        from core.harness.inference import InferenceResult

        mock_result = InferenceResult(
            run_id="r",
            accuracy=1.0,
            total=2,
            correct=2,
            per_task_type={},
            steps_mean=1.0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0},
            stop_reason_counts={},
        )

        with patch(
            "core.harness.runner.run_inference", return_value=mock_result
        ) as mock_fn:
            with patch(
                "core.harness.runner.load_benchmark", return_value=fake_questions
            ):
                runner.infer()

        called_questions = mock_fn.call_args[1]["questions"]
        assert len(called_questions) == 2


class TestRunnerAutoWorkspace:
    """Runner 自动创建 workspace 测试。"""

    def test_auto_creates_workspace(self, workspace_env: dict[str, Path]) -> None:
        """workspace 不存在时自动创建 manifest.json。"""
        new_ws = workspace_env["workspace_dir"].parent / "auto_ws"
        config = _make_config(workspace_env, workspace_dir=new_ws)
        runner = Runner(config)
        assert (new_ws / "manifest.json").exists()
        assert runner._paths.workspace_dir == new_ws.resolve()


class TestRunnerDiagnose:
    """Runner.diagnose() 测试。"""

    def test_diagnose_calls_run_diagnosis(self, workspace_env: dict[str, Path]) -> None:
        """diagnose() 应将路径、run_id 与筛选器透传给 run_diagnosis。"""
        config = _make_config(workspace_env, mode="diagnose", run_id="run-123")
        runner = Runner(config)

        from core.harness.diagnose import DiagnosisResult

        mock_result = DiagnosisResult(run_id="run-123")

        with patch(
            "core.harness.diagnose.run_diagnosis", return_value=mock_result
        ) as mock_fn:
            result = runner.diagnose(task_types=["temporal_reasoning"])

        assert result is mock_result
        assert mock_fn.call_args.kwargs["run_id"] == "run-123"
        assert mock_fn.call_args.kwargs["workspace_dir"] == config.workspace_dir
        assert mock_fn.call_args.kwargs["skills_dir"] == runner._paths.skills_dir
        assert "prompts_dir" not in mock_fn.call_args.kwargs
        assert mock_fn.call_args.kwargs["task_types"] == ["temporal_reasoning"]

    def test_diagnose_requires_run_id(self, workspace_env: dict[str, Path]) -> None:
        """缺少 run_id 时应抛出 ValueError。"""
        config = _make_config(workspace_env, mode="diagnose", run_id="")
        runner = Runner(config)

        with pytest.raises(ValueError, match="run_id"):
            runner.diagnose()
