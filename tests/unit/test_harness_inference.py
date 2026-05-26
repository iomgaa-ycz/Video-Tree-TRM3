from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.harness.inference import InferenceResult


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
