"""训练循环使用的 forward() 输出容器。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    """封装训练循环一次 forward() 推理输出的不可变结果结构。"""

    run_id: str
    accuracy: float
    total: int
    correct: int
    per_task_type: dict[str, dict]
    steps_mean: float
    token_usage: dict[str, int]
    stop_reason_counts: dict[str, int]
