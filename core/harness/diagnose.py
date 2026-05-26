"""诊断数据结构，对应 backward() 三阶段流水线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanEvaluation:
    """片段级评估结果。"""

    step: int
    tool_name: str
    extraction_completeness: float
    accuracy: float
    score: float
    missed_info: str
    hallucinations: str


@dataclass
class QuestionDiagnosis:
    """单题诊断结果。"""

    question_id: str
    video_id: str
    task_type: str
    correct: bool
    root_causes: list[dict[str, Any]] = field(default_factory=list)
    effective_patterns: list[dict[str, Any]] = field(default_factory=list)
    missed_nodes: list[str] = field(default_factory=list)
    key_insight: str = ""


@dataclass
class DiagnosisResult:
    """整次运行的诊断汇总结果。"""

    run_id: str
    question_diagnoses: list[QuestionDiagnosis] = field(default_factory=list)
    aggregations: list[dict[str, Any]] = field(default_factory=list)
