"""Harness 评估系统：自我进化循环的基础设施。"""

from __future__ import annotations

from core.harness.config import RunConfig, load_config
from core.harness.diagnose import (
    CaseSample,
    DiagnosisResult,
    ErrorAttribution,
    QuestionMetrics,
    SkillCasePack,
    SkillStepAdherence,
    SpanMetrics,
    SystemCasePack,
    ToolCasePack,
    run_diagnosis,
)
from core.harness.evolve import EvolutionRecord, EvolutionResult, ValidationResult
from core.harness.inference import InferenceResult, TracePlugin, run_inference
from core.harness.log import HarnessLog
from core.harness.question_gen import (
    GeneratedQuestion,
    QuestionGenResult,
    load_benchmark,
)
from core.harness.runner import Runner

__all__ = [
    "CaseSample",
    "DiagnosisResult",
    "ErrorAttribution",
    "EvolutionRecord",
    "EvolutionResult",
    "GeneratedQuestion",
    "HarnessLog",
    "InferenceResult",
    "QuestionGenResult",
    "QuestionMetrics",
    "RunConfig",
    "Runner",
    "SkillCasePack",
    "SkillStepAdherence",
    "SpanMetrics",
    "SystemCasePack",
    "ToolCasePack",
    "TracePlugin",
    "ValidationResult",
    "load_benchmark",
    "load_config",
    "run_diagnosis",
    "run_inference",
]
