"""进化数据结构，对应 optimizer.step()。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetSuggestionSet:
    """表示目标文件的建议集合。"""

    target: str
    kind: str
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    success_anchors: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationResult:
    """表示一次校验的结果。"""

    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EvolutionRecord:
    """表示单个目标文件的一次进化记录。"""

    target_file: str
    original_content: str
    evolved_content: str
    reason: str
    status: str
    suggestions: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class EvolutionResult:
    """表示一次整体进化流程的汇总结果。"""

    skills_version: str | None
    prompts_version: str | None
    records: list[EvolutionRecord] = field(default_factory=list)
    accepted_count: int = 0
    rejected_count: int = 0
