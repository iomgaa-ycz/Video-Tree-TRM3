"""进化数据结构与核心逻辑，对应 optimizer.step()。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """格式验证的结果。"""

    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EvolutionRecord:
    """单个目标文件的一次进化记录。"""

    target_file: str
    """目标文件名，如 'temporal-reasoning.md'。"""

    target_type: str
    """目标类型: 'skill' / 'system' / 'tool'。"""

    original_content: str
    """改写前原文。"""

    evolved_content: str
    """改写后内容；rejected 时与 original_content 相同。"""

    reason: str
    """状态说明。"""

    status: str
    """'accepted' / 'rejected' / 'skipped'。"""

    source_version: str
    """改写前版本号，如 'v1'。"""

    result_version: str | None = None
    """改写后版本号；rejected/skipped 时为 None。"""

    suggestions: list[dict[str, Any]] = field(default_factory=list)
    """LLM 输出的改动建议列表。"""

    attempts: list[dict[str, Any]] = field(default_factory=list)
    """每次 LLM 调用的原始响应摘要。"""

    validation_errors: list[str] = field(default_factory=list)
    """验证失败的具体原因。"""


@dataclass
class EvolutionResult:
    """一次整体进化流程的汇总结果。"""

    skills_version: str | None
    """新 skills 版本号；无改动时为 None。"""

    prompts_version: str | None
    """新 prompts 版本号；无改动时为 None。"""

    records: list[EvolutionRecord] = field(default_factory=list)
    """所有目标的进化记录。"""

    accepted_count: int = 0
    """通过验证的改写数。"""

    rejected_count: int = 0
    """未通过验证的改写数。"""

    skipped_count: int = 0
    """因无失败案例而跳过的目标数。"""
