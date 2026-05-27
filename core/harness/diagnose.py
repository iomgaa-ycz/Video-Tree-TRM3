"""诊断模块 — 两阶段流水线的数据结构与 Stage 2 聚合。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanMetrics:
    """单次工具调用的输出质量指标。"""

    step: int
    """工具调用所在的步骤编号。"""

    tool_name: str
    """本次调用使用的工具名称。"""

    extraction_completeness: float
    """信息提取完整度。"""

    hallucination_rate: float
    """幻觉内容占比。"""

    missed_info_tags: list[str] = field(default_factory=list)
    """未提取信息的标签列表。"""

    hallucination_tags: list[str] = field(default_factory=list)
    """幻觉内容的标签列表。"""


@dataclass
class SkillStepAdherence:
    """单个 skill step 的遵循判定。"""

    step_label: str
    """被判定的步骤标签。"""

    adhered: bool
    """该步骤是否被遵循。"""

    description: str
    """对遵循情况的文字说明。"""


@dataclass
class QuestionMetrics:
    """单题的 13 个原始指标，即 Stage 1 输出。"""

    question_id: str
    """题目唯一标识。"""

    video_id: str
    """对应视频唯一标识。"""

    task_type: str
    """题目任务类型。"""

    correct: bool
    """该题最终是否答对。"""

    format_compliance: float
    """输出格式遵循程度。"""

    budget_usage: float
    """预算使用比例。"""

    confidence_calibration: str
    """置信度校准结论。"""

    repeat_visit_rate: float
    """重复访问节点的比例。"""

    search_keyword_repetition: float
    """搜索关键词重复率。"""

    level_jump_pattern: str
    """层级跳转模式描述。"""

    tool_usage: dict[str, int]
    """各工具的调用次数统计。"""

    span_metrics: list[SpanMetrics]
    """该题全部工具调用的片段级质量指标。"""

    missed_nodes: list[str]
    """该题遗漏的节点列表。"""

    skill_adherence: list[SkillStepAdherence]
    """该题对 skill 步骤的遵循情况。"""

    confirmation_bias: bool
    """是否出现确认偏误。"""

    evidence_sufficient: bool
    """当前证据是否充足。"""


@dataclass
class ErrorAttribution:
    """D1 错误归因。"""

    question_id: str
    """发生错误归因的题目唯一标识。"""

    primary_cause: str
    """错误的主要原因。"""

    reasoning_failure_type: str | None
    """推理失败类型；若不适用则为 None。"""


@dataclass
class DiagnosisResult:
    """完整诊断报告，即 Stage 2 输出。"""

    run_id: str
    """本次诊断运行的唯一标识。"""

    filter_summary: dict[str, Any] = field(default_factory=dict)
    """筛选条件与筛选结果摘要。"""

    error_attributions: list[ErrorAttribution] = field(default_factory=list)
    """错误归因结果列表。"""

    attribution_distribution: dict[str, int] = field(default_factory=dict)
    """各归因类别的分布统计。"""

    reasoning_failure_types: dict[str, int] = field(default_factory=dict)
    """各推理失败类型的分布统计。"""

    tool_quality: dict[str, dict[str, Any]] = field(default_factory=dict)
    """按工具聚合的质量分析结果。"""

    search_effectiveness: dict[str, dict[str, Any]] = field(default_factory=dict)
    """搜索有效性的聚合统计。"""

    skill_compliance: dict[str, dict[str, Any]] = field(default_factory=dict)
    """技能遵循情况的聚合统计。"""

    decision_patterns: dict[str, Any] = field(default_factory=dict)
    """决策模式与行为模式摘要。"""


# 兼容现有包级导出名称；不新增额外数据结构定义。
SpanEvaluation = SpanMetrics
QuestionDiagnosis = QuestionMetrics
