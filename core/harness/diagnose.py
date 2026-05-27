"""诊断模块 — 两阶段流水线的数据结构与 Stage 2 聚合。"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from loguru import logger

from core.harness.log import HarnessLog
from core.harness.metrics import (
    compute_question_metrics,
    extract_json_from_response,
    load_diagnose_prompt,
)
from core.harness.question_gen import GeneratedQuestion, load_benchmark
from core.llm_client import LLMClient
from core.workspace import resolve_paths


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

    error_type: str
    """错误的主要类别。"""

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


def _now_iso() -> str:
    """返回当前 UTC 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float:
    """计算均值；空列表返回 0.0。"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> float:
    """按线性插值计算分位数；空列表返回 0.0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _parse_steps_json(raw_steps: Any) -> list[dict[str, Any]]:
    """将数据库中的 steps_json 解析为步骤列表。"""
    if isinstance(raw_steps, list):
        return [step for step in raw_steps if isinstance(step, dict)]
    if not isinstance(raw_steps, str) or not raw_steps.strip():
        return []
    try:
        parsed = json.loads(raw_steps)
    except json.JSONDecodeError:
        logger.warning("steps_json 解析失败，回退为空列表")
        return []
    if isinstance(parsed, list):
        return [step for step in parsed if isinstance(step, dict)]
    return []


def _format_trace_text(traces: list[dict[str, Any]]) -> str:
    """将 trace 列表格式化为推理失败分类可读文本。"""
    lines: list[str] = []
    for trace in traces:
        args = trace.get("tool_args", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"Step {trace.get('step', '')}: thought={trace.get('thought', '')} | "
            f"tool={trace.get('tool_name', '')} | args={args} | "
            f"output={trace.get('tool_output', '')}"
        )
    return "\n".join(lines)


def _parse_level_sequence(level_jump_pattern: str) -> list[str]:
    """从层级跳转文本中提取层级序列。"""
    return re.findall(r"L\d+", level_jump_pattern or "")


def _extract_level_from_node(node_id: str) -> str | None:
    """从节点 ID 中提取 L1/L2/L3 层级。"""
    match = re.search(r"L([123])", node_id or "")
    if match is None:
        return None
    return f"L{match.group(1)}"


def _load_json(path: Path) -> dict[str, Any]:
    """加载 JSON 文件。"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_skill_content(skills_dir: Path, task_type: str) -> str:
    """按题型加载 skill 文件，不存在时回退默认策略。"""
    task_filename = f"{task_type.lower().replace(' ', '-')}.md"
    task_path = skills_dir / task_filename
    if task_path.exists():
        return task_path.read_text(encoding="utf-8")
    fallback_path = skills_dir / "default-strategy.md"
    return fallback_path.read_text(encoding="utf-8")


def _lookup_question(
    prediction_row: dict[str, Any],
    question_lookup: dict[tuple[str, str], GeneratedQuestion],
) -> GeneratedQuestion | None:
    """按 video_id 与 question_id 查找题目原文。"""
    key = (str(prediction_row["video_id"]), str(prediction_row["question_id"]))
    return question_lookup.get(key)


def _normalize_prediction_row(
    prediction_row: dict[str, Any],
    question_lookup: dict[tuple[str, str], GeneratedQuestion],
) -> dict[str, Any]:
    """将数据库预测记录补全为指标计算所需结构。"""
    steps = _parse_steps_json(prediction_row.get("steps_json"))
    question = _lookup_question(prediction_row, question_lookup)
    normalized = dict(prediction_row)
    normalized["steps_json"] = steps
    normalized["correct"] = prediction_row.get("prediction") == prediction_row.get(
        "answer"
    )
    if question is not None:
        normalized["question"] = question.question
        normalized["options"] = question.options
        normalized["task_type"] = question.task_type
    else:
        normalized.setdefault("question", "")
        normalized.setdefault("options", [])
    return normalized


def _load_run_max_steps(
    log: HarnessLog, run_id: str, predictions: list[dict[str, Any]]
) -> int:
    """从 _runs.config 中读取 max_steps，失败时回退到观测值。"""
    rows = log.query("SELECT config FROM _runs WHERE run_id = ?", (run_id,))
    if rows:
        raw_config = rows[0].get("config")
        if isinstance(raw_config, str) and raw_config.strip():
            try:
                parsed = json.loads(raw_config)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("max_steps"), int):
                return max(parsed["max_steps"], 1)
    observed = [
        len(prediction.get("steps_json", []))
        for prediction in predictions
        if isinstance(prediction.get("steps_json"), list)
    ]
    return max(max(observed, default=0), 1)


def _ensure_diagnosis_tables(log: HarnessLog) -> None:
    """创建诊断阶段所需的数据表。"""
    log.execute("""
        CREATE TABLE IF NOT EXISTS question_metrics (
            run_id TEXT,
            video_id TEXT,
            question_id TEXT,
            is_correct INTEGER,
            error_type TEXT,
            reasoning_failure_type TEXT,
            evidence_sufficient INTEGER,
            missed_nodes_json JSON,
            budget_usage REAL,
            format_compliant INTEGER,
            high_confidence INTEGER,
            low_confidence INTEGER,
            confirmation_bias_detected INTEGER,
            created_at TEXT
        )
    """)
    log.execute("""
        CREATE TABLE IF NOT EXISTS span_evaluations (
            run_id TEXT,
            video_id TEXT,
            question_id TEXT,
            step INTEGER,
            tool_name TEXT,
            extraction_completeness REAL,
            hallucination_rate REAL,
            missed_tags_json JSON,
            hallucinated_tags_json JSON
        )
    """)
    log.execute("""
        CREATE TABLE IF NOT EXISTS diagnose_traces (
            run_id TEXT,
            video_id TEXT,
            question_id TEXT,
            stage TEXT,
            input_json JSON,
            output_json JSON,
            created_at TEXT
        )
    """)


def _clear_existing_diagnosis_rows(log: HarnessLog, run_id: str) -> None:
    """删除同一 run 的旧诊断结果，确保重复执行时幂等。"""
    for table_name in ("question_metrics", "span_evaluations", "diagnose_traces"):
        log.execute(f"DELETE FROM {table_name} WHERE run_id = ?", (run_id,))


def _insert_question_metrics_row(
    log: HarnessLog,
    run_id: str,
    qm: QuestionMetrics,
    attribution: ErrorAttribution | None,
) -> None:
    """写入单题汇总指标。"""
    log.execute(
        """
        INSERT INTO question_metrics (
            run_id, video_id, question_id, is_correct, error_type,
            reasoning_failure_type, evidence_sufficient, missed_nodes_json,
            budget_usage, format_compliant, high_confidence, low_confidence,
            confirmation_bias_detected, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            qm.video_id,
            qm.question_id,
            int(qm.correct),
            attribution.error_type if attribution is not None else None,
            attribution.reasoning_failure_type if attribution is not None else None,
            int(qm.evidence_sufficient),
            json.dumps(qm.missed_nodes, ensure_ascii=False),
            qm.budget_usage,
            int(qm.format_compliance >= 1.0),
            int(qm.confidence_calibration == "high_conf_wrong"),
            int(qm.confidence_calibration == "low_conf_right"),
            int(qm.confirmation_bias),
            _now_iso(),
        ),
    )


def _insert_span_rows(log: HarnessLog, run_id: str, qm: QuestionMetrics) -> None:
    """写入单题 span 级评估。"""
    for span in qm.span_metrics:
        log.execute(
            """
            INSERT INTO span_evaluations (
                run_id, video_id, question_id, step, tool_name,
                extraction_completeness, hallucination_rate, missed_tags_json,
                hallucinated_tags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                qm.video_id,
                qm.question_id,
                span.step,
                span.tool_name,
                span.extraction_completeness,
                span.hallucination_rate,
                json.dumps(span.missed_info_tags, ensure_ascii=False),
                json.dumps(span.hallucination_tags, ensure_ascii=False),
            ),
        )


def _insert_diagnose_trace(
    log: HarnessLog,
    run_id: str,
    video_id: str,
    question_id: str,
    stage: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
) -> None:
    """写入诊断阶段的中间痕迹。"""
    log.execute(
        """
        INSERT INTO diagnose_traces (
            run_id, video_id, question_id, stage, input_json, output_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            video_id,
            question_id,
            stage,
            json.dumps(input_payload, ensure_ascii=False),
            json.dumps(output_payload, ensure_ascii=False),
            _now_iso(),
        ),
    )


def _classify_reasoning_failure(
    judge_client: LLMClient,
    prompts_dir: Path,
    prediction: dict[str, Any],
    traces: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """调用 judge 模型细分推理失败类型。"""
    system_prompt = load_diagnose_prompt(prompts_dir, "diagnose_reasoning_failure.md")
    user_prompt = (
        f"## 题目\n{prediction.get('question', '')}\n\n"
        f"## 正确答案\n{prediction.get('answer', '')}\n\n"
        f"## Agent 错误预测\n{prediction.get('prediction', '')}\n\n"
        f"## 执行轨迹\n{_format_trace_text(traces)}"
    )
    response = judge_client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    raw_content = response.choices[0].message.content
    parsed = extract_json_from_response(raw_content)
    failure_type = parsed.get("type")
    if not isinstance(failure_type, str) or not failure_type.strip():
        failure_type = None
    return failure_type, parsed


def attribute_error(qm: QuestionMetrics) -> ErrorAttribution:
    """按瀑布规则归因单题错误类型。"""
    avg_completeness = _mean([span.extraction_completeness for span in qm.span_metrics])
    max_hallucination = max(
        (span.hallucination_rate for span in qm.span_metrics), default=0.0
    )

    if avg_completeness < 0.5 or max_hallucination > 0.5:
        error_type = "extraction_failure"
    elif len(qm.missed_nodes) > 0:
        error_type = "search_failure"
    elif qm.evidence_sufficient is True:
        error_type = "reasoning_failure"
    else:
        error_type = "mixed"

    return ErrorAttribution(
        question_id=qm.question_id,
        error_type=error_type,
        reasoning_failure_type=None,
    )


def aggregate_d2(all_metrics: list[QuestionMetrics]) -> dict[str, dict]:
    """按工具聚合 span 级质量指标。"""
    grouped: dict[str, list[SpanMetrics]] = defaultdict(list)
    for qm in all_metrics:
        for span in qm.span_metrics:
            grouped[span.tool_name].append(span)

    result: dict[str, dict] = {}
    for tool_name, spans in grouped.items():
        missed_counter: Counter[str] = Counter()
        hallucinated_counter: Counter[str] = Counter()
        for span in spans:
            missed_counter.update(span.missed_info_tags)
            hallucinated_counter.update(span.hallucination_tags)
        result[tool_name] = {
            "avg_completeness": _mean([span.extraction_completeness for span in spans]),
            "avg_hallucination": _mean([span.hallucination_rate for span in spans]),
            "n_calls": len(spans),
            "top_missed": [[tag, count] for tag, count in missed_counter.most_common()],
            "top_hallucinated": [
                [tag, count] for tag, count in hallucinated_counter.most_common()
            ],
        }
    return result


def aggregate_d3(all_metrics: list[QuestionMetrics]) -> dict[str, dict]:
    """按题型与正误拆分搜索行为统计。"""
    grouped: dict[str, dict[str, list[QuestionMetrics]]] = defaultdict(
        lambda: {"correct": [], "incorrect": []}
    )
    for qm in all_metrics:
        bucket = "correct" if qm.correct else "incorrect"
        grouped[qm.task_type][bucket].append(qm)

    result: dict[str, dict] = {}
    for task_type, task_groups in grouped.items():
        task_result: dict[str, Any] = {}
        for bucket_name, metrics_group in task_groups.items():
            task_result[bucket_name] = {
                "repeat_visit_rate": _mean(
                    [qm.repeat_visit_rate for qm in metrics_group]
                ),
                "keyword_repetition": _mean(
                    [qm.search_keyword_repetition for qm in metrics_group]
                ),
                "l3_usage_rate": _mean(
                    [
                        1.0
                        if "L3" in _parse_level_sequence(qm.level_jump_pattern)
                        else 0.0
                        for qm in metrics_group
                    ]
                ),
                "observe_frame_rate": _mean(
                    [
                        1.0 if qm.tool_usage.get("observe_frame", 0) > 0 else 0.0
                        for qm in metrics_group
                    ]
                ),
                "avg_steps": _mean([qm.budget_usage for qm in metrics_group]),
                "n_questions": len(metrics_group),
            }

        incorrect_group = task_groups["incorrect"]
        level_counts = {"L1": 0, "L2": 0, "L3": 0}
        for qm in incorrect_group:
            for node_id in qm.missed_nodes:
                level = _extract_level_from_node(node_id)
                if level in level_counts:
                    level_counts[level] += 1

        task_result["incorrect"]["missed_nodes_rate"] = _mean(
            [1.0 if qm.missed_nodes else 0.0 for qm in incorrect_group]
        )
        task_result["incorrect"]["missed_node_levels"] = level_counts
        result[task_type] = task_result
    return result


def aggregate_d4(all_metrics: list[QuestionMetrics]) -> dict[str, dict]:
    """按题型聚合 skill step 遵循与收益差异。"""
    grouped: dict[str, list[QuestionMetrics]] = defaultdict(list)
    for qm in all_metrics:
        grouped[qm.task_type].append(qm)

    result: dict[str, dict] = {}
    for task_type, metrics_group in grouped.items():
        total_steps = 0
        adhered_steps = 0
        step_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "adhered": 0,
                "deviated": 0,
                "correct_adhered": 0,
                "correct_deviated": 0,
            }
        )

        for qm in metrics_group:
            for step in qm.skill_adherence:
                total_steps += 1
                if step.adhered:
                    adhered_steps += 1
                    step_stats[step.step_label]["adhered"] += 1
                    step_stats[step.step_label]["correct_adhered"] += int(qm.correct)
                else:
                    step_stats[step.step_label]["deviated"] += 1
                    step_stats[step.step_label]["correct_deviated"] += int(qm.correct)

        task_steps: dict[str, dict[str, float]] = {}
        for step_label, stats in step_stats.items():
            adhered_count = stats["adhered"]
            deviated_count = stats["deviated"]
            total_count = adhered_count + deviated_count
            acc_adhered = (
                stats["correct_adhered"] / adhered_count if adhered_count > 0 else 0.0
            )
            acc_deviated = (
                stats["correct_deviated"] / deviated_count
                if deviated_count > 0
                else 0.0
            )
            task_steps[step_label] = {
                "adherence_rate": adhered_count / total_count if total_count else 0.0,
                "acc_adhered": acc_adhered,
                "acc_deviated": acc_deviated,
                "delta": acc_adhered - acc_deviated,
            }

        result[task_type] = {
            "overall_adherence": adhered_steps / total_steps if total_steps else 0.0,
            "n_questions": len(metrics_group),
            "steps": task_steps,
        }
    return result


def aggregate_d5(all_metrics: list[QuestionMetrics]) -> dict[str, Any]:
    """跨题型聚合决策与校准模式。"""
    if not all_metrics:
        return {
            "format_compliance_rate": 0.0,
            "budget_usage_median": 0.0,
            "budget_usage_p25": 0.0,
            "budget_usage_p75": 0.0,
            "early_submit_rate": 0.0,
            "high_conf_wrong_rate": 0.0,
            "low_conf_right_rate": 0.0,
            "confirmation_bias_rate": 0.0,
            "per_type_bias": {},
        }

    budget_values = [qm.budget_usage for qm in all_metrics]
    wrong_metrics = [qm for qm in all_metrics if not qm.correct]
    per_type_groups: dict[str, list[QuestionMetrics]] = defaultdict(list)
    for qm in all_metrics:
        per_type_groups[qm.task_type].append(qm)

    return {
        "format_compliance_rate": _mean([qm.format_compliance for qm in all_metrics]),
        "budget_usage_median": median(budget_values),
        "budget_usage_p25": _percentile(budget_values, 0.25),
        "budget_usage_p75": _percentile(budget_values, 0.75),
        "early_submit_rate": (
            sum(1 for qm in wrong_metrics if qm.budget_usage < 0.3) / len(wrong_metrics)
            if wrong_metrics
            else 0.0
        ),
        "high_conf_wrong_rate": _mean(
            [
                1.0 if qm.confidence_calibration == "high_conf_wrong" else 0.0
                for qm in all_metrics
            ]
        ),
        "low_conf_right_rate": _mean(
            [
                1.0 if qm.confidence_calibration == "low_conf_right" else 0.0
                for qm in all_metrics
            ]
        ),
        "confirmation_bias_rate": _mean(
            [1.0 if qm.confirmation_bias else 0.0 for qm in all_metrics]
        ),
        "per_type_bias": {
            task_type: _mean([1.0 if qm.confirmation_bias else 0.0 for qm in group])
            for task_type, group in per_type_groups.items()
        },
    }


_DIAGNOSE_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def run_diagnosis(
    log: HarnessLog,
    run_id: str,
    workspace_dir: Path,
    skills_dir: Path,
    concurrency: int = 1,
    task_types: list[str] | None = None,
    video_ids: list[str] | None = None,
    question_ids: list[str] | None = None,
    only_incorrect: bool = False,
    stop_reasons: list[str] | None = None,
    diagnose_prompts_dir: Path = _DIAGNOSE_PROMPTS_DIR,
) -> DiagnosisResult:
    """执行两阶段诊断流水线并写出聚合报告。

    参数:
        diagnose_prompts_dir: 诊断 prompt 目录，默认项目根目录 prompts/。
            与推理 prompt（store/prompts/v1/）分开管理，诊断 prompt 不参与版本化。
    """
    prompts_dir = diagnose_prompts_dir
    effective_stop_reasons = stop_reasons or ["finished", "budget_exceeded"]
    task_type_filter = set(task_types or [])
    video_filter = set(video_ids or [])
    question_filter = set(question_ids or [])

    prediction_rows = log.query(
        "SELECT * FROM predictions WHERE run_id = ?",
        (run_id,),
    )
    filtered_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        is_correct = row.get("prediction") == row.get("answer")
        if (
            effective_stop_reasons
            and row.get("stop_reason") not in effective_stop_reasons
        ):
            continue
        if task_type_filter and row.get("task_type") not in task_type_filter:
            continue
        if video_filter and row.get("video_id") not in video_filter:
            continue
        if question_filter and row.get("question_id") not in question_filter:
            continue
        if only_incorrect and is_correct:
            continue
        filtered_rows.append(row)

    paths = resolve_paths(workspace_dir)
    benchmark_questions = load_benchmark(paths.questions_dir)
    question_lookup = {(qa.video_id, qa.question_id): qa for qa in benchmark_questions}
    normalized_predictions = [
        _normalize_prediction_row(row, question_lookup) for row in filtered_rows
    ]

    tree_cache = {
        video_id: _load_json(paths.videos_dir / video_id / "tree.json")
        for video_id in sorted({row["video_id"] for row in normalized_predictions})
    }
    skill_cache = {
        task_type: _load_skill_content(skills_dir, task_type)
        for task_type in sorted({row["task_type"] for row in normalized_predictions})
    }

    _ensure_diagnosis_tables(log)
    _clear_existing_diagnosis_rows(log, run_id)

    trace_rows = log.query(
        """
        SELECT video_id, question_id, step, tool_name, tool_args, tool_output, thought
        FROM traces
        WHERE run_id = ?
        ORDER BY video_id, question_id, step
        """,
        (run_id,),
    )
    traces_by_question: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        traces_by_question[(row["video_id"], row["question_id"])].append(row)

    max_steps = _load_run_max_steps(log, run_id, normalized_predictions)

    def _worker(prediction: dict[str, Any]) -> dict[str, Any]:
        judge_client = LLMClient.from_env("JUDGE_LLM", thinking=False)
        key = (prediction["video_id"], prediction["question_id"])
        traces = traces_by_question.get(key, [])
        qm = compute_question_metrics(
            prediction=prediction,
            traces=traces,
            tree_data=tree_cache[prediction["video_id"]],
            skill_content=skill_cache[prediction["task_type"]],
            judge_client=judge_client,
            prompts_dir=prompts_dir,
            max_steps=max_steps,
        )
        attribution = attribute_error(qm) if not qm.correct else None
        return {
            "prediction": prediction,
            "traces": traces,
            "metrics": qm,
            "attribution": attribution,
        }

    worker_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_worker, prediction): prediction
            for prediction in normalized_predictions
        }
        for future in as_completed(futures):
            prediction = futures[future]
            try:
                worker_results.append(future.result())
            except Exception:
                logger.exception(
                    "诊断失败: {} / {}",
                    prediction["video_id"],
                    prediction["question_id"],
                )
                raise

    reasoning_client = LLMClient.from_env("JUDGE_LLM", thinking=False)
    for item in worker_results:
        attribution = item["attribution"]
        if attribution is None or attribution.error_type != "reasoning_failure":
            continue
        reasoning_type, reasoning_output = _classify_reasoning_failure(
            reasoning_client,
            prompts_dir,
            item["prediction"],
            item["traces"],
        )
        attribution.reasoning_failure_type = reasoning_type
        _insert_diagnose_trace(
            log,
            run_id,
            item["metrics"].video_id,
            item["metrics"].question_id,
            "reasoning_failure",
            {
                "prediction": item["prediction"],
                "traces": item["traces"],
            },
            reasoning_output,
        )

    all_metrics = [item["metrics"] for item in worker_results]
    error_attributions = [
        item["attribution"]
        for item in worker_results
        if item["attribution"] is not None
    ]
    attribution_distribution = dict(
        Counter(attr.error_type for attr in error_attributions)
    )
    reasoning_failure_types = dict(
        Counter(
            attr.reasoning_failure_type
            for attr in error_attributions
            if attr.reasoning_failure_type
        )
    )

    for item in worker_results:
        qm = item["metrics"]
        attribution = item["attribution"]
        _insert_question_metrics_row(log, run_id, qm, attribution)
        _insert_span_rows(log, run_id, qm)
        _insert_diagnose_trace(
            log,
            run_id,
            qm.video_id,
            qm.question_id,
            "compute_question_metrics",
            {
                "prediction": item["prediction"],
                "traces": item["traces"],
            },
            {
                "question_metrics": asdict(qm),
                "error_attribution": asdict(attribution) if attribution else None,
            },
        )

    result = DiagnosisResult(
        run_id=run_id,
        filter_summary={
            "stop_reasons": effective_stop_reasons,
            "task_types": sorted(task_type_filter),
            "video_ids": sorted(video_filter),
            "question_ids": sorted(question_filter),
            "only_incorrect": only_incorrect,
            "total_predictions": len(prediction_rows),
            "selected_predictions": len(normalized_predictions),
        },
        error_attributions=error_attributions,
        attribution_distribution=attribution_distribution,
        reasoning_failure_types=reasoning_failure_types,
        tool_quality=aggregate_d2(all_metrics),
        search_effectiveness=aggregate_d3(all_metrics),
        skill_compliance=aggregate_d4(all_metrics),
        decision_patterns=aggregate_d5(all_metrics),
    )

    output_path = Path(workspace_dir) / "analyses" / f"diagnosis_{run_id}.json"
    output_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
