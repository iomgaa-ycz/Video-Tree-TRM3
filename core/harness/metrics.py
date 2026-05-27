"""Stage 1 指标计算 — 规则指标与 LLM judge 编排。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from json_repair import repair_json

if TYPE_CHECKING:
    from core.harness.diagnose import QuestionMetrics, SkillStepAdherence, SpanMetrics


def _parse_json_object(raw: str) -> dict | None:
    """将原始字符串解析为字典；失败时返回 None。"""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = json.loads(repair_json(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def calc_format_compliance(raw_contents: list[str]) -> float:
    """每步 JSON 是否包含 reflect/plan/action 三个字段。合规步数/总步数。空列表返回 1.0。"""
    if not raw_contents:
        return 1.0

    compliant_count = 0
    for raw in raw_contents:
        parsed = _parse_json_object(raw)
        if parsed is not None and all(
            key in parsed for key in ("reflect", "plan", "action")
        ):
            compliant_count += 1

    return compliant_count / len(raw_contents)


def calc_budget_usage(steps_used: int, max_steps: int) -> float:
    """steps_used / max_steps。"""
    return steps_used / max_steps


def calc_confidence_calibration(confidence: float, correct: bool) -> str:
    """置信度校准分类。>=0.7 且答错→'high_conf_wrong'；<0.5 且答对→'low_conf_right'；否则→'calibrated'。"""
    if confidence >= 0.7 and not correct:
        return "high_conf_wrong"
    if confidence < 0.5 and correct:
        return "low_conf_right"
    return "calibrated"


def calc_repeat_visit_rate(view_node_ids: list[str]) -> float:
    """重复访问率。1 - (unique / total)。空列表返回 0.0。"""
    if not view_node_ids:
        return 0.0
    return 1 - (len(set(view_node_ids)) / len(view_node_ids))


def _trigrams(text: str) -> set[str]:
    """返回字符串的字符级 trigram 集合。"""
    if len(text) < 3:
        return set()
    return {text[index : index + 3] for index in range(len(text) - 2)}


def calc_search_keyword_repetition(queries: list[str]) -> float:
    """连续 search_similar 查询的最大字符级 trigram Jaccard 相似度。

    不足 2 个查询时返回 0.0。
    Trigram: sliding window of 3 chars over each query string.
    Jaccard(a,b) = |trigrams(a) ∩ trigrams(b)| / |trigrams(a) ∪ trigrams(b)|
    Return max Jaccard over all consecutive pairs.
    """
    if len(queries) < 2:
        return 0.0

    max_score = 0.0
    for left, right in zip(queries, queries[1:], strict=False):
        left_trigrams = _trigrams(left)
        right_trigrams = _trigrams(right)
        union = left_trigrams | right_trigrams
        if not union:
            score = 0.0
        else:
            score = len(left_trigrams & right_trigrams) / len(union)
        if score > max_score:
            max_score = score
    return max_score


def calc_level_jump_pattern(view_node_ids: list[str]) -> str:
    """从 node_id 提取层级，拼成 'L1→L2→L3' 格式。

    Level extraction: use regex r'_L(\\d+)_' to find level number in each node_id.
    Node ids without a match are skipped.
    Empty result returns ''.
    """
    levels: list[str] = []
    for node_id in view_node_ids:
        match = re.search(r"_L(\d+)_", node_id)
        if match is not None:
            levels.append(f"L{match.group(1)}")
    return "→".join(levels)


def calc_tool_usage(tool_names: list[str]) -> dict[str, int]:
    """按 tool_name 计数，返回 dict。"""
    return dict(Counter(tool_names))


def _extract_last_confidence(raw_contents: list[str]) -> float:
    """从末步 raw_content 提取 reflect.confidence。失败时返回 0.5。"""
    try:
        parsed = _parse_json_object(raw_contents[-1])
        if parsed is None:
            raise ValueError("末步内容不是字典。")
        return float(parsed["reflect"]["confidence"])
    except Exception:
        return 0.5


def extract_rule_metrics(
    prediction: dict, raw_contents: list[str], max_steps: int
) -> dict:
    """从 prediction 和 raw_contents 提取全部 7 个规则指标。

    prediction dict shape:
      prediction['steps_json']: list[dict], each dict may have:
        - 'tool_call': dict with 'tool' and 'args' (dict)
        - For tool_call.tool == 'view_node': args may have 'node_id' (str)
        - For tool_call.tool == 'search_similar': args may have 'query' (str)
      prediction['correct']: bool

    Logic:
      1. Parse steps_json to collect: view_node_ids, search_queries, tool_names
      2. Extract confidence from raw_contents via _extract_last_confidence
      3. correct = prediction.get('correct', False)
      4. Call all 7 calc_* functions
      5. Return dict with keys:
         'format_compliance', 'budget_usage', 'confidence_calibration',
         'repeat_visit_rate', 'search_keyword_repetition', 'level_jump_pattern', 'tool_usage'
    """
    view_node_ids: list[str] = []
    search_queries: list[str] = []
    tool_names: list[str] = []

    for step in prediction.get("steps_json", []):
        tool_call = step.get("tool_call", {})
        if not isinstance(tool_call, dict):
            continue

        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if isinstance(tool_name, str):
            tool_names.append(tool_name)

        if tool_name == "view_node":
            node_id = args.get("node_id")
            if isinstance(node_id, str):
                view_node_ids.append(node_id)

        if tool_name == "search_similar":
            query = args.get("query")
            if isinstance(query, str):
                search_queries.append(query)

    confidence = prediction.get("answer_confidence", 0.5)
    if raw_contents:
        last_step = _parse_json_object(raw_contents[-1])
        if isinstance(last_step, dict):
            confidence = _extract_last_confidence(raw_contents)

    correct = bool(prediction.get("correct", False))
    steps_used = len(prediction.get("steps_json", []))

    return {
        "format_compliance": calc_format_compliance(raw_contents),
        "budget_usage": calc_budget_usage(steps_used, max_steps),
        "confidence_calibration": calc_confidence_calibration(confidence, correct),
        "repeat_visit_rate": calc_repeat_visit_rate(view_node_ids),
        "search_keyword_repetition": calc_search_keyword_repetition(search_queries),
        "level_jump_pattern": calc_level_jump_pattern(view_node_ids),
        "tool_usage": calc_tool_usage(tool_names),
    }


def extract_json_from_response(raw: str) -> dict:
    """从 LLM 回复中提取 JSON。

    Strategy (in order):
    1. Try to find markdown code block: ```json or ```
       Extract content between first ```...```, then json.loads.
    2. Try to find outermost {...}: from first '{' to last '}'.
       Then json.loads.
    3. Use repair_json from json_repair, then json.loads on the result.
    Raise ValueError if all three strategies fail.
    """
    block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if block_match is not None:
        try:
            parsed = json.loads(block_match.group(1))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            if isinstance(parsed, dict):
                return parsed

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and start <= end:
        try:
            parsed = json.loads(raw[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        else:
            if isinstance(parsed, dict):
                return parsed

    try:
        parsed = json.loads(repair_json(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("无法从 LLM 回复中提取 JSON。") from exc

    if isinstance(parsed, dict):
        return parsed
    raise ValueError("无法从 LLM 回复中提取 JSON。")


def load_diagnose_prompt(prompts_dir: Path, filename: str) -> str:
    """加载 prompt 文件内容。文件不存在时 raise FileNotFoundError。"""
    return (prompts_dir / filename).read_text(encoding="utf-8")


_SPAN_EVAL_TOOLS = {"view_node", "search_similar", "observe_frame"}


def _call_judge(judge_client, system_prompt, user_prompt) -> str:
    """调用 judge 模型并返回文本内容。"""
    from loguru import logger

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    logger.debug("调用 LLM judge，消息数={}", len(messages))
    response = judge_client.chat(messages)
    return response.choices[0].message.content


def _stringify_tool_args(tool_args) -> str:
    """将工具参数转换为紧凑文本。"""
    if isinstance(tool_args, str):
        return tool_args
    return json.dumps(tool_args, ensure_ascii=False, sort_keys=True)


def _parse_tool_args(tool_args) -> dict[str, object]:
    """解析 trace 中的工具参数。"""
    from loguru import logger

    if isinstance(tool_args, dict):
        return tool_args
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
        except json.JSONDecodeError:
            logger.warning("tool_args 解析失败，回退为空字典: {}", tool_args)
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _format_trace_text(traces: list[dict]) -> str:
    """将 trace 列表格式化为 judge 可读文本。"""
    lines: list[str] = []
    for trace in traces:
        step = trace.get("step", "")
        thought = str(trace.get("thought", ""))[:100]
        tool_name = trace.get("tool_name", "")
        tool_args = _stringify_tool_args(trace.get("tool_args", {}))
        tool_output = str(trace.get("tool_output", ""))[:200]
        lines.append(
            f'Step {step}: thinking="{thought}" → {tool_name}({tool_args}) → {tool_output}'
        )
    return "\n".join(lines)


def _load_tree_content(tree_data: dict) -> str:
    """将树结构内容整理为文本。"""
    nodes = tree_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return ""

    chunks: list[str] = []
    for node_id in sorted(nodes):
        node = nodes.get(node_id, {})
        if not isinstance(node, dict):
            continue
        level = node.get("level", "")
        time_range = node.get("time_range", [0, 0])
        if not isinstance(time_range, list | tuple) or len(time_range) < 2:
            time_range = [0, 0]
        t_start, t_end = time_range[0], time_range[1]
        card_json = json.dumps(node.get("card", {}), ensure_ascii=False, sort_keys=True)
        chunks.append(
            f"### {node_id} | L{level} | {float(t_start):.0f}-{float(t_end):.0f}s\n{card_json}"
        )
    return "\n\n".join(chunks)


def evaluate_span(
    judge_client,
    prompts_dir,
    question,
    tool_name,
    tool_args,
    tool_output,
    ground_truth,
    step,
) -> "SpanMetrics":
    """评估单次 span 级工具调用质量。"""
    from core.harness.diagnose import SpanMetrics

    system_prompt = load_diagnose_prompt(Path(prompts_dir), "diagnose_span.md")
    user_prompt = (
        f"## 问题\n{question}\n\n"
        f"## 工具调用\n工具: {tool_name}\n参数: {json.dumps(tool_args, ensure_ascii=False)}\n\n"
        f"## 工具输出\n{tool_output}\n\n"
        f"## 原始数据（ground truth）\n{ground_truth}"
    )
    response_text = _call_judge(judge_client, system_prompt, user_prompt)
    parsed = extract_json_from_response(response_text)
    return SpanMetrics(
        step=int(step),
        tool_name=tool_name,
        extraction_completeness=float(parsed.get("extraction_completeness", 0.0)),
        hallucination_rate=float(parsed.get("hallucination_rate", 0.0)),
        missed_info_tags=list(parsed.get("missed_info_tags", [])),
        hallucination_tags=list(parsed.get("hallucination_tags", [])),
    )


def judge_missed_nodes(
    judge_client, prompts_dir, question, options, answer, tree_content, visited_node_ids
) -> list[str]:
    """评估是否遗漏关键节点。"""
    system_prompt = load_diagnose_prompt(Path(prompts_dir), "diagnose_missed_nodes.md")
    options_text = (
        "\n".join(options) if isinstance(options, list | tuple) else str(options)
    )
    user_prompt = (
        f"## 问题\n{question}\n\n"
        f"## 选项\n{options_text}\n\n"
        f"## 答案\n{answer}\n\n"
        f"## 树内容\n{tree_content}\n\n"
        f"## 已访问节点\n{json.dumps(visited_node_ids, ensure_ascii=False)}"
    )
    response_text = _call_judge(judge_client, system_prompt, user_prompt)
    parsed = extract_json_from_response(response_text)
    missed = parsed.get("missed_nodes", [])
    if isinstance(missed, list):
        return [str(nid) for nid in missed]
    return []


def judge_skill_adherence(
    judge_client, prompts_dir, skill_content, trace_text
) -> list["SkillStepAdherence"]:
    """评估技能步骤遵循情况。"""
    system_prompt = load_diagnose_prompt(
        Path(prompts_dir), "diagnose_skill_adherence.md"
    )
    user_prompt = f"## Skill 内容\n{skill_content}\n\n## 执行轨迹\n{trace_text}"
    response_text = _call_judge(judge_client, system_prompt, user_prompt)
    parsed = extract_json_from_response(response_text)
    from core.harness.diagnose import SkillStepAdherence

    steps = parsed.get("steps", [])
    if not isinstance(steps, list):
        return []

    results: list[SkillStepAdherence] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        results.append(
            SkillStepAdherence(
                step_label=str(item.get("step_label", "")),
                adhered=bool(item.get("adhered", False)),
                description=str(item.get("description", "")),
            )
        )
    return results


def judge_confirmation_bias(
    judge_client, prompts_dir, question, options, trace_text
) -> tuple[bool, str]:
    """评估是否存在确认偏误。"""
    system_prompt = load_diagnose_prompt(
        Path(prompts_dir), "diagnose_confirmation_bias.md"
    )
    options_text = (
        "\n".join(options) if isinstance(options, list | tuple) else str(options)
    )
    user_prompt = (
        f"## 问题\n{question}\n\n## 选项\n{options_text}\n\n## 执行轨迹\n{trace_text}"
    )
    response_text = _call_judge(judge_client, system_prompt, user_prompt)
    parsed = extract_json_from_response(response_text)
    return bool(parsed.get("has_bias", False)), str(parsed.get("evidence", ""))


def judge_evidence_sufficiency(
    judge_client, prompts_dir, question, options, answer, all_tool_outputs
) -> tuple[bool, str]:
    """评估当前证据是否充足。"""
    system_prompt = load_diagnose_prompt(
        Path(prompts_dir), "diagnose_evidence_sufficiency.md"
    )
    options_text = (
        "\n".join(options) if isinstance(options, list | tuple) else str(options)
    )
    user_prompt = (
        f"## 问题\n{question}\n\n"
        f"## 选项\n{options_text}\n\n"
        f"## 答案\n{answer}\n\n"
        f"## 所有工具输出\n{all_tool_outputs}"
    )
    response_text = _call_judge(judge_client, system_prompt, user_prompt)
    parsed = extract_json_from_response(response_text)
    return bool(parsed.get("sufficient", False)), str(parsed.get("reasoning", ""))


def _get_ground_truth_for_trace(
    tree_data: dict, tool_name: str, tool_args: dict
) -> str:
    """按工具类型获取对应节点的 ground truth。"""
    nodes = tree_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return ""

    node_id = ""
    if tool_name == "observe_frame":
        node_ids = tool_args.get("node_ids", [])
        if isinstance(node_ids, list) and node_ids:
            node_id = str(node_ids[0])
    else:
        node_id = str(tool_args.get("node_id", ""))
        if not node_id:
            node_ids = tool_args.get("node_ids", [])
            if isinstance(node_ids, list) and node_ids:
                node_id = str(node_ids[0])

    node = nodes.get(node_id, {})
    if not isinstance(node, dict):
        return ""
    return json.dumps(node.get("card", {}), ensure_ascii=False, sort_keys=True)


def compute_question_metrics(
    prediction,
    traces,
    tree_data,
    skill_content,
    judge_client,
    prompts_dir,
    max_steps,
    raw_contents=None,
) -> "QuestionMetrics":
    """编排单题规则指标与 LLM judge 指标。"""
    from core.harness.diagnose import QuestionMetrics

    if raw_contents is None:
        raw_contents = [
            str(step.get("tool_output", ""))
            for step in prediction.get("steps_json", [])
        ]

    rule_metrics_dict = extract_rule_metrics(prediction, raw_contents, max_steps)

    span_evals_list: list["SpanMetrics"] = []
    visited_node_ids: list[str] = []
    seen_node_ids: set[str] = set()

    for trace in traces:
        tool_name = trace.get("tool_name")
        tool_args = _parse_tool_args(trace.get("tool_args", {}))
        if tool_name in _SPAN_EVAL_TOOLS:
            span_evals_list.append(
                evaluate_span(
                    judge_client=judge_client,
                    prompts_dir=prompts_dir,
                    question=prediction.get("question", ""),
                    tool_name=str(tool_name),
                    tool_args=tool_args,
                    tool_output=str(trace.get("tool_output", "")),
                    ground_truth=_get_ground_truth_for_trace(
                        tree_data, str(tool_name), tool_args
                    ),
                    step=int(trace.get("step", 0)),
                )
            )

        if tool_name == "view_node":
            node_id = tool_args.get("node_id")
            if isinstance(node_id, str) and node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                visited_node_ids.append(node_id)

    all_tool_outputs = "\n".join(
        str(trace.get("tool_output", ""))
        for trace in traces
        if trace.get("tool_name") in _SPAN_EVAL_TOOLS
    )
    options_list = (
        prediction.get("options", "").split("\n")
        if isinstance(prediction.get("options"), str)
        else prediction.get("options", [])
    )
    trace_text = _format_trace_text(traces)
    tree_content = _load_tree_content(tree_data)

    missed_nodes_list = judge_missed_nodes(
        judge_client=judge_client,
        prompts_dir=prompts_dir,
        question=prediction.get("question", ""),
        options=options_list,
        answer=prediction.get("answer", ""),
        tree_content=tree_content,
        visited_node_ids=visited_node_ids,
    )
    skill_adherence_list = judge_skill_adherence(
        judge_client=judge_client,
        prompts_dir=prompts_dir,
        skill_content=skill_content,
        trace_text=trace_text,
    )
    has_bias, bias_evidence = judge_confirmation_bias(
        judge_client=judge_client,
        prompts_dir=prompts_dir,
        question=prediction.get("question", ""),
        options=options_list,
        trace_text=trace_text,
    )
    sufficient, reasoning = judge_evidence_sufficiency(
        judge_client=judge_client,
        prompts_dir=prompts_dir,
        question=prediction.get("question", ""),
        options=options_list,
        answer=prediction.get("answer", ""),
        all_tool_outputs=all_tool_outputs,
    )

    question_metrics = QuestionMetrics(
        question_id=prediction["question_id"],
        video_id=prediction["video_id"],
        task_type=prediction["task_type"],
        correct=bool(prediction.get("correct", False)),
        format_compliance=rule_metrics_dict["format_compliance"],
        budget_usage=rule_metrics_dict["budget_usage"],
        confidence_calibration=rule_metrics_dict["confidence_calibration"],
        repeat_visit_rate=rule_metrics_dict["repeat_visit_rate"],
        search_keyword_repetition=rule_metrics_dict["search_keyword_repetition"],
        level_jump_pattern=rule_metrics_dict["level_jump_pattern"],
        tool_usage=rule_metrics_dict["tool_usage"],
        span_metrics=span_evals_list,
        missed_nodes=missed_nodes_list,
        skill_adherence=skill_adherence_list,
        confirmation_bias=has_bias,
        evidence_sufficient=sufficient,
    )
    return question_metrics
