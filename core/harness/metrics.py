"""Stage 1 指标计算 — 规则指标与 LLM judge 编排。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from json_repair import repair_json


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
