"""`core.harness.metrics` 单元测试。"""

from __future__ import annotations

import pytest

from core.harness.metrics import (
    calc_budget_usage,
    calc_confidence_calibration,
    calc_format_compliance,
    calc_level_jump_pattern,
    calc_repeat_visit_rate,
    calc_search_keyword_repetition,
    calc_tool_usage,
    extract_json_from_response,
    extract_rule_metrics,
    load_diagnose_prompt,
)


def test_format_compliance_all_compliant() -> None:
    raw = ["{reflect:a,plan:b,action:c}", "{reflect:x,plan:y,action:z}"]
    assert calc_format_compliance(raw) == 1.0


def test_format_compliance_partial() -> None:
    raw = ["{reflect:a,plan:b,action:c}", "{reflect:x}"]
    assert calc_format_compliance(raw) == 0.5


def test_format_compliance_empty() -> None:
    assert calc_format_compliance([]) == 1.0


def test_budget_usage() -> None:
    assert calc_budget_usage(3, 10) == pytest.approx(0.3)


def test_confidence_calibration_high_conf_wrong() -> None:
    assert calc_confidence_calibration(0.8, False) == "high_conf_wrong"


def test_confidence_calibration_low_conf_right() -> None:
    assert calc_confidence_calibration(0.3, True) == "low_conf_right"


def test_confidence_calibration_calibrated_correct() -> None:
    assert calc_confidence_calibration(0.8, True) == "calibrated"


def test_confidence_calibration_boundary() -> None:
    assert calc_confidence_calibration(0.6, False) == "calibrated"


def test_repeat_visit_rate_no_repeat() -> None:
    assert calc_repeat_visit_rate(["a", "b", "c"]) == pytest.approx(0.0)


def test_repeat_visit_rate_with_repeat() -> None:
    ids = ["a", "b", "a", "c", "b"]
    assert calc_repeat_visit_rate(ids) == pytest.approx(0.4)


def test_repeat_visit_rate_empty() -> None:
    assert calc_repeat_visit_rate([]) == 0.0


def test_search_keyword_repetition_empty() -> None:
    assert calc_search_keyword_repetition([]) == 0.0


def test_search_keyword_repetition_single() -> None:
    assert calc_search_keyword_repetition(["hello world"]) == 0.0


def test_search_keyword_repetition_identical() -> None:
    query = "the quick brown fox"
    assert calc_search_keyword_repetition([query, query]) == pytest.approx(1.0)


def test_search_keyword_repetition_different() -> None:
    assert calc_search_keyword_repetition(["abc", "xyz"]) < 0.5


def test_level_jump_pattern_normal() -> None:
    ids = ["seg_L1_001", "seg_L2_003", "seg_L3_007"]
    assert calc_level_jump_pattern(ids) == "L1→L2→L3"


def test_level_jump_pattern_empty() -> None:
    assert calc_level_jump_pattern([]) == ""


def test_tool_usage() -> None:
    names = ["view_node", "view_node", "search_similar", "submit_answer"]
    result = calc_tool_usage(names)
    assert result == {"view_node": 2, "search_similar": 1, "submit_answer": 1}


def test_extract_rule_metrics_full() -> None:
    prediction = {
        "steps_json": [
            {
                "tool_call": {
                    "tool": "view_node",
                    "args": {"node_id": "seg_L1_001"},
                }
            },
            {
                "tool_call": {
                    "tool": "search_similar",
                    "args": {"query": "red car driving"},
                }
            },
            {
                "tool_call": {
                    "tool": "view_node",
                    "args": {"node_id": "seg_L2_005"},
                }
            },
            {"tool_call": {"tool": "submit_answer", "args": {"answer": "yes"}}},
        ],
        "correct": True,
    }
    raw_contents = [
        '{"reflect":"ok","plan":"go","action":"view"}',
        '{"reflect":"ok","plan":"go","action":"search"}',
        '{"reflect":"ok","plan":"go","action":"view"}',
        '{"reflect":{"confidence":0.85},"plan":"done","action":"submit"}',
    ]

    result = extract_rule_metrics(prediction, raw_contents, max_steps=10)

    assert result["format_compliance"] == 1.0
    assert result["budget_usage"] == pytest.approx(0.4)
    assert result["repeat_visit_rate"] == pytest.approx(0.0)
    assert result["tool_usage"]["view_node"] == 2
    assert result["level_jump_pattern"] == "L1→L2"


def test_extract_json_from_response_plain() -> None:
    raw = '{"key": "value"}'
    assert extract_json_from_response(raw) == {"key": "value"}


def test_extract_json_from_response_markdown() -> None:
    raw = 'Here is result:\n```json\n{"score": 9}\n```'
    assert extract_json_from_response(raw) == {"score": 9}


def test_extract_json_from_response_embedded() -> None:
    raw = 'Analysis done. {"verdict": "pass"} end.'
    assert extract_json_from_response(raw) == {"verdict": "pass"}


def test_load_diagnose_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "test_prompt.md"
    prompt_file.write_text("hello prompt", encoding="utf-8")
    assert load_diagnose_prompt(tmp_path, "test_prompt.md") == "hello prompt"
