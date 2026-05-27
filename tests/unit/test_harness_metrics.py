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


def _make_mock_judge(response_json: dict):
    import json
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        response_json, ensure_ascii=False
    )
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    mock.chat.return_value = mock_response
    return mock


def _write_prompt_file(tmp_path, filename: str) -> None:
    (tmp_path / filename).write_text("你是评估器。", encoding="utf-8")


def test_evaluate_span(tmp_path) -> None:
    from core.harness.metrics import evaluate_span

    _write_prompt_file(tmp_path, "diagnose_span.md")
    judge = _make_mock_judge(
        {
            "extraction_completeness": 0.85,
            "hallucination_rate": 0.1,
            "missed_info_tags": ["subtitle_quote"],
            "hallucination_tags": ["wrong_attribute"],
        }
    )

    result = evaluate_span(
        judge_client=judge,
        prompts_dir=tmp_path,
        question="发生了什么？",
        tool_name="view_node",
        tool_args={"node_id": "n1"},
        tool_output="输出内容",
        ground_truth='{"caption":"真值"}',
        step=2,
    )

    assert result.tool_name == "view_node"
    assert result.step == 2
    assert result.extraction_completeness == pytest.approx(0.85)
    assert result.hallucination_rate == pytest.approx(0.1)
    assert result.missed_info_tags == ["subtitle_quote"]
    assert result.hallucination_tags == ["wrong_attribute"]


def test_judge_missed_nodes(tmp_path) -> None:
    from core.harness.metrics import judge_missed_nodes

    _write_prompt_file(tmp_path, "diagnose_missed_nodes.md")
    judge = _make_mock_judge({"missed_nodes": ["n1", "n2"]})

    result = judge_missed_nodes(
        judge_client=judge,
        prompts_dir=tmp_path,
        question="问题",
        options=["A", "B"],
        answer="A",
        tree_content="tree",
        visited_node_ids=["n0"],
    )

    assert result == ["n1", "n2"]


def test_judge_skill_adherence(tmp_path) -> None:
    from core.harness.metrics import judge_skill_adherence

    _write_prompt_file(tmp_path, "diagnose_skill_adherence.md")
    judge = _make_mock_judge(
        {
            "steps": [
                {
                    "step_label": "全局扫描",
                    "adhered": True,
                    "description": "Agent 扫描了 L1",
                },
            ]
        }
    )

    result = judge_skill_adherence(
        judge_client=judge,
        prompts_dir=tmp_path,
        skill_content="step1",
        trace_text="trace",
    )

    assert len(result) == 1
    assert result[0].step_label == "全局扫描"
    assert result[0].adhered is True
    assert result[0].description == "Agent 扫描了 L1"


def test_judge_confirmation_bias(tmp_path) -> None:
    from core.harness.metrics import judge_confirmation_bias

    _write_prompt_file(tmp_path, "diagnose_confirmation_bias.md")
    judge = _make_mock_judge({"has_bias": True, "evidence": "bias found"})

    result = judge_confirmation_bias(
        judge_client=judge,
        prompts_dir=tmp_path,
        question="问题",
        options=["A", "B"],
        trace_text="trace",
    )

    assert result == (True, "bias found")


def test_judge_evidence_sufficiency(tmp_path) -> None:
    from core.harness.metrics import judge_evidence_sufficiency

    _write_prompt_file(tmp_path, "diagnose_evidence_sufficiency.md")
    judge = _make_mock_judge({"sufficient": False, "reasoning": "not enough"})

    result = judge_evidence_sufficiency(
        judge_client=judge,
        prompts_dir=tmp_path,
        question="问题",
        options=["A", "B"],
        answer="A",
        all_tool_outputs="out1\nout2",
    )

    assert result == (False, "not enough")


def test_compute_question_metrics(tmp_path) -> None:
    import json
    from unittest.mock import MagicMock

    from core.harness.metrics import compute_question_metrics

    for filename in (
        "diagnose_span.md",
        "diagnose_missed_nodes.md",
        "diagnose_skill_adherence.md",
        "diagnose_confirmation_bias.md",
        "diagnose_evidence_sufficiency.md",
    ):
        _write_prompt_file(tmp_path, filename)

    responses = [
        {
            "extraction_completeness": 0.9,
            "hallucination_rate": 0.1,
            "missed_info_tags": [],
            "hallucination_tags": [],
        },
        {
            "extraction_completeness": 0.7,
            "hallucination_rate": 0.2,
            "missed_info_tags": ["entity"],
            "hallucination_tags": [],
        },
        {"missed_nodes": ["n3"]},
        {
            "steps": [
                {"step_label": "先看节点", "adhered": True, "description": "已执行"}
            ]
        },
        {"has_bias": False, "evidence": ""},
        {"sufficient": True, "reasoning": "证据充分"},
    ]
    judge = MagicMock()
    judge.chat.side_effect = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content=json.dumps(item, ensure_ascii=False))
                )
            ],
            usage=MagicMock(prompt_tokens=100, completion_tokens=50),
        )
        for item in responses
    ]

    prediction = {
        "question_id": "q1",
        "video_id": "v1",
        "task_type": "mcq",
        "question": "主角在哪个节点出现？",
        "options": "A. n1\nB. n2",
        "prediction": "A",
        "answer": "A",
        "correct": True,
        "steps_json": [
            {
                "tool_call": {"tool": "view_node", "args": {"node_id": "n1"}},
                "tool_output": '{"reflect":"r1","plan":"p1","action":"a1"}',
            },
            {
                "tool_call": {"tool": "search_similar", "args": {"query": "主角出现"}},
                "tool_output": '{"reflect":{"confidence":0.88},"plan":"p2","action":"a2"}',
            },
        ],
    }
    traces = [
        {
            "step": 1,
            "tool_name": "view_node",
            "tool_args": json.dumps({"node_id": "n1"}, ensure_ascii=False),
            "tool_output": "节点 n1 输出",
            "thought": "先看关键节点",
        },
        {
            "step": 2,
            "tool_name": "search_similar",
            "tool_args": json.dumps(
                {"node_id": "n2", "query": "主角出现"}, ensure_ascii=False
            ),
            "tool_output": "节点 n2 输出",
            "thought": "再搜索相似节点",
        },
    ]
    tree_data = {
        "nodes": {
            "n1": {
                "level": 1,
                "time_range": [0, 5],
                "card": {"summary": "节点1"},
            },
            "n2": {
                "level": 2,
                "time_range": [5, 10],
                "card": {"summary": "节点2"},
            },
        }
    }

    result = compute_question_metrics(
        prediction=prediction,
        traces=traces,
        tree_data=tree_data,
        skill_content="1. 先看节点\n2. 再核验证据",
        judge_client=judge,
        prompts_dir=tmp_path,
        max_steps=4,
    )

    assert result.question_id == "q1"
    assert result.video_id == "v1"
    assert result.task_type == "mcq"
    assert result.correct is True
    assert result.budget_usage == pytest.approx(0.5)
    assert len(result.span_metrics) == 2
    assert result.span_metrics[0].extraction_completeness == pytest.approx(0.9)
    assert result.span_metrics[1].missed_info_tags == ["entity"]
    assert result.missed_nodes == ["n3"]
    assert len(result.skill_adherence) == 1
    assert result.skill_adherence[0].step_label == "先看节点"
    assert result.confirmation_bias is False
    assert result.evidence_sufficient is True
