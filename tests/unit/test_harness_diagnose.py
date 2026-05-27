"""`core.harness.diagnose` 数据结构与聚合单元测试。"""

from __future__ import annotations

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
    aggregate_d2,
    aggregate_d5,
    attribute_error,
)


def _make_question_metrics(**overrides) -> QuestionMetrics:
    """构造测试用 QuestionMetrics。"""
    defaults = {
        "question_id": "q_001",
        "video_id": "video_001",
        "task_type": "temporal_reasoning",
        "correct": False,
        "format_compliance": 1.0,
        "budget_usage": 0.6,
        "confidence_calibration": "calibrated",
        "repeat_visit_rate": 0.2,
        "search_keyword_repetition": 0.1,
        "level_jump_pattern": "L1→L2",
        "tool_usage": {"view_node": 2},
        "span_metrics": [],
        "missed_nodes": [],
        "skill_adherence": [],
        "confirmation_bias": False,
        "evidence_sufficient": False,
    }
    defaults.update(overrides)
    return QuestionMetrics(**defaults)


def test_span_metrics_construction_and_defaults() -> None:
    metrics = SpanMetrics(
        step=2,
        tool_name="search_nodes",
        extraction_completeness=0.92,
        hallucination_rate=0.08,
    )

    assert metrics.step == 2
    assert metrics.tool_name == "search_nodes"
    assert metrics.extraction_completeness == 0.92
    assert metrics.hallucination_rate == 0.08
    assert metrics.missed_info_tags == []
    assert metrics.hallucination_tags == []


def test_skill_step_adherence_construction() -> None:
    adherence = SkillStepAdherence(
        step_label="step_1_locate_key_frame",
        adhered=True,
        description="先定位关键帧后再做细节确认。",
    )

    assert adherence.step_label == "step_1_locate_key_frame"
    assert adherence.adhered is True
    assert adherence.description == "先定位关键帧后再做细节确认。"


def test_question_metrics_construction() -> None:
    span_metrics = [
        SpanMetrics(
            step=1,
            tool_name="observe_frame",
            extraction_completeness=0.9,
            hallucination_rate=0.1,
            missed_info_tags=["actor_motion"],
            hallucination_tags=["vehicle_color"],
        )
    ]
    skill_adherence = [
        SkillStepAdherence(
            step_label="step_2_verify_evidence",
            adhered=False,
            description="跳过了交叉验证。",
        )
    ]

    metrics = QuestionMetrics(
        question_id="q_001",
        video_id="video_001",
        task_type="temporal_reasoning",
        correct=False,
        format_compliance=0.95,
        budget_usage=0.73,
        confidence_calibration="high_conf_wrong",
        repeat_visit_rate=0.25,
        search_keyword_repetition=0.4,
        level_jump_pattern="L1→L3→L2",
        tool_usage={"search_nodes": 3, "observe_frame": 2},
        span_metrics=span_metrics,
        missed_nodes=["node_7", "node_9"],
        skill_adherence=skill_adherence,
        confirmation_bias=True,
        evidence_sufficient=False,
    )

    assert metrics.question_id == "q_001"
    assert metrics.video_id == "video_001"
    assert metrics.task_type == "temporal_reasoning"
    assert metrics.correct is False
    assert metrics.format_compliance == 0.95
    assert metrics.budget_usage == 0.73
    assert metrics.confidence_calibration == "high_conf_wrong"
    assert metrics.repeat_visit_rate == 0.25
    assert metrics.search_keyword_repetition == 0.4
    assert metrics.level_jump_pattern == "L1→L3→L2"
    assert metrics.tool_usage == {"search_nodes": 3, "observe_frame": 2}
    assert metrics.span_metrics == span_metrics
    assert metrics.missed_nodes == ["node_7", "node_9"]
    assert metrics.skill_adherence == skill_adherence
    assert metrics.confirmation_bias is True
    assert metrics.evidence_sufficient is False


def test_error_attribution_construction() -> None:
    attribution = ErrorAttribution(
        question_id="q_002",
        error_type="retrieval_failure",
        reasoning_failure_type="premature_conclusion",
    )

    assert attribution.question_id == "q_002"
    assert attribution.error_type == "retrieval_failure"
    assert attribution.reasoning_failure_type == "premature_conclusion"


def test_diagnosis_result_construction_and_defaults() -> None:
    attribution = ErrorAttribution(
        question_id="q_003",
        error_type="search_strategy",
        reasoning_failure_type=None,
    )

    result = DiagnosisResult(run_id="run_001", error_attributions=[attribution])

    assert result.run_id == "run_001"
    assert result.filter_summary == {}
    assert result.error_attributions == [attribution]
    assert result.attribution_distribution == {}
    assert result.reasoning_failure_types == {}
    assert result.tool_quality == {}
    assert result.search_effectiveness == {}
    assert result.skill_compliance == {}
    assert result.decision_patterns == {}


def test_attribute_error_extraction_failure() -> None:
    metrics = _make_question_metrics(
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.4,
                hallucination_rate=0.2,
            )
        ]
    )

    result = attribute_error(metrics)

    assert result.error_type == "extraction_failure"
    assert result.reasoning_failure_type is None


def test_attribute_error_search_failure() -> None:
    metrics = _make_question_metrics(
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.9,
                hallucination_rate=0.1,
            )
        ],
        missed_nodes=["node_L2_3"],
    )

    assert attribute_error(metrics).error_type == "search_failure"


def test_attribute_error_reasoning_failure() -> None:
    metrics = _make_question_metrics(
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.9,
                hallucination_rate=0.1,
            )
        ],
        evidence_sufficient=True,
    )

    assert attribute_error(metrics).error_type == "reasoning_failure"


def test_attribute_error_mixed() -> None:
    metrics = _make_question_metrics(
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.9,
                hallucination_rate=0.1,
            )
        ],
        evidence_sufficient=False,
        missed_nodes=[],
    )

    assert attribute_error(metrics).error_type == "mixed"


def test_aggregate_d2() -> None:
    metrics_a = _make_question_metrics(
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.8,
                hallucination_rate=0.1,
                missed_info_tags=["time"],
                hallucination_tags=["color"],
            ),
            SpanMetrics(
                step=2,
                tool_name="search_similar",
                extraction_completeness=0.6,
                hallucination_rate=0.2,
                missed_info_tags=["actor"],
                hallucination_tags=[],
            ),
        ]
    )
    metrics_b = _make_question_metrics(
        question_id="q_002",
        span_metrics=[
            SpanMetrics(
                step=1,
                tool_name="view_node",
                extraction_completeness=0.4,
                hallucination_rate=0.5,
                missed_info_tags=["time"],
                hallucination_tags=["color", "object"],
            )
        ],
    )

    result = aggregate_d2([metrics_a, metrics_b])

    assert set(result) == {"view_node", "search_similar"}
    assert abs(result["view_node"]["avg_completeness"] - 0.6) < 1e-9
    assert abs(result["view_node"]["avg_hallucination"] - 0.3) < 1e-9
    assert result["view_node"]["n_calls"] == 2
    assert result["view_node"]["top_missed"][0] == ["time", 2]
    assert result["view_node"]["top_hallucinated"][0] == ["color", 2]
    assert result["search_similar"]["n_calls"] == 1


def test_aggregate_d5() -> None:
    metrics = [
        _make_question_metrics(
            question_id="q1",
            task_type="type_a",
            correct=False,
            format_compliance=1.0,
            budget_usage=0.2,
            confidence_calibration="high_conf_wrong",
            confirmation_bias=True,
        ),
        _make_question_metrics(
            question_id="q2",
            task_type="type_a",
            correct=True,
            format_compliance=0.5,
            budget_usage=0.5,
            confidence_calibration="low_conf_right",
            confirmation_bias=False,
        ),
        _make_question_metrics(
            question_id="q3",
            task_type="type_b",
            correct=False,
            format_compliance=0.0,
            budget_usage=0.8,
            confidence_calibration="calibrated",
            confirmation_bias=True,
        ),
    ]

    result = aggregate_d5(metrics)

    assert result["format_compliance_rate"] == 0.5
    assert result["budget_usage_median"] == 0.5
    assert result["budget_usage_p25"] == 0.35
    assert result["budget_usage_p75"] == 0.65
    assert result["early_submit_rate"] == 0.5
    assert result["high_conf_wrong_rate"] == 1 / 3
    assert result["low_conf_right_rate"] == 1 / 3
    assert result["confirmation_bias_rate"] == 2 / 3
    assert result["per_type_bias"] == {"type_a": 0.5, "type_b": 1.0}


# ---------------------------------------------------------------------------
# 案例包数据结构测试
# ---------------------------------------------------------------------------


def test_case_sample_construction() -> None:
    """CaseSample 应能正常构造并保存全部字段。"""
    sample = CaseSample(
        question_id="q_001",
        video_id="video_001",
        task_type="Temporal Reasoning",
        question="What happened first?",
        options=["A. X", "B. Y", "C. Z", "D. W"],
        answer="A",
        prediction="B",
        correct=False,
        error_type="search_failure",
        selection_reason="missed_nodes=3, budget_usage=1.0",
        metrics={"budget_usage": 1.0, "missed_nodes": ["L2_001", "L2_002", "L3_005"]},
        trace=[
            {"step": 0, "tool_name": "view_node", "tool_args": {"node_id": "L1_000"}}
        ],
    )
    assert sample.question_id == "q_001"
    assert sample.error_type == "search_failure"
    assert len(sample.trace) == 1


def test_skill_case_pack_construction() -> None:
    """SkillCasePack 应能正常构造，默认列表为空。"""
    pack = SkillCasePack(
        task_type="Temporal Reasoning",
        target_file="temporal-reasoning.md",
        stats={"n_total": 60, "accuracy": 0.5},
        failure_cases=[],
        success_cases=[],
    )
    assert pack.task_type == "Temporal Reasoning"
    assert pack.target_file == "temporal-reasoning.md"
    assert pack.failure_cases == []


def test_system_case_pack_construction() -> None:
    """SystemCasePack 应能正常构造。"""
    pack = SystemCasePack(
        stats={"early_submit_rate": 0.15, "early_submit_count": 12},
        failure_cases=[],
        success_cases=[],
    )
    assert pack.stats["early_submit_count"] == 12


def test_tool_case_pack_construction() -> None:
    """ToolCasePack 应能正常构造。"""
    pack = ToolCasePack(
        tool_name="view_node",
        target_files=["view_node_extract.md", "view_node_verify.md"],
        stats={"avg_completeness": 0.78, "n_calls": 500},
        failure_spans=[],
        success_spans=[],
    )
    assert pack.tool_name == "view_node"
    assert len(pack.target_files) == 2


def test_diagnosis_result_case_pack_defaults() -> None:
    """DiagnosisResult 新字段的默认值应允许无案例包构造。"""
    result = DiagnosisResult(run_id="test_run")
    assert result.skill_case_packs == {}
    assert result.system_case_pack is None
    assert result.tool_case_packs == {}


def test_diagnosis_result_with_case_packs() -> None:
    """DiagnosisResult 应能携带案例包。"""
    skill_pack = SkillCasePack(
        task_type="Temporal Reasoning",
        target_file="temporal-reasoning.md",
        stats={},
    )
    result = DiagnosisResult(
        run_id="test_run",
        skill_case_packs={"Temporal Reasoning": skill_pack},
    )
    assert "Temporal Reasoning" in result.skill_case_packs
    assert (
        result.skill_case_packs["Temporal Reasoning"].target_file
        == "temporal-reasoning.md"
    )
