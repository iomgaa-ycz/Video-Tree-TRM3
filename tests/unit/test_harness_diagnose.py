"""`core.harness.diagnose` 数据结构单元测试。"""

from __future__ import annotations

from core.harness.diagnose import (
    DiagnosisResult,
    ErrorAttribution,
    QuestionMetrics,
    SkillStepAdherence,
    SpanMetrics,
)


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
        confidence_calibration="overconfident",
        repeat_visit_rate=0.25,
        search_keyword_repetition=0.4,
        level_jump_pattern="L1->L3->L2",
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
    assert metrics.confidence_calibration == "overconfident"
    assert metrics.repeat_visit_rate == 0.25
    assert metrics.search_keyword_repetition == 0.4
    assert metrics.level_jump_pattern == "L1->L3->L2"
    assert metrics.tool_usage == {"search_nodes": 3, "observe_frame": 2}
    assert metrics.span_metrics == span_metrics
    assert metrics.missed_nodes == ["node_7", "node_9"]
    assert metrics.skill_adherence == skill_adherence
    assert metrics.confirmation_bias is True
    assert metrics.evidence_sufficient is False


def test_error_attribution_construction() -> None:
    attribution = ErrorAttribution(
        question_id="q_002",
        primary_cause="retrieval_failure",
        reasoning_failure_type="premature_conclusion",
    )

    assert attribution.question_id == "q_002"
    assert attribution.primary_cause == "retrieval_failure"
    assert attribution.reasoning_failure_type == "premature_conclusion"


def test_diagnosis_result_construction_and_defaults() -> None:
    attribution = ErrorAttribution(
        question_id="q_003",
        primary_cause="search_strategy",
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
