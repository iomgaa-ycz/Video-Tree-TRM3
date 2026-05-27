"""`core.harness.diagnose` 数据结构与聚合单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

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
    _build_skill_case_packs,
    _build_system_case_pack,
    _build_tool_case_packs,
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


# ---------------------------------------------------------------------------
# 案例包构建函数测试
# ---------------------------------------------------------------------------


def test_build_skill_case_packs_basic() -> None:
    """基本场景：2 个 error_type 各 2+ 题，应各取 top 2；成功案例按比例。"""
    metrics = [
        _make_question_metrics(
            question_id="q_sf_1",
            correct=False,
            missed_nodes=["L2_a", "L2_b", "L3_c"],
            budget_usage=1.0,
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", True, "ok")],
        ),
        _make_question_metrics(
            question_id="q_sf_2",
            correct=False,
            missed_nodes=["L2_a", "L3_b"],
            budget_usage=0.8,
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", True, "ok")],
        ),
        _make_question_metrics(
            question_id="q_sf_3",
            correct=False,
            missed_nodes=["L3_a"],
            budget_usage=0.6,
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", False, "no")],
        ),
        _make_question_metrics(
            question_id="q_rf_1",
            correct=False,
            missed_nodes=[],
            evidence_sufficient=True,
            budget_usage=0.9,
            confidence_calibration="high_conf_wrong",
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", True, "ok")],
        ),
        _make_question_metrics(
            question_id="q_rf_2",
            correct=False,
            missed_nodes=[],
            evidence_sufficient=True,
            budget_usage=0.5,
            confidence_calibration="calibrated",
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", True, "ok")],
        ),
        _make_question_metrics(
            question_id="q_ok_1",
            correct=True,
            budget_usage=0.4,
            task_type="Temporal Reasoning",
            skill_adherence=[SkillStepAdherence("s1", True, "ok")],
        ),
        _make_question_metrics(
            question_id="q_ok_2",
            correct=True,
            budget_usage=0.6,
            task_type="Temporal Reasoning",
            skill_adherence=[
                SkillStepAdherence("s1", True, "ok"),
                SkillStepAdherence("s2", False, "no"),
            ],
        ),
        _make_question_metrics(
            question_id="q_ok_3",
            correct=True,
            budget_usage=0.3,
            task_type="Temporal Reasoning",
            skill_adherence=[
                SkillStepAdherence("s1", True, "ok"),
                SkillStepAdherence("s2", True, "ok"),
            ],
        ),
    ]
    attributions = [
        ErrorAttribution("q_sf_1", "search_failure", None),
        ErrorAttribution("q_sf_2", "search_failure", None),
        ErrorAttribution("q_sf_3", "search_failure", None),
        ErrorAttribution("q_rf_1", "reasoning_failure", None),
        ErrorAttribution("q_rf_2", "reasoning_failure", None),
    ]
    traces = {(m.video_id, m.question_id): [] for m in metrics}
    predictions = [
        {
            "video_id": m.video_id,
            "question_id": m.question_id,
            "task_type": m.task_type,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "prediction": "A" if m.correct else "B",
        }
        for m in metrics
    ]

    packs = _build_skill_case_packs(
        all_metrics=metrics,
        error_attributions=attributions,
        traces_by_question=traces,
        predictions=predictions,
        d3_stats={},
        d4_stats={},
    )

    assert "Temporal Reasoning" in packs
    pack = packs["Temporal Reasoning"]

    assert len(pack.failure_cases) == 4
    failure_ids = [c.question_id for c in pack.failure_cases]
    assert "q_sf_1" in failure_ids
    assert "q_sf_2" in failure_ids
    assert "q_rf_1" in failure_ids

    assert len(pack.success_cases) == 2
    success_ids = [c.question_id for c in pack.success_cases]
    assert "q_ok_3" in success_ids


def test_build_skill_case_packs_low_accuracy() -> None:
    """accuracy <= 30% 时成功案例放宽标准。"""
    metrics = []
    attributions = []
    for i in range(8):
        m = _make_question_metrics(
            question_id=f"q_wrong_{i}",
            correct=False,
            missed_nodes=["L2_x"] if i < 4 else [],
            evidence_sufficient=(i >= 4),
            task_type="OCR Problems",
            skill_adherence=[],
        )
        metrics.append(m)
        error_type = "search_failure" if i < 4 else "reasoning_failure"
        attributions.append(ErrorAttribution(f"q_wrong_{i}", error_type, None))

    for i in range(2):
        metrics.append(
            _make_question_metrics(
                question_id=f"q_right_{i}",
                correct=True,
                task_type="OCR Problems",
                skill_adherence=[],
            )
        )

    traces = {(m.video_id, m.question_id): [] for m in metrics}
    predictions = [
        {
            "video_id": m.video_id,
            "question_id": m.question_id,
            "task_type": m.task_type,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "prediction": "A" if m.correct else "B",
        }
        for m in metrics
    ]

    packs = _build_skill_case_packs(
        all_metrics=metrics,
        error_attributions=attributions,
        traces_by_question=traces,
        predictions=predictions,
        d3_stats={},
        d4_stats={},
    )

    pack = packs["OCR Problems"]
    assert len(pack.success_cases) >= 2
    for case in pack.success_cases:
        assert "low_accuracy_pool" in case.selection_reason


def test_build_system_case_pack_basic() -> None:
    """行为模式达到 min_pattern_count 时应产出案例。"""
    metrics = []
    for i in range(4):
        metrics.append(
            _make_question_metrics(
                question_id=f"q_early_{i}",
                correct=False,
                budget_usage=0.1 + i * 0.05,
                confidence_calibration="calibrated",
                confirmation_bias=False,
            )
        )
    for i in range(2):
        metrics.append(
            _make_question_metrics(
                question_id=f"q_hcw_{i}",
                correct=False,
                budget_usage=0.7,
                confidence_calibration="high_conf_wrong",
                confirmation_bias=False,
            )
        )
    for i in range(3):
        metrics.append(
            _make_question_metrics(
                question_id=f"q_good_{i}",
                correct=True,
                budget_usage=0.45 + i * 0.05,
                confidence_calibration="calibrated",
                confirmation_bias=False,
            )
        )

    traces = {(m.video_id, m.question_id): [] for m in metrics}
    predictions = [
        {
            "video_id": m.video_id,
            "question_id": m.question_id,
            "task_type": m.task_type,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "prediction": "A" if m.correct else "B",
        }
        for m in metrics
    ]

    pack = _build_system_case_pack(
        all_metrics=metrics,
        traces_by_question=traces,
        predictions=predictions,
        d5_stats={"early_submit_rate": 0.4},
    )

    assert pack is not None
    assert len(pack.failure_cases) == 2
    failure_ids = [c.question_id for c in pack.failure_cases]
    assert failure_ids[0] == "q_early_0"
    assert failure_ids[1] == "q_early_1"
    assert len(pack.success_cases) >= 2


def test_build_system_case_pack_returns_none() -> None:
    """所有行为模式都不足阈值时应返回 None。"""
    metrics = [
        _make_question_metrics(question_id="q_1", correct=False, budget_usage=0.5),
        _make_question_metrics(question_id="q_2", correct=True, budget_usage=0.5),
    ]
    traces = {(m.video_id, m.question_id): [] for m in metrics}
    predictions = [
        {
            "video_id": m.video_id,
            "question_id": m.question_id,
            "task_type": m.task_type,
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "answer": "A",
            "prediction": "A" if m.correct else "B",
        }
        for m in metrics
    ]

    pack = _build_system_case_pack(
        all_metrics=metrics,
        traces_by_question=traces,
        predictions=predictions,
        d5_stats={},
    )
    assert pack is None


def test_build_tool_case_packs_basic() -> None:
    """从 span_evaluations 和 traces 中抽取工具级案例。"""
    mock_log = MagicMock()
    mock_log.query.return_value = [
        {
            "video_id": "v1",
            "question_id": "q1",
            "step": 0,
            "tool_name": "view_node",
            "extraction_completeness": 0.3,
            "hallucination_rate": 0.6,
            "missed_tags_json": '["entity"]',
            "hallucinated_tags_json": '["fabricated_action"]',
        },
        {
            "video_id": "v1",
            "question_id": "q1",
            "step": 1,
            "tool_name": "view_node",
            "extraction_completeness": 0.5,
            "hallucination_rate": 0.4,
            "missed_tags_json": '["subtitle_quote"]',
            "hallucinated_tags_json": "[]",
        },
        {
            "video_id": "v1",
            "question_id": "q2",
            "step": 0,
            "tool_name": "view_node",
            "extraction_completeness": 0.95,
            "hallucination_rate": 0.0,
            "missed_tags_json": "[]",
            "hallucinated_tags_json": "[]",
        },
    ]

    traces_by_question = {
        ("v1", "q1"): [
            {
                "step": 0,
                "tool_name": "view_node",
                "tool_args": '{"node_id": "L1_000", "question": "Q?"}',
                "tool_output": "some output step 0",
            },
            {
                "step": 1,
                "tool_name": "view_node",
                "tool_args": '{"node_id": "L2_001", "question": "Q?"}',
                "tool_output": "some output step 1",
            },
        ],
        ("v1", "q2"): [
            {
                "step": 0,
                "tool_name": "view_node",
                "tool_args": '{"node_id": "L1_001", "question": "Q?"}',
                "tool_output": "good output",
            },
        ],
    }

    tree_cache = {
        "v1": {
            "nodes": {
                "L1_000": {
                    "card": {"scene_summary": "intro"},
                    "level": 1,
                    "time_range": [0, 300],
                },
                "L2_001": {
                    "card": {"event_description": "event"},
                    "level": 2,
                    "time_range": [0, 30],
                },
                "L1_001": {
                    "card": {"scene_summary": "outro"},
                    "level": 1,
                    "time_range": [300, 600],
                },
            }
        },
    }

    packs = _build_tool_case_packs(
        log=mock_log,
        run_id="test_run",
        traces_by_question=traces_by_question,
        d2_stats={"view_node": {"avg_completeness": 0.58, "n_calls": 3}},
        tree_cache=tree_cache,
    )

    assert "view_node" in packs
    vn_pack = packs["view_node"]
    assert vn_pack.tool_name == "view_node"
    assert "view_node_extract.md" in vn_pack.target_files
    assert len(vn_pack.failure_spans) >= 2
    assert len(vn_pack.success_spans) >= 1
    assert vn_pack.success_spans[0]["extraction_completeness"] == 0.95
