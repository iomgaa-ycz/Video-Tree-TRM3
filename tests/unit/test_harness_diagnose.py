"""diagnose 数据结构单元测试。"""

from __future__ import annotations

from core.harness.diagnose import DiagnosisResult, QuestionDiagnosis, SpanEvaluation


def test_span_evaluation_construction() -> None:
    evaluation = SpanEvaluation(
        step=1,
        tool_name="observe_frame",
        extraction_completeness=0.9,
        accuracy=0.8,
        score=0.85,
        missed_info="遗漏了人物动作",
        hallucinations="错误提到了红色汽车",
    )

    assert evaluation.step == 1
    assert evaluation.tool_name == "observe_frame"
    assert evaluation.extraction_completeness == 0.9
    assert evaluation.accuracy == 0.8
    assert evaluation.score == 0.85
    assert evaluation.missed_info == "遗漏了人物动作"
    assert evaluation.hallucinations == "错误提到了红色汽车"


def test_question_diagnosis_correct_case() -> None:
    diagnosis = QuestionDiagnosis(
        question_id="q1",
        video_id="video_001",
        task_type="temporal_reasoning",
        correct=True,
        effective_patterns=[{"pattern": "先定位关键帧"}],
        missed_nodes=["L2_003"],
        key_insight="关键帧检索有效",
    )

    assert diagnosis.question_id == "q1"
    assert diagnosis.video_id == "video_001"
    assert diagnosis.task_type == "temporal_reasoning"
    assert diagnosis.correct is True
    assert diagnosis.effective_patterns == [{"pattern": "先定位关键帧"}]
    assert diagnosis.missed_nodes == ["L2_003"]
    assert diagnosis.key_insight == "关键帧检索有效"


def test_question_diagnosis_incorrect_case_with_root_causes() -> None:
    diagnosis = QuestionDiagnosis(
        question_id="q2",
        video_id="video_002",
        task_type="counting",
        correct=False,
        root_causes=[{"stage": "retrieve", "reason": "漏掉关键节点"}],
        key_insight="召回不足导致错误",
    )

    assert diagnosis.correct is False
    assert diagnosis.root_causes == [{"stage": "retrieve", "reason": "漏掉关键节点"}]
    assert diagnosis.key_insight == "召回不足导致错误"


def test_question_diagnosis_default_values() -> None:
    diagnosis = QuestionDiagnosis(
        question_id="q3",
        video_id="video_003",
        task_type="grounding",
        correct=True,
    )

    assert diagnosis.root_causes == []
    assert diagnosis.effective_patterns == []
    assert diagnosis.missed_nodes == []
    assert diagnosis.key_insight == ""


def test_diagnosis_result_construction() -> None:
    question_diagnosis = QuestionDiagnosis(
        question_id="q4",
        video_id="video_004",
        task_type="spatial_reasoning",
        correct=False,
    )
    result = DiagnosisResult(
        run_id="run_001",
        question_diagnoses=[question_diagnosis],
        aggregations=[{"metric": "accuracy", "value": 0.5}],
    )

    assert result.run_id == "run_001"
    assert result.question_diagnoses == [question_diagnosis]
    assert result.aggregations == [{"metric": "accuracy", "value": 0.5}]


def test_diagnosis_result_default_values() -> None:
    result = DiagnosisResult(run_id="run_002")

    assert result.run_id == "run_002"
    assert result.question_diagnoses == []
    assert result.aggregations == []
