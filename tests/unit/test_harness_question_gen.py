from __future__ import annotations

from core.harness.question_gen import GeneratedQuestion, QuestionGenResult


def test_generated_question_full_construction() -> None:
    """测试 GeneratedQuestion 显式传入全部字段时的构造结果。"""

    question = GeneratedQuestion(
        question_id="q-001",
        video_id="video-001",
        task_type="multiple_choice",
        question="视频里主角先做了什么？",
        options={"A": "开门", "B": "坐下", "C": "起身", "D": "离开"},
        answer="A",
        source_nodes=["node-1", "node-2"],
        difficulty="hard",
    )

    assert question.question_id == "q-001"
    assert question.video_id == "video-001"
    assert question.task_type == "multiple_choice"
    assert question.question == "视频里主角先做了什么？"
    assert question.options == {
        "A": "开门",
        "B": "坐下",
        "C": "起身",
        "D": "离开",
    }
    assert question.answer == "A"
    assert question.source_nodes == ["node-1", "node-2"]
    assert question.difficulty == "hard"


def test_generated_question_defaults() -> None:
    """测试 GeneratedQuestion 仅传必填字段时的默认值。"""

    question = GeneratedQuestion(
        question_id="q-002",
        video_id="video-002",
        task_type="multiple_choice",
        question="视频结尾发生了什么？",
        options={"A": "关灯", "B": "开灯"},
        answer="B",
    )

    assert question.source_nodes == []
    assert question.difficulty == "medium"


def test_question_gen_result_full_construction() -> None:
    """测试 QuestionGenResult 显式传入全部字段时的构造结果。"""

    result = QuestionGenResult(
        version="v1.0.0",
        total=6,
        per_task_type={"multiple_choice": 4, "boolean": 2},
        per_video={"video-001": 3, "video-002": 3},
    )

    assert result.version == "v1.0.0"
    assert result.total == 6
    assert result.per_task_type == {"multiple_choice": 4, "boolean": 2}
    assert result.per_video == {"video-001": 3, "video-002": 3}


def test_question_gen_result_defaults() -> None:
    """测试 QuestionGenResult 仅传必填字段时的默认值。"""

    result = QuestionGenResult(version="v1.0.1", total=0)

    assert result.per_task_type == {}
    assert result.per_video == {}
