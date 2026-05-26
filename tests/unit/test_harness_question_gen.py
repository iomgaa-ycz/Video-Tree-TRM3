from __future__ import annotations

import json
from pathlib import Path

from core.harness.question_gen import (
    GeneratedQuestion,
    QuestionGenResult,
    load_benchmark,
)


def _write_benchmark_file(path: Path, qa_list: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(qa_list, ensure_ascii=False), encoding="utf-8")


def test_generated_question_full_construction() -> None:
    """测试 GeneratedQuestion 显式传入全部字段时的构造结果。"""

    question = GeneratedQuestion(
        question_id="q-001",
        video_id="video-001",
        task_type="multiple_choice",
        question="视频里主角先做了什么？",
        options=["A. 开门", "B. 坐下", "C. 起身", "D. 离开"],
        answer="A",
        source_nodes=["node-1", "node-2"],
        difficulty="hard",
    )

    assert question.question_id == "q-001"
    assert question.video_id == "video-001"
    assert question.task_type == "multiple_choice"
    assert question.question == "视频里主角先做了什么？"
    assert question.options == ["A. 开门", "B. 坐下", "C. 起身", "D. 离开"]
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
        options=["A. 关灯", "B. 开灯"],
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


def test_load_benchmark_single_video(tmp_path: Path) -> None:
    _write_benchmark_file(
        tmp_path / "717.json",
        [
            {
                "question_id": "717-1",
                "task_type": "Information Synopsis",
                "question": "发生了什么？",
                "options": ["A. 甲", "B. 乙", "C. 丙", "D. 丁"],
                "answer": "C",
            }
        ],
    )

    questions = load_benchmark(tmp_path)

    assert len(questions) == 1
    assert questions[0] == GeneratedQuestion(
        question_id="717-1",
        video_id="717",
        task_type="Information Synopsis",
        question="发生了什么？",
        options=["A. 甲", "B. 乙", "C. 丙", "D. 丁"],
        answer="C",
    )


def test_load_benchmark_multiple_videos(tmp_path: Path) -> None:
    _write_benchmark_file(
        tmp_path / "b_video.json",
        [
            {
                "question_id": "b-1",
                "task_type": "Information Synopsis",
                "question": "B?",
                "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
                "answer": "B",
            }
        ],
    )
    _write_benchmark_file(
        tmp_path / "a_video.json",
        [
            {
                "question_id": "a-1",
                "task_type": "Information Synopsis",
                "question": "A?",
                "options": ["A. x", "B. y", "C. z", "D. w"],
                "answer": "A",
            }
        ],
    )

    questions = load_benchmark(tmp_path)

    assert [question.video_id for question in questions] == ["a_video", "b_video"]
    assert [question.question_id for question in questions] == ["a-1", "b-1"]


def test_load_benchmark_empty_dir(tmp_path: Path) -> None:
    assert load_benchmark(tmp_path) == []


def test_load_benchmark_skips_non_json(tmp_path: Path) -> None:
    _write_benchmark_file(
        tmp_path / "real_video.json",
        [
            {
                "question_id": "real-1",
                "task_type": "Information Synopsis",
                "question": "Real?",
                "options": ["A. yes", "B. no", "C. maybe", "D. later"],
                "answer": "A",
            }
        ],
    )
    (tmp_path / "ignored.txt").write_text("[]", encoding="utf-8")

    questions = load_benchmark(tmp_path)

    assert len(questions) == 1
    assert questions[0].video_id == "real_video"
