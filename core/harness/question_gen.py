"""出题数据结构，对应 DataLoader。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneratedQuestion:
    """表示单条生成题目的结构化结果。"""

    question_id: str
    video_id: str
    task_type: str
    question: str
    options: list[str]
    answer: str
    source_nodes: list[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class QuestionGenResult:
    """表示一次出题流程的汇总统计结果。"""

    version: str
    total: int
    per_task_type: dict[str, int] = field(default_factory=dict)
    per_video: dict[str, int] = field(default_factory=dict)


def load_benchmark(questions_dir: Path) -> list[GeneratedQuestion]:
    """从目录中的 benchmark JSON 文件加载题目列表。"""

    results: list[GeneratedQuestion] = []
    for path in sorted(questions_dir.glob("*.json")):
        video_id = path.stem
        with open(path, encoding="utf-8") as f:
            qa_list: list[dict] = json.load(f)
        for qa in qa_list:
            results.append(
                GeneratedQuestion(
                    question_id=qa["question_id"],
                    video_id=video_id,
                    task_type=qa["task_type"],
                    question=qa["question"],
                    options=qa["options"],
                    answer=qa["answer"],
                )
            )
    return results
