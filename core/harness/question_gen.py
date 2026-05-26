"""出题数据结构，对应 DataLoader。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneratedQuestion:
    """表示单条生成题目的结构化结果。"""

    question_id: str
    video_id: str
    task_type: str
    question: str
    options: dict[str, str]
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
