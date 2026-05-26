---
type: plan
node_id: plan:2026-05-26-inference-step
title: 推理步骤（inference）实现计划
date: 2026-05-26
---

# 推理步骤（inference）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 TRM4 已有的 AgentLoop、TreeEnvironment、Tools、PromptManager、SkillRegistry、HarnessLog、Workspace 组装为可独立调用的 `run_inference()` 函数，对应训练循环的 forward()。

**Architecture:** 组装式设计——`run_inference()` 编排函数 + `_run_single_question()` 内部函数。per-question ThreadPool 并行，pluggy TracePlugin 记录轨迹，HarnessLog 加锁保证并发写安全。调用方通过 `load_benchmark()` 或 `question_gen()` 准备 `list[GeneratedQuestion]` 传入。

**Tech Stack:** Python 3.11, pluggy, loguru, sqlite3, ThreadPoolExecutor, sentence-transformers

**Design doc:** `research-wiki/designs/2026-05-26-inference-step.md`

---

## 文件结构

| 文件 | 职责 | 变更类型 |
|------|------|---------|
| `core/harness/question_gen.py` | 题目数据结构 + benchmark 加载 | 修改 |
| `tests/unit/test_harness_question_gen.py` | 题目加载测试 | 修改 |
| `core/harness/log.py` | SQLite 日志（加线程锁） | 修改 |
| `tests/unit/test_harness_log.py` | 线程安全测试 | 新建 |
| `core/search/skills.py` | 技能注册表 + discover_skills | 修改 |
| `tests/unit/test_skills.py` | discover_skills 测试 | 修改 |
| `core/tree/environment.py` | TreeEnvironment（embedding 加锁） | 修改 |
| `core/harness/inference.py` | run_inference + _run_single_question + TracePlugin | 扩充 |
| `tests/unit/test_harness_inference.py` | 推理编排测试 | 扩充 |

---

### Task 1: GeneratedQuestion options 格式修复 + load_benchmark

**Files:**
- Modify: `core/harness/question_gen.py`
- Modify: `tests/unit/test_harness_question_gen.py`

- [ ] **Step 1: 修改 GeneratedQuestion.options 类型并更新测试**

将 `core/harness/question_gen.py` 中 `options: dict[str, str]` 改为 `options: list[str]`，同时在文件顶部添加 `import json` 和 `from pathlib import Path`（后续 `load_benchmark` 需要）：

```python
# core/harness/question_gen.py — 完整文件

"""出题数据结构与 benchmark 加载，对应 DataLoader。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneratedQuestion:
    """表示单条题目的结构化结果。

    options 对齐 Video-MME benchmark 格式：["A. 选项文本", "B. 选项文本", ...]。
    """

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
```

更新 `tests/unit/test_harness_question_gen.py`——将所有 `dict` options 改为 `list[str]`：

```python
# tests/unit/test_harness_question_gen.py — 完整文件

from __future__ import annotations

from core.harness.question_gen import GeneratedQuestion, QuestionGenResult


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
```

- [ ] **Step 2: 运行测试确认 options 格式修改正确**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_question_gen.py -v
```

预期：全部 4 个测试 PASS。

- [ ] **Step 3: 为 load_benchmark 编写测试**

在 `tests/unit/test_harness_question_gen.py` 末尾追加：

```python
import json
from pathlib import Path

from core.harness.question_gen import load_benchmark


def test_load_benchmark_single_video(tmp_path: Path) -> None:
    """测试单个视频文件加载。"""
    video_id = "abc123"
    qa_data = [
        {
            "question_id": "q-1",
            "task_type": "Counting Problem",
            "question": "How many?",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "answer": "B",
        }
    ]
    (tmp_path / f"{video_id}.json").write_text(
        json.dumps(qa_data, ensure_ascii=False)
    )

    questions = load_benchmark(tmp_path)
    assert len(questions) == 1
    q = questions[0]
    assert q.video_id == video_id
    assert q.question_id == "q-1"
    assert q.task_type == "Counting Problem"
    assert q.options == ["A. 1", "B. 2", "C. 3", "D. 4"]
    assert q.answer == "B"
    assert q.source_nodes == []
    assert q.difficulty == "medium"


def test_load_benchmark_multiple_videos(tmp_path: Path) -> None:
    """测试多视频文件加载，验证按 video_id 排序。"""
    for vid in ["zzz", "aaa"]:
        data = [
            {
                "question_id": f"{vid}-q1",
                "task_type": "OCR",
                "question": "What text?",
                "options": ["A. Yes", "B. No"],
                "answer": "A",
            },
            {
                "question_id": f"{vid}-q2",
                "task_type": "OCR",
                "question": "Second?",
                "options": ["A. X", "B. Y"],
                "answer": "B",
            },
        ]
        (tmp_path / f"{vid}.json").write_text(json.dumps(data))

    questions = load_benchmark(tmp_path)
    assert len(questions) == 4
    assert questions[0].video_id == "aaa"
    assert questions[2].video_id == "zzz"


def test_load_benchmark_empty_dir(tmp_path: Path) -> None:
    """空目录应返回空列表。"""
    assert load_benchmark(tmp_path) == []


def test_load_benchmark_skips_non_json(tmp_path: Path) -> None:
    """非 .json 文件应被跳过。"""
    (tmp_path / "readme.md").write_text("not a question file")
    (tmp_path / "vid1.json").write_text(
        json.dumps([{
            "question_id": "q1",
            "task_type": "T",
            "question": "Q?",
            "options": ["A. a"],
            "answer": "A",
        }])
    )
    questions = load_benchmark(tmp_path)
    assert len(questions) == 1
```

- [ ] **Step 4: 运行新测试确认失败**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_question_gen.py::test_load_benchmark_single_video -v
```

预期：FAIL — `ImportError: cannot import name 'load_benchmark'`

- [ ] **Step 5: 实现 load_benchmark**

在 `core/harness/question_gen.py` 的 `QuestionGenResult` 类后追加：

```python
def load_benchmark(questions_dir: Path) -> list[GeneratedQuestion]:
    """从 Store benchmark 目录加载题目，转为 GeneratedQuestion 列表。

    每个 {video_id}.json 包含 list[dict]，video_id 从文件名提取。
    按 video_id 排序后展平返回。

    参数:
        questions_dir: benchmark 题目目录路径。

    返回:
        所有题目的 GeneratedQuestion 列表。
    """
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
```

- [ ] **Step 6: 运行全部 question_gen 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_question_gen.py -v
```

预期：全部 8 个测试 PASS。

- [ ] **Step 7: 更新 `__init__.py` 导出并提交**

在 `core/harness/__init__.py` 中添加 `load_benchmark` 到 import 和 `__all__`：

```python
from core.harness.question_gen import GeneratedQuestion, QuestionGenResult, load_benchmark
```

```bash
git add core/harness/question_gen.py core/harness/__init__.py tests/unit/test_harness_question_gen.py
git commit -m "feat(harness): unify options to list[str] and add load_benchmark"
```

---

### Task 2: HarnessLog 线程安全

**Files:**
- Modify: `core/harness/log.py`
- Create: `tests/unit/test_harness_log.py`

- [ ] **Step 1: 编写线程安全测试**

```python
# tests/unit/test_harness_log.py — 完整文件

from __future__ import annotations

import threading
from pathlib import Path

from core.harness.log import HarnessLog


def test_concurrent_inserts(tmp_path: Path) -> None:
    """多线程并发写入不丢数据且不抛异常。"""
    db_path = str(tmp_path / "test.db")
    log = HarnessLog(db_path, "run-concurrent", git_sha="abc123")
    log.create_table("items", {"value": "INTEGER"})

    n_threads = 8
    n_per_thread = 50
    errors: list[Exception] = []

    def worker(start: int) -> None:
        try:
            for i in range(n_per_thread):
                log.insert("items", {"value": start + i})
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(t * n_per_thread,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"并发写入异常: {errors}"

    rows = log.query(
        "SELECT COUNT(*) as cnt FROM items WHERE run_id = ?", ("run-concurrent",)
    )
    assert rows[0]["cnt"] == n_threads * n_per_thread
    log.close()


def test_context_manager(tmp_path: Path) -> None:
    """测试上下文管理器正常关闭。"""
    db_path = str(tmp_path / "ctx.db")
    with HarnessLog(db_path, "run-ctx", git_sha="abc") as log:
        log.create_table("t", {"x": "TEXT"})
        log.insert("t", {"x": "hello"})

    log2 = HarnessLog(db_path, "run-ctx-check", git_sha="abc")
    rows = log2.query("SELECT status FROM _runs WHERE run_id = ?", ("run-ctx",))
    assert rows[0]["status"] == "completed"
    log2.close()


def test_context_manager_on_exception(tmp_path: Path) -> None:
    """异常时上下文管理器设置 status=failed。"""
    db_path = str(tmp_path / "err.db")
    try:
        with HarnessLog(db_path, "run-err", git_sha="abc") as log:
            raise ValueError("boom")
    except ValueError:
        pass

    log2 = HarnessLog(db_path, "run-check", git_sha="abc")
    rows = log2.query("SELECT status FROM _runs WHERE run_id = ?", ("run-err",))
    assert rows[0]["status"] == "failed"
    log2.close()
```

- [ ] **Step 2: 运行测试确认并发测试行为**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_log.py -v
```

预期：`test_context_manager` 和 `test_context_manager_on_exception` 应 PASS；`test_concurrent_inserts` 可能失败（无锁时 SQLite 并发写可能报 `database is locked`）。

- [ ] **Step 3: 给 HarnessLog 添加 threading.Lock**

修改 `core/harness/log.py`：

1. 文件顶部添加 `import threading`
2. `__init__` 中 `self._conn = sqlite3.connect(db_path)` 改为 `self._conn = sqlite3.connect(db_path, check_same_thread=False)`
3. `__init__` 末尾添加 `self._lock = threading.Lock()`
4. `insert` 方法中 `self._conn.execute` 和 `self._conn.commit()` 包裹 `with self._lock:`
5. `execute` 方法同上
6. `log_event` 方法同上
7. `close` 方法同上

具体改动——`insert` 方法的最后两行改为：

```python
        with self._lock:
            self._conn.execute(sql, values)
            self._conn.commit()
```

`execute` 方法改为：

```python
    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()
```

`log_event` 方法改为：

```python
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO _events (run_id, timestamp, event_type, payload) VALUES (?, ?, ?, ?)",
                (self._run_id, _now_iso(), event_type, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()
```

`close` 方法改为：

```python
    def close(self, status: str = "completed") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE _runs SET finished_at = ?, status = ? WHERE run_id = ?",
                (_now_iso(), status, self._run_id),
            )
            self._conn.commit()
            self._conn.close()
```

- [ ] **Step 4: 运行全部 log 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_log.py -v
```

预期：全部 3 个测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add core/harness/log.py tests/unit/test_harness_log.py
git commit -m "feat(harness): add threading.Lock to HarnessLog for concurrent writes"
```

---

### Task 3: discover_skills 函数

**Files:**
- Modify: `core/search/skills.py`
- Modify: `tests/unit/test_skills.py`

- [ ] **Step 1: 编写 discover_skills 测试**

在 `tests/unit/test_skills.py` 末尾追加：

```python
from core.search.skills import discover_skills

_ALWAYS_SKILL = """---
name: general-strategy
description: 通用搜索策略
always: true
---

通用策略正文内容。
"""

_TASK_SKILL = """---
name: counting-problem
description: 计数类问题搜索策略
task_type: Counting Problem
---

计数题专用策略。
"""

_TASK_SKILL_2 = """---
name: ocr-problems
description: OCR 文字识别策略
task_type: OCR Problems
---

OCR 专用策略。
"""


class TestDiscoverSkills:
    """测试技能目录扫描与分类。"""

    def test_empty_dir(self, tmp_path: Path) -> None:
        always_text, task_map, catalog, registry = discover_skills(tmp_path)
        assert always_text == ""
        assert task_map == {}
        assert catalog == ""

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        always_text, task_map, catalog, registry = discover_skills(
            tmp_path / "nonexistent"
        )
        assert always_text == ""
        assert task_map == {}

    def test_mixed_skills(self, tmp_path: Path) -> None:
        (tmp_path / "always.md").write_text(_ALWAYS_SKILL)
        (tmp_path / "counting.md").write_text(_TASK_SKILL)
        (tmp_path / "ocr.md").write_text(_TASK_SKILL_2)

        always_text, task_map, catalog, registry = discover_skills(tmp_path)

        assert "通用策略正文内容" in always_text
        assert "Counting Problem" in task_map
        assert "计数题专用策略" in task_map["Counting Problem"]
        assert "OCR Problems" in task_map
        assert "counting-problem" in catalog
        assert "ocr-problems" in catalog
        assert registry.read("counting-problem").strip() == "计数题专用策略。"

    def test_always_skill_not_in_catalog(self, tmp_path: Path) -> None:
        """always 技能不应出现在 catalog 和 registry 中。"""
        (tmp_path / "always.md").write_text(_ALWAYS_SKILL)
        _, _, catalog, registry = discover_skills(tmp_path)
        assert "general-strategy" not in catalog
        with pytest.raises(KeyError):
            registry.read("general-strategy")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_skills.py::TestDiscoverSkills -v
```

预期：FAIL — `ImportError: cannot import name 'discover_skills'`

- [ ] **Step 3: 实现 discover_skills**

在 `core/search/skills.py` 末尾追加：

```python
def discover_skills(
    skills_dir: Path,
) -> tuple[str, dict[str, str], str, SkillRegistry]:
    """扫描 skills 目录，按 frontmatter 分类返回。

    遍历 *.md 文件，根据 frontmatter 的 always/task_type 字段分类：
    - always=true 的 skill 拼入 always_skills_text
    - 有 task_type 的 skill 加入 task_skill_map
    - 非 always 的 skill 生成 catalog_text 并注册到 registry

    参数:
        skills_dir: Skill 文件目录。

    返回:
        (always_skills_text, task_skill_map, catalog_text, registry) 四元组。
    """
    if not skills_dir.exists():
        return "", {}, "", SkillRegistry()

    always_parts: list[str] = []
    task_skill_map: dict[str, str] = {}
    catalog_lines: list[str] = []
    registry_paths: dict[str, Path] = {}

    for path in sorted(skills_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta = parse_frontmatter(raw)
        if "name" not in meta:
            logger.warning("跳过无 name 的 skill 文件: {}", path)
            continue

        body = strip_frontmatter(raw)
        name = meta["name"]
        desc = meta.get("description", "")
        task_type = meta.get("task_type", "")
        is_always = str(meta.get("always", "false")).lower() == "true"

        if is_always:
            always_parts.append(body)
        else:
            if task_type:
                task_skill_map[task_type] = body
            catalog_lines.append(f"- **{name}**: {desc}")
            registry_paths[name] = path

    always_text = "\n\n---\n\n".join(always_parts)
    catalog_text = "\n".join(catalog_lines)

    registry = SkillRegistry()
    registry.set_paths(registry_paths)

    return always_text, task_skill_map, catalog_text, registry
```

- [ ] **Step 4: 运行全部 skills 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_skills.py -v
```

预期：全部测试 PASS（原有 8 个 + 新增 4 个 = 12 个）。

- [ ] **Step 5: 提交**

```bash
git add core/search/skills.py tests/unit/test_skills.py
git commit -m "feat(search): add discover_skills for skill directory scanning"
```

---

### Task 4: TreeEnvironment embedding 懒加载线程安全

**Files:**
- Modify: `core/tree/environment.py`

- [ ] **Step 1: 添加 threading.Lock 到 _ensure_embeddings**

在 `core/tree/environment.py` 顶部添加 `import threading`。

在 `TreeEnvironment.__init__` 末尾（`self._model: Any = None` 之后）添加：

```python
        self._embed_lock = threading.Lock()
```

修改 `_ensure_embeddings` 方法为双重检查锁定模式：

```python
    def _ensure_embeddings(self) -> None:
        """延迟加载或生成 embedding 索引（支持 chunking，线程安全）。"""
        if self._embeddings is not None:
            return

        with self._embed_lock:
            if self._embeddings is not None:
                return

            cache_path = self._tree_dir / f"{self._video_id}.embeddings.npz"

            if cache_path.exists():
                data = np.load(cache_path, allow_pickle=True)
                self._chunk_node_ids = data["chunk_node_ids"].tolist()
                self._embeddings = data["embeddings"]
                return

            chunk_node_ids: list[str] = []
            texts: list[str] = []
            for nid, node in self._nodes.items():
                full_text = self._node_full_text(node)
                for chunk in self._chunk_text(full_text):
                    chunk_node_ids.append(nid)
                    texts.append(chunk)

            self._chunk_node_ids = chunk_node_ids

            model = self._get_model()
            self._embeddings = model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            np.savez(
                cache_path,
                chunk_node_ids=np.array(self._chunk_node_ids),
                embeddings=self._embeddings,
            )
```

- [ ] **Step 2: 运行现有 environment 测试确认无回归**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_environment.py -v
```

预期：全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add core/tree/environment.py
git commit -m "fix(tree): add threading.Lock to embedding lazy initialization"
```

---

### Task 5: TracePlugin（pluggy 轨迹记录插件）

**Files:**
- Modify: `core/harness/inference.py`
- Modify: `tests/unit/test_harness_inference.py`

- [ ] **Step 1: 编写 TracePlugin 测试**

在 `tests/unit/test_harness_inference.py` 末尾追加：

```python
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

from core.harness.inference import TracePlugin
from core.harness.log import HarnessLog
from core.loop import LoopResult, Step


def _make_step(tool_name: str, node_id: str = "") -> Step:
    """构造测试用 Step。"""
    return Step(
        thought="thinking...",
        reflect={},
        plan={},
        tool_call={"tool": tool_name, "args": {"node_id": node_id}},
        tool_output="output",
        raw_content="{}",
    )


class TestTracePlugin:
    """TracePlugin pluggy 插件测试。"""

    def test_after_tool_writes_trace(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        log = HarnessLog(db_path, "run-trace", git_sha="abc")
        log.create_table(
            "traces",
            {
                "video_id": "TEXT",
                "question_id": "TEXT",
                "step": "INTEGER",
                "tool_name": "TEXT",
                "tool_args": "JSON",
                "tool_output": "TEXT",
                "thought": "TEXT",
            },
        )

        plugin = TracePlugin(log, "vid-1", "q-1")
        step = _make_step("view_node", "vid-1_L1_000")
        plugin.after_tool(iteration=0, step=step)

        rows = log.query("SELECT * FROM traces WHERE question_id = ?", ("q-1",))
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "view_node"
        assert rows[0]["video_id"] == "vid-1"
        assert rows[0]["thought"] == "thinking..."
        log.close()

    def test_on_finish_writes_validation_flags(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        log = HarnessLog(db_path, "run-flags", git_sha="abc")
        log.create_table(
            "validation_flags",
            {
                "video_id": "TEXT",
                "question_id": "TEXT",
                "has_l3_visit": "INTEGER",
                "l1_count": "INTEGER",
                "l2_count": "INTEGER",
                "l3_count": "INTEGER",
            },
        )

        plugin = TracePlugin(log, "vid-1", "q-1")
        result = LoopResult(
            result={"answer": "A"},
            steps=[
                _make_step("view_node", "vid-1_L1_000"),
                _make_step("view_node", "vid-1_L1_000_L2_001"),
                _make_step("view_node", "vid-1_L1_000_L2_001_L3_002"),
                _make_step("search_similar"),
            ],
            steps_used=4,
            stop_reason="finished",
        )
        plugin.on_finish(result=result)

        rows = log.query(
            "SELECT * FROM validation_flags WHERE question_id = ?", ("q-1",)
        )
        assert len(rows) == 1
        assert rows[0]["has_l3_visit"] == 1
        assert rows[0]["l1_count"] == 1
        assert rows[0]["l2_count"] == 1
        assert rows[0]["l3_count"] == 1
        log.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py::TestTracePlugin -v
```

预期：FAIL — `ImportError: cannot import name 'TracePlugin'`

- [ ] **Step 3: 实现 TracePlugin**

将 `core/harness/inference.py` 的完整内容替换为：

```python
"""训练循环的 forward() 步骤：推理编排。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from core.loop import LoopResult, Step, hookimpl

_NODE_LEVEL_RE = re.compile(r"_L(\d+)_")


@dataclass(frozen=True)
class InferenceResult:
    """封装训练循环一次 forward() 推理输出的不可变结果结构。"""

    run_id: str
    accuracy: float
    total: int
    correct: int
    per_task_type: dict[str, dict]
    steps_mean: float
    token_usage: dict[str, int]
    stop_reason_counts: dict[str, int]


class TracePlugin:
    """pluggy 插件：记录工具调用轨迹和树遍历验证标记。

    参数:
        log: HarnessLog 实例。
        video_id: 当前视频 ID。
        question_id: 当前题目 ID。
    """

    def __init__(self, log: Any, video_id: str, question_id: str) -> None:
        self._log = log
        self._video_id = video_id
        self._question_id = question_id

    @hookimpl
    def after_tool(self, iteration: int, step: Step) -> str | None:
        """每次工具调用后写入 traces 表。"""
        self._log.insert(
            "traces",
            {
                "video_id": self._video_id,
                "question_id": self._question_id,
                "step": iteration,
                "tool_name": step.tool_call["tool"],
                "tool_args": json.dumps(step.tool_call["args"], ensure_ascii=False),
                "tool_output": step.tool_output,
                "thought": step.thought,
            },
        )
        return None

    @hookimpl
    def on_finish(self, result: LoopResult) -> None:
        """循环结束后写入验证标记。"""
        l1, l2, l3 = 0, 0, 0
        for s in result.steps:
            if s.tool_call.get("tool") == "view_node":
                node_id = s.tool_call.get("args", {}).get("node_id", "")
                matches = _NODE_LEVEL_RE.findall(node_id)
                if matches:
                    level = int(matches[-1])
                    if level == 1:
                        l1 += 1
                    elif level == 2:
                        l2 += 1
                    elif level == 3:
                        l3 += 1
        self._log.insert(
            "validation_flags",
            {
                "video_id": self._video_id,
                "question_id": self._question_id,
                "has_l3_visit": 1 if l3 > 0 else 0,
                "l1_count": l1,
                "l2_count": l2,
                "l3_count": l3,
            },
        )
```

- [ ] **Step 4: 运行全部 inference 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py -v
```

预期：原有 2 个 + 新增 2 个 = 全部 4 个 PASS。

- [ ] **Step 5: 提交**

```bash
git add core/harness/inference.py tests/unit/test_harness_inference.py
git commit -m "feat(harness): add TracePlugin as pluggy hookimpl for trace logging"
```

---

### Task 6: _run_single_question 内部函数

**Files:**
- Modify: `core/harness/inference.py`
- Modify: `tests/unit/test_harness_inference.py`

- [ ] **Step 1: 编写 _run_single_question 测试**

在 `tests/unit/test_harness_inference.py` 追加：

```python
from unittest.mock import patch, MagicMock
from core.harness.inference import _run_single_question
from core.harness.question_gen import GeneratedQuestion
from core.search.skills import SkillRegistry


def _make_qa() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_id="q-1",
        video_id="vid-1",
        task_type="Counting Problem",
        question="How many cats?",
        options=["A. 1", "B. 2", "C. 3", "D. 4"],
        answer="B",
    )


def test_run_single_question_finished(tmp_path: Path) -> None:
    """正常完成时，prediction 写入 log 且返回正确结构。"""
    db_path = str(tmp_path / "test.db")
    log = HarnessLog(db_path, "run-sq", git_sha="abc")
    log.create_table("predictions", {
        "video_id": "TEXT", "question_id": "TEXT", "task_type": "TEXT",
        "prediction": "TEXT", "answer": "TEXT", "evidence": "TEXT",
        "reasoning": "TEXT", "steps_used": "INTEGER",
        "prompt_tokens": "INTEGER", "completion_tokens": "INTEGER",
        "stop_reason": "TEXT", "steps_json": "JSON",
    })
    log.create_table("traces", {
        "video_id": "TEXT", "question_id": "TEXT", "step": "INTEGER",
        "tool_name": "TEXT", "tool_args": "JSON", "tool_output": "TEXT",
        "thought": "TEXT",
    })
    log.create_table("validation_flags", {
        "video_id": "TEXT", "question_id": "TEXT", "has_l3_visit": "INTEGER",
        "l1_count": "INTEGER", "l2_count": "INTEGER", "l3_count": "INTEGER",
    })

    mock_loop_result = LoopResult(
        result={"answer": "B", "evidence": "saw 2 cats", "reasoning": "counted them"},
        steps=[_make_step("view_node", "vid-1_L1_000"), _make_step("submit_answer")],
        steps_used=2,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        stop_reason="finished",
    )

    mock_prompt_mgr = MagicMock()
    mock_prompt_mgr.build_inference_prompt.return_value = "system prompt"
    mock_prompt_mgr.format_user_prompt.return_value = "user prompt"

    with patch("core.harness.inference.LLMClient") as MockClient, \
         patch("core.harness.inference.AgentLoop") as MockLoop:
        mock_client_instance = MagicMock()
        MockClient.from_env.return_value = mock_client_instance
        mock_loop_instance = MagicMock()
        mock_loop_instance.run.return_value = mock_loop_result
        MockLoop.return_value = mock_loop_instance

        mock_env = MagicMock()
        mock_env._nodes = {"vid-1_L1_000": {"level": 1, "children_ids": []}}

        result = _run_single_question(
            qa=_make_qa(),
            env=mock_env,
            vl_client=MagicMock(),
            prompt_mgr=mock_prompt_mgr,
            skill_registry=SkillRegistry(),
            log=log,
            max_steps=15,
            skill_mode="auto",
            always_skills_text="",
            task_skill_map={},
            catalog_text="",
            prompts_dir=tmp_path,
        )

    assert result["prediction"] == "B"
    assert result["stop_reason"] == "finished"

    rows = log.query("SELECT * FROM predictions WHERE question_id = ?", ("q-1",))
    assert len(rows) == 1
    assert rows[0]["prediction"] == "B"
    log.close()


def test_run_single_question_error(tmp_path: Path) -> None:
    """异常时写入 stop_reason=error 的兜底记录。"""
    db_path = str(tmp_path / "err.db")
    log = HarnessLog(db_path, "run-err", git_sha="abc")
    log.create_table("predictions", {
        "video_id": "TEXT", "question_id": "TEXT", "task_type": "TEXT",
        "prediction": "TEXT", "answer": "TEXT", "evidence": "TEXT",
        "reasoning": "TEXT", "steps_used": "INTEGER",
        "prompt_tokens": "INTEGER", "completion_tokens": "INTEGER",
        "stop_reason": "TEXT", "steps_json": "JSON",
    })
    log.create_table("traces", {
        "video_id": "TEXT", "question_id": "TEXT", "step": "INTEGER",
        "tool_name": "TEXT", "tool_args": "JSON", "tool_output": "TEXT",
        "thought": "TEXT",
    })
    log.create_table("validation_flags", {
        "video_id": "TEXT", "question_id": "TEXT", "has_l3_visit": "INTEGER",
        "l1_count": "INTEGER", "l2_count": "INTEGER", "l3_count": "INTEGER",
    })

    mock_prompt_mgr = MagicMock()
    mock_prompt_mgr.build_inference_prompt.return_value = "sys"
    mock_prompt_mgr.format_user_prompt.return_value = "usr"

    with patch("core.harness.inference.LLMClient") as MockClient, \
         patch("core.harness.inference.AgentLoop") as MockLoop:
        MockClient.from_env.return_value = MagicMock()
        mock_loop_instance = MagicMock()
        mock_loop_instance.run.side_effect = RuntimeError("LLM down")
        MockLoop.return_value = mock_loop_instance

        mock_env = MagicMock()
        mock_env._nodes = {}

        result = _run_single_question(
            qa=_make_qa(),
            env=mock_env,
            vl_client=MagicMock(),
            prompt_mgr=mock_prompt_mgr,
            skill_registry=SkillRegistry(),
            log=log,
            max_steps=15,
            skill_mode="none",
            always_skills_text="",
            task_skill_map={},
            catalog_text="",
            prompts_dir=tmp_path,
        )

    assert result["stop_reason"] == "error"
    rows = log.query("SELECT * FROM predictions WHERE question_id = ?", ("q-1",))
    assert len(rows) == 1
    assert rows[0]["stop_reason"] == "error"
    log.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py::test_run_single_question_finished -v
```

预期：FAIL — `ImportError: cannot import name '_run_single_question'`

- [ ] **Step 3: 实现 _run_single_question**

在 `core/harness/inference.py` 中，文件顶部 import 区域扩展为：

```python
"""训练循环的 forward() 步骤：推理编排。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from core.harness.log import HarnessLog
from core.harness.question_gen import GeneratedQuestion
from core.llm_client import LLMClient
from core.loop import AgentLoop, LoopResult, Step, hookimpl
from core.search.prompt import PromptManager
from core.search.skills import SkillRegistry
from core.tree.tools import dispatch
```

在 `TracePlugin` 类后追加 `_run_single_question` 函数：

```python
def _run_single_question(
    qa: GeneratedQuestion,
    env: Any,
    vl_client: Any,
    prompt_mgr: PromptManager,
    skill_registry: SkillRegistry,
    log: HarnessLog,
    max_steps: int,
    skill_mode: str,
    always_skills_text: str,
    task_skill_map: dict[str, str],
    catalog_text: str,
    prompts_dir: Path,
) -> dict[str, Any]:
    """执行单道题目的 Agent 推理。

    创建独立的 search_client 和 AgentLoop（线程安全），
    通过 PromptManager 组装 prompt，运行循环，结果写入 log。

    参数:
        qa: 待推理的题目。
        env: TreeEnvironment 实例（同 video_id 的题目共享）。
        vl_client: 视觉模型 LLMClient（共享）。
        prompt_mgr: PromptManager 实例。
        skill_registry: SkillRegistry 实例。
        log: HarnessLog 实例（线程安全）。
        max_steps: AgentLoop 最大步数。
        skill_mode: "auto" / "manual" / "none"。
        always_skills_text: always 层 skill 全文。
        task_skill_map: {task_type: skill_body} 映射。
        catalog_text: manual 模式的 skill 目录文本。
        prompts_dir: prompt 文件目录。

    返回:
        预测结果字典（prediction, answer, steps_used 等）。
    """
    record: dict[str, Any] = {
        "video_id": qa.video_id,
        "question_id": qa.question_id,
        "task_type": qa.task_type,
        "prediction": None,
        "answer": qa.answer,
        "evidence": "",
        "reasoning": "",
        "steps_used": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "stop_reason": "error",
        "steps_json": "[]",
    }

    try:
        search_client = LLMClient.from_env("SEARCH_LLM", thinking=True)
        loop = AgentLoop(search_client, max_steps=max_steps)

        system_prompt = prompt_mgr.build_inference_prompt(
            skill_mode=skill_mode,
            task_type=qa.task_type,
            always_skills_text=always_skills_text,
            task_skill_map=task_skill_map,
            catalog_text=catalog_text,
        )

        l1_ids = sorted(
            nid for nid, node in env._nodes.items() if node.get("level") == 1
        )
        qa_dict = {"question": qa.question, "options": qa.options}
        user_prompt = prompt_mgr.format_user_prompt(qa_dict, l1_ids)

        trace_plugin = TracePlugin(log, qa.video_id, qa.question_id)

        tool_fn = partial(
            dispatch,
            env=env,
            vl_client=vl_client,
            prompts_dir=prompts_dir,
            skills=skill_registry if skill_mode == "manual" else None,
        )

        loop_result = loop.run(
            system_prompt, user_prompt, tool_fn, plugins=[trace_plugin]
        )

        result_dict = loop_result.result if isinstance(loop_result.result, dict) else {}
        record.update({
            "prediction": result_dict.get("answer"),
            "evidence": result_dict.get("evidence", ""),
            "reasoning": result_dict.get("reasoning", ""),
            "steps_used": loop_result.steps_used,
            "prompt_tokens": loop_result.token_usage["prompt_tokens"],
            "completion_tokens": loop_result.token_usage["completion_tokens"],
            "stop_reason": loop_result.stop_reason,
            "steps_json": json.dumps(
                [
                    {
                        "thought": s.thought,
                        "tool_call": s.tool_call,
                        "tool_output": s.tool_output,
                    }
                    for s in loop_result.steps
                ],
                ensure_ascii=False,
            ),
        })
    except Exception:
        logger.exception("[{}] QA {} 执行异常", qa.video_id, qa.question_id)

    log.insert("predictions", record)
    return record
```

- [ ] **Step 4: 运行全部 inference 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py -v
```

预期：全部 6 个 PASS（原有 2 + TracePlugin 2 + single_question 2）。

- [ ] **Step 5: 提交**

```bash
git add core/harness/inference.py tests/unit/test_harness_inference.py
git commit -m "feat(harness): add _run_single_question internal function"
```

---

### Task 7: run_inference 编排函数

**Files:**
- Modify: `core/harness/inference.py`
- Modify: `tests/unit/test_harness_inference.py`
- Modify: `core/harness/__init__.py`

- [ ] **Step 1: 编写 run_inference 测试**

在 `tests/unit/test_harness_inference.py` 追加：

```python
from core.harness.inference import run_inference
from core.workspace import init_workspace


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """创建最小 store + workspace 用于测试。"""
    store = tmp_path / "store"
    store.mkdir()
    (store / "videos").mkdir()
    vid_dir = store / "videos" / "vid-1"
    vid_dir.mkdir()
    tree_data = {
        "video_id": "vid-1",
        "videoID": "vid-1",
        "duration_category": "short",
        "duration_seconds": 60.0,
        "domain": "test",
        "built_at": "2026-01-01",
        "build_stats": {},
        "failures": [],
        "nodes": {
            "vid-1_L1_000": {
                "node_id": "vid-1_L1_000",
                "level": 1,
                "time_range": [0.0, 60.0],
                "parent_id": None,
                "children_ids": [],
                "card": {"scene_summary": "A test scene"},
                "subtitle": "",
            }
        },
    }
    (vid_dir / "tree.json").write_text(json.dumps(tree_data))

    skills_dir = store / "skills" / "v1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "meta.json").write_text(
        '{"version":"v1","source":"manual","created_at":"2026-01-01"}'
    )

    prompts_dir = store / "prompts" / "v1"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "system.md").write_text("You are a test agent.")
    (prompts_dir / "meta.json").write_text(
        '{"version":"v1","source":"manual","created_at":"2026-01-01"}'
    )

    questions_dir = store / "questions" / "benchmarks" / "test"
    questions_dir.mkdir(parents=True)

    ws = tmp_path / "ws"
    init_workspace(ws, store, "benchmarks/test", "v1", "v1")

    return ws, store


def test_run_inference_basic(tmp_path: Path) -> None:
    """基础编排：单题推理，验证 InferenceResult 结构。"""
    ws, store = _setup_workspace(tmp_path)

    questions = [
        GeneratedQuestion(
            question_id="q-1",
            video_id="vid-1",
            task_type="Test",
            question="What is this?",
            options=["A. Yes", "B. No"],
            answer="A",
        )
    ]

    mock_loop_result = LoopResult(
        result={"answer": "A", "evidence": "saw it", "reasoning": "obvious"},
        steps=[_make_step("submit_answer")],
        steps_used=1,
        token_usage={"prompt_tokens": 50, "completion_tokens": 20},
        stop_reason="finished",
    )

    with patch("core.harness.inference.LLMClient") as MockClient, \
         patch("core.harness.inference.AgentLoop") as MockLoop, \
         patch("core.harness.inference.TreeEnvironment") as MockEnv:
        MockClient.from_env.return_value = MagicMock()
        mock_loop_instance = MagicMock()
        mock_loop_instance.run.return_value = mock_loop_result
        MockLoop.return_value = mock_loop_instance
        mock_env_instance = MagicMock()
        mock_env_instance._nodes = {
            "vid-1_L1_000": {"level": 1, "children_ids": []}
        }
        MockEnv.return_value = mock_env_instance

        result = run_inference(
            workspace_dir=ws,
            questions=questions,
            concurrency=1,
            max_steps=15,
            skill_mode="none",
        )

    assert result.run_id is not None
    assert result.total == 1
    assert result.correct == 1
    assert result.accuracy == 1.0
    assert result.stop_reason_counts == {"finished": 1}


def test_run_inference_empty_questions(tmp_path: Path) -> None:
    """空题目列表应直接返回零结果。"""
    ws, store = _setup_workspace(tmp_path)

    result = run_inference(
        workspace_dir=ws,
        questions=[],
        concurrency=1,
        max_steps=15,
        skill_mode="none",
    )

    assert result.total == 0
    assert result.accuracy == 0.0
    assert result.correct == 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py::test_run_inference_basic -v
```

预期：FAIL — `ImportError: cannot import name 'run_inference'`

- [ ] **Step 3: 实现 run_inference 和 _aggregate_results**

在 `core/harness/inference.py` 顶部 import 区域扩展为最终版本：

```python
"""训练循环的 forward() 步骤：推理编排。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from core.harness.log import HarnessLog
from core.harness.question_gen import GeneratedQuestion
from core.llm_client import LLMClient
from core.loop import AgentLoop, LoopResult, Step, hookimpl
from core.search.prompt import PromptManager
from core.search.skills import SkillRegistry, discover_skills
from core.tree.environment import TreeEnvironment
from core.tree.tools import dispatch
from core.workspace import record_run, resolve_paths
```

在 `_run_single_question` 函数后追加：

```python
def _aggregate_results(log: HarnessLog, run_id: str) -> InferenceResult:
    """从 predictions 表聚合推理指标。

    参数:
        log: HarnessLog 实例。
        run_id: 当前 run ID。

    返回:
        InferenceResult 冻结实例。
    """
    rows = log.query("SELECT * FROM predictions WHERE run_id = ?", (run_id,))

    total = len(rows)
    if total == 0:
        return InferenceResult(
            run_id=run_id,
            accuracy=0.0,
            total=0,
            correct=0,
            per_task_type={},
            steps_mean=0.0,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0},
            stop_reason_counts={},
        )

    correct = sum(1 for r in rows if r["prediction"] == r["answer"])
    steps_total = sum(r["steps_used"] for r in rows)
    prompt_total = sum(r["prompt_tokens"] for r in rows)
    completion_total = sum(r["completion_tokens"] for r in rows)

    stop_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        stop_counts[r["stop_reason"]] += 1

    task_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        task_groups[r["task_type"]].append(r)

    per_task_type = {}
    for task_type, group in task_groups.items():
        t_total = len(group)
        t_correct = sum(1 for r in group if r["prediction"] == r["answer"])
        per_task_type[task_type] = {
            "accuracy": t_correct / t_total if t_total > 0 else 0.0,
            "total": t_total,
            "correct": t_correct,
        }

    return InferenceResult(
        run_id=run_id,
        accuracy=correct / total,
        total=total,
        correct=correct,
        per_task_type=per_task_type,
        steps_mean=steps_total / total,
        token_usage={
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
        },
        stop_reason_counts=dict(stop_counts),
    )


def run_inference(
    workspace_dir: Path,
    questions: list[GeneratedQuestion],
    concurrency: int,
    max_steps: int,
    skill_mode: str,
) -> InferenceResult:
    """在视频树上执行 Agent 推理，对应训练循环的 forward()。

    参数:
        workspace_dir: Workspace 根目录。
        questions: 待推理的题目列表（已筛选）。
        concurrency: 最大并行 worker 数。
        max_steps: AgentLoop 单题最大步数。
        skill_mode: "auto" / "manual" / "none"。

    返回:
        InferenceResult（含 accuracy、per_task_type 等聚合指标）。
    """
    paths = resolve_paths(workspace_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    record_run(workspace_dir, run_id)

    config_snapshot = {
        "concurrency": concurrency,
        "max_steps": max_steps,
        "skill_mode": skill_mode,
        "total_questions": len(questions),
    }

    with HarnessLog(str(paths.db_path), run_id, config_snapshot=config_snapshot) as log:
        if not questions:
            return _aggregate_results(log, run_id)

        log.create_table("predictions", {
            "video_id": "TEXT",
            "question_id": "TEXT",
            "task_type": "TEXT",
            "prediction": "TEXT",
            "answer": "TEXT",
            "evidence": "TEXT",
            "reasoning": "TEXT",
            "steps_used": "INTEGER",
            "prompt_tokens": "INTEGER",
            "completion_tokens": "INTEGER",
            "stop_reason": "TEXT",
            "steps_json": "JSON",
        })
        log.create_table("traces", {
            "video_id": "TEXT",
            "question_id": "TEXT",
            "step": "INTEGER",
            "tool_name": "TEXT",
            "tool_args": "JSON",
            "tool_output": "TEXT",
            "thought": "TEXT",
        })
        log.create_table("validation_flags", {
            "video_id": "TEXT",
            "question_id": "TEXT",
            "has_l3_visit": "INTEGER",
            "l1_count": "INTEGER",
            "l2_count": "INTEGER",
            "l3_count": "INTEGER",
        })

        prompt_mgr = PromptManager(paths.prompts_dir)
        always_text, task_skill_map, catalog_text, skill_registry = discover_skills(
            paths.skills_dir
        )

        video_ids = {qa.video_id for qa in questions}
        tool_client = LLMClient.from_env("SEARCH_LLM", thinking=False)
        vl_client = LLMClient.from_env("VL_LLM", thinking=False)

        video_envs: dict[str, TreeEnvironment] = {}
        for vid in video_ids:
            tree_path = paths.videos_dir / vid / "tree.json"
            video_envs[vid] = TreeEnvironment(
                tree_path, tool_client, paths.prompts_dir
            )

        done_count = 0
        total_count = len(questions)

        def _worker(qa: GeneratedQuestion) -> dict[str, Any]:
            return _run_single_question(
                qa=qa,
                env=video_envs[qa.video_id],
                vl_client=vl_client,
                prompt_mgr=prompt_mgr,
                skill_registry=skill_registry,
                log=log,
                max_steps=max_steps,
                skill_mode=skill_mode,
                always_skills_text=always_text,
                task_skill_map=task_skill_map,
                catalog_text=catalog_text,
                prompts_dir=paths.prompts_dir,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, qa): qa for qa in questions}
            for future in as_completed(futures):
                qa = futures[future]
                done_count += 1
                try:
                    future.result()
                    logger.info(
                        "[{}/{}] {} QA {} 完成",
                        done_count, total_count, qa.video_id, qa.question_id,
                    )
                except Exception:
                    logger.exception(
                        "[{}/{}] {} QA {} 失败",
                        done_count, total_count, qa.video_id, qa.question_id,
                    )

        result = _aggregate_results(log, run_id)

    logger.info(
        "推理完成: accuracy={:.2%} ({}/{})",
        result.accuracy, result.correct, result.total,
    )
    return result
```

- [ ] **Step 4: 运行全部 inference 测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/test_harness_inference.py -v
```

预期：全部 8 个测试 PASS。

- [ ] **Step 5: 更新 `__init__.py` 导出并提交**

在 `core/harness/__init__.py` 中更新 inference 的导出：

```python
from core.harness.inference import InferenceResult, TracePlugin, run_inference
from core.harness.question_gen import GeneratedQuestion, QuestionGenResult, load_benchmark
```

并更新 `__all__` 列表，添加 `"TracePlugin"`, `"run_inference"`, `"load_benchmark"`。

```bash
git add core/harness/inference.py core/harness/__init__.py tests/unit/test_harness_inference.py
git commit -m "feat(harness): add run_inference orchestration with per-question parallelism"
```

---

### Task 8: 代码质量 & 最终验证

**Files:**
- All modified files

- [ ] **Step 1: ruff 格式化**

```bash
conda run -n Video-Tree-TRM ruff format core/harness/question_gen.py core/harness/inference.py core/harness/log.py core/harness/__init__.py core/search/skills.py core/tree/environment.py
```

- [ ] **Step 2: ruff 检查**

```bash
conda run -n Video-Tree-TRM ruff check core/harness/ core/search/skills.py core/tree/environment.py --fix
```

- [ ] **Step 3: 运行全部单元测试**

```bash
conda run -n Video-Tree-TRM pytest tests/unit/ -v --tb=short
```

预期：全部 PASS，无回归。

- [ ] **Step 4: 修复任何失败，提交最终状态**

```bash
git add -u
git commit -m "style: format and lint all inference step changes"
```
