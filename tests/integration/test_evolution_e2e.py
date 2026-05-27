"""进化模块端到端冒烟测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.harness.diagnose import CaseSample, DiagnosisResult, SkillCasePack
from core.harness.evolve import run_evolution


def _make_case(qid: str, *, correct: bool) -> CaseSample:
    """构造最小化的 CaseSample。"""
    return CaseSample(
        question_id=qid,
        video_id="v1",
        task_type="T",
        question="问题?",
        options=["A", "B", "C", "D"],
        answer="A",
        prediction="B" if not correct else "A",
        correct=correct,
        error_type="search_failure" if not correct else None,
        selection_reason="test",
        metrics={},
        trace=[
            {
                "step": 1,
                "tool_name": "view_node",
                "tool_args": '{"node_id": "L1_000"}',
                "tool_output": "场景摘要...",
            }
        ],
    )


def test_evolution_e2e_skill(tmp_path: Path) -> None:
    """完整流程：案例包 → LLM 调用 → 验证 → 版本写入。"""
    # 构造 store
    store = tmp_path / "store"
    skills_v1 = store / "skills" / "v1"
    skills_v1.mkdir(parents=True)
    prompts_v1 = store / "prompts" / "v1"
    prompts_v1.mkdir(parents=True)

    original = "---\nname: t\ndescription: d\ntask_type: T\n---\n## 适用场景\n旧内容\n"
    (skills_v1 / "t.md").write_text(original)
    (skills_v1 / "meta.json").write_text('{"version":"v1","source":"manual"}')
    (prompts_v1 / "system.md").write_text("## 角色\ntest")
    (prompts_v1 / "meta.json").write_text('{"version":"v1","source":"manual"}')

    # 构造 workspace
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "analyses").mkdir()

    evolved = "---\nname: t\ndescription: d\ntask_type: T\n---\n## 适用场景\n新内容\n"
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = json.dumps(
        {
            "suggestions": [
                {
                    "section": "适用场景",
                    "problem": "旧描述不够具体",
                    "change": "更新为新描述",
                    "related_cases": ["q1"],
                }
            ],
            "evolved_content": evolved,
        },
        ensure_ascii=False,
    )

    mock_client = MagicMock()
    mock_client.chat.return_value = mock_resp

    diagnosis = DiagnosisResult(
        run_id="run_e2e",
        skill_case_packs={
            "T": SkillCasePack(
                task_type="T",
                target_file="t.md",
                stats={"accuracy": 0.5},
                failure_cases=[_make_case("q1", correct=False)],
                success_cases=[_make_case("q2", correct=True)],
            ),
        },
        system_case_pack=None,
        tool_case_packs={},
    )

    with patch("core.harness.evolve.LLMClient") as MockLLM:
        MockLLM.from_env.return_value = mock_client
        result = run_evolution(
            diagnosis=diagnosis,
            workspace_dir=ws,
            store_dir=store,
            skills_dir=skills_v1,
            prompts_dir=prompts_v1,
            db_path=ws / "harness.db",
            targets={"skills"},
        )

    assert result.skills_version == "v2"
    assert result.prompts_version is None
    assert result.accepted_count == 1
    assert result.rejected_count == 0

    # 验证 store 中有 v2
    v2_file = store / "skills" / "v2" / "t.md"
    assert v2_file.exists()
    assert "新内容" in v2_file.read_text()

    # 验证 meta.json
    meta = json.loads((store / "skills" / "v2" / "meta.json").read_text())
    assert meta["source"] == "evolution"
    assert meta["parent"] == "v1"

    # 验证 JSON 快照
    snapshot = ws / "analyses" / "evolution_run_e2e.json"
    assert snapshot.exists()
    data = json.loads(snapshot.read_text())
    assert data["skills_version"] == "v2"
    assert len(data["records"]) == 1

    # 验证 DB
    import sqlite3

    conn = sqlite3.connect(str(ws / "harness.db"))
    rows = conn.execute(
        "SELECT * FROM evolution_records WHERE run_id = 'run_e2e'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1


def test_evolution_e2e_all_skipped(tmp_path: Path) -> None:
    """所有案例包无失败案例时，应全部跳过且不创建新版本。"""
    store = tmp_path / "store"
    skills_v1 = store / "skills" / "v1"
    skills_v1.mkdir(parents=True)
    prompts_v1 = store / "prompts" / "v1"
    prompts_v1.mkdir(parents=True)

    ws = tmp_path / "workspace"
    ws.mkdir()

    diagnosis = DiagnosisResult(
        run_id="run_skip",
        skill_case_packs={
            "T": SkillCasePack(
                task_type="T",
                target_file="t.md",
                stats={},
                failure_cases=[],
                success_cases=[],
            )
        },
        system_case_pack=None,
        tool_case_packs={},
    )

    result = run_evolution(
        diagnosis=diagnosis,
        workspace_dir=ws,
        store_dir=store,
        skills_dir=skills_v1,
        prompts_dir=prompts_v1,
        db_path=ws / "harness.db",
    )

    assert result.skills_version is None
    assert result.prompts_version is None
    assert result.skipped_count == 1
    assert result.accepted_count == 0


def test_evolution_e2e_targets_filter(tmp_path: Path) -> None:
    """targets 参数应过滤掉未指定的目标类型。"""
    store = tmp_path / "store"
    skills_v1 = store / "skills" / "v1"
    skills_v1.mkdir(parents=True)
    prompts_v1 = store / "prompts" / "v1"
    prompts_v1.mkdir(parents=True)

    ws = tmp_path / "workspace"
    ws.mkdir()

    diagnosis = DiagnosisResult(
        run_id="run_filter",
        skill_case_packs={
            "T": SkillCasePack(
                task_type="T",
                target_file="t.md",
                stats={},
                failure_cases=[_make_case("q1", correct=False)],
                success_cases=[],
            )
        },
        system_case_pack=None,
        tool_case_packs={},
    )

    # targets={"system"} 但没有 system_case_pack → 不会调 LLM
    result = run_evolution(
        diagnosis=diagnosis,
        workspace_dir=ws,
        store_dir=store,
        skills_dir=skills_v1,
        prompts_dir=prompts_v1,
        db_path=ws / "harness.db",
        targets={"system"},
    )

    assert result.skills_version is None
    assert result.accepted_count == 0
