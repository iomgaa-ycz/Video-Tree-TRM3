"""Workspace + Store 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

import json

from core.workspace import (
    ResolvedPaths,
    advance_version,
    init_store,
    init_workspace,
    list_versions,
    list_video_ids,
    load_manifest,
    next_version,
    record_run,
    resolve_paths,
    update_manifest,
)


class TestListVersions:
    """列出 store 中某类资源的版本号。"""

    def test_empty_store(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills").mkdir(parents=True)
        assert list_versions(store, "skills") == []

    def test_single_version(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills" / "v1").mkdir(parents=True)
        assert list_versions(store, "skills") == ["v1"]

    def test_multiple_versions_sorted(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        for v in ["v3", "v1", "v10", "v2"]:
            (store / "skills" / v).mkdir(parents=True)
        assert list_versions(store, "skills") == ["v1", "v2", "v3", "v10"]

    def test_ignores_non_version_dirs(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills" / "v1").mkdir(parents=True)
        (store / "skills" / "README.md").touch()
        (store / "skills" / "temp").mkdir()
        assert list_versions(store, "skills") == ["v1"]

    def test_nonexistent_resource_dir(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        assert list_versions(store, "skills") == []

    def test_nested_resource_type(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "questions" / "generated" / "v1").mkdir(parents=True)
        (store / "questions" / "generated" / "v2").mkdir(parents=True)
        assert list_versions(store, "questions/generated") == ["v1", "v2"]


class TestNextVersion:
    """返回下一个可用版本号。"""

    def test_empty_returns_v1(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills").mkdir(parents=True)
        assert next_version(store, "skills") == "v1"

    def test_after_v1_returns_v2(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills" / "v1").mkdir(parents=True)
        assert next_version(store, "skills") == "v2"

    def test_after_v10_returns_v11(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        for v in ["v1", "v5", "v10"]:
            (store / "skills" / v).mkdir(parents=True)
        assert next_version(store, "skills") == "v11"


@pytest.fixture()
def seed_dirs(tmp_path: Path) -> dict[str, Path]:
    """创建测试用的原始资源目录。"""
    videos = tmp_path / "raw_videos"
    (videos / "video_001").mkdir(parents=True)
    (videos / "video_001" / "tree.json").write_text('{"nodes": []}')
    (videos / "video_002").mkdir()
    (videos / "video_002" / "tree.json").write_text('{"nodes": []}')

    skills = tmp_path / "raw_skills"
    skills.mkdir()
    (skills / "search_strategy.md").write_text("# Search Strategy")

    prompts = tmp_path / "raw_prompts"
    prompts.mkdir()
    (prompts / "react_system.md").write_text("# System Prompt")

    return {"videos": videos, "skills": skills, "prompts": prompts}


class TestInitStore:
    """初始化 store 目录结构。"""

    def test_creates_store_structure(
        self, tmp_path: Path, seed_dirs: dict[str, Path]
    ) -> None:
        store = tmp_path / "store"
        init_store(store, seed_dirs["videos"], seed_dirs["skills"], seed_dirs["prompts"])

        assert (store / "videos" / "video_001" / "tree.json").exists()
        assert (store / "videos" / "video_002" / "tree.json").exists()
        assert (store / "questions" / "benchmarks").is_dir()
        assert (store / "questions" / "generated").is_dir()
        assert (store / "skills" / "v1" / "search_strategy.md").exists()
        assert (store / "prompts" / "v1" / "react_system.md").exists()

    def test_writes_meta_json(
        self, tmp_path: Path, seed_dirs: dict[str, Path]
    ) -> None:
        store = tmp_path / "store"
        init_store(store, seed_dirs["videos"], seed_dirs["skills"], seed_dirs["prompts"])

        skills_meta = json.loads((store / "skills" / "v1" / "meta.json").read_text())
        assert skills_meta["version"] == "v1"
        assert skills_meta["parent"] is None
        assert skills_meta["source"] == "manual"

        prompts_meta = json.loads((store / "prompts" / "v1" / "meta.json").read_text())
        assert prompts_meta["version"] == "v1"

    def test_raises_if_store_exists(
        self, tmp_path: Path, seed_dirs: dict[str, Path]
    ) -> None:
        store = tmp_path / "store"
        store.mkdir()
        with pytest.raises(FileExistsError):
            init_store(
                store, seed_dirs["videos"], seed_dirs["skills"], seed_dirs["prompts"]
            )


class TestAdvanceVersion:
    """将新版本资源写入 store。"""

    def test_creates_v2_from_v1(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills" / "v1").mkdir(parents=True)
        (store / "skills" / "v1" / "search.md").write_text("original")

        evolved = tmp_path / "evolved_skills"
        evolved.mkdir()
        (evolved / "search.md").write_text("improved")

        meta = {
            "parent": "v1",
            "source": "evolution",
            "trigger_run": "run_001",
            "trigger_workspace": "exp1",
            "description": "test evolution",
        }
        version = advance_version(store, "skills", evolved, meta)

        assert version == "v2"
        assert (store / "skills" / "v2" / "search.md").read_text() == "improved"

    def test_writes_meta_json_with_version(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "skills" / "v1").mkdir(parents=True)

        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "a.md").write_text("content")

        meta = {"parent": "v1", "source": "evolution"}
        advance_version(store, "skills", evolved, meta)

        written_meta = json.loads(
            (store / "skills" / "v2" / "meta.json").read_text()
        )
        assert written_meta["version"] == "v2"
        assert written_meta["parent"] == "v1"
        assert written_meta["source"] == "evolution"
        assert "created_at" in written_meta

    def test_returns_correct_version_number(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        for v in ["v1", "v2", "v3"]:
            (store / "skills" / v).mkdir(parents=True)

        evolved = tmp_path / "evolved"
        evolved.mkdir()
        (evolved / "a.md").write_text("content")

        version = advance_version(store, "skills", evolved, {"source": "manual"})
        assert version == "v4"
        assert (store / "skills" / "v4" / "a.md").exists()

    def test_nested_resource_type(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "questions" / "generated" / "v1").mkdir(parents=True)

        new_questions = tmp_path / "new_qs"
        new_questions.mkdir()
        (new_questions / "video_001.json").write_text("[]")

        version = advance_version(
            store, "questions/generated", new_questions, {"source": "auto-gen"}
        )
        assert version == "v2"
        assert (
            store / "questions" / "generated" / "v2" / "video_001.json"
        ).exists()


# ---------------------------------------------------------------------------
# Workspace 函数测试
# ---------------------------------------------------------------------------


@pytest.fixture()
def ready_store(tmp_path: Path, seed_dirs: dict[str, Path]) -> Path:
    """返回已初始化的 store 路径。"""
    store = tmp_path / "store"
    init_store(store, seed_dirs["videos"], seed_dirs["skills"], seed_dirs["prompts"])
    (store / "questions" / "benchmarks" / "Video-MME").mkdir(parents=True)
    (store / "questions" / "benchmarks" / "Video-MME" / "video_001.json").write_text(
        "[]"
    )
    return store


class TestInitWorkspace:
    """创建 workspace 并写入 manifest。"""

    def test_creates_workspace_structure(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        assert (ws / "manifest.json").exists()
        assert (ws / "analyses").is_dir()
        assert (ws / "runs").is_dir()

    def test_manifest_content(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        manifest = json.loads((ws / "manifest.json").read_text())
        assert manifest["name"] == "exp1"
        assert manifest["current"]["videos"] == "videos"
        assert manifest["current"]["questions"] == "questions/benchmarks/Video-MME"
        assert manifest["current"]["skills"] == "skills/v1"
        assert manifest["current"]["prompts"] == "prompts/v1"
        assert manifest["history"] == []
        assert "store" in manifest
        assert "created_at" in manifest

    def test_raises_if_workspace_exists(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        ws.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

    def test_raises_if_skills_version_missing(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        with pytest.raises(FileNotFoundError):
            init_workspace(ws, ready_store, "benchmarks/Video-MME", "v99", "v1")

    def test_raises_if_questions_missing(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        with pytest.raises(FileNotFoundError):
            init_workspace(ws, ready_store, "benchmarks/NONEXISTENT", "v1", "v1")


class TestLoadManifest:
    """读取 manifest.json。"""

    def test_loads_manifest(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        manifest = load_manifest(ws)
        assert manifest["name"] == "exp1"
        assert manifest["current"]["skills"] == "skills/v1"

    def test_raises_if_no_manifest(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspaces" / "no_such"
        ws.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            load_manifest(ws)


class TestResolvePaths:
    """从 manifest 解析绝对路径。"""

    def test_resolves_all_paths(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        paths = resolve_paths(ws)
        assert paths.store_dir == ready_store.resolve()
        assert paths.videos_dir == ready_store.resolve() / "videos"
        assert (
            paths.questions_dir
            == ready_store.resolve() / "questions" / "benchmarks" / "Video-MME"
        )
        assert paths.skills_dir == ready_store.resolve() / "skills" / "v1"
        assert paths.prompts_dir == ready_store.resolve() / "prompts" / "v1"
        assert paths.workspace_dir == ws.resolve()
        assert paths.db_path == ws.resolve() / "harness.db"
        assert paths.analyses_dir == ws.resolve() / "analyses"
        assert paths.runs_dir == ws.resolve() / "runs"

    def test_returns_resolved_paths_type(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        paths = resolve_paths(ws)
        assert isinstance(paths, ResolvedPaths)


class TestListVideoIds:
    """列出 workspace 引用的视频 ID。"""

    def test_lists_videos_with_tree_json(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        video_ids = list_video_ids(ws)
        assert video_ids == ["video_001", "video_002"]

    def test_ignores_dirs_without_tree_json(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        (ready_store / "videos" / "no_tree_dir").mkdir()

        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        video_ids = list_video_ids(ws)
        assert "no_tree_dir" not in video_ids


class TestUpdateManifest:
    """更新 manifest 的 current 字段。"""

    def test_updates_skills_version(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")
        (ready_store / "skills" / "v2").mkdir()

        update_manifest(ws, skills="skills/v2")

        manifest = load_manifest(ws)
        assert manifest["current"]["skills"] == "skills/v2"
        assert manifest["current"]["prompts"] == "prompts/v1"

    def test_updates_multiple_fields(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")
        (ready_store / "skills" / "v2").mkdir()
        (ready_store / "prompts" / "v2").mkdir()

        update_manifest(ws, skills="skills/v2", prompts="prompts/v2")

        manifest = load_manifest(ws)
        assert manifest["current"]["skills"] == "skills/v2"
        assert manifest["current"]["prompts"] == "prompts/v2"

    def test_rejects_invalid_key(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        with pytest.raises(KeyError):
            update_manifest(ws, nonexistent_field="value")


class TestRecordRun:
    """将 current 版本快照追加到 history，创建 run 目录。"""

    def test_appends_to_history(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        record_run(ws, "run_001")

        manifest = load_manifest(ws)
        assert len(manifest["history"]) == 1
        entry = manifest["history"][0]
        assert entry["run_id"] == "run_001"
        assert entry["skills"] == "skills/v1"
        assert entry["prompts"] == "prompts/v1"
        assert entry["questions"] == "questions/benchmarks/Video-MME"
        assert "started_at" in entry

    def test_creates_run_directory(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        run_dir = record_run(ws, "run_001")
        assert run_dir.is_dir()
        assert run_dir.name == "run_001"

    def test_creates_per_video_wiki_dirs(
        self, tmp_path: Path, ready_store: Path
    ) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        run_dir = record_run(ws, "run_001")
        assert (run_dir / "video_001" / "wiki").is_dir()
        assert (run_dir / "video_002" / "wiki").is_dir()

    def test_multiple_runs_append(self, tmp_path: Path, ready_store: Path) -> None:
        ws = tmp_path / "workspaces" / "exp1"
        init_workspace(ws, ready_store, "benchmarks/Video-MME", "v1", "v1")

        record_run(ws, "run_001")

        (ready_store / "skills" / "v2").mkdir()
        update_manifest(ws, skills="skills/v2")
        record_run(ws, "run_002")

        manifest = load_manifest(ws)
        assert len(manifest["history"]) == 2
        assert manifest["history"][0]["skills"] == "skills/v1"
        assert manifest["history"][1]["skills"] == "skills/v2"


# ---------------------------------------------------------------------------
# HarnessLog 扩展测试
# ---------------------------------------------------------------------------

from core.harness.log import HarnessLog


class TestHarnessLogVersionColumns:
    """验证 _runs 表包含版本追踪列。"""

    def test_runs_table_has_version_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "harness.db"
        with HarnessLog(str(db_path), "test_run") as log:
            columns = log.query("PRAGMA table_info(_runs)")
        col_names = {c["name"] for c in columns}
        assert "skills_version" in col_names
        assert "prompts_version" in col_names
        assert "questions_ref" in col_names

    def test_insert_run_with_version_info(self, tmp_path: Path) -> None:
        db_path = tmp_path / "harness.db"
        with HarnessLog(str(db_path), "run_001") as log:
            log.execute(
                "UPDATE _runs SET skills_version=?, prompts_version=?, questions_ref=? WHERE run_id=?",
                ("v2", "v1", "benchmarks/Video-MME", "run_001"),
            )
            rows = log.query(
                "SELECT skills_version, prompts_version, questions_ref FROM _runs WHERE run_id=?",
                ("run_001",),
            )
        assert rows[0]["skills_version"] == "v2"
        assert rows[0]["prompts_version"] == "v1"
        assert rows[0]["questions_ref"] == "benchmarks/Video-MME"
