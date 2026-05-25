"""Workspace + Store 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

import json

from core.workspace import (
    ResolvedPaths,
    advance_version,
    init_store,
    list_versions,
    next_version,
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
