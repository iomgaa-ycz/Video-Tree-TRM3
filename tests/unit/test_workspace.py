"""Workspace + Store 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.workspace import ResolvedPaths, list_versions, next_version


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
