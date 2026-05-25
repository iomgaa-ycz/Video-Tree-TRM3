"""Store（资源仓库）与 Workspace（实验工作区）管理。

Store 存储版本化资源（视频、题目、Skill、Prompt），
Workspace 通过 manifest.json 引用 Store 中特定版本并记录实验过程。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ResolvedPaths:
    """manifest 解析后的绝对路径集合。

    属性:
        store_dir: Store 根目录绝对路径。
        videos_dir: 视频数据目录。
        questions_dir: 当前引用的题目目录。
        skills_dir: 当前引用的 Skill 版本目录。
        prompts_dir: 当前引用的 Prompt 版本目录。
        workspace_dir: Workspace 根目录。
        db_path: harness.db 路径。
        analyses_dir: 分析报告目录。
        runs_dir: 运行临时状态目录。
    """

    store_dir: Path
    videos_dir: Path
    questions_dir: Path
    skills_dir: Path
    prompts_dir: Path
    workspace_dir: Path
    db_path: Path
    analyses_dir: Path
    runs_dir: Path


def _parse_version(name: str) -> int:
    """解析版本目录名为整数。

    参数:
        name: 版本目录名，如 "v1"、"v10"。

    返回:
        版本号整数。

    异常:
        ValueError: 版本目录名格式不合法。
    """
    match = re.match(r"v(\d+)$", name)
    if not match:
        raise ValueError(f"无效版本号: {name}")
    return int(match.group(1))


def list_versions(store_dir: Path, resource_type: str) -> list[str]:
    """列出 store 中某类资源的所有版本号，按数字排序。

    参数:
        store_dir: Store 根目录。
        resource_type: 资源类型路径，如 "skills"、"questions/generated"。

    返回:
        排序后的版本号列表，如 ["v1", "v2", "v10"]。
    """
    resource_dir = store_dir / resource_type
    if not resource_dir.is_dir():
        return []
    versions = []
    for entry in resource_dir.iterdir():
        if entry.is_dir() and re.match(r"v\d+$", entry.name):
            versions.append(entry.name)
    return sorted(versions, key=_parse_version)


def next_version(store_dir: Path, resource_type: str) -> str:
    """返回某类资源的下一个可用版本号。

    参数:
        store_dir: Store 根目录。
        resource_type: 资源类型路径。

    返回:
        下一个版本号字符串，如 "v3"。
    """
    versions = list_versions(store_dir, resource_type)
    if not versions:
        return "v1"
    latest = _parse_version(versions[-1])
    return f"v{latest + 1}"


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _write_meta(
    target_dir: Path, version: str, source: str, **extra: str | None
) -> None:
    """写入版本元数据文件。

    参数:
        target_dir: 版本目录。
        version: 版本号。
        source: 来源标识（"manual" / "evolution" / "auto-gen"）。
        **extra: 额外字段（parent, trigger_run, trigger_workspace, description）。
    """
    meta = {
        "version": version,
        "created_at": _now_iso(),
        "parent": extra.get("parent"),
        "source": source,
        "trigger_run": extra.get("trigger_run"),
        "trigger_workspace": extra.get("trigger_workspace"),
        "description": extra.get("description", ""),
    }
    (target_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )


def init_store(
    store_dir: Path,
    videos_source: Path,
    skills_dir: Path,
    prompts_dir: Path,
) -> None:
    """初始化 Store：拷贝视频数据，创建 skills/v1、prompts/v1 和 questions 目录。

    参数:
        store_dir: Store 目标路径（不得已存在）。
        videos_source: 视频数据源目录。
        skills_dir: 初始 Skill 文件目录。
        prompts_dir: 初始 Prompt 文件目录。

    异常:
        FileExistsError: Store 目录已存在。
    """
    if store_dir.exists():
        raise FileExistsError(f"Store 已存在: {store_dir}")
    store_dir.mkdir(parents=True)
    shutil.copytree(videos_source, store_dir / "videos")
    (store_dir / "questions" / "benchmarks").mkdir(parents=True)
    (store_dir / "questions" / "generated").mkdir(parents=True)
    shutil.copytree(skills_dir, store_dir / "skills" / "v1")
    _write_meta(
        store_dir / "skills" / "v1",
        "v1",
        "manual",
        description="手工创建的初始版本",
    )
    shutil.copytree(prompts_dir, store_dir / "prompts" / "v1")
    _write_meta(
        store_dir / "prompts" / "v1",
        "v1",
        "manual",
        description="手工创建的初始版本",
    )


def advance_version(
    store_dir: Path,
    resource_type: str,
    source_dir: Path,
    meta: dict,
) -> str:
    """将 source_dir 的内容写入 store 的下一个版本目录，写入 meta.json。

    参数:
        store_dir: Store 根目录。
        resource_type: 资源类型路径，如 "skills"、"questions/generated"。
        source_dir: 包含新版本资源文件的源目录。
        meta: 元数据字典，至少包含 source 字段。

    返回:
        新版本号字符串，如 "v2"。
    """
    version = next_version(store_dir, resource_type)
    target = store_dir / resource_type / version
    shutil.copytree(source_dir, target)
    _write_meta(
        target,
        version,
        meta.get("source", "manual"),
        parent=meta.get("parent"),
        trigger_run=meta.get("trigger_run"),
        trigger_workspace=meta.get("trigger_workspace"),
        description=meta.get("description", ""),
    )
    return version


# ---------------------------------------------------------------------------
# Workspace 函数
# ---------------------------------------------------------------------------

_MANIFEST_CURRENT_KEYS = {"videos", "questions", "skills", "prompts"}


def init_workspace(
    workspace_dir: Path,
    store_dir: Path,
    questions: str,
    skills_version: str,
    prompts_version: str,
) -> None:
    """创建 Workspace 目录并写入初始 manifest.json。

    参数:
        workspace_dir: Workspace 目标路径（不得已存在）。
        store_dir: Store 根目录。
        questions: 题目在 questions/ 下的相对路径，如 "benchmarks/Video-MME"。
        skills_version: Skills 版本号，如 "v1"。
        prompts_version: Prompts 版本号，如 "v1"。

    异常:
        FileExistsError: Workspace 目录已存在。
        FileNotFoundError: 引用的资源在 Store 中不存在。
    """
    if workspace_dir.exists():
        raise FileExistsError(f"Workspace 已存在: {workspace_dir}")
    store_abs = store_dir.resolve()
    refs = {
        "skills": f"skills/{skills_version}",
        "prompts": f"prompts/{prompts_version}",
        "questions": f"questions/{questions}",
    }
    for label, rel in refs.items():
        full = store_abs / rel
        if not full.is_dir():
            raise FileNotFoundError(f"Store 中不存在 {label}: {full}")

    workspace_dir.mkdir(parents=True)
    (workspace_dir / "analyses").mkdir()
    (workspace_dir / "runs").mkdir()
    store_rel = os.path.relpath(store_abs, workspace_dir.resolve())
    manifest = {
        "name": workspace_dir.name,
        "created_at": _now_iso(),
        "store": store_rel,
        "current": {
            "videos": "videos",
            "questions": refs["questions"],
            "skills": refs["skills"],
            "prompts": refs["prompts"],
        },
        "history": [],
    }
    (workspace_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


def load_manifest(workspace_dir: Path) -> dict:
    """读取并返回 workspace 的 manifest.json。

    参数:
        workspace_dir: Workspace 根目录。

    返回:
        manifest 字典。

    异常:
        FileNotFoundError: manifest.json 不存在。
    """
    manifest_path = workspace_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json 不存在: {manifest_path}")
    return json.loads(manifest_path.read_text())


def resolve_paths(workspace_dir: Path) -> ResolvedPaths:
    """读取 manifest，解析 current 中所有资源的绝对路径。

    参数:
        workspace_dir: Workspace 根目录。

    返回:
        ResolvedPaths 实例，包含所有资源的绝对路径。
    """
    manifest = load_manifest(workspace_dir)
    ws_abs = workspace_dir.resolve()
    store_abs = (ws_abs / manifest["store"]).resolve()
    current = manifest["current"]
    return ResolvedPaths(
        store_dir=store_abs,
        videos_dir=store_abs / current["videos"],
        questions_dir=store_abs / current["questions"],
        skills_dir=store_abs / current["skills"],
        prompts_dir=store_abs / current["prompts"],
        workspace_dir=ws_abs,
        db_path=ws_abs / "harness.db",
        analyses_dir=ws_abs / "analyses",
        runs_dir=ws_abs / "runs",
    )


def list_video_ids(workspace_dir: Path) -> list[str]:
    """列出 workspace 引用的所有视频 ID（含 tree.json 的子目录名）。

    参数:
        workspace_dir: Workspace 根目录。

    返回:
        排序后的视频 ID 列表。
    """
    paths = resolve_paths(workspace_dir)
    video_ids = []
    for entry in paths.videos_dir.iterdir():
        if entry.is_dir() and (entry / "tree.json").exists():
            video_ids.append(entry.name)
    return sorted(video_ids)


def update_manifest(workspace_dir: Path, **version_updates: str) -> None:
    """更新 manifest 的 current 字段。

    参数:
        workspace_dir: Workspace 根目录。
        **version_updates: 要更新的字段及其新值，如 skills="skills/v2"。

    异常:
        KeyError: 更新的字段不在 current 允许的 key 中。
    """
    invalid = set(version_updates) - _MANIFEST_CURRENT_KEYS
    if invalid:
        raise KeyError(f"无效的 manifest current 字段: {invalid}")
    manifest = load_manifest(workspace_dir)
    manifest["current"].update(version_updates)
    (workspace_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


def record_run(workspace_dir: Path, run_id: str) -> Path:
    """将 current 版本快照追加到 manifest history，创建 run 目录和 per-video wiki 目录。

    参数:
        workspace_dir: Workspace 根目录。
        run_id: 本次运行的唯一标识，如 "run_001"。

    返回:
        创建的 run 目录路径。
    """
    manifest = load_manifest(workspace_dir)
    current = manifest["current"]
    entry = {
        "run_id": run_id,
        "started_at": _now_iso(),
        "skills": current["skills"],
        "prompts": current["prompts"],
        "questions": current["questions"],
    }
    manifest["history"].append(entry)
    (workspace_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    run_dir = workspace_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    for video_id in list_video_ids(workspace_dir):
        (run_dir / video_id / "wiki").mkdir(parents=True)
    return run_dir
