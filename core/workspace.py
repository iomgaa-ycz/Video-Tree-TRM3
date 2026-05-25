"""Store（资源仓库）与 Workspace（实验工作区）管理。

Store 存储版本化资源（视频、题目、Skill、Prompt），
Workspace 通过 manifest.json 引用 Store 中特定版本并记录实验过程。
"""

from __future__ import annotations

import json
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
        store_dir / "skills" / "v1", "v1", "manual",
        description="手工创建的初始版本",
    )
    shutil.copytree(prompts_dir, store_dir / "prompts" / "v1")
    _write_meta(
        store_dir / "prompts" / "v1", "v1", "manual",
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
