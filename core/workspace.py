"""Store（资源仓库）与 Workspace（实验工作区）管理。

Store 存储版本化资源（视频、题目、Skill、Prompt），
Workspace 通过 manifest.json 引用 Store 中特定版本并记录实验过程。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
