"""从 research-wiki 构建 manifest.json 数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.wiki_site._wiki_helpers import ENTITY_TYPES, read_frontmatter


@dataclass
class ManifestEntry:
    """单个实体的 manifest 条目。"""

    node_id: str
    title: str
    entity_type: str
    date: str
    page_path: str
    edges: list[dict[str, Any]] = field(default_factory=list)


def _load_edges(wiki_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """按 node_id 分组加载所有边。"""
    graph_path = wiki_dir / "graph" / "edges.json"
    if not graph_path.exists():
        return {}

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for link in graph.get("links", []):
        source = str(link.get("source", ""))
        grouped.setdefault(source, []).append(link)
        target = str(link.get("target", ""))
        if target != source:
            grouped.setdefault(target, []).append(link)
    return grouped


def build_manifest(wiki_dir: Path) -> list[ManifestEntry]:
    """扫描 wiki 目录构建 manifest 条目列表。

    参数:
        wiki_dir: research-wiki/ 根目录。

    返回:
        ManifestEntry 列表。
    """
    edges_by_node = _load_edges(wiki_dir)
    entries: list[ManifestEntry] = []

    for entity_type, subdir_name in ENTITY_TYPES.items():
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for md_path in sorted(subdir.glob("*.md")):
            fm = read_frontmatter(md_path)
            node_id = str(fm.get("node_id", f"{entity_type}:{md_path.stem}"))
            title = str(fm.get("title", md_path.stem))
            date = str(fm.get("date", ""))
            page_path = f"{subdir_name}/{md_path.stem}"

            entry = ManifestEntry(
                node_id=node_id,
                title=title,
                entity_type=entity_type,
                date=date,
                page_path=page_path,
                edges=edges_by_node.get(node_id, []),
            )
            entries.append(entry)

    return entries


def write_manifest_json(entries: list[ManifestEntry], output_path: Path) -> None:
    """将 manifest 写入 JSON 文件。

    参数:
        entries: ManifestEntry 列表。
        output_path: 输出 JSON 路径。
    """
    data = [
        {
            "node_id": e.node_id,
            "title": e.title,
            "entity_type": e.entity_type,
            "date": e.date,
            "page_path": e.page_path,
            "edges": e.edges,
        }
        for e in entries
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
