"""从 research_wiki.py 提取的共享解析函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENTITY_TYPES = {
    "paper": "papers",
    "idea": "ideas",
    "experiment": "experiments",
    "claim": "claims",
    "gap": "gaps",
    "design": "designs",
    "finding": "findings",
    "plan": "plans",
    "review": "reviews",
    "schema": "schemas",
    "metric": "metrics",
}


def read_frontmatter(filepath: Path) -> dict[str, Any]:
    """从 markdown 文件读取 YAML frontmatter。"""
    text = filepath.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}

    result: dict[str, Any] = {}
    for line in text.splitlines()[1:]:
        if line == "---":
            break
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        value = raw_value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                result[key.strip()] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        result[key.strip()] = value
    return result


def markdown_body(text: str) -> str:
    """提取 markdown 中 frontmatter 之后的正文。"""
    if not text.startswith("---\n"):
        return text
    lines = text.splitlines()
    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            return "\n".join(lines[index + 1 :])
    return ""
