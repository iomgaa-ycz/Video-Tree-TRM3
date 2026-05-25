"""技能注册表与 Markdown frontmatter 解析工具。"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_FRONTMATTER_FIELDS = {"name", "description", "always", "task_type"}


def _extract_frontmatter_lines(text: str) -> tuple[list[str], int] | None:
    """提取 frontmatter 行与正文起始偏移。"""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None

    offset = len(lines[0])
    frontmatter_lines: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return frontmatter_lines, offset + len(line)
        frontmatter_lines.append(line)
        offset += len(line)

    logger.debug("frontmatter 缺少结束分隔符，按普通正文处理")
    return None


def strip_frontmatter(text: str) -> str:
    """移除 Markdown 文本开头的 frontmatter，并返回正文。

    参数:
        text: 原始 Markdown 文本。

    返回:
        去除 frontmatter 后的正文；若 frontmatter 不完整或不存在，则返回原文。
    """
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return text

    _, body_start = extracted
    return text[body_start:]


def parse_frontmatter(text: str) -> dict[str, str]:
    """解析 Markdown frontmatter 中的目标字段。

    参数:
        text: 原始 Markdown 文本。

    返回:
        仅包含 `name`、`description`、`always`、`task_type` 的字符串字典。
        若不存在完整 frontmatter，则返回空字典。
    """
    extracted = _extract_frontmatter_lines(text)
    if extracted is None:
        return {}

    frontmatter_lines, _ = extracted
    parsed: dict[str, str] = {}
    for raw_line in frontmatter_lines:
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        key, _, raw_value = line.partition(":")
        normalized_key = key.strip()
        if normalized_key not in _FRONTMATTER_FIELDS:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        parsed[normalized_key] = value

    return parsed


class SkillRegistry:
    """管理技能名称到文件路径映射并读取技能正文。"""

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}

    def set_paths(self, mapping: dict[str, Path]) -> None:
        """注入技能名称到文件路径的映射。

        参数:
            mapping: 技能名到 Markdown 文件路径的映射。
        """
        self._paths = dict(mapping)
        logger.debug("SkillRegistry 已载入 {} 个技能路径", len(self._paths))

    def read(self, name: str) -> str:
        """读取指定技能文件，并返回去除 frontmatter 后的正文。

        参数:
            name: 技能名称。

        返回:
            技能 Markdown 正文。

        异常:
            KeyError: 技能名称未注册时抛出。
        """
        try:
            path = self._paths[name]
        except KeyError:
            logger.error("技能未注册: {}", name)
            raise

        logger.debug("读取技能文件: name={}, path={}", name, path)
        return strip_frontmatter(path.read_text(encoding="utf-8"))
