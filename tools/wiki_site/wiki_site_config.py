"""Wiki-Site 配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WikiSiteConfig:
    """Wiki-Site 渲染系统配置。"""

    project_name: str = "Research Wiki"
    primary_color: str = "#4c6ef5"
    port: int = 8686
    model: str = "claude-opus-4-6"
    temperature: int = 0
    debounce_seconds: int = 2
    auto_shutdown_minutes: int = 30
    max_retries_on_compile_error: int = 1


def load_config(config_path: Path) -> WikiSiteConfig:
    """从 YAML 文件加载配置，缺失字段用默认值填充。

    参数:
        config_path: 配置文件路径。

    返回:
        WikiSiteConfig 实例。
    """
    if not config_path.exists():
        return WikiSiteConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return WikiSiteConfig()

    known_fields = {f.name for f in WikiSiteConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}
    return WikiSiteConfig(**filtered)
