"""运行配置：RunConfig dataclass 与 YAML+CLI 加载。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_MODES = {'infer', 'train'}
_VALID_SKILL_MODES = {'auto', 'manual', 'none'}
_PATH_FIELDS = {'workspace_dir', 'store_dir'}


@dataclass(frozen=True)
class RunConfig:
    """实验运行配置，所有参数的唯一归口。

    字段:
        workspace_dir: Workspace 根目录。
        store_dir: Store 根目录。
        mode: 运行模式，"infer" 或 "train"。
        concurrency: 并行 worker 数。
        max_steps: AgentLoop 单题最大步数。
        skill_mode: Skill 加载模式，"auto" / "manual" / "none"。
        n_samples: 题目截取数，0 表示全量。
        questions: 题目在 questions/ 下的相对路径。
        skills_version: Skills 版本号。
        prompts_version: Prompts 版本号。
        epochs: 训练轮数。
    """

    workspace_dir: Path
    store_dir: Path
    mode: str
    concurrency: int
    max_steps: int
    skill_mode: str
    n_samples: int
    questions: str
    skills_version: str
    prompts_version: str
    epochs: int


def _validate(config: RunConfig) -> None:
    """校验 RunConfig 字段约束。

    参数:
        config: 待校验的配置实例。

    异常:
        ValueError: 字段值不合法。
    """
    if config.mode not in _VALID_MODES:
        raise ValueError(f'mode 必须为 {_VALID_MODES} 之一，实际: {config.mode!r}')
    if config.skill_mode not in _VALID_SKILL_MODES:
        raise ValueError(
            f'skill_mode 必须为 {_VALID_SKILL_MODES} 之一，实际: {config.skill_mode!r}'
        )
    if config.concurrency <= 0:
        raise ValueError(f'concurrency 必须 > 0，实际: {config.concurrency}')
    if config.max_steps <= 0:
        raise ValueError(f'max_steps 必须 > 0，实际: {config.max_steps}')
    if config.n_samples < 0:
        raise ValueError(f'n_samples 必须 >= 0，实际: {config.n_samples}')
    if config.epochs <= 0:
        raise ValueError(f'epochs 必须 > 0，实际: {config.epochs}')


def load_config(yaml_path: Path, cli_overrides: dict) -> RunConfig:
    """从 YAML 加载配置，CLI 参数覆盖非 None 字段。

    参数:
        yaml_path: YAML 配置文件路径。
        cli_overrides: CLI 参数字典，值为 None 表示未传入。

    返回:
        构造并校验后的 RunConfig 实例。
    """
    with open(yaml_path, encoding='utf-8') as f:
        yaml_data: dict = yaml.safe_load(f)

    valid_fields = {f.name for f in dataclasses.fields(RunConfig)}
    for key, value in cli_overrides.items():
        if value is not None and key in valid_fields:
            yaml_data[key] = value

    for field_name in _PATH_FIELDS:
        yaml_data[field_name] = Path(yaml_data[field_name])

    config = RunConfig(**{k: v for k, v in yaml_data.items() if k in valid_fields})
    _validate(config)
    return config
