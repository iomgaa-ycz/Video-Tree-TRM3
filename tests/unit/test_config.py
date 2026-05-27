"""RunConfig 与 load_config 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.harness.config import load_config


@pytest.fixture()
def yaml_file(tmp_path: Path) -> Path:
    """创建临时 YAML 配置文件。"""
    data = {
        "workspace_dir": "workspaces/default",
        "store_dir": "store",
        "mode": "infer",
        "run_id": "",
        "concurrency": 12,
        "max_steps": 15,
        "skill_mode": "auto",
        "n_samples": 0,
        "questions": "benchmarks/Video-MME",
        "skills_version": "v1",
        "prompts_version": "v1",
        "epochs": 1,
    }
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.dump(data))
    return p


class TestLoadConfig:
    """load_config 加载与覆盖逻辑测试。"""

    def test_load_from_yaml(self, yaml_file: Path) -> None:
        """纯 YAML 加载，无 CLI 覆盖。"""
        config = load_config(yaml_file, cli_overrides={})
        assert config.concurrency == 12
        assert config.mode == "infer"
        assert config.run_id == ""
        assert config.max_steps == 15
        assert config.skill_mode == "auto"
        assert config.n_samples == 0
        assert config.questions == "benchmarks/Video-MME"
        assert config.skills_version == "v1"
        assert config.prompts_version == "v1"
        assert config.epochs == 1

    def test_cli_overrides_yaml(self, yaml_file: Path) -> None:
        """CLI 参数覆盖 YAML 值。"""
        overrides = {"concurrency": 4, "n_samples": 30, "mode": "train"}
        config = load_config(yaml_file, cli_overrides=overrides)
        assert config.concurrency == 4
        assert config.n_samples == 30
        assert config.mode == "train"
        assert config.max_steps == 15

    def test_none_overrides_ignored(self, yaml_file: Path) -> None:
        """CLI 中为 None 的字段不覆盖 YAML 值。"""
        overrides = {"concurrency": None, "n_samples": None}
        config = load_config(yaml_file, cli_overrides=overrides)
        assert config.concurrency == 12
        assert config.n_samples == 0

    def test_path_fields_are_path_objects(self, yaml_file: Path) -> None:
        """workspace_dir 和 store_dir 是 Path 类型。"""
        config = load_config(yaml_file, cli_overrides={})
        assert isinstance(config.workspace_dir, Path)
        assert isinstance(config.store_dir, Path)

    def test_config_key_not_in_yaml(self, yaml_file: Path) -> None:
        """CLI 传入非法 key 时，仅使用 YAML 中存在的字段。"""
        overrides = {"config": "some/path.yaml", "unknown_field": 42}
        config = load_config(yaml_file, cli_overrides=overrides)
        assert config.concurrency == 12


class TestRunConfigValidation:
    """RunConfig 字段约束测试。"""

    def test_invalid_mode(self, yaml_file: Path) -> None:
        """非法 mode 值应报错。"""
        with pytest.raises(ValueError, match="mode"):
            load_config(yaml_file, cli_overrides={"mode": "invalid"})

    def test_invalid_skill_mode(self, yaml_file: Path) -> None:
        """非法 skill_mode 值应报错。"""
        with pytest.raises(ValueError, match="skill_mode"):
            load_config(yaml_file, cli_overrides={"skill_mode": "wrong"})

    def test_negative_concurrency(self, yaml_file: Path) -> None:
        """concurrency <= 0 应报错。"""
        with pytest.raises(ValueError, match="concurrency"):
            load_config(yaml_file, cli_overrides={"concurrency": 0})

    def test_diagnose_mode_requires_run_id(self, yaml_file: Path) -> None:
        """diagnose 模式缺少 run_id 应报错。"""
        with pytest.raises(ValueError, match="run_id"):
            load_config(yaml_file, cli_overrides={"mode": "diagnose", "run_id": ""})
