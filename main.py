"""命令行入口：解析参数 → 加载配置 → 调度 Runner。"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from core.harness.config import load_config
from core.harness.runner import Runner


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    所有参数 default=None，表示未传入，使用 YAML 默认值。

    返回:
        配置好的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(
        description="Video-Tree 实验运行器",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML 配置文件路径（默认: config/default.yaml）",
    )
    parser.add_argument("--workspace-dir", type=Path, dest="workspace_dir")
    parser.add_argument("--store-dir", type=Path, dest="store_dir")
    parser.add_argument("--mode", choices=["infer", "train"])
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument(
        "--skill-mode",
        choices=["auto", "manual", "none"],
        dest="skill_mode",
    )
    parser.add_argument("--n-samples", type=int, dest="n_samples")
    parser.add_argument("--questions", type=str)
    parser.add_argument("--skills-version", type=str, dest="skills_version")
    parser.add_argument("--prompts-version", type=str, dest="prompts_version")
    parser.add_argument("--epochs", type=int)
    return parser


def _print_result(result) -> None:
    """打印推理结果摘要。

    参数:
        result: InferenceResult 实例。
    """
    logger.info("=" * 60)
    logger.info("运行 ID: {}", result.run_id)
    logger.info(
        "总体准确率: {:.2%} ({}/{})",
        result.accuracy,
        result.correct,
        result.total,
    )
    logger.info("平均步数: {:.1f}", result.steps_mean)
    logger.info(
        "Token 用量: prompt={}, completion={}",
        result.token_usage["prompt_tokens"],
        result.token_usage["completion_tokens"],
    )
    if result.per_task_type:
        logger.info("--- 按任务类型 ---")
        for task_type, stats in sorted(result.per_task_type.items()):
            logger.info(
                "  {}: {:.2%} ({}/{})",
                task_type,
                stats["accuracy"],
                stats["correct"],
                stats["total"],
            )
    logger.info(
        "停止原因: {}",
        ", ".join(f"{k}={v}" for k, v in result.stop_reason_counts.items()),
    )
    logger.info("=" * 60)


def main() -> None:
    """入口函数。"""
    parser = _build_parser()
    args = parser.parse_args()

    config_path = args.config
    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}
    config = load_config(config_path, cli_overrides)

    logger.info(
        "配置加载完成: mode={}, workspace={}", config.mode, config.workspace_dir
    )

    runner = Runner(config)

    if config.mode == "infer":
        result = runner.infer()
        _print_result(result)
    elif config.mode == "train":
        runner.train()


if __name__ == "__main__":
    main()
