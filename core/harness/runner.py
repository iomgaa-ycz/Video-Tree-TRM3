"""实验运行器，对标 PyTorch Trainer。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from core.harness.config import RunConfig
from core.harness.inference import InferenceResult, run_inference
from core.harness.question_gen import load_benchmark
from core.workspace import ResolvedPaths, init_workspace, resolve_paths

if TYPE_CHECKING:
    from core.harness.diagnose import DiagnosisResult


class Runner:
    """实验运行器，通过 RunConfig 驱动不同阶段。

    参数:
        config: 运行配置。
    """

    def __init__(self, config: RunConfig) -> None:
        self._config = config
        self._ensure_workspace()
        self._paths: ResolvedPaths = resolve_paths(config.workspace_dir)

    def _ensure_workspace(self) -> None:
        """若 workspace 不存在则自动创建。"""
        manifest = self._config.workspace_dir / "manifest.json"
        if not manifest.exists():
            logger.info("Workspace 不存在，自动创建: {}", self._config.workspace_dir)
            init_workspace(
                workspace_dir=self._config.workspace_dir,
                store_dir=self._config.store_dir,
                questions=self._config.questions,
                skills_version=self._config.skills_version,
                prompts_version=self._config.prompts_version,
            )

    def infer(self) -> InferenceResult:
        """执行单次推理（forward-only）。

        加载 benchmark 题目，可选截取前 n_samples 条，
        调用 run_inference 执行 Agent 推理并返回聚合结果。

        返回:
            InferenceResult 冻结实例。
        """
        questions = load_benchmark(self._paths.questions_dir)
        if self._config.n_samples > 0:
            questions = questions[: self._config.n_samples]

        logger.info(
            "启动推理: {} 道题, concurrency={}, max_steps={}, skill_mode={}",
            len(questions),
            self._config.concurrency,
            self._config.max_steps,
            self._config.skill_mode,
        )

        return run_inference(
            workspace_dir=self._config.workspace_dir,
            questions=questions,
            concurrency=self._config.concurrency,
            max_steps=self._config.max_steps,
            skill_mode=self._config.skill_mode,
        )

    def diagnose(self, run_id: str | None = None, **filters) -> "DiagnosisResult":
        """执行指定 run 的两阶段诊断。"""
        effective_run_id = run_id or self._config.run_id
        if not effective_run_id:
            raise ValueError("diagnose 模式必须提供 run_id。")

        from core.harness.diagnose import run_diagnosis
        from core.harness.log import HarnessLog

        with HarnessLog(str(self._paths.db_path), effective_run_id) as log:
            return run_diagnosis(
                log=log,
                run_id=effective_run_id,
                workspace_dir=self._config.workspace_dir,
                skills_dir=self._paths.skills_dir,
                concurrency=self._config.concurrency,
                **filters,
            )
