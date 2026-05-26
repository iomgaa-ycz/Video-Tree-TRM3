from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from core.harness.log import HarnessLog
from core.harness.question_gen import GeneratedQuestion
from core.llm_client import LLMClient
from core.loop import AgentLoop, LoopResult, Step, hookimpl
from core.search.prompt import PromptManager
from core.search.skills import SkillRegistry
from core.tree.tools import dispatch


@dataclass(frozen=True)
class InferenceResult:
    run_id: str
    accuracy: float
    total: int
    correct: int
    per_task_type: dict[str, dict]
    steps_mean: float
    token_usage: dict[str, int]
    stop_reason_counts: dict[str, int]


class TracePlugin:
    """记录单题推理轨迹与层级访问统计的 pluggy 插件。"""

    def __init__(self, log: Any, video_id: str, question_id: str) -> None:
        self._log = log
        self._video_id = video_id
        self._question_id = question_id

    @hookimpl
    def after_tool(self, iteration: int, step: Step) -> str | None:
        self._log.insert(
            "traces",
            {
                "video_id": self._video_id,
                "question_id": self._question_id,
                "step": iteration,
                "tool_name": step.tool_call["tool"],
                "tool_args": json.dumps(step.tool_call.get("args", {})),
                "tool_output": step.tool_output,
                "thought": step.thought,
            },
        )
        return None

    @hookimpl
    def on_finish(self, result: LoopResult) -> None:
        """统计 view_node 的层级访问次数，写入 validation_flags 表。"""
        l1, l2, l3 = 0, 0, 0
        for step in result.steps:
            if step.tool_call.get("tool") != "view_node":
                continue
            node_id = step.tool_call.get("args", {}).get("node_id", "")
            matches = re.findall(r"_L(\d+)_", node_id)
            if matches:
                level = int(matches[-1])
                if level == 1:
                    l1 += 1
                elif level == 2:
                    l2 += 1
                elif level == 3:
                    l3 += 1
        self._log.insert(
            "validation_flags",
            {
                "video_id": self._video_id,
                "question_id": self._question_id,
                "has_l3_visit": 1 if l3 > 0 else 0,
                "l1_count": l1,
                "l2_count": l2,
                "l3_count": l3,
            },
        )


def _run_single_question(
    qa: GeneratedQuestion,
    env: Any,
    vl_client: Any,
    prompt_mgr: PromptManager,
    skill_registry: SkillRegistry,
    log: HarnessLog,
    max_steps: int,
    skill_mode: str,
    always_skills_text: str,
    task_skill_map: dict[str, str],
    catalog_text: str,
    prompts_dir: Path,
) -> dict[str, Any]:
    """执行单道题目的 Agent 推理。

    创建独立的 search_client 和 AgentLoop（线程安全），
    通过 PromptManager 组装 prompt，运行循环，结果写入 log。

    参数:
        qa: 待推理的题目。
        env: TreeEnvironment 实例（同 video_id 的题目共享）。
        vl_client: 视觉模型 LLMClient（共享）。
        prompt_mgr: PromptManager 实例。
        skill_registry: SkillRegistry 实例。
        log: HarnessLog 实例（线程安全）。
        max_steps: AgentLoop 最大步数。
        skill_mode: "auto" / "manual" / "none"。
        always_skills_text: always 层 skill 全文。
        task_skill_map: {task_type: skill_body} 映射。
        catalog_text: manual 模式的 skill 目录文本。
        prompts_dir: prompt 文件目录。

    返回:
        预测结果字典。
    """
    record: dict[str, Any] = {
        "video_id": qa.video_id,
        "question_id": qa.question_id,
        "task_type": qa.task_type,
        "prediction": None,
        "answer": qa.answer,
        "evidence": "",
        "reasoning": "",
        "steps_used": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "stop_reason": "error",
        "steps_json": "[]",
    }

    try:
        search_client = LLMClient.from_env("SEARCH_LLM", thinking=True)
        loop = AgentLoop(search_client, max_steps=max_steps)

        system_prompt = prompt_mgr.build_inference_prompt(
            skill_mode=skill_mode,
            task_type=qa.task_type,
            always_skills_text=always_skills_text,
            task_skill_map=task_skill_map,
            catalog_text=catalog_text,
        )

        l1_ids = sorted(
            nid for nid, node in env._nodes.items() if node.get("level") == 1
        )
        qa_dict = {"question": qa.question, "options": qa.options}
        user_prompt = prompt_mgr.format_user_prompt(qa_dict, l1_ids)

        trace_plugin = TracePlugin(log, qa.video_id, qa.question_id)
        tool_fn = partial(
            dispatch,
            env=env,
            vl_client=vl_client,
            prompts_dir=prompts_dir,
            skills=skill_registry if skill_mode == "manual" else None,
        )

        loop_result = loop.run(
            system_prompt, user_prompt, tool_fn, plugins=[trace_plugin]
        )

        result_dict = loop_result.result if isinstance(loop_result.result, dict) else {}
        record.update(
            {
                "prediction": result_dict.get("answer"),
                "evidence": result_dict.get("evidence", ""),
                "reasoning": result_dict.get("reasoning", ""),
                "steps_used": loop_result.steps_used,
                "prompt_tokens": loop_result.token_usage["prompt_tokens"],
                "completion_tokens": loop_result.token_usage["completion_tokens"],
                "stop_reason": loop_result.stop_reason,
                "steps_json": json.dumps(
                    [
                        {
                            "thought": s.thought,
                            "tool_call": s.tool_call,
                            "tool_output": s.tool_output,
                        }
                        for s in loop_result.steps
                    ],
                    ensure_ascii=False,
                ),
            }
        )
    except Exception:
        logger.exception("[{}] QA {} 执行异常", qa.video_id, qa.question_id)

    log.insert("predictions", record)
    return record
