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
        counts = {"1": 0, "2": 0, "3": 0}
        for step in result.steps:
            if step.tool_call["tool"] != "view_node":
                continue
            node_id = step.tool_call.get("args", {}).get("node_id", "")
            match = re.search(r"_L(\d+)_", node_id)
            if match and match.group(1) in counts:
                counts[match.group(1)] += 1
        self._log.insert(
            "validation_flags",
            {
                "video_id": self._video_id,
                "question_id": self._question_id,
                "l1_visits": counts["1"],
                "l2_visits": counts["2"],
                "l3_visits": counts["3"],
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
    record = {
        "video_id": qa.video_id,
        "question_id": qa.question_id,
        "prediction": "",
        "answer": qa.answer,
        "correct": False,
        "stop_reason": "error",
        "steps": "[]",
        "token_usage": "{}",
        "steps_count": 0,
    }

    try:
        search_client = LLMClient.from_env("SEARCH_LLM", thinking=True)
        loop = AgentLoop(search_client, max_steps=max_steps)
        try:
            system_prompt = prompt_mgr.build_inference_prompt(
                always_skills_text=always_skills_text,
                task_skill_map=task_skill_map,
                catalog_text=catalog_text,
            )
        except TypeError:
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
        prediction = getattr(loop_result, "answer", None)
        if prediction is None:
            result_payload = getattr(loop_result, "result", None)
            if isinstance(result_payload, dict):
                prediction = result_payload.get("answer")
        record["prediction"] = prediction or ""
        record["stop_reason"] = loop_result.stop_reason
        record["token_usage"] = json.dumps(loop_result.token_usage)
        record["steps_count"] = len(loop_result.steps)
        record["correct"] = record["prediction"] == qa.answer
        record["steps"] = json.dumps(
            [
                {
                    "thought": s.thought,
                    "tool_call": s.tool_call,
                    "tool_output": s.tool_output,
                }
                for s in loop_result.steps
            ]
        )
    except Exception:
        logger.exception("推理失败 question_id=%s", qa.question_id)
    finally:
        log.insert("predictions", record)

    return record
