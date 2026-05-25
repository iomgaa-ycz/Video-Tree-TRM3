"""搜索 Agent 工具层 — JSON 解析、dispatch 和工具描述。

替代 TRM3 的 cli.py。放弃 Function Calling，模型输出纯 JSON，
本模块负责解析 action 字段并分发到对应工具函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from core.search.skills import SkillRegistry
    from core.tree.environment import TreeEnvironment


_BASE_DESCRIPTIONS = """\
## 可用工具

在 action 中指定 tool 和 args 来调用工具。

### view_node
查看节点信息，获取与问题相关的内容摘要和子节点概览。
- args: {"node_id": "节点 ID", "question": "当前关注的具体问题"}

### search_similar
语义检索最相关的节点，返回与问题相关的内容摘要。
- args: {"query": "搜索关键词（2-4 词）", "question": "当前关注的具体问题", "k": 返回数量（可选，默认 5）}

### observe_frame
调用视觉模型查看关键帧图像，回答针对性的视觉问题。
- args: {"node_ids": ["L3 节点 ID 列表（1-4 个），或单个 L2 节点 ID"], "question": "针对帧内容的具体视觉问题"}

### submit_answer
提交最终答案。
- args: {"answer": "选项字母 A/B/C/D", "evidence": "关键证据摘要", "reasoning": "每个选项的判断理由"}"""

_SKILL_DESCRIPTION = """

### read_skill
加载指定题型技能的详细搜索策略。
- args: {"name": "技能名称"}"""


def get_tool_descriptions(include_read_skill: bool = False) -> str:
    """返回工具描述文本，用于写入 system prompt。

    参数:
        include_read_skill: 是否包含 read_skill 工具（manual 模式用）。

    返回:
        Markdown 格式的工具描述文本。
    """
    text = _BASE_DESCRIPTIONS
    if include_read_skill:
        text += _SKILL_DESCRIPTION
    return text


def parse_action(content: str) -> tuple[str, dict[str, Any]]:
    """解析模型输出的 JSON content，提取工具调用。

    参数:
        content: 模型的 content 文本（应为合法 JSON）。

    返回:
        (tool_name, args_dict)。

    异常:
        ValueError: JSON 解析失败、缺少 action 字段、或 action 缺少 tool/args。
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}") from e

    if "action" not in data:
        raise ValueError(f"缺少 action 字段，收到的 keys: {list(data.keys())}")

    action = data["action"]
    if "tool" not in action or "args" not in action:
        raise ValueError(f"action 缺少 tool 或 args: {action}")

    return action["tool"], action["args"]


def dispatch(
    name: str,
    args: dict[str, Any],
    env: TreeEnvironment,
    vl_client: LLMClient | None = None,
    prompts_dir: Path | None = None,
    skills: SkillRegistry | None = None,
) -> str:
    """按工具名称分发到对应方法。

    参数:
        name: 工具名称。
        args: 工具参数字典。
        env: TreeEnvironment 实例。
        vl_client: VL 用 LLMClient（observe_frame 需要）。
        prompts_dir: prompt 文件目录（observe_frame 需要）。
        skills: SkillRegistry 实例（read_skill 需要）。

    返回:
        工具执行结果文本。
    """
    try:
        if name == "view_node":
            return env.view_node(args["node_id"], args["question"])

        if name == "search_similar":
            return env.search_similar(args["query"], args["question"], args.get("k", 5))

        if name == "submit_answer":
            return f"[ok] 答案已提交: {args['answer']}"

        if name == "observe_frame":
            question = args.get("question", "")
            if not question.strip():
                return "工具执行错误: question 不能为空"

            from core.tree.vision import observe_frame as observe_frame_fn

            frame_paths = env.resolve_frame_paths(args["node_ids"])
            subtitle = env.get_subtitle(args["node_ids"][0])

            assert vl_client is not None, "observe_frame 需要 vl_client"
            assert prompts_dir is not None, "observe_frame 需要 prompts_dir"

            vl_result = observe_frame_fn(vl_client, frame_paths, question, prompts_dir)
            if subtitle:
                return f"[字幕上下文] {subtitle}\n{vl_result}"
            return vl_result

        if name == "read_skill":
            if skills is None:
                return "错误: skills 未启用"
            return skills.read(args["name"])

        return f"未知工具: {name}"

    except (KeyError, ValueError, FileNotFoundError, AssertionError) as e:
        return f"工具执行错误: {e}"
