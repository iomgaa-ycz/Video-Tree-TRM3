"""提示词管理模块。

提供 PromptManager 类，统一管理四步循环（出题/推理/诊断/进化）的
prompt 加载与组装。工具级 prompt（extract/verify）不在管理范围内。
"""

from __future__ import annotations

from pathlib import Path

from core.tree.tools import get_tool_descriptions


class PromptManager:
    """管理循环级 prompt 的加载与组装。

    构造时缓存 system.md 作为 inference 基础模板。
    后续步骤（diagnose/evolve/question_gen）通过 load() 按文件名读取。

    参数:
        prompts_dir: prompt 文件目录的绝对路径。
    """

    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        system_path = prompts_dir / "system.md"
        if not system_path.exists():
            raise FileNotFoundError(f"system.md 不存在: {system_path}")
        self._system_base = system_path.read_text(encoding="utf-8")

    def build_inference_prompt(
        self,
        skill_mode: str,
        task_type: str,
        always_skills_text: str,
        task_skill_map: dict[str, str],
        catalog_text: str,
    ) -> str:
        """组装 inference 步骤的完整 system prompt。

        参数:
            skill_mode: "auto" / "manual" / "none"。
            task_type: 当前 QA 的题型。
            always_skills_text: always 层 skill 正文（已拼接）。
            task_skill_map: {task_type: skill_body} 映射。
            catalog_text: manual 模式的 skill 目录文本。

        返回:
            拼装后的完整 system prompt。
        """
        include_read_skill = skill_mode == "manual"
        parts = [
            self._system_base,
            f"\n\n---\n\n{get_tool_descriptions(include_read_skill=include_read_skill)}",
        ]
        if always_skills_text:
            parts.append(f"\n\n---\n\n# 通用搜索策略\n\n{always_skills_text}")
        if skill_mode == "auto":
            skill_text = task_skill_map.get(task_type)
            if skill_text:
                parts.append(f"\n\n---\n\n# 当前题型搜索策略\n\n{skill_text}")
        elif skill_mode == "manual":
            if catalog_text:
                parts.append(
                    "\n\n---\n\n# 可用搜索策略\n\n"
                    "以下技能扩展了你的导航能力。当问题匹配某技能的适用题型时，"
                    "用 read_skill 工具加载该技能，然后按其指引操作。\n\n"
                    f"{catalog_text}"
                )
        return "".join(parts)

    def format_user_prompt(
        self,
        qa: dict,
        l1_node_ids: list[str],
        task_type: str | None = None,
    ) -> str:
        """格式化 inference 步骤的用户提示词。

        参数:
            qa: 包含 question / options 字段的 QA 字典。
            l1_node_ids: L1 根节点 ID 列表（如 ["L1_000", "L1_001"]）。
            task_type: 可选题型标签，非 None 时插入题型行（oracle 实验用）。

        返回:
            格式化后的用户提示词。
        """
        options_text = "\n".join(qa["options"])
        roots_text = ", ".join(l1_node_ids)
        task_type_line = f"**题型**: {task_type}\n" if task_type else ""
        return (
            f"请回答以下关于这个视频的多选题：\n\n"
            f"{task_type_line}"
            f"**问题**: {qa['question']}\n"
            f"**选项**:\n{options_text}\n\n"
            f"**视频树 L1 根节点**: {roots_text}\n"
            f"请从以上 L1 节点开始导航，收集证据后回答。"
        )

    def load(self, name: str) -> str:
        """按文件名加载 prompt 内容。

        参数:
            name: prompt 文件名（如 "diagnose_span.md"）。

        返回:
            文件内容字符串。

        异常:
            FileNotFoundError: 文件不存在。
        """
        path = self._prompts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"prompt 文件不存在: {path}")
        return path.read_text(encoding="utf-8")
