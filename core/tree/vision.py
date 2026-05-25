"""视觉模型调用模块 — 两轮 VL 调用查看关键帧图像。

提取轮：带防幻觉 system prompt，提取原始视觉证据。
验证轮：带核实 system prompt，逐条核实并给置信度。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from core.llm_client import LLMClient


def _load_prompt(prompts_dir: Path, filename: str) -> str:
    """从 prompts 目录加载 system prompt 文件。

    参数:
        prompts_dir: prompt 文件所在目录。
        filename: prompt 文件名。

    返回:
        文件内容字符串。
    """
    return (prompts_dir / filename).read_text(encoding="utf-8")


def _encode_frame(frame_path: Path) -> str:
    """将帧文件编码为 base64 data URI。

    参数:
        frame_path: JPEG 帧文件路径。

    返回:
        data:image/jpeg;base64,... 格式的字符串。

    异常:
        FileNotFoundError: 帧文件不存在。
    """
    raw = frame_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _call_vl(
    client: LLMClient, system_prompt: str, user_content: list[dict[str, Any]]
) -> str:
    """调用 VL 模型（thinking 已在 client 构造时关闭）。

    参数:
        client: VL LLMClient 实例（thinking=False）。
        system_prompt: 系统提示词。
        user_content: 用户消息的 content 列表（图片 + 文本块）。

    返回:
        模型回答文本。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    response = client.chat(messages)
    return response.choices[0].message.content


def observe_frame(
    vl_client: LLMClient,
    frame_paths: list[Path],
    question: str,
    prompts_dir: Path,
) -> str:
    """调用 VL 模型查看帧图像：提取轮 + 验证轮。

    参数:
        vl_client: VL 用 LLMClient（thinking=False）。
        frame_paths: 帧文件路径列表。
        question: 针对帧内容的视觉问题。
        prompts_dir: prompt 文件目录。

    返回:
        "[视觉观察] {证据}\\n[验证] {核实结果}" 或错误信息。
    """
    try:
        image_uris = [_encode_frame(p) for p in frame_paths]
    except FileNotFoundError as e:
        return f"[VL错误] 帧文件不存在: {e}"

    image_parts: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": uri}} for uri in image_uris
    ]

    extract_content = image_parts + [{"type": "text", "text": question}]
    try:
        raw_evidence = _call_vl(
            vl_client,
            _load_prompt(prompts_dir, "observe_frame_extract.md"),
            extract_content,
        )
    except Exception as e:
        return f"[VL错误] {e}"

    verify_text = (
        f"问题: {question}\n\n"
        f"以下是另一个模型基于这些图片生成的描述，请核实：\n{raw_evidence}"
    )
    verify_content = image_parts + [{"type": "text", "text": verify_text}]
    try:
        verify_result = _call_vl(
            vl_client,
            _load_prompt(prompts_dir, "observe_frame_verify.md"),
            verify_content,
        )
        return f"[视觉观察] {raw_evidence}\n[验证] {verify_result}"
    except Exception as e:
        logger.warning("验证轮调用失败，跳过: {}", e)
        return f"[视觉观察] {raw_evidence}\n[验证] 跳过（调用失败）"
