"""节点内容摘要模块 — 两轮 LLM 调用生成 question-conditioned 摘要。

提取轮：带防幻觉 system prompt，提取与问题相关的信息。
验证轮：带核实 system prompt，逐条核实并给置信度。
与 vision.py 的 observe_frame 同构设计。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _call_llm(client: LLMClient, system_prompt: str, user_text: str) -> str:
    """调用 LLM（thinking 已在 client 构造时关闭）。

    参数:
        client: LLMClient 实例（thinking=False）。
        system_prompt: 系统提示词。
        user_text: 用户消息文本。

    返回:
        模型回答文本。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    response = client.chat(messages)
    return response.choices[0].message.content


def summarize_node(
    client: LLMClient, raw_text: str, question: str, prompts_dir: Path
) -> str:
    """对单个节点做 question-conditioned 两轮摘要。

    参数:
        client: 工具用 LLMClient（thinking=False）。
        raw_text: 节点原始文本（card + subtitle 拼接）。
        question: Agent 当前关注的具体问题。
        prompts_dir: prompt 文件目录。

    返回:
        "[内容摘要] {提取结果}\\n[核实] {验证结果}" 或错误信息。
    """
    extract_input = f"问题: {question}\n\n以下是视频片段的描述和字幕:\n{raw_text}"
    try:
        raw_summary = _call_llm(
            client, _load_prompt(prompts_dir, "view_node_extract.md"), extract_input
        )
    except Exception as e:
        return f"[摘要错误] {e}"

    verify_input = (
        f"问题: {question}\n\n"
        f"原始内容:\n{raw_text}\n\n"
        f"以下是另一个模型基于上述内容生成的摘要，请核实:\n{raw_summary}"
    )
    try:
        verify_result = _call_llm(
            client, _load_prompt(prompts_dir, "view_node_verify.md"), verify_input
        )
        return f"[内容摘要] {raw_summary}\n[核实] {verify_result}"
    except Exception as e:
        logger.warning("验证轮调用失败，跳过: {}", e)
        return f"[内容摘要] {raw_summary}\n[核实] 跳过（调用失败）"


def summarize_children(
    client: LLMClient,
    children_info: list[dict[str, Any]],
    question: str,
    prompts_dir: Path,
) -> str:
    """对子节点列表做 question-conditioned 相关性标注（两轮）。

    参数:
        client: 工具用 LLMClient（thinking=False）。
        children_info: 子节点信息列表，每项含 id, time_range, summary。
        question: Agent 当前关注的具体问题。
        prompts_dir: prompt 文件目录。

    返回:
        带相关性标注的子节点概览文本。
    """
    lines = []
    for child in children_info:
        t_start, t_end = child["time_range"]
        lines.append(
            f"- {child['id']} ({t_start:.0f}-{t_end:.0f}s): {child['summary']}"
        )
    children_text = "\n".join(lines)

    extract_input = f"问题: {question}\n\n{children_text}"
    try:
        raw_ranking = _call_llm(
            client,
            _load_prompt(prompts_dir, "view_node_children_extract.md"),
            extract_input,
        )
    except Exception as e:
        logger.warning("子节点标注失败，回退原始列表: {}", e)
        return children_text

    verify_input = (
        f"问题: {question}\n\n"
        f"原始子节点列表:\n{children_text}\n\n"
        f"以下是另一个模型基于上述信息生成的相关性标注，请核实:\n{raw_ranking}"
    )
    try:
        verify_result = _call_llm(
            client,
            _load_prompt(prompts_dir, "view_node_children_verify.md"),
            verify_input,
        )
        return f"{raw_ranking}\n[核实] {verify_result}"
    except Exception as e:
        logger.warning("子节点标注验证轮失败，跳过: {}", e)
        return raw_ranking


def _summarize_search_result(
    client: LLMClient, raw_text: str, question: str, prompts_dir: Path
) -> str:
    """对搜索结果做两轮摘要（search_similar 专用）。

    参数:
        client: 工具用 LLMClient（thinking=False）。
        raw_text: 节点原始文本。
        question: Agent 当前关注的具体问题。
        prompts_dir: prompt 文件目录。

    返回:
        "[内容摘要] {提取结果}\\n[核实] {验证结果}" 或错误信息。
    """
    extract_input = (
        f"问题: {question}\n\n以下是语义搜索命中的视频节点描述和字幕:\n{raw_text}"
    )
    try:
        raw_summary = _call_llm(
            client,
            _load_prompt(prompts_dir, "search_similar_extract.md"),
            extract_input,
        )
    except Exception as e:
        return f"[摘要错误] {e}"

    verify_input = (
        f"问题: {question}\n\n"
        f"原始内容:\n{raw_text}\n\n"
        f"以下是另一个模型基于上述内容生成的摘要，请核实:\n{raw_summary}"
    )
    try:
        verify_result = _call_llm(
            client,
            _load_prompt(prompts_dir, "search_similar_verify.md"),
            verify_input,
        )
        return f"[内容摘要] {raw_summary}\n[核实] {verify_result}"
    except Exception as e:
        logger.warning("搜索结果验证轮失败，跳过: {}", e)
        return f"[内容摘要] {raw_summary}\n[核实] 跳过（调用失败）"


def summarize_nodes_batch(
    client: LLMClient,
    items: list[tuple[str, str, str]],
    question: str,
    prompts_dir: Path,
    max_workers: int = 5,
) -> list[tuple[str, str]]:
    """并发对多个搜索结果做两轮摘要。

    参数:
        client: 工具用 LLMClient（thinking=False）。
        items: [(node_id, raw_text, extra_info), ...] 列表。
        question: Agent 当前关注的具体问题。
        prompts_dir: prompt 文件目录。
        max_workers: 并发线程数。

    返回:
        [(node_id, summary_text), ...] 列表，顺序与输入一致。
    """
    results: dict[int, tuple[str, str]] = {}

    def _worker(idx: int, node_id: str, raw_text: str) -> tuple[int, str, str]:
        summary = _summarize_search_result(client, raw_text, question, prompts_dir)
        return idx, node_id, summary

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_worker, i, nid, text): i
            for i, (nid, text, _) in enumerate(items)
        }
        for future in as_completed(futures):
            idx, node_id, summary = future.result()
            results[idx] = (node_id, summary)

    return [results[i] for i in range(len(items))]
