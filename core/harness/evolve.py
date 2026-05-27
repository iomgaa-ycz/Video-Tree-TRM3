"""进化数据结构与核心逻辑，对应 optimizer.step()。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml as _yaml


@dataclass
class ValidationResult:
    """格式验证的结果。"""

    passed: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class EvolutionRecord:
    """单个目标文件的一次进化记录。"""

    target_file: str
    """目标文件名，如 'temporal-reasoning.md'。"""

    target_type: str
    """目标类型: 'skill' / 'system' / 'tool'。"""

    original_content: str
    """改写前原文。"""

    evolved_content: str
    """改写后内容；rejected 时与 original_content 相同。"""

    reason: str
    """状态说明。"""

    status: str
    """'accepted' / 'rejected' / 'skipped'。"""

    source_version: str
    """改写前版本号，如 'v1'。"""

    result_version: str | None = None
    """改写后版本号；rejected/skipped 时为 None。"""

    suggestions: list[dict[str, Any]] = field(default_factory=list)
    """LLM 输出的改动建议列表。"""

    attempts: list[dict[str, Any]] = field(default_factory=list)
    """每次 LLM 调用的原始响应摘要。"""

    validation_errors: list[str] = field(default_factory=list)
    """验证失败的具体原因。"""


@dataclass
class EvolutionResult:
    """一次整体进化流程的汇总结果。"""

    skills_version: str | None
    """新 skills 版本号；无改动时为 None。"""

    prompts_version: str | None
    """新 prompts 版本号；无改动时为 None。"""

    records: list[EvolutionRecord] = field(default_factory=list)
    """所有目标的进化记录。"""

    accepted_count: int = 0
    """通过验证的改写数。"""

    rejected_count: int = 0
    """未通过验证的改写数。"""

    skipped_count: int = 0
    """因无失败案例而跳过的目标数。"""


# ---------------------------------------------------------------------------
# 格式验证
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """解析 YAML frontmatter，失败返回 None。

    参数:
        text: Markdown 文件全文。

    返回:
        frontmatter 字典，无有效 frontmatter 时返回 None。
    """
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    try:
        return _yaml.safe_load(match.group(1))
    except _yaml.YAMLError:
        return None


def _check_length(original: str, evolved: str, errors: list[str]) -> None:
    """检查改写后长度是否在 0.3x ~ 2.0x 范围内。"""
    orig_len = len(original)
    if orig_len == 0:
        return
    ratio = len(evolved) / orig_len
    if ratio > 2.0:
        errors.append(
            f"长度超限: 改写后 {len(evolved)} 字符是原文 {orig_len} 的 {ratio:.1f} 倍 (上限 2.0)"
        )
    if ratio < 0.3:
        errors.append(
            f"长度不足: 改写后 {len(evolved)} 字符是原文 {orig_len} 的 {ratio:.1f} 倍 (下限 0.3)"
        )


def _check_code_blocks(text: str, errors: list[str]) -> None:
    """检查代码块是否闭合。"""
    count = text.count("```")
    if count % 2 != 0:
        errors.append(f"Markdown 格式错误: 代码块未闭合 (``` 出现 {count} 次)")


def _extract_section(text: str, heading: str) -> str | None:
    """提取 ## heading 到下一个 ## 之间的文本。

    参数:
        text: Markdown 全文。
        heading: 二级标题名。

    返回:
        该 section 的完整文本（含标题行），未找到时返回 None。
    """
    pattern = rf"(## {re.escape(heading)}.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def validate_skill(original: str, evolved: str) -> ValidationResult:
    """校验 Skill 改写结果。

    检查项: frontmatter 保留、长度合理、代码块闭合。

    参数:
        original: 改写前的 Skill 文件全文。
        evolved: 改写后的 Skill 文件全文。

    返回:
        ValidationResult 实例。
    """
    errors: list[str] = []
    orig_fm = _parse_frontmatter(original)
    evol_fm = _parse_frontmatter(evolved)
    if orig_fm is None:
        errors.append("原文缺少有效 frontmatter")
    elif evol_fm is None:
        errors.append("改写后缺少有效 frontmatter")
    else:
        for key in ("name", "description", "task_type"):
            if orig_fm.get(key) != evol_fm.get(key):
                errors.append(
                    f"frontmatter 字段 {key} 被修改: "
                    f"{orig_fm.get(key)!r} → {evol_fm.get(key)!r}"
                )
    _check_length(original, evolved, errors)
    _check_code_blocks(evolved, errors)
    return ValidationResult(passed=len(errors) == 0, errors=errors)


def validate_system(original: str, evolved: str) -> ValidationResult:
    """校验 System Prompt 改写结果。

    检查项: 冻结区保留（能力边界、输出格式、视频树结构）、长度合理、代码块闭合。

    参数:
        original: 改写前的 system.md 全文。
        evolved: 改写后的 system.md 全文。

    返回:
        ValidationResult 实例。
    """
    errors: list[str] = []
    frozen_sections = ["能力边界", "输出格式", "视频树结构"]
    for section_name in frozen_sections:
        orig_section = _extract_section(original, section_name)
        if orig_section is None:
            continue
        evol_section = _extract_section(evolved, section_name)
        if evol_section is None:
            errors.append(f"冻结区 '## {section_name}' 在改写后缺失")
        elif orig_section != evol_section:
            errors.append(f"冻结区 '## {section_name}' 在改写后被修改")
    _check_length(original, evolved, errors)
    _check_code_blocks(evolved, errors)
    return ValidationResult(passed=len(errors) == 0, errors=errors)


def validate_tool(
    original_extract: str,
    evolved_extract: str,
    original_verify: str,
    evolved_verify: str,
) -> ValidationResult:
    """校验 Tool Prompt 改写结果。

    检查项: 输出格式 section 保留、长度合理。

    参数:
        original_extract: 改写前的 extract prompt。
        evolved_extract: 改写后的 extract prompt。
        original_verify: 改写前的 verify prompt。
        evolved_verify: 改写后的 verify prompt。

    返回:
        ValidationResult 实例。
    """
    errors: list[str] = []
    for label, orig, evol in [
        ("extract", original_extract, evolved_extract),
        ("verify", original_verify, evolved_verify),
    ]:
        orig_fmt = _extract_section(orig, "输出格式")
        if orig_fmt is not None:
            evol_fmt = _extract_section(evol, "输出格式")
            if evol_fmt is None:
                errors.append(f"{label}: 冻结区 '## 输出格式' 在改写后缺失")
            elif orig_fmt != evol_fmt:
                errors.append(f"{label}: 冻结区 '## 输出格式' 在改写后被修改")
        _check_length(orig, evol, errors)
    return ValidationResult(passed=len(errors) == 0, errors=errors)
