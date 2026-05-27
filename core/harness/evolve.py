"""进化数据结构与核心逻辑，对应 optimizer.step()。"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml as _yaml
from loguru import logger

from core.harness.diagnose import (
    CaseSample,
    DiagnosisResult,
    SkillCasePack,
    SystemCasePack,
    ToolCasePack,
)
from core.harness.log import HarnessLog
from core.llm_client import LLMClient
from core.workspace import advance_version


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
    evol_len = len(evolved)
    if ratio > 2.0:
        errors.append(
            f"长度超限: {evol_len} 字符是原文 {orig_len} 的"
            f" {ratio:.1f} 倍 (上限 2.0)"
        )
    if ratio < 0.3:
        errors.append(
            f"长度不足: {evol_len} 字符是原文 {orig_len} 的"
            f" {ratio:.1f} 倍 (下限 0.3)"
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


# ---------------------------------------------------------------------------
# LLM 交互辅助函数
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_prompt_template(template_name: str) -> str:
    """从项目根 prompts/ 目录加载 prompt 模板。

    参数:
        template_name: 模板文件名，如 "evolve_skill.md"。

    返回:
        模板文件全文。
    """
    path = _PROJECT_ROOT / "prompts" / template_name
    return path.read_text(encoding="utf-8")


def _format_case_samples(cases: list[CaseSample | dict]) -> str:
    """将 CaseSample 列表格式化为 LLM 可读文本。

    参数:
        cases: CaseSample 实例列表（也兼容 dict）。

    返回:
        格式化后的多行文本。
    """
    lines: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            case = asdict(case)
        lines.append(f"### {case.get('question_id', 'unknown')}")
        lines.append(f"- question: {case.get('question', '')}")
        options = case.get("options", [])
        if options:
            lines.append(f"- options: {json.dumps(options, ensure_ascii=False)}")
        lines.append(f"- answer: {case.get('answer', '')}")
        lines.append(f"- prediction: {case.get('prediction', '')}")
        lines.append(f"- error_type: {case.get('error_type', '')}")
        lines.append(f"- selection_reason: {case.get('selection_reason', '')}")
        trace = case.get("trace", [])
        if trace:
            lines.append("- trace:")
            for step in trace:
                output_text = str(step.get("tool_output", ""))
                if len(output_text) > 500:
                    output_text = output_text[:500] + "..."
                lines.append(
                    f"  - step {step.get('step', '?')}: "
                    f"tool={step.get('tool_name', '')} "
                    f"args={json.dumps(step.get('tool_args', {}), ensure_ascii=False)} "
                    f"output={output_text}"
                )
        lines.append("")
    return "\n".join(lines)


def _format_spans(spans: list[dict[str, Any]]) -> str:
    """将工具 span 字典列表格式化为 LLM 可读文本。

    参数:
        spans: span 字典列表，每个包含 step, tool_name, tool_args 等字段。

    返回:
        格式化后的多行文本。
    """
    lines: list[str] = []
    for span in spans:
        lines.append(f"### step {span.get('step', '?')}")
        lines.append(f"- tool_name: {span.get('tool_name', '')}")
        lines.append(
            f"- tool_args: {json.dumps(span.get('tool_args', {}), ensure_ascii=False)}"
        )
        output_text = str(span.get("tool_output", ""))
        if len(output_text) > 500:
            output_text = output_text[:500] + "..."
        lines.append(f"- tool_output: {output_text}")
        lines.append(
            f"- extraction_completeness: {span.get('extraction_completeness', '')}"
        )
        lines.append(f"- hallucination_rate: {span.get('hallucination_rate', '')}")
        missed = span.get("missed_info_tags", [])
        if missed:
            lines.append(
                f"- missed_info_tags: {json.dumps(missed, ensure_ascii=False)}"
            )
        hall_tags = span.get("hallucination_tags", [])
        if hall_tags:
            lines.append(
                f"- hallucination_tags: {json.dumps(hall_tags, ensure_ascii=False)}"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_llm_json(raw: str) -> dict | None:
    """从 LLM 响应中解析 JSON。

    处理 ```json 代码块包裹的情况。

    参数:
        raw: LLM 原始输出文本。

    返回:
        解析后的字典，失败返回 None。
    """
    text = raw.strip()
    # 提取 ```json ... ``` 代码块
    code_block = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _now_iso() -> str:
    """返回当前 UTC 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 单目标进化函数
# ---------------------------------------------------------------------------


def _evolve_single_skill(
    client: LLMClient,
    pack: SkillCasePack,
    skills_dir: Path,
    source_version: str,
) -> EvolutionRecord:
    """进化单个 Skill 文件。

    参数:
        client: LLM 客户端。
        pack: 该题型的案例包。
        skills_dir: 当前版本的 skills 目录。
        source_version: 当前版本号。

    返回:
        EvolutionRecord 实例。
    """
    target_file = pack.target_file
    original = (skills_dir / target_file).read_text(encoding="utf-8")
    system_prompt = _load_prompt_template("evolve_skill.md")

    stats_json = json.dumps(pack.stats, ensure_ascii=False, indent=2)
    user_msg = (
        f"## 当前 Skill 文件\n\n{original}\n\n"
        f"## 聚合统计\n\n```json\n{stats_json}\n```\n\n"
        f"## 失败案例\n\n{_format_case_samples(pack.failure_cases)}\n\n"
        f"## 成功案例\n\n{_format_case_samples(pack.success_cases)}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    attempts: list[dict[str, Any]] = []

    # 最多两次尝试（首次 + 一次重试）
    for attempt_idx in range(2):
        response = client.chat(messages)
        raw_content = response.choices[0].message.content
        attempts.append({"attempt": attempt_idx + 1, "raw_length": len(raw_content)})
        parsed = _parse_llm_json(raw_content)

        if parsed is None:
            logger.warning("Skill 进化 LLM 响应 JSON 解析失败: {}", target_file)
            if attempt_idx == 0:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "你的输出不是合法 JSON，请重新输出。",
                    }
                )
                continue
            return EvolutionRecord(
                target_file=target_file,
                target_type="skill",
                original_content=original,
                evolved_content=original,
                reason="LLM 响应 JSON 解析失败",
                status="rejected",
                source_version=source_version,
                attempts=attempts,
                validation_errors=["JSON 解析失败"],
            )

        evolved_content = parsed.get("evolved_content", "")
        suggestions = parsed.get("suggestions", [])
        validation = validate_skill(original, evolved_content)

        if validation.passed:
            return EvolutionRecord(
                target_file=target_file,
                target_type="skill",
                original_content=original,
                evolved_content=evolved_content,
                reason="验证通过",
                status="accepted",
                source_version=source_version,
                suggestions=suggestions,
                attempts=attempts,
            )

        # 验证失败，首次时重试
        if attempt_idx == 0:
            error_feedback = "\n".join(validation.errors)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": f"验证失败，请修正后重新输出：\n{error_feedback}",
                }
            )
            continue

        return EvolutionRecord(
            target_file=target_file,
            target_type="skill",
            original_content=original,
            evolved_content=original,
            reason="验证失败（重试后仍未通过）",
            status="rejected",
            source_version=source_version,
            suggestions=suggestions,
            attempts=attempts,
            validation_errors=validation.errors,
        )

    # 不应到达此处，但作为防御性兜底
    return EvolutionRecord(
        target_file=target_file,
        target_type="skill",
        original_content=original,
        evolved_content=original,
        reason="未知错误",
        status="rejected",
        source_version=source_version,
        attempts=attempts,
    )


def _evolve_system_prompt(
    client: LLMClient,
    pack: SystemCasePack,
    prompts_dir: Path,
    source_version: str,
) -> EvolutionRecord:
    """进化 System Prompt。

    参数:
        client: LLM 客户端。
        pack: 跨题型行为模式案例包。
        prompts_dir: 当前版本的 prompts 目录。
        source_version: 当前版本号。

    返回:
        EvolutionRecord 实例。
    """
    target_file = "system.md"
    original = (prompts_dir / target_file).read_text(encoding="utf-8")
    system_prompt = _load_prompt_template("evolve_system.md")

    stats_json = json.dumps(pack.stats, ensure_ascii=False, indent=2)
    user_msg = (
        f"## 当前 System Prompt\n\n{original}\n\n"
        f"## D5 行为模式统计\n\n```json\n{stats_json}\n```\n\n"
        f"## 失败案例\n\n{_format_case_samples(pack.failure_cases)}\n\n"
        f"## 成功案例\n\n{_format_case_samples(pack.success_cases)}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    attempts: list[dict[str, Any]] = []

    for attempt_idx in range(2):
        response = client.chat(messages)
        raw_content = response.choices[0].message.content
        attempts.append({"attempt": attempt_idx + 1, "raw_length": len(raw_content)})
        parsed = _parse_llm_json(raw_content)

        if parsed is None:
            logger.warning("System 进化 LLM 响应 JSON 解析失败")
            if attempt_idx == 0:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "你的输出不是合法 JSON，请重新输出。",
                    }
                )
                continue
            return EvolutionRecord(
                target_file=target_file,
                target_type="system",
                original_content=original,
                evolved_content=original,
                reason="LLM 响应 JSON 解析失败",
                status="rejected",
                source_version=source_version,
                attempts=attempts,
                validation_errors=["JSON 解析失败"],
            )

        evolved_content = parsed.get("evolved_content", "")
        suggestions = parsed.get("suggestions", [])
        validation = validate_system(original, evolved_content)

        if validation.passed:
            return EvolutionRecord(
                target_file=target_file,
                target_type="system",
                original_content=original,
                evolved_content=evolved_content,
                reason="验证通过",
                status="accepted",
                source_version=source_version,
                suggestions=suggestions,
                attempts=attempts,
            )

        if attempt_idx == 0:
            error_feedback = "\n".join(validation.errors)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": f"验证失败，请修正后重新输出：\n{error_feedback}",
                }
            )
            continue

        return EvolutionRecord(
            target_file=target_file,
            target_type="system",
            original_content=original,
            evolved_content=original,
            reason="验证失败（重试后仍未通过）",
            status="rejected",
            source_version=source_version,
            suggestions=suggestions,
            attempts=attempts,
            validation_errors=validation.errors,
        )

    return EvolutionRecord(
        target_file=target_file,
        target_type="system",
        original_content=original,
        evolved_content=original,
        reason="未知错误",
        status="rejected",
        source_version=source_version,
        attempts=attempts,
    )


def _evolve_single_tool(
    client: LLMClient,
    pack: ToolCasePack,
    prompts_dir: Path,
    source_version: str,
) -> EvolutionRecord:
    """进化单个工具的 extract + verify prompt。

    参数:
        client: LLM 客户端。
        pack: 该工具的案例包。
        prompts_dir: 当前版本的 prompts 目录。
        source_version: 当前版本号。

    返回:
        EvolutionRecord 实例。
    """
    tool_name = pack.tool_name
    target_file = f"{tool_name}_extract.md"
    extract_path = prompts_dir / f"{tool_name}_extract.md"
    verify_path = prompts_dir / f"{tool_name}_verify.md"
    orig_extract = extract_path.read_text(encoding="utf-8")
    orig_verify = verify_path.read_text(encoding="utf-8")
    original_combined = json.dumps(
        {"extract": orig_extract, "verify": orig_verify}, ensure_ascii=False
    )

    system_prompt = _load_prompt_template("evolve_tool.md")

    stats_json = json.dumps(pack.stats, ensure_ascii=False, indent=2)
    user_msg = (
        f"## 当前 extract prompt\n\n{orig_extract}\n\n"
        f"## 当前 verify prompt\n\n{orig_verify}\n\n"
        f"## 工具质量统计\n\n```json\n{stats_json}\n```\n\n"
        f"## 失败 span 案例\n\n{_format_spans(pack.failure_spans)}\n\n"
        f"## 成功 span 案例\n\n{_format_spans(pack.success_spans)}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    attempts: list[dict[str, Any]] = []

    for attempt_idx in range(2):
        response = client.chat(messages)
        raw_content = response.choices[0].message.content
        attempts.append({"attempt": attempt_idx + 1, "raw_length": len(raw_content)})
        parsed = _parse_llm_json(raw_content)

        if parsed is None:
            logger.warning("Tool 进化 LLM 响应 JSON 解析失败: {}", tool_name)
            if attempt_idx == 0:
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": "你的输出不是合法 JSON，请重新输出。",
                    }
                )
                continue
            return EvolutionRecord(
                target_file=target_file,
                target_type="tool",
                original_content=original_combined,
                evolved_content=original_combined,
                reason="LLM 响应 JSON 解析失败",
                status="rejected",
                source_version=source_version,
                attempts=attempts,
                validation_errors=["JSON 解析失败"],
            )

        evolved_extract = parsed.get("evolved_extract", "")
        evolved_verify = parsed.get("evolved_verify", "")
        suggestions = parsed.get("suggestions", [])
        validation = validate_tool(
            orig_extract, evolved_extract, orig_verify, evolved_verify
        )

        evolved_combined = json.dumps(
            {"extract": evolved_extract, "verify": evolved_verify},
            ensure_ascii=False,
        )

        if validation.passed:
            return EvolutionRecord(
                target_file=target_file,
                target_type="tool",
                original_content=original_combined,
                evolved_content=evolved_combined,
                reason="验证通过",
                status="accepted",
                source_version=source_version,
                suggestions=suggestions,
                attempts=attempts,
            )

        if attempt_idx == 0:
            error_feedback = "\n".join(validation.errors)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": f"验证失败，请修正后重新输出：\n{error_feedback}",
                }
            )
            continue

        return EvolutionRecord(
            target_file=target_file,
            target_type="tool",
            original_content=original_combined,
            evolved_content=original_combined,
            reason="验证失败（重试后仍未通过）",
            status="rejected",
            source_version=source_version,
            suggestions=suggestions,
            attempts=attempts,
            validation_errors=validation.errors,
        )

    return EvolutionRecord(
        target_file=target_file,
        target_type="tool",
        original_content=original_combined,
        evolved_content=original_combined,
        reason="未知错误",
        status="rejected",
        source_version=source_version,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# DB 持久化
# ---------------------------------------------------------------------------


def write_evolution_records(log: HarnessLog, records: list[EvolutionRecord]) -> None:
    """将进化记录写入数据库。

    参数:
        log: HarnessLog 实例。
        records: 进化记录列表。
    """
    log.execute("""
        CREATE TABLE IF NOT EXISTS evolution_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT,
            target_file TEXT,
            target_type TEXT,
            status TEXT,
            suggestions JSON,
            validation_errors JSON,
            source_version TEXT,
            result_version TEXT
        )
    """)
    for record in records:
        if record.status == "skipped":
            continue
        log.insert(
            "evolution_records",
            {
                "target_file": record.target_file,
                "target_type": record.target_type,
                "status": record.status,
                "suggestions": json.dumps(record.suggestions, ensure_ascii=False),
                "validation_errors": json.dumps(
                    record.validation_errors, ensure_ascii=False
                ),
                "source_version": record.source_version,
                "result_version": record.result_version,
            },
        )


# ---------------------------------------------------------------------------
# 编排函数
# ---------------------------------------------------------------------------


def run_evolution(
    diagnosis: DiagnosisResult,
    workspace_dir: Path,
    store_dir: Path,
    skills_dir: Path,
    prompts_dir: Path,
    db_path: Path,
    targets: set[str] | None = None,
    concurrency: int = 4,
) -> EvolutionResult:
    """执行一轮进化流程：收集任务、并发调用 LLM、验证、版本写入。

    参数:
        diagnosis: 诊断结果。
        workspace_dir: Workspace 根目录。
        store_dir: Store 根目录。
        skills_dir: 当前版本 skills 目录。
        prompts_dir: 当前版本 prompts 目录。
        db_path: harness.db 路径。
        targets: 进化目标集合，可选 {"skills", "system", "tools"}。
        concurrency: 并发数。

    返回:
        EvolutionResult 实例。
    """
    if targets is None:
        targets = {"skills", "system", "tools"}

    # 从目录路径提取当前版本号
    source_skills_version = skills_dir.name
    source_prompts_version = prompts_dir.name

    # Phase 1: 收集任务
    tasks: list[tuple[str, Any]] = []
    skipped_records: list[EvolutionRecord] = []

    if "skills" in targets:
        for task_type, pack in diagnosis.skill_case_packs.items():
            if pack.failure_cases:
                tasks.append(("skill", pack))
            else:
                skipped_records.append(
                    EvolutionRecord(
                        target_file=pack.target_file,
                        target_type="skill",
                        original_content="",
                        evolved_content="",
                        reason="无失败案例",
                        status="skipped",
                        source_version=source_skills_version,
                    )
                )

    if "system" in targets and diagnosis.system_case_pack is not None:
        if diagnosis.system_case_pack.failure_cases:
            tasks.append(("system", diagnosis.system_case_pack))
        else:
            skipped_records.append(
                EvolutionRecord(
                    target_file="system.md",
                    target_type="system",
                    original_content="",
                    evolved_content="",
                    reason="无失败案例",
                    status="skipped",
                    source_version=source_prompts_version,
                )
            )

    if "tools" in targets:
        for tool_name, pack in diagnosis.tool_case_packs.items():
            if pack.failure_spans:
                tasks.append(("tool", pack))
            else:
                skipped_records.append(
                    EvolutionRecord(
                        target_file=f"{pack.tool_name}_extract.md",
                        target_type="tool",
                        original_content="",
                        evolved_content="",
                        reason="无失败案例",
                        status="skipped",
                        source_version=source_prompts_version,
                    )
                )

    if not tasks:
        return EvolutionResult(
            skills_version=None,
            prompts_version=None,
            records=skipped_records,
            skipped_count=len(skipped_records),
        )

    # Phase 2: 并发执行进化
    client = LLMClient.from_env("EVOLVE_LLM", thinking=True)

    def _dispatch(task_type: str, pack: Any) -> EvolutionRecord:
        if task_type == "skill":
            return _evolve_single_skill(client, pack, skills_dir, source_skills_version)
        if task_type == "system":
            return _evolve_system_prompt(
                client, pack, prompts_dir, source_prompts_version
            )
        return _evolve_single_tool(client, pack, prompts_dir, source_prompts_version)

    evolution_records: list[EvolutionRecord] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_dispatch, task_type, pack): (task_type, pack)
            for task_type, pack in tasks
        }
        for future in as_completed(futures):
            task_type, pack = futures[future]
            try:
                record = future.result()
                evolution_records.append(record)
            except Exception:
                label = getattr(pack, "target_file", getattr(pack, "tool_name", "?"))
                logger.exception("进化失败: type={}, target={}", task_type, label)
                evolution_records.append(
                    EvolutionRecord(
                        target_file=str(label),
                        target_type=task_type,
                        original_content="",
                        evolved_content="",
                        reason="进化过程异常",
                        status="rejected",
                        source_version=(
                            source_skills_version
                            if task_type == "skill"
                            else source_prompts_version
                        ),
                    )
                )

    all_records = skipped_records + evolution_records
    accepted = [r for r in evolution_records if r.status == "accepted"]
    rejected_count = sum(1 for r in evolution_records if r.status == "rejected")
    skipped_count = len(skipped_records)

    # Phase 3: 版本写入 — 将 accepted 的改动写入新版本
    new_skills_version: str | None = None
    new_prompts_version: str | None = None

    accepted_skills = [r for r in accepted if r.target_type == "skill"]
    accepted_prompts = [r for r in accepted if r.target_type in ("system", "tool")]

    if accepted_skills:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copytree(skills_dir, tmp_dir / "skills", dirs_exist_ok=True)
            for record in accepted_skills:
                (tmp_dir / "skills" / record.target_file).write_text(
                    record.evolved_content, encoding="utf-8"
                )
            new_skills_version = advance_version(
                store_dir,
                "skills",
                tmp_dir / "skills",
                {
                    "source": "evolution",
                    "parent": source_skills_version,
                    "trigger_run": diagnosis.run_id,
                    "description": f"进化 {len(accepted_skills)} 个 skill",
                },
            )
            for record in accepted_skills:
                record.result_version = new_skills_version

    if accepted_prompts:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            shutil.copytree(prompts_dir, tmp_dir / "prompts", dirs_exist_ok=True)
            for record in accepted_prompts:
                if record.target_type == "system":
                    (tmp_dir / "prompts" / record.target_file).write_text(
                        record.evolved_content, encoding="utf-8"
                    )
                elif record.target_type == "tool":
                    content = json.loads(record.evolved_content)
                    tool_name = record.target_file.replace("_extract.md", "")
                    (tmp_dir / "prompts" / f"{tool_name}_extract.md").write_text(
                        content["extract"], encoding="utf-8"
                    )
                    (tmp_dir / "prompts" / f"{tool_name}_verify.md").write_text(
                        content["verify"], encoding="utf-8"
                    )
            new_prompts_version = advance_version(
                store_dir,
                "prompts",
                tmp_dir / "prompts",
                {
                    "source": "evolution",
                    "parent": source_prompts_version,
                    "trigger_run": diagnosis.run_id,
                    "description": f"进化 {len(accepted_prompts)} 个 prompt",
                },
            )
            for record in accepted_prompts:
                record.result_version = new_prompts_version

    # Phase 4: 写 JSON 快照
    analyses_dir = workspace_dir / "analyses"
    analyses_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "run_id": diagnosis.run_id,
        "timestamp": _now_iso(),
        "skills_version": new_skills_version,
        "prompts_version": new_prompts_version,
        "accepted_count": len(accepted),
        "rejected_count": rejected_count,
        "skipped_count": skipped_count,
        "records": [asdict(r) for r in all_records],
    }
    snapshot_path = analyses_dir / f"evolution_{diagnosis.run_id}.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("进化快照已写入: {}", snapshot_path)

    # Phase 5: DB 持久化
    with HarnessLog(str(db_path), diagnosis.run_id) as log:
        write_evolution_records(log, all_records)

    return EvolutionResult(
        skills_version=new_skills_version,
        prompts_version=new_prompts_version,
        records=all_records,
        accepted_count=len(accepted),
        rejected_count=rejected_count,
        skipped_count=skipped_count,
    )
