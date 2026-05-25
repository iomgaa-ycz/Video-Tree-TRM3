"""HarnessJudge：通过 LLM API 实现语义评估和诊断。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Verdict:
    """评估判定结果。"""

    passed: bool
    score: float
    reasoning: str
    suggestions: list[str]
    raw_response: str


@dataclass
class Diagnosis:
    """问题诊断结果。"""

    root_cause: str
    evidence: list[str]
    fix_suggestions: list[str]
    severity: str


class HarnessJudge:
    """通过 LLM API 对运行结果进行语义评估。

    参数:
        llm_client: LLMClient 实例。
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm_client = llm_client

    def _call_llm(self, prompt: str) -> str:
        """通过 LLMClient 调用 LLM。

        参数:
            prompt: 发送给模型的完整 prompt。

        返回:
            模型的文本回复。
        """
        response = self._llm_client.chat(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _build_evaluate_prompt(
        self,
        criteria: str,
        evidence: dict[str, Any],
        rubric: dict[str, Any] | None = None,
    ) -> str:
        """构建评估 prompt。

        参数:
            criteria: 评判标准描述。
            evidence: 证据数据字典。
            rubric: 可选的评分维度字典。

        返回:
            完整的 prompt 字符串。
        """
        parts = [
            "你是一个科研实验评估专家。请根据以下标准和证据进行评估。",
            f"\n## 评判标准\n{criteria}",
            f"\n## 证据数据\n```json\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n```",
        ]
        if rubric:
            parts.append(
                f"\n## 评分维度\n```json\n{json.dumps(rubric, ensure_ascii=False, indent=2)}\n```"
            )
        parts.append(
            "\n## 输出格式\n请严格按以下 JSON 格式输出：\n"
            '{"passed": true/false, "score": 0.0-1.0, "reasoning": "...", "suggestions": ["..."]}'
        )
        return "\n".join(parts)

    def _build_diagnose_prompt(self, error_context: dict[str, Any]) -> str:
        """构建诊断 prompt。

        参数:
            error_context: 错误上下文信息字典。

        返回:
            完整的诊断 prompt 字符串。
        """
        return (
            "你是一个科研实验诊断专家。请分析以下错误上下文，找出根因并给出修复建议。\n\n"
            f"## 错误上下文\n```json\n{json.dumps(error_context, ensure_ascii=False, indent=2)}\n```\n\n"
            "## 输出格式\n请严格按以下 JSON 格式输出：\n"
            '{"root_cause": "...", "evidence": ["..."], "fix_suggestions": ["..."], '
            '"severity": "critical/important/minor"}'
        )

    def _parse_verdict(self, raw: str) -> Verdict:
        """从 LLM 回复中解析 Verdict。

        参数:
            raw: LLM 原始回复文本。

        返回:
            解析后的 Verdict 对象。
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return Verdict(
                passed=False,
                score=0.0,
                reasoning="无法解析 LLM 回复",
                suggestions=[],
                raw_response=raw,
            )
        data = json.loads(text[start:end])
        return Verdict(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            suggestions=list(data.get("suggestions", [])),
            raw_response=raw,
        )

    def _parse_diagnosis(self, raw: str) -> Diagnosis:
        """从 LLM 回复中解析 Diagnosis。

        参数:
            raw: LLM 原始回复文本。

        返回:
            解析后的 Diagnosis 对象。
        """
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return Diagnosis(
                root_cause="无法解析 LLM 回复",
                evidence=[],
                fix_suggestions=[],
                severity="critical",
            )
        data = json.loads(text[start:end])
        return Diagnosis(
            root_cause=str(data.get("root_cause", "")),
            evidence=list(data.get("evidence", [])),
            fix_suggestions=list(data.get("fix_suggestions", [])),
            severity=str(data.get("severity", "important")),
        )

    def evaluate(
        self,
        criteria: str,
        evidence: dict[str, Any],
        rubric: dict[str, Any] | None = None,
    ) -> Verdict:
        """对实验结果进行语义评估。

        参数:
            criteria: 评判标准描述。
            evidence: 证据数据。
            rubric: 评分维度及权重。

        返回:
            Verdict 评估结果。
        """
        prompt = self._build_evaluate_prompt(criteria, evidence, rubric)
        raw = self._call_llm(prompt)
        return self._parse_verdict(raw)

    def compare_runs(
        self,
        baseline_id: str,
        current_id: str,
        metrics_table: str,
        log: Any,
    ) -> Verdict:
        """对比两次运行的 metrics。

        参数:
            baseline_id: 基线运行 ID。
            current_id: 当前运行 ID。
            metrics_table: metrics 表名。
            log: HarnessLog 实例。

        返回:
            Verdict 对比结果。
        """
        baseline_data = log.query(
            f"SELECT * FROM {metrics_table} WHERE run_id = ?", (baseline_id,)
        )
        current_data = log.query(
            f"SELECT * FROM {metrics_table} WHERE run_id = ?", (current_id,)
        )
        evidence = {"baseline": baseline_data, "current": current_data}
        criteria = (
            f"对比 run {current_id} 相对于基线 {baseline_id} 在 {metrics_table} 表中的各项指标，"
            "判断是否有提升或回退。"
        )
        return self.evaluate(criteria, evidence)

    def diagnose(self, error_context: dict[str, Any], log: Any) -> Diagnosis:
        """根据日志和错误信息诊断问题根因。

        参数:
            error_context: 错误上下文信息。
            log: HarnessLog 实例。

        返回:
            Diagnosis 诊断结果。
        """
        events = log.query(
            "SELECT * FROM _events WHERE run_id = ? ORDER BY timestamp DESC LIMIT 50",
            (error_context.get("run_id", ""),),
        )
        error_context["recent_events"] = events
        prompt = self._build_diagnose_prompt(error_context)
        raw = self._call_llm(prompt)
        return self._parse_diagnosis(raw)
