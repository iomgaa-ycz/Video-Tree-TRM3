"""HarnessJudge 单元测试。"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.harness.judge import HarnessJudge


class TestHarnessJudgeCallLLM:
    """_call_llm 委托给 LLMClient.chat()。"""

    def test_call_llm_delegates_to_client(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"passed": true, "score": 0.9, "reasoning": "ok", "suggestions": []}'
                    )
                )
            ]
        )

        judge = HarnessJudge(llm_client=mock_client)
        result = judge._call_llm("test prompt")

        mock_client.chat.assert_called_once()
        messages = mock_client.chat.call_args.kwargs["messages"]
        assert messages == [{"role": "user", "content": "test prompt"}]
        assert (
            result
            == '{"passed": true, "score": 0.9, "reasoning": "ok", "suggestions": []}'
        )


class TestHarnessJudgeEvaluate:
    """evaluate() 端到端。"""

    def test_evaluate_returns_verdict(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"passed": true, "score": 0.85, "reasoning": "结果良好", "suggestions": ["可以更好"]}'
                    )
                )
            ]
        )

        judge = HarnessJudge(llm_client=mock_client)
        verdict = judge.evaluate(criteria="准确率 > 80%", evidence={"accuracy": 0.85})

        assert verdict.passed is True
        assert verdict.score == 0.85
        assert verdict.reasoning == "结果良好"
