"""Unit tests for the HF DeepEval judge wiring. No LLM, no deepeval install."""

from __future__ import annotations

from pathlib import Path

CI = Path(__file__).resolve().parent


def test_judge_uses_hf_router_not_openai():
    text = (CI / "judge_model.py").read_text(encoding="utf-8")
    assert "class HfJudge" in text
    assert "DeepEvalBaseLLM" in text
    assert "router.huggingface.co/v1" in text
    assert "HF_TOKEN" in text
    assert "OPENAI_API_KEY" not in text
    assert "schema" in text


def test_trajectory_metric_passes_hf_judge():
    text = (CI / "test_last_mile_trajectory.py").read_text(encoding="utf-8")
    assert "from judge_model import HfJudge" in text
    wired = "TaskCompletionMetric(threshold=0.5, model=HfJudge())"
    assert wired in text
    leftover = text.replace(wired, "")
    assert "TaskCompletionMetric(" not in leftover


def test_assert_test_uses_llm_test_case_not_golden_plus_metrics():
    text = (CI / "test_last_mile_trajectory.py").read_text(encoding="utf-8")
    assert "assert_test(golden=golden, metrics=" not in text
    assert "LLMTestCase" in text
    assert "test_case=" in text
    assert "judge_input" in text
    assert "apply_metric_fields" in text
    assert 'deepeval_success"]) and bool(row["task_complete"])' not in text
