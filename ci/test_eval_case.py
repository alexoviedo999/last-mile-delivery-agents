"""Unit tests for gold pass vs DeepEval case text. No LLM."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_case import apply_metric_fields, gold_passed, judge_actual, judge_input


def test_gold_passed_is_task_complete_only():
    assert gold_passed(True) is True
    assert gold_passed(False) is False
    assert gold_passed(None) is False


def test_apply_metric_fields_does_not_and_deepeval_into_passed():
    row = {"task_complete": True, "passed": False}
    metric = SimpleNamespace(score=0.0, success=False, reason="no tools called")
    apply_metric_fields(row, metric, error="Metrics: Task Completion failed.")
    assert row["passed"] is True
    assert row["deepeval_success"] is False
    assert row["deepeval_score"] == 0.0
    assert "Task Completion" in row["error"]


def test_gold_fail_stays_failed_when_deepeval_passes():
    row = {"task_complete": False}
    metric = SimpleNamespace(score=1.0, success=True, reason="picked an action")
    apply_metric_fields(row, metric)
    assert row["passed"] is False
    assert row["deepeval_success"] is True


def test_judge_input_states_gold_resolution_and_context():
    text = judge_input(
        {
            "shipment_id": "SHP-003",
            "expected_resolution": "RESCHEDULE",
            "expected_tone": "CASUAL",
            "should_escalate": "NO",
            "is_exception": "YES",
            "ground_truth_reasoning": "address issue is resolvable by rescheduling",
        }
    )
    assert "SHP-003" in text
    assert "RESCHEDULE" in text
    assert "Escalate: NO" in text
    assert "rescheduling" in text


def test_judge_actual_includes_resolution_tools_and_trajectory():
    pred = {
        "state": {
            "resolution_output": {
                "resolution": "REROUTE_TO_LOCKER",
                "is_exception": "YES",
                "rationale": "eligible locker nearby",
            },
            "communication_output": {"tone_label": "CASUAL"},
            "escalated": True,
            "tool_calls_log": ["AGENT: resolution_agent invoked"],
            "trajectory_log": ["critic_resolution: decision=REVISE"],
        }
    }
    text = judge_actual(pred)
    assert "REROUTE_TO_LOCKER" in text
    assert "resolution_agent" in text
    assert "REVISE" in text
    assert "escalated=True" in text
