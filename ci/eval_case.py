"""Gold pass flag and DeepEval case text. No deepeval import."""

from __future__ import annotations

from typing import Any, Optional


def gold_passed(task_complete: Any) -> bool:
    """Ship gate: gold equality only. DeepEval is advisory."""
    return bool(task_complete)


def judge_input(gold: dict[str, Any]) -> str:
    """TaskCompletionMetric infers the task from `input`. State the gold action."""
    return (
        f"Task: resolve last-mile shipment {gold.get('shipment_id')} "
        f"per the operations playbook. "
        f"Required resolution: {gold.get('expected_resolution')}. "
        f"Required tone: {gold.get('expected_tone')}. "
        f"Escalate: {gold.get('should_escalate')}. "
        f"Exception: {gold.get('is_exception')}. "
        f"Context: {gold.get('ground_truth_reasoning') or ''}"
    )


def judge_expected(gold: dict[str, Any]) -> str:
    return (
        f"resolution={gold.get('expected_resolution')}; "
        f"tone={gold.get('expected_tone')}; "
        f"escalate={gold.get('should_escalate')}"
    )


def judge_actual(pred: dict[str, Any]) -> str:
    state = pred.get("state") or {}
    res = state.get("resolution_output") or {}
    comm = state.get("communication_output") or {}
    tools = state.get("tool_calls_log") or []
    traj = state.get("trajectory_log") or []
    lines = [
        f"resolution={res.get('resolution')}",
        f"is_exception={res.get('is_exception')}",
        f"tone={comm.get('tone_label')}",
        f"escalated={state.get('escalated')}",
        f"rationale={res.get('rationale') or ''}",
    ]
    if tools:
        lines.append("tools: " + " | ".join(str(t) for t in tools[:20]))
    if traj:
        lines.append("trajectory: " + " | ".join(str(t) for t in traj[:20]))
    return "\n".join(lines)


def apply_metric_fields(
    row: dict[str, Any],
    metric: Any,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    row["deepeval_score"] = getattr(metric, "score", None)
    success = getattr(metric, "success", None)
    row["deepeval_success"] = None if success is None else bool(success)
    row["deepeval_reason"] = (getattr(metric, "reason", None) or "")[:800] or None
    row["passed"] = gold_passed(row.get("task_complete"))
    if error:
        row["error"] = error[:800]
    return row
