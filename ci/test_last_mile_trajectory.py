"""CI job: score the last-mile LangGraph trajectory with DeepEval.

Gold equality (`compute_task_completion`, `compute_escalation_accuracy`) remains
the ship gate in the Space UI. This file scores the *ordered run* — nodes, LLM
calls, tools — via CallbackHandler on invoke.

Default `pytest` skips the live graph. Set DEEPEVAL_LIVE=1 (and HF_TOKEN or
OPENAI_API_KEY depending on the model) then:

    DEEPEVAL_TRACE=1 deepeval test run ci/test_last_mile_trajectory.py

See https://deepeval.com/integrations/frameworks/langgraph
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parent
sys.path.insert(0, str(CI))

from deepeval_invoke import find_app_py, find_gold_csv, invoke_config, trace_enabled  # noqa: E402
from report import build_report, write_latest  # noqa: E402

LIVE = os.environ.get("DEEPEVAL_LIVE", "").strip() == "1"
_LIVE_ROWS: list[dict] = []


def _gold_path() -> Path:
    return find_gold_csv()


def _last_yes_rows(path: Path) -> list[dict]:
    """Notebook consolidation rule: last YES row per shipment_id wins."""
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sid = row["shipment_id"]
            if sid not in by_id or row.get("is_exception") == "YES":
                by_id[sid] = row
    return list(by_id.values())


def test_gold_file_has_ten_shipments():
    path = _gold_path()
    assert path.exists(), path
    rows = _last_yes_rows(path)
    ids = [r["shipment_id"] for r in rows]
    assert ids == [f"SHP-{n:03d}" for n in range(1, 11)]


def test_ci_env_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_TRACE", raising=False)
    assert trace_enabled() is False
    assert invoke_config(env={}, handler_factory=lambda: object()) is None


def _live_goldens():
    try:
        return _last_yes_rows(_gold_path())
    except FileNotFoundError:
        return []


@pytest.fixture(scope="session", autouse=True)
def _persist_live_report():
    yield
    if not LIVE or not _LIVE_ROWS:
        return
    write_latest(
        build_report(
            _LIVE_ROWS,
            run_url=os.environ.get("GITHUB_RUN_URL"),
            source="github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        )
    )


@pytest.mark.skipif(not LIVE, reason="set DEEPEVAL_LIVE=1 to run LLM trajectory eval")
@pytest.mark.parametrize("gold", _live_goldens())
def test_last_mile_trajectory(gold):
    pytest.importorskip("deepeval")
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric
    from deepeval.test_case import LLMTestCase

    from judge_model import HfJudge

    os.environ["DEEPEVAL_TRACE"] = "1"
    sys.path.insert(0, str(find_app_py().parent))
    try:
        from app import run_pipeline  # type: ignore
    except Exception as exc:
        pytest.skip(f"cannot import last-mile app.py: {exc}")

    assert invoke_config() is not None, "DEEPEVAL_TRACE must be set for trajectory scoring"
    metric = TaskCompletionMetric(threshold=0.5, model=HfJudge())
    row = {
        "shipment_id": gold["shipment_id"],
        "expected_resolution": gold.get("expected_resolution"),
        "passed": False,
        "task_complete": None,
        "escalation_correct": None,
        "resolution": None,
        "deepeval_score": None,
        "deepeval_success": None,
        "deepeval_reason": None,
        "error": None,
    }
    try:
        pred = run_pipeline(gold["shipment_id"])
        tc = pred.get("task_completion") or {}
        row["task_complete"] = tc.get("task_complete")
        row["escalation_correct"] = pred.get("escalation_correct")
        row["resolution"] = (pred.get("state") or {}).get("resolution_output", {}).get("resolution")
        # Installed deepeval rejects golden=+metrics=. test_case=+metrics= is valid
        # on 3.x and 4.x; CallbackHandler still records the LangGraph run.
        assert_test(
            test_case=LLMTestCase(
                input=gold["shipment_id"],
                actual_output=row["resolution"] or "",
                expected_output=gold.get("expected_resolution") or "",
            ),
            metrics=[metric],
            run_async=False,
        )
        row["deepeval_score"] = getattr(metric, "score", None)
        row["deepeval_success"] = bool(getattr(metric, "success", False))
        row["deepeval_reason"] = (getattr(metric, "reason", None) or "")[:800]
        row["passed"] = bool(row["deepeval_success"]) and bool(row["task_complete"])
    except Exception as exc:
        row["error"] = str(exc)[:800]
        raise
    finally:
        _LIVE_ROWS.append(row)
