"""Unit tests for the eval JSON report. No LLM."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import build_report, empty_report, load_latest, write_latest


def test_empty_report_has_schema_and_no_shipments():
    r = empty_report()
    assert r["schema"] == "last-mile-deepeval-v1"
    assert r["shipments"] == []
    assert r["generated_at"] is None


def test_write_and_load_latest(tmp_path):
    path = tmp_path / "latest.json"
    report = build_report(
        [
            {
                "shipment_id": "SHP-001",
                "passed": True,
                "task_complete": True,
                "deepeval_score": 0.9,
            }
        ],
        run_url="https://example.test/run/1",
        source="test",
    )
    write_latest(report, path)
    loaded = load_latest(path)
    assert loaded["total"] == 1
    assert loaded["passed"] == 1
    assert loaded["shipments"][0]["shipment_id"] == "SHP-001"
    hist = tmp_path / "history.jsonl"
    assert hist.exists()
    assert "SHP-001" in hist.read_text(encoding="utf-8")
