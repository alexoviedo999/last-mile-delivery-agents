"""JSON report for the DeepEval results Space. Writer only — no LLM."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = "last-mile-deepeval-v1"
CI_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = CI_DIR.parent / "evals" / "latest.json"
HISTORY_PATH = CI_DIR.parent / "evals" / "history.jsonl"


def empty_report(*, note: Optional[str] = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": None,
        "source": "none",
        "run_url": None,
        "note": note
        or "No live DeepEval run yet. GitHub Actions → last-mile trajectory → live.",
        "shipments": [],
    }


def build_report(
    shipments: list[dict[str, Any]],
    *,
    run_url: Optional[str] = None,
    source: str = "github-actions",
) -> dict[str, Any]:
    passed = sum(1 for s in shipments if s.get("passed"))
    deepeval_passed = sum(1 for s in shipments if s.get("deepeval_success"))
    n = len(shipments)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "run_url": run_url or os.environ.get("GITHUB_RUN_URL") or None,
        "note": None,
        "passed": passed,
        "deepeval_passed": deepeval_passed,
        "total": n,
        "shipments": shipments,
    }


def write_latest(report: dict[str, Any], path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hist = path.parent / "history.jsonl"
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(report, separators=(",", ":")) + "\n")
    return path


def load_latest(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    if not path.exists():
        return empty_report()
    return json.loads(path.read_text(encoding="utf-8"))
