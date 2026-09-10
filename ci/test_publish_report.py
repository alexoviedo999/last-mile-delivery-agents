"""Unit tests for evals-branch publish. No git remote, no LLM."""

from __future__ import annotations

from pathlib import Path

CI = Path(__file__).resolve().parent


def test_publish_fetches_evals_into_origin_ref():
    text = (CI / "publish_report.py").read_text(encoding="utf-8")
    assert "+refs/heads/evals:refs/remotes/origin/evals" in text
    assert 'fetch", "origin", "evals"' not in text
    assert "--verify" in text and "origin/evals" in text
    assert "non-fast-forward" in text
    assert "--rebase" in text
