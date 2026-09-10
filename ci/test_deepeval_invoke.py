"""Unit tests for the CI DeepEval invoke gate. No LLM, no deepeval install required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parent
GUIDE = CI.parent
sys.path.insert(0, str(CI))

from deepeval_invoke import (  # noqa: E402
    APP_HELPER,
    apply_last_mile_app_patch,
    find_app_py,
    find_requirements,
    invoke_config,
    invoke_graph,
    trace_enabled,
)


class FakeHandler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGraph:
    def __init__(self):
        self.calls = []

    def invoke(self, state, config=None, **kwargs):
        self.calls.append({"state": state, "config": config, "kwargs": kwargs})
        return {"ok": True, "state": state}


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("no", False),
    ],
)
def test_trace_enabled_parses_flag(value, expected):
    assert trace_enabled({"DEEPEVAL_TRACE": value}) is expected


def test_trace_disabled_when_unset():
    assert trace_enabled({}) is False


def test_invoke_config_none_when_disabled():
    assert invoke_config(env={}) is None
    assert invoke_config(env={"DEEPEVAL_TRACE": "0"}) is None


def test_invoke_config_none_when_deepeval_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("deepeval"):
            raise ImportError("deepeval not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert invoke_config(env={"DEEPEVAL_TRACE": "1"}) is None


def test_invoke_config_attaches_handler_when_enabled():
    config = invoke_config(
        env={"DEEPEVAL_TRACE": "1"},
        handler_factory=lambda: FakeHandler(name="last-mile"),
    )
    assert config is not None
    assert len(config["callbacks"]) == 1
    assert config["callbacks"][0].kwargs["name"] == "last-mile"


def test_invoke_graph_skips_config_when_disabled():
    graph = FakeGraph()
    result = invoke_graph(graph, {"shipment_id": "SHP-001"}, env={})
    assert result["ok"] is True
    assert graph.calls[0]["config"] is None


def test_invoke_graph_passes_callback_when_enabled(monkeypatch):
    graph = FakeGraph()

    def fake_invoke_config(env=None, handler_factory=None):
        return {"callbacks": [FakeHandler(name="deepeval")]}

    monkeypatch.setattr("deepeval_invoke.invoke_config", fake_invoke_config)
    invoke_graph(
        graph,
        {"shipment_id": "SHP-001"},
        env={"DEEPEVAL_TRACE": "1"},
        config={"tags": ["eval"]},
    )
    call = graph.calls[0]
    assert call["config"]["tags"] == ["eval"]
    assert call["config"]["callbacks"][0].kwargs["name"] == "deepeval"


def test_invoke_graph_merges_existing_callbacks(monkeypatch):
    graph = FakeGraph()
    extra = FakeHandler(name="existing")

    def fake_invoke_config(env=None, handler_factory=None):
        return {"callbacks": [FakeHandler(name="deepeval")]}

    monkeypatch.setattr("deepeval_invoke.invoke_config", fake_invoke_config)
    invoke_graph(
        graph,
        {"shipment_id": "SHP-002"},
        env={"DEEPEVAL_TRACE": "1"},
        config={"callbacks": [extra]},
    )
    callbacks = graph.calls[0]["config"]["callbacks"]
    assert [c.kwargs["name"] for c in callbacks] == ["existing", "deepeval"]


def test_snapshot_script_reapplies_patch():
    script = GUIDE / "snapshot_sources.py"
    if not script.exists():
        pytest.skip("guide snapshot_sources.py not in this repo")
    text = script.read_text(encoding="utf-8")
    assert "_patch_last_mile_deepeval_callback" in text
    assert "apply_last_mile_app_patch" in text


def test_app_py_snapshot_wires_callback():
    app_py = find_app_py()
    text = app_py.read_text(encoding="utf-8")
    assert "def _deepeval_invoke_config" in text
    assert "DEEPEVAL_TRACE" in text
    assert "last_mile_app.invoke(initial_state, config=config)" in text
    assert "from deepeval.integrations.langchain import CallbackHandler" in text
    req = find_requirements().read_text(encoding="utf-8")
    assert "deepeval" not in req.lower()


def test_patcher_is_idempotent(tmp_path):
    stub = tmp_path / "app.py"
    stub.write_text(
        "def run_pipeline(shipment_id: str) -> dict:\n"
        "    initial_state = {}\n"
        "    result = last_mile_app.invoke(initial_state)\n"
        "    return result\n",
        encoding="utf-8",
    )
    assert apply_last_mile_app_patch(stub) is True
    once = stub.read_text(encoding="utf-8")
    assert HELPER_MARKER_PRESENT(once)
    assert "config=config" in once
    assert apply_last_mile_app_patch(stub) is False
    assert stub.read_text(encoding="utf-8") == once


def HELPER_MARKER_PRESENT(text: str) -> bool:
    return "# ─── DEEPEVAL CI TRACE" in text and "def _deepeval_invoke_config" in text


def test_helper_snippet_mentions_no_space_dependency():
    assert "not in Space requirements.txt" in APP_HELPER
    assert "CallbackHandler" in APP_HELPER
