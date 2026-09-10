"""Attach DeepEval's LangGraph CallbackHandler on invoke for CI trajectory scoring.

Streamlit clicks leave DEEPEVAL_TRACE unset, so the Space does not import deepeval
and does not score every operator click. A CI job sets DEEPEVAL_TRACE=1 and runs
`deepeval test run` against the recorded spans.

Official pattern (https://deepeval.com/integrations/frameworks/langgraph):

    graph.invoke(state, config={"callbacks": [CallbackHandler()]})
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

TRACE_ENV = "DEEPEVAL_TRACE"
TRUE_VALUES = {"1", "true", "yes", "on"}
HANDLER_NAME = "last-mile"
HANDLER_TAGS = ["ci", "trajectory"]
CI_DIR = Path(__file__).resolve().parent


def find_app_py() -> Path:
    for candidate in (
        CI_DIR.parent / "app.py",
        CI_DIR.parent / "sources" / "last-mile" / "app.py",
        Path.cwd() / "app.py",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("last-mile app.py")


def find_gold_csv() -> Path:
    for candidate in (
        CI_DIR.parent / "data" / "ground_truth.csv",
        CI_DIR.parent.parent.parent / "Project-3-last-mile" / "data" / "ground_truth.csv",
        Path.cwd() / "data" / "ground_truth.csv",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ground_truth.csv")


def find_requirements() -> Path:
    for candidate in (
        CI_DIR.parent / "requirements.txt",
        CI_DIR.parent / "sources" / "last-mile" / "requirements.txt",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("requirements.txt")

HELPER_MARKER = "# ─── DEEPEVAL CI TRACE"
INVOKE_BARE = "    result = last_mile_app.invoke(initial_state)"
INVOKE_GATED = """    config = _deepeval_invoke_config()
    result = (
        last_mile_app.invoke(initial_state, config=config)
        if config
        else last_mile_app.invoke(initial_state)
    )"""

APP_HELPER = '''# ─── DEEPEVAL CI TRACE ────────────────────────────────────────────────────────
# Streamlit clicks leave DEEPEVAL_TRACE unset. CI sets it so CallbackHandler
# records the LangGraph trajectory. deepeval is not in Space requirements.txt.


def _deepeval_invoke_config():
    flag = os.environ.get("DEEPEVAL_TRACE", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None
    try:
        from deepeval.integrations.langchain import CallbackHandler
    except ImportError:
        return None
    return {
        "callbacks": [
            CallbackHandler(
                name="last-mile",
                tags=["ci", "trajectory"],
                metadata={"graph": "last_mile_app"},
            )
        ]
    }


'''


def trace_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(TRACE_ENV, "")).strip().lower() in TRUE_VALUES


def invoke_config(
    env: Optional[Mapping[str, str]] = None,
    handler_factory: Optional[Callable[[], Any]] = None,
) -> Optional[dict]:
    """Return LangGraph invoke config with CallbackHandler, or None.

    `handler_factory` is a test seam so unit tests do not import deepeval.
    """
    if not trace_enabled(env):
        return None
    if handler_factory is None:
        try:
            from deepeval.integrations.langchain import CallbackHandler
        except ImportError:
            return None

        def handler_factory() -> Any:
            return CallbackHandler(
                name=HANDLER_NAME,
                tags=list(HANDLER_TAGS),
                metadata={"graph": "last_mile_app"},
            )

    return {"callbacks": [handler_factory()]}


def invoke_graph(graph: Any, state: Any, env: Optional[Mapping[str, str]] = None, **kwargs: Any) -> Any:
    """Invoke a compiled LangGraph app, attaching CallbackHandler when tracing."""
    extra = invoke_config(env)
    config = dict(kwargs.pop("config", None) or {})
    if extra:
        callbacks = list(config.get("callbacks", [])) + list(extra["callbacks"])
        config["callbacks"] = callbacks
        return graph.invoke(state, config=config, **kwargs)
    if config:
        return graph.invoke(state, config=config, **kwargs)
    return graph.invoke(state, **kwargs)


def apply_last_mile_app_patch(app_py: Path) -> bool:
    """Idempotently wire `_deepeval_invoke_config` into last-mile Space `app.py`.

    Snapshots of `app.py` are fetched from GitHub and would otherwise lose the
    callback. Returns True if the file changed.
    """
    text = app_py.read_text(encoding="utf-8")
    original = text
    if HELPER_MARKER not in text:
        needle = "def run_pipeline(shipment_id: str) -> dict:"
        if needle not in text:
            raise ValueError(f"{app_py}: cannot find run_pipeline to insert DeepEval helper")
        text = text.replace(needle, APP_HELPER + needle, 1)
    if "last_mile_app.invoke(initial_state, config=config)" not in text:
        if INVOKE_BARE not in text:
            raise ValueError(f"{app_py}: cannot find bare last_mile_app.invoke to patch")
        text = text.replace(INVOKE_BARE, INVOKE_GATED, 1)
    if text == original:
        return False
    app_py.write_text(text, encoding="utf-8")
    return True
