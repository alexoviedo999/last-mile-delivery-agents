"""CI reader for last-mile DeepEval trajectory reports.

This is not the operator console. It never calls last_mile_app.invoke.
It only loads JSON published by GitHub Actions (evals branch).
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_URLS = [
    os.environ.get("RESULTS_URL") or "",
    "https://raw.githubusercontent.com/alexoviedo999/last-mile-delivery-agents/evals/evals/latest.json",
    "https://raw.githubusercontent.com/alexoviedo999/last-mile-delivery-agents/main/evals/latest.json",
]

st.set_page_config(
    page_title="Last-mile DeepEval results",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
<style>
.hero {
  background: linear-gradient(135deg, #1a2332, #243044);
  color: #e7ecf3; padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1.2rem;
  border: 1px solid #2d3a4f;
}
.hero h1 { margin: 0 0 .35rem; font-size: 1.6rem; }
.hero p { margin: 0; color: #9aa8bc; }
.pill {
  display: inline-block; padding: .2rem .6rem; border-radius: 999px;
  background: #0b1020; border: 1px solid #2d3a4f; color: #6ee7b7;
  font-size: .8rem; margin-right: .35rem;
}
</style>
<div class="hero">
  <h1>Last-mile DeepEval trajectory results</h1>
  <p>CI reader. Not the operator console — it does not run the graph.
  Gold equality still ships on the live Space. These rows are the ordered-run scores.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def fetch_report() -> tuple[dict[str, Any], str]:
    errors = []
    for url in DEFAULT_URLS:
        if not url:
            continue
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json(), url
            errors.append(f"{url} → HTTP {r.status_code}")
        except Exception as exc:
            errors.append(f"{url} → {exc}")
    local = os.path.join(os.path.dirname(__file__), "..", "..", "evals", "latest.json")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as fh:
            return json.load(fh), local
    return {
        "schema": "last-mile-deepeval-v1",
        "generated_at": None,
        "source": "none",
        "note": "No report found. Merge the CI report PR, then run Actions → last-mile trajectory → live.",
        "shipments": [],
        "errors": errors,
    }, ""


report, source = fetch_report()

c1, c2, c3, c4 = st.columns(4)
shipments = report.get("shipments") or []
c1.metric("Shipments", report.get("total") if report.get("total") is not None else len(shipments))
c2.metric("Passed", report.get("passed") if report.get("passed") is not None else "—")
c3.metric("Generated", report.get("generated_at") or "never")
c4.metric("Source", report.get("source") or "none")

if source:
    st.caption(f"Loaded from `{source}`")
if report.get("run_url"):
    st.markdown(f"[GitHub Actions run]({report['run_url']})")
if report.get("note"):
    st.info(report["note"])

operator = "https://huggingface.co/spaces/alexoviedo999/last-mile-delivery-agents"
st.markdown(
    f'<span class="pill">DeepEval CI</span> '
    f'<span class="pill">no invoke</span> '
    f'<a href="{operator}">Operator Space (gold equality)</a>',
    unsafe_allow_html=True,
)

if not shipments:
    st.warning(
        "Empty report. After `HF_TOKEN` is a GitHub secret, run "
        "**Actions → last-mile trajectory → Run workflow → live**."
    )
    st.stop()

rows = []
for s in shipments:
    rows.append(
        {
            "shipment": s.get("shipment_id"),
            "passed": s.get("passed"),
            "gold task complete": s.get("task_complete"),
            "escalation": s.get("escalation_correct"),
            "resolution": s.get("resolution"),
            "expected": s.get("expected_resolution"),
            "DeepEval score": s.get("deepeval_score"),
            "DeepEval pass": s.get("deepeval_success"),
            "error": s.get("error"),
        }
    )
st.subheader("Shipments")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

with st.expander("DeepEval reasons"):
    for s in shipments:
        reason = s.get("deepeval_reason") or s.get("error") or "—"
        st.markdown(f"**{s.get('shipment_id')}** — {reason}")

st.caption(
    "Task complete / escalation are gold equality from the graph state. "
    "DeepEval score is TaskCompletionMetric on the traced trajectory."
)
