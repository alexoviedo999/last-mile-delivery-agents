---
title: Last-mile DeepEval results
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.52.1
app_file: app.py
pinned: false
---

# Last-mile DeepEval trajectory results

A **reader**, not the operator console. It does not run the LangGraph app.
It displays `evals/latest.json` published by GitHub Actions after a live
`deepeval test run`.

Operator Space (gold equality, one shipment):
https://huggingface.co/spaces/alexoviedo999/last-mile-delivery-agents

Override the JSON URL with Space variable `RESULTS_URL` if needed.
