# CI trajectory scoring

The Streamlit UI scores one shipment with gold equality. This folder scores
the ordered LangGraph run so CI can fail a build when the trajectory regresses.

Do not add `deepeval` to `requirements.txt`. Operator clicks leave
`DEEPEVAL_TRACE` unset.

## Unit (no LLM)

GitHub Actions runs this on pull requests and on `main`:

```bash
python -m pytest ci -q
```

Checks: env gate, `_deepeval_invoke_config` on `invoke`, gold file has ten
shipments, Space requirements stay free of deepeval, live metric uses `HfJudge`
(HF router, not OpenAI), publish fetches `origin/evals` by refspec.

## Live DeepEval (LLM)

Opt-in. In the repo: **Actions → last-mile trajectory → Run workflow → live**.
Needs a GitHub secret named `HF_TOKEN` (Inference Providers; same as the Space).
`TaskCompletionMetric` is constructed with `HfJudge` so it does not read
`OPENAI_API_KEY`.

Or locally:

```bash
pip install -r requirements.txt -r ci/requirements-ci.txt
export DEEPEVAL_TRACE=1 DEEPEVAL_LIVE=1 HF_TOKEN=hf_...
deepeval test run ci/test_last_mile_trajectory.py
```

Live import of `app.py` needs Streamlit, data next to the app, and a model key.
Start with one shipment if you are debugging.

A live run writes `evals/latest.json` and pushes it to the `evals` branch.
The results Space reads that file; it does not invoke the graph:
https://huggingface.co/spaces/alexoviedo999/last-mile-deepeval-results

Docs: https://deepeval.com/integrations/frameworks/langgraph
