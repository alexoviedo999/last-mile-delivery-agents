"""DeepEval judge that uses the same HF Inference router as the Space.

TaskCompletionMetric defaults to OpenAI. This repo's live CI only has HF_TOKEN.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from deepeval.models.base_model import DeepEvalBaseLLM

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
HF_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge JSON was not an object")
    return value


class HfJudge(DeepEvalBaseLLM):
    def __init__(self) -> None:
        self._chat: Optional[ChatOpenAI] = None
        super().__init__()

    def load_model(self):
        if self._chat is None:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for the DeepEval judge")
            self._chat = ChatOpenAI(
                model=HF_CHAT_MODEL,
                temperature=0,
                api_key=token,
                base_url=HF_ROUTER_BASE_URL,
            )
        return self._chat

    def generate(self, prompt: str, schema: Any = None) -> Any:
        chat = self.load_model()
        if schema is not None:
            prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema:\n"
                + json.dumps(schema.model_json_schema())
            )
        text = _content_to_text(chat.invoke(prompt).content)
        if schema is None:
            return text
        return schema.model_validate(_parse_json(text))

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        chat = self.load_model()
        if schema is not None:
            prompt = (
                prompt
                + "\n\nReturn ONLY valid JSON matching this schema:\n"
                + json.dumps(schema.model_json_schema())
            )
        msg = await chat.ainvoke(prompt)
        text = _content_to_text(msg.content)
        if schema is None:
            return text
        return schema.model_validate(_parse_json(text))

    def get_model_name(self) -> str:
        return HF_CHAT_MODEL
