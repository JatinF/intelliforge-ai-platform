"""OpenAI-compatible chat completions for model replacement.

vLLM serves this same POST /v1/chat/completions shape. This process serves the
local heads on that shape. The Qwen model id returns 503 until a measured
weight load exists.
"""

from __future__ import annotations

import re

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from intelliforge.governance.auth import PlatformStore
from intelliforge.investigate import investigate
from intelliforge.serving.cost import whitespace_tokens
from intelliforge.serving.preflight import assess_qwen_load, read_available_bytes
from intelliforge.serving.remote import QWEN_MODEL

CASE_ID = re.compile(r"SYN-CASE-[A-Z0-9]+")
LOCAL_MODELS = {
    "numpy-lora-r2-seed7": "quality",
    "local-linear-w0-zeros": "cost",
}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage]
    stream: bool = False


def completion_payload(model: str, content: str, prompt: str, record: dict | None = None) -> dict:
    prompt_tokens = whitespace_tokens(prompt)
    completion_tokens = whitespace_tokens(content)
    body = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if record is not None:
        body["intelliforge"] = {
            "status": record["status"],
            "fault_class": record["fault_class"],
            "model_version": record["model_version"],
            "remote_called": record["remote_called"],
            "setpoint_applied": record["setpoint_applied"],
        }
    return body


def create_openai_app(available_bytes: int | None = None) -> FastAPI:
    app = FastAPI(title="IntelliForge OpenAI-compatible")
    app.state.store = PlatformStore()

    def _available() -> int:
        if available_bytes is None:
            return read_available_bytes()
        return available_bytes

    @app.post("/v1/chat/completions")
    def chat(body: ChatRequest, authorization: str | None = Header(default=None)) -> dict:
        if body.stream:
            raise HTTPException(status_code=400, detail="stream is unsupported")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="unauthorized")
        token = authorization.split(" ", 1)[1].strip()
        principal = app.state.store.principal(token)
        if principal is None:
            raise HTTPException(status_code=401, detail="unauthorized")
        prompt = "\n".join(message.content for message in body.messages if message.role == "user")
        if body.model == QWEN_MODEL:
            decision = dict(assess_qwen_load(_available()))
            decision["qwen_weight_load"] = "not-run"
            if decision["decision"] == "allow":
                decision["reason"] = "preflight allows a load and this process has not loaded the weights"
            raise HTTPException(status_code=503, detail=decision)
        policy = LOCAL_MODELS.get(body.model)
        if policy is None:
            raise HTTPException(status_code=400, detail="unknown model")
        found = CASE_ID.search(prompt)
        if found is None:
            raise HTTPException(status_code=400, detail="prompt is missing a case id")
        try:
            record = investigate(principal, found.group(0), policy, "diagnose", app.state.store)
        except PermissionError:
            raise HTTPException(status_code=403, detail="forbidden") from None
        except KeyError:
            raise HTTPException(status_code=404, detail="case not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return completion_payload(body.model, record["answer"], prompt, record)

    return app
