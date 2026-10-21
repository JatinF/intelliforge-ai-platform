"""Remote model adapters.

Both adapters fail closed. The local benchmark never calls them.
"""

from __future__ import annotations

import os

import httpx

from serving.cost import GROK_MODEL

QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"


def vllm_base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "").strip()


def xai_api_key() -> str:
    return os.environ.get("XAI_API_KEY", "").strip()


def require_vllm() -> str:
    base = vllm_base_url()
    if not base:
        raise RuntimeError("VLLM_BASE_URL is unset")
    return base.rstrip("/")


def require_xai() -> str:
    key = xai_api_key()
    if not key:
        raise RuntimeError("XAI_API_KEY is unset")
    return key


def chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    *,
    client: httpx.Client | None = None,
    headers: dict | None = None,
) -> dict:
    """POST {base}/v1/chat/completions. vLLM and the local stand-in share this shape."""
    root = base_url.rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if client is None:
        response = httpx.post(f"{root}/v1/chat/completions", json=payload, headers=headers, timeout=30.0)
    else:
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def chat_vllm(prompt: str, base_url: str | None = None) -> dict:
    """OpenAI-compatible chat call. Reached only when a base URL is present."""
    root = require_vllm() if base_url is None else base_url.rstrip("/")
    if not root:
        raise RuntimeError("VLLM_BASE_URL is unset")
    return chat_completion(root, QWEN_MODEL, prompt)


def chat_xai(prompt: str, api_key: str | None = None) -> dict:
    """Chat completion against the xAI API. Reached only when a key is present."""
    key = require_xai() if api_key is None else api_key
    if not key:
        raise RuntimeError("XAI_API_KEY is unset")
    response = httpx.post(
        XAI_CHAT_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": GROK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()
