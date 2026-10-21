"""Policy routes for the local head and the remote adapters."""

from __future__ import annotations

from serving.cost import GROK_MODEL
from serving.remote import QWEN_MODEL, vllm_base_url, xai_api_key

POLICIES = ("quality", "cost", "open-weight", "commercial")


def choose_route(policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy == "quality":
        return {
            "policy": policy,
            "route_id": "lora-local",
            "model_version": "numpy-lora-r2-seed7",
            "allowed": True,
            "reason": "quality policy uses the local LoRA head",
        }
    if policy == "cost":
        return {
            "policy": policy,
            "route_id": "baseline-local",
            "model_version": "local-linear-w0-zeros",
            "allowed": True,
            "reason": "cost policy uses the unadapted linear head",
        }
    if policy == "open-weight":
        if not vllm_base_url():
            return {
                "policy": policy,
                "route_id": "vllm-qwen",
                "model_version": QWEN_MODEL,
                "allowed": False,
                "reason": "VLLM_BASE_URL is unset",
            }
        return {
            "policy": policy,
            "route_id": "vllm-qwen",
            "model_version": QWEN_MODEL,
            "allowed": True,
            "reason": "open-weight policy selected the OpenAI-compatible endpoint",
        }
    if not xai_api_key():
        return {
            "policy": policy,
            "route_id": "commercial-xai",
            "model_version": GROK_MODEL,
            "allowed": False,
            "reason": "XAI_API_KEY is unset",
        }
    return {
        "policy": policy,
        "route_id": "commercial-xai",
        "model_version": GROK_MODEL,
        "allowed": True,
        "reason": "commercial policy selected the xAI endpoint",
    }
