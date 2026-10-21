"""Open-weight preflight and the OpenAI-compatible replacement call."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from routing.policy import choose_route
from serving.openai_compat import LOCAL_MODELS, create_openai_app
from serving.preflight import (
    QWEN_PARAMETER_BYTES,
    QWEN_PARAMETER_COUNT,
    REQUIRED_AVAILABLE_BYTES,
    assess_qwen_load,
)
from serving.remote import QWEN_MODEL, chat_completion
from training import peft_lora

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Why is SYN-PROC-ETCH-01 producing abnormal chamber temperature readings? SYN-CASE-001"


def test_three_billion_checkpoint_is_refused_below_the_weight_size():
    decision = assess_qwen_load(1024)
    assert decision["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert decision["parameter_count"] == QWEN_PARAMETER_COUNT == 3_085_938_688
    assert decision["parameter_bytes"] == QWEN_PARAMETER_BYTES
    assert decision["decision"] == "refuse"
    assert decision["qwen_weight_load"] == "not-run"
    assert REQUIRED_AVAILABLE_BYTES > QWEN_PARAMETER_BYTES


def test_model_field_selects_the_local_head_or_stops_qwen():
    assert LOCAL_MODELS["numpy-lora-r2-seed7"] == "quality"
    assert choose_route("quality")["model_version"] == "numpy-lora-r2-seed7"
    assert choose_route("cost")["model_version"] == "local-linear-w0-zeros"
    app = create_openai_app(available_bytes=1024)
    headers = {"Authorization": "Bearer dev-engineer-a"}
    with TestClient(app) as client:
        lora = chat_completion("http://testserver", "numpy-lora-r2-seed7", PROMPT, client=client, headers=headers)
        baseline = chat_completion("http://testserver", "local-linear-w0-zeros", PROMPT, client=client, headers=headers)
        refused = client.post(
            "/v1/chat/completions",
            json={"model": QWEN_MODEL, "messages": [{"role": "user", "content": PROMPT}], "stream": False},
            headers=headers,
        )
    assert lora["model"] == "numpy-lora-r2-seed7"
    assert lora["intelliforge"]["status"] == "COMPLETED"
    assert lora["intelliforge"]["fault_class"] == "cooling_flow_low"
    assert "The setpoint was not applied." in lora["choices"][0]["message"]["content"]
    assert baseline["intelliforge"]["status"] == "STOPPED"
    assert baseline["intelliforge"]["fault_class"] is None
    assert refused.status_code == 503
    assert refused.json()["detail"]["qwen_weight_load"] == "not-run"
    assert refused.json()["detail"]["parameter_count"] == 3_085_938_688


def test_peft_entry_exits_before_importing_torch(monkeypatch):
    monkeypatch.setattr(peft_lora, "read_available_bytes", lambda: 1024)
    with pytest.raises(SystemExit) as caught:
        peft_lora.main()
    assert caught.value.code == 2


def test_committed_probe_matches_the_preflight_constants():
    payload = json.loads((ROOT / "benchmarks" / "results" / "openweight_probe.json").read_text(encoding="utf-8"))
    preflight = payload["preflight"]
    assert preflight["parameter_count"] == QWEN_PARAMETER_COUNT
    assert preflight["parameter_bytes"] == QWEN_PARAMETER_BYTES
    assert preflight["available_bytes"] < preflight["required_available_bytes"]
    assert preflight["decision"] == "refuse"
    assert payload["gates"]["vllm-qwen"] == "not-run"
    assert payload["gates"]["peft-hf-train"] == "not-run"
    assert payload["gates"]["qwen-weight-load"] == "not-run"
    assert payload["imports"]["vllm"] == "not-installed"
    assert payload["imports"]["peft"] == "not-installed"
    assert payload["replacement"]["qwen_http_status"] == 503
    assert payload["replacement"]["local_lora_status"] == "COMPLETED"
    assert payload["replacement"]["baseline_status"] == "STOPPED"
