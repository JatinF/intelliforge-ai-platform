"""Local diagnosis, retrieval, and remote fail-closed routes."""

import httpx
from fastapi.testclient import TestClient

from gateway.app import create_app
from governance.auth import PlatformStore
from gateway.investigate import investigate
from knowledge.store import load_cases, retrieve
from serving.local_lora import train_and_compare


def _engineer(tenant: str = "tenant-a") -> dict:
    subject = "engineer-a" if tenant == "tenant-a" else "engineer-b"
    return {"tenant": tenant, "role": "engineer", "subject": subject, "token": "dev-engineer-a"}


def test_cooling_case_retrieves_only_the_caller_tenant():
    question = "Why is SYN-PROC-ETCH-01 producing abnormal chamber temperature readings?"
    ranked = retrieve("tenant-a", question)
    assert ranked[0]["id"] == "SYN-DOC-COOLING"
    assert "SYN-DOC-OTHER" not in {item["id"] for item in ranked}


def test_quality_route_completes_without_applying_a_setpoint():
    adapter = train_and_compare()
    result = investigate(_engineer(), "SYN-CASE-001", "quality", "diagnose", PlatformStore(), adapter)
    assert result["status"] == "COMPLETED"
    assert result["completion_claimed"] is True
    assert result["setpoint_applied"] is False
    assert result["fault_class"] == "cooling_flow_low"
    assert result["evidence_ids"] == ["SYN-DOC-COOLING"]
    assert "The setpoint was not applied." in result["answer"]
    assert 0.9 <= result["recommended_flow"] <= 1.0
    assert result["remote_called"] is False
    assert result["model_version"] == "numpy-lora-r2-seed7"


def test_baseline_route_stops_when_the_head_disagrees():
    adapter = train_and_compare()
    result = investigate(_engineer(), "SYN-CASE-001", "cost", "diagnose", PlatformStore(), adapter)
    assert result["status"] == "STOPPED"
    assert result["completion_claimed"] is False
    assert result["fault_class"] is None
    assert result["predicted_fault"] == "normal"


def test_open_weight_and_commercial_routes_do_not_call_out(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("httpx was called")

    monkeypatch.setattr(httpx, "post", explode)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    adapter = train_and_compare()
    store = PlatformStore()
    open_weight = investigate(_engineer(), "SYN-CASE-001", "open-weight", "diagnose", store, adapter)
    commercial = investigate(_engineer(), "SYN-CASE-001", "commercial", "diagnose", store, adapter)
    assert open_weight["status"] == "STOPPED"
    assert commercial["status"] == "STOPPED"
    assert open_weight["remote_called"] is False
    assert commercial["remote_called"] is False
    assert open_weight["model_version"] == "Qwen/Qwen2.5-3B-Instruct"
    assert commercial["model_version"] == "grok-4.7"


def test_other_tenant_case_is_hidden():
    client = TestClient(create_app())
    response = client.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer dev-engineer-b"},
        json={"case_id": "SYN-CASE-001"},
    )
    assert response.status_code == 404
    assert load_cases()[0]["tenant"] == "tenant-a"


def test_gateway_rejects_an_injected_fault_and_reports_health():
    client = TestClient(create_app())
    health = client.get("/healthz")
    assert health.json()["real_equipment"] == 0
    rejected = client.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer dev-engineer-a"},
        json={"case_id": "SYN-CASE-001", "expected_fault": "normal"},
    )
    assert rejected.status_code == 422
    missing = client.post("/v1/investigations", json={"case_id": "SYN-CASE-001"})
    assert missing.status_code == 401
