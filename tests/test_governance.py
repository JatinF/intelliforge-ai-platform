"""Approval separation and audit visibility."""

from fastapi.testclient import TestClient

from gateway.app import create_app
from governance.auth import can_decide


def test_apply_waits_for_a_different_approver_and_rereads_the_tool():
    client = TestClient(create_app())
    engineer = {"Authorization": "Bearer dev-engineer-a"}
    approver = {"Authorization": "Bearer dev-approver-a"}
    opened = client.post(
        "/v1/investigations",
        headers=engineer,
        json={"case_id": "SYN-CASE-001", "action": "apply_setpoint"},
    )
    assert opened.status_code == 200
    body = opened.json()
    assert body["status"] == "AWAITING_APPROVAL"
    assert body["completion_claimed"] is False
    assert body["setpoint_applied"] is False
    denied = client.post(f"/v1/approvals/{body['approval_id']}/decision", headers=engineer, json={"decision": "approve"})
    assert denied.status_code == 403
    hidden = client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers={"Authorization": "Bearer dev-engineer-b"},
        json={"decision": "approve"},
    )
    assert hidden.status_code == 404
    accepted = client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        headers=approver,
        json={"decision": "approve"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "APPLIED"
    assert accepted.json()["setpoint_applied"] is True
    assert accepted.json()["completion_claimed"] is True
    audit = client.get(f"/v1/audit/{body['audit_id']}", headers=engineer)
    assert audit.status_code == 200
    assert audit.json()["model_version"] == "numpy-lora-r2-seed7"
    foreign = client.get(f"/v1/audit/{body['audit_id']}", headers={"Authorization": "Bearer dev-engineer-b"})
    assert foreign.status_code == 404


def test_rejected_approval_does_not_apply():
    client = TestClient(create_app())
    opened = client.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer dev-engineer-a"},
        json={"case_id": "SYN-CASE-001", "action": "apply_setpoint"},
    )
    approval_id = opened.json()["approval_id"]
    rejected = client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers={"Authorization": "Bearer dev-approver-a"},
        json={"decision": "reject"},
    )
    assert rejected.json()["status"] == "REJECTED"
    assert rejected.json()["setpoint_applied"] is False
    assert rejected.json()["completion_claimed"] is False


def test_reader_cannot_request_apply_and_same_subject_cannot_decide():
    client = TestClient(create_app())
    response = client.post(
        "/v1/investigations",
        headers={"Authorization": "Bearer dev-reader-a"},
        json={"case_id": "SYN-CASE-001", "action": "apply_setpoint"},
    )
    assert response.status_code == 403
    assert can_decide(
        {"tenant": "tenant-a", "role": "approver", "subject": "approver-a"},
        {"tenant": "tenant-a", "requester": "approver-a"},
    ) == (False, "same subject")
