"""Synthetic equipment investigation.

A run retrieves tenant documents, selects a route, classifies with the local
head, and may call the SciPy tool. Applying a setpoint waits for a different approver.
"""

from __future__ import annotations

import time

from governance.auth import can_decide, can_diagnose, can_request_apply
from knowledge.store import load_cases, load_documents, load_edges, retrieve, walk
from serving.local_lora import load_adapter, predict_label
from routing.policy import choose_route
from serving.cost import estimate_local_cost, whitespace_tokens
from simulation.setpoint import recommend_flow


def case_by_id(case_id: str, tenant: str, cases: list[dict] | None = None) -> dict | None:
    for case in cases if cases is not None else load_cases():
        if case["case_id"] == case_id and case["tenant"] == tenant:
            return case
    return None


def _answer(case: dict, fault: str, evidence_ids: list[str], relationships: list[str], flow: float, applied: bool) -> str:
    applied_text = "The setpoint was applied." if applied else "The setpoint was not applied."
    relation_text = ", ".join(relationships) if relationships else "none"
    evidence_text = ", ".join(evidence_ids) if evidence_ids else "none"
    return (
        f"Process {case['process_id']} fault class is {fault}. "
        f"Evidence: {evidence_text}. "
        f"Relationships: {relation_text}. "
        f"Recommended coolant flow fraction is {flow:.4f}. "
        f"{applied_text}"
    )


def _local_decision(case: dict, route: dict, adapter: dict, documents: list[dict], edges: list[dict]) -> dict:
    question = case["question"]
    ranked = retrieve(case["tenant"], question, documents)
    relationships = walk(case["tenant"], case["process_id"], edges)
    top = ranked[0] if ranked else None
    predicted = predict_label(case["readings"], adapter, route["route_id"])
    tool = recommend_flow(case["readings"]["temperature_c"], case["readings"]["target_temperature_c"])
    supported = top is not None and top["fault_class"] == predicted
    evidence_ids = [top["id"]] if supported else []
    if supported:
        status = "COMPLETED"
        completion_claimed = True
        reason = "top document supports the local head"
    else:
        status = "STOPPED"
        completion_claimed = False
        reason = "local head and top document disagree"
    answer = _answer(case, predicted, evidence_ids, relationships, tool["flow_fraction"], False)
    prompt = question + "\n" + "\n".join(document["text"] for document in ranked)
    return {
        "status": status,
        "completion_claimed": completion_claimed,
        "reason": reason,
        "fault_class": predicted if supported else None,
        "predicted_fault": predicted,
        "evidence_ids": evidence_ids,
        "retrieved_ids": [document["id"] for document in ranked],
        "relationships": relationships,
        "tool": tool,
        "recommended_flow": tool["flow_fraction"],
        "setpoint_applied": False,
        "answer": answer,
        "prompt_tokens": whitespace_tokens(prompt),
        "completion_tokens": whitespace_tokens(answer),
    }


def investigate(
    principal: dict,
    case_id: str,
    policy: str = "quality",
    action: str = "diagnose",
    store=None,
    adapter: dict | None = None,
    now: float | None = None,
) -> dict:
    started = time.perf_counter()
    if action not in {"diagnose", "apply_setpoint"}:
        raise ValueError("unknown action")
    if action == "diagnose" and not can_diagnose(principal):
        raise PermissionError("role")
    if action == "apply_setpoint" and not can_request_apply(principal):
        raise PermissionError("role")
    case = case_by_id(case_id, principal["tenant"])
    if case is None:
        raise KeyError(case_id)
    route = choose_route(policy)
    run_id = store.next_id("SYN-RUN") if store is not None else "SYN-RUN-LOCAL"
    if route["route_id"] in {"vllm-qwen", "commercial-xai"}:
        if route["allowed"]:
            route = dict(route)
            route["reason"] = "remote route is recorded and not executed by the local path"
        return _stopped_remote(principal, case, route, run_id, started, store)
    head = adapter if adapter is not None else load_adapter()
    decision = _local_decision(case, route, head, load_documents(), load_edges())
    if action == "apply_setpoint":
        return _open_approval(principal, case, route, decision, run_id, started, store, now)
    return _finish(principal, case, route, decision, run_id, started, store, "diagnose")


def _stopped_remote(principal, case, route, run_id, started, store) -> dict:
    latency_ms = (time.perf_counter() - started) * 1000.0
    cost = estimate_local_cost(0, 0)
    record = {
        "run_id": run_id,
        "case_id": case["case_id"],
        "tenant": principal["tenant"],
        "subject": principal["subject"],
        "action": "diagnose",
        "status": "STOPPED",
        "completion_claimed": False,
        "reason": route["reason"],
        "fault_class": None,
        "evidence_ids": [],
        "relationships": [],
        "model_route": route["route_id"],
        "model_version": route["model_version"],
        "policy": route["policy"],
        "setpoint_applied": False,
        "recommended_flow": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": round(latency_ms, 3),
        "estimated_cost": cost,
        "answer": "",
        "remote_called": False,
    }
    return _audit(store, record)


def _open_approval(principal, case, route, decision, run_id, started, store, now) -> dict:
    if decision["status"] != "COMPLETED":
        return _finish(principal, case, route, decision, run_id, started, store, "apply_setpoint")
    moment = time.time() if now is None else now
    approval_id = store.next_id("SYN-APR") if store is not None else "SYN-APR-LOCAL"
    approval = {
        "approval_id": approval_id,
        "run_id": run_id,
        "tenant": principal["tenant"],
        "requester": principal["subject"],
        "case_id": case["case_id"],
        "recommended_flow": decision["recommended_flow"],
        "readings": dict(case["readings"]),
        "expires_at": moment + 3600.0,
        "status": "AWAITING_APPROVAL",
    }
    if store is not None:
        store.approvals[approval_id] = approval
    decision = dict(decision)
    decision["status"] = "AWAITING_APPROVAL"
    decision["completion_claimed"] = False
    decision["setpoint_applied"] = False
    decision["reason"] = "apply waits for a different approver"
    finished = _finish(principal, case, route, decision, run_id, started, store, "apply_setpoint")
    finished["approval_id"] = approval_id
    return finished


def decide_approval(principal: dict, approval_id: str, decision: str, store, now: float | None = None) -> dict:
    approval = store.approvals.get(approval_id)
    if approval is None or approval["tenant"] != principal["tenant"]:
        raise KeyError(approval_id)
    allowed, reason = can_decide(principal, approval)
    moment = time.time() if now is None else now
    if not allowed:
        raise PermissionError(reason)
    if moment > float(approval["expires_at"]):
        approval["status"] = "EXPIRED"
        return _decision_record(principal, approval, "EXPIRED", False, "approval expired")
    if decision == "reject":
        approval["status"] = "REJECTED"
        return _decision_record(principal, approval, "REJECTED", False, "approver rejected")
    if decision != "approve":
        raise ValueError("unknown decision")
    fresh = recommend_flow(approval["readings"]["temperature_c"], approval["readings"]["target_temperature_c"])
    if abs(fresh["flow_fraction"] - float(approval["recommended_flow"])) > 1e-6:
        approval["status"] = "UNVERIFIED"
        return _decision_record(principal, approval, "UNVERIFIED", False, "recomputed flow does not match")
    approval["status"] = "APPLIED"
    record = _decision_record(principal, approval, "APPLIED", True, "approver accepted and the tool reread matched")
    record["setpoint_applied"] = True
    record["completion_claimed"] = True
    record["recommended_flow"] = fresh["flow_fraction"]
    return record


def _decision_record(principal, approval, status, applied, reason) -> dict:
    return {
        "approval_id": approval["approval_id"],
        "run_id": approval["run_id"],
        "tenant": principal["tenant"],
        "subject": principal["subject"],
        "status": status,
        "completion_claimed": False,
        "setpoint_applied": applied,
        "reason": reason,
        "recommended_flow": approval["recommended_flow"],
    }


def _finish(principal, case, route, decision, run_id, started, store, action) -> dict:
    latency_ms = (time.perf_counter() - started) * 1000.0
    cost = estimate_local_cost(decision["prompt_tokens"], decision["completion_tokens"])
    record = {
        "run_id": run_id,
        "case_id": case["case_id"],
        "tenant": principal["tenant"],
        "subject": principal["subject"],
        "action": action,
        "status": decision["status"],
        "completion_claimed": decision["completion_claimed"],
        "reason": decision["reason"],
        "fault_class": decision["fault_class"],
        "predicted_fault": decision["predicted_fault"],
        "evidence_ids": decision["evidence_ids"],
        "retrieved_ids": decision["retrieved_ids"],
        "relationships": decision["relationships"],
        "model_route": route["route_id"],
        "model_version": route["model_version"],
        "policy": route["policy"],
        "setpoint_applied": decision["setpoint_applied"],
        "recommended_flow": decision["recommended_flow"],
        "tool": decision["tool"],
        "prompt_tokens": decision["prompt_tokens"],
        "completion_tokens": decision["completion_tokens"],
        "latency_ms": round(latency_ms, 3),
        "estimated_cost": cost,
        "answer": decision["answer"],
        "remote_called": False,
    }
    return _audit(store, record)


def _audit(store, record: dict) -> dict:
    if store is None:
        record["audit_id"] = "SYN-AUD-LOCAL"
        return record
    record["audit_id"] = store.next_id("SYN-AUD")
    store.runs[record["run_id"]] = record
    store.append_audit(
        {
            "audit_id": record["audit_id"],
            "run_id": record["run_id"],
            "tenant": record["tenant"],
            "subject": record["subject"],
            "status": record["status"],
            "model_route": record["model_route"],
            "model_version": record["model_version"],
            "evidence_ids": list(record["evidence_ids"]),
            "prompt_tokens": record["prompt_tokens"],
            "completion_tokens": record["completion_tokens"],
            "latency_ms": record["latency_ms"],
            "estimated_cost_usd": record["estimated_cost"]["usd"],
            "setpoint_applied": record["setpoint_applied"],
        }
    )
    return record
