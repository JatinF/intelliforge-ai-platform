"""Score a local investigation against a synthetic case."""

from __future__ import annotations


def score_case(result: dict, case: dict) -> dict:
    expected_evidence = set(case["expected_evidence"])
    evidence_ok = expected_evidence.issubset(set(result["evidence_ids"]))
    fault_ok = result["fault_class"] == case["expected_fault"]
    relationship_text = " ".join(result.get("relationships") or [])
    relationships_ok = all(item in relationship_text or item in " ".join(result["evidence_ids"]) for item in case["expected_mentions"])
    applied_ok = result["setpoint_applied"] is False and "The setpoint was not applied." in result.get("answer", "")
    completed = result["status"] == "COMPLETED" and result["completion_claimed"] is True
    passed = bool(evidence_ok and fault_ok and relationships_ok and applied_ok and completed)
    return {
        "case_id": case["case_id"],
        "passed": passed,
        "fault_ok": fault_ok,
        "evidence_ok": evidence_ok,
        "relationships_ok": relationships_ok,
        "applied_ok": applied_ok,
        "completed": completed,
    }


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    if count == 0:
        return {"cases": 0, "task_completion_rate": 0.0, "fault_accuracy": 0.0}
    return {
        "cases": count,
        "task_completion_rate": round(sum(1 for row in rows if row["passed"]) / count, 6),
        "fault_accuracy": round(sum(1 for row in rows if row["fault_ok"]) / count, 6),
    }
