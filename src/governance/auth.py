"""Fixture principals, approvals, and an append-only audit list."""

from __future__ import annotations

DEV_TOKENS = {
    "dev-reader-a": {"tenant": "tenant-a", "role": "reader", "subject": "reader-a"},
    "dev-engineer-a": {"tenant": "tenant-a", "role": "engineer", "subject": "engineer-a"},
    "dev-approver-a": {"tenant": "tenant-a", "role": "approver", "subject": "approver-a"},
    "dev-engineer-b": {"tenant": "tenant-b", "role": "engineer", "subject": "engineer-b"},
}


class PlatformStore:
    def __init__(self) -> None:
        self.audits: list[dict] = []
        self.approvals: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self._sequence = 0

    def next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}-{self._sequence:04d}"

    def principal(self, token: str | None) -> dict | None:
        if not token:
            return None
        record = DEV_TOKENS.get(token)
        if record is None:
            return None
        return {"token": token, **record}

    def append_audit(self, record: dict) -> dict:
        self.audits.append(record)
        return record


def can_diagnose(principal: dict) -> bool:
    return principal["role"] in {"reader", "engineer", "approver"}


def can_request_apply(principal: dict) -> bool:
    return principal["role"] == "engineer"


def can_decide(principal: dict, approval: dict) -> tuple[bool, str]:
    if principal["tenant"] != approval["tenant"]:
        return False, "tenant boundary"
    if principal["role"] != "approver":
        return False, "role"
    if principal["subject"] == approval["requester"]:
        return False, "same subject"
    return True, "allowed"
