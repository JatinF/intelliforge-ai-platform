"""Run one local quality workflow on the main gateway and write its result."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from gateway.app import create_app  # noqa: E402

PROMPT = "SYN-CASE-001\nWhy is SYN-PROC-ETCH-01 producing abnormal chamber temperature readings?"


def main() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-engineer-a"}
    health = client.get("/healthz")
    completed = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "numpy-lora-r2-seed7",
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
        },
    )
    if completed.status_code != 200:
        raise RuntimeError(completed.text)
    body = completed.json()
    meta = body["intelliforge"]
    report = {
        "workflow": "quality-chat-completions",
        "deployment_decision": "NO-GO",
        "real_equipment": health.json()["real_equipment"],
        "mode": health.json()["mode"],
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "model": "numpy-lora-r2-seed7",
            "case_id": "SYN-CASE-001",
        },
        "status_code": completed.status_code,
        "status": meta["status"],
        "fault_class": meta["fault_class"],
        "model_version": meta["model_version"],
        "setpoint_applied": meta["setpoint_applied"],
        "remote_called": meta["remote_called"],
        "answer": body["choices"][0]["message"]["content"],
        "usage": body["usage"],
        "gates": {
            "vllm-qwen": "not-run",
            "hosted-commercial-model": "not-run",
            "qwen-weight-load": "not-run",
        },
    }
    destination = ROOT / "benchmarks" / "results" / "workflow_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "fault_class": report["fault_class"], "setpoint_applied": report["setpoint_applied"]}))


if __name__ == "__main__":
    main()
