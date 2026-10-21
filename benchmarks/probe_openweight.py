"""Record whether this workstation can load Qwen2.5-3B and start vLLM.

Usage: python benchmarks/probe_openweight.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from benchmarks.analyze_results import percentile
from serving.openai_compat import create_openai_app
from serving.preflight import assess_qwen_load, read_available_bytes
from serving.remote import QWEN_MODEL, chat_completion


def machine_spec() -> str:
    processor = platform.processor() or platform.machine()
    return (
        f"{platform.system()} {platform.release()} {platform.machine()} "
        f"processor={processor} cpu_count={os.cpu_count()} python={platform.python_version()}"
    )


def command_text(program: str, args: list[str]) -> str:
    if shutil.which(program) is None:
        return "absent"
    try:
        raw = subprocess.check_output([program, *args], timeout=30, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raw = exc.output or b""
    except subprocess.TimeoutExpired:
        return "timeout"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff") or b"\x00" in raw[:8]:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return " ".join(text.split())


def module_status(name: str) -> str:
    return "installed" if importlib.util.find_spec(name) else "not-installed"


def time_local_models(available_bytes: int) -> dict:
    app = create_openai_app(available_bytes=available_bytes)
    headers = {"Authorization": "Bearer dev-engineer-a"}
    prompt = "Why is SYN-PROC-ETCH-01 producing abnormal chamber temperature readings? SYN-CASE-001"
    latencies = []
    with TestClient(app) as client:
        for _ in range(20):
            started = time.perf_counter()
            body = chat_completion("http://testserver", "numpy-lora-r2-seed7", prompt, client=client, headers=headers)
            latencies.append((time.perf_counter() - started) * 1000.0)
        baseline = chat_completion(
            "http://testserver",
            "local-linear-w0-zeros",
            prompt,
            client=client,
            headers=headers,
        )
        qwen = client.post(
            "/v1/chat/completions",
            json={"model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
            headers=headers,
        )
    return {
        "iterations": 20,
        "local_lora_p95_ms": percentile(latencies, 95),
        "local_lora_status": body["intelliforge"]["status"],
        "local_lora_fault_class": body["intelliforge"]["fault_class"],
        "baseline_status": baseline["intelliforge"]["status"],
        "baseline_fault_class": baseline["intelliforge"]["fault_class"],
        "qwen_http_status": qwen.status_code,
        "tokenizer": "whitespace",
        "server": "in-process TestClient",
    }


def main() -> None:
    available = read_available_bytes()
    decision = assess_qwen_load(available)
    replacement = time_local_models(available)
    report = {
        "machine": machine_spec(),
        "preflight": decision,
        "commands": {
            "nvidia-smi": command_text("nvidia-smi", []),
            "docker": command_text("docker", ["version", "--format", "{{.Server.Version}}"]),
            "wsl": command_text("wsl", ["-l", "-v"]),
        },
        "imports": {
            "torch": module_status("torch"),
            "transformers": module_status("transformers"),
            "peft": module_status("peft"),
            "vllm": module_status("vllm"),
        },
        "replacement": replacement,
        "gates": {
            "vllm-qwen": "not-run",
            "peft-hf-train": "not-run",
            "qwen-weight-load": "not-run",
            "gpu-memory": "not-run",
        },
    }
    destination = ROOT / "benchmarks" / "results" / "openweight_probe.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"decision": decision["decision"], "available_bytes": available, "qwen_http_status": replacement["qwen_http_status"]}))


if __name__ == "__main__":
    main()
