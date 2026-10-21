"""Time the local investigation path.

Usage: python benchmarks/run_latency_eval.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmarks.analyze_results import percentile, render_markdown
from intelliforge.evaluation.tracking import write_run
from intelliforge.governance.auth import PlatformStore
from intelliforge.investigate import investigate
from intelliforge.lora.trainer import load_adapter
from intelliforge.serving.cost import GROK_INPUT_USD_PER_1M, GROK_OUTPUT_USD_PER_1M, GROK_PRICE_SOURCE, LOCAL_MOCK_USD_PER_1K, PRICE_AS_OF
from intelliforge.serving.quantize import adapter_reconstruction_error
import numpy as np


def machine_spec() -> str:
    processor = platform.processor() or platform.machine()
    return (
        f"{platform.system()} {platform.release()} {platform.machine()} "
        f"processor={processor} cpu_count={os.cpu_count()} python={platform.python_version()}"
    )


def run(iterations: int = 100) -> dict:
    adapter = load_adapter()
    principal = {"tenant": "tenant-a", "role": "engineer", "subject": "engineer-a", "token": "dev-engineer-a"}
    latencies = []
    token_counts = []
    costs = []
    started = time.perf_counter()
    for _ in range(iterations):
        result = investigate(principal, "SYN-CASE-001", "quality", "diagnose", PlatformStore(), adapter)
        latencies.append(float(result["latency_ms"]))
        total_tokens = int(result["prompt_tokens"]) + int(result["completion_tokens"])
        token_counts.append(total_tokens)
        costs.append(float(result["estimated_cost"]["usd"]))
    wall_s = time.perf_counter() - started
    tokens = sum(token_counts)
    quant = adapter_reconstruction_error(
        np.asarray(adapter["A"], dtype=np.float64),
        np.asarray(adapter["B"], dtype=np.float64),
    )
    return {
        "iterations": iterations,
        "p95_ms": percentile(latencies, 95),
        "mean_ms": round(sum(latencies) / len(latencies), 6),
        "wall_clock_s": round(wall_s, 6),
        "tokens_per_second": round(tokens / wall_s, 6) if wall_s else 0.0,
        "tokenizer": "whitespace",
        "mean_usd": round(sum(costs) / len(costs), 8),
        "quantization": quant,
        "sample_latency_ms": [round(value, 6) for value in latencies[:10]],
    }


def write_summary(quality: dict, latency: dict, destination: Path) -> dict:
    metrics = quality["lora_metrics"]
    report = {
        "deployment_decision": "NO-GO",
        "real_equipment": 0,
        "machine": machine_spec(),
        "lora": {
            "lora_heldout_accuracy": metrics["lora_heldout_accuracy"],
            "baseline_heldout_accuracy": metrics["baseline_heldout_accuracy"],
            "lora_prototype_accuracy": metrics["lora_prototype_accuracy"],
            "baseline_prototype_accuracy": metrics["baseline_prototype_accuracy"],
            "train_loss_start": metrics["train_loss_start"],
            "train_loss_end": metrics["train_loss_end"],
        },
        "quality": {
            "cases": quality["cases"],
            "lora_task_completion_rate": quality["lora_task_completion_rate"],
            "baseline_task_completion_rate": quality["baseline_task_completion_rate"],
            "lora_fault_accuracy": quality["lora_fault_accuracy"],
            "baseline_fault_accuracy": quality["baseline_fault_accuracy"],
        },
        "latency": {
            "iterations": latency["iterations"],
            "p95_ms": latency["p95_ms"],
            "tokens_per_second": latency["tokens_per_second"],
        },
        "quantization": {
            "max_abs_error": round(latency["quantization"]["max_abs_error"], 8),
        },
        "cost": {
            "mean_usd": latency["mean_usd"],
            "usd_per_1k": LOCAL_MOCK_USD_PER_1K,
            "as_of": PRICE_AS_OF,
        },
        "commercial_list_price": {
            "model": "grok-4.7",
            "input_usd_per_1m": GROK_INPUT_USD_PER_1M,
            "output_usd_per_1m": GROK_OUTPUT_USD_PER_1M,
            "as_of": PRICE_AS_OF,
            "source": GROK_PRICE_SOURCE,
            "calls": 0,
        },
        "gates": {
            "vllm-qwen": "not-run",
            "peft-hf-train": "not-run",
            "bitsandbytes": "not-run",
            "gpu-memory": "not-run",
            "neo4j": "not-run",
            "pgvector": "not-run",
            "hosted-commercial-model": "not-run",
            "mlflow-client": "not-run",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "latency_report.json").write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8", newline="\n")
    (destination / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (destination / "benchmark_report.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    write_run(
        destination / "tracking_run.json",
        {"route": "lora-local", "iterations": latency["iterations"]},
        {"p95_ms": latency["p95_ms"], "lora_task_completion_rate": quality["lora_task_completion_rate"]},
    )
    return report


def main() -> None:
    from benchmarks.run_quality_eval import run as run_quality

    destination = ROOT / "benchmarks" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    quality = run_quality()
    (destination / "quality_report.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8", newline="\n")
    latency = run()
    report = write_summary(quality, latency, destination)
    print(json.dumps({"p95_ms": report["latency"]["p95_ms"], "deployment_decision": report["deployment_decision"]}))


if __name__ == "__main__":
    main()
