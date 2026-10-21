"""Render the committed benchmark. Figures stay tied to the run that produced them."""

from __future__ import annotations

import math


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return round(ordered[index], 6)


def render_markdown(report: dict) -> str:
    quality = report["quality"]
    latency = report["latency"]
    lora = report["lora"]
    quant = report["quantization"]
    cost = report["cost"]
    return f"""# IntelliForge benchmark report

This file records one workstation run of the local mock. It is not a vLLM serving result and it is not a Qwen 2.5 3B measurement.

| Item | Value | Class |
| --- | --- | --- |
| Deployment decision | NO-GO | GATE |
| Real equipment records | {report['real_equipment']} | SPEC |
| Machine | {report['machine']} | SPEC |
| LoRA held-out accuracy | {lora['lora_heldout_accuracy']} | MEASURED NumPy trainer |
| Baseline held-out accuracy | {lora['baseline_heldout_accuracy']} | MEASURED unadapted head |
| LoRA prototype accuracy | {lora['lora_prototype_accuracy']} | MEASURED |
| Baseline prototype accuracy | {lora['baseline_prototype_accuracy']} | MEASURED |
| Train loss start / end | {lora['train_loss_start']} / {lora['train_loss_end']} | MEASURED |
| Quality cases | {quality['cases']} | MEASURED |
| LoRA-route task completion | {quality['lora_task_completion_rate']} | MEASURED |
| Baseline-route task completion | {quality['baseline_task_completion_rate']} | MEASURED |
| LoRA-route fault accuracy | {quality['lora_fault_accuracy']} | MEASURED |
| Baseline-route fault accuracy | {quality['baseline_fault_accuracy']} | MEASURED |
| Local latency iterations | {latency['iterations']} | MEASURED |
| Local latency P95 | {latency['p95_ms']} ms | MEASURED |
| Whitespace tokens per second | {latency['tokens_per_second']} | MEASURED on the mock answers |
| Int8 max absolute error | {quant['max_abs_error']} | MEASURED on the LoRA matrices |
| Derived local cost, mean USD | {cost['mean_usd']} | DERIVED |
| Local price assumption | {cost['usd_per_1k']} USD per 1,000 whitespace tokens, as of {cost['as_of']} | ASSUMPTION |
| grok-4.7 list price, prompts under 200k tokens | input {report['commercial_list_price']['input_usd_per_1m']} USD and output {report['commercial_list_price']['output_usd_per_1m']} USD per 1,000,000 tokens | SPEC from the vendor page on {report['commercial_list_price']['as_of']} |
| Qwen / vLLM | not-run | GATE |
| Hugging Face PEFT | not-run | GATE |
| bitsandbytes | not-run | GATE |
| GPU memory | not-run | GATE |
| Neo4j | not-run | GATE |
| pgvector | not-run | GATE |
| Hosted commercial call | not-run | GATE |
| MLflow client | not-run | GATE |

The LoRA numbers come from `training/lora_train.py`. The task scores come from the synthetic cases under `datasets/synthetic_equipment/`. Latency is the in-process investigation, including retrieval, the local head, and the SciPy tool.
"""
