"""The committed report matches the renderer and the qualification gates."""

import json
from pathlib import Path

from benchmarks.analyze_results import render_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_committed_benchmark_matches_the_renderer():
    payload = json.loads((ROOT / "benchmarks" / "results" / "benchmark_report.json").read_text(encoding="utf-8"))
    text = (ROOT / "benchmarks" / "results" / "benchmark_report.md").read_text(encoding="utf-8")
    assert payload["deployment_decision"] == "NO-GO"
    assert payload["real_equipment"] == 0
    assert payload["quality"]["lora_task_completion_rate"] == 1
    assert payload["quality"]["baseline_task_completion_rate"] < 1
    assert payload["lora"]["lora_prototype_accuracy"] == 1
    assert payload["gates"]["vllm-qwen"] == "not-run"
    assert payload["gates"]["gpu-memory"] == "not-run"
    assert payload["gates"]["hosted-commercial-model"] == "not-run"
    assert payload["commercial_list_price"]["calls"] == 0
    assert text == render_markdown(payload)
