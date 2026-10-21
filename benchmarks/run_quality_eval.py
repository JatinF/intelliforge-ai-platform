"""Score the synthetic cases on the LoRA route and the baseline route.

Usage: python benchmarks/run_quality_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.score import score_case, summarize
from governance.auth import PlatformStore
from gateway.investigate import investigate
from knowledge.store import load_cases
from serving.local_lora import save_adapter, train_and_compare


def run() -> dict:
    adapter = train_and_compare()
    save_adapter(adapter)
    principal = {"tenant": "tenant-a", "role": "engineer", "subject": "engineer-a", "token": "dev-engineer-a"}
    cases = [case for case in load_cases() if case["suite"] == "qualification"]
    scored = {}
    for policy, name in (("quality", "lora"), ("cost", "baseline")):
        rows = []
        for case in cases:
            result = investigate(principal, case["case_id"], policy, "diagnose", PlatformStore(), adapter)
            rows.append(score_case(result, case))
        summary = summarize(rows)
        summary["rows"] = rows
        scored[name] = summary
    return {
        "cases": len(cases),
        "lora_task_completion_rate": scored["lora"]["task_completion_rate"],
        "baseline_task_completion_rate": scored["baseline"]["task_completion_rate"],
        "lora_fault_accuracy": scored["lora"]["fault_accuracy"],
        "baseline_fault_accuracy": scored["baseline"]["fault_accuracy"],
        "lora_rows": scored["lora"]["rows"],
        "baseline_rows": scored["baseline"]["rows"],
        "lora_metrics": adapter["metrics"],
    }


def main() -> None:
    destination = ROOT / "benchmarks" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    report = run()
    (destination / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("cases", "lora_task_completion_rate", "baseline_task_completion_rate")}))


if __name__ == "__main__":
    main()
