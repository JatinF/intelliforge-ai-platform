"""Compare the unadapted head with the NumPy LoRA head.

Usage: python training/compare_adapters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from serving.local_lora import train_and_compare


def compare(report: dict | None = None) -> dict:
    trained = report or train_and_compare()
    metrics = trained["metrics"]
    return {
        "baseline_heldout_accuracy": metrics["baseline_heldout_accuracy"],
        "lora_heldout_accuracy": metrics["lora_heldout_accuracy"],
        "accuracy_delta": round(metrics["lora_heldout_accuracy"] - metrics["baseline_heldout_accuracy"], 6),
        "baseline_prototype_accuracy": metrics["baseline_prototype_accuracy"],
        "lora_prototype_accuracy": metrics["lora_prototype_accuracy"],
        "peft_hf_train": "not-run",
    }


def main() -> None:
    print(json.dumps(compare()))


if __name__ == "__main__":
    main()
