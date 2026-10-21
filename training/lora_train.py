"""Train the local LoRA head and write the adapter artifact.

Usage: python training/lora_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from serving.local_lora import save_adapter, train_and_compare


def main() -> None:
    report = train_and_compare()
    path = save_adapter(report)
    print(json.dumps({"path": str(path), "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
