"""Train a Hugging Face PEFT LoRA on Qwen/Qwen2.5-3B-Instruct.

Usage: python training/peft_lora.py

The process exits 2 when the memory preflight refuses the weight load.
torch, transformers, and peft are imported only after that check passes.
Install them with: python -m pip install -e ".[openweight]"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from knowledge.store import load_cases
from serving.preflight import assess_qwen_load, read_available_bytes
from serving.remote import QWEN_MODEL

OUTPUT = ROOT / "training" / "artifacts" / "qwen25_3b_lora"
STEPS = 20
RANK = 8
ALPHA = 16


def training_rows() -> list[str]:
    rows = []
    for case in load_cases():
        if case["suite"] != "qualification":
            continue
        readings = case["readings"]
        rows.append(
            "Instruction: "
            + case["question"]
            + "\n"
            + f"temperature_c={readings['temperature_c']} target_temperature_c={readings['target_temperature_c']} "
            + f"pressure_kpa={readings['pressure_kpa']} target_pressure_kpa={readings['target_pressure_kpa']}\n"
            + f"Answer: fault class is {case['expected_fault']}.\n"
        )
    return rows


def train() -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = training_rows()
    if not rows:
        raise RuntimeError("qualification split is empty")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    losses = []
    model.train()
    for step in range(STEPS):
        text = rows[step % len(rows)]
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        labels = batch["input_ids"].clone()
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(round(float(loss.detach()), 6))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    return {
        "model": QWEN_MODEL,
        "steps": STEPS,
        "rank": RANK,
        "alpha": ALPHA,
        "examples": len(rows),
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "output": str(OUTPUT),
        "peft_hf_train": "measured",
    }


def main() -> None:
    decision = assess_qwen_load(read_available_bytes())
    print(json.dumps(decision))
    if decision["decision"] != "allow":
        raise SystemExit(2)
    print(json.dumps(train()))


if __name__ == "__main__":
    main()
