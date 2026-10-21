"""The measured NumPy LoRA head and the int8 round-trip."""

import numpy as np

from serving.local_lora import train_and_compare
from serving.quantize import adapter_reconstruction_error
from training.compare_adapters import compare


def test_lora_heldout_beats_the_zero_head():
    report = train_and_compare()
    metrics = report["metrics"]
    assert metrics["baseline_heldout_accuracy"] < metrics["lora_heldout_accuracy"]
    assert metrics["lora_prototype_accuracy"] == 1.0
    assert metrics["peft_hf_train"] == "not-run"
    delta = compare(report)
    assert delta["accuracy_delta"] > 0


def test_int8_round_trip_stays_bounded():
    report = train_and_compare()
    error = adapter_reconstruction_error(np.asarray(report["A"]), np.asarray(report["B"]))
    assert error["bitsandbytes"] == "not-run"
    assert 0.0 <= error["max_abs_error"] < 0.05
