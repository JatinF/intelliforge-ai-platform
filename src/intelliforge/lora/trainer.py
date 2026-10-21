"""NumPy LoRA on a linear head.

delta_w = (alpha / rank) * B @ A, with B initialized at zero.
Hugging Face PEFT is not imported. The gate peft-hf-train stays not-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from intelliforge.paths import adapter_path

CLASSES = ("normal", "cooling_flow_low", "pressure_leak")
SEED = 7
RANK = 2
ALPHA = 4.0
STEPS = 400
LEARNING_RATE = 0.8


def featurize(readings: dict) -> np.ndarray:
    """Synthetic sensor encoder. Axis 0, 1, or 2 is the class signal."""
    temp_delta = float(readings["temperature_c"]) - float(readings["target_temperature_c"])
    pressure_delta = float(readings["pressure_kpa"]) - float(readings["target_pressure_kpa"])
    features = np.zeros(4, dtype=np.float64)
    if pressure_delta <= -5.0:
        features[2] = 1.0
    elif temp_delta >= 10.0:
        features[1] = 1.0
    else:
        features[0] = 1.0
    features[3] = 0.05
    return features


def _split(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for class_index in range(len(CLASSES)):
        for _ in range(n_per_class):
            row = rng.normal(0.0, 0.01, size=4)
            row[class_index] += 1.0
            row[3] = 0.05 + float(rng.normal(0.0, 0.001))
            rows.append(row)
            labels.append(class_index)
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _accuracy(features: np.ndarray, labels: np.ndarray, matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    predicted = predict_matrix(features, matrix_a, matrix_b)
    return float(np.mean(predicted == labels))


def predict_matrix(features: np.ndarray, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    scale = ALPHA / RANK
    delta = scale * (matrix_b @ matrix_a)
    logits = np.asarray(features, dtype=np.float64) @ delta.T
    if logits.ndim == 1:
        return np.asarray([int(np.argmax(logits))])
    return np.argmax(logits, axis=1).astype(np.int64)


def predict_label(readings: dict, adapter: dict, route_id: str) -> str:
    if route_id == "baseline-local":
        return CLASSES[0]
    features = featurize(readings)
    matrix_a = np.asarray(adapter["A"], dtype=np.float64)
    matrix_b = np.asarray(adapter["B"], dtype=np.float64)
    index = int(predict_matrix(features, matrix_a, matrix_b)[0])
    return CLASSES[index]


def train_and_compare(
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict:
    features, labels = _split(30, seed)
    held_features, held_labels = _split(10, seed + 1)
    rng = np.random.default_rng(seed)
    matrix_a = rng.normal(0.0, 0.02, size=(RANK, 4))
    matrix_b = np.zeros((len(CLASSES), RANK), dtype=np.float64)
    scale = ALPHA / RANK
    losses = []
    for _ in range(steps):
        delta = scale * (matrix_b @ matrix_a)
        logits = features @ delta.T
        probs = _softmax(logits)
        picked = probs[np.arange(len(labels)), labels]
        losses.append(float(-np.mean(np.log(picked + 1e-12))))
        grad_logits = probs.copy()
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        grad_logits /= len(labels)
        grad_delta = grad_logits.T @ features
        grad_b = scale * grad_delta @ matrix_a.T
        grad_a = scale * matrix_b.T @ grad_delta
        matrix_a -= learning_rate * grad_a
        matrix_b -= learning_rate * grad_b
    zeros_a = np.zeros_like(matrix_a)
    zeros_b = np.zeros_like(matrix_b)
    baseline_held = _accuracy(held_features, held_labels, zeros_a, zeros_b)
    adapted_held = _accuracy(held_features, held_labels, matrix_a, matrix_b)
    prototypes = []
    prototype_labels = []
    for class_index, _name in enumerate(CLASSES):
        row = np.zeros(4, dtype=np.float64)
        row[class_index] = 1.0
        row[3] = 0.05
        prototypes.append(row)
        prototype_labels.append(class_index)
    prototype_matrix = np.vstack(prototypes)
    prototype_y = np.asarray(prototype_labels)
    return {
        "format": "numpy-lora",
        "seed": seed,
        "rank": RANK,
        "alpha": ALPHA,
        "steps": steps,
        "learning_rate": learning_rate,
        "classes": list(CLASSES),
        "A": matrix_a.round(8).tolist(),
        "B": matrix_b.round(8).tolist(),
        "metrics": {
            "train_loss_start": round(losses[0], 6),
            "train_loss_end": round(losses[-1], 6),
            "baseline_heldout_accuracy": round(baseline_held, 6),
            "lora_heldout_accuracy": round(adapted_held, 6),
            "baseline_prototype_accuracy": round(_accuracy(prototype_matrix, prototype_y, zeros_a, zeros_b), 6),
            "lora_prototype_accuracy": round(_accuracy(prototype_matrix, prototype_y, matrix_a, matrix_b), 6),
            "heldout_examples": int(len(held_labels)),
            "train_examples": int(len(labels)),
            "peft_hf_train": "not-run",
        },
    }


def save_adapter(report: dict, path: Path | None = None) -> Path:
    destination = path or adapter_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return destination


def load_adapter(path: Path | None = None) -> dict:
    source = path or adapter_path()
    return json.loads(source.read_text(encoding="utf-8"))
