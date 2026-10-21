"""Int8 round-trip for small local matrices.

bitsandbytes is the GPU quantization slot. It is not imported here.
"""

from __future__ import annotations

import numpy as np


def quantize_int8(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(matrix, dtype=np.float64)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
    quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return quantized, scale


def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(quantized, dtype=np.float64) * float(scale)


def reconstruction_error(matrix: np.ndarray) -> dict:
    quantized, scale = quantize_int8(matrix)
    restored = dequantize_int8(quantized, scale)
    delta = np.abs(restored - np.asarray(matrix, dtype=np.float64))
    return {
        "method": "numpy-int8",
        "max_abs_error": float(np.max(delta) if delta.size else 0.0),
        "scale": scale,
        "bitsandbytes": "not-run",
    }


def adapter_reconstruction_error(matrix_a: np.ndarray, matrix_b: np.ndarray) -> dict:
    first = reconstruction_error(matrix_a)
    second = reconstruction_error(matrix_b)
    return {
        "method": "numpy-int8",
        "max_abs_error": max(first["max_abs_error"], second["max_abs_error"]),
        "bitsandbytes": "not-run",
    }
