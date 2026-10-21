"""Decide whether this process may load Qwen2.5-3B-Instruct.

The parameter count and storage size are copied from the Hugging Face model
API. The extra resident headroom is an assumption. The available-byte reading
is measured on the machine that calls read_available_bytes.
"""

from __future__ import annotations

import ctypes
import sys

from intelliforge.serving.remote import QWEN_MODEL

# https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct on 2026-10-21.
# safetensors.parameters.BF16 and usedStorage.
QWEN_PARAMETER_COUNT = 3_085_938_688
QWEN_PARAMETER_DTYPE = "BF16"
QWEN_PARAMETER_BYTES = QWEN_PARAMETER_COUNT * 2
QWEN_REPO_USED_STORAGE_BYTES = 6_171_926_992
QWEN_CARD_SOURCE = "https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct"
QWEN_CARD_AS_OF = "2026-10-21"

# Room for activations, the allocator, and the process beside the BF16 weights.
RESIDENT_HEADROOM_BYTES = 2 * 1024**3
REQUIRED_AVAILABLE_BYTES = QWEN_PARAMETER_BYTES + RESIDENT_HEADROOM_BYTES


def read_available_bytes() -> int:
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError("MemAvailable is missing from /proc/meminfo")


def assess_qwen_load(available_bytes: int) -> dict:
    available = int(available_bytes)
    allowed = available >= REQUIRED_AVAILABLE_BYTES
    if allowed:
        reason = "available memory covers the BF16 weights plus the resident headroom"
    else:
        reason = "available memory is below the BF16 weights plus the resident headroom"
    return {
        "model": QWEN_MODEL,
        "parameter_count": QWEN_PARAMETER_COUNT,
        "parameter_dtype": QWEN_PARAMETER_DTYPE,
        "parameter_bytes": QWEN_PARAMETER_BYTES,
        "repo_used_storage_bytes": QWEN_REPO_USED_STORAGE_BYTES,
        "card_source": QWEN_CARD_SOURCE,
        "card_as_of": QWEN_CARD_AS_OF,
        "card_evidence_class": "SPEC",
        "resident_headroom_bytes": RESIDENT_HEADROOM_BYTES,
        "headroom_evidence_class": "ASSUMPTION",
        "required_available_bytes": REQUIRED_AVAILABLE_BYTES,
        "available_bytes": available,
        "available_evidence_class": "MEASURED",
        "decision": "allow" if allowed else "refuse",
        "qwen_weight_load": "not-run",
        "reason": reason,
    }
