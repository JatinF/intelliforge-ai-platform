"""Token accounting and estimated cost.

Whitespace token counts are measured on the strings this process built.
Prices are assumptions or a vendor list price. Neither is an invoice.
"""

from __future__ import annotations

PRICE_AS_OF = "2026-10-21"
LOCAL_MOCK_USD_PER_1K = 0.00015

# Vendor list price for grok-4.7 prompts under 200k tokens, copied from
# https://docs.x.ai/docs/models on 2026-10-21. The commercial route does not
# call that API in the default path.
GROK_MODEL = "grok-4.7"
GROK_INPUT_USD_PER_1M = 2.0
GROK_OUTPUT_USD_PER_1M = 6.0
GROK_PRICE_SOURCE = "https://docs.x.ai/docs/models"


def whitespace_tokens(text: str) -> int:
    parts = [part for part in text.split() if part]
    return len(parts)


def estimate_local_cost(prompt_tokens: int, completion_tokens: int) -> dict:
    total = int(prompt_tokens) + int(completion_tokens)
    usd = total / 1000.0 * LOCAL_MOCK_USD_PER_1K
    return {
        "usd": round(usd, 8),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": total,
        "tokenizer": "whitespace",
        "evidence_class": "DERIVED",
        "price": {
            "evidence_class": "ASSUMPTION",
            "as_of": PRICE_AS_OF,
            "usd_per_1k": LOCAL_MOCK_USD_PER_1K,
            "note": "Local mock price assumption. Not an invoice and not a GPU quote.",
        },
    }
