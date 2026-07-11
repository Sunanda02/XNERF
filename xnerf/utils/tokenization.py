from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_token_id(value: Any, vocab_size: int = 4096, offset: int = 1) -> int:
    """Map a string/object to a stable non-zero token id."""

    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    digest = hashlib.blake2b(value.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (vocab_size - offset) + offset


def tokens_to_ids(values: list[Any], vocab_size: int = 4096, max_len: int = 256, prefix: str = "") -> list[int]:
    return [stable_token_id(f"{prefix}:{v}", vocab_size=vocab_size) for v in values[:max_len]]

