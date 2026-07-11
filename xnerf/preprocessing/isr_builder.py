from __future__ import annotations

import torch

from xnerf.preprocessing.ontology import ARCH_TO_ID, TOKEN_TO_ID
from xnerf.utils.base import Processor


class ISRBuilderProcessor(Processor):
    """Build an Intermediate Semantic Representation tensor.

    Inputs:
        {"semantic": list[dict], "arch": str} or list[dict]
    Outputs:
        torch.LongTensor [max_len, 4]
        Columns: semantic_id, arch_id, address_delta_bucket, size
    Tensor dimensions:
        [T, 4] where T=max_len. PAD rows use semantic_id=0.
    Usage:
        isr = ISRBuilderProcessor(max_len=1024).process({"semantic": mapped, "arch": "x64"})
    """

    def __init__(self, max_len: int = 1024, max_delta_bucket: int = 255):
        self.max_len = max_len
        self.max_delta_bucket = max_delta_bucket

    def process(self, item) -> torch.Tensor:
        arch = "unknown"
        rows = item
        if isinstance(item, dict):
            rows = item["semantic"]
            arch = item.get("arch", arch).lower()
        arch_id = ARCH_TO_ID.get(arch, ARCH_TO_ID["unknown"])
        out = torch.zeros(self.max_len, 4, dtype=torch.long)
        prev_addr = rows[0]["address"] if rows else 0
        for i, row in enumerate(rows[: self.max_len]):
            delta = max(0, int(row["address"]) - prev_addr)
            prev_addr = int(row["address"])
            out[i, 0] = int(row.get("semantic_id", TOKEN_TO_ID["UNKNOWN"]))
            out[i, 1] = arch_id
            out[i, 2] = min(delta, self.max_delta_bucket)
            out[i, 3] = int(row.get("size", 0))
        return out
