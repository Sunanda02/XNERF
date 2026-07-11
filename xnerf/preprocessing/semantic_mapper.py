from __future__ import annotations

from dataclasses import asdict

from xnerf.preprocessing.disassembler import Instruction
from xnerf.preprocessing.ontology import MNEMONIC_TO_SEMANTIC, semantic_id
from xnerf.utils.base import Processor


class SemanticMapperProcessor(Processor):
    """Map architecture-specific instructions to semantic ontology classes.

    Inputs:
        list[Instruction]
    Outputs:
        list[dict] with address, mnemonic, semantic, semantic_id
    Tensor dimensions:
        none; symbolic sequence consumed by ISRBuilderProcessor.
    Usage:
        mapped = SemanticMapperProcessor().process(instructions)
    """

    def __init__(self, mnemonic_map: dict[str, str] | None = None):
        self.mnemonic_map = mnemonic_map or MNEMONIC_TO_SEMANTIC

    def process(self, item: list[Instruction]) -> list[dict]:
        rows = []
        for ins in item:
            label = self.mnemonic_map.get(ins.mnemonic, "UNKNOWN")
            row = asdict(ins)
            row["semantic"] = label
            row["semantic_id"] = semantic_id(label)
            rows.append(row)
        return rows

