from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from xnerf.utils.base import Processor

try:
    from capstone import (
        Cs,
        CS_ARCH_ARM,
        CS_ARCH_ARM64,
        CS_ARCH_MIPS,
        CS_ARCH_RISCV,
        CS_ARCH_X86,
        CS_MODE_32,
        CS_MODE_64,
        CS_MODE_ARM,
        CS_MODE_LITTLE_ENDIAN,
        CS_MODE_MIPS32,
        CS_MODE_RISCV64,
    )
except Exception:  # pragma: no cover - import guard for docs-only envs
    Cs = None


@dataclass
class Instruction:
    address: int
    mnemonic: str
    op_str: str
    size: int


class DisassemblerProcessor(Processor):
    """Disassemble raw bytes into normalized instruction records.

    Inputs:
        item: bytes or {"bytes": bytes, "arch": str, "base": int}
    Outputs:
        list[Instruction]
    Tensor dimensions:
        none, this processor emits symbolic records.
    Usage:
        p = DisassemblerProcessor("x86")
        ins = p.process(b"\\x89\\xd8")
    """

    def __init__(self, arch: str = "unknown", max_instructions: int = 4096):
        self.arch = arch.lower()
        self.max_instructions = max_instructions

    def _capstone_mode(self, arch: str):
        if Cs is None:
            raise RuntimeError("capstone is not installed. Install requirements.txt first.")
        mapping = {
            "x86": (CS_ARCH_X86, CS_MODE_32),
            "x64": (CS_ARCH_X86, CS_MODE_64),
            "arm": (CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
            "arm64": (CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN),
            "mips": (CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_LITTLE_ENDIAN),
            "riscv": (CS_ARCH_RISCV, CS_MODE_RISCV64 + CS_MODE_LITTLE_ENDIAN),
        }
        if arch not in mapping:
            raise ValueError(f"Unsupported architecture: {arch}")
        return mapping[arch]

    def process(self, item) -> list[Instruction]:
        arch = self.arch
        base = 0
        blob = item
        if isinstance(item, dict):
            blob = item["bytes"]
            arch = item.get("arch", arch).lower()
            base = int(item.get("base", 0))
        cs_arch, cs_mode = self._capstone_mode(arch)
        md = Cs(cs_arch, cs_mode)
        md.detail = False
        records = []
        for idx, ins in enumerate(md.disasm(blob, base)):
            if idx >= self.max_instructions:
                break
            records.append(Instruction(ins.address, ins.mnemonic.lower(), ins.op_str.lower(), ins.size))
        return records
