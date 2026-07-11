from __future__ import annotations

SEMANTIC_CLASSES = [
    "PAD",
    "UNKNOWN",
    "DATA_TRANSFER",
    "ARITHMETIC",
    "LOGIC",
    "CONTROL_FLOW",
    "CALL",
    "RETURN",
    "STACK",
    "MEMORY_LOAD",
    "MEMORY_STORE",
    "CRYPTO_HINT",
    "SYSTEM",
    "PRIVILEGE",
    "NETWORK",
    "ANTI_ANALYSIS",
]

TOKEN_TO_ID = {token: idx for idx, token in enumerate(SEMANTIC_CLASSES)}

ARCH_TO_ID = {
    "unknown": 0,
    "x86": 1,
    "x64": 2,
    "arm": 3,
    "arm64": 4,
    "mips": 5,
    "riscv": 6,
}

MNEMONIC_TO_SEMANTIC = {
    "mov": "DATA_TRANSFER", "movzx": "DATA_TRANSFER", "movsx": "DATA_TRANSFER", "lea": "DATA_TRANSFER",
    "ldr": "MEMORY_LOAD", "ldp": "MEMORY_LOAD", "lw": "MEMORY_LOAD", "ld": "MEMORY_LOAD", "lb": "MEMORY_LOAD",
    "str": "MEMORY_STORE", "stp": "MEMORY_STORE", "sw": "MEMORY_STORE", "sd": "MEMORY_STORE", "sb": "MEMORY_STORE",
    "push": "STACK", "pop": "STACK", "enter": "STACK", "leave": "STACK", "stm": "STACK", "ldm": "STACK",
    "add": "ARITHMETIC", "sub": "ARITHMETIC", "mul": "ARITHMETIC", "imul": "ARITHMETIC", "div": "ARITHMETIC",
    "and": "LOGIC", "or": "LOGIC", "xor": "LOGIC", "not": "LOGIC", "shl": "LOGIC", "shr": "LOGIC", "ror": "LOGIC",
    "jmp": "CONTROL_FLOW", "jz": "CONTROL_FLOW", "jnz": "CONTROL_FLOW", "je": "CONTROL_FLOW", "jne": "CONTROL_FLOW",
    "b": "CONTROL_FLOW", "beq": "CONTROL_FLOW", "bne": "CONTROL_FLOW", "cbz": "CONTROL_FLOW",
    "call": "CALL", "bl": "CALL", "jal": "CALL", "jalr": "CALL",
    "ret": "RETURN", "retn": "RETURN", "bx": "RETURN", "jr": "RETURN",
    "syscall": "SYSTEM", "svc": "SYSTEM", "int": "SYSTEM", "ecall": "SYSTEM",
    "aesenc": "CRYPTO_HINT", "aesdec": "CRYPTO_HINT", "sha1rnds4": "CRYPTO_HINT", "crc32": "CRYPTO_HINT",
    "rdtsc": "ANTI_ANALYSIS", "cpuid": "ANTI_ANALYSIS", "isb": "ANTI_ANALYSIS",
}


def semantic_id(label: str) -> int:
    return TOKEN_TO_ID.get(label, TOKEN_TO_ID["UNKNOWN"])
