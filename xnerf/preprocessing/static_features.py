from __future__ import annotations

import hashlib
import logging
import re
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import torch

from xnerf.preprocessing.disassembler import DisassemblerProcessor, Instruction
from xnerf.preprocessing.ontology import ARCH_TO_ID
from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline
from xnerf.preprocessing.semantic_mapper import SemanticMapperProcessor
from xnerf.utils.tokenization import tokens_to_ids


LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = 256
API_VOCAB_SIZE = 8192
NETWORK_VOCAB_SIZE = 4096
SEQ_LEN = 256
MEMORY_LEN = 512
MEMORY_DIM = 8
ISR_LEN = 1024
CFG_NODE_DIM = 4

PREPROCESSING_CACHE_VERSION = "static_features_v1"

NETWORK_IMPORT_HINTS = (
    "accept",
    "bind",
    "connect",
    "dns",
    "download",
    "ftp",
    "getaddrinfo",
    "gethost",
    "http",
    "https",
    "inet_",
    "internet",
    "irc",
    "recv",
    "resolv",
    "send",
    "socket",
    "tcp",
    "udp",
    "url",
    "websocket",
    "winhttp",
    "wininet",
    "ws2_32",
)


@dataclass
class ExecutableMetadata:
    format: str = "unknown"
    arch: str = "unknown"
    arch_raw: str = "unknown"
    endian: str = "little"
    entrypoint: int | None = None
    image_base: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class ExtractionLog:
    warnings: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.warnings.append(message)
        LOGGER.warning(message)


def canonical_arch(value: str | None) -> str:
    text = str(value or "unknown").strip().lower().replace("-", "")
    aliases = {
        "amd64": "x64",
        "x86_64": "x64",
        "i386": "x86",
        "i486": "x86",
        "i586": "x86",
        "i686": "x86",
        "aarch64": "arm64",
        "armv8": "arm64",
        "armv7": "arm",
        "armv6": "arm",
        "mipsel": "mips",
        "mipsle": "mips",
        "mips32": "mips",
        "mips32el": "mips",
        "riscv32": "riscv",
        "riscv64": "riscv",
    }
    text = aliases.get(text, text)
    return text if text in ARCH_TO_ID else "unknown"


def infer_arch_from_path(path: str | Path) -> str:
    text = str(path).lower()
    for raw in ("arm64", "aarch64", "arm", "mipsel", "mips", "riscv64", "riscv", "x86_64", "x64", "x86"):
        if raw in text:
            return canonical_arch(raw)
    return "unknown"


def binary_image_from_bytes(data: bytes, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    values = np.frombuffer(data[: image_size * image_size], dtype=np.uint8)
    if values.size == 0:
        values = np.zeros(1, dtype=np.uint8)
    values = np.pad(values, (0, max(0, image_size * image_size - values.size)))[: image_size * image_size]
    return torch.from_numpy(values.reshape(1, image_size, image_size).astype("float32") / 255.0)


def binary_image_from_file(path: str | Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    return binary_image_from_bytes(Path(path).read_bytes(), image_size=image_size)


def build_memory_trace(features: Iterable[float], rows: int = MEMORY_LEN, cols: int = MEMORY_DIM) -> torch.Tensor:
    out = torch.zeros(rows * cols, dtype=torch.float32)
    values_list = list(features)
    if values_list:
        values = torch.tensor(values_list[: rows * cols], dtype=torch.float32)
        if values.numel() > 1:
            values = (values - values.mean()) / values.std().clamp_min(1e-6)
        out[: values.numel()] = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return out.view(rows, cols)


def zero_memory_trace(reason: str | None = None, log: ExtractionLog | None = None) -> torch.Tensor:
    if reason and log is not None:
        log.add(reason)
    return torch.zeros(MEMORY_LEN, MEMORY_DIM, dtype=torch.float32)


def _read_c_string(data: bytes, offset: int, limit: int | None = None) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset, len(data) if limit is None else min(len(data), offset + limit))
    if end < 0:
        end = len(data) if limit is None else min(len(data), offset + limit)
    return data[offset:end].decode("utf-8", errors="ignore").strip()


def _detect_pe(data: bytes) -> ExecutableMetadata | None:
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_off + 24 > len(data) or data[pe_off : pe_off + 4] != b"PE\x00\x00":
        return ExecutableMetadata(format="PE", notes=["PE DOS header present but NT header was not readable"])
    machine = struct.unpack_from("<H", data, pe_off + 4)[0]
    arch_map = {
        0x014C: "x86",
        0x8664: "x64",
        0x01C0: "arm",
        0x01C4: "arm",
        0xAA64: "arm64",
        0x0166: "mips",
        0x0266: "mips",
        0x0366: "mips",
        0x0466: "mips",
        0x5032: "riscv",
        0x5064: "riscv",
    }
    opt_off = pe_off + 24
    entrypoint = None
    image_base = 0
    if opt_off + 32 <= len(data):
        magic = struct.unpack_from("<H", data, opt_off)[0]
        entry_rva = struct.unpack_from("<I", data, opt_off + 16)[0]
        entrypoint = entry_rva
        if magic == 0x10B and opt_off + 32 <= len(data):
            image_base = struct.unpack_from("<I", data, opt_off + 28)[0]
        elif magic == 0x20B and opt_off + 32 <= len(data):
            image_base = struct.unpack_from("<Q", data, opt_off + 24)[0]
    arch = arch_map.get(machine, "unknown")
    return ExecutableMetadata(format="PE", arch=arch, arch_raw=f"pe_machine_0x{machine:04x}", entrypoint=entrypoint, image_base=image_base)


def _detect_elf(data: bytes) -> ExecutableMetadata | None:
    if len(data) < 20 or data[:4] != b"\x7fELF":
        return None
    elf_class = data[4]
    endian = "little" if data[5] == 1 else "big" if data[5] == 2 else "little"
    prefix = "<" if endian == "little" else ">"
    machine = struct.unpack_from(prefix + "H", data, 18)[0]
    arch_map = {
        3: "x86",
        8: "mips",
        40: "arm",
        62: "x64",
        183: "arm64",
        243: "riscv",
    }
    entrypoint = None
    try:
        if elf_class == 1 and len(data) >= 28:
            entrypoint = struct.unpack_from(prefix + "I", data, 24)[0]
        elif elf_class == 2 and len(data) >= 32:
            entrypoint = struct.unpack_from(prefix + "Q", data, 24)[0]
    except struct.error:
        entrypoint = None
    raw = "mipsel" if machine == 8 and endian == "little" else f"elf_machine_{machine}"
    return ExecutableMetadata(format="ELF", arch=arch_map.get(machine, "unknown"), arch_raw=raw, endian=endian, entrypoint=entrypoint)


def _detect_apk(path: Path, data: bytes) -> ExecutableMetadata | None:
    if path.suffix.lower() != ".apk" and not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return None
    if "AndroidManifest.xml" not in names and not any(name.endswith(".dex") for name in names):
        return None
    arch = "unknown"
    for name in names:
        lowered = name.lower()
        if "lib/arm64-v8a/" in lowered:
            arch = "arm64"
            break
        if "lib/armeabi" in lowered:
            arch = "arm"
        elif "lib/x86_64/" in lowered:
            arch = "x64"
        elif "lib/x86/" in lowered and arch == "unknown":
            arch = "x86"
        elif "lib/mips" in lowered and arch == "unknown":
            arch = "mips"
    return ExecutableMetadata(format="APK", arch=arch, arch_raw=arch, notes=["APK architecture inferred from native library ABIs"])


def detect_executable(path: str | Path, data: bytes | None = None) -> ExecutableMetadata:
    sample_path = Path(path)
    blob = sample_path.read_bytes() if data is None else data
    detected = _detect_pe(blob) or _detect_elf(blob) or _detect_apk(sample_path, blob)
    if detected is None:
        return ExecutableMetadata(format="unknown", arch=infer_arch_from_path(sample_path), arch_raw="path_hint")
    detected.arch = canonical_arch(detected.arch)
    return detected


def _pe_sections(data: bytes) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    number_sections = struct.unpack_from("<H", data, pe_off + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    opt_off = pe_off + 24
    sec_off = opt_off + opt_size
    sections = []
    for i in range(number_sections):
        off = sec_off + i * 40
        if off + 40 > len(data):
            break
        virtual_size, virtual_address, raw_size, raw_ptr = struct.unpack_from("<IIII", data, off + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_ptr, raw_size))
    return opt_off, opt_size, sections


def _pe_rva_to_offset(rva: int, sections: list[tuple[int, int, int, int]]) -> int | None:
    for va, vsize, raw_ptr, raw_size in sections:
        if va <= rva < va + max(vsize, raw_size):
            offset = raw_ptr + (rva - va)
            return offset if 0 <= offset < raw_ptr + raw_size else None
    return rva


def extract_pe_imports(data: bytes) -> list[str]:
    imports: list[str] = []
    try:
        opt_off, _opt_size, sections = _pe_sections(data)
        magic = struct.unpack_from("<H", data, opt_off)[0]
        data_dir_off = opt_off + (96 if magic == 0x10B else 112 if magic == 0x20B else 0)
        if not data_dir_off or data_dir_off + 8 > len(data):
            return []
        import_rva, _import_size = struct.unpack_from("<II", data, data_dir_off + 8)
        desc_off = _pe_rva_to_offset(import_rva, sections)
        if desc_off is None:
            return []
        thunk_width = 8 if magic == 0x20B else 4
        ordinal_mask = 0x8000000000000000 if thunk_width == 8 else 0x80000000
        for desc_idx in range(512):
            off = desc_off + desc_idx * 20
            if off + 20 > len(data):
                break
            original_thunk, _time, _chain, name_rva, first_thunk = struct.unpack_from("<IIIII", data, off)
            if not any((original_thunk, name_rva, first_thunk)):
                break
            dll_off = _pe_rva_to_offset(name_rva, sections)
            dll_name = _read_c_string(data, dll_off or -1).lower()
            thunk_rva = original_thunk or first_thunk
            thunk_off = _pe_rva_to_offset(thunk_rva, sections)
            if thunk_off is None:
                continue
            for thunk_idx in range(2048):
                item_off = thunk_off + thunk_idx * thunk_width
                if item_off + thunk_width > len(data):
                    break
                value = struct.unpack_from("<Q" if thunk_width == 8 else "<I", data, item_off)[0]
                if value == 0:
                    break
                if value & ordinal_mask:
                    imports.append(f"{dll_name}:ordinal_{value & 0xFFFF}")
                    continue
                hint_name_off = _pe_rva_to_offset(int(value), sections)
                if hint_name_off is None or hint_name_off + 2 >= len(data):
                    continue
                name = _read_c_string(data, hint_name_off + 2)
                if name:
                    imports.append(f"{dll_name}:{name}" if dll_name else name)
    except Exception:
        return imports
    return imports


def _elf_header(data: bytes) -> tuple[str, int, int, int, int, int, int] | None:
    if len(data) < 52 or data[:4] != b"\x7fELF":
        return None
    elf_class = data[4]
    prefix = "<" if data[5] == 1 else ">"
    if elf_class == 1:
        e_shoff = struct.unpack_from(prefix + "I", data, 32)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(prefix + "HHH", data, 46)
    elif elf_class == 2:
        e_shoff = struct.unpack_from(prefix + "Q", data, 40)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(prefix + "HHH", data, 58)
    else:
        return None
    return prefix, elf_class, e_shoff, e_shentsize, e_shnum, e_shstrndx, 64 if elf_class == 2 else 52


def extract_elf_imports(data: bytes) -> list[str]:
    imports: list[str] = []
    try:
        header = _elf_header(data)
        if header is None:
            return []
        prefix, elf_class, e_shoff, e_shentsize, e_shnum, e_shstrndx, _ = header
        sections = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            if off + e_shentsize > len(data):
                break
            if elf_class == 1:
                sh = struct.unpack_from(prefix + "IIIIIIIIII", data, off)
                name, sh_type, _flags, _addr, sh_offset, sh_size, sh_link, _info, _align, sh_entsize = sh
            else:
                name, sh_type, _flags, _addr, sh_offset, sh_size, sh_link, _info, _align, sh_entsize = struct.unpack_from(prefix + "IIQQQQIIQQ", data, off)
            sections.append({"name_off": name, "type": sh_type, "offset": sh_offset, "size": sh_size, "link": sh_link, "entsize": sh_entsize})
        if not (0 <= e_shstrndx < len(sections)):
            return []
        shstr = sections[e_shstrndx]
        shstr_data = data[shstr["offset"] : shstr["offset"] + shstr["size"]]
        for sec in sections:
            name = _read_c_string(shstr_data, sec["name_off"])
            sec["name"] = name
        for sec in sections:
            if sec.get("name") not in {".dynsym", ".symtab"} or sec["entsize"] == 0:
                continue
            if not (0 <= sec["link"] < len(sections)):
                continue
            strtab = sections[sec["link"]]
            strings = data[strtab["offset"] : strtab["offset"] + strtab["size"]]
            count = min(sec["size"] // sec["entsize"], 20000)
            for idx in range(count):
                off = sec["offset"] + idx * sec["entsize"]
                if off + sec["entsize"] > len(data):
                    break
                if elf_class == 1:
                    st_name, st_value, _st_size, _st_info, _st_other, st_shndx = struct.unpack_from(prefix + "IIIBBH", data, off)
                else:
                    st_name, _st_info, _st_other, st_shndx, st_value, _st_size = struct.unpack_from(prefix + "IBBHQQ", data, off)
                if st_name == 0:
                    continue
                symbol = _read_c_string(strings, st_name)
                if symbol and (st_shndx == 0 or st_value == 0):
                    imports.append(symbol)
    except Exception:
        return imports
    return imports


def extract_apk_symbols(path: Path) -> list[str]:
    tokens: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                lowered = name.lower()
                if lowered.startswith("lib/") and lowered.endswith(".so"):
                    tokens.append(name)
                    try:
                        tokens.extend(extract_elf_imports(zf.read(name)))
                    except Exception:
                        continue
                elif lowered.endswith(".dex"):
                    blob = zf.read(name)
                    for match in re.finditer(rb"L(?:android|java|javax|dalvik|kotlin|okhttp|retrofit)/[A-Za-z0-9_/$-]+;", blob):
                        tokens.append(match.group(0).decode("utf-8", errors="ignore"))
                        if len(tokens) >= 4096:
                            break
    except (OSError, zipfile.BadZipFile):
        return tokens
    return tokens


def extract_import_tokens(path: str | Path, data: bytes, metadata: ExecutableMetadata) -> list[str]:
    sample_path = Path(path)
    if metadata.format == "PE":
        return extract_pe_imports(data)
    if metadata.format == "ELF":
        return extract_elf_imports(data)
    if metadata.format == "APK":
        return extract_apk_symbols(sample_path)
    return []


def network_tokens_from_imports(imports: Iterable[str]) -> list[str]:
    out = []
    for token in imports:
        lower = token.lower()
        if any(hint in lower for hint in NETWORK_IMPORT_HINTS):
            out.append(token)
    return out


def ids_from_tokens(tokens: list[Any], vocab_size: int, prefix: str, max_len: int = SEQ_LEN) -> torch.Tensor:
    out = torch.zeros(max_len, dtype=torch.long)
    values = tokens_to_ids(tokens, vocab_size=vocab_size, max_len=max_len, prefix=prefix)
    if values:
        tensor = torch.tensor(values, dtype=torch.long)
        out[: tensor.numel()] = tensor
    return out


def _branch_target(op_str: str) -> int | None:
    match = re.search(r"0x[0-9a-fA-F]+|\b\d+\b", op_str)
    if not match:
        return None
    try:
        return int(match.group(0), 0)
    except ValueError:
        return None


def _is_control_flow(mnemonic: str) -> bool:
    return mnemonic.startswith("j") or mnemonic in {"b", "bl", "bx", "beq", "bne", "cbz", "call", "ret", "retn", "jr", "jal", "jalr"}


def cfg_from_instructions(instructions: list[Instruction]) -> tuple[torch.Tensor, torch.Tensor]:
    if not instructions:
        return torch.zeros((0, CFG_NODE_DIM), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long)
    max_nodes = 2048
    instructions = instructions[:max_nodes]
    addr_to_idx = {ins.address: idx for idx, ins in enumerate(instructions)}
    edges: set[tuple[int, int]] = set()
    mapper = SemanticMapperProcessor()
    semantic_rows = mapper.process(instructions)
    semantic_values = [float(row.get("semantic_id", 1)) for row in semantic_rows]
    sizes = [float(max(0, ins.size)) for ins in instructions]
    for idx, ins in enumerate(instructions):
        mnemonic = ins.mnemonic.lower()
        target = _branch_target(ins.op_str)
        if _is_control_flow(mnemonic) and target in addr_to_idx:
            edges.add((idx, addr_to_idx[target]))
        if idx + 1 < len(instructions) and mnemonic not in {"ret", "retn", "jr"}:
            edges.add((idx, idx + 1))
    undirected = set(edges)
    for src, dst in list(edges):
        undirected.add((dst, src))
    degree = [0.0 for _ in instructions]
    for src, dst in undirected:
        degree[src] += 1.0
        degree[dst] += 0.0
    max_degree = max(degree) if degree else 1.0
    x = torch.tensor(
        [
            [
                degree[idx],
                semantic_values[idx] / 15.0,
                sizes[idx] / 16.0,
                degree[idx] / max(max_degree, 1.0),
            ]
            for idx in range(len(instructions))
        ],
        dtype=torch.float32,
    )
    if not undirected:
        return x, torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor(sorted(undirected), dtype=torch.long).t().contiguous()
    return x, edge_index


def load_edgelist_graph(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        graph = nx.read_edgelist(path, nodetype=str)
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        edges = []
        for src, dst in graph.edges():
            edges.append([node_to_idx[src], node_to_idx[dst]])
            edges.append([node_to_idx[dst], node_to_idx[src]])
        if not edges:
            return torch.zeros((0, CFG_NODE_DIM), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        x = torch.tensor(
            [[float(degrees[node]), float(degrees[node]), float(degrees[node]), float(degrees[node]) / max(max_degree, 1)] for node in nodes],
            dtype=torch.float32,
        )
        return x, edge_index
    except Exception:
        return torch.zeros((0, CFG_NODE_DIM), dtype=torch.float32), torch.zeros((2, 0), dtype=torch.long)


def extract_isr_and_cfg(data: bytes, arch: str, log: ExtractionLog | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arch = canonical_arch(arch)
    if arch == "unknown":
        if log is not None:
            log.add("ISR/CFG unavailable: executable architecture could not be detected.")
        return (
            torch.zeros(ISR_LEN, 4, dtype=torch.long),
            torch.zeros((0, CFG_NODE_DIM), dtype=torch.float32),
            torch.zeros((2, 0), dtype=torch.long),
        )
    try:
        disassembler = DisassemblerProcessor(arch=arch)
        instructions = disassembler.process({"bytes": data, "arch": arch})
        isr = ArchitectureNormalizationPipeline(arch=arch, disassembler=disassembler).process({"bytes": data, "arch": arch})
        graph_x, graph_edge_index = cfg_from_instructions(instructions)
        if graph_x.numel() == 0 and log is not None:
            log.add("CFG unavailable: disassembler produced no instructions.")
        return isr[:ISR_LEN].long(), graph_x, graph_edge_index
    except Exception as exc:
        if log is not None:
            log.add(f"ISR/CFG extraction failed for arch={arch}: {type(exc).__name__}: {exc}")
        return (
            torch.zeros(ISR_LEN, 4, dtype=torch.long),
            torch.zeros((0, CFG_NODE_DIM), dtype=torch.float32),
            torch.zeros((2, 0), dtype=torch.long),
        )


def sandbox_cache_path(path: str | Path, cache_dir: str | Path = ".xnerf_cache/sandbox_features", arch_hint: str = "unknown") -> Path:
    sample_path = Path(path)
    stat = sample_path.stat()
    key = "|".join(
        [
            PREPROCESSING_CACHE_VERSION,
            str(sample_path.resolve()),
            str(stat.st_size),
            str(int(stat.st_mtime_ns)),
            canonical_arch(arch_hint),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()
    return Path(cache_dir) / digest[:2] / f"{digest}.pt"


def extract_static_modalities(
    path: str | Path,
    arch_hint: str = "unknown",
    cache_dir: str | Path | None = ".xnerf_cache/sandbox_features",
    use_cache: bool = True,
) -> dict[str, Any]:
    sample_path = Path(path)
    cache_file = sandbox_cache_path(sample_path, cache_dir or ".xnerf_cache/sandbox_features", arch_hint=arch_hint)
    if use_cache and cache_dir and cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu")
        if isinstance(cached, dict) and cached.get("cache_version") == PREPROCESSING_CACHE_VERSION:
            features = cached["features"]
            if isinstance(features, dict) and isinstance(features.get("metadata"), dict):
                features["metadata"]["cache_hit"] = True
            return features

    log = ExtractionLog()
    data = sample_path.read_bytes()
    metadata = detect_executable(sample_path, data)
    hint = canonical_arch(arch_hint)
    if hint != "unknown" and hint != metadata.arch:
        log.add(f"Architecture hint '{arch_hint}' overrides detected architecture '{metadata.arch}'.")
        metadata.arch = hint
        metadata.arch_raw = str(arch_hint)
    elif metadata.arch == "unknown":
        path_arch = infer_arch_from_path(sample_path)
        if path_arch != "unknown":
            metadata.arch = path_arch
            metadata.arch_raw = "path_hint"
            log.add(f"Architecture inferred from path because file headers were inconclusive: {path_arch}.")

    imports = extract_import_tokens(sample_path, data, metadata)
    if not imports:
        log.add(f"API imports unavailable: no imported symbols could be extracted from {metadata.format}.")
    network_tokens = network_tokens_from_imports(imports)
    if not network_tokens:
        log.add("Network tokens unavailable: no networking-related imports were found.")
    isr, graph_x, graph_edge_index = extract_isr_and_cfg(data, metadata.arch, log=log)
    memory_trace = zero_memory_trace(
        "Memory trace unavailable: training uses cached feature-vector tensors for memory modality; no equivalent cache exists for this standalone binary.",
        log=log,
    )
    sha256 = hashlib.sha256(data).hexdigest()
    features = {
        "binary_image": binary_image_from_bytes(data),
        "graph_x": graph_x,
        "graph_edge_index": graph_edge_index,
        "api_ids": ids_from_tokens(imports, API_VOCAB_SIZE, "api"),
        "network_ids": ids_from_tokens(network_tokens, NETWORK_VOCAB_SIZE, "net"),
        "memory_trace": memory_trace,
        "isr": isr,
        "arch_id": torch.tensor(ARCH_TO_ID.get(metadata.arch, ARCH_TO_ID["unknown"]), dtype=torch.long),
        "label": torch.tensor(0, dtype=torch.long),
        "metadata": {
            "path": str(sample_path),
            "file_name": sample_path.name,
            "size_bytes": len(data),
            "sha256": sha256,
            "format": metadata.format,
            "arch": metadata.arch,
            "arch_raw": metadata.arch_raw,
            "entrypoint": metadata.entrypoint,
            "image_base": metadata.image_base,
            "api_token_count": len(imports),
            "network_token_count": len(network_tokens),
            "cfg_node_count": int(graph_x.shape[0]),
            "cfg_edge_count": int(graph_edge_index.shape[1]),
            "warnings": [*metadata.notes, *log.warnings],
            "cache_hit": False,
        },
    }
    if use_cache and cache_dir:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cache_version": PREPROCESSING_CACHE_VERSION, "features": features}, cache_file)
    return features
