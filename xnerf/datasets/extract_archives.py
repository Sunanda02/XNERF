from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ARCHIVE_SUFFIXES = {
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".tbz2",
    ".xz",
    ".txz",
}

DATA_FILE_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".parquet"}


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in ARCHIVE_SUFFIXES) or name.endswith(".tar.gz")


def is_data_file(path: Path) -> bool:
    return path.suffix.lower() in DATA_FILE_SUFFIXES


def _extract_archive(archive: Path, dst_dir: Path, overwrite: bool) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / f".extracted_{archive.stem.replace('.', '_')}"
    if marker.exists() and not overwrite:
        print(f"already extracted: {archive}")
        return False
    print(f"extracting {archive} -> {dst_dir}")
    shutil.unpack_archive(str(archive), str(dst_dir))
    marker.write_text(str(archive), encoding="utf-8")
    return True


def extract_dataset_archives(archive_root: Path, data_root: Path, overwrite: bool = False) -> dict[str, int]:
    """Extract user-provided dataset archives into Kaggle working storage.

    Expected input locations. Every dataset and subdirectory is optional:
        archive_root/malnet_tiny/images/*.zip|*.tar|*.tar.gz
        archive_root/malnet_tiny/graphs/*.zip|*.tar|*.tar.gz
        archive_root/andmal2020/static/*.zip|*.tar|*.tar.gz
        archive_root/andmal2020/dynamic/*.zip|*.tar|*.tar.gz
        archive_root/cicmaldroid2020/**/*.zip|*.tar|*.tar.gz
        archive_root/drebin/*.zip|*.tar|*.tar.gz
        archive_root/ember/*.zip|*.tar|*.tar.gz
        archive_root/virusshare/*.zip|*.tar|*.tar.gz
        archive_root/cape/*.zip|*.tar|*.tar.gz

    Outputs:
        data_root/raw/<same relative subdirectory>/...
    """

    counts: dict[str, int] = {}
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    if not archive_root.exists():
        print(f"archive root not present, skipping extraction: {archive_root}")
        return counts

    for archive in archive_root.rglob("*"):
        if not archive.is_file():
            continue
        relative_parent = archive.parent.relative_to(archive_root)
        dst_dir = raw_root / relative_parent
        key = relative_parent.as_posix() or "root"
        counts.setdefault(key, 0)
        if is_archive(archive):
            if _extract_archive(archive, dst_dir, overwrite):
                counts[key] += 1
            continue
        if not is_data_file(archive):
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / archive.name
        if dst.exists() and not overwrite:
            print(f"already copied: {archive}")
            continue
        print(f"copying {archive} -> {dst}")
        shutil.copy2(archive, dst)
        counts[key] += 1

    extracted_nested = True
    while extracted_nested:
        extracted_nested = False
        for archive in raw_root.rglob("*"):
            if not archive.is_file() or not is_archive(archive):
                continue
            relative_parent = archive.parent.relative_to(raw_root)
            key = relative_parent.as_posix() or "root"
            counts.setdefault(key, 0)
            dst_dir = archive.parent / archive.stem
            if _extract_archive(archive, dst_dir, overwrite):
                counts[key] += 1
                extracted_nested = True
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kaggle-mounted X-NERF++ dataset archives")
    parser.add_argument("--archive-root", type=Path, default=Path("/kaggle/input/xnerf-malware-archives/archives"))
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = extract_dataset_archives(args.archive_root, args.data_root, args.overwrite)
    print(counts)


if __name__ == "__main__":
    main()
