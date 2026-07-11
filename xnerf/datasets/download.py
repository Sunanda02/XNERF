from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import requests


DATASET_HINTS = {
    "malnet_tiny": "Download from https://mal-net.org/ and place archives under data/raw/malnet_tiny.",
    "cicmaldroid2020": "Download from CIC dataset portal after accepting terms.",
    "drebin": "Download feature vectors from the official Drebin distribution after authorization.",
    "ember": "Use the EMBER public feature release or ember package scripts.",
    "virusshare": "Requires VIRUSSHARE_API_KEY and explicit authorization to handle samples.",
    "cape": "Export CAPE reports as JSON into data/raw/cape/reports.",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_url(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def download_ember(root: Path) -> None:
    out = ensure_dir(root / "raw" / "ember")
    print(f"EMBER: place official feature files in {out}")
    print("If the ember package is installed, run its documented downloader from this directory.")


def download_virusshare(root: Path) -> None:
    api_key = os.getenv("VIRUSSHARE_API_KEY")
    out = ensure_dir(root / "raw" / "virusshare")
    if not api_key:
        print("Skipping VirusShare: set VIRUSSHARE_API_KEY after confirming authorization.")
        return
    hashes_file = out / "hashes.txt"
    if not hashes_file.exists():
        print(f"Create {hashes_file} with approved sample hashes, one per line.")
        return
    for sha256 in hashes_file.read_text(encoding="utf-8").splitlines():
        sha256 = sha256.strip()
        if not sha256:
            continue
        dest = out / f"{sha256}.bin"
        if dest.exists():
            continue
        url = f"https://virusshare.com/apiv2/download?apikey={api_key}&hash={sha256}"
        print(f"Downloading authorized VirusShare sample {sha256}")
        download_url(url, dest)


def print_manual_hint(root: Path, name: str) -> None:
    ensure_dir(root / "raw" / name)
    print(f"{name}: {DATASET_HINTS[name]}")


DOWNLOADERS: dict[str, Callable[[Path], None]] = {
    "malnet_tiny": lambda root: print_manual_hint(root, "malnet_tiny"),
    "cicmaldroid2020": lambda root: print_manual_hint(root, "cicmaldroid2020"),
    "drebin": lambda root: print_manual_hint(root, "drebin"),
    "ember": download_ember,
    "virusshare": download_virusshare,
    "cape": lambda root: print_manual_hint(root, "cape"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="X-NERF++ dataset acquisition helper")
    parser.add_argument("--root", default="data", type=Path)
    parser.add_argument("--dataset", choices=[*DOWNLOADERS.keys(), "all"], default="all")
    args = parser.parse_args()
    names = DOWNLOADERS.keys() if args.dataset == "all" else [args.dataset]
    for name in names:
        DOWNLOADERS[name](args.root)


if __name__ == "__main__":
    main()

