from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sandbox_dir = Path(__file__).resolve().parent
    repo_root = sandbox_dir.parent
    sys.path.insert(0, str(sandbox_dir))
    sys.path.insert(0, str(repo_root))

from config import load_sandbox_config
from feature_extractor import FeatureExtractionError
from inference import InferenceError, format_terminal_report, run_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Run terminal XNERF inference on one file")
    parser.add_argument("file_path", type=Path)
    parser.add_argument("--config", default=None, help="YAML config path; defaults to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override configured checkpoint path")
    parser.add_argument("--arch", default=None, help="Optional architecture override, e.g. x86, x64, arm, arm64, mips, mipsel, riscv")
    args = parser.parse_args()

    try:
        config = load_sandbox_config(args.config)
        if args.checkpoint:
            config.checkpoint = Path(args.checkpoint)
        if args.arch:
            config.arch = args.arch
        result = run_inference(args.file_path, config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FeatureExtractionError, InferenceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(format_terminal_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
