from __future__ import annotations

from importlib import util
from pathlib import Path
import sys


def _load_main() -> callable:
    sandbox_dir = Path(__file__).resolve().parent / "sandbox"
    module_path = sandbox_dir / "sandbox.py"
    spec = util.spec_from_file_location("xnerf_sandbox_entrypoint", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load sandbox entrypoint: {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    raise SystemExit(_load_main()())

