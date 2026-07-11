from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from xnerf.explainability.report_generator import ReportGenerator
from xnerf.model import XNERFPlusPlus
from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline


def load_local_model(checkpoint_path: Path, device: torch.device) -> XNERFPlusPlus:
    payload = torch.load(checkpoint_path, map_location=device)
    model_config = payload.get("model_config", {})
    model = XNERFPlusPlus(
        num_classes=int(model_config.get("num_classes", 2)),
        num_families=int(model_config.get("num_families", 32)),
    ).to(device)
    model.load_state_dict(payload.get("state_dict", payload), strict=False)
    return model.eval()


def make_single_batch(sample_path: Path, arch: str, device: torch.device) -> dict[str, torch.Tensor]:
    from xnerf.datasets.loaders import MalwareManifestDataset

    tmp = Path("runs/local/tmp_manifest.jsonl")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps({"path": str(sample_path), "label": 0, "family": "unknown", "arch": arch}) + "\n", encoding="utf-8")
    ds = MalwareManifestDataset(tmp)
    batch = ds[0]
    batch["isr"] = ArchitectureNormalizationPipeline(arch=arch).process({"bytes": sample_path.read_bytes(), "arch": arch})
    return {k: v.unsqueeze(0).to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local X-NERF++ inference with a Kaggle-trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/xnerf_local_inference.pt"))
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--arch", default="x86")
    parser.add_argument("--report", type=Path, default=Path("runs/local/report.pdf"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_local_model(args.checkpoint, device)
    batch = make_single_batch(args.sample, args.arch, device)
    with torch.no_grad():
        outputs = model(batch)
    graph = model.renderer.reconstruct_graphs(outputs["stage_logits"])[0]
    reporter = ReportGenerator()
    summary = reporter.summarize(outputs, graph)
    reporter.generate_pdf(summary, args.report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
