from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.evaluation.test_after_training import load_model
from xnerf.utils.base import collate_dicts, move_to_device
from xnerf.utils.config import load_config
from xnerf.zero_shot.prototypes import save_prototype_bank


@torch.no_grad()
def build_family_prototypes(config_path: str, checkpoint_path: str, manifest_path: str, output_path: str) -> Path:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), cfg, device)
    ds = MalwareManifestDataset(manifest_path, require_cache=True)
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 4), shuffle=False, num_workers=cfg["training"].get("num_workers", 2), collate_fn=collate_dicts)

    vectors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in tqdm(loader, desc="build zero-shot prototypes"):
        families = batch.get("family", ["unknown"] * len(batch["label"]))
        batch = move_to_device(batch, device)
        outputs = model(batch)
        emb = outputs["zero_shot_embedding"].detach().cpu()
        for family, vector in zip(families, emb):
            vectors[str(family)].append(vector)

    labels = sorted(vectors)
    prototypes = torch.stack([torch.stack(vectors[label]).mean(dim=0) for label in labels])
    return save_prototype_bank(
        output_path,
        prototypes,
        labels,
        metadata={"source_manifest": manifest_path, "config": config_path, "checkpoint": checkpoint_path},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zero-shot prototype bank from trained X-NERF++ embeddings")
    parser.add_argument("--config", default="xnerf/configs/kaggle.yaml")
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/best.pt")
    parser.add_argument("--manifest", default="/kaggle/working/data/processed/train_manifest.jsonl")
    parser.add_argument("--output", default="/kaggle/working/runs/zero_shot/prototypes.pt")
    args = parser.parse_args()
    out = build_family_prototypes(args.config, args.checkpoint, args.manifest, args.output)
    print(json.dumps({"prototype_bank": str(out)}, indent=2))


if __name__ == "__main__":
    main()
