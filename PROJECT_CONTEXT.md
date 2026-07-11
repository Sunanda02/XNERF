# PROJECT_CONTEXT.md
## X-NERF++ Permanent Project Context

Audited and remediated on 2026-06-24 in `D:\GITrepos\XNERF`.

X-NERF++ is a research-stage defensive malware ML framework for multimodal malware detection, family classification, and cross-architecture representation learning. It ingests heterogeneous malware-research artifacts into JSONL manifests, loads tensors for byte images, API sequences, network events, numeric traces, CFG graphs, and ISR representations, then trains `XNERFPlusPlus`.

## Remediation Summary

The repository previously had three critical wiring problems: ISR tensors were loaded but unused, graph embeddings were encoded under a modality key ignored by fusion, and dataset-name placeholders could contribute to family loss. This pass fixed those issues:

- Added `xnerf/encoders/isr.py::ISREncoder`.
- Wired `XNERFPlusPlus.forward` so `batch["isr"] -> ISREncoder -> SemanticFieldSynchronizer`.
- Changed graph fusion key from `graph` to `cfg`, matching `SemanticFieldSynchronizer.modalities`.
- Added `isr` to `SemanticFieldSynchronizer.modalities`.
- Masked invalid malware-family placeholders as `family_label=-1`.
- Changed family CE to use `ignore_index=-1`.

## Current Architecture

```text
batch
  binary_image -> BinaryImageEncoder --.
  api_ids      -> APIEncoder -----------.
  network_ids  -> NetworkEncoder -------.
  memory_trace -> MemoryEncoder --------+-> SemanticFieldSynchronizer -> semantic [B,T,2048]
  graph_x/edge -> CFGEncoder -----------'
  isr          -> ISREncoder -----------'

semantic.mean -> CrossArchitectureAligner -> malware_logits, family_logits, zero_shot_embedding, arch_logits
semantic + memory + arch embeddings -> MNEF -> field, behavior_logits -> TrajectoryDecoder -> stage/transition logits
```

## Live Loss

```text
final_loss =
  malware_ce
  + 0.1 * family_ce(ignore_index=-1)
  + 0.1 * arch_adv
  + 0.01 * field_smooth
```

Behavior/stage heads are still not supervised because the dataset does not provide behavior-stage targets. Paired cross-architecture alignment is still not live because the dataset does not provide paired same-malware samples across architectures.

## Scientific Status

Training-ready engineering state:

- ISR and graph branches now influence the fused representation used by malware and family heads.
- Invalid family placeholders no longer train the family classifier.
- The trainer can save checkpoints and validates finite tensors/losses/gradients when dependencies are installed.

Still not publication-ready:

- No committed checkpoint or metrics exist.
- No end-to-end smoke test could be run in this environment because available Python runtimes lack `torch`/`pytest`.
- No supervised attack-stage labels exist.
- No paired cross-architecture corpus exists.
- `num_families` remains manifest-dependent and must match the active `family_vocab.json`.

## Important Files

- `xnerf/model.py`: end-to-end model wiring.
- `xnerf/encoders/isr.py`: ISR encoder.
- `xnerf/synchronization/sfs.py`: multimodal fusion.
- `xnerf/datasets/loaders.py`: manifest dataset and invalid-family masking.
- `xnerf/training/losses.py`: live training losses.
- `config.yaml`: local debug config with `num_families: 32`.
- `config_balanced_90k.yaml`: balanced 90k config with `num_families: 64`.

## Next Required Work

1. Install/repair the project Python environment and run the full tests.
2. Run a one-mini-batch smoke test with real manifests and confirm checkpoint save/validation.
3. Build a manifest and compute actual valid/ignored family counts.
4. Add behavior-stage labels before claiming meaningful trajectory reconstruction.
5. Add paired or class-conditional cross-architecture batches before claiming cross-architecture alignment beyond the adversarial head.

