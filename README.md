# X-NERF++

Cross-Architecture Neural Execution Rendering Framework for defensive malware intelligence.

## Current Status

X-NERF++ is a research-stage PyTorch framework for multimodal malware detection, family classification, and cross-architecture representation learning. The repository now includes a remediation pass for the highest-risk wiring issues found in the audit:

- `batch["isr"]` is encoded by `ISREncoder` and fused with the shared semantic feature space.
- CFG/graph embeddings are passed to the synchronizer under the active `cfg` modality key.
- Placeholder dataset-name family labels are masked with `family_label = -1` and ignored by family CE.
- The live training loss remains malware CE + masked family CE + architecture adversarial CE + field smoothness.

No large-scale training results, checkpoint, held-out metrics, or publication-quality claims are committed in this repository. Treat the model as training-ready code, not as a validated malware detector.

## What Is Included

- Dataset ingestion for heterogeneous malware research data: feature CSV/Parquet, API sequences, CAPE/Avast-style dynamic reports, graph `.edgelist` files, and authorized binary samples.
- Encoders for byte images, API sequences, network events, memory/numeric traces, CFG graphs, and ISR tensors.
- Semantic Field Synchronizer producing `[batch,time,2048]` shared representations.
- Malware Neural Execution Field `F(x,t,s,m,a)` and trajectory decoder.
- Gradient-reversal architecture adversarial head.
- Masked family classification for invalid or placeholder family labels.
- Training, validation, testing, zero-shot prototype evaluation, export, FastAPI, local CLI, and Docker scaffolding.

## Defensible Claims

Safe to claim from source inspection:

- The repository implements a unified ingestion and manifest pipeline.
- The model architecture supports multimodal fusion when the corresponding tensors are present.
- ISR and graph branches now feed the shared semantic fusion path.
- Family loss ignores invalid malware-family placeholders via `ignore_index=-1`.
- The trainer implements AMP, gradient accumulation, checkpointing, validation, and non-finite checks.

Do not claim yet:

- Any accuracy, F1, ROC-AUC, zero-shot, or cross-architecture result.
- Supervised behavior-stage or attack-stage reconstruction.
- Paired same-malware cross-architecture alignment.
- Successful Docker/API deployment against a trained checkpoint.
- Five-modality real-data training unless the manifest proves each modality is populated for the same samples.

See `CLAIM_VALIDATION_FINAL.md` for the strict claim table.

## Training Commands

Local debug run:

```powershell
C:\Users\Mayukh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m xnerf.pipeline.local_run pipeline --config config.yaml
```

Kaggle run:

```bash
pip install -q torch-geometric transformers fastapi uvicorn capstone networkx ray umap-learn reportlab
python -m xnerf.pipeline.kaggle_run pipeline --config xnerf/configs/kaggle.yaml
```

Balanced 90k local config:

```powershell
python -m xnerf.pipeline.local_run pipeline --config config_balanced_90k.yaml
```

`config_balanced_90k.yaml` currently sets `model.num_families: 64`. That value must match the generated `family_vocab.json` for the exact manifest used in training.

## Dataset Layout

Place authorized archives under:

```text
data/
  archives/
    malnet_tiny/
    AndMal2020/
    cicmaldroid2020/
    drebin/
    ember/
    virusshare/
    cape/reports/
```

The pipeline writes:

```text
data/raw/
data/cache/isr/
data/processed/manifest.jsonl
data/processed/train_manifest.jsonl
data/processed/val_manifest.jsonl
data/processed/test_manifest.jsonl
data/processed/family_vocab.json
```

Raw malware samples require explicit authorization. X-NERF++ treats binaries as analysis inputs and does not execute samples.

## Local Inference

After training and export:

```powershell
$env:XNERF_CHECKPOINT="models/xnerf_local_inference.pt"
uvicorn xnerf.api.app:app --reload
```

Single-file CLI:

```powershell
python -m xnerf.deployment.local_analyze --checkpoint models/xnerf_local_inference.pt --sample path\to\sample.bin --arch x86
```

## Audit Reports

The remediation reports generated for the current state are:

- `AUDIT_FIX_REPORT.md`
- `ISR_INTEGRATION_REPORT.md`
- `GRAPH_INTEGRATION_REPORT.md`
- `CROSS_ARCH_AUDIT.md`
- `FAMILY_REPAIR_REPORT.md`
- `LOSS_GRAPH.md`
- `TRAINING_READINESS_REPORT.md`
- `CLAIM_VALIDATION_FINAL.md`
- `PROJECT_CONTEXT.md`

## Verification

Performed in this environment:

- AST syntax parse for edited Python files: passed.

Not performed here:

- `pytest` and model smoke tests, because the available Python runtimes either point to a stale venv or do not include `pytest`/`torch`.

