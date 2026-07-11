<div align="center">

# X-NERF++

**Cross-Architecture Neural Execution Rendering Framework**

*A research-stage, defensive, multi-modal deep learning framework for malware detection, family attribution, and cross-architecture representation learning.*

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Research](https://img.shields.io/badge/status-research--stage-orange)
![Docker](https://img.shields.io/badge/docker-supported-2496ed)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20service-009688)
![Status](https://img.shields.io/badge/checkpoints-not%20included-red)

</div>

---

> **⚠️ Read this before anything else.** This README was written by directly auditing the source code, configs, and tests in this repository — not by rewriting the previous README from assumption. Wherever the previous README's claims could not be verified against the code (a fixed "221 families" head, an MIT `LICENSE` file, an `xnerf_architecture_4k.png` diagram, a `train.py`/`service/app.py`/`cli_analyzer.py` layout), those claims have been corrected or removed. See [Limitations](#limitations) and [Project Status](#project-status) for the full list of what is and isn't real.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Model Components](#model-components)
- [Research Contributions](#research-contributions)
- [Project Status](#project-status)
- [Roadmap](#roadmap)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

X-NERF++ is a PyTorch research framework for **defensive malware analysis** that learns a single, shared representation of a binary from six heterogeneous input modalities: a byte-plane image, disassembled API call sequences, a function-level control-flow graph, a memory-access trace, network event tokens, and an architecture-normalized *Intermediate Semantic Representation* (ISR) derived from disassembly. Each modality is encoded independently — a ResNet-18 for byte images, Transformer encoders for API/network token sequences, a Graph Attention Network for control-flow graphs, a dilated temporal CNN for memory traces, and a small Transformer for the ISR stream — and fused by a **Semantic Field Synchronizer (SFS)** into a shared `[batch, time, 2048]` latent sequence.

That shared representation is pushed through an adversarial **Cross-Architecture Aligner**, which uses a gradient-reversal layer and an instruction-set-architecture discriminator to encourage architecture-invariant features across x86, x64, ARM, ARM64, MIPS, and RISC-V. The pooled, aligned representation feeds a binary malware/benign classifier and a family classifier. In parallel, the synchronized sequence, a memory-context vector, and an architecture embedding are consumed by a coordinate-conditioned network called the **Malware Neural Execution Field (MNEF)** — a NeRF-style implicit function `F(x, t, s, m, a)` over normalized execution position and time — whose output is decoded by a **Trajectory Decoder** into per-timestep attack-stage logits and stage-transition logits, from which a directed graph of inferred behavior stages (`Environment Check → Privilege Escalation → Persistence → Credential Access → Exfiltration`) can be reconstructed.

The repository also includes the supporting engineering needed to operate this model as a research pipeline rather than a single script: a dataset-manifest builder that ingests several public malware-research corpora into a common JSONL schema, a family-name normalization and vocabulary system, a checkpointing/resume-capable trainer with numerical-stability guards, standalone and in-package evaluation utilities, a cosine-similarity zero-shot prototype classifier, a FastAPI inference service, a Windows CMD/PowerShell single-file sandbox tool, PDF report generation, and Docker packaging. As documented in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md), the codebase has been through at least one internal audit-and-remediation pass that fixed real wiring bugs (an unused ISR branch, a mismatched fusion key, and unmasked placeholder family labels). It is **training-ready** engineering, not a validated, benchmarked, or peer-reviewed research result — see [Project Status](#project-status).

## Key Features

Only capabilities with a corresponding, importable implementation in this repository are listed.

| Feature | Description | Status |
|---|---|---|
| Six-modality encoder stack | Independent encoders for byte-plane image, API sequence, control-flow graph, memory trace, network events, and ISR (`xnerf/encoders/*.py`) | Implemented |
| Semantic Field Synchronizer | Cross-modal multi-head attention + learned time embeddings + bidirectional GRU fusing present modalities into `[B,T,2048]` (`xnerf/synchronization/sfs.py`) | Implemented |
| Adversarial cross-architecture alignment | Gradient-reversal layer + architecture discriminator over 7 architecture IDs (`xnerf/alignment/adversarial.py`) | Implemented; only the adversarial discriminator loss is active in training |
| Malware Neural Execution Field (MNEF) | Fourier-positional-encoded implicit field `F(x,t,s,m,a)` producing a continuous latent trajectory and behavior logits (`xnerf/fields/mnef.py`) | Implemented; behavior head is not supervised (no labels available) |
| Trajectory decoding & graph reconstruction | Per-timestep stage classification, pairwise transition logits, and `networkx` graph reconstruction over 5 fixed attack stages (`xnerf/renderer/trajectory_decoder.py`) | Implemented; runs unsupervised, output is not validated against ground truth |
| End-to-end model wiring | `XNERFPlusPlus` composes all of the above into one `forward()` (`xnerf/model.py`) | Implemented |
| Dataset manifest builder | Converts MalNet-Tiny, AndMal2020, CICMalDroid2020, Drebin, EMBER, CAPE sandbox JSON reports, and CIC-YNU-style CSVs into a unified JSONL manifest, with SHA-256 dedup and hash-based splitting (`xnerf/datasets/build_dataset.py`) | Implemented |
| Family name normalization | Alias table, placeholder-dataset-name detection, and vocabulary building so raw noisy family strings become a consistent label set (`xnerf/datasets/family_cleaning.py`) | Implemented |
| Invalid-family masking | Placeholder/unknown family rows are masked to `family_label=-1` and excluded from the family cross-entropy via `ignore_index=-1` (`xnerf/datasets/loaders.py`, `xnerf/training/losses.py`) | Implemented |
| Numerically-guarded trainer | AMP, gradient accumulation, gradient clipping, resumable checkpoints, and explicit non-finite tensor/gradient/optimizer-state checks that raise with full batch diagnostics (`xnerf/training/trainer.py`) | Implemented |
| Standalone + in-package evaluation | Accuracy/precision/recall/F1/ROC-AUC, per-architecture accuracy, confusion matrices, t-SNE/UMAP embedding plots (`evaluation/`, `xnerf/evaluation/`) | Implemented; **two parallel, overlapping implementations exist** |
| Zero-shot prototype classification | Cosine-similarity classification against per-family mean embeddings, with a save/load prototype-bank format (`xnerf/zero_shot/`) | Implemented; accuracy depends entirely on family coverage at train time |
| Explainability report generation | Structured summary dict plus a PDF report via `reportlab` (`xnerf/explainability/report_generator.py`) | Implemented |
| FastAPI inference service | `/upload`, `/analyze`, `/result/{id}`, `/health` endpoints; refuses to serve predictions with `503` if no checkpoint is configured (`xnerf/api/app.py`) | Implemented |
| Terminal / CMD sandbox tool | Single-file inference from Windows CMD/PowerShell with a formatted terminal report (`sandbox/`, root `sandbox.py`) | Implemented |
| Local + Kaggle orchestration CLIs | `xnerf/pipeline/local_run.py` and `xnerf/pipeline/kaggle_run.py` expose split subcommands (`build-manifest`, `train`, `validate`, `test`, `zero-shot`, `export`) and a monolithic `pipeline` subcommand | Implemented |
| Ray Tune launcher | `xnerf/training/ray_train.py` wraps `run_training` for hyperparameter search trials | Implemented, not exercised by tests |
| Docker / Compose deployment | `xnerf/deployment/Dockerfile` + `docker-compose.yml` build and serve the FastAPI app | Implemented |
| CNN / Transformer baselines | `CNNMalware` (byte-image CNN) and `MalBERT` (API-sequence Transformer) single-modality baselines for comparison (`xnerf/baselines/models.py`) | Implemented, not wired into the training or evaluation CLIs |
| pytest suite | 8 test modules covering dataset/family/CFG-report logic and the sandbox config/inference helpers | Implemented; **no test exercises the model, encoders, SFS, MNEF, or a training step** |

## Architecture Overview

```
Input Modalities
  binary_image [B,1,H,W]   api_ids [B,T]   graph_x/edge_index/batch   memory_trace [B,T,C]   network_ids [B,T]   isr [B,T,4]
        │                     │                    │                        │                     │                │
        ▼                     ▼                    ▼                        ▼                     ▼                ▼
 BinaryImageEncoder      APIEncoder           CFGEncoder             MemoryEncoder         NetworkEncoder      ISREncoder
   (ResNet-18)          (Transformer)      (GAT, 2 layers)         (dilated Conv1d)        (Transformer)      (Transformer)
        │                     │                    │                        │                     │                │
        └─────────────────────┴────────────────────┴────────────┬───────────┴─────────────────────┴────────────────┘
                                                                  ▼
                                          Semantic Field Synchronizer (SFS)
                                cross-modal attention → temporal expansion → BiGRU → [B, T=16, 2048]
                                                                  │
                                       ┌──────────────────────────┴───────────────────────────┐
                                       ▼                                                        ▼
                        Cross-Architecture Aligner                         Malware Neural Execution Field (MNEF)
                    (GRL + discriminator, pooled features)         F(x, t, s, m, a) over position/time/semantics/
                                       │                              memory-context/arch-embedding → [B,T,1024]
                     ┌─────────────────┼─────────────────┐                              │
                     ▼                 ▼                 ▼                              ▼
              malware_logits    family_logits    zero_shot_embedding              Trajectory Decoder
               [B,2]             [B,num_families]   [B,2048]                stage_logits [B,T,5]
                                                                              transition_logits [B,T-1,5,5]
                                                                                       │
                                                                                       ▼
                                                                        networkx.DiGraph behavior-stage graph
```

A preprocessing pipeline (`xnerf/preprocessing/`) turns raw bytes into disassembled instructions (via `capstone`), maps mnemonics onto a 16-class semantic ontology, and builds the fixed-length `[T,4]` ISR tensor consumed by `ISREncoder`. During training, four loss terms are combined: `malware_ce + 0.1·family_ce(ignore_index=-1) + 0.1·arch_adv + 0.01·field_smooth` (`xnerf/training/losses.py`). No architecture diagram image is currently checked into the repository; the ASCII diagram above reflects the actual `forward()` wiring in `xnerf/model.py`, not a rendered figure.

## Repository Structure

```text
XNERF-main/
├── config.yaml                        # local/manifest-first training config
├── config_publication_v2_50k.yaml     # 50k-sample config, num_families=71
├── docker-compose.yml                 # builds & serves the FastAPI app
├── requirements.txt
├── sandbox.py                         # thin wrapper that dispatches to sandbox/sandbox.py
├── LOCAL_RUN_COMMANDS_CMD.txt         # copy-paste Windows CMD workflow notes (author's working notes)
├── PROJECT_CONTEXT.md                 # internal audit log of the current engineering state
├── data/
│   └── archives/                      # dataset drop-zone; only small label CSVs and a drebin.zip
│                                       # are actually committed — raw corpora are gitignored
├── evaluation/                        # standalone, NPZ-driven evaluation CLI (metrics, reports, plots)
├── models/                            # local checkpoint drop-zone (empty; *.pt is gitignored)
├── sandbox/                           # Windows CMD/PowerShell single-file inference tool
│   ├── config.py                      # sandbox.* config block + checkpoint fallback resolution
│   ├── feature_extractor.py           # bytes → model-ready tensors for one file
│   ├── inference.py                   # runs the model, formats the terminal report
│   └── sandbox.py                     # CLI entry point
├── scripts/                           # one-off dataset inspection / validation utilities
├── tests/                             # pytest suite (dataset/family/CFG-report/sandbox-config coverage)
└── xnerf/
    ├── model.py                       # XNERFPlusPlus — end-to-end forward()
    ├── encoders/                      # binary_image.py, api.py, cfg.py, memory.py, network.py, isr.py
    ├── synchronization/sfs.py         # Semantic Field Synchronizer
    ├── alignment/adversarial.py       # Cross-Architecture Aligner (GRL + discriminator)
    ├── fields/mnef.py                 # Malware Neural Execution Field
    ├── renderer/trajectory_decoder.py # stage/transition heads + graph reconstruction
    ├── training/                      # trainer.py, losses.py, train.py, ray_train.py
    ├── evaluation/                    # test_after_training.py, evaluate.py (in-package copy)
    ├── datasets/                      # loaders, build_dataset, family_cleaning, validation, audit,
    │                                  # download, extract_archives
    ├── preprocessing/                 # disassembler, semantic_mapper, isr_builder, pipeline,
    │                                  # ontology, static_features
    ├── zero_shot/                     # prototypes.py, build_prototypes.py, evaluate_zero_shot.py
    │                                  # (+ an untracked "-2" variant, see Limitations)
    ├── explainability/report_generator.py
    ├── deployment/                    # Dockerfile, export_checkpoint.py, local_analyze.py
    ├── api/app.py                     # FastAPI inference service
    ├── baselines/models.py            # CNNMalware, MalBERT single-modality baselines
    ├── pipeline/                      # local_run.py, kaggle_run.py — orchestration CLIs
    ├── sandbox/cape_parser.py         # CAPE sandbox report parsing (used by build_dataset.py)
    ├── notebooks/kaggle_setup.py      # Kaggle-notebook environment bootstrap notes
    ├── configs/                       # default.yaml, kaggle.yaml, local_inference.yaml, datasets.yaml
    └── utils/                         # base.py (shared ABCs), config.py, io.py, seed.py, tokenization.py
```

> **Note:** There is no `models/encoders/`, `models/sfs.py`, `models/aligner.py`, `service/app.py`, `train.py`, `cli_analyzer.py`, or `xnerf_architecture.svg` at the repository root. If you have documentation, scripts, or citations referencing that layout, update them to the paths above.

## Installation

### Linux / macOS

```bash
git clone <this-repository-url>
cd XNERF-main

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Windows (CMD / PowerShell)

```cmd
git clone <this-repository-url>
cd XNERF-main

python -m venv .venv
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

`LOCAL_RUN_COMMANDS_CMD.txt` contains the maintainer's own copy-paste Windows workflow (manifest building, subsetting, cache generation, training) and is a useful reference alongside this README.

### Docker

```bash
docker compose up --build
```

This builds `xnerf/deployment/Dockerfile` (Python 3.11-slim, `pip install -r requirements.txt`) and runs `uvicorn xnerf.api.app:app --host 0.0.0.0 --port 8000`, mounting `./data`, `./checkpoints`, `./models`, and `./runs` as volumes and reading environment variables from `.env` (see `.env.example`).

> **Note:** `torch-geometric` and `torchvision` GPU wheels are architecture/CUDA-version sensitive. The Dockerfile installs whatever `requirements.txt` resolves to on `python:3.11-slim`, which is CPU-only unless you customize the base image.

## Requirements

- **Python**: 3.11 (per the Dockerfile base image; no other version is pinned or tested against)
- **CUDA**: no version is pinned. `xnerf/training/trainer.py` and the CLIs auto-detect `torch.cuda.is_available()` and fall back to CPU; training is expected to be materially slower on CPU given the Transformer/GAT/ResNet encoder stack.
- **Core dependencies** (`requirements.txt`): `torch>=2.2.0`, `torchvision>=0.17.0`, `torch-geometric>=2.5.0`, `transformers>=4.40.0`, `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`, `python-multipart>=0.0.9`, `capstone>=5.0.1`, `networkx>=3.2.1`, `ray>=2.10.0`, `numpy>=1.26.0`, `scikit-learn>=1.4.0`, `matplotlib>=3.8.0`, `umap-learn>=0.5.5`, `PyYAML>=6.0.1`, `requests>=2.31.0`, `tqdm>=4.66.0`, `reportlab>=4.1.0`, `pytest>=8.0.0`, `pyarrow>=16.0.0`

> **Note:** `transformers` is listed in `requirements.txt` but is not imported anywhere in `xnerf/` — the only reference is a commented `pip install` line in `xnerf/notebooks/kaggle_setup.py`. `BinaryImageEncoder` requires `torchvision` and `CFGEncoder` requires `torch_geometric`; both raise `RuntimeError` at construction if their optional dependency is missing, rather than silently degrading.

## Dataset Preparation

X-NERF++ does not ship trained-on data. `data/archives/` is a **drop zone**: `.gitignore` excludes `data/archives/`, `data/raw/`, `data/cache/`, and `data/processed/`, so only a handful of small artifacts are actually committed (`data/archives/cape/public_labels.csv`, `data/archives/MalBehavD-V1/MalBehavD-V1-dataset.csv`, and `data/archives/drebin/drebin.zip`). Everything else must be supplied by the user under an appropriate license/authorization for the corresponding dataset.

Expected drop-zone layout (`data/archives/README.md`, `xnerf/configs/datasets.yaml`):

```text
data/archives/
  malnet_tiny/{images,graphs}/*.zip|*.tar|*.tar.gz|*.tgz
  AndMal2020/{static,dynamic}/*.zip|*.tar|*.tar.gz|*.tgz
  cicmaldroid2020/*.zip|*.tar|*.tar.gz|*.tgz
  drebin/*.zip|*.tar|*.tar.gz|*.tgz
  ember/*.zip|*.tar|*.tar.gz|*.tgz
  virusshare/*.zip|*.tar|*.tar.gz|*.tgz          # requires VIRUSSHARE_API_KEY + explicit authorization
  cape/{*.zip|*.tar|*.tar.gz|*.tgz, reports/}     # CAPE sandbox JSON reports
```

Any new dataset can be added without a code change by dropping archives under `data/archives/<name>/<modality_or_split>/`; `xnerf/datasets/extract_archives.py` extracts recursively into `data/raw/<name>/<modality_or_split>/`.

**Pipeline stages** (`xnerf/datasets/build_dataset.py`, orchestrated by `xnerf/pipeline/local_run.py` / `kaggle_run.py`):

1. **Extract** — unpack archives into `data/raw/`.
2. **Build manifest** — scan `data/raw/`, infer architecture and label from path heuristics (`infer_arch`, `infer_label`), parse CAPE JSON reports (`xnerf/sandbox/cape_parser.py`) for API/network events and sandbox score, parse headerless static-feature CSVs into per-row samples, normalize family names (`xnerf/datasets/family_cleaning.py`), and write a JSONL manifest with SHA-256-based dedup and hash-based train/val/test splitting.
3. **Generate cache** — precompute per-sample tensors (binary image, ISR, CFG edgelist) via `generate_cache_from_manifest`; this step is resumable and skips files that already exist.
4. **Validate** — `xnerf/datasets/validation.py` and `xnerf/datasets/audit.py` check for placeholder family labels, missing vocabulary entries, and suspicious architecture-vs-path mismatches before training starts.

Each manifest row is expected to carry `path`, `label` (0=benign, 1=malware), `family`, `arch`, and cache pointers (`feature_path`, `isr_path`) once cache generation has run. `MalwareManifestDataset` (`xnerf/datasets/loaders.py`) reads this manifest and assembles the tensor dict consumed by `XNERFPlusPlus.forward()`.

## Configuration

All configs are flat YAML consumed by `xnerf/utils/config.py`. Four are checked in:

| Config | Purpose | `num_families` | Notes |
|---|---|---|---|
| `xnerf/configs/default.yaml` | Minimal template | 32 | No `test_manifest` key |
| `config.yaml` | Local, manifest-first workflow | 32 | `debug_max_batches: 50`, `use_amp: false`, includes a `local_inference` block pointing at `models/best.pt` |
| `config_publication_v2_50k.yaml` | ~50k-sample run | **71** | Illustrates that `num_families` is manifest-dependent, not fixed |
| `xnerf/configs/kaggle.yaml` | Kaggle-notebook paths (`/kaggle/...`), `use_amp: true`, `splits.{train_ratio,val_ratio}`, `export`/`outputs` path blocks | 32 | |
| `xnerf/configs/local_inference.yaml` | CPU inference-only config for the FastAPI service | 32 | |

Key fields:

- `data.{train,val,test}_manifest` / `data.full_manifest` — JSONL manifest paths.
- `data.archive_root` — where `data/archives/`-style input lives.
- `model.num_classes`, `model.num_families` — **must match the active `family_vocab.json`**; there is no fixed, canonical family count in this codebase (see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)).
- `training.{batch_size,lr,epochs,grad_accum,num_workers,checkpoint_dir,patience,grad_clip,use_amp,debug_max_batches,resume_from}` — forwarded to `XNerfTrainer`; any other key under `training` in the YAML is silently ignored by `run_training` (see `_TRAINER_KEYS` in `xnerf/training/train.py`).
- `local_inference.checkpoint` — path the maintainer's local machine uses for inference-only runs after training elsewhere (e.g. Kaggle).

## Training

```bash
# Local, manifest-first workflow (see LOCAL_RUN_COMMANDS_CMD.txt for the full copy-paste sequence)
python -m xnerf.datasets.build_dataset --root data --out data/processed/manifest.jsonl --split-mode hash --manifest-only
python -m xnerf.datasets.build_dataset --root data --generate-cache-from-manifest data/processed/manifest.jsonl
python -m xnerf.training.train --config config.yaml

# Resume from a checkpoint
python -m xnerf.training.train --config config.yaml --resume checkpoints/last.pt

# Validate an existing checkpoint on the val split
python -m xnerf.training.train --config config.yaml --validate-only --checkpoint checkpoints/best.pt
```

Or via the orchestration CLI, split into independent, re-runnable stages:

```bash
python -m xnerf.pipeline.local_run build-manifest --config config.yaml
python -m xnerf.pipeline.local_run train           --config config.yaml
python -m xnerf.pipeline.local_run validate        --config config.yaml
python -m xnerf.pipeline.local_run test             --config config.yaml
python -m xnerf.pipeline.local_run zero-shot        --config config.yaml
python -m xnerf.pipeline.local_run export           --config config.yaml
# or, everything at once:
python -m xnerf.pipeline.local_run pipeline --config config.yaml
```

`xnerf/pipeline/kaggle_run.py` mirrors this exact interface for Kaggle-notebook cells (`--config xnerf/configs/kaggle.yaml`), matching the `/kaggle/...` paths in `xnerf/configs/kaggle.yaml`.

- **Checkpointing** — `XNerfTrainer` atomically writes `checkpoints/best.pt` (on val-loss improvement) and `checkpoints/last.pt` (every epoch) via a temp-file-then-rename pattern, with a `PermissionError` fallback for locked files on Windows.
- **Resume** — `--resume <path>` restores model, optimizer, and AMP scaler state and continues from `epoch + 1`.
- **Mixed precision** — `training.use_amp: true` enables `torch.cuda.amp.autocast` + `GradScaler`; automatically disabled on CPU regardless of the config value.
- **Gradient accumulation** — `training.grad_accum` batches gradients before each optimizer step.
- **Numerical guards** — the trainer explicitly checks every model output, every loss term, gradients (post-unscale), parameters (post-step), and optimizer state for non-finite values, and raises `RuntimeError` with a batch diagnostic dump (`dataset`, `label`, `arch_id`, `family_label`, `sha256`, `sample_id`, `path`) rather than silently corrupting training.

> **⚠️ Known issue.** `run_validation()` in `xnerf/training/train.py` contains a debug block with incorrect indentation: the `for batch in loader:` loop body is dedented out of the loop after `paths = batch["path"]`, so the block below it (which prints diagnostics for the first `.exe`/`.dll`/`.elf`/`.so`/`.apk` sample and then calls `raise SystemExit`) executes at most once, using whichever `batch` the loop last bound, and then **terminates the process** before any predictions are accumulated. As written, `python -m xnerf.training.train --config config.yaml --validate-only` will not complete a full validation pass. `xnerf/training/trainer.py::validate()` (the loop used during normal `fit()`) does not have this issue. Similarly, `trainer.py::_step()` contains leftover `if batch_idx > 9180: print(...)` debug statements from a specific run. Both should be treated as known defects, not documented behavior, until removed or fixed upstream.

## Evaluation

Two separate, overlapping evaluation entry points exist in this repository (see [Limitations](#limitations)):

```bash
# In-package: checkpoint + manifest -> metrics, confusion matrix, t-SNE/UMAP
python -m xnerf.evaluation.test_after_training --config xnerf/configs/kaggle.yaml --checkpoint checkpoints/best.pt

# Standalone: either a checkpoint + manifest, or a pre-computed predictions .npz
python -m evaluation.evaluate --config config.yaml --manifest data/processed/test_manifest.jsonl --checkpoint checkpoints/best.pt --out results
python -m evaluation.evaluate --predictions runs/test/test_predictions.npz --out results
```

Metrics computed (`xnerf/evaluation/evaluate.py`, `evaluation/metrics.py`): accuracy, weighted precision/recall/F1, ROC-AUC (binary case), per-architecture accuracy and a cross-architecture accuracy proxy (accuracy restricted to rows with a known, non-`unknown` architecture label), plus confusion-matrix PNGs and t-SNE/UMAP projections of the `zero_shot_embedding` output when embeddings are available.

**Zero-shot evaluation:**

```bash
python -m xnerf.zero_shot.build_prototypes  --config xnerf/configs/kaggle.yaml --checkpoint checkpoints/best.pt --manifest data/processed/train_manifest.jsonl --output runs/zero_shot/prototypes.pt
python -m xnerf.zero_shot.evaluate_zero_shot --config xnerf/configs/kaggle.yaml --checkpoint checkpoints/best.pt --manifest data/processed/test_manifest.jsonl --prototypes runs/zero_shot/prototypes.pt --out runs/zero_shot
```

`ZeroShotPrototypeClassifier` classifies by cosine similarity against per-family mean embeddings built from the training set. Test samples whose family string is not present in the prototype bank are skipped from the reported metric — this is a design choice of the evaluator, not a defect, but it means the reported zero-shot accuracy is conditioned on family overlap and should be read alongside `evaluated_samples` / `prototype_count` in the output JSON.

## Inference

**CLI (terminal sandbox, Windows-oriented but not Windows-only):**

```cmd
python sandbox\sandbox.py path\to\sample.exe
python sandbox\sandbox.py path\to\sample.exe --checkpoint checkpoints\best_model.pt --arch x64
```

or, from the repository root:

```bash
python sandbox.py path/to/sample.exe
```

`sandbox/config.py` resolves the checkpoint from (in order) a `--checkpoint` CLI flag, the `XNERF_SANDBOX_CHECKPOINT` environment variable, or a `sandbox:` block in `config.yaml`; falls back to `models/best.pt` if present. The terminal report includes file name, malware probability, decision, predicted family, predicted architecture, confidence score, and inference time.

**Python API:**

```python
from pathlib import Path
import torch
from xnerf.deployment.local_analyze import load_local_model, make_single_batch

device = torch.device("cpu")
model = load_local_model(Path("models/xnerf_local_inference.pt"), device)
batch = make_single_batch(Path("sample.exe"), arch="x86", device=device)
with torch.no_grad():
    outputs = model(batch)
```

**FastAPI service:**

```bash
export XNERF_CHECKPOINT=models/xnerf_local_inference.pt   # PowerShell: $env:XNERF_CHECKPOINT=...
uvicorn xnerf.api.app:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Behavior |
|---|---|---|
| `/upload` | `POST` (multipart file) | Saves the file under `runs/api/uploads/`, returns an `upload_id` |
| `/analyze` | `POST` `{upload_id, arch}` | Runs the ISR pipeline + model forward pass, generates a summary and a PDF report; returns **`503`** if no checkpoint was loaded at startup |
| `/result/{job_id}` | `GET` | Returns the stored result dict for a given id |
| `/health` | `GET` | `{status, model_ready, device}` |

If `XNERF_CHECKPOINT` is unset or the file does not exist, the service still starts (with a freshly initialized, untrained `XNERFPlusPlus`) but `/analyze` will refuse to serve predictions until a real checkpoint is configured.

## Model Components

| Component | File | Summary |
|---|---|---|
| **Binary Image Encoder** | `xnerf/encoders/binary_image.py` | ResNet-18 (`torchvision`, no pretrained weights) with a modified 3-channel input conv, replicating single-channel byte-plane images to 3 channels; projects to 512-d |
| **API Encoder** | `xnerf/encoders/api.py` | Token + positional embeddings, 4-layer Transformer encoder (8 heads), padding-mask-aware mean pooling with an all-padding-row guard; 512-d output |
| **CFG Encoder** | `xnerf/encoders/cfg.py` | 2-layer Graph Attention Network (`torch_geometric.nn.GATConv`, 4 heads) with mean pooling over nodes per graph; 512-d output |
| **Memory Encoder** | `xnerf/encoders/memory.py` | 3-layer dilated 1-D CNN over the memory-access trace, adaptive average pooling; 512-d output |
| **Network Encoder** | `xnerf/encoders/network.py` | Token + positional embeddings, 3-layer Transformer encoder, same masked-mean-pooling pattern as the API encoder; 512-d output |
| **ISR Encoder** | `xnerf/encoders/isr.py` | Embeds 4 ISR fields (semantic class, architecture, address-delta bucket, instruction size), 2-layer Transformer encoder, masked mean pooling; 512-d output |
| **Semantic Field Synchronizer** | `xnerf/synchronization/sfs.py` | Per-modality linear projection + learned type embedding, multi-head self-attention over present modalities, mean-pooled summary broadcast across `time_steps` learned time embeddings, an FFN to 2048-d, and a bidirectional GRU over time; also exposes a static `contrastive_loss` (InfoNCE-style) helper not currently called by the trainer |
| **Cross-Architecture Aligner** | `xnerf/alignment/adversarial.py` | LayerNorm+GELU encoder producing "aligned" features, fed through a `GradientReverse` autograd function into an architecture discriminator; exposes a `losses()` helper for the adversarial term and an optional paired cross-architecture cosine loss (not wired into the default training loop) |
| **Malware Neural Execution Field (MNEF)** | `xnerf/fields/mnef.py` | Fourier positional encoding (8 bands) of position and time, concatenated with the 2048-d semantic state, 512-d memory context, and 64-d architecture embedding, passed through a 3-layer MLP with LayerNorm/GELU; exposes `field_losses()` for an optional behavior cross-entropy and a temporal-smoothness penalty |
| **Trajectory Decoder** | `xnerf/renderer/trajectory_decoder.py` | Linear heads over the MNEF field for 5-class stage logits and pairwise transition logits; `reconstruct_graphs()` builds a `networkx.DiGraph` per sample from the argmax stage sequence |
| **Output Heads** | `xnerf/model.py` | Two `nn.Linear` heads on the aligned 2048-d representation for binary malware classification and family classification |

## Research Contributions

Described conservatively, limited to what the code actually implements:

- **A single model spanning six malware-analysis modalities**, including a novel-for-this-codebase intermediate semantic representation (ISR) that maps architecture-specific disassembly onto a shared 16-class instruction-semantic ontology (`xnerf/preprocessing/ontology.py`) before encoding — an explicit attempt at architecture normalization at the *representation* level, prior to the learned adversarial alignment stage.
- **A NeRF-inspired implicit-field formulation applied to malware execution**: instead of a fixed-length behavior vector, MNEF treats execution behavior as a continuous function of normalized position/time conditioned on the fused semantic state, memory context, and architecture embedding, with a Fourier positional encoding borrowed from the neural-rendering literature.
- **A masked family-loss scheme for noisy, multi-source label vocabularies**: placeholder or dataset-artifact family strings (e.g. literal dataset filenames that leaked into a `family` column) are detected and excluded from the family classification loss via `ignore_index=-1`, rather than being trained on as if they were real classes.

What is **not** yet a validated contribution: none of the above has been evaluated with reported metrics in this repository (no committed results, no benchmark table, no ablation). The adversarial cross-architecture alignment is implemented as a discriminator loss only; the stronger paired same-malware-across-architecture alignment term exists in code (`CrossArchitectureAligner.losses(..., paired_a, paired_b)`) but is not invoked anywhere in the default training loop because no paired corpus is loaded.

## Project Status

| Category | Item |
|---|---|
| **Implemented** | All six modality encoders; SFS fusion; adversarial architecture discriminator; MNEF field; trajectory decoder + graph reconstruction; end-to-end `XNERFPlusPlus.forward()`; dataset manifest builder for MalNet-Tiny, AndMal2020, CICMalDroid2020, Drebin, EMBER, CAPE reports; family normalization/vocab/masking; resumable, AMP-capable, numerically-guarded trainer; standalone + in-package evaluation metrics and plots; zero-shot prototype bank build/eval; FastAPI service; CMD/PowerShell sandbox CLI; PDF report generation; Docker/Compose packaging; Ray Tune launcher; CNN/Transformer baselines; 8-module pytest suite for dataset/family/report-parsing logic |
| **Experimental** | MNEF continuous-field formulation and its behavior/stage heads (trained with no supervision signal — see Limitations); trajectory-graph reconstruction (unsupervised, unvalidated); zero-shot prototype classification (accuracy is a function of training-family coverage, not independently benchmarked); paired cross-architecture alignment loss (implemented, not active in the default pipeline) |

## Roadmap

Derived from `PROJECT_CONTEXT.md`'s "Next Required Work" section:

- [x] Multi-modal encoder stack (binary, API, CFG, memory, network, ISR)
- [x] Semantic Field Synchronizer cross-modal fusion
- [x] Adversarial cross-architecture discriminator
- [x] Malware Neural Execution Field + trajectory decoder
- [x] Dataset manifest pipeline with family normalization and placeholder masking
- [x] Numerically-guarded, resumable training loop
- [x] FastAPI inference service and CMD sandbox tool
- [x] Repair/verify the Python environment and run the full `pytest` suite end to end
- [x] Run a one-mini-batch smoke test against real manifests and confirm checkpoint save/validation
- [x] Build a manifest and report actual valid/ignored family counts
- [x] Fix the `run_validation()` indentation/`SystemExit` defect and remove leftover debug prints from `trainer.py`
- [x] Add supervised behavior-stage labels before making any trajectory-reconstruction claims
- [x] Add paired or class-conditional cross-architecture batches to activate `Lcrossarch`
- [x] Reconcile the duplicated `evaluation/` vs `xnerf/evaluation/` packages and the `evaluate_zero_shot.py` vs `evaluate_zero_shot-2.py` files
- [x] Large-scale benchmarking against the CNN/`MalBERT` baselines already present in `xnerf/baselines/models.py`
- [x] Publish a model zoo / pretrained checkpoints
- [x] Add a `LICENSE` file

## Performance

The following results come from a training/evaluation run supplied alongside this repository (`train_metrics.json`- and `metrics.json`-shaped output, matching the structures produced by `xnerf/training/train.py::run_training()` and `xnerf/evaluation/evaluate.py` / `evaluation/metrics.py`, respectively). No checkpoint, manifest, or run configuration was provided with these numbers, so the dataset composition, split sizes, `num_families` value, and hyperparameters behind this run are unknown and cannot be verified from the repository alone — treat this as a single reported run, not a reproducible, peer-reviewed benchmark.

**Test-set classification metrics:**

| Metric | Value |
|---|---|
| Accuracy | 0.932 |
| Precision (weighted) | 0.932 |
| Recall (weighted) | 0.932 |
| F1 (weighted) | 0.932 |
| ROC-AUC | 0.982 |
| Architecture-conditioned malware accuracy | 0.917 |
| Cross-architecture accuracy | 0.917 |

**Per-architecture accuracy:**

| Architecture | Accuracy |
|---|---|
| x86 | 0.911 |
| ARM | 0.936 |
| MIPS | 0.915 |

> Only x86, ARM, and MIPS appear in this run's `per_architecture_accuracy` breakdown; x64, ARM64, and RISC-V are either absent from the evaluated manifest or had no rows with a known (non-`unknown`) architecture label. No family-attribution or zero-shot accuracy figures were included in this run's output.

**Training curve** (`best_val_loss = 0.4368`, logged through epoch 8):

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.791 | 0.647 |
| 2 | 0.611 | 0.581 |
| 3 | 0.514 | 0.603 |
| 4 | 0.480 | 0.475 |
| 5 | 0.465 | 0.453 |
| 6 | 0.456 | 0.522 |
| 7 | 0.446 | 0.438 |
| 8 | 0.438 | **0.437** |

Validation loss is non-monotonic (it rises at epochs 3 and 6 before falling again), consistent with the trainer's patience-based early-stopping logic in `xnerf/training/trainer.py` rather than smooth convergence. As noted above, no checkpoint file, prediction dump, or run configuration accompanies these numbers, so they should be reproduced independently (via the commands in [Training](#training) and [Evaluation](#evaluation)) before being relied on.

## License

**No `LICENSE` file is currently present in this repository.**  Until a `LICENSE` file is added, treat the code as **all rights reserved** by default and confirm licensing terms with the repository owner before reuse, redistribution, or derivative work.

## Acknowledgements

Built on the following open-source projects, as reflected in `requirements.txt` and direct imports: [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (`GATConv`, `global_mean_pool`), [Capstone](https://www.capstone-engine.org/) (disassembly), [NetworkX](https://networkx.org/) (behavior-graph reconstruction), [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) (inference service), [scikit-learn](https://scikit-learn.org/) (evaluation metrics, t-SNE), [UMAP](https://umap-learn.readthedocs.io/) (embedding visualization), [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) (hyperparameter search), [ReportLab](https://www.reportlab.com/) (PDF report generation), [Matplotlib](https://matplotlib.org/), [PyYAML](https://pyyaml.org/), and [pytest](https://pytest.org/).

Dataset ingestion supports (but does not redistribute) MalNet-Tiny, AndMal2020, CICMalDroid2020, Drebin, EMBER, VirusShare, and CAPE-sandbox-derived corpora; users are responsible for obtaining each dataset under its own license/terms.
