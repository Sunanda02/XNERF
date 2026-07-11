# XNERF Terminal Sandbox

Run single-file inference from CMD or PowerShell after a trained checkpoint is available.

```cmd
cd D:\GITrepos\XNERF
python sandbox\sandbox.py path\to\sample.exe
```

Checkpoint selection needs no code change. Use one of:

```cmd
python sandbox\sandbox.py path\to\sample.exe --checkpoint checkpoints\best_model.pt
set XNERF_SANDBOX_CHECKPOINT=checkpoints\best_model.pt
python sandbox\sandbox.py path\to\sample.exe
```

The sandbox reads `config.yaml` by default. Optional config block:

```yaml
sandbox:
  checkpoint: checkpoints/best_model.pt
  decision_threshold: 0.5
  arch: x86
  device:
```

The terminal report includes file name, malware probability, decision, predicted family, predicted architecture, confidence score, and inference time.

