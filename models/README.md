# Local Trained Models

After training on Kaggle, download:

```text
/kaggle/working/export/xnerf_local_inference.pt
```

Place it here:

```text
models/xnerf_local_inference.pt
```

Then run local inference/API with:

```powershell
$env:XNERF_CHECKPOINT="models/xnerf_local_inference.pt"
uvicorn xnerf.api.app:app --host 127.0.0.1 --port 8000
```

