from __future__ import annotations

import os
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from xnerf.datasets.loaders import MalwareManifestDataset
from xnerf.explainability.report_generator import ReportGenerator
from xnerf.model import XNERFPlusPlus
from xnerf.preprocessing.pipeline import ArchitectureNormalizationPipeline

app = FastAPI(title="X-NERF++ Malware Intelligence API")
UPLOAD_DIR = Path("runs/api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS: dict[str, dict] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_READY = False


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def load_model_from_env() -> XNERFPlusPlus:
    global MODEL_READY
    checkpoint = os.getenv("XNERF_CHECKPOINT", "models/xnerf_local_inference.pt")
    if checkpoint and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=DEVICE)
        model_config = payload.get("model_config", {})
        model = XNERFPlusPlus(
            num_classes=int(model_config.get("num_classes", 2)),
            num_families=int(model_config.get("num_families", 32)),
        ).to(DEVICE)
        state = payload.get("state_dict", payload.get("model", payload))
        model.load_state_dict(_strip_module_prefix(state), strict=False)
        MODEL_READY = True
        return model.eval()
    MODEL_READY = False
    return XNERFPlusPlus().to(DEVICE).eval()


MODEL = load_model_from_env()
REPORTER = ReportGenerator()


class AnalyzeRequest(BaseModel):
    upload_id: str
    arch: str = "x86"


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    path.write_bytes(await file.read())
    RESULTS[upload_id] = {"status": "uploaded", "path": str(path)}
    return {"upload_id": upload_id, "status": "uploaded"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="trained checkpoint missing; set XNERF_CHECKPOINT or place models/xnerf_local_inference.pt")
    if req.upload_id not in RESULTS:
        return {"error": "unknown upload_id"}
    path = Path(RESULTS[req.upload_id]["path"])
    normalizer = ArchitectureNormalizationPipeline(arch=req.arch)
    isr = normalizer.process({"bytes": path.read_bytes(), "arch": req.arch})
    row = {"path": str(path), "label": 0, "family": "unknown", "arch": req.arch}
    tmp_manifest = UPLOAD_DIR / f"{req.upload_id}.jsonl"
    tmp_manifest.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
    ds = MalwareManifestDataset(tmp_manifest)
    batch = ds[0]
    batch["isr"] = isr
    tensor_batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = MODEL(tensor_batch)
    graph = MODEL.renderer.reconstruct_graphs(outputs["stage_logits"])[0]
    summary = REPORTER.summarize(outputs, graph)
    report_path = REPORTER.generate_pdf(summary, Path("runs/api/reports") / f"{req.upload_id}.pdf")
    RESULTS[req.upload_id] = {"status": "done", "summary": summary, "report": str(report_path)}
    return {"upload_id": req.upload_id, **RESULTS[req.upload_id]}


@app.get("/result/{job_id}")
def result(job_id: str):
    return RESULTS.get(job_id, {"error": "unknown id"})


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_READY, "device": str(DEVICE)}
