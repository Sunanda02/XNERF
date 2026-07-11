from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from xnerf.renderer.trajectory_decoder import STAGES


class ReportGenerator:
    """Generate analyst-facing explainability reports.

    Inputs:
        model outputs, family names, optional trajectory graph.
    Outputs:
        dict summary and PDF file.
    Usage:
        ReportGenerator(["benign","malware"], families).generate_pdf(summary, "report.pdf")
    """

    def __init__(self, class_names: list[str] | None = None, family_names: list[str] | None = None):
        self.class_names = class_names or ["benign", "malware"]
        self.family_names = family_names or ["unknown"]

    def summarize(self, outputs: dict[str, torch.Tensor], graph: nx.DiGraph | None = None) -> dict[str, Any]:
        malware_prob = torch.softmax(outputs["malware_logits"], dim=-1)[0, -1].item()
        family_idx = int(torch.softmax(outputs["family_logits"], dim=-1)[0].argmax().item())
        stage_ids = outputs["stage_logits"][0].argmax(dim=-1).detach().cpu().tolist()
        stages = [STAGES[i] for i in stage_ids]
        return {
            "malware_probability": malware_prob,
            "family": self.family_names[family_idx] if family_idx < len(self.family_names) else str(family_idx),
            "behavior_summary": " -> ".join(dict.fromkeys(stages)),
            "trajectory_nodes": graph.number_of_nodes() if graph else len(stages),
            "trajectory_edges": graph.number_of_edges() if graph else max(0, len(stages) - 1),
            "zero_shot_relation": "embedding available for nearest-neighbor or text-prototype comparison",
        }

    def generate_pdf(self, summary: dict[str, Any], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        c = canvas.Canvas(str(output_path), pagesize=letter)
        y = 750
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, y, "X-NERF++ Malware Intelligence Report")
        c.setFont("Helvetica", 10)
        for key, value in summary.items():
            y -= 28
            c.drawString(72, y, f"{key}: {value}")
        c.save()
        return output_path

