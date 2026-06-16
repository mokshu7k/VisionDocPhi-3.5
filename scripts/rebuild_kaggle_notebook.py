"""Rebuild visiondocphi-3-5.ipynb for Kaggle T4 Phase B workflow."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "visiondocphi-3-5.ipynb"

cells = []

def md(text: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def code(text: str):
    cells.append({"cell_type": "code", "metadata": {}, "source": [text], "outputs": [], "execution_count": None})

md("# VisionDocPhi-3.5 — Phase B Agentic OCR Evaluation (Kaggle T4)\n\n**Modes:** Baseline (`vision_only`) vs Adaptive (`ocr_adaptive`) on 200 stratified samples.")

code(
"""import os
import sys

PROJECT_NAME = "VisionDocPhi-3.5"
GITHUB_REPO = "https://github.com/mokshu7k/VisionDocPhi-3.5.git"

# Kaggle working directory (fallback to /content for Colab)
if os.path.exists("/kaggle/working"):
    PROJECT_PATH = os.path.join("/kaggle/working", PROJECT_NAME)
else:
    PROJECT_PATH = os.path.join("/content", PROJECT_NAME)

if not os.path.exists(PROJECT_PATH):
  print("Cloning repository...")
  parent = os.path.dirname(PROJECT_PATH)
  os.makedirs(parent, exist_ok=True)
  os.chdir(parent)
  os.system(f"git clone {GITHUB_REPO} {PROJECT_NAME}")

os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)
os.environ["DEVICE"] = "cuda"
os.environ["USE_8BIT_QUANTIZATION"] = "false"
print(f"Project path: {PROJECT_PATH}")

# Pull latest if repo already cloned
if os.path.exists(os.path.join(PROJECT_PATH, ".git")):
    os.system("git pull --rebase || git pull")
"""
)

md("## Install Phase B dependencies")

code(
"""!pip install -q sentence-transformers opencv-python-headless
"""
)

md("## GPU check")

code(
"""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
"""
)

md("## Build eval subset (200 samples: 100 layout + 100 visual)")

code(
"""import os
from pathlib import Path

subset_path = Path("data/outputs/eval_subset_200.json")
if subset_path.exists():
    print(f"Subset exists: {subset_path}")
else:
    !python scripts/build_eval_subset.py
"""
)

md("## Baseline evaluation (`vision_only`)")

code(
"""!python scripts/chunked_evaluation.py --mode baseline --subset data/outputs/eval_subset_200.json --chunk-size 20 --resume
"""
)

md("## Adaptive evaluation (`ocr_adaptive`)")

code(
"""!python scripts/chunked_evaluation.py --mode ocr_adaptive --subset data/outputs/eval_subset_200.json --chunk-size 20 --resume
"""
)

md("## Compare baseline vs adaptive")

code(
"""!python scripts/compare_baselines.py
"""
)

md("## Git backup chunk outputs")

code(
"""import os
os.system("git config user.email 'mokshu7k@users.noreply.github.com'")
os.system("git config user.name 'mokshu7k'")
os.system("git add data/outputs/eval_subset_200.json")
os.system("git add data/outputs/baseline_200/chunks data/outputs/baseline_200/chunked_progress.json")
os.system("git add data/outputs/ocr_adaptive_200/chunks data/outputs/ocr_adaptive_200/chunked_progress.json")
os.system("git add data/outputs/comparisons/")
os.system("git status")
os.system("git commit -m 'Phase B chunk backup' || true")
# Uncomment and set token to push from Kaggle:
# os.system("git push origin main")
"""
)

md(
"""## Notes

- Phi-3.5 float16 ~8.5 GB VRAM with `use_cache=False` — fits T4 15 GB
- Embeddings + OpenCV run on CPU — no extra GPU memory
- Each chunk ~20 samples (~35 min at ~100 s/sample)
- Use `--resume` across Kaggle sessions
- Archive legacy `chunks_val/` before first run — see `docs/ARCHIVE_LEGACY_CHUNKS.md`
"""
)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "dockerImageVersionId": 31328,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "cells": cells,
}

NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Written {NOTEBOOK} with {len(cells)} cells")
