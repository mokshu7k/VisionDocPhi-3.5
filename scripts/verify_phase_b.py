#!/usr/bin/env python
"""Verify Phase B implementation against the plan checklist."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    EVAL_SUBSET_FILE,
    OCR_MAX_CHARS,
    VAL_ANNOTATIONS,
    IMAGES_DIR,
    CHUNK_SIZE,
)
from src.data.ocr_loader import load_ocr_lines
from src.agents.formatter_agent import LayoutFormatterAgent
from src.agents.retriever_agent import HybridRetrieverAgent
from src.agents.router_agent import RouterAgent


def check_subset():
    with open(EVAL_SUBSET_FILE) as f:
        data = json.load(f)
    ids = data["question_ids"]
    ok = len(ids) == 200 and len(set(ids)) == 200
    ok &= data["by_cohort"].get("layout_heavy") == 100
    ok &= data["by_cohort"].get("visual_heavy") == 100
    print(f"[{'OK' if ok else 'FAIL'}] eval_subset_200.json: {len(ids)} unique, cohorts={data['by_cohort']}")
    return ok, ids


def check_chunk_overlap(ids):
    # Simulate chunk indices without torch
    chunk0 = ids[0:CHUNK_SIZE]
    chunk1 = ids[CHUNK_SIZE:CHUNK_SIZE * 2]
    overlap = set(chunk0) & set(chunk1)
    ok = len(overlap) == 0 and len(chunk0) == CHUNK_SIZE
    print(f"[{'OK' if ok else 'FAIL'}] chunk_000 vs chunk_001 overlap: {len(overlap)}")
    return ok


def check_agents(ids):
    with open(VAL_ANNOTATIONS) as f:
        val = json.load(f)
    sample = next(s for s in val["data"] if s["questionId"] == ids[0])
    from PIL import Image

    img_path = Path(IMAGES_DIR) / sample["image"].replace("documents/", "")
    img = Image.open(img_path).convert("RGB")
    lines = load_ocr_lines(sample["ucsf_document_id"], sample["ucsf_document_page_no"])

    router = RouterAgent()
    routing = router.decide(
        sample["question"],
        sample.get("question_types", []),
        img,
        sample["ucsf_document_id"],
        sample["ucsf_document_page_no"],
    )
    print(f"[OK] Router: route={routing.route}, reason={routing.reason}")

    retriever = HybridRetrieverAgent()
    scored = retriever.retrieve(
        sample["question"],
        lines,
        ucsf_id=sample["ucsf_document_id"],
        page_no=sample["ucsf_document_page_no"],
    )
    has_scores = scored and all(
        hasattr(s, "dense_score") and hasattr(s, "sparse_score") and hasattr(s, "final_score")
        for s in scored
    )
    print(f"[{'OK' if has_scores else 'FAIL'}] Retriever: {len(scored)} lines with hybrid scores")

    if scored:
        fmt = LayoutFormatterAgent().format(scored[:min(5, len(scored))])
        order_changed = fmt.presentation_order != list(range(len(scored[:5])))
        under_budget = fmt.char_count <= OCR_MAX_CHARS
        print(f"[{'OK' if under_budget else 'FAIL'}] Formatter: chars={fmt.char_count} (max {OCR_MAX_CHARS})")
        print(f"[{'OK' if order_changed or len(scored) < 2 else 'WARN'}] Formatter layout reorder: {fmt.presentation_order}")
    return has_scores


def check_no_ocr_always():
    root = PROJECT_ROOT
    hits = []
    skip = {Path(__file__).resolve()}
    for path in root.rglob("*"):
        if path.resolve() in skip:
            continue
        if path.suffix not in (".py", ".ipynb"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "ocr_always" in text:
            hits.append(str(path))
    ok = len(hits) == 0
    print(f"[{'OK' if ok else 'FAIL'}] ocr_always absent from codebase")
    if hits:
        print("  found in:", hits)
    return ok


def main():
    print("Phase B verification\n" + "=" * 40)
    ok_subset, ids = check_subset()
    ok_chunks = check_chunk_overlap(ids)
    ok_agents = check_agents(ids)
    ok_modes = check_no_ocr_always()
    all_ok = ok_subset and ok_chunks and ok_agents and ok_modes
    print("=" * 40)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
