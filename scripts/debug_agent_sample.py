#!/usr/bin/env python
"""Debug a single eval-subset sample through the agentic pipeline."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EVAL_SUBSET_FILE, IMAGES_DIR, VAL_ANNOTATIONS
from src.agents.pipeline import AgenticDocVQAPipeline
from src.data.dataset import DocVQADataset, load_subset_metadata
from src.models.inference import DocVQAInference, InferenceMode


def _find_sample(subset_path: Path, question_id: int | None, index: int | None) -> dict:
    with open(subset_path, "r") as f:
        subset = json.load(f)
    entries = subset.get("entries", subset.get("samples", []))
    if question_id is not None:
        for entry in entries:
            if entry.get("question_id") == question_id:
                return entry
        raise ValueError(f"question_id {question_id} not found in subset")
    if index is not None:
        if index < 0 or index >= len(entries):
            raise ValueError(f"index {index} out of range (0-{len(entries) - 1})")
        return entries[index]
    raise ValueError("Provide --question-id or --index")


def _print_section(title: str, body: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(body)


def main():
    parser = argparse.ArgumentParser(description="Debug agent pipeline for one eval sample")
    parser.add_argument("--question-id", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--subset", type=str, default=str(EVAL_SUBSET_FILE))
    parser.add_argument("--run-vlm", action="store_true", help="Run VLM inference (requires GPU)")
    args = parser.parse_args()

    subset_path = Path(args.subset)
    entry = _find_sample(subset_path, args.question_id, args.index)
    qid = entry["question_id"]

    question_ids, cohort_map = load_subset_metadata(str(subset_path))
    dataset = DocVQADataset(
        str(VAL_ANNOTATIONS),
        str(IMAGES_DIR),
        split="val",
        question_ids=question_ids,
        cohort_map=cohort_map,
    )

    sample = None
    for i in range(len(dataset)):
        s = dataset[i]
        if s["question_id"] == qid:
            sample = s
            break
    if sample is None:
        raise RuntimeError(f"Sample {qid} not found in dataset")

    image = sample["image"]
    question = sample["question"]
    pipeline = AgenticDocVQAPipeline()
    result = pipeline.prepare(
        image=image,
        question=question,
        question_types=sample.get("question_types", []),
        ucsf_id=sample.get("ucsf_document_id", ""),
        page_no=str(sample.get("ucsf_document_page_no", "")),
    )

    inference = DocVQAInference(mode=InferenceMode.VISION_ONLY)
    baseline_prompt = inference._build_prompt(question)
    adaptive_prompt = inference._build_prompt(
        question,
        ocr_snippets=result.ocr_snippet if result.used_ocr else None,
    )

    routing_text = json.dumps(result.routing.to_dict() if result.routing else {}, indent=2)
    retrieval_text = json.dumps(result.retrieval or {}, indent=2)
    formatting_text = json.dumps(result.formatting or {}, indent=2)

    _print_section(f"Sample question_id={qid}", f"Question: {question}\nGT: {sample['answers']}")
    _print_section("Routing", routing_text)
    _print_section("Retrieval", retrieval_text)
    _print_section("Formatting", formatting_text)
    _print_section("OCR snippet (pre-XML)", result.ocr_snippet or "(none — vision_only)")

    _print_section("BASELINE PROMPT", baseline_prompt)
    _print_section("ADAPTIVE PROMPT", adaptive_prompt)

    if args.run_vlm:
        adaptive_inference = DocVQAInference(mode=InferenceMode.OCR_ADAPTIVE)
        baseline_answer = inference.generate_answer(image, question)
        adaptive_answer = adaptive_inference.generate_answer(
            image,
            question,
            ocr_snippets=result.ocr_snippet if result.used_ocr else None,
        )
        _print_section(
            "VLM answers",
            f"Baseline: {baseline_answer}\nAdaptive: {adaptive_answer}",
        )


if __name__ == "__main__":
    main()
