#!/usr/bin/env python
"""Run the DocVQA evaluation harness (offline by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BATCH_SIZE,
    EVAL_SUBSET_FILE,
    IMAGES_DIR,
    NUM_WORKERS,
    OUTPUT_DIR,
    VAL_ANNOTATIONS,
)
from src.evals.harness import EvalHarness, HARNESS_DIR
from src.utils.metrics import latency_percentiles


def run_latency_smoke(n: int, adaptive_version: str = "v2") -> dict:
    """
    Time N stratified subset samples for baseline and OCR-adaptive modes.

    Requires local dataset + GPU/CPU model load. Raises on missing data.
    """
    from src.data.dataset import (
        DocVQADataset,
        create_subset_dataloader,
        load_subset_metadata,
    )
    from src.models.inference import DocVQAInference, InferenceMode

    if n < 1:
        raise ValueError("--latency-smoke N must be >= 1")

    subset_path = Path(EVAL_SUBSET_FILE)
    if not subset_path.exists():
        raise FileNotFoundError(
            f"Eval subset not found: {subset_path}. "
            "Run scripts/build_eval_subset.py first."
        )
    if not Path(VAL_ANNOTATIONS).exists() or not Path(IMAGES_DIR).exists():
        raise FileNotFoundError(
            "SP-DocVQA annotations/images missing; cannot run latency smoke."
        )

    question_ids, cohort_map = load_subset_metadata(str(subset_path))
    # Stratified-ish: take from front of subset (already 100 layout + 100 visual)
    take = min(n, len(question_ids))
    qids = question_ids[:take]
    cohort = {qid: cohort_map[qid] for qid in qids if qid in cohort_map}

    dataset = DocVQADataset(
        str(VAL_ANNOTATIONS),
        str(IMAGES_DIR),
        split="val",
        question_ids=qids,
        cohort_map=cohort,
    )
    loader = create_subset_dataloader(
        dataset, 0, len(dataset), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    smoke: dict = {"n": take, "modes": {}}

    for mode in (InferenceMode.VISION_ONLY, InferenceMode.OCR_ADAPTIVE):
        print(f"\n⏱ Latency smoke: mode={mode}, n={take}")
        inference = DocVQAInference(mode=mode)
        metrics = inference.evaluate(loader, num_samples=take, mode=mode)
        rows = metrics.get("results", [])
        total = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
        prep = [float(r["ocr_prep_ms"]) for r in rows if r.get("ocr_prep_ms") is not None]
        key = "baseline" if mode == InferenceMode.VISION_ONLY else "adaptive"
        smoke[key] = latency_percentiles(total)
        smoke[key]["n"] = len(total)
        if prep:
            smoke[f"{key}_ocr_prep"] = latency_percentiles(prep)
            smoke[f"{key}_ocr_prep"]["n"] = len(prep)

        # Persist smoke rows for inspection
        out = HARNESS_DIR / f"latency_smoke_{key}.json"
        HARNESS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"mode": mode, "results": rows}, f, indent=2)
        print(f"Wrote {out}")

        del inference

    smoke["adaptive_version_note"] = adaptive_version
    return smoke


def main():
    parser = argparse.ArgumentParser(
        description="DocVQA eval harness: taxonomy + ANLS/EM + optional latency smoke"
    )
    parser.add_argument("--baseline-version", type=str, default="")
    parser.add_argument("--adaptive-version", type=str, default="v2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify a few samples and exit (no report write)",
    )
    parser.add_argument(
        "--latency-smoke",
        type=int,
        default=0,
        metavar="N",
        help="Optionally time N samples through baseline + adaptive (needs data/GPU)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=f"Report directory (default: {HARNESS_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else HARNESS_DIR
    harness = EvalHarness(
        baseline_version=args.baseline_version,
        adaptive_version=args.adaptive_version,
        output_dir=out_dir,
    )

    if args.dry_run:
        result = harness.dry_run()
        print(json.dumps(result, indent=2))
        print("Dry run OK")
        return

    latency_smoke = None
    if args.latency_smoke:
        try:
            latency_smoke = run_latency_smoke(
                args.latency_smoke,
                adaptive_version=args.adaptive_version,
            )
        except Exception as exc:
            print(f"Latency smoke skipped: {exc}")
            latency_smoke = {
                "error": str(exc),
                "n": args.latency_smoke,
                "note": (
                    "Offline harness still runs. Provide SP-DocVQA data and a "
                    "working model runtime to collect latency smoke timings."
                ),
            }

    report = harness.run(latency_smoke=latency_smoke, write=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"

    o = report["overall"]
    print(f"Report written to {json_path}")
    print(f"Markdown written to {md_path}")
    print(
        f"\nOverall: baseline_anls={o['baseline_mean_anls']:.4f}, "
        f"adaptive_anls={o['adaptive_mean_anls']:.4f}, "
        f"delta={o['delta_anls']:+.4f}"
    )
    print(
        f"EM: baseline={o['baseline_exact_match']:.4f}, "
        f"adaptive={o['adaptive_exact_match']:.4f}"
    )
    adaptive_tax = report["error_taxonomy"]["adaptive"]
    print("\nAdaptive error taxonomy:")
    for label, stats in adaptive_tax.items():
        print(f"  {label}: n={stats['count']} ({stats['rate']:.1%})")


if __name__ == "__main__":
    main()
