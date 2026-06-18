#!/usr/bin/env python
"""
Run chunked evaluation on the 200-sample eval subset.

Usage:
    python scripts/chunked_evaluation.py --mode baseline --resume
    python scripts/chunked_evaluation.py --mode ocr_adaptive --resume
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EVAL_SUBSET_FILE, CHUNK_SIZE
from src.pipelines.chunked_evaluation import run_chunked_evaluation


def main():
    parser = argparse.ArgumentParser(description="Chunked DocVQA evaluation (baseline vs ocr_adaptive)")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "vision_only", "ocr_adaptive", "adaptive"],
        default="baseline",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=str(EVAL_SUBSET_FILE),
        help="Path to eval subset JSON",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Samples per chunk",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Output dir suffix (e.g. v2 -> ocr_adaptive_200_v2)",
    )

    args = parser.parse_args()
    resume = not args.no_resume

    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║        CHUNKED DOCVQA EVALUATION — Phase B Agentic Pipeline        ║
╚════════════════════════════════════════════════════════════════════╝

  Mode:       {args.mode}
  Subset:     {args.subset}
  Chunk size: {args.chunk_size}
  Resume:     {resume}
    """)

    results = run_chunked_evaluation(
        mode=args.mode,
        subset_file=args.subset,
        chunk_size=args.chunk_size,
        split=args.split,
        resume=resume,
        version=args.version,
    )

    if results:
        print("\n✅ CHUNKED EVALUATION COMPLETE!")
        for key, value in results.items():
            if key not in ("results", "predictions", "ground_truths") and isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
