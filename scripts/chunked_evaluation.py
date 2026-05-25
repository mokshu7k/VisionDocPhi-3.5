#!/usr/bin/env python
"""
Script to run chunked evaluation on DocVQA dataset

Usage:
    python scripts/chunked_evaluation.py --split val --chunk_size 200 --resume
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.chunked_evaluation import run_chunked_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Run chunked evaluation on DocVQA dataset for memory efficiency"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Number of samples to process per chunk (default: 200)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh evaluation (don't resume from checkpoint)"
    )
    
    args = parser.parse_args()
    
    # Handle resume flag
    resume = not args.no_resume
    
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║           CHUNKED DOCVQA EVALUATION - GPU OPTIMIZED               ║
╚════════════════════════════════════════════════════════════════════╝

Configuration:
  • Split: {args.split.upper()}
  • Chunk Size: {args.chunk_size} samples
  • Resume: {'Yes' if resume else 'No'}

This approach:
  ✅ Processes data in manageable chunks
  ✅ Saves progress after each chunk
  ✅ Can be interrupted and resumed
  ✅ Works well with Colab's runtime limits
  ✅ Merges final results automatically
    """)
    
    # Run chunked evaluation
    results = run_chunked_evaluation(
        split=args.split,
        chunk_size=args.chunk_size,
        resume=resume
    )
    
    if results:
        print("\n" + "="*70)
        print("✅ CHUNKED EVALUATION COMPLETE!")
        print("="*70)
        print("\nFinal Metrics:")
        for key, value in results.items():
            if key != 'results' and isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print()
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
