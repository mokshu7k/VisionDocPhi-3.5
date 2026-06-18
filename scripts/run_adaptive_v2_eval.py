#!/usr/bin/env python
"""Run ocr_adaptive v2 eval and compare against baseline_200 (Kaggle workflow)."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    py = sys.executable
    scripts = PROJECT_ROOT / "scripts"

    print("Step 1: Run ocr_adaptive evaluation to ocr_adaptive_200_v2/")
    subprocess.run(
        [
            py,
            str(scripts / "chunked_evaluation.py"),
            "--mode",
            "ocr_adaptive",
            "--version",
            "v2",
            "--no-resume",
        ],
        check=False,
    )

    print("\nStep 2: Compare baseline vs adaptive v2")
    subprocess.run([py, str(scripts / "compare_baselines.py")], check=False)

    print("\nStep 3: Ablation report (baseline vs adaptive v2)")
    subprocess.run(
        [
            py,
            str(scripts / "ablation_report.py"),
            "--adaptive-version",
            "v2",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
