#!/usr/bin/env python
"""Compare baseline vs ocr_adaptive ANLS on the 200-sample eval subset."""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import OUTPUT_DIR, get_mode_output_dir


def load_merged(mode_dir: Path) -> dict:
    path = mode_dir / "results_merged.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing merged results: {path}")
    with open(path, "r") as f:
        return json.load(f)


def mean_anls(rows: list) -> float:
    if not rows:
        return 0.0
    return sum(r.get("anls_score", 0.0) for r in rows) / len(rows)


def group_mean(rows: list, key_fn) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r.get("anls_score", 0.0))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in groups.items()}


def compare_samples(baseline_rows: list, adaptive_rows: list) -> dict:
    b_map = {r["question_id"]: r for r in baseline_rows}
    a_map = {r["question_id"]: r for r in adaptive_rows}
    wins = {"baseline": 0, "adaptive": 0, "tie": 0}
    for qid in b_map:
        if qid not in a_map:
            continue
        b_score = b_map[qid].get("anls_score", 0.0)
        a_score = a_map[qid].get("anls_score", 0.0)
        if abs(b_score - a_score) < 1e-6:
            wins["tie"] += 1
        elif a_score > b_score:
            wins["adaptive"] += 1
        else:
            wins["baseline"] += 1
    return wins


def routing_breakdown(adaptive_rows: list) -> dict:
    groups = defaultdict(list)
    for r in adaptive_rows:
        routing = r.get("routing", {})
        reason = routing.get("reason", "unknown")
        groups[reason].append(r.get("anls_score", 0.0))
    return {
        reason: {
            "count": len(scores),
            "mean_anls": sum(scores) / len(scores) if scores else 0.0,
        }
        for reason, scores in groups.items()
    }


def main():
    baseline_dir = get_mode_output_dir("baseline")
    adaptive_dir = get_mode_output_dir("ocr_adaptive")

    baseline_data = load_merged(baseline_dir)
    adaptive_data = load_merged(adaptive_dir)

    baseline_rows = baseline_data.get("results", [])
    adaptive_rows = adaptive_data.get("results", [])

    report = {
        "overall": {
            "baseline_mean_anls": mean_anls(baseline_rows),
            "adaptive_mean_anls": mean_anls(adaptive_rows),
            "delta": mean_anls(adaptive_rows) - mean_anls(baseline_rows),
            "sample_count": len(baseline_rows),
        },
        "by_cohort": {
            "baseline": group_mean(baseline_rows, lambda r: r.get("cohort", "unknown")),
            "adaptive": group_mean(adaptive_rows, lambda r: r.get("cohort", "unknown")),
        },
        "by_question_type": {
            "baseline": group_mean(
                baseline_rows,
                lambda r: r.get("question_types", ["unknown"])[0] if r.get("question_types") else "unknown",
            ),
            "adaptive": group_mean(
                adaptive_rows,
                lambda r: r.get("question_types", ["unknown"])[0] if r.get("question_types") else "unknown",
            ),
        },
        "wins": compare_samples(baseline_rows, adaptive_rows),
        "routing_breakdown": routing_breakdown(adaptive_rows),
    }

    comparisons_dir = OUTPUT_DIR / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    json_path = comparisons_dir / "baseline_vs_adaptive_report.json"
    md_path = comparisons_dir / "baseline_vs_adaptive_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Baseline vs OCR-Adaptive Comparison",
        "",
        f"- Baseline mean ANLS: **{report['overall']['baseline_mean_anls']:.4f}**",
        f"- Adaptive mean ANLS: **{report['overall']['adaptive_mean_anls']:.4f}**",
        f"- Delta (adaptive - baseline): **{report['overall']['delta']:+.4f}**",
        f"- Samples: {report['overall']['sample_count']}",
        "",
        "## Wins per sample",
        f"- Adaptive wins: {report['wins']['adaptive']}",
        f"- Baseline wins: {report['wins']['baseline']}",
        f"- Ties: {report['wins']['tie']}",
        "",
        "## By cohort",
    ]
    for cohort in sorted(set(report["by_cohort"]["baseline"].keys()) | set(report["by_cohort"]["adaptive"].keys())):
        b = report["by_cohort"]["baseline"].get(cohort, 0.0)
        a = report["by_cohort"]["adaptive"].get(cohort, 0.0)
        md_lines.append(f"- {cohort}: baseline={b:.4f}, adaptive={a:.4f}, delta={a - b:+.4f}")

    md_lines.append("")
    md_lines.append("## Routing breakdown (adaptive)")
    for reason, stats in report["routing_breakdown"].items():
        md_lines.append(f"- {reason}: n={stats['count']}, mean_anls={stats['mean_anls']:.4f}")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"✓ Report written to {json_path}")
    print(f"✓ Markdown written to {md_path}")
    print(f"\nOverall: baseline={report['overall']['baseline_mean_anls']:.4f}, "
          f"adaptive={report['overall']['adaptive_mean_anls']:.4f}, "
          f"delta={report['overall']['delta']:+.4f}")


if __name__ == "__main__":
    main()
