#!/usr/bin/env python
"""Ablation report: baseline vs adaptive slices, damage and hijacking cases."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import OUTPUT_DIR, get_mode_output_dir

HIJACK_RE = re.compile(r"#|INSTRUCTION|User:", re.IGNORECASE)


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


def snippet_preview(row: dict, max_len: int = 120) -> str:
    fmt = row.get("formatting", {})
    retrieval = row.get("retrieval", {})
    ids = retrieval.get("expanded_line_ids", retrieval.get("line_ids", []))
    preview = f"line_ids={ids}"
    if len(preview) > max_len:
        return preview[:max_len] + "..."
    return preview


def is_hijacking_case(row: dict) -> bool:
    pred = row.get("predicted_answer", "")
    if not HIJACK_RE.search(pred):
        return False
    gt_list = row.get("ground_truth_answers", [])
    if not gt_list:
        return False
    from src.utils.metrics import anls_score
    return anls_score([pred], gt_list) < 1.0


def build_case_list(
    baseline_rows: list,
    adaptive_rows: list,
    filter_fn,
    limit: int = 20,
) -> list:
    b_map = {r["question_id"]: r for r in baseline_rows}
    cases = []
    for row in adaptive_rows:
        if not filter_fn(row, b_map.get(row["question_id"])):
            continue
        b_row = b_map.get(row["question_id"], {})
        cases.append({
            "question_id": row["question_id"],
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth_answers", []),
            "baseline_pred": b_row.get("predicted_answer", ""),
            "adaptive_pred": row.get("predicted_answer", ""),
            "baseline_anls": b_row.get("anls_score", 0.0),
            "adaptive_anls": row.get("anls_score", 0.0),
            "used_ocr": row.get("used_ocr", False),
            "routing_reason": (row.get("routing") or {}).get("reason", ""),
            "snippet_preview": snippet_preview(row),
        })
    cases.sort(key=lambda c: c["baseline_anls"] - c["adaptive_anls"], reverse=True)
    return cases[:limit]


def main():
    parser = argparse.ArgumentParser(description="Ablation report for baseline vs adaptive")
    parser.add_argument("--baseline-version", type=str, default="")
    parser.add_argument("--adaptive-version", type=str, default="v2")
    args = parser.parse_args()

    baseline_dir = get_mode_output_dir("baseline", version=args.baseline_version)
    adaptive_dir = get_mode_output_dir("ocr_adaptive", version=args.adaptive_version)

    baseline_data = load_merged(baseline_dir)
    adaptive_data = load_merged(adaptive_dir)

    baseline_rows = baseline_data.get("results", [])
    adaptive_rows = adaptive_data.get("results", [])

    used_ocr_rows = [r for r in adaptive_rows if r.get("used_ocr")]
    gate_rows = [
        r for r in adaptive_rows
        if (r.get("routing") or {}).get("reason") == "low_retrieval_confidence"
    ]

    damage_cases = build_case_list(
        baseline_rows,
        adaptive_rows,
        lambda a, b: (
            a.get("used_ocr")
            and b is not None
            and b.get("anls_score", 0.0) > a.get("anls_score", 0.0)
        ),
    )

    hijack_cases = build_case_list(
        baseline_rows,
        adaptive_rows,
        lambda a, b: is_hijacking_case(a),
    )

    report = {
        "overall": {
            "baseline_mean_anls": mean_anls(baseline_rows),
            "adaptive_mean_anls": mean_anls(adaptive_rows),
            "delta": mean_anls(adaptive_rows) - mean_anls(baseline_rows),
            "sample_count": len(baseline_rows),
        },
        "used_ocr_subset": {
            "count": len(used_ocr_rows),
            "mean_anls": mean_anls(used_ocr_rows),
        },
        "low_retrieval_confidence_gate": {
            "count": len(gate_rows),
            "mean_anls": mean_anls(gate_rows),
        },
        "by_cohort": {
            "baseline": defaultdict(list),
            "adaptive": defaultdict(list),
        },
        "damage_cases": damage_cases,
        "hijacking_cases": hijack_cases,
    }

    for row in baseline_rows:
        report["by_cohort"]["baseline"][row.get("cohort", "unknown")].append(
            row.get("anls_score", 0.0)
        )
    for row in adaptive_rows:
        report["by_cohort"]["adaptive"][row.get("cohort", "unknown")].append(
            row.get("anls_score", 0.0)
        )

    report["by_cohort"] = {
        "baseline": {
            k: sum(v) / len(v) if v else 0.0
            for k, v in report["by_cohort"]["baseline"].items()
        },
        "adaptive": {
            k: sum(v) / len(v) if v else 0.0
            for k, v in report["by_cohort"]["adaptive"].items()
        },
    }

    out_dir = OUTPUT_DIR / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_report.json"
    md_path = out_dir / "ablation_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Ablation Report",
        "",
        f"- Baseline mean ANLS: **{report['overall']['baseline_mean_anls']:.4f}**",
        f"- Adaptive mean ANLS: **{report['overall']['adaptive_mean_anls']:.4f}**",
        f"- used_ocr=true subset: n={report['used_ocr_subset']['count']}, "
        f"mean={report['used_ocr_subset']['mean_anls']:.4f}",
        f"- low_retrieval_confidence gate: n={report['low_retrieval_confidence_gate']['count']}",
        "",
        "## Damage cases (top 20)",
    ]
    for case in damage_cases:
        md_lines.append(
            f"- qid={case['question_id']}: baseline={case['baseline_anls']:.2f} "
            f"adaptive={case['adaptive_anls']:.2f} | {case['question'][:60]}"
        )

    md_lines.append("")
    md_lines.append("## Hijacking cases (top 20)")
    for case in hijack_cases:
        md_lines.append(
            f"- qid={case['question_id']}: pred={case['adaptive_pred']!r} "
            f"gt={case['ground_truth']}"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Report written to {json_path}")
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
