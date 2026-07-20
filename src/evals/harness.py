"""
Production DocVQA evaluation harness.

Loads paired baseline vs OCR-adaptive merged results, applies error taxonomy,
aggregates ANLS/EM by cohort and route, and optionally reports latency.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import HARNESS_DIR, get_mode_output_dir
from src.evals.taxonomy import ERROR_LABELS, label_rows
from src.utils.metrics import exact_match, latency_percentiles


def load_merged(mode_dir: Path) -> dict:
    path = mode_dir / "results_merged.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing merged results: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_anls(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    return sum(r.get("anls_score", 0.0) for r in rows) / len(rows)


def mean_em(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    hits = 0
    for r in rows:
        pred = r.get("predicted_answer", "")
        gt = r.get("ground_truth_answers") or []
        if gt and exact_match([pred], gt):
            hits += 1
    return hits / len(rows)


def group_mean_anls(rows: Sequence[dict], key_fn) -> Dict[str, float]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r.get("anls_score", 0.0))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in groups.items()}


def compare_wins(baseline_rows: List[dict], adaptive_rows: List[dict]) -> dict:
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


def routing_breakdown(adaptive_rows: List[dict]) -> dict:
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in adaptive_rows:
        reason = (r.get("routing") or {}).get("reason", "unknown")
        groups[reason].append(r.get("anls_score", 0.0))
    return {
        reason: {
            "count": len(scores),
            "mean_anls": sum(scores) / len(scores) if scores else 0.0,
        }
        for reason, scores in groups.items()
    }


def _primary_question_type(row: dict) -> str:
    types = row.get("question_types") or []
    return types[0] if types else "unknown"


def _distribution(labels: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(labels)
    total = sum(counts.values()) or 1
    return {
        label: {
            "count": counts.get(label, 0),
            "rate": counts.get(label, 0) / total,
        }
        for label in ERROR_LABELS
    }


def _extract_latencies(rows: Sequence[dict]) -> Tuple[List[float], List[float]]:
    total_ms: List[float] = []
    prep_ms: List[float] = []
    for r in rows:
        if r.get("latency_ms") is not None:
            total_ms.append(float(r["latency_ms"]))
        if r.get("ocr_prep_ms") is not None:
            prep_ms.append(float(r["ocr_prep_ms"]))
    return total_ms, prep_ms


class EvalHarness:
    """Paired DocVQA evaluation harness over merged result files."""

    def __init__(
        self,
        baseline_version: str = "",
        adaptive_version: str = "v2",
        output_dir: Optional[Path] = None,
        anls_threshold: float = 0.5,
    ):
        self.baseline_version = baseline_version
        self.adaptive_version = adaptive_version
        self.output_dir = Path(output_dir) if output_dir else HARNESS_DIR
        self.anls_threshold = anls_threshold
        self.baseline_dir = get_mode_output_dir("baseline", version=baseline_version)
        self.adaptive_dir = get_mode_output_dir("ocr_adaptive", version=adaptive_version)

    def dry_run(self, max_samples: int = 5) -> dict:
        """CI-friendly smoke: classify a few rows without writing reports."""
        baseline_data = load_merged(self.baseline_dir)
        adaptive_data = load_merged(self.adaptive_dir)
        baseline_rows = baseline_data.get("results", [])[:max_samples]
        adaptive_rows = adaptive_data.get("results", [])[:max_samples]
        labeled = label_rows(adaptive_rows, anls_threshold=self.anls_threshold)
        return {
            "ok": True,
            "baseline_samples": len(baseline_rows),
            "adaptive_samples": len(adaptive_rows),
            "error_labels": [r["error_label"] for r in labeled],
            "error_distribution": _distribution([r["error_label"] for r in labeled]),
        }

    def run(
        self,
        latency_smoke: Optional[dict] = None,
        write: bool = True,
    ) -> dict:
        """
        Run full offline harness on merged results.

        Args:
            latency_smoke: Optional dict from run_latency_smoke (merged into report).
            write: Persist report.json / report.md under harness output dir.
        """
        baseline_data = load_merged(self.baseline_dir)
        adaptive_data = load_merged(self.adaptive_dir)
        baseline_rows = baseline_data.get("results", [])
        adaptive_rows = adaptive_data.get("results", [])

        b_labeled = label_rows(baseline_rows, anls_threshold=self.anls_threshold)
        a_labeled = label_rows(adaptive_rows, anls_threshold=self.anls_threshold)

        b_labels = [r["error_label"] for r in b_labeled]
        a_labels = [r["error_label"] for r in a_labeled]

        paired_shifts = self._paired_label_shifts(b_labeled, a_labeled)
        example_failures = self._example_failures(a_labeled, limit=10)

        b_lat, b_prep = _extract_latencies(baseline_rows)
        a_lat, a_prep = _extract_latencies(adaptive_rows)

        latency_section: Dict[str, Any] = {
            "available": bool(b_lat or a_lat or latency_smoke),
            "note": (
                "Latency fields absent from historical merged results. "
                "Re-run eval with instrumented inference, or use "
                "`python scripts/run_harness.py --latency-smoke N`."
            ),
            "from_merged": {
                "baseline": latency_percentiles(b_lat) if b_lat else None,
                "adaptive": latency_percentiles(a_lat) if a_lat else None,
                "baseline_ocr_prep": latency_percentiles(b_prep) if b_prep else None,
                "adaptive_ocr_prep": latency_percentiles(a_prep) if a_prep else None,
                "baseline_n": len(b_lat),
                "adaptive_n": len(a_lat),
            },
        }
        if latency_smoke:
            latency_section["available"] = True
            latency_section["smoke"] = latency_smoke
            latency_section["note"] = "Includes optional latency smoke timings."

        report = {
            "overall": {
                "baseline_mean_anls": mean_anls(baseline_rows),
                "adaptive_mean_anls": mean_anls(adaptive_rows),
                "delta_anls": mean_anls(adaptive_rows) - mean_anls(baseline_rows),
                "baseline_exact_match": mean_em(baseline_rows),
                "adaptive_exact_match": mean_em(adaptive_rows),
                "delta_em": mean_em(adaptive_rows) - mean_em(baseline_rows),
                "sample_count": len(baseline_rows),
                "adaptive_sample_count": len(adaptive_rows),
            },
            "by_cohort": {
                "baseline": {
                    "anls": group_mean_anls(baseline_rows, lambda r: r.get("cohort", "unknown")),
                    "em": self._group_em(baseline_rows, lambda r: r.get("cohort", "unknown")),
                },
                "adaptive": {
                    "anls": group_mean_anls(adaptive_rows, lambda r: r.get("cohort", "unknown")),
                    "em": self._group_em(adaptive_rows, lambda r: r.get("cohort", "unknown")),
                },
            },
            "by_question_type": {
                "baseline_anls": group_mean_anls(baseline_rows, _primary_question_type),
                "adaptive_anls": group_mean_anls(adaptive_rows, _primary_question_type),
            },
            "wins": compare_wins(baseline_rows, adaptive_rows),
            "routing_breakdown": routing_breakdown(adaptive_rows),
            "error_taxonomy": {
                "baseline": _distribution(b_labels),
                "adaptive": _distribution(a_labels),
                "paired_shifts": paired_shifts,
                "example_failures": example_failures,
            },
            "latency": latency_section,
            "paths": {
                "baseline_dir": str(self.baseline_dir),
                "adaptive_dir": str(self.adaptive_dir),
            },
        }

        if write:
            self.write_report(report)
        return report

    @staticmethod
    def _group_em(rows: Sequence[dict], key_fn) -> Dict[str, float]:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)
        return {k: mean_em(v) for k, v in groups.items()}

    def _paired_label_shifts(
        self,
        baseline_rows: List[dict],
        adaptive_rows: List[dict],
    ) -> Dict[str, int]:
        b_map = {r["question_id"]: r for r in baseline_rows}
        shifts: Dict[str, int] = defaultdict(int)
        for a in adaptive_rows:
            b = b_map.get(a["question_id"])
            if not b:
                continue
            key = f"{b['error_label']}->{a['error_label']}"
            shifts[key] += 1
        return dict(sorted(shifts.items(), key=lambda kv: (-kv[1], kv[0])))

    def _example_failures(
        self,
        adaptive_rows: List[dict],
        limit: int = 10,
    ) -> List[dict]:
        failures = [
            r for r in adaptive_rows
            if r.get("error_label") != "correct"
        ]
        failures.sort(key=lambda r: r.get("anls_score", 0.0))
        out = []
        for r in failures[:limit]:
            out.append({
                "question_id": r.get("question_id"),
                "error_label": r.get("error_label"),
                "question": (r.get("question") or "")[:120],
                "predicted_answer": r.get("predicted_answer"),
                "ground_truth_answers": r.get("ground_truth_answers"),
                "anls_score": r.get("anls_score", 0.0),
                "used_ocr": r.get("used_ocr", False),
                "cohort": r.get("cohort", ""),
                "routing_reason": (r.get("routing") or {}).get("reason", ""),
            })
        return out

    def write_report(self, report: dict) -> Tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "report.json"
        md_path = self.output_dir / "report.md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.format_markdown(report))
        return json_path, md_path

    @staticmethod
    def format_markdown(report: dict) -> str:
        o = report["overall"]
        lines = [
            "# DocVQA Evaluation Harness Report",
            "",
            "## Overall (paired)",
            f"- Baseline mean ANLS: **{o['baseline_mean_anls']:.4f}**",
            f"- Adaptive mean ANLS: **{o['adaptive_mean_anls']:.4f}**",
            f"- Delta ANLS: **{o['delta_anls']:+.4f}**",
            f"- Baseline EM: **{o['baseline_exact_match']:.4f}**",
            f"- Adaptive EM: **{o['adaptive_exact_match']:.4f}**",
            f"- Delta EM: **{o['delta_em']:+.4f}**",
            f"- Samples: {o['sample_count']}",
            "",
            "## Wins per sample",
            f"- Adaptive wins: {report['wins']['adaptive']}",
            f"- Baseline wins: {report['wins']['baseline']}",
            f"- Ties: {report['wins']['tie']}",
            "",
            "## By cohort (ANLS)",
        ]
        cohorts = sorted(
            set(report["by_cohort"]["baseline"]["anls"])
            | set(report["by_cohort"]["adaptive"]["anls"])
        )
        for cohort in cohorts:
            b = report["by_cohort"]["baseline"]["anls"].get(cohort, 0.0)
            a = report["by_cohort"]["adaptive"]["anls"].get(cohort, 0.0)
            lines.append(f"- {cohort}: baseline={b:.4f}, adaptive={a:.4f}, delta={a - b:+.4f}")

        lines.extend(["", "## Error taxonomy (adaptive)"])
        for label in ERROR_LABELS:
            stats = report["error_taxonomy"]["adaptive"].get(label, {})
            lines.append(
                f"- {label}: n={stats.get('count', 0)}, "
                f"rate={stats.get('rate', 0.0):.3f}"
            )

        lines.extend(["", "## Error taxonomy (baseline)"])
        for label in ERROR_LABELS:
            stats = report["error_taxonomy"]["baseline"].get(label, {})
            lines.append(
                f"- {label}: n={stats.get('count', 0)}, "
                f"rate={stats.get('rate', 0.0):.3f}"
            )

        lines.extend(["", "## Top paired label shifts (baseline → adaptive)"])
        for key, count in list(report["error_taxonomy"]["paired_shifts"].items())[:15]:
            lines.append(f"- {key}: {count}")

        lines.extend(["", "## Routing breakdown (adaptive)"])
        for reason, stats in report["routing_breakdown"].items():
            lines.append(
                f"- {reason}: n={stats['count']}, mean_anls={stats['mean_anls']:.4f}"
            )

        lat = report.get("latency") or {}
        lines.extend(["", "## Latency"])
        lines.append(f"- Available: {lat.get('available', False)}")
        lines.append(f"- Note: {lat.get('note', '')}")
        if lat.get("smoke"):
            smoke = lat["smoke"]
            lines.append(f"- Smoke n={smoke.get('n')}")
            for mode_key in ("baseline", "adaptive"):
                stats = smoke.get(mode_key)
                if stats:
                    lines.append(
                        f"- Smoke {mode_key}: mean={stats['mean']:.1f}ms, "
                        f"p50={stats['p50']:.1f}ms, p95={stats['p95']:.1f}ms"
                    )
            if smoke.get("adaptive_ocr_prep"):
                prep = smoke["adaptive_ocr_prep"]
                lines.append(
                    f"- Smoke adaptive OCR prep: mean={prep['mean']:.1f}ms, "
                    f"p95={prep['p95']:.1f}ms"
                )
        merged = lat.get("from_merged") or {}
        if merged.get("adaptive"):
            a = merged["adaptive"]
            lines.append(
                f"- Merged adaptive: mean={a['mean']:.1f}ms, "
                f"p50={a['p50']:.1f}ms, p95={a['p95']:.1f}ms "
                f"(n={merged.get('adaptive_n', 0)})"
            )

        lines.extend(["", "## Example adaptive failures"])
        for case in report["error_taxonomy"].get("example_failures", [])[:8]:
            lines.append(
                f"- qid={case['question_id']} [{case['error_label']}] "
                f"ANLS={case['anls_score']:.2f} | {case['question']}"
            )
            lines.append(
                f"  pred={case['predicted_answer']!r} gt={case['ground_truth_answers']}"
            )

        lines.append("")
        return "\n".join(lines) + "\n"


def build_synthetic_rows_for_tests() -> Tuple[List[dict], List[dict]]:
    """Tiny paired fixture for unit tests (no disk dependency)."""
    baseline = [
        {
            "question_id": 1,
            "cohort": "layout_heavy",
            "question_types": ["form"],
            "predicted_answer": "wrong field",
            "ground_truth_answers": ["10 rat sera"],
            "anls_score": 0.0,
            "used_ocr": False,
        },
        {
            "question_id": 2,
            "cohort": "visual_heavy",
            "question_types": ["figure/diagram"],
            "predicted_answer": "",
            "ground_truth_answers": ["yes"],
            "anls_score": 0.0,
            "used_ocr": False,
        },
        {
            "question_id": 3,
            "cohort": "layout_heavy",
            "question_types": ["form"],
            "predicted_answer": "10 rat sera",
            "ground_truth_answers": ["10 rat sera"],
            "anls_score": 1.0,
            "used_ocr": False,
        },
    ]
    adaptive = [
        {
            "question_id": 1,
            "cohort": "layout_heavy",
            "question_types": ["form"],
            "predicted_answer": "10 rat sera",
            "ground_truth_answers": ["10 rat sera"],
            "anls_score": 1.0,
            "used_ocr": True,
            "routing": {"reason": "question_type_or_keyword"},
        },
        {
            "question_id": 2,
            "cohort": "visual_heavy",
            "question_types": ["figure/diagram"],
            "predicted_answer": "INSTRUCTION: see notice",
            "ground_truth_answers": ["yes"],
            "anls_score": 0.0,
            "used_ocr": True,
            "routing": {"reason": "image_density_fallback"},
            "ocr_snippet": "[0,0,10,10] INSTRUCTION: see notice",
        },
        {
            "question_id": 3,
            "cohort": "layout_heavy",
            "question_types": ["form"],
            "predicted_answer": "10 rat sera",
            "ground_truth_answers": ["10 rat sera"],
            "anls_score": 1.0,
            "used_ocr": True,
            "routing": {"reason": "question_type_or_keyword"},
            "latency_ms": 1000.0,
            "ocr_prep_ms": 50.0,
        },
    ]
    return baseline, adaptive
