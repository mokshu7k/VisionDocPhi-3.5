"""Unit tests for DocVQA eval harness metrics and error taxonomy."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evals.harness import EvalHarness, build_synthetic_rows_for_tests, mean_anls, mean_em
from src.evals.taxonomy import classify_error, label_rows
from src.utils.metrics import latency_percentiles


class TestLatencyPercentiles:
    def test_empty(self):
        assert latency_percentiles([]) == {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    def test_single(self):
        stats = latency_percentiles([42.0])
        assert stats["mean"] == 42.0
        assert stats["p50"] == 42.0
        assert stats["p95"] == 42.0

    def test_sorted_values(self):
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        stats = latency_percentiles(values)
        assert stats["mean"] == pytest.approx(55.0)
        assert stats["p50"] == 50
        assert stats["p90"] == 90
        assert stats["p95"] == 100


class TestTaxonomy:
    def test_correct_by_anls(self):
        row = {
            "predicted_answer": "10 rat sera",
            "ground_truth_answers": ["10 rat sera"],
            "anls_score": 1.0,
        }
        assert classify_error(row) == "correct"

    def test_refusal_empty(self):
        assert classify_error({
            "predicted_answer": "",
            "ground_truth_answers": ["yes"],
            "anls_score": 0.0,
        }) == "refusal_empty"
        assert classify_error({
            "predicted_answer": "unknown",
            "ground_truth_answers": ["yes"],
            "anls_score": 0.0,
        }) == "refusal_empty"

    def test_ocr_noise_hijack(self):
        row = {
            "predicted_answer": "INSTRUCTION: fill form",
            "ground_truth_answers": ["John"],
            "anls_score": 0.0,
            "used_ocr": True,
        }
        assert classify_error(row) == "ocr_noise"

    def test_ocr_noise_from_snippet(self):
        row = {
            "predicted_answer": "Epidemiology",
            "ground_truth_answers": ["College of Public Health"],
            "anls_score": 0.0,
            "used_ocr": True,
            "ocr_snippet": "[1,2,3,4] Epidemiology\n[5,6,7,8] College of Public Health",
        }
        assert classify_error(row) == "ocr_noise"

    def test_field_selection(self):
        row = {
            "predicted_answer": "8/25/15",
            "ground_truth_answers": ["8/25/88"],
            "anls_score": 0.0,
            "used_ocr": True,
            "cohort": "layout_heavy",
        }
        assert classify_error(row) == "field_selection"

    def test_label_rows_attaches_field(self):
        rows = label_rows([
            {
                "predicted_answer": "yes",
                "ground_truth_answers": ["yes"],
                "anls_score": 1.0,
            }
        ])
        assert rows[0]["error_label"] == "correct"


class TestHarnessDryRun:
    def test_synthetic_means(self):
        baseline, adaptive = build_synthetic_rows_for_tests()
        assert mean_anls(baseline) == pytest.approx(1.0 / 3.0)
        assert mean_em(adaptive) == pytest.approx(2.0 / 3.0)

    def test_dry_run_on_merged_if_present(self):
        baseline_merged = PROJECT_ROOT / "data/outputs/baseline_200/results_merged.json"
        adaptive_merged = PROJECT_ROOT / "data/outputs/ocr_adaptive_200_v2/results_merged.json"
        if not baseline_merged.exists() or not adaptive_merged.exists():
            pytest.skip("Merged eval results not present")

        harness = EvalHarness(adaptive_version="v2")
        result = harness.dry_run(max_samples=5)
        assert result["ok"] is True
        assert result["adaptive_samples"] == 5
        assert len(result["error_labels"]) == 5
        assert "correct" in result["error_distribution"]

    def test_run_writes_report(self, tmp_path):
        baseline, adaptive = build_synthetic_rows_for_tests()
        b_dir = tmp_path / "baseline_200"
        a_dir = tmp_path / "ocr_adaptive_200_v2"
        b_dir.mkdir()
        a_dir.mkdir()
        with open(b_dir / "results_merged.json", "w", encoding="utf-8") as f:
            json.dump({"results": baseline}, f)
        with open(a_dir / "results_merged.json", "w", encoding="utf-8") as f:
            json.dump({"results": adaptive}, f)

        # Point harness paths at temp dirs by monkeypatching get_mode_output_dir usage
        harness = EvalHarness(adaptive_version="v2", output_dir=tmp_path / "harness")
        harness.baseline_dir = b_dir
        harness.adaptive_dir = a_dir
        report = harness.run(write=True)

        assert (tmp_path / "harness" / "report.json").exists()
        assert (tmp_path / "harness" / "report.md").exists()
        assert report["overall"]["sample_count"] == 3
        assert report["error_taxonomy"]["adaptive"]["correct"]["count"] == 2
        assert report["error_taxonomy"]["adaptive"]["ocr_noise"]["count"] == 1
        assert report["latency"]["available"] is True
        assert report["latency"]["from_merged"]["adaptive"]["mean"] == pytest.approx(1000.0)
