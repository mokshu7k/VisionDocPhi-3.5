"""Unit tests for OCR retrieval and context expansion."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.context_expander import (
    _apply_expansion_budget,
    _in_column,
    _y_overlap,
    expand_context,
    is_label_only,
)
from src.agents.retriever_agent import extract_query_tokens, sparse_match_score
from src.agents.types import ScoredLine
from src.data.ocr_loader import OcrLine


def _scored(line_id: int, text: str, bbox: list, score: float = 0.8) -> ScoredLine:
    return ScoredLine(
        line_id=line_id,
        text=text,
        bbox=bbox,
        dense_score=score,
        sparse_score=score,
        final_score=score,
    )


class TestStopwordBoost:
    def test_the_does_not_trigger_max_sparse(self):
        tokens = extract_query_tokens("Which is the college?")
        assert "THE" not in tokens
        score = sparse_match_score(
            "Which is the college?",
            "THE UNIVERSITY OF ARKANSAS",
            tokens,
        )
        assert score < 1.0

    def test_college_still_matches(self):
        tokens = extract_query_tokens("Which is the college?")
        assert "COLLEGE" in tokens
        score = sparse_match_score(
            "Which is the college?",
            "College of Medicine",
            tokens,
        )
        assert score == 1.0


class TestLabelDetection:
    def test_colon_label(self):
        assert is_label_only("College:")

    def test_keyword_without_colon(self):
        assert is_label_only("College")

    def test_value_line_not_label(self):
        assert not is_label_only("School of Public Health")


class TestGeometry:
    def test_y_overlap_midpoint_tolerance(self):
        b1 = [10, 100, 100, 120]
        b2 = [10, 104, 100, 124]
        assert _y_overlap(b1, b2, tolerance=6)

    def test_in_column_within_pad(self):
        anchor = [100, 50, 200, 70]
        candidate = [110, 80, 190, 100]
        assert _in_column(candidate, anchor, pad=20)

    def test_in_column_rejects_other_column(self):
        anchor = [100, 50, 200, 70]
        candidate = [400, 80, 500, 100]
        assert not _in_column(candidate, anchor, pad=20)

    def test_wide_value_below_label_in_column(self):
        anchor = [100, 50, 180, 70]
        value = [105, 75, 350, 95]
        assert _in_column(value, anchor, pad=20)


class TestExpansion:
    def test_label_below_expands_value(self):
        anchor = _scored(0, "College", [100, 50, 180, 70], score=0.9)
        value = OcrLine(1, "School of Public Health", [105, 75, 350, 95])
        label = OcrLine(0, "College", [100, 50, 180, 70])
        other_col = OcrLine(2, "College of Medicine", [400, 75, 600, 95])

        result = expand_context([anchor], [label, value, other_col])
        result_ids = {line.line_id for line in result}
        assert 1 in result_ids
        assert 2 not in result_ids

    def test_expansion_budget_caps_lines(self):
        anchors = [_scored(0, "College", [0, 0, 50, 10], score=1.0)]
        extras = [
            _scored(i, f"line{i}" * 20, [0, i * 15, 100, i * 15 + 10], score=0.5)
            for i in range(1, 30)
        ]
        capped = _apply_expansion_budget(anchors, anchors + extras, max_lines=5, max_chars=500)
        assert len(capped) <= 5

    def test_expansion_output_layout_sorted(self):
        anchor = _scored(0, "College", [100, 200, 180, 220], score=0.9)
        lines = [
            OcrLine(0, "College", [100, 200, 180, 220]),
            OcrLine(1, "School of Public Health", [105, 230, 350, 250]),
            OcrLine(2, "Name", [100, 50, 150, 70]),
            OcrLine(3, "John Doe", [105, 80, 250, 100]),
        ]
        result = expand_context([anchor], lines)
        ys = [line.bbox[1] for line in result]
        assert ys == sorted(ys)
