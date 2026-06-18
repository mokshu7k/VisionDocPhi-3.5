"""Expand retrieved OCR lines with spatial neighbors and label-value pairs."""

import re
from typing import Dict, List, Set

from config.settings import NEIGHBOR_Y_GAP
from src.agents.types import ScoredLine
from src.data.ocr_loader import OcrLine

LABEL_ONLY_RE = re.compile(r"^[\w\s/.-]{1,40}:\s*$")


def _x_overlap(b1: List[int], b2: List[int]) -> bool:
    return b1[0] <= b2[2] and b2[0] <= b1[2]


def _y_overlap(b1: List[int], b2: List[int]) -> bool:
    return b1[1] <= b2[3] and b2[1] <= b1[3]


def _is_label_only(text: str) -> bool:
    t = text.strip()
    return bool(LABEL_ONLY_RE.match(t)) or (t.endswith(":") and ":" in t and len(t.split(":")[-1].strip()) == 0)


def _scored_from_ocr(line: OcrLine, anchor: ScoredLine) -> ScoredLine:
    return ScoredLine(
        line_id=line.line_id,
        text=line.text,
        bbox=line.bbox,
        dense_score=anchor.dense_score,
        sparse_score=anchor.sparse_score,
        final_score=anchor.final_score * 0.9,
    )


def expand_context(
    scored_lines: List[ScoredLine],
    all_lines: List[OcrLine],
    neighbor_y_gap: int = NEIGHBOR_Y_GAP,
) -> List[ScoredLine]:
    """Add neighbor and label-value lines to the retrieved set."""
    if not scored_lines or not all_lines:
        return scored_lines

    by_id: Dict[int, OcrLine] = {line.line_id: line for line in all_lines}
    scored_map: Dict[int, ScoredLine] = {s.line_id: s for s in scored_lines}
    expanded_ids: Set[int] = set(scored_map.keys())

    for anchor in scored_lines:
        ocr_line = by_id.get(anchor.line_id)
        if ocr_line is None:
            continue
        bbox = ocr_line.bbox
        x1, y1, x2, y2 = bbox

        # Vertical neighbor below within gap and horizontal overlap
        for candidate in all_lines:
            if candidate.line_id in expanded_ids:
                continue
            cx1, cy1, cx2, cy2 = candidate.bbox
            gap = cy1 - y2
            if 0 <= gap <= neighbor_y_gap and _x_overlap(bbox, candidate.bbox):
                expanded_ids.add(candidate.line_id)
                scored_map[candidate.line_id] = _scored_from_ocr(candidate, anchor)

        # Label-value: force next line below when label ends with colon
        if _is_label_only(ocr_line.text):
            below = [
                ln for ln in all_lines
                if ln.line_id not in expanded_ids
                and ln.bbox[1] >= y2
                and (ln.bbox[1] - y2) <= neighbor_y_gap
            ]
            if below:
                below.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
                best = below[0]
                expanded_ids.add(best.line_id)
                scored_map[best.line_id] = _scored_from_ocr(best, anchor)

        # Same-row right (form fields)
        if _is_label_only(ocr_line.text):
            same_row = [
                ln for ln in all_lines
                if ln.line_id not in expanded_ids
                and _y_overlap(bbox, ln.bbox)
                and ln.bbox[0] > x2
            ]
            if same_row:
                same_row.sort(key=lambda ln: ln.bbox[0])
                right = same_row[0]
                expanded_ids.add(right.line_id)
                scored_map[right.line_id] = _scored_from_ocr(right, anchor)

    return list(scored_map.values())
