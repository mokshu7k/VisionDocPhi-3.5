"""Expand retrieved OCR lines with spatial neighbors and label-value pairs."""

import re
from typing import Dict, List, Set

from config.settings import (
    COLUMN_X_PAD,
    FIELD_LABEL_KEYWORDS,
    NEIGHBOR_Y_GAP,
    OCR_MAX_CHARS,
    OCR_MAX_LINES,
    Y_OVERLAP_TOLERANCE,
)
from src.agents.types import ScoredLine
from src.data.ocr_loader import OcrLine

LABEL_ONLY_RE = re.compile(r"^[\w\s/.-]{1,40}:\s*$")
FIELD_LABEL_PREFIX_RE = re.compile(
    r"^(" + "|".join(re.escape(k) for k in sorted(FIELD_LABEL_KEYWORDS)) + r")[\s:.\-/]*$",
    re.IGNORECASE,
)


def _x_overlap(b1: List[int], b2: List[int]) -> bool:
    return b1[0] <= b2[2] and b2[0] <= b1[2]


def _y_overlap(b1: List[int], b2: List[int], tolerance: int = Y_OVERLAP_TOLERANCE) -> bool:
    """Horizontal row alignment via vertical midpoint tolerance."""
    center1 = (b1[1] + b1[3]) / 2
    center2 = (b2[1] + b2[3]) / 2
    return abs(center1 - center2) <= tolerance


def _in_column(candidate_bbox: List[int], anchor_bbox: List[int], pad: int = COLUMN_X_PAD) -> bool:
    ax1, _, ax2, _ = anchor_bbox
    cx1, _, cx2, _ = candidate_bbox
    return cx1 <= (ax2 + pad) and cx2 >= (ax1 - pad)


def is_label_only(text: str) -> bool:
    t = text.strip()
    if LABEL_ONLY_RE.match(t):
        return True
    if t.endswith(":") and ":" in t and len(t.split(":")[-1].strip()) == 0:
        return True
    if len(t) < 40 and FIELD_LABEL_PREFIX_RE.match(t):
        return True
    return False


def _scored_from_ocr(line: OcrLine, anchor: ScoredLine) -> ScoredLine:
    return ScoredLine(
        line_id=line.line_id,
        text=line.text,
        bbox=line.bbox,
        dense_score=anchor.dense_score,
        sparse_score=anchor.sparse_score,
        final_score=anchor.final_score * 0.9,
    )


def _apply_expansion_budget(
    anchors: List[ScoredLine],
    all_expanded: List[ScoredLine],
    max_lines: int = OCR_MAX_LINES,
    max_chars: int = OCR_MAX_CHARS,
) -> List[ScoredLine]:
    anchor_ids = {a.line_id for a in anchors}
    selected = list(anchors)
    char_count = sum(len(a.text) + 30 for a in anchors)

    extras = [line for line in all_expanded if line.line_id not in anchor_ids]
    extras.sort(key=lambda s: s.final_score, reverse=True)

    for line in extras:
        if len(selected) >= max_lines:
            break
        line_len = len(line.text) + 30
        if char_count + line_len > max_chars:
            continue
        selected.append(line)
        char_count += line_len

    selected.sort(key=lambda s: (s.bbox[1], s.bbox[0]))
    return selected


def expand_context(
    scored_lines: List[ScoredLine],
    all_lines: List[OcrLine],
    neighbor_y_gap: int = NEIGHBOR_Y_GAP,
) -> List[ScoredLine]:
    """Add neighbor and label-value lines; return layout-ordered, budget-capped set."""
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

        for candidate in all_lines:
            if candidate.line_id in expanded_ids:
                continue
            cx1, cy1, cx2, cy2 = candidate.bbox
            gap = cy1 - y2
            if (
                0 <= gap <= neighbor_y_gap
                and _x_overlap(bbox, candidate.bbox)
                and _in_column(candidate.bbox, bbox)
            ):
                expanded_ids.add(candidate.line_id)
                scored_map[candidate.line_id] = _scored_from_ocr(candidate, anchor)

        if is_label_only(ocr_line.text):
            below = [
                ln for ln in all_lines
                if ln.line_id not in expanded_ids
                and ln.bbox[1] >= y2
                and (ln.bbox[1] - y2) <= neighbor_y_gap
                and _in_column(ln.bbox, bbox)
            ]
            if below:
                below.sort(key=lambda ln: (abs(ln.bbox[0] - x1), ln.bbox[1]))
                best = below[0]
                expanded_ids.add(best.line_id)
                scored_map[best.line_id] = _scored_from_ocr(best, anchor)

        if is_label_only(ocr_line.text):
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

    all_expanded = list(scored_map.values())
    return _apply_expansion_budget(scored_lines, all_expanded)
