"""
Rule-based DocVQA error taxonomy for evaluation harness.

Labels:
- correct: ANLS >= threshold (default 0.5) or exact match
- refusal_empty: empty / unknown / n/a style answers
- ocr_noise: hijack/boilerplate patterns or pred copied from OCR but not GT
- field_selection: plausible wrong span on OCR/layout samples
- other: remaining failures
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.utils.metrics import anls_score, exact_match

ERROR_LABELS = (
    "correct",
    "refusal_empty",
    "ocr_noise",
    "field_selection",
    "other",
)

HIJACK_RE = re.compile(r"#|INSTRUCTION|User:|NOTICE", re.IGNORECASE)
REFUSAL_RE = re.compile(
    r"^\s*(unknown|n/?a|none|not\s+sure|cannot\s+answer|no\s+answer|"
    r"i\s+don'?t\s+know|unable\s+to\s+(?:answer|determine))\.?\s*$",
    re.IGNORECASE,
)
BOILERPLATE_TOKENS = frozenset({"INSTRUCTION", "NOTICE", "USER"})


def _normalize_loose(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_refusal_empty(pred: str) -> bool:
    if not (pred or "").strip():
        return True
    return bool(REFUSAL_RE.match(pred.strip()))


def _pred_in_ocr_not_gt(pred: str, gt_list: List[str], ocr_text: str) -> bool:
    """True if prediction appears in OCR context but matches no ground truth."""
    if not ocr_text or not pred.strip():
        return False
    pred_n = _normalize_loose(pred)
    ocr_n = _normalize_loose(ocr_text)
    if len(pred_n) < 2 or pred_n not in ocr_n:
        return False
    for gt in gt_list or []:
        gt_n = _normalize_loose(gt)
        if gt_n and (pred_n == gt_n or pred_n in gt_n or gt_n in pred_n):
            return False
    return True


def _has_boilerplate(pred: str) -> bool:
    tokens = set(re.findall(r"[A-Za-z]+", pred or ""))
    upper = {t.upper() for t in tokens}
    return bool(upper & BOILERPLATE_TOKENS) or bool(HIJACK_RE.search(pred or ""))


def _looks_like_field_value(pred: str) -> bool:
    """Short, non-sentence answers typical of wrong field copy."""
    text = (pred or "").strip()
    if not text or len(text) > 80:
        return False
    if text.count(" ") > 8:
        return False
    return True


def classify_error(
    row: Dict[str, Any],
    *,
    anls_threshold: float = 0.5,
    ocr_text: Optional[str] = None,
) -> str:
    """
    Classify a single evaluation row into an error taxonomy label.

    Args:
        row: Result dict with predicted_answer, ground_truth_answers, anls_score, etc.
        anls_threshold: ANLS cutoff for "correct" (DocVQA default 0.5).
        ocr_text: Optional OCR snippet text for noise detection.
    """
    pred = row.get("predicted_answer", "") or ""
    gt_list = row.get("ground_truth_answers") or []
    anls = row.get("anls_score")
    if anls is None:
        anls = anls_score([pred], gt_list) if gt_list else 0.0

    if anls >= anls_threshold or (gt_list and exact_match([pred], gt_list)):
        return "correct"

    if _is_refusal_empty(pred):
        return "refusal_empty"

    snippet = ocr_text
    if snippet is None:
        snippet = row.get("ocr_snippet") or row.get("ocr_text") or ""
        formatting = row.get("formatting") or {}
        if not snippet and isinstance(formatting.get("snippet"), str):
            snippet = formatting["snippet"]

    if _has_boilerplate(pred) or _pred_in_ocr_not_gt(pred, gt_list, snippet):
        return "ocr_noise"

    used_ocr = bool(row.get("used_ocr"))
    cohort = (row.get("cohort") or "").lower()
    layoutish = cohort == "layout_heavy" or used_ocr
    if layoutish and _looks_like_field_value(pred):
        return "field_selection"

    return "other"


def label_rows(
    rows: List[Dict[str, Any]],
    *,
    anls_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return shallow copies of rows with error_label attached."""
    labeled = []
    for row in rows:
        copy = dict(row)
        copy["error_label"] = classify_error(copy, anls_threshold=anls_threshold)
        labeled.append(copy)
    return labeled
