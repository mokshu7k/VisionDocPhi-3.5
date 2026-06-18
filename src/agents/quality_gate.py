"""Confidence gating before OCR infusion."""

import re
from typing import List

from config.settings import OCR_GATE_MIN_SCORE
from src.agents.retriever_agent import extract_query_tokens
from src.agents.types import FormattedSnippet, ScoredLine

WHAT_IS_RE = re.compile(r"\bwhat\s+is\b|\bwhat\s+are\b", re.IGNORECASE)


def _has_label_value_content(formatted: FormattedSnippet) -> bool:
    for line_text in formatted.text_block.split("\n"):
        if "] " in line_text:
            text = line_text.split("] ", 1)[-1].strip()
        else:
            text = line_text.strip()
        if ":" in text:
            after = text.split(":", 1)[1].strip()
            if after and re.search(r"[A-Za-z0-9]", after):
                return True
        if text and not text.endswith(":") and re.search(r"[A-Za-z0-9]{2,}", text):
            return True
    return False


def passes_quality_gate(
    scored: List[ScoredLine],
    formatted: FormattedSnippet,
    question: str,
) -> bool:
    if not scored or not formatted.text_block.strip():
        return False

    top_score = max(s.final_score for s in scored)
    if top_score < OCR_GATE_MIN_SCORE:
        return False

    q_tokens = set(extract_query_tokens(question))
    snippet_tokens = set(extract_query_tokens(formatted.text_block))
    new_tokens = snippet_tokens - q_tokens
    if not new_tokens and len(snippet_tokens) <= len(q_tokens):
        return False

    if WHAT_IS_RE.search(question):
        if not _has_label_value_content(formatted):
            return False

    return True
