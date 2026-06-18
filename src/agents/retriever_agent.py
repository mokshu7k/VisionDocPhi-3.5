"""Hybrid dense-sparse retriever for OCR line selection."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    EMBEDDING_MODEL,
    FIELD_LABEL_KEYWORDS,
    HYBRID_ALPHA,
    OCR_CACHE_DIR,
    OCR_MAX_CHARS,
    OCR_MAX_LINES,
    OCR_MIN_SCORE,
    OCR_TOP_K,
    QUERY_STOPWORDS,
)
from src.agents.context_expander import is_label_only
from src.agents.types import ScoredLine
from src.data.ocr_loader import OcrLine

EMBEDDING_CACHE_SUFFIX = "_w1"

ALNUM_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?|[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*"
)


def extract_query_tokens(question: str) -> List[str]:
    tokens = ALNUM_TOKEN_RE.findall(question)
    normalized = []
    for t in tokens:
        n = t.upper().strip()
        if len(n) >= 1 and n not in QUERY_STOPWORDS:
            normalized.append(n)
    return normalized


def _token_matches_line(token: str, line_upper: str) -> bool:
    if token in FIELD_LABEL_KEYWORDS:
        return bool(re.search(rf"\b{re.escape(token)}\b", line_upper))
    return token in line_upper


def sparse_match_score(question: str, line_text: str, tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    line_upper = line_text.upper()
    matched = 0
    for token in tokens:
        if _token_matches_line(token, line_upper):
            matched += 1
    score = matched / len(tokens)
    for token in tokens:
        if len(token) >= 3 and _token_matches_line(token, line_upper):
            return 1.0
    return score


def _windowed_texts(lines: List[OcrLine]) -> List[str]:
    texts = []
    for i, line in enumerate(lines):
        parts = []
        if i > 0:
            parts.append(lines[i - 1].text)
        parts.append(line.text)
        if i < len(lines) - 1:
            parts.append(lines[i + 1].text)
        texts.append(" ".join(parts))
    return texts


def _sort_key(scored: ScoredLine) -> Tuple[float, int]:
    label_boost = 1 if is_label_only(scored.text) else 0
    return (scored.final_score, label_boost)


class HybridRetrieverAgent:
    def __init__(self, alpha: float = HYBRID_ALPHA):
        self.alpha = alpha
        self._encoder = None
        self._embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def _load_encoder(self):
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._encoder = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        self._load_encoder()
        if hasattr(self._encoder, "encode"):
            return self._encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self._encoder.fit_transform(texts).toarray()

    def _cache_path(self, ucsf_id: str, page_no: str) -> Optional[Path]:
        if not ucsf_id or not page_no:
            return None
        return OCR_CACHE_DIR / f"{ucsf_id}_{page_no}{EMBEDDING_CACHE_SUFFIX}.npz"

    def _load_line_embeddings_cache(
        self, ucsf_id: str, page_no: str, n_lines: int
    ) -> Optional[np.ndarray]:
        path = self._cache_path(ucsf_id, page_no)
        if path is None or not path.exists():
            return None
        try:
            data = np.load(path)
            embs = data["embeddings"]
            if embs.shape[0] == n_lines:
                return embs
        except Exception:
            return None
        return None

    def _save_line_embeddings_cache(
        self, ucsf_id: str, page_no: str, embeddings: np.ndarray
    ):
        path = self._cache_path(ucsf_id, page_no)
        if path is None:
            return
        try:
            np.savez_compressed(path, embeddings=embeddings)
        except Exception:
            pass

    def _dense_scores(
        self,
        question: str,
        lines: List[OcrLine],
        ucsf_id: Optional[str] = None,
        page_no: Optional[str] = None,
    ) -> List[float]:
        if not lines:
            return []
        texts = _windowed_texts(lines)
        self._load_encoder()
        if hasattr(self._encoder, "encode"):
            q_emb = self._encode_texts([question])[0]
            cached = None
            if ucsf_id and page_no:
                cached = self._load_line_embeddings_cache(ucsf_id, page_no, len(lines))
            if cached is not None:
                line_embs = cached
            else:
                line_embs = self._encode_texts(texts)
                if ucsf_id and page_no:
                    self._save_line_embeddings_cache(ucsf_id, page_no, line_embs)
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0:
                return [0.0] * len(lines)
            scores = []
            for emb in line_embs:
                denom = np.linalg.norm(emb)
                if denom == 0:
                    scores.append(0.0)
                else:
                    scores.append(float(np.dot(q_emb, emb) / (q_norm * denom)))
            return scores

        all_texts = [question] + texts
        matrix = self._encoder.fit_transform(all_texts).toarray()
        q_emb = matrix[0]
        line_embs = matrix[1:]
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return [0.0] * len(lines)
        scores = []
        for emb in line_embs:
            denom = np.linalg.norm(emb)
            if denom == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(q_emb, emb) / (q_norm * denom)))
        return scores

    def retrieve(
        self,
        question: str,
        lines: List[OcrLine],
        ucsf_id: Optional[str] = None,
        page_no: Optional[str] = None,
    ) -> List[ScoredLine]:
        if not lines:
            return []

        tokens = extract_query_tokens(question)
        dense_scores = self._dense_scores(question, lines, ucsf_id=ucsf_id, page_no=page_no)

        scored: List[ScoredLine] = []
        for i, line in enumerate(lines):
            sparse = sparse_match_score(question, line.text, tokens)
            dense = dense_scores[i] if i < len(dense_scores) else 0.0
            final = self.alpha * dense + (1 - self.alpha) * sparse
            if final < OCR_MIN_SCORE:
                continue
            scored.append(
                ScoredLine(
                    line_id=line.line_id,
                    text=line.text,
                    bbox=line.bbox,
                    dense_score=dense,
                    sparse_score=sparse,
                    final_score=final,
                )
            )

        scored.sort(key=_sort_key, reverse=True)
        pool = scored[:OCR_TOP_K]

        selected: List[ScoredLine] = []
        char_count = 0
        for line in pool:
            if len(selected) >= OCR_MAX_LINES:
                break
            line_len = len(line.text) + 30
            if char_count + line_len > OCR_MAX_CHARS and selected:
                break
            selected.append(line)
            char_count += line_len

        return selected
