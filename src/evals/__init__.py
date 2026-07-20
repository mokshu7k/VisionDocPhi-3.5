"""DocVQA evaluation harness and error taxonomy."""

from src.evals.taxonomy import ERROR_LABELS, classify_error
from src.evals.harness import EvalHarness

__all__ = [
    "ERROR_LABELS",
    "classify_error",
    "EvalHarness",
]
