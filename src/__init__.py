"""
VisionDocPhi-3.5 Source Package
"""
__version__ = "1.0.0"
__author__ = "DocVQA Team"

from src.models.inference import DocVQAInference
from src.data.dataset import DocVQADataset, create_dataloader, get_dataset_stats
from src.utils.metrics import calculate_metrics, anls_score

__all__ = [
    "DocVQAInference",
    "DocVQADataset",
    "create_dataloader",
    "get_dataset_stats",
    "calculate_metrics",
    "anls_score",
]
