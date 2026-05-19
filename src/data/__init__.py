"""
Data module for dataset handling
"""
from .dataset import DocVQADataset, create_dataloader, get_dataset_stats

__all__ = ["DocVQADataset", "create_dataloader", "get_dataset_stats"]
