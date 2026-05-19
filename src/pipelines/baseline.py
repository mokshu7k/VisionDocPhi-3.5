"""
Zero-Shot Baseline Pipeline

Main evaluation pipeline for DocVQA zero-shot baseline
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any
import logging

from config.settings import (
    MODEL_NAME, DEVICE, VAL_ANNOTATIONS, TEST_ANNOTATIONS,
    IMAGES_DIR, BATCH_SIZE, NUM_WORKERS, PREDICTIONS_FILE,
    RESULTS_FILE, SAVE_PREDICTIONS
)
from src.models.inference import DocVQAInference
from src.data.dataset import create_dataloader, get_dataset_stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_zero_shot_baseline(
    split: str = "val",
    num_samples: int = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run zero-shot VQA baseline on specified dataset split
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to evaluate (None = all)
        save_results: Whether to save predictions and results
    
    Returns:
        Dictionary with results and metrics
    """
    print("\n" + "="*70)
    print(f"🚀 DOCVQA ZERO-SHOT BASELINE - {split.upper()} SET")
    print("="*70 + "\n")
    
    # Select evaluation split
    if split == "val":
        annotations_file = str(VAL_ANNOTATIONS)
    elif split == "test":
        annotations_file = str(TEST_ANNOTATIONS)
    else:
        raise ValueError(f"Unknown split: {split}. Choose from 'train', 'val', 'test'")
    
    # Get dataset stats
    print("📊 Dataset Statistics:")
    stats = get_dataset_stats(annotations_file)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize inference engine
    device = "cuda" if torch.cuda.is_available() else DEVICE
    print(f"💾 Using device: {device}\n")
    
    inference = DocVQAInference(model_name=MODEL_NAME, device=device)
    
    # Create dataloader
    print(f"📂 Loading {split} dataset...")
    dataloader = create_dataloader(
        annotations_file=annotations_file,
        image_dir=str(IMAGES_DIR),
        split=split,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )
    
    # Run evaluation
    print(f"⏳ Starting evaluation on {split} set...")
    eval_results = inference.evaluate(dataloader, num_samples=num_samples)
    
    # Print metrics
    metrics = {k: v for k, v in eval_results.items() if k != 'results'}
    print("\n" + "="*70)
    print("📈 EVALUATION METRICS")
    print("="*70)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")
    print("="*70 + "\n")
    
    # Save results
    if save_results and SAVE_PREDICTIONS:
        print("💾 Saving results...")
        
        # Save predictions
        predictions_data = {
            'model': MODEL_NAME,
            'split': split,
            'device': device,
            'metrics': metrics,
            'results': eval_results.get('results', [])
        }
        
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"  ✓ Predictions saved to: {PREDICTIONS_FILE}")
        
        # Save summary results
        results_data = {
            'model': MODEL_NAME,
            'split': split,
            'device': device,
            'metrics': metrics
        }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  ✓ Results summary saved to: {RESULTS_FILE}")
    
    return eval_results


if __name__ == "__main__":
    # Run baseline on validation set
    results = run_zero_shot_baseline(split="val", num_samples=None)
    print("\n✅ Baseline evaluation completed!")
