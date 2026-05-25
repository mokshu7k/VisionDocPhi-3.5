"""
Chunked Evaluation Pipeline

Process large datasets in chunks to fit within GPU memory constraints.
Useful for Google Colab or systems with limited GPU VRAM.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm

from config.settings import (
    MODEL_NAME, DEVICE, VAL_ANNOTATIONS, TEST_ANNOTATIONS,
    IMAGES_DIR, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR
)
from src.models.inference import DocVQAInference
from src.data.dataset import create_dataloader, get_dataset_stats

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkedEvaluator:
    """Evaluates DocVQA dataset in memory-efficient chunks"""
    
    def __init__(self, split: str = "val", chunk_size: int = 200):
        """
        Args:
            split: Dataset split ('val' or 'test')
            chunk_size: Number of samples per chunk
        """
        self.split = split
        self.chunk_size = chunk_size
        
        # Select annotations file
        if split == "val":
            self.annotations_file = str(VAL_ANNOTATIONS)
        elif split == "test":
            self.annotations_file = str(TEST_ANNOTATIONS)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.image_dir = str(IMAGES_DIR)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking files
        self.progress_file = self.output_dir / f"chunked_progress_{split}.json"
        self.chunks_dir = self.output_dir / f"chunks_{split}"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset stats
        self.dataset_stats = get_dataset_stats(self.annotations_file)
        self.total_samples = self.dataset_stats.get('total_samples', 0)
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from previous runs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'split': self.split,
            'chunk_size': self.chunk_size,
            'total_samples': self.total_samples,
            'completed_chunks': [],
            'all_predictions': [],
            'all_results': [],
            'merged_metrics': None
        }
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def save_chunk_results(self, chunk_id: int, results: Dict[str, Any]):
        """Save individual chunk results"""
        chunk_file = self.chunks_dir / f"chunk_{chunk_id:03d}.json"
        with open(chunk_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_chunk_results(self, chunk_id: int) -> Dict[str, Any]:
        """Load individual chunk results"""
        chunk_file = self.chunks_dir / f"chunk_{chunk_id:03d}.json"
        if chunk_file.exists():
            with open(chunk_file, 'r') as f:
                return json.load(f)
        return None
    
    def process_chunk(self, chunk_id: int, inference: DocVQAInference) -> Dict[str, Any]:
        """
        Process a single chunk
        
        Args:
            chunk_id: Index of the chunk (0-based)
            inference: DocVQAInference instance
        
        Returns:
            Dictionary with chunk results
        """
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        chunk_samples = end_idx - start_idx
        
        print(f"\n{'='*70}")
        print(f"📦 CHUNK {chunk_id + 1}/{self.num_chunks}")
        print(f"Samples: {start_idx} - {end_idx} ({chunk_samples} samples)")
        print(f"{'='*70}\n")
        
        # Create dataloader for this chunk
        dataloader = create_dataloader(
            annotations_file=self.annotations_file,
            image_dir=self.image_dir,
            split=self.split,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        
        # Process chunk
        chunk_results = inference.evaluate(dataloader, num_samples=chunk_samples)
        
        # Save chunk results
        self.save_chunk_results(chunk_id, chunk_results)
        print(f"✅ Chunk {chunk_id + 1} saved!\n")
        
        return chunk_results
    
    def merge_chunk_results(self, all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from all chunks
        
        Args:
            all_chunks: List of chunk result dictionaries
        
        Returns:
            Merged results dictionary
        """
        print("\n" + "="*70)
        print("🔗 MERGING RESULTS FROM ALL CHUNKS")
        print("="*70 + "\n")
        
        # Collect all predictions and results
        all_predictions = []
        all_ground_truths = []
        all_results = []
        
        for chunk_idx, chunk_data in enumerate(all_chunks):
            if 'results' in chunk_data:
                all_results.extend(chunk_data['results'])
            
            predictions = chunk_data.get('predictions', [])
            ground_truths = chunk_data.get('ground_truths', [])
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
            
            print(f"Chunk {chunk_idx + 1}: {len(predictions)} predictions added")
        
        print(f"\n📊 Total predictions merged: {len(all_predictions)}")
        
        # Recalculate metrics on merged data
        from src.utils.metrics import calculate_metrics
        merged_metrics = calculate_metrics(all_predictions, all_ground_truths)
        merged_metrics['results'] = all_results
        
        print("\n📈 MERGED METRICS:")
        for metric_name, metric_value in merged_metrics.items():
            if metric_name != 'results' and isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")
        
        return merged_metrics
    
    def run(self, resume: bool = True) -> Dict[str, Any]:
        """
        Run chunked evaluation
        
        Args:
            resume: Whether to resume from previous checkpoint
        
        Returns:
            Dictionary with final merged results
        """
        # Load progress
        progress = self.load_progress()
        
        print("\n" + "="*70)
        print("🚀 CHUNKED EVALUATION PIPELINE")
        print("="*70)
        print(f"📊 Split: {self.split.upper()}")
        print(f"📦 Total Samples: {self.total_samples}")
        print(f"📦 Chunk Size: {self.chunk_size}")
        print(f"📦 Total Chunks: {self.num_chunks}")
        
        if resume and progress['completed_chunks']:
            print(f"📋 Resuming from previous run...")
            print(f"✅ Already processed: {len(progress['completed_chunks'])} chunks")
        else:
            print(f"🔄 Starting fresh evaluation")
        
        print("="*70 + "\n")
        
        # Initialize inference engine
        device = "cuda" if torch.cuda.is_available() else DEVICE
        print(f"💾 Using device: {device}\n")
        
        inference = DocVQAInference(model_name=MODEL_NAME, device=device)
        
        # Process chunks
        all_chunks = []
        for chunk_id in range(self.num_chunks):
            # Skip if already completed
            if resume and chunk_id in progress['completed_chunks']:
                print(f"⏭️  Skipping chunk {chunk_id + 1} (already processed)")
                chunk_data = self.load_chunk_results(chunk_id)
                if chunk_data:
                    all_chunks.append(chunk_data)
                continue
            
            # Process chunk
            try:
                chunk_results = self.process_chunk(chunk_id, inference)
                all_chunks.append(chunk_results)
                
                # Update progress
                progress['completed_chunks'].append(chunk_id)
                self.save_progress(progress)
                
                # Memory cleanup
                inference.cleanup_memory()
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_id + 1}: {e}")
                print(f"⚠️  Skipping chunk {chunk_id + 1} and continuing...")
                continue
        
        # Merge and return results
        if all_chunks:
            merged_results = self.merge_chunk_results(all_chunks)
            
            # Save final merged results
            final_predictions_file = self.output_dir / f"predictions_{self.split}_merged.json"
            final_results_file = self.output_dir / f"results_{self.split}_merged.json"
            
            with open(final_predictions_file, 'w') as f:
                json.dump(merged_results.get('results', []), f, indent=2)
            
            results_summary = {k: v for k, v in merged_results.items() if k != 'results'}
            with open(final_results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\n💾 Final results saved!")
            print(f"  📄 Predictions: {final_predictions_file}")
            print(f"  📊 Metrics: {final_results_file}")
            
            return merged_results
        else:
            print("❌ No chunks were processed successfully!")
            return {}


def run_chunked_evaluation(split: str = "val", chunk_size: int = 200, resume: bool = True) -> Dict[str, Any]:
    """
    Main entry point for chunked evaluation
    
    Args:
        split: Dataset split ('val' or 'test')
        chunk_size: Number of samples per chunk (default 200)
        resume: Whether to resume from checkpoint
    
    Returns:
        Dictionary with final results and metrics
    """
    evaluator = ChunkedEvaluator(split=split, chunk_size=chunk_size)
    return evaluator.run(resume=resume)


if __name__ == "__main__":
    # Example usage
    results = run_chunked_evaluation(split="val", chunk_size=200, resume=True)
    print("\n✅ Chunked evaluation complete!")
