"""
Chunked Evaluation Pipeline

Process eval subset in chunks with mode-specific outputs and resume support.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from config.settings import (
    MODEL_NAME, DEVICE, VAL_ANNOTATIONS, TEST_ANNOTATIONS,
    IMAGES_DIR, BATCH_SIZE, NUM_WORKERS, OUTPUT_DIR,
    CHUNK_SIZE, EVAL_SUBSET_FILE, get_mode_output_dir,
)
from src.models.inference import DocVQAInference, MODE_ALIASES, InferenceMode
from src.data.dataset import (
    DocVQADataset,
    create_subset_dataloader,
    load_subset_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkedEvaluator:
    """Evaluates DocVQA eval subset in memory-efficient chunks."""

    def __init__(
        self,
        mode: str = "baseline",
        subset_file: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
        split: str = "val",
    ):
        self.mode = MODE_ALIASES.get(mode, mode)
        self.subset_file = Path(subset_file or EVAL_SUBSET_FILE)
        self.chunk_size = chunk_size
        self.split = split

        if split == "val":
            self.annotations_file = str(VAL_ANNOTATIONS)
        elif split == "test":
            self.annotations_file = str(TEST_ANNOTATIONS)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_dir = str(IMAGES_DIR)
        self.output_dir = get_mode_output_dir(self.mode)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "chunked_progress.json"
        self.merged_file = self.output_dir / "results_merged.json"

        self.question_ids, self.cohort_map = load_subset_metadata(str(self.subset_file))
        self.dataset = DocVQADataset(
            self.annotations_file,
            self.image_dir,
            split=split,
            question_ids=self.question_ids,
            cohort_map=self.cohort_map,
        )
        self.total_samples = len(self.dataset)
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size

    def load_progress(self) -> Dict[str, Any]:
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "mode": self.mode,
            "subset_file": str(self.subset_file),
            "chunk_size": self.chunk_size,
            "total_samples": self.total_samples,
            "completed_chunks": [],
        }

    def save_progress(self, progress: Dict[str, Any]):
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def save_chunk_results(self, chunk_id: int, results: Dict[str, Any]):
        chunk_file = self.chunks_dir / f"chunk_{chunk_id:03d}.json"
        with open(chunk_file, "w") as f:
            json.dump(results, f, indent=2)

    def load_chunk_results(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        chunk_file = self.chunks_dir / f"chunk_{chunk_id:03d}.json"
        if chunk_file.exists():
            with open(chunk_file, "r") as f:
                return json.load(f)
        return None

    def process_chunk(self, chunk_id: int, inference: DocVQAInference) -> Dict[str, Any]:
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        chunk_samples = end_idx - start_idx

        print(f"\n{'='*70}")
        print(f"📦 CHUNK {chunk_id + 1}/{self.num_chunks} (mode={self.mode})")
        print(f"Samples: {start_idx} - {end_idx} ({chunk_samples} samples)")
        print(f"{'='*70}\n")

        dataloader = create_subset_dataloader(
            self.dataset,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        chunk_results = inference.evaluate(dataloader, mode=self.mode)
        chunk_results["chunk_id"] = chunk_id
        chunk_results["start_idx"] = start_idx
        chunk_results["end_idx"] = end_idx

        self.save_chunk_results(chunk_id, chunk_results)
        print(f"✅ Chunk {chunk_id + 1} saved!\n")
        return chunk_results

    def merge_chunk_results(self, all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("🔗 MERGING RESULTS FROM ALL CHUNKS")
        print("="*70 + "\n")

        all_predictions = []
        all_ground_truths = []
        all_results = []

        for chunk_idx, chunk_data in enumerate(all_chunks):
            chunk_results = chunk_data.get("results", [])
            all_results.extend(chunk_results)

            for row in chunk_results:
                all_predictions.append(row.get("predicted_answer", ""))
                all_ground_truths.append(row.get("ground_truth_answers", []))

            print(f"Chunk {chunk_idx + 1}: {len(chunk_results)} results added")

        print(f"\n📊 Total results merged: {len(all_results)}")

        from src.utils.metrics import calculate_metrics
        merged_metrics = calculate_metrics(all_predictions, all_ground_truths)
        merged_metrics["results"] = all_results
        merged_metrics["mode"] = self.mode
        merged_metrics["total_samples"] = len(all_results)

        print("\n📈 MERGED METRICS:")
        for metric_name, metric_value in merged_metrics.items():
            if metric_name not in ("results", "predictions", "ground_truths") and isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")

        return merged_metrics

    def run(self, resume: bool = True) -> Dict[str, Any]:
        progress = self.load_progress()

        print("\n" + "="*70)
        print("🚀 CHUNKED EVALUATION PIPELINE")
        print("="*70)
        print(f"📊 Mode: {self.mode}")
        print(f"📊 Subset: {self.subset_file}")
        print(f"📦 Total Samples: {self.total_samples}")
        print(f"📦 Chunk Size: {self.chunk_size}")
        print(f"📦 Total Chunks: {self.num_chunks}")
        print(f"📁 Output: {self.output_dir}")

        if resume and progress.get("completed_chunks"):
            print(f"📋 Resuming — {len(progress['completed_chunks'])} chunks done")
        else:
            print("🔄 Starting fresh evaluation")
        print("="*70 + "\n")

        device = "cuda" if torch.cuda.is_available() else DEVICE
        print(f"💾 Using device: {device}\n")

        inference = DocVQAInference(model_name=MODEL_NAME, device=device, mode=self.mode)

        all_chunks = []
        for chunk_id in range(self.num_chunks):
            if resume and chunk_id in progress.get("completed_chunks", []):
                print(f"⏭️  Skipping chunk {chunk_id + 1} (already processed)")
                chunk_data = self.load_chunk_results(chunk_id)
                if chunk_data:
                    all_chunks.append(chunk_data)
                continue

            try:
                chunk_results = self.process_chunk(chunk_id, inference)
                all_chunks.append(chunk_results)
                progress.setdefault("completed_chunks", []).append(chunk_id)
                self.save_progress(progress)
                inference.cleanup_memory()
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_id + 1}: {e}")
                continue

        if all_chunks:
            merged_results = self.merge_chunk_results(all_chunks)
            with open(self.merged_file, "w") as f:
                json.dump(merged_results, f, indent=2)
            print(f"\n💾 Merged results: {self.merged_file}")
            return merged_results

        print("❌ No chunks were processed successfully!")
        return {}


def run_chunked_evaluation(
    mode: str = "baseline",
    subset_file: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    split: str = "val",
    resume: bool = True,
) -> Dict[str, Any]:
    evaluator = ChunkedEvaluator(
        mode=mode,
        subset_file=subset_file,
        chunk_size=chunk_size,
        split=split,
    )
    return evaluator.run(resume=resume)


if __name__ == "__main__":
    results = run_chunked_evaluation(mode="baseline", resume=True)
    print("\n✅ Chunked evaluation complete!")
