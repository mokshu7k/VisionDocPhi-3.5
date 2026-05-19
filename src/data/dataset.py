"""
DocVQA Dataset Loader

Handles loading the Single Page Document VQA dataset with proper data structures
and batch processing.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class DocVQADataset(Dataset):
    """
    Single Page Document VQA Dataset
    
    Loads document images and their corresponding VQA annotations.
    """
    
    def __init__(self, annotations_file: str, image_dir: str, split: str = "train"):
        """
        Args:
            annotations_file: Path to JSON annotation file
            image_dir: Path to directory containing document images
            split: Dataset split (train, val, test)
        """
        self.annotations_file = Path(annotations_file)
        self.image_dir = Path(image_dir)
        self.split = split
        
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        # Extract data samples from nested or flat structure
        self.samples = self._extract_samples(data)
        
        print(f"✓ Loaded {len(self.samples)} samples from '{split}' split")
    
    def _extract_samples(self, data: Dict) -> List[Dict]:
        """Extract samples from JSON in nested or flat structure"""
        samples = []
        
        if isinstance(data, dict):
            # Check for nested structure with 'data' key
            if 'data' in data and isinstance(data['data'], list):
                samples = data['data']
            else:
                # Try flat structure (each value could be a sample)
                for key, value in data.items():
                    if isinstance(value, dict) and 'questionId' in value:
                        samples.append(value)
        elif isinstance(data, list):
            # Direct list structure
            samples = data
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample
        
        Returns:
            Dict containing:
                - image: PIL Image
                - question: str
                - answers: List[str]
                - question_types: List[str]
                - image_path: str
                - question_id: int
                - doc_id: int
        """
        sample = self.samples[idx]
        
        # Load image - handle path prefixes
        image_file = sample['image']
        if image_file.startswith('documents/'):
            image_file = image_file.replace('documents/', '', 1)
        
        image_path = self.image_dir / image_file
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            # Create dummy image on error
            image = Image.new('RGB', (224, 224), color='white')
        
        return {
            'image': image,
            'question': sample['question'],
            'answers': sample['answers'],
            'question_types': sample.get('question_types', []),
            'image_path': str(sample['image']),
            'question_id': sample['questionId'],
            'doc_id': sample['docId'],
        }


def create_dataloader(
    annotations_file: str,
    image_dir: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False
) -> DataLoader:
    """
    Create a DataLoader for DocVQA dataset
    
    Args:
        annotations_file: Path to JSON annotation file
        image_dir: Path to directory containing images
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size (typically 1 for variable image sizes)
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the dataset
    
    Returns:
        DataLoader instance
    """
    dataset = DocVQADataset(annotations_file, image_dir, split=split)
    
    def collate_fn(batch):
        """Custom collate function to handle PIL images"""
        return batch
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return dataloader


def get_dataset_stats(annotations_file: str) -> Dict[str, Any]:
    """
    Get statistics about the dataset
    
    Args:
        annotations_file: Path to JSON annotation file
    
    Returns:
        Dictionary with dataset statistics
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Extract samples
    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            samples = data['data']
        else:
            samples = [v for v in data.values() if isinstance(v, dict) and 'questionId' in v]
    elif isinstance(data, list):
        samples = data
    else:
        samples = []
    
    # Collect question types
    question_types = {}
    for sample in samples:
        for qtype in sample.get('question_types', []):
            question_types[qtype] = question_types.get(qtype, 0) + 1
    
    return {
        'total_samples': len(samples),
        'question_types': question_types,
        'num_question_types': len(question_types)
    }


if __name__ == "__main__":
    # Test dataset loading
    print("Testing DocVQADataset...")
    from config.settings import VAL_ANNOTATIONS, IMAGES_DIR
    
    dataset = DocVQADataset(str(VAL_ANNOTATIONS), str(IMAGES_DIR), split='val')
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Question: {sample['question']}")
    print(f"  Answers: {sample['answers']}")
    print(f"  Image size: {sample['image'].size}")
