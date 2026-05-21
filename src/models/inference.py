"""
Zero-Shot VQA Inference using Phi-3.5 Vision

This module implements a zero-shot Document VQA system using Phi-3.5 Vision model.
No OCR data or training is used - pure vision-language inference.
"""

import torch
from typing import Dict, List, Any
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

# Attempt to disable FlashAttention2 validation check
# Different transformers versions have different internal function names
try:
    import transformers.modeling_utils as modeling_utils
    
    # Try to find and patch the FlashAttention2 check function
    if hasattr(modeling_utils, '_flash_attn_2_can_dispatch'):
        original_check = modeling_utils._flash_attn_2_can_dispatch
        modeling_utils._flash_attn_2_can_dispatch = lambda self, is_init_check=False: False
    elif hasattr(modeling_utils, 'is_flash_attn_2_available'):
        original_check = modeling_utils.is_flash_attn_2_available
        modeling_utils.is_flash_attn_2_available = lambda: False
    
    # Also try to patch at the generation_utils level
    import transformers.generation.utils as gen_utils
    if hasattr(gen_utils, '_flash_attn_2_can_dispatch'):
        gen_utils._flash_attn_2_can_dispatch = lambda self, is_init_check=False: False
except Exception as patch_error:
    # If patching fails, that's okay - we'll handle it in model loading
    print(f"⚠️  Warning: Could not patch FlashAttention2 check: {patch_error}")

from config.settings import (
    MODEL_NAME, DEVICE, TORCH_DTYPE, ATTN_IMPLEMENTATION, 
    MAX_NEW_TOKENS, TEMPERATURE
)
from src.utils.metrics import calculate_metrics, anls_score


class DocVQAInference:
    """Zero-shot VQA inference using Phi-3.5 Vision"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        """
        Initialize the model and processor
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on (cuda/cpu). If None, uses config DEVICE
        """
        self.device = device or DEVICE
        self.model_name = model_name
        
        print(f"🚀 Loading model: {model_name}")
        print(f"📍 Device: {self.device}")
        
        # Load model with appropriate dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("Loading model...")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = None
        
        # Strategy 1: Try loading with eager attention
        try:
            print("Strategy 1: Attempting to load with eager attention...")
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config._attn_implementation = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                config=config,
                device_map="auto" if self.device == "cuda" else None
            )
            print("✅ Strategy 1: Loaded with eager attention!\n")
        except Exception as e:
            print(f"⚠️  Strategy 1 failed: {str(e)[:100]}...\n")
            self.model = None
        
        # Strategy 2: Try loading on CPU without specifying attention
        if self.model is None:
            try:
                print("Strategy 2: Loading on CPU (no attention override)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="cpu"
                )
                # Move to target device
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                print("✅ Strategy 2: Loaded on CPU and moved to device!\n")
            except Exception as e:
                print(f"⚠️  Strategy 2 failed: {str(e)[:100]}...\n")
                self.model = None
        
        # Strategy 3: Try with attn_implementation parameter directly
        if self.model is None:
            try:
                print("Strategy 3: Loading with attn_implementation='eager' parameter...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                    device_map="cpu"
                )
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                print("✅ Strategy 3: Loaded with parameter!\n")
            except Exception as e:
                print(f"⚠️  Strategy 3 failed: {str(e)[:100]}...\n")
                self.model = None
        
        # If all strategies fail, raise error
        if self.model is None:
            print("❌ All loading strategies failed!")
            raise RuntimeError(
                "Could not load model. Please ensure you have:\n"
                "1. Sufficient GPU memory (8GB+)\n"
                "2. Latest transformers library (pip install --upgrade transformers)\n"
                "3. Valid HuggingFace model access"
            )
        
        self.model.eval()
        print("✅ Model initialized and ready for inference!\n")
    
    def generate_answer(self, image: Image.Image, question: str, max_length: int = None) -> str:
        """
        Generate answer for a given image and question
        
        Args:
            image: PIL Image object
            question: Question text
            max_length: Maximum length of generated answer. If None, uses config MAX_NEW_TOKENS
        
        Returns:
            Generated answer string
        """
        if max_length is None:
            max_length = MAX_NEW_TOKENS
        
        # Create prompt in Phi-3.5 Vision format
        prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with compatibility fixes for different transformers versions
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=False,  # Disable cache to avoid DynamicCache compatibility issues
                )
            
            # Decode
            response = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            # Extract answer (after "assistant" token)
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response.strip()
            
            return answer
        
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return ""
    
    def evaluate(self, dataloader, num_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate on a dataset
        
        Args:
            dataloader: DataLoader for the dataset
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            Dictionary with results and metrics
        """
        results = []
        predictions = []
        ground_truths = []
        
        total_samples = len(dataloader) if num_samples is None else min(num_samples, len(dataloader))
        
        print(f"\n📊 Running inference on {total_samples} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=total_samples, desc="Evaluating")):
                if num_samples and batch_idx >= num_samples:
                    break
                
                # Batch contains list of samples
                for sample in batch:
                    image = sample['image']
                    question = sample['question']
                    ground_truth_answers = sample['answers']
                    question_id = sample['question_id']
                    doc_id = sample['doc_id']
                    
                    # Generate answer
                    predicted_answer = self.generate_answer(image, question)
                    
                    # Calculate ANLS score
                    anls = anls_score([predicted_answer], ground_truth_answers)
                    
                    # Store results
                    results.append({
                        'question_id': question_id,
                        'doc_id': doc_id,
                        'question': question,
                        'predicted_answer': predicted_answer,
                        'ground_truth_answers': ground_truth_answers,
                        'anls_score': anls,
                    })
                    
                    predictions.append(predicted_answer)
                    ground_truths.append(ground_truth_answers)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        metrics['results'] = results
        
        return metrics


if __name__ == "__main__":
    # Quick test
    print("Testing DocVQAInference...")
    inference = DocVQAInference()
    print("✅ Inference engine initialized!")
