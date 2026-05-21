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

from transformers import AutoModelForCausalLM, AutoProcessor

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
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with appropriate dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = None
        
        # Strategy 1: Try with FlashAttention2 first (fastest on CUDA if available)
        if self.device == "cuda":
            try:
                print("Strategy 1: Attempting to load with FlashAttention2...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map="auto"
                )
                print("✅ Strategy 1: Loaded with FlashAttention2\n")
            except (ImportError, RuntimeError) as e:
                if "flash_attn" in str(e).lower() or "Flash" in str(e):
                    print(f"⚠️  Strategy 1 failed: FlashAttention2 not available\n")
                    self.model = None
                else:
                    raise
        
        # Strategy 2: Load on CPU first with eager attention, then move to GPU
        # This bypasses FlashAttention2 validation during loading
        if self.model is None:
            try:
                print("Strategy 2: Loading on CPU with eager attention, then moving to device...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="cpu",  # Load on CPU to avoid FA2 validation on GPU
                    attn_implementation="eager"
                )
                # Now move to target device
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                print("✅ Strategy 2: Loaded on CPU and moved to device\n")
            except Exception as e:
                print(f"⚠️  Strategy 2 failed: {str(e)[:80]}...\n")
                self.model = None
        
        # Strategy 3: Load on CPU with sdpa attention
        if self.model is None:
            try:
                print("Strategy 3: Loading on CPU with sdpa attention...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="cpu",
                    attn_implementation="sdpa"
                )
                # Move to target device
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                print("✅ Strategy 3: Loaded with sdpa on CPU and moved to device\n")
            except Exception as e:
                print(f"⚠️  Strategy 3 failed: {str(e)[:80]}...\n")
                self.model = None
        
        # Strategy 4: Load on CPU with default attention
        if self.model is None:
            try:
                print("Strategy 4: Loading on CPU with default attention...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map="cpu"
                )
                # Move to target device
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                print("✅ Strategy 4: Loaded with default attention on CPU and moved to device\n")
            except Exception as e:
                print(f"❌ All strategies failed. Last error:\n{str(e)[:200]}\n")
                raise
        
        # Move to device (skip if device_map was used)
        if self.device != "cuda" or not hasattr(self.model, 'device_map'):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ Model loaded successfully!\n")
    
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
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    temperature=TEMPERATURE,
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
