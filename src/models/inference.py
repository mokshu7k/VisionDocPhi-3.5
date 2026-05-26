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
    MAX_NEW_TOKENS, TEMPERATURE,
    USE_GRADIENT_CHECKPOINTING, LOW_CPU_MEM_USAGE, MEMORY_CLEANUP_INTERVAL,
    USE_8BIT_QUANTIZATION
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
        
        # Use snapshot_download to cache model locally first.
        # This avoids the huggingface_hub strict repo ID validation bug
        # that breaks trust_remote_code loading for phi-3.5-vision-instruct
        # when a newer huggingface_hub version is installed.
        # NOTE: Skip snapshot_download if using quantization (compatibility issue)
        load_path = model_name
        if not (USE_8BIT_QUANTIZATION and self.device == "cuda"):
            try:
                from huggingface_hub import snapshot_download
                print("📥 Downloading model snapshot to local cache...")
                local_model_path = snapshot_download(
                    repo_id=model_name,
                    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot"]
                )
                print(f"✅ Model cached at: {local_model_path}")
                load_path = local_model_path
            except Exception as snap_err:
                print(f"⚠️  snapshot_download failed ({snap_err}), falling back to direct load...")
                load_path = model_name

        # Load from the model path
        # Note: Phi-3.5 Vision requires BitsAndBytesConfig for quantization
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
            "attn_implementation": "eager",
            "_attn_implementation": "eager",
            "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
        }
        
        if USE_8BIT_QUANTIZATION and self.device == "cuda":
            print("⚙️  Using 8-bit quantization via bitsandbytes...")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                # torch_dtype is typically determined by quantization, but we can still set it
                model_kwargs["torch_dtype"] = torch_dtype
            except Exception as quant_err:
                print(f"⚠️  Quantization config failed: {quant_err}")
                print("   Falling back to full precision (float16)...")
                model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                **model_kwargs
            )
        except Exception as load_err:
            print(f"❌ Model loading failed: {load_err}")
            print("   Retrying without quantization...")
            # Remove quantization config if present
            if "quantization_config" in model_kwargs:
                del model_kwargs["quantization_config"]
            self.model = AutoModelForCausalLM.from_pretrained(
                load_path,
                **model_kwargs
            )
        
        # Enable gradient checkpointing to save memory
        if USE_GRADIENT_CHECKPOINTING and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for memory efficiency")
        
        # Verify model loaded correctly
        if not hasattr(self.model, 'generate'):
            raise RuntimeError(
                f"Model loaded but missing 'generate' method!\n"
                f"Model type: {type(self.model)}\n"
                f"This may indicate a model loading or version issue."
            )
        
        self.model.eval()
        print("✅ Model loaded successfully and ready for inference!\n")
    
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
        prompt = f"<|user|>\n<|image_1|>\nAnswer briefly with only the exact value or phrase from the document. Do not use full sentences.\nQuestion: {question}<|end|>\n<|assistant|>\n"
        
        try:
            # Step 1: Process inputs
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            except Exception as proc_error:
                print(f"❌ Processor error: {proc_error}")
                return ""
            
            # Step 2: Move to device
            try:
                inputs = inputs.to(self.device)
            except Exception as device_error:
                print(f"❌ Device move error: {device_error}")
                return ""
            
            # Step 3: Generate tokens
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
            except Exception as gen_error:
                print(f"❌ Generation error: {gen_error}")
                # Try alternative without explicit pad token
                try:
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_length,
                            do_sample=False,
                        )
                except Exception as gen_error2:
                    print(f"❌ Generation retry failed: {gen_error2}")
                    return ""
            
            # Step 4: Decode output
            try:
                # Get only the new tokens (excluding input tokens)
                input_len = inputs["input_ids"].shape[1]
                output_tokens = output_ids[0][input_len:].cpu().numpy()
                
                # Filter out tokens that might cause decode errors
                valid_tokens = [t for t in output_tokens if 0 <= t < self.processor.tokenizer.vocab_size]
                
                if not valid_tokens:
                    print(f"❌ All output tokens are invalid")
                    return ""
                
                # Decode using tokenizer directly (more reliable than processor.decode)
                response = self.processor.tokenizer.decode(
                    valid_tokens,
                    skip_special_tokens=True
                )
            except Exception as decode_error:
                print(f"❌ Decode error (attempt 1): {decode_error}")
                # Try alternative: use processor.decode without filtering
                try:
                    response = self.processor.decode(
                        output_ids[0][input_len:],
                        skip_special_tokens=True
                    )
                except Exception as decode_error2:
                    print(f"❌ Decode error (attempt 2): {decode_error2}")
                    # Last resort: just return empty string
                    return ""
            
            # Step 5: Extract and clean answer
            # The model sometimes repeats the question or includes it in the response
            # We need to extract just the core answer without being too aggressive
            answer = response.strip()
            
            # Since the model sometimes hallucinates extra text (like "Here is an article...", "Instruction 1:"),
            # we should take the first non-empty line as our answer, because we asked for a brief exact value.
            response_lines = [line.strip() for line in response.split('\n') if line.strip()]
            if response_lines:
                answer = response_lines[0]
                
            # If the model hallucinates "Here is..." or "Instruction..." on the same line, try to split by those markers
            for marker in [" Here is ", " Instruction ", " Article:", " Summary:", " Here is a task"]:
                if marker in answer:
                    answer = answer.split(marker)[0].strip()
            
            # Remove common filler phrases if they dominate the start
            filler_phrases = [
                "the answer is ",
                "based on the image, ",
                "according to the document, ",
                "looking at ",
                "from the document, ",
            ]
            
            answer_lower = answer.lower()
            for filler in filler_phrases:
                if answer_lower.startswith(filler):
                    answer = answer[len(filler):].strip()
                    break
            
            return answer
        
        except Exception as e:
            print(f"❌ Unexpected error in generate_answer: {e}")
            return ""
    
    def cleanup_memory(self):
        """Clear GPU memory cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
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
                
                # Periodic memory cleanup
                if (batch_idx + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                    self.cleanup_memory()
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, ground_truths)
        metrics['results'] = results
        
        return metrics


if __name__ == "__main__":
    # Quick test
    print("Testing DocVQAInference...")
    inference = DocVQAInference()
    print("✅ Inference engine initialized!")
