"""
Zero-Shot VQA Inference using Phi-3.5 Vision

This module implements a zero-shot Document VQA system using Phi-3.5 Vision model.
No OCR data or training is used - pure vision-language inference.
"""
import os
import sys
import gc
import types
import importlib
import importlib

# ===========================================================================
# CUDA SETUP: Only set BNB_CUDA_VERSION if quantization is enabled
# This avoids CUDA compatibility issues when quantization is disabled
# ===========================================================================
if os.getenv("USE_QUANTIZATION", "false").lower() == "true":
    os.environ["BNB_CUDA_VERSION"] = "129"

from typing import Dict, List, Any
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# ===========================================================================
# FIX 1: triton.ops mock — bitsandbytes imports triton.ops.matmul_perf_model
# which was removed in modern PyTorch/triton. We create a fake package with
# __path__ set (required for sub-module imports to work without crashing).
# ===========================================================================
try:
    import triton as _triton
    _ops = types.ModuleType('triton.ops')
    _ops.__path__ = []          # CRITICAL: makes it a package, not just a module
    _ops.__package__ = 'triton.ops'
    _ops.__spec__ = importlib.util.spec_from_loader('triton.ops', loader=None)
    sys.modules['triton.ops'] = _ops
    sys.modules.setdefault('triton.ops.matmul_perf_model',
                           types.ModuleType('triton.ops.matmul_perf_model'))
except ImportError:
    pass

# ===========================================================================
# CRITICAL: Patch _check_and_enable_flash_attn_2 on PreTrainedModel itself.
# This is a classmethod — patching the module does NOT work because Python
# resolves `cls._check_and_enable_flash_attn_2()` via the class MRO, not 
# the module namespace. We must patch the class directly, and do it BEFORE
# any model class is imported or used.
# ===========================================================================
from transformers import PreTrainedModel

@classmethod
def _disabled_flash_attn_check(cls, config, *args, **kwargs):
    """Force eager attention. Never enable FlashAttention2 — flash_attn is not installed."""
    if hasattr(config, '_attn_implementation'):
        config._attn_implementation = 'eager'
    return config

PreTrainedModel._check_and_enable_flash_attn_2 = _disabled_flash_attn_check

# Now it is safe to import the model classes
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig

from config.settings import (
    MODEL_NAME, DEVICE, TORCH_DTYPE, ATTN_IMPLEMENTATION, 
    MAX_NEW_TOKENS, TEMPERATURE,
    USE_GRADIENT_CHECKPOINTING, LOW_CPU_MEM_USAGE, MEMORY_CLEANUP_INTERVAL,
    USE_8BIT_QUANTIZATION
)
from src.utils.metrics import calculate_metrics, anls_score
from src.agents.pipeline import AgenticDocVQAPipeline


class InferenceMode:
    VISION_ONLY = "vision_only"
    OCR_ADAPTIVE = "ocr_adaptive"
    BASELINE = "baseline"  # alias for vision_only


MODE_ALIASES = {
    "baseline": InferenceMode.VISION_ONLY,
    "vision_only": InferenceMode.VISION_ONLY,
    "ocr_adaptive": InferenceMode.OCR_ADAPTIVE,
    "adaptive": InferenceMode.OCR_ADAPTIVE,
}


class DocVQAInference:
    """Zero-shot VQA inference using Phi-3.5 Vision"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = None, mode: str = InferenceMode.VISION_ONLY):
        """
        Initialize the model and processor
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on (cuda/cpu). If None, uses config DEVICE
            mode: Inference mode — vision_only/baseline or ocr_adaptive
        """
        self.mode = MODE_ALIASES.get(mode, mode)
        self.agent_pipeline = AgenticDocVQAPipeline() if self.mode == InferenceMode.OCR_ADAPTIVE else None
        self.device = device or DEVICE
        self.model_name = model_name
        
        # 🧹 CRITICAL: Clear CUDA memory before loading model to prevent fragmentation
        if torch.cuda.is_available():
            import gc
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
        
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
        
        # Load directly from HuggingFace (no snapshot_download)
        # HuggingFace hub automatically caches models
        # snapshot_download is incompatible with quantization config
        
        # Configure model loading parameters
        # Note: Phi-3.5 Vision requires BitsAndBytesConfig for quantization
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "max_memory": {0: "15GB", "cpu": "30GB"} if self.device == "cuda" else None,
            "low_cpu_mem_usage": LOW_CPU_MEM_USAGE,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if USE_8BIT_QUANTIZATION and self.device == "cuda":
            # --- TIER 1: bitsandbytes (fastest, but requires matching CUDA binary) ---
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["torch_dtype"] = torch_dtype
                print("🔷 Tier 1: bitsandbytes 4-bit NF4 quantization configured")
            except Exception:
                # --- TIER 2: quanto (pure Python, no CUDA binary needed — works on Kaggle!) ---
                try:
                    from transformers import QuantoConfig
                    model_kwargs["quantization_config"] = QuantoConfig(weights="int4")
                    model_kwargs["torch_dtype"] = torch_dtype
                    print("🔶 Tier 2: quanto int4 quantization configured (no CUDA binary needed)")
                except ImportError:
                    # --- TIER 3: plain float16, no quantization ---
                    print("⬜ Tier 3: No quantization available — using plain float16")
                    model_kwargs["torch_dtype"] = torch_dtype
        else:
            print(f"📍 Using float16 precision (quantization disabled)")
            print("   → Memory optimization via KV cache leak fix (use_cache=False)")
            model_kwargs["torch_dtype"] = torch_dtype

        print(f"📥 Loading from HuggingFace Hub: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            print("✅ Model loaded successfully!")
        except Exception as load_err:
            print(f"❌ Model loading failed with current config: {load_err}")
            if "quantization_config" in model_kwargs:
                print("   → Retrying without quantization...")
                del model_kwargs["quantization_config"]
                
                # CRITICAL: If the first load failed mid-way, PyTorch will still hold the 
                # partially-allocated tensors in GPU memory. We MUST empty the cache before 
                # retrying, otherwise the fallback will instantly throw an OutOfMemoryError.
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                raise
        
        # Enable gradient checkpointing to save memory
        if USE_GRADIENT_CHECKPOINTING and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for memory efficiency")

        # CRITICAL: Force eager attention on the loaded model.
        # Even though we patched the load-time check, the model config may still
        # have _attn_implementation='flash_attention_2' baked in from HuggingFace,
        # which causes `flash_attn_func is not defined` during generate().
        # We must overwrite it on the config AND on every attention sub-module.
        self.model.config._attn_implementation = 'eager'
        for module in self.model.modules():
            if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
                module.config._attn_implementation = 'eager'
            # Some layers store it directly as an attribute
            if hasattr(module, '_attn_implementation'):
                module._attn_implementation = 'eager'
        print("✅ Attention mode forced to: eager (flash_attn disabled)")
        
        # Verify model loaded correctly - patch if missing generate method
        if not hasattr(self.model, 'generate'):
            print("⚠️  Model missing 'generate' method - attempting to patch...")
            try:
                # Try to explicitly inherit from GenerationMixin
                from transformers import GenerationMixin
                # Add GenerationMixin methods to the model
                for attr in dir(GenerationMixin):
                    if not attr.startswith('_') and callable(getattr(GenerationMixin, attr)):
                        if not hasattr(self.model, attr):
                            setattr(self.model, attr, getattr(GenerationMixin, attr).__get__(self.model, type(self.model)))
                
                # Specifically ensure generate is available
                if not hasattr(self.model, 'generate'):
                    raise RuntimeError("Failed to patch generate method")
                print("✅ Successfully patched generate method!")
            except Exception as patch_err:
                raise RuntimeError(
                    f"Model loaded but missing 'generate' method!\n"
                    f"Model type: {type(self.model)}\n"
                    f"Attempted patch failed: {patch_err}\n"
                    f"This may indicate a model loading or transformers version issue.\n"
                    f"Suggestion: Try downgrading transformers to 4.40.0"
                )
        
        self.model.eval()
        print("✅ Model loaded successfully and ready for inference!\n")
    
    def _build_prompt(self, question: str, ocr_snippets: str = None) -> str:
        ocr_block = ""
        if ocr_snippets:
            ocr_block = f"{ocr_snippets}\n"
        return (
            f"<|user|>\n<|image_1|>\n"
            f"Answer briefly with only the exact value or phrase from the document. "
            f"Do not use full sentences.\n"
            f"{ocr_block}"
            f"Question: {question}<|end|>\n<|assistant|>\n"
        )

    def generate_answer(
        self,
        image: Image.Image,
        question: str,
        max_length: int = None,
        ocr_snippets: str = None,
        mode: str = None,
    ) -> str:
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
        
        effective_mode = MODE_ALIASES.get(mode or self.mode, mode or self.mode)
        prompt = self._build_prompt(
            question,
            ocr_snippets if effective_mode != InferenceMode.VISION_ONLY else None,
        )
        
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
    
    def evaluate(self, dataloader, num_samples: int = None, mode: str = None) -> Dict[str, Any]:
        """
        Evaluate on a dataset
        
        Args:
            dataloader: DataLoader for the dataset (may be a Subset slice)
            num_samples: Number of samples to evaluate (None = all in dataloader)
            mode: Override inference mode for this evaluation run
        
        Returns:
            Dictionary with results and metrics
        """
        effective_mode = MODE_ALIASES.get(mode or self.mode, mode or self.mode)
        results = []
        predictions = []
        ground_truths = []
        
        total_samples = len(dataloader) if num_samples is None else min(num_samples, len(dataloader))
        
        print(f"\n📊 Running inference on {total_samples} samples (mode={effective_mode})...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=total_samples, desc="Evaluating")):
                if num_samples and batch_idx >= num_samples:
                    break
                
                for sample in batch:
                    image = sample['image']
                    question = sample['question']
                    ground_truth_answers = sample['answers']
                    question_id = sample['question_id']
                    doc_id = sample['doc_id']

                    ocr_snippet = None
                    pipeline_result = None
                    used_ocr = False

                    if effective_mode == InferenceMode.OCR_ADAPTIVE and self.agent_pipeline:
                        pipeline_result = self.agent_pipeline.prepare(
                            image=image,
                            question=question,
                            question_types=sample.get('question_types', []),
                            ucsf_id=sample.get('ucsf_document_id', ''),
                            page_no=str(sample.get('ucsf_document_page_no', '')),
                        )
                        used_ocr = pipeline_result.used_ocr
                        ocr_snippet = pipeline_result.ocr_snippet

                    predicted_answer = self.generate_answer(
                        image,
                        question,
                        ocr_snippets=ocr_snippet,
                        mode=effective_mode,
                    )

                    anls = anls_score([predicted_answer], ground_truth_answers)

                    result_row = {
                        'question_id': question_id,
                        'doc_id': doc_id,
                        'cohort': sample.get('cohort', ''),
                        'question_types': sample.get('question_types', []),
                        'mode': effective_mode,
                        'question': question,
                        'predicted_answer': predicted_answer,
                        'ground_truth_answers': ground_truth_answers,
                        'anls_score': anls,
                        'used_ocr': used_ocr,
                    }

                    if pipeline_result:
                        audit = pipeline_result.to_audit_dict()
                        if audit.get('routing'):
                            result_row['routing'] = audit['routing']
                        if audit.get('retrieval'):
                            result_row['retrieval'] = audit['retrieval']
                        if audit.get('formatting'):
                            result_row['formatting'] = audit['formatting']

                    results.append(result_row)
                    predictions.append(predicted_answer)
                    ground_truths.append(ground_truth_answers)
                
                if (batch_idx + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                    self.cleanup_memory()
        
        metrics = calculate_metrics(predictions, ground_truths)
        metrics['results'] = results
        metrics['predictions'] = predictions
        metrics['ground_truths'] = ground_truths
        metrics['mode'] = effective_mode
        
        return metrics


if __name__ == "__main__":
    # Quick test
    print("Testing DocVQAInference...")
    inference = DocVQAInference()
    print("✅ Inference engine initialized!")
