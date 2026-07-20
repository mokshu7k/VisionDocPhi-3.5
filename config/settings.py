"""
Configuration Settings for DocVQA Zero-Shot Baseline
Environment-specific configurations
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
ANNOTATIONS_DIR = RAW_DATA_DIR / "spdocvqa_qas"
IMAGES_DIR = RAW_DATA_DIR / "spdocvqa_images"

TRAIN_ANNOTATIONS = ANNOTATIONS_DIR / "train_v1.0_withQT.json"
VAL_ANNOTATIONS = ANNOTATIONS_DIR / "val_v1.0_withQT.json"
TEST_ANNOTATIONS = ANNOTATIONS_DIR / "test_v1.0.json"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Model identifier from HuggingFace Hub
MODEL_NAME = "microsoft/phi-3.5-vision-instruct"

# Device configuration
DEVICE = os.getenv("DEVICE", "cpu")  # Can be set via env: DEVICE=cuda
if DEVICE == "cuda":
    # GPU settings
    TORCH_DTYPE = "float16"
    ATTN_IMPLEMENTATION = "eager"  # Use eager instead of flash_attention_2 for memory efficiency
else:
    # CPU settings
    TORCH_DTYPE = "float32"
    ATTN_IMPLEMENTATION = "eager"

# Inference configuration
MAX_LENGTH = 2048
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 128

# ============================================================================
# GPU MEMORY OPTIMIZATION
# ============================================================================
# Quantization memory estimates:
#   float16 (no quant) : ~8.0 GB  — fits on T4 (15GB)
#   4-bit NF4 quant    : ~2.5 GB  — fits on P100/T4 with huge headroom
# 
# ⚠️  KAGGLE: Set to False to avoid CUDA 13.x compatibility issues
# Use float16 instead + KV cache memory leak fix (use_cache=False)
# 
# MEMORY LEAK FIX: use_cache=False in generate() prevents KV cache accumulation
# With the leak fixed, float16 alone (8 GB) fits comfortably on Kaggle's 15GB T4
# Kaggle: set USE_8BIT_QUANTIZATION=false (env) to avoid bitsandbytes CUDA issues
USE_8BIT_QUANTIZATION = os.getenv("USE_8BIT_QUANTIZATION", "false").lower() in ("true", "1", "yes")

# Enable gradient checkpointing to reduce memory during inference
USE_GRADIENT_CHECKPOINTING = True

# Memory cleanup frequency (clear cache every N batches)
MEMORY_CLEANUP_INTERVAL = 5

# Use low_cpu_mem_usage for model loading
LOW_CPU_MEM_USAGE = True

# ============================================================================
# DATALOADER CONFIGURATION
# ============================================================================
BATCH_SIZE = 1  # Process one at a time due to variable image sizes
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Default split for evaluation
EVAL_SPLIT = "val"  # Can be "val" or "test"

# ANLS metric configuration
ANLS_THRESHOLD = 0.5  # Threshold for exact match in ANLS

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Save predictions and results
SAVE_PREDICTIONS = True
PREDICTIONS_FILE = OUTPUT_DIR / "predictions_zeroshot.json"
RESULTS_FILE = OUTPUT_DIR / "results_zeroshot.json"

# ============================================================================
# OCR & AGENTIC PIPELINE CONFIGURATION
# ============================================================================
OCR_DIR = RAW_DATA_DIR / "spdocvqa_ocr"
OCR_CACHE_DIR = DATA_DIR / "cache" / "ocr_embeddings"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SUBSET_SIZE = 200
EVAL_SUBSET_FILE = OUTPUT_DIR / "eval_subset_200.json"
LAYOUT_HEAVY_COUNT = 100
VISUAL_HEAVY_COUNT = 100
EVAL_SUBSET_SEED = 42

LAYOUT_HEAVY_TYPES = frozenset({
    "layout", "table/list", "form", "handwritten", "others",
})
VISUAL_HEAVY_TYPES = frozenset({
    "figure/diagram", "Image/Photo", "Yes/No", "free_text",
})
OCR_CAUTIOUS_TYPES = frozenset({
    "figure/diagram", "free_text",
})

OCR_MAX_CHARS = 1200
OCR_MAX_LINES = 25
OCR_MIN_SCORE = 0.20
OCR_TOP_K = 40
OCR_GATE_MIN_SCORE = 0.45
HYBRID_ALPHA = 0.7
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NEIGHBOR_Y_GAP = 45
COLUMN_X_PAD = 20
Y_OVERLAP_TOLERANCE = 6
ENABLE_CONTEXT_EXPANSION = True

QUERY_STOPWORDS = frozenset({
    "A", "AN", "AND", "ARE", "AT", "FOR", "IN", "IS", "OF", "ON",
    "OR", "THE", "TO", "WAS", "WHAT", "WHICH", "WHO", "WHOM",
})

FIELD_LABEL_KEYWORDS = frozenset({
    "COLLEGE", "NAME", "SPECIMEN", "TYPE", "ID", "NO", "DATE",
    "ADDRESS", "PHONE", "EMAIL", "DEPARTMENT", "SCHOOL", "TITLE",
})

OCR_BOILERPLATE_DENYLIST = frozenset({
    "INSTRUCTION", "NOTICE", "USER",
})

DOCVQA_MAX_NEW_TOKENS = 20
DOCVQA_STOP_STRINGS = ["\n", "#", "<|user|>", "<|end|>", "<document_ocr_context>"]

DENSITY_EDGE_THRESHOLD = 0.12
DENSITY_EDGE_MID = 0.08
DENSITY_EDGE_VERY_HIGH = 0.18
DENSITY_RESOLUTION_THRESHOLD = 2000
DENSITY_CONTRAST_THRESHOLD = 35.0
DENSITY_ANALYSIS_MAX_DIM = 1024

LAYOUT_KEYWORDS = frozenset({
    "row", "column", "table", "total", "amount", "sum",
    "left", "right", "above", "below", "header", "footer", "field", "box",
})

# ============================================================================
# CHUNKED PROCESSING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 20  # 10 chunks × 20 = 200 eval subset
ENABLE_CHUNKED_MODE = False
RESUME_FROM_CHECKPOINT = True


def get_mode_output_dir(mode: str, version: str = "") -> Path:
    """Return output directory for baseline or ocr_adaptive evaluation."""
    mode_key = mode.replace("vision_only", "baseline")
    suffix = f"_{version}" if version else ""
    if mode_key == "baseline":
        return OUTPUT_DIR / f"baseline_200{suffix}"
    if mode_key in ("ocr_adaptive", "adaptive"):
        return OUTPUT_DIR / f"ocr_adaptive_200{suffix}"
    return OUTPUT_DIR / f"{mode_key}_200{suffix}"


# Eval harness output (taxonomy + paired ANLS/EM + latency reports)
HARNESS_DIR = OUTPUT_DIR / "harness"
HARNESS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
