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
# Note: Phi-3.5 Vision model does not support load_in_8bit parameter
# Instead, we use gradient checkpointing, eager attention, and chunked processing

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
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
