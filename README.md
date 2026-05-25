# VisionDocPhi-3.5: DocVQA Zero-Shot Baseline with Phi-3.5 Vision

Production-ready implementation of a zero-shot Document Visual Question Answering (VQA) system using the **Phi-3.5 Vision** model on the **DocVQA** dataset.

## 📋 Project Structure

```
VisionDocPhi-3.5/
├── config/                      # Configuration management
│   ├── __init__.py
│   └── settings.py              # Main configuration file
├── src/                         # Source code (main package)
│   ├── models/                  # Model inference code
│   │   ├── __init__.py
│   │   └── inference.py         # DocVQAInference class
│   ├── data/                    # Data loading and processing
│   │   ├── __init__.py
│   │   └── dataset.py           # DocVQADataset, dataloader utilities
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   └── metrics.py           # ANLS, exact match, other metrics
│   └── pipelines/               # End-to-end pipelines
│       ├── __init__.py
│       └── baseline.py          # Zero-shot baseline pipeline
├── scripts/                     # Executable scripts
│   ├── baseline_evaluation.py   # Main evaluation script
│   └── verify_setup.py          # Setup verification script
├── notebooks/                   # Jupyter notebooks
│   └── DocVQA_Colab_Production.ipynb  # Google Colab notebook
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_metrics.py          # Metrics tests
├── data/                        # Data directory
│   ├── raw/                     # Raw data
│   │   ├── spdocvqa_images/     # Document images
│   │   └── spdocvqa_qas/        # JSON annotations
│   └── outputs/                 # Results and predictions
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── pyproject.toml               # Project metadata
└── README.md                    # This file
```

## 🎯 About the Project

This is a **Phase 1 zero-shot baseline** for the DocVQA task, meaning:
- ❌ NO fine-tuning on DocVQA data
- ❌ NO OCR coordinate information
- ✅ ONLY raw image + vision language model
- ✅ Direct answer prediction

This establishes a foundation before adding OCR in Phase 2.

## 📊 About the Dataset

**DocVQA** is a dataset of single-page document images with VQA annotations.

### Dataset Format
- **Images**: Document images (PNG/JPEG) in `data/raw/spdocvqa_images/`
- **Annotations**: JSON files in `data/raw/spdocvqa_qas/`
  - `train_v1.0_withQT.json` - Training set (~39K QA pairs)
  - `val_v1.0_withQT.json` - Validation set (~5.4K QA pairs)
  - `test_v1.0.json` - Test set (ground truth not public)

### Annotation Structure
```json
{
  "questionId": 12345,
  "question": "What is the total amount?",
  "question_types": ["numerical", "layout"],
  "image": "documents/document_1.png",
  "answers": ["$100.00", "100.00"],
  "docId": 123,
  "data_split": "train"
}
```

## 🚀 Zero-Shot Approach

**What is Zero-Shot?**
- No training on DocVQA data
- No OCR information or spatial coordinates
- Pure visual understanding using the vision-language model
- Direct question-answering from image pixels

**Why Zero-Shot First?**
- Establishes a baseline performance floor
- Tests the model's inherent VQA capabilities
- Identifies gaps that prompt engineering (Phase 2) aims to fill

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have CUDA installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure for Your System

Edit `config.py` to set:
- `DEVICE`: Use `"cuda"` for GPU or `"cpu"` for CPU-only
- Model parameters if needed

## Running the Baseline

### Standard Evaluation

```bash
python scripts/baseline_evaluation.py --split val
```

### Chunked Evaluation (Recommended for Large Datasets)

For processing large datasets on systems with limited GPU memory (Google Colab, etc.):

```bash
# Process in chunks of 200 samples
python scripts/chunked_evaluation.py --split val --chunk_size 200

# Or resume from previous checkpoint
python scripts/chunked_evaluation.py --split val --chunk_size 200 --resume

# Start fresh (ignore previous checkpoints)
python scripts/chunked_evaluation.py --split val --chunk_size 200 --no-resume
```

**Chunked Processing Features:**
- ✅ Automatically splits dataset into manageable chunks
- ✅ Saves progress after each chunk
- ✅ Can be interrupted and resumed
- ✅ Automatically merges final results
- ✅ Perfect for Google Colab (restart runtime between chunks)

**Workflow for Colab:**
```python
# Cell 1: Run chunk processing
!python scripts/chunked_evaluation.py --split val --chunk_size 200

# If interrupted, restart runtime and run again - it will resume automatically!
# Cell 2: (after restart)
!python scripts/chunked_evaluation.py --split val --chunk_size 200 --resume
```

### Limited Samples (Testing)

```bash
python scripts/baseline_evaluation.py --split val --num_samples 100
```

## Output

The script generates:

1. **Predictions JSON** (`outputs/predictions_zeroshot.json`)
   - Contains predicted answers, ground truth, and ANLS scores for each sample

2. **Results JSON** (`outputs/results_zeroshot.json`)
   - Summary metrics:
     - **ANLS**: Average Normalized Levenshtein Similarity (0-1 scale)
     - **Exact Match**: Percentage of exact match answers
     - **Total Samples**: Number of evaluated samples

3. **Console Output**
   - Real-time progress with TQDM
   - Sample predictions for inspection

## Metrics Explanation

### ANLS (Average Normalized Levenshtein Similarity)
- Measures similarity between predicted and ground truth answers
- Range: 0-1 (higher is better)
- Formula: `ANLS = max(0, (1 - (edit_distance / max_len)) - 0.5) / 0.5`
- Normalized by max string length to handle various answer lengths
- Threshold (0.5) allows partial credit for near-matches

### Exact Match
- Simple percentage of answers that exactly match ground truth (after normalization)
- Less informative but easier to interpret

## Model Details

**Phi-3.5 Vision**
- Model size: 4.2B parameters
- Vision encoder: Handles document images
- Language model: Generates answers
- Context length: 2048 tokens
- Input format: Supports multi-modal prompts with images and text

## Understanding the Results

A typical zero-shot baseline on DocVQA:
- **ANLS**: 0.30 - 0.50 (varies by question type)
- **Exact Match**: 0.10 - 0.30

**Why is the zero-shot performance lower?**
- No OCR grounding (spatial/textual anchors)
- Limited context understanding of document structure
- Difficulty with numerical/exact matching
- Generic prompting without domain adaptation

**Next Phase (Phase 2): Prompt Engineering with OCR**
- Add OCR metadata with spatial coordinates
- Improve prompt instructions for document understanding
- Expected improvement: ANLS → 0.50-0.70+

## Troubleshooting

### GPU Memory Optimization for Google Colab

If you encounter GPU out-of-memory errors in Colab, use **chunked processing** (the most effective and reliable approach):

### 🎯 Best Solution: Chunked Processing

Process large datasets in manageable chunks that fit in GPU memory:

```bash
python scripts/chunked_evaluation.py --split val --chunk_size 200
```

**Why chunked processing is best:**
- ✅ Automatically saves progress after each chunk
- ✅ Can be interrupted and resumed without data loss
- ✅ Works reliably across Colab runtime restarts
- ✅ Merges all results automatically
- ✅ No additional dependencies or complex setup

**Example Colab workflow:**
```python
# Run in Colab cell
!cd /content/VisionDocPhi-3.5 && python scripts/chunked_evaluation.py --split val --chunk_size 200

# If interrupted or runtime crashes:
# Just run the same command again - it will resume from the last completed chunk!
```

---

### Additional Memory Optimizations (Already Enabled)

These are automatically applied:

#### 1. **Gradient Checkpointing** (~10-20% memory savings)
- Reduces peak memory during inference
- Already enabled by default

#### 2. **Eager Attention** (~5-10% memory savings)
- More memory efficient than other attention mechanisms
- Already enabled for Colab compatibility

#### 3. **Memory Cleanup** (prevents accumulation)
- Automatically clears GPU cache every 5 batches
- Already enabled

#### 4. **Batch Size = 1** (significant!)
- Processes one image at a time
- Already configured

**Total from these optimizations: ~20-30% memory savings** ✓

---

### Advanced: Manual Chunking Without Script

If you need more control, manually split processing:

```bash
# Process only first 200 samples
python scripts/baseline_evaluation.py --split val --num_samples 200

# [Restart Colab runtime]

# Process samples 200-400
python scripts/baseline_evaluation.py --split val --num_samples 200
```

Then manually merge results.

---

### Configuration (in `config/settings.py`)

```python
CHUNK_SIZE = 200                 # Samples per chunk
ENABLE_CHUNKED_MODE = False      # Manual flag (scripts use it automatically)
RESUME_FROM_CHECKPOINT = True    # Auto-resume from interruptions
```

**Note on 8-bit Quantization:** Phi-3.5 Vision model does not support 8-bit quantization. Chunked processing is the recommended memory solution.

### Out of Memory (OOM) Error
- Reduce `BATCH_SIZE` in `config.py` (keep at 1 for safety)
- Reduce `MAX_LENGTH` in `config.py`
- Use `--num_samples` to test on fewer examples
- Consider using `torch_dtype=torch.float32` for lower memory

### Image Loading Errors
- Ensure image paths in JSON match actual file paths
- Check that `spdocvqa_images/` contains all referenced images
- Some corrupted images are auto-handled with white placeholder

### Model Download Issues
- First run downloads the model (~8-10GB for Phi-3.5)
- Ensure stable internet connection
- Models cached in `~/.cache/huggingface/`

## Code Organization

### `config.py`
- All paths and hyperparameters
- Single source of truth for configuration

### `data/docvqa_loader.py`
- `DocVQADataset`: PyTorch Dataset class
- `create_dataloader`: Convenience function for DataLoader
- `get_dataset_stats`: Dataset statistics

### `utils/metrics.py`
- `anls_score()`: Calculate ANLS for single sample
- `calculate_metrics()`: Aggregate metrics
- Text normalization for fair comparison

### `baseline_zero_shot.py`
- `DocVQAInference`: Model wrapper for inference
- `main()`: Evaluation loop
- Argument parsing for CLI

## Next Steps (Phase 2)

After establishing the zero-shot baseline:

1. **OCR Integration**
   - Extract text from images using OCR (Tesseract/PaddleOCR)
   - Store coordinates and text content
   - Create structured metadata

2. **Prompt Engineering**
   - Include OCR results in prompts
   - Add spatial reasoning instructions
   - Test different prompt templates

3. **Fine-tuning** (Optional)
   - Add small training loop with DocVQA data
   - Compute forward/backward passes
   - Save fine-tuned weights

4. **Deployment**
   - Create Streamlit/Gradio UI
   - Add real-time inference
   - Support document upload and Q&A

## References

- **DocVQA Dataset**: https://www.docvqa.org/
- **Phi-3.5 Vision**: https://huggingface.co/microsoft/phi-3.5-vision-instruct
- **ANLS Metric**: Reference implementation from VQA literature

## License

This implementation is for research purposes. Refer to the DocVQA dataset license and Phi-3.5 model license.

---

**Questions or Issues?**
Check the code comments in each file for more details.
