# VisionDocPhi-3.5 Kaggle Setup Guide (Updated)

## Quick Summary
- ✅ **Use float16 precision** (not quantization - avoids CUDA compatibility issues)
- ✅ **Memory leak fix is built-in** (`use_cache=False` in model.generate())
- ✅ **8.5 GB VRAM usage** ← fits easily on Kaggle's T4 (15GB)
- ✅ **No bitsandbytes needed** ← avoids CUDA 13.x dependency errors

## Why This Works

### The Problem (Original)
- Kaggle's CUDA 13.x libraries are missing (`libnvJitLink.so.13`)
- bitsandbytes needs these libraries to compile quantization kernels
- Without them, the model tries to load in float32 → uses 25GB RAM → OOM crash

### The Solution
Instead of quantization (which requires CUDA libs), we:
1. **Load in float16** (~8GB) - fits on T4
2. **Disable KV cache** (`use_cache=False`) - prevents memory leak
3. **Use eager attention** - no flash_attn required

Result: **Perfect stability at 8.5 GB VRAM!**

---

## Kaggle Notebook Steps

### Step 1: Clone & Setup
```python
import os
import sys

PROJECT_NAME = "VisionDocPhi-3.5"
GITHUB_REPO = "https://github.com/mokshu7k/VisionDocPhi-3.5.git"
PROJECT_PATH = f"/kaggle/working/{PROJECT_NAME}"

if not os.path.exists(PROJECT_PATH):
    print("📥 Cloning repository...")
    !git clone {GITHUB_REPO} {PROJECT_PATH}
else:
    print(f"✓ Repository already exists")

os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)
print(f"📂 Working directory: {os.getcwd()}")
```

### Step 2: Install Dependencies
```python
import os
os.chdir('/kaggle/working/VisionDocPhi-3.5')

print("📦 Installing dependencies...")
!pip install -q -r requirements.txt

print("✅ Dependencies installed!")
```

**Important:** Do NOT install `bitsandbytes` - it's not needed and causes issues on Kaggle!

### Step 3: Verify Setup
```python
import torch
import os
import sys

os.chdir('/kaggle/working/VisionDocPhi-3.5')
sys.path.insert(0, '/kaggle/working/VisionDocPhi-3.5')

print("🔍 VERIFICATION CHECKLIST\n")

# Check GPU
print("📦 GPU Status:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check project structure
print("\n📁 Project Structure:")
required = ['config', 'src', 'scripts', 'data/raw', 'data/outputs']
for d in required:
    exists = "✓" if os.path.exists(d) else "✗"
    print(f"  {exists} {d}/")

# Check imports
print("\n📥 Checking imports...")
try:
    from config.settings import MODEL_NAME, DEVICE, USE_8BIT_QUANTIZATION
    print(f"  ✓ config.settings")
    print(f"    Model: {MODEL_NAME}")
    print(f"    Quantization: {USE_8BIT_QUANTIZATION}")
except Exception as e:
    print(f"  ✗ {e}")

print("\n✅ Setup verification complete!")
```

### Step 4: Load Model
```python
import os
import sys
import torch

os.chdir('/kaggle/working/VisionDocPhi-3.5')
sys.path.insert(0, '/kaggle/working/VisionDocPhi-3.5')

from config.settings import MODEL_NAME
from src.models.inference import DocVQAInference

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Initializing Phi-3.5 Vision model...\n")
print(f"Device: {device}")
print(f"Memory Mode: float16 (no quantization)")
print(f"Memory Optimization: KV cache fix (use_cache=False)")
print()

inference = DocVQAInference(model_name=MODEL_NAME, device=device)

print("✅ Model loaded successfully!")
print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Step 5: Quick Test (5 Samples)
```python
import os
import sys
import json
from PIL import Image
from tqdm import tqdm
import torch

os.chdir('/kaggle/working/VisionDocPhi-3.5')
sys.path.insert(0, '/kaggle/working/VisionDocPhi-3.5')

from config.settings import VAL_ANNOTATIONS, IMAGES_DIR
from src.data.dataset import create_dataloader
from src.utils.metrics import calculate_metrics

print("\n" + "="*70)
print("🧪 QUICK TEST (First 5 Samples)")
print("="*70 + "\n")

try:
    # Create dataloader
    dataloader = create_dataloader(
        annotations_file=str(VAL_ANNOTATIONS),
        image_dir=str(IMAGES_DIR),
        split='val',
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

    test_results = []
    predictions = []
    ground_truths = []

    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        
        for sample in batch:
            image = sample['image']
            question = sample['question']
            ground_truth = sample['answers'][0] if sample['answers'] else "N/A"
            
            # Generate answer
            predicted_answer = inference.generate_answer(image, question)
            
            print(f"Sample {i+1}:")
            print(f"  Q: {question}")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Ground Truth: {ground_truth}\n")
            
            predictions.append(predicted_answer)
            ground_truths.append([ground_truth])

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    print("📊 Metrics on 5 samples:")
    print(f"  ANLS Score: {metrics.get('anls', 0):.4f}")
    print(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
    
    print("\n✅ Quick test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

### Step 6: Full Evaluation
```python
import os
import sys
os.chdir('/kaggle/working/VisionDocPhi-3.5')
sys.path.insert(0, '/kaggle/working/VisionDocPhi-3.5')

from src.pipelines.baseline import run_zero_shot_baseline

print("\n" + "="*70)
print("📊 FULL EVALUATION (VAL SET)")
print("="*70 + "\n")

try:
    eval_results = run_zero_shot_baseline(
        split="val",
        num_samples=None,  # Use all samples
        save_results=True
    )
    
    print("\n✅ Evaluation completed!")
    print("📊 Results saved to: data/outputs/")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
```

### Step 7: Download Results
```python
import os

results_dir = 'data/outputs'
if os.path.exists(results_dir):
    print("📥 Available results:")
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            print(f"  - {file}")
else:
    print("No results found")
```

---

## Troubleshooting

### "CUDA out of memory" Error
**✅ This should NOT happen with the current setup!**

If it does:
1. Restart the kernel
2. Clear GPU memory:
```python
import torch
torch.cuda.empty_cache()
```
3. Verify KV cache fix is in place (check `use_cache=False` in inference.py line 260)

### "ModuleNotFoundError" Errors
1. Make sure you ran Step 1 (clone repo)
2. Make sure `sys.path.insert(0, PROJECT_PATH)` is in your cell
3. Restart kernel if needed

### "Image file not found" Error
You need to upload/mount the dataset:
1. Go to `/kaggle/working/VisionDocPhi-3.5/data/raw/`
2. Upload `spdocvqa_images.zip`
3. Run: `!unzip -q spdocvqa_images.zip`

### Model Download Takes Forever
- HuggingFace Hub can be slow
- Model is ~8.3 GB - first download takes 5-10 minutes
- Wait patiently or restart kernel and try again

---

## Memory Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| **Phi-3.5-Vision Model** | 8.3 GB | Loaded in float16 |
| **Processor/Tokenizer** | 0.2 GB | Lightweight |
| **Batch Image Processing** | 0.5 GB | Cached during inference |
| **Outputs/Scratch** | 0.5 GB | Temporary allocations |
| **Total** | ~9.5 GB max | ← Fits on T4 (15GB) |

**KV Cache Prevention:** With `use_cache=False`, memory stays constant ~9.5 GB throughout evaluation!

---

## Key Configuration

File: `config/settings.py`

Current settings for Kaggle:
```python
USE_8BIT_QUANTIZATION = False     # ← Disable quantization
TORCH_DTYPE = "float16"           # ← Use half precision
ATTN_IMPLEMENTATION = "eager"     # ← No FlashAttention required
LOW_CPU_MEM_USAGE = True          # ← Efficient loading
USE_GRADIENT_CHECKPOINTING = True # ← Save memory
BATCH_SIZE = 1                    # ← Process one image at a time
```

File: `src/models/inference.py`

Line ~260 has the critical fix:
```python
output_ids = self.model.generate(
    **inputs,
    max_new_tokens=max_length,
    do_sample=False,
    use_cache=False,  # ← CRITICAL: Prevents KV cache memory leak
    ...
)
```

---

## Success Indicator

When working correctly, you should see:
```
🚀 Initializing Phi-3.5 Vision model...

Device: cuda
📝 Using float16 precision (quantization disabled)
   → Memory optimization via KV cache leak fix (use_cache=False)

✅ Model loaded successfully!
Memory used: 8.5 GB
```

✅ Ready for evaluation!
