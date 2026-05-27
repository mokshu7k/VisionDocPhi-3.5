# 🎯 Kaggle Setup Guide for VisionDocPhi-3.5

## Overview
This guide provides step-by-step instructions to set up VisionDocPhi-3.5 in Kaggle notebooks with proper dependency management.

## ⚠️ Key Differences from Colab

| Aspect | Kaggle | Colab |
|--------|--------|-------|
| PyTorch | ✅ Pre-installed | ✅ Pre-installed |
| CUDA | ✅ Available | ✅ Available |
| Pre-installed packages | Many (numpy, scipy, scikit-learn, etc.) | Fewer than Kaggle |
| Dependency conflicts | Higher risk | Lower risk |
| Build tools | Limited | Better |
| flash-attn | ❌ Not recommended | ⚠️ May work |

## 📋 Setup Instructions

### Step 1: Clone Repository (if not already present)
```python
import os
import subprocess

# Clone the repository to /kaggle/working/
os.chdir('/kaggle/working/')
subprocess.run(['git', 'clone', 'https://github.com/YOUR-REPO/VisionDocPhi-3.5.git'], 
               check=False)
os.chdir('/kaggle/working/VisionDocPhi-3.5')
```

### Step 2: Install Dependencies (Kaggle-Optimized)
```python
import os
import sys

# Navigate to project root
os.chdir('/kaggle/working/VisionDocPhi-3.5')

print("📦 Installing Kaggle-optimized dependencies...\n")

# Install from Kaggle-specific requirements with --no-build-isolation
# This is CRITICAL to avoid "subprocess-exited-with-error" when building wheels
print("🔧 Installing from requirements_kaggle.txt (with --no-build-isolation)...")
!pip install -q -r requirements_kaggle.txt --no-build-isolation 2>&1 | grep -E "Successfully|error|ERROR" | head -20

# If you get dependency conflicts, try with --upgrade flag
# !pip install -q -r requirements_kaggle.txt --no-build-isolation --upgrade

print("\n✅ Dependencies installed!")
```

### Step 3: Handle Dependency Conflicts (Automatic)
```python
print("🔍 Resolving dependency conflicts...\n")

# Install critical dependencies that Kaggle might be missing
print("📥 Installing missing critical packages...")
!pip install -q --upgrade gradio transformers huggingface-hub 2>&1 | grep -E "(Successfully|ERROR|WARNING)" || echo "✅ Updates completed"

print("\n✅ Conflict resolution complete!")
```

### ⚠️ Common Issues & Solutions

#### Issue 1: "subprocess-exited-with-error" when installing
**Cause**: Pip is trying to build wheels from source code, but Kaggle's build environment is incompatible
**Solution**: Use the `--no-build-isolation` flag to skip building from source:
```python
# CORRECT - This should work:
!pip install -q -r requirements_kaggle.txt --no-build-isolation

# WRONG - This will fail with subprocess-exited-with-error:
!pip install -q -r requirements_kaggle.txt
```

This is the most common issue. The `--no-build-isolation` flag tells pip to use pre-built wheels instead of trying to compile packages from source.

#### Issue 2: Dependency resolution conflicts (ResolutionImpossible or warnings about incompatible packages)
**Cause**: Kaggle has many pre-installed packages (ydata-profiling, tensorflow, numba, etc.) that have conflicting dependencies. These are NOT needed for DocVQA.

**Why warnings/conflicts appear:**
- ydata-profiling requires numpy<2.4 but Kaggle has numpy 2.4.6
- google-colab requires pandas==2.2.2 but Kaggle has 2.3.3
- tensorflow requires numpy<2.2.0 but Kaggle has 2.4.6
- etc.

**Solution**: These warnings are **safe to ignore**. The requirements_kaggle.txt is intentionally minimal and only includes what's needed for DocVQA:
```python
# This is fine to run - warnings about unrelated packages can be ignored:
!pip install -q -r requirements_kaggle.txt --no-build-isolation

# The minimal requirements include ONLY:
# - transformers, huggingface-hub (model loading)
# - peft, accelerate (model optimization)
# - bitsandbytes (quantization)
# - pillow, opencv (image processing)
# - tqdm, datasets, gradio (utilities)
#
# Pre-installed Kaggle packages like numpy, scipy, tensorflow, ydata-profiling
# are NOT in the requirements and their conflicts are irrelevant to DocVQA
```

**Why fixed versions (==) don't work on Kaggle:**
- Kaggle pre-installs specific versions of many packages
- If requirements.txt has `huggingface-hub==0.19.0` but Kaggle has 0.22.0, pip fails
- Flexible constraints (>=) allow pip to work with Kaggle's versions
- But warnings appear because Kaggle's other pre-installed packages have conflicts with each other (not with our code)

#### Issue 3: gym version conflict
**Cause**: dopamine-rl requires `gym<=0.25.2`, but Kaggle has newer
**Solution**: These are optional dependencies; not required for DocVQA inference

#### Issue 4: gradio compatibility with Pillow
**Cause**: gradio <6.15 requires Pillow <12.0
**Solution**: Already handled in `requirements_kaggle.txt`

#### Issue 5: bitsandbytes installation fails
**Cause**: Missing CUDA build tools
**Solution**:
```python
# Option A: Install pre-built wheel
!pip install -q bitsandbytes-cuda12x  # For CUDA 12.x

# Option B: Continue without bitsandbytes (uses 16-bit instead)
# Code will fall back to float16 if bitsandbytes unavailable
!pip install -q bitsandbytes || echo "⚠️  Continuing without bitsandbytes"
```

#### Issue 6: FlashAttention not available
**Cause**: Flash-attn requires specific CUDA and build tools
**Solution**: Already disabled in `requirements_kaggle.txt`. Code uses eager attention by default.

---

## 🚀 Complete Setup Code Block (Copy-Paste Ready)

```python
import os
import sys
import subprocess

print("=" * 70)
print("🎯 VisionDocPhi-3.5 Kaggle Setup")
print("=" * 70)

# Step 1: Navigate to working directory
os.chdir('/kaggle/working/')
project_path = '/kaggle/working/VisionDocPhi-3.5'

# Check if repo exists, if not clone it
if not os.path.exists(project_path):
    print("\n📥 Cloning repository...")
    subprocess.run(['git', 'clone', 'https://github.com/YOUR-REPO/VisionDocPhi-3.5.git'], 
                   check=True)

os.chdir(project_path)
print(f"\n✅ Working directory: {os.getcwd()}\n")

# Step 2: Install minimal dependencies (warnings about unrelated packages are safe to ignore)
print("📦 Installing dependencies (this may take 1-2 minutes)...\n")
print("⚠️  Note: You may see warnings about unrelated Kaggle packages - these are safe to ignore!\n")

# Install with --no-build-isolation
!pip install -q -r requirements_kaggle.txt --no-build-isolation 2>&1 | grep -E "Successfully|Collecting|Installing" | head -20

# Step 3: Verify critical packages for DocVQA
print("\n🔍 Verifying critical packages...\n")

critical_packages = ['transformers', 'huggingface_hub', 'torch', 'PIL', 'peft', 'accelerate']
all_good = True
for package in critical_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"❌ {package}: {str(e)}")
        all_good = False

if not all_good:
    print("\n⚠️  Some packages failed. Trying with --upgrade...")
    !pip install -q -r requirements_kaggle.txt --no-build-isolation --upgrade

# Step 4: Display configuration
print("\n" + "=" * 70)
print("📊 Environment Configuration")
print("=" * 70)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU(s): {torch.cuda.device_count()}")
    
print("\n✅ Setup complete! You're ready to run VisionDocPhi-3.5!")
print("✅ Note: Dependency warnings about google-adk, ydata-profiling, tensorflow, etc. are OK")
```

---

## 🔧 Troubleshooting

### If you see "Getting requirements to build wheel did not run successfully":
```python
# Try installing without build isolation
!pip install --no-build-isolation -r requirements_kaggle.txt

# Or install packages individually
!pip install -q transformers
!pip install -q huggingface-hub
!pip install -q peft
!pip install -q accelerate
!pip install -q datasets
```

### Check your environment:
```python
# Verify installed packages
!pip freeze | grep -E "transformers|huggingface|torch|scipy|gradio"

# Check for conflicts
!pip check
```

---

## 📝 Notes
- **PyTorch**: Already pre-installed in Kaggle. Don't reinstall.
- **GPU Memory**: VisionDocPhi-3.5 with 8-bit quantization uses ~8-12GB VRAM
- **Timeout**: If installation times out, try `--no-build-isolation` flag
- **Kaggle TPU**: GPU setup recommended; TPU may have additional compatibility issues

---

## ✅ Verification Script
After setup, run this to verify everything works:

```python
# Test imports
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import bitsandbytes as bnb

print("✅ All critical imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")
```

