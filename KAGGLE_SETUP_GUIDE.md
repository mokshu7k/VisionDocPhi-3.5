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

#### Issue 2: scipy version conflict
**Cause**: ydata-profiling requires `scipy<1.17`, but newer versions installed
**Solution**: Already handled in `requirements_kaggle.txt` (scipy constraint included)

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

# Step 2: Install dependencies with --no-build-isolation (CRITICAL for Kaggle)
print("📦 Installing dependencies (this may take 2-3 minutes)...\n")

# Install with --no-build-isolation to avoid subprocess-exited-with-error
!pip install -q -r requirements_kaggle.txt --no-build-isolation 2>&1 | grep -E "Successfully|error|ERROR" | head -20

# Step 3: If bitsandbytes fails, try the CUDA-specific version
print("\n🔷 Attempting to ensure bitsandbytes is available...")
!pip install -q bitsandbytes-cuda12x --no-build-isolation 2>&1 | grep -E "Successfully|already" || echo "⚠️  Continuing with fallback to float16"

# Step 4: Verify critical packages
print("\n🔍 Verifying critical packages...\n")

required_packages = ['transformers', 'huggingface_hub', 'torch', 'PIL', 'scipy']
for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"⚠️  {package}: {str(e)}")

# Step 5: Display configuration
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

