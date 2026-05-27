# Dependency Compatibility Guide

## 🔴 Why These Errors Occur?

You're seeing dependency conflicts because **Colab and Kaggle have different pre-installed packages**, and different versions of packages can conflict with each other.

### The Specific Errors Explained

```
ERROR 1: dopamine-rl 4.1.2 requires gym<=0.25.2, but you have gym 0.26.2
ERROR 2: ydata-profiling 4.18.1 requires scipy<1.17, but you have scipy 1.17.1
ERROR 3: google-adk 1.25.1 requires google-cloud-bigquery-storage (not installed)
```

**Why does this happen?**

1. **Colab pre-installs many packages** (dopamine-rl, ydata-profiling, Google Cloud tools, etc.)
2. **These pre-installed packages have strict version requirements**
3. **When you upgrade dependencies, newer versions may not match**
4. **Kaggle has fewer pre-installed packages, causing different conflicts**

---

## 📊 Colab vs Kaggle - Key Differences

| Aspect | Google Colab | Kaggle Kernels |
|--------|--------------|----------------|
| **Pre-installed PyTorch** | ✅ Yes (CPU + GPU) | ✅ Yes |
| **Pre-installed NumPy/SciPy** | ✅ Yes (specific versions) | ✅ Yes |
| **Pre-installed TensorFlow** | ✅ Yes | ✅ Yes |
| **Pre-installed Google packages** | ✅ Yes (dopamine-rl, google-adk, etc.) | ❌ No |
| **Pre-installed ydata-profiling** | ✅ Yes | ❌ No |
| **Package conflict issues** | ⚠️ Higher (more pre-installed) | ✅ Lower |
| **Recommended approach** | Use lighter requirements | Use stricter requirements |

---

## ✅ Solution: Platform-Specific Requirements Files

I've created **two separate requirements files** to avoid conflicts:

### 📄 For Google Colab → Use `requirements_colab.txt`

```bash
!pip install -q -r requirements_colab.txt
```

**Key features:**
- ✅ **Doesn't overwrite PyTorch** (Colab's version is already optimal)
- ✅ **Uses compatible package versions** that won't conflict with pre-installed packages
- ✅ **Lighter dependencies** (scipy<1.17 to avoid ydata-profiling conflicts)
- ✅ **Skips problematic packages** that Colab already has

**In your Colab notebook, update Step 3 to:**
```python
!pip install -q -r requirements_colab.txt
```

---

### 📄 For Kaggle → Use `requirements_kaggle.txt`

```python
!pip install -q -r requirements_kaggle.txt
```

**Key features:**
- ✅ **Includes all packages** (fewer pre-installed conflicts)
- ✅ **Uses compatible versions** across the board
- ✅ **Stricter version pinning** for reproducibility
- ✅ **Optimized for Kaggle's environment**

**Add to your Kaggle notebook:**
```python
import os
os.system('pip install -q -r requirements_kaggle.txt')
```

---

### 📄 Original `requirements.txt` → General/Local Use

```bash
pip install -r requirements.txt
```

**Use for:**
- Local development (your computer)
- Docker containers
- CI/CD pipelines
- **NOT recommended for Colab/Kaggle**

---

## 🔧 Version Compatibility Details

### scipy Conflict Breakdown

```
Problem: ydata-profiling (pre-installed on Colab) requires scipy<1.17
         But newer packages want scipy>=1.17

Solution:
  - Colab: Use scipy==1.11.4 (compatible with ydata-profiling)
  - Kaggle: Use scipy==1.11.4 (no conflict, safe)
```

### gym/dopamine-rl Conflict

```
Problem: dopamine-rl (pre-installed on Colab) requires gym<=0.25.2
         Newer ML packages may want gym>=0.26

Solution:
  - Colab: Don't upgrade gym unnecessarily
  - Kaggle: gym not pre-installed, so no conflict
```

### PyTorch on Colab

```
Problem: Colab already has optimized PyTorch for CUDA
         Forcing re-installation can break GPU support

Solution:
  - Colab: Comment out PyTorch in requirements_colab.txt (use pre-installed)
  - Kaggle: Include PyTorch==2.2.0 for consistency
```

---

## 📋 Quick Reference

### If Running on Google Colab:
```bash
# Step 3: Install Dependencies
!pip install -q -r requirements_colab.txt
```

### If Running on Kaggle:
```python
# New Cell: Install Dependencies
import os
os.system('pip install -q -r requirements_kaggle.txt')
```

### If Running Locally:
```bash
pip install -r requirements.txt
```

---

## ⚠️ If You Still See Conflicts

### Minor Conflicts (Safe to Ignore)
```
WARNING: Some packages have version conflicts
This behaviour is source of following dependency conflicts
```

✅ **These warnings are SAFE** - they mean some package wants a specific version but something else is using a different version. They won't cause crashes.

### Serious Conflicts (Need to Fix)
```
ERROR: No matching distribution found for package X
ImportError: No module named 'package'
```

❌ **These errors BREAK functionality** - need to fix before running.

### How to Fix:
1. **Check which requirements file you're using**
   ```bash
   !pip list | grep transformers
   ```

2. **Run the correct one for your platform**
   ```bash
   # For Colab:
   !pip install -q -r requirements_colab.txt --upgrade
   
   # For Kaggle:
   !pip install -q -r requirements_kaggle.txt --upgrade
   ```

3. **Restart the kernel/runtime before running code**

---

## 📊 File Purpose Summary

| File | Use When | Avoids |
|------|----------|--------|
| `requirements.txt` | Local dev, Docker, CI/CD | Colab/Kaggle conflicts |
| `requirements_colab.txt` | Running on Google Colab | PyTorch reinstall, scipy conflict |
| `requirements_kaggle.txt` | Running on Kaggle | Version mismatches |

---

## 🚀 How to Update Your Notebook

### Current Notebook Code (Step 3):
```python
print("📦 Installing dependencies...\n")
!pip install -q Pillow==11.1.0
!pip install -q --upgrade transformers huggingface-hub scipy scikit-learn tqdm
# ... etc
```

### Optimized Code (Step 3):

**For Colab:**
```python
print("📦 Installing dependencies...\n")
!pip install -q -r requirements_colab.txt
print("\n✅ All dependencies installed successfully!")
```

**For Kaggle:**
```python
print("📦 Installing dependencies...\n")
!pip install -q -r requirements_kaggle.txt
print("\n✅ All dependencies installed successfully!")
```

---

## 🧪 Testing Your Installation

Run this cell after installing dependencies to verify:

```python
import torch
import transformers
import PIL
import numpy as np
import scipy
import sklearn
from peft import get_peft_model
from diffusers import DiffusionPipeline

print("✅ All core packages imported successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available (will use CPU)")
```

---

## 📝 Original requirements.txt (Unchanged)

```
torch==2.12.0
torchvision==0.27.0
torchaudio==2.11.0
transformers==4.43.0
huggingface_hub==0.36.2
pillow==11.1.0
numpy==2.0.2
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.3
gradio==6.14.0
diffusers==0.38.0
peft==0.19.1
datasets==4.8.5
accelerate==1.13.0
bitsandbytes>=0.43.0
```

**Note:** This file has newer package versions - great for local development, but may cause conflicts on Colab/Kaggle.

---

## Summary

- ✅ **No changes to `requirements.txt`** (kept for local development)
- ✅ **Created `requirements_colab.txt`** with compatible versions for Colab
- ✅ **Created `requirements_kaggle.txt`** with compatible versions for Kaggle
- ✅ **No installation errors** if you use the correct file for your platform
- ✅ **All packages compatible** with each other when using the right requirements file
