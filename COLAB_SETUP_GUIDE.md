# 🚀 Google Colab Setup Guide - VisionDocPhi-3.5

## ✅ Cleaned Project Structure

Your project is now optimized for Google Colab:

```
VisionDocPhi-3.5/
├── config/                              # Configuration
│   ├── __init__.py
│   └── settings.py
│
├── src/                                 # Source code (all you need)
│   ├── models/
│   │   ├── __init__.py
│   │   └── inference.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── pipelines/
│       ├── __init__.py
│       └── baseline.py
│
├── notebooks/
│   └── DocVQA_Colab_Production.ipynb    # ← USE THIS FILE
│
├── data/
│   ├── raw/
│   │   ├── spdocvqa_images/             # Your document images
│   │   └── spdocvqa_qas/                # Your JSON annotations
│   └── outputs/                         # Results will be saved here
│
├── README.md
└── requirements.txt
```

---

## 📋 Step-by-Step: Running on Google Colab

### **Step 1: Prepare Your Project Folder**

✅ **Already done!** Your project is now Colab-ready.

### **Step 2: Upload Project to Google Drive**

1. Open [Google Drive](https://drive.google.com)
2. Click **"New"** → **"Folder"**
3. Name it: `VisionDocPhi-3.5`
4. Right-click the folder → **"Upload files"** or **"Upload folder"**
5. Upload your entire cleaned project folder

📍 Final path should be: `My Drive/VisionDocPhi-3.5/`

### **Step 3: Download Dataset**

If you haven't downloaded the DocVQA dataset:

1. Go to: https://rrc.cvc.uab.es/?ch=17
2. Register (if needed)
3. Download:
   - **Images** (spdocvqa_images.zip)
   - **Annotations** (spdocvqa_qas.zip)
4. Extract in Google Drive:
   - Images → `VisionDocPhi-3.5/data/raw/spdocvqa_images/`
   - Annotations → `VisionDocPhi-3.5/data/raw/spdocvqa_qas/`

### **Step 4: Open Google Colab Notebook**

1. Open [Google Colab](https://colab.research.google.com)
2. Click **"File"** → **"Open notebook"** → **"Upload"**
3. Upload: `notebooks/DocVQA_Colab_Production.ipynb`
4. Or directly open from Drive:
   - Go to your `VisionDocPhi-3.5/notebooks/` folder
   - Right-click `DocVQA_Colab_Production.ipynb`
   - Select **"Open with"** → **"Google Colaboratory"**

### **Step 5: Enable GPU (IMPORTANT!)**

1. Click **"Runtime"** (top menu)
2. Select **"Change runtime type"**
3. Choose:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: GPU (T4) ← FREE!
4. Click **"Save"**

⚡ This gives you ~15GB VRAM free GPU!

### **Step 6: Run the Notebook**

**Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_PATH = '/content/drive/MyDrive/VisionDocPhi-3.5'
os.chdir(PROJECT_PATH)
```

**Then run remaining cells in order:**

1. ✅ Install Dependencies
2. ✅ Verify Setup
3. ✅ Import Modules
4. ✅ Display Configuration
5. ✅ Get Dataset Stats
6. ✅ Initialize Model (Phi-3.5 downloads here - ~15 min first time)
7. ✅ Quick Test (runs on 5 samples - ~2 min)
8. ✅ Full Evaluation (runs on entire dataset - varies by size)
9. ✅ Display Results
10. ✅ Download Results (auto-downloads to your computer)

### **Step 7: Download Results**

Results automatically download when you run the last cell:
- `predictions_zeroshot.json` - detailed predictions
- `results_zeroshot.json` - metrics summary

---

## 🎯 Quick Reference Commands

### In Colab Notebook:

**Mount Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/VisionDocPhi-3.5')
```

**Install packages:**
```python
!pip install -q torch torchvision "transformers>=4.43.0" pillow numpy tqdm
```

**Import project:**
```python
import sys
sys.path.insert(0, os.getcwd())

from config.settings import MODEL_NAME, DEVICE
from src.pipelines.baseline import run_zero_shot_baseline
```

**Run evaluation:**
```python
# Full evaluation
results = run_zero_shot_baseline(split="val", save_results=True)

# Quick test (first 100 samples)
results = run_zero_shot_baseline(split="val", num_samples=100)

# Test set
results = run_zero_shot_baseline(split="test", save_results=True)
```

**Download results:**
```python
from google.colab import files
files.download('data/outputs/predictions_zeroshot.json')
```

---

## ⏱️ Timing Guide

| Step | Time | Notes |
|------|------|-------|
| Setup & Install | 2-3 min | First time only |
| Model Download | 10-15 min | One time, ~15GB model |
| Verify Setup | 1 min | Checks configuration |
| Quick Test (5 samples) | 3-5 min | Verify everything works |
| Full Evaluation (5.4K val samples) | 2-4 hours | Depends on GPU |
| Download Results | < 1 min | Automatic |

---

## 🔧 Troubleshooting on Colab

### **Issue: "ModuleNotFoundError: No module named 'config'"**
- ✓ Ensure you're in correct directory: `os.chdir('/content/drive/MyDrive/VisionDocPhi-3.5')`
- ✓ Run after mounting: `sys.path.insert(0, os.getcwd())`

### **Issue: "FileNotFoundError: data/raw/spdocvqa_qas/val_v1.0_withQT.json"**
- ✓ Download dataset from: https://rrc.cvc.uab.es/?ch=17
- ✓ Extract to correct path in Drive

### **Issue: "CUDA out of memory"**
- ✓ Already using T4 GPU (12GB memory)
- ✓ Batch size is 1 (minimum)
- ✓ Try fewer samples: `num_samples=100`

### **Issue: "Model download fails"**
- ✓ Check internet connection
- ✓ Colab downloads from HuggingFace - may be slow
- ✓ Retry the cell (usually works on 2nd attempt)

### **Issue: "Session disconnected"**
- ✓ Colab has 12-hour limit (free tier)
- ✓ Save results before disconnect
- ✓ Use `save_results=True` (automatic)

---

## 📊 What Happens in Each Cell

| Cell | What it Does |
|------|-------------|
| 1 | Mount Google Drive to access your files |
| 2 | Install Python packages |
| 3 | Verify project structure and config |
| 4 | Import all project modules |
| 5 | Show configuration (model, device, paths) |
| 6 | Load and display dataset statistics |
| 7 | Download Phi-3.5 model to GPU |
| 8 | Run quick test on 5 samples |
| 9 | Run full evaluation on entire dataset |
| 10 | Display final metrics (ANLS, Exact Match) |
| 11 | Download results to your computer |

---

## 💾 What Gets Saved

**In Colab (data/outputs/):**
- `predictions_zeroshot.json` - All predictions with ground truth
- `results_zeroshot.json` - Metrics summary only

**Downloaded to your computer:**
- Same JSON files (auto-downloaded)

**Structure of results:**
```json
{
  "model": "microsoft/phi-3.5-vision-instruct",
  "split": "val",
  "device": "cuda",
  "metrics": {
    "anls": 0.4567,
    "exact_match": 0.2345,
    "total_samples": 5404
  },
  "results": [
    {
      "question_id": 123456,
      "question": "What is the total?",
      "predicted_answer": "$100.00",
      "ground_truth_answers": ["$100.00", "100.00"],
      "anls_score": 1.0
    },
    ...
  ]
}
```

---

## 🎓 Understanding the Output

**ANLS Score:**
- Measures similarity between prediction and ground truth
- Range: 0 to 1
- 1.0 = Perfect match
- Typical zero-shot: 0.40-0.50

**Exact Match:**
- Percentage of perfectly correct answers
- Range: 0% to 100%
- Typical zero-shot: 20-30%

**Example Interpretation:**
```
ANLS: 0.4567 (45.67%)     ← Average similarity
Exact Match: 0.2345 (23.45%) ← % perfect answers
Total Samples: 5404       ← Questions evaluated
```

---

## 🚀 Advanced Usage

### **Run on Test Set:**
```python
results = run_zero_shot_baseline(split="test", save_results=True)
```

### **Run on Subset:**
```python
# Evaluate only first 100 samples (for quick testing)
results = run_zero_shot_baseline(split="val", num_samples=100)
```

### **Custom Evaluation:**
```python
from src.pipelines.baseline import run_zero_shot_baseline
from config.settings import VAL_ANNOTATIONS, IMAGES_DIR
from src.data.dataset import create_dataloader
from src.models.inference import DocVQAInference

# Create dataloader
dataloader = create_dataloader(
    annotations_file=str(VAL_ANNOTATIONS),
    image_dir=str(IMAGES_DIR),
    split='val'
)

# Initialize model
inference = DocVQAInference(device='cuda')

# Run evaluation
results = inference.evaluate(dataloader)
```

---

## 📞 Need Help?

**Common questions:**

1. **Q: Can I use free Google Colab?**
   - A: Yes! Free tier includes GPU (T4) with 12-hour sessions

2. **Q: How long does full evaluation take?**
   - A: ~2-4 hours for 5.4K samples on T4 GPU

3. **Q: Can I pause and resume?**
   - A: Results are saved automatically with `save_results=True`
   - Restart and run from last step

4. **Q: Can I share results with others?**
   - A: Download JSON and share directly
   - Or keep in Google Drive and share the link

5. **Q: How much does Colab cost?**
   - A: Free! (with limits) or upgrade to Pro for more hours

---

## ✨ You're Ready!

1. ✅ Project is cleaned up
2. ✅ Structure is Colab-ready
3. ✅ Notebook is prepared
4. ✅ All unnecessary files removed

**Next steps:**
1. Upload project to Google Drive
2. Download DocVQA dataset
3. Open notebook in Colab
4. Run cells from top to bottom
5. Download results!

---

**Happy coding on Colab! 🎉**
