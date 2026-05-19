# GitHub + Google Colab Workflow Guide

## 🚀 Complete Setup & Running Guide

This guide explains how to run the VisionDocPhi-3.5 project on Google Colab using GitHub as the single source of truth.

---

## 📋 Table of Contents
1. [Initial Setup (One-time)](#initial-setup-one-time)
2. [Running the Project on Colab](#running-the-project-on-colab)
3. [Updating Code & Syncing](#updating-code--syncing)
4. [Troubleshooting](#troubleshooting)
5. [Workflow Summary](#workflow-summary)

---

## Initial Setup (One-time)

### ✅ Already Done!
Your project is already set up on GitHub at:
```
https://github.com/mokshu7k/VisionDocPhi-3.5.git
```

The notebook has been updated to work with GitHub. No further setup needed!

---

## Running the Project on Colab

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Click **"New notebook"** (or open the existing one if you have it)

### Step 2: Upload or Link Your Notebook
**Option A: Upload the notebook from local**
- File → Upload notebook
- Select `notebooks/DocVQA_Colab_Production.ipynb`

**Option B: Open from GitHub**
- Go to: `https://colab.research.google.com/github/mokshu7k/VisionDocPhi-3.5/blob/main/notebooks/DocVQA_Colab_Production.ipynb`
- Click the GitHub icon, paste your repo URL, and select the notebook

### Step 3: Run the Notebook Cells in Order

```
1️⃣  Step 1: Clone from GitHub & Setup Project
   - Clones your repo from GitHub
   - Sets up the working directory
   - Takes ~10-20 seconds

2️⃣  Step 2: Pull Latest Changes (Optional)
   - If you made changes to the repo, uncomment to pull them
   - Can skip if using latest version

3️⃣  Step 3: Install Dependencies
   - Installs all required packages
   - Installs FlashAttention2 for GPU optimization (optional)
   - Takes ~3-5 minutes

🆕 3️⃣.5️⃣  Step 3.5: Download Dataset (Optional)
   - Shows 3 options to get dataset images
   - Images NOT on GitHub (too large - see DATASET_GUIDE.md)
   - Only needed for Step 10 (full evaluation)
   - Quick test (Step 9) doesn't need images
   - Takes 5-20 minutes if downloading

4️⃣  Step 4: Verify Project Setup
   - Checks all directories and files
   - Verifies CUDA/GPU availability
   - Verifies PyTorch installation

5️⃣  Step 5: Import Project Modules
   - Imports all custom modules from your project
   - Should show ✅ for all imports

6️⃣  Step 6: Display Configuration
   - Shows model name, device, paths, etc.
   - Verify everything looks correct

7️⃣  Step 7: Get Dataset Statistics
   - Loads dataset info
   - Shows breakdown by question type
   - Only works if dataset files are present

8️⃣  Step 8: Initialize Model
   - Loads the Phi-3.5 Vision model (~8.29 GB)
   - Takes 1-3 minutes depending on GPU
   - Uses fallback attention if FlashAttention2 fails

9️⃣  Step 9: Quick Test (5 Samples)
   - Tests inference on 5 samples
   - Takes ~2-5 minutes
   - Shows predictions vs ground truth
   - ✅ WORKS WITHOUT DATASET

🔟 Step 10: Full Evaluation Pipeline
   - Runs complete evaluation on all samples
   - Requires dataset from Step 3.5
   - Takes time depending on dataset size
   - Saves results to `data/outputs/`

1️⃣1️⃣ Step 11: Display Results
   - Shows final metrics (ANLS, Exact Match, etc.)
   - Shows where results are saved

1️⃣2️⃣ Step 12: Download Results (Optional)
   - Downloads JSON files with predictions and metrics
   - Use this to save results locally
```

---

## ⏱️ Estimated Time Breakdown

| Step | Time |
|------|------|
| Clone & Setup | 20 sec |
| Install Dependencies | 3-5 min |
| **Download Dataset** | **5-20 min*** |
| Verify Setup | 30 sec |
| Load Model | 1-3 min |
| Quick Test (5 samples) | 2-5 min |
| Full Evaluation (all samples) | 5-30 min** |
| **Total (with dataset)** | **~17-65 min** |
| **Total (model test only)** | **~12-15 min** |

*Only needed for full evaluation; download in step 3.5  
**Depends on number of samples

---

## Updating Code & Syncing

### 📝 Workflow: Edit Code → Push to GitHub → Run in Colab

#### Step 1: Make Changes Locally

Edit your code on your Windows machine (VS Code):
```
- Edit files in d:\Projects\VisionDocPhi-3.5\
- Examples: src/models/inference.py, config/settings.py, etc.
```

#### Step 2: Push to GitHub

Run in PowerShell from your project directory:
```powershell
cd d:\Projects\VisionDocPhi-3.5
git add .
git commit -m "Your meaningful commit message"
git push origin main
```

**Example commit messages:**
```powershell
git commit -m "Fix FlashAttention fallback"
git commit -m "Improve model initialization"
git commit -m "Update inference parameters"
```

#### Step 3: Pull Changes in Colab

In your Colab notebook, run Step 2 (uncomment the pull command):
```python
!git pull origin main
```

**Or** - Run the entire notebook from the start (cloning will get latest code)

---

## 🎯 Common Tasks

### Task 1: Run a Quick Test
```
Run only cells: 1, 3, 4, 5, 6, 8, 9
(Skip the full evaluation)
Takes ~5-10 minutes
```

### Task 2: Update Model Parameters
```
1. Edit src/models/inference.py or config/settings.py locally
2. Push to GitHub: git push origin main
3. In Colab, run: !git pull origin main
4. Restart the kernel (Runtime → Restart runtime)
5. Rerun the notebook
```

### Task 3: Add New Features
```
1. Create new files in src/
2. Add imports to __init__.py files
3. Push to GitHub
4. Pull in Colab
5. Test in Colab notebook
```

### Task 4: Fix Issues
```
1. Reproduce the issue locally
2. Fix the code
3. Push to GitHub
4. Pull in Colab
5. Test again
```

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```python
# Make sure this runs first (Step 1)
# Then restart the kernel: Runtime → Restart runtime
# Then run all cells in order
```

### Issue: "CUDA out of memory"

**Solutions:**
```python
# Option 1: Reduce batch size
# Edit config/settings.py, change BATCH_SIZE = 1

# Option 2: Use CPU (slower but works)
# Edit config/settings.py, change DEVICE = "cpu"
```

### Issue: FlashAttention2 installation fails

**Don't worry!** The code automatically falls back to eager attention.
```
⚠️  FlashAttention2 not available (will use eager attention)
✅ Model still works, just slower
```

### Issue: "Dataset files not found"

**The dataset is only needed if you run Steps 7-10.**

**Important:** Images are NOT on GitHub (too large). Download them in Colab:

**See: [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed instructions**

Quick options:
```python
# Option 1: If images are on Google Drive
!pip install -q gdown
!gdown YOUR_FILE_ID -O data/raw/images.zip
!cd data/raw && unzip -q images.zip

# Option 2: Manual upload
from google.colab import files
files.upload()  # Select images.zip
```

- If you just want to test the model (Quick Test), skip the dataset
- For full evaluation, follow [DATASET_GUIDE.md](DATASET_GUIDE.md)

### Issue: Changes not appearing in Colab

**Make sure to:**
```bash
# 1. Push to GitHub
git push origin main

# 2. Pull in Colab (uncomment Step 2)
!git pull origin main

# 3. Restart kernel
# Runtime → Restart runtime

# 4. Rerun cells
```

---

## 💡 Best Practices

### ✅ DO:
- ✅ Test code locally before pushing
- ✅ Write meaningful commit messages
- ✅ Push regularly (don't wait too long)
- ✅ Keep `.gitignore` updated
- ✅ Use branches for experimental features
- ✅ Delete old/unused notebooks from Colab

### ❌ DON'T:
- ❌ Don't upload large files (models, datasets) to GitHub
- ❌ Don't push unfinished/broken code
- ❌ Don't manually edit code in Colab and expect it to persist
- ❌ Don't store Google Drive paths in code
- ❌ Don't commit large `.ipynb` notebooks to GitHub (they have binary data)

---

## 📊 Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│         YOUR LOCAL MACHINE (Windows)                    │
│  - Edit code in VS Code                                 │
│  - Test locally if needed                               │
│  - git push to GitHub                                   │
└─────────────────────────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │      GITHUB REPOSITORY           │
                    │  (Single source of truth)        │
                    │  - All code stored here          │
                    │  - Version history               │
                    │  - Easy to access anywhere       │
                    └──────────────────┬───────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │    GOOGLE COLAB                  │
                    │  - Clone/Pull from GitHub        │
                    │  - Install dependencies          │
                    │  - Load model & run inference    │
                    │  - Use GPU for fast processing   │
                    │  - Download results              │
                    └──────────────────────────────────┘
```

---

## 🎓 Example: Complete Workflow

### Scenario: You want to improve inference speed

```bash
# 1. On your Windows machine:
cd d:\Projects\VisionDocPhi-3.5

# 2. Edit the file
# Open src/models/inference.py in VS Code
# Change MAX_NEW_TOKENS from 128 to 64

# 3. Push to GitHub
git add src/models/inference.py
git commit -m "Reduce max tokens to 64 for faster inference"
git push origin main

# 4. In Colab notebook:
# Run Step 2 (uncomment and execute):
!git pull origin main

# 5. Restart kernel and rerun
# Runtime → Restart runtime
# Then run Step 8 and 9 to test

# 6. If it works well, that's it!
# If not, edit again and push another commit
```

---

## ❓ FAQ

**Q: Can I edit files directly in Colab and save them?**
A: No, edits in Colab are temporary. Always edit locally and push to GitHub.

**Q: What if I forget to push before using Colab?**
A: No problem! Just push whenever you want, then pull in Colab.

**Q: Do I need Google Drive at all?**
A: No! GitHub + Colab is all you need. Google Drive is completely optional.

**Q: How do I switch between different versions of my code?**
A: Use GitHub branches:
```bash
git branch new-feature
git checkout new-feature
# Make changes and push
# Later: git checkout main to go back
```

**Q: Can I share my project with others?**
A: Yes! Your GitHub repo is public. Others can clone it and run it on their own Colab.

---

## 📞 Quick Reference

### Essential Commands

**Local (Windows PowerShell):**
```powershell
git status              # Check what changed
git add .               # Stage all changes
git commit -m "msg"     # Commit changes
git push origin main    # Push to GitHub
git pull origin main    # Pull latest (if working locally)
```

**Colab:**
```python
!git pull origin main   # Pull latest from GitHub
```

---

## 🎉 You're All Set!

Your project is now set up for:
- ✅ Version control with GitHub
- ✅ Easy collaboration
- ✅ Free GPU with Google Colab
- ✅ No Google Drive space waste
- ✅ Always using latest code

**Next Step:** Go to Colab, upload/open the notebook, and run it! 🚀

---

*Last Updated: May 19, 2026*
*Project: VisionDocPhi-3.5*
*Repository: https://github.com/mokshu7k/VisionDocPhi-3.5*
